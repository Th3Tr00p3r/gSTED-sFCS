"""
Bead positioning - put the bead where you want it with the stepper stage.

A bead lands where it lands. To measure a PSF, or to map the field across the sample, it
has to be brought to the centre of the scan, or to some other chosen point. This dialog
does that in one move:

  1. Scan, and put the crosshair on the bead (the auto-crosshair usually does it).
  2. 'Read bead' here.
  3. Choose the target: the centre of the scan, the remembered red crosshair, or typed
     coordinates in microns.
  4. 'Move bead to target'.

Before any of that the stage has to be calibrated against the scan, which measures the
two things the steps-per-micron in the settings cannot say: which stage axis moves the
image along which axis, and in which direction. Two moves do it:

  'Read bead' -> type an amount -> 'Move +X' -> rescan -> 'Record X'
  'Read bead' -> type an amount -> 'Move +Y' -> rescan -> 'Record Y'
  'Build calibration'

Save it, and the next session starts at step 1. If you just want to get going, 'From
steps/um' builds a calibration from the scalar in the stage settings plus a guess at the
swap and the signs - good enough to find a bead, not good enough to trust.

Every move is planned to finish with the same fixed approach vector: the stage is sent
past the target and then brought back along the approach direction, so backlash enters
every positioning identically instead of randomly. The stage is still a stepper, though
- at 2 steps/um one step is 500 nm, so the bead lands within about half a step of where
you asked. That is fine, because the next scan measures where it actually is.
"""

import logging
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)

from utilities.stage_mapping import (
    AXIS_NAMES,
    DEFAULT_BACKLASH_STEPS,
    StageMappingError,
    StageMove,
    StageScanCalibration,
    calibrate,
    from_steps_per_um,
)

DEFAULT_CALIBRATION_PATH = Path("./settings/stage_scan_calibration.json")


class BeadPositioningDialog(QDialog):
    """Doc."""

    def __init__(self, impl, parent=None):
        super().__init__(parent)
        self._impl = impl
        self._bead_um = None  # the bead position last read, in um
        self._pending = None  # (axis, steps) of a move made but not yet recorded
        self._moves = []
        self._calibration = None

        self.setWindowTitle("Bead Positioning")
        self.setMinimumWidth(540)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Move a bead to a chosen point in the scan with the stage."))
        layout.addWidget(self._position_group())
        layout.addWidget(self._calibration_group())
        layout.addWidget(self._move_group())
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(110)
        layout.addWidget(self._log)
        self._load_if_present()

    # ------------------------------------------------------------------ layout

    def _position_group(self) -> QGroupBox:
        box = QGroupBox("Bead position")
        grid = QGridLayout(box)
        self._bead_label = QLabel("-")
        grid.addWidget(QLabel("at the crosshair:"), 0, 0)
        grid.addWidget(self._bead_label, 0, 1)
        read_btn = QPushButton("Read bead")
        read_btn.setToolTip("Take the scan image crosshair as the bead's current position")
        read_btn.clicked.connect(self._read_bead)
        grid.addWidget(read_btn, 0, 2)
        return box

    def _calibration_group(self) -> QGroupBox:
        box = QGroupBox("Stage calibration")
        grid = QGridLayout(box)

        grid.addWidget(QLabel("Move by (steps)"), 0, 0)
        self._amount_box = QSpinBox()
        self._amount_box.setRange(-2000, 2000)
        self._amount_box.setValue(200)
        self._amount_box.setToolTip(
            "Large enough that the bead clearly moves, small enough that it stays in the scan"
        )
        grid.addWidget(self._amount_box, 0, 1)
        for col, axis in enumerate(AXIS_NAMES):
            btn = QPushButton(f"Move {axis.replace('stage_', '').upper()}")
            btn.clicked.connect(lambda _, a=axis: self._calibration_move(a))
            grid.addWidget(btn, 0, 2 + col)

        grid.addWidget(QLabel("then rescan and:"), 1, 0, 1, 2)
        for col, axis in enumerate(AXIS_NAMES):
            btn = QPushButton(f"Record {axis.replace('stage_', '').upper()}")
            btn.setToolTip("Read the bead again and record how far the last move took it")
            btn.clicked.connect(lambda _, a=axis: self._record_move(a))
            grid.addWidget(btn, 1, 2 + col)

        build_btn = QPushButton("Build calibration")
        build_btn.clicked.connect(self._build)
        scalar_btn = QPushButton("From steps/um")
        scalar_btn.setToolTip(
            "A starting guess from the stage settings - replace it with a real one"
        )
        scalar_btn.clicked.connect(self._build_from_scalar)
        grid.addWidget(build_btn, 2, 0, 1, 2)
        grid.addWidget(scalar_btn, 2, 2, 1, 2)

        self._swap_box = QCheckBox("swap axes")
        self._invert_x = QCheckBox("invert x")
        self._invert_y = QCheckBox("invert y")
        row = QHBoxLayout()
        for widget in (self._swap_box, self._invert_x, self._invert_y):
            row.addWidget(widget)
        grid.addLayout(row, 3, 0, 1, 4)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load)
        grid.addWidget(save_btn, 4, 0, 1, 2)
        grid.addWidget(load_btn, 4, 2, 1, 2)
        return box

    def _move_group(self) -> QGroupBox:
        box = QGroupBox("Move the bead")
        grid = QGridLayout(box)

        grid.addWidget(QLabel("Target"), 0, 0)
        self._target_combo = QComboBox()
        self._target_combo.addItems(
            ["centre of the scan", "remembered crosshair", "these coordinates (um)"]
        )
        grid.addWidget(self._target_combo, 0, 1, 1, 3)

        self._target_x = QDoubleSpinBox()
        self._target_y = QDoubleSpinBox()
        for spin in (self._target_x, self._target_y):
            spin.setRange(-1e4, 1e4)
            spin.setDecimals(2)
            spin.setSuffix(" um")
        grid.addWidget(QLabel("x"), 1, 0)
        grid.addWidget(self._target_x, 1, 1)
        grid.addWidget(QLabel("y"), 1, 2)
        grid.addWidget(self._target_y, 1, 3)

        grid.addWidget(QLabel("Backlash approach (steps)"), 2, 0, 1, 2)
        self._backlash_box = QSpinBox()
        self._backlash_box.setRange(0, 500)
        self._backlash_box.setValue(DEFAULT_BACKLASH_STEPS)
        self._backlash_box.setToolTip(
            "Every move finishes with this, in +x +y, so backlash is the same every time"
        )
        grid.addWidget(self._backlash_box, 2, 2, 1, 2)

        plan_btn = QPushButton("Plan only")
        plan_btn.clicked.connect(lambda: self._go(dry_run=True))
        move_btn = QPushButton("Move bead to target")
        move_btn.clicked.connect(lambda: self._go(dry_run=False))
        grid.addWidget(plan_btn, 3, 0, 1, 2)
        grid.addWidget(move_btn, 3, 2, 1, 2)
        return box

    # ------------------------------------------------------------------ helpers

    def _say(self, text: str):
        self._log.append(text)
        logging.info(f"Bead positioning: {text.splitlines()[0]}")

    def _read_bead(self):
        try:
            self._bead_um = self._impl.read_scan_image_position_um()
        except RuntimeError as exc:
            self._say(f"<b>{exc}</b>")
            return None
        self._bead_label.setText(f"{self._bead_um[0]:.2f}, {self._bead_um[1]:.2f} um")
        return self._bead_um

    def _calibration_move(self, axis: str):
        if self._read_bead() is None:
            return
        amount = self._amount_box.value()
        if not amount:
            self._say("Set a non-zero amount first.")
            return
        steps = (amount, 0) if axis == AXIS_NAMES[0] else (0, amount)
        self._pending = (axis, steps, self._bead_um)
        self._impl.move_stage_steps([steps])
        self._say(
            f"Moved {axis} by {amount} steps - rescan, put the crosshair on the bead, "
            f"then press 'Record {axis.replace('stage_', '').upper()}'."
        )

    def _record_move(self, axis: str):
        if self._pending is None or self._pending[0] != axis:
            self._say(f"Press 'Move {axis.replace('stage_', '').upper()}' first.")
            return
        before = self._pending[2]
        after = self._read_bead()
        if after is None:
            return
        displacement = (after[0] - before[0], after[1] - before[1])
        if np.hypot(*displacement) < 1e-6:
            self._say("The bead did not move - is the stage on, and was the scan repeated?")
            return
        self._moves = [m for m in self._moves if m[0] != axis]
        self._moves.append((axis, StageMove(self._pending[1], displacement)))
        self._pending = None
        self._say(
            f"Recorded {axis}: {self._pending_steps_text(self._moves[-1][1])} "
            f"-> {displacement[0]:+.2f}, {displacement[1]:+.2f} um"
        )

    @staticmethod
    def _pending_steps_text(move: StageMove) -> str:
        return f"{move.steps[0]:+.0f}, {move.steps[1]:+.0f} steps"

    def _apply(self, calibration: StageScanCalibration):
        calibration.backlash_steps = self._backlash_box.value()
        self._calibration = calibration
        warning = (
            ""
            if calibration.is_well_conditioned
            else f"\n<b>Condition number {calibration.condition_number:.0f} - "
            f"the two moves were nearly parallel; redo them.</b>"
        )
        self._say(f"<pre>{calibration.describe_axes()}</pre>{warning}")

    def _build(self):
        try:
            self._apply(
                calibrate([m for _, m in self._moves], backlash_steps=self._backlash_box.value())
            )
        except StageMappingError as exc:
            self._say(f"<b>{exc}</b>")

    def _build_from_scalar(self):
        try:
            steps_per_um = self._impl.stage_steps_per_um()
            self._apply(
                from_steps_per_um(
                    steps_per_um,
                    swap_axes=self._swap_box.isChecked(),
                    invert_x=self._invert_x.isChecked(),
                    invert_y=self._invert_y.isChecked(),
                    backlash_steps=self._backlash_box.value(),
                    notes="from the stage settings, not measured",
                )
            )
        except (StageMappingError, RuntimeError) as exc:
            self._say(f"<b>{exc}</b>")

    def _target_um(self):
        choice = self._target_combo.currentIndex()
        if choice == 0:
            return self._impl.scan_image_centre_um()
        if choice == 1:
            return self._impl.read_scan_image_position_um(use_reference=True)
        return (self._target_x.value(), self._target_y.value())

    def _go(self, dry_run: bool):
        if self._calibration is None:
            self._say("<b>No calibration yet.</b>")
            return
        bead = self._read_bead()
        if bead is None:
            return
        try:
            target = self._target_um()
        except RuntimeError as exc:
            self._say(f"<b>{exc}</b>")
            return

        self._calibration.backlash_steps = self._backlash_box.value()
        plan = self._calibration.plan_move((target[0] - bead[0], target[1] - bead[1]))
        text = (
            f"{bead[0]:.2f}, {bead[1]:.2f} -> {target[0]:.2f}, {target[1]:.2f} um"
            f"<br>&nbsp;&nbsp;{plan['total_steps'][0]:+d}, {plan['total_steps'][1]:+d} steps "
            f"as {' then '.join(str(m) for m in plan['moves'])}"
            f"<br>&nbsp;&nbsp;lands within {plan['rounding_error_um'] * 1000:.0f} nm "
            f"of the target (whole steps)"
        )
        self._say(text)
        if not dry_run:
            self._impl.move_stage_steps(plan["moves"])
            self._say("Moving - rescan to see where the bead actually landed.")

    # ------------------------------------------------------------------ storage

    def _save(self):
        if self._calibration is None:
            self._say("<b>Nothing to save.</b>")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save stage calibration", str(DEFAULT_CALIBRATION_PATH), "JSON (*.json)"
        )
        if path:
            self._say(f"Saved to {self._calibration.save(path)}")

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load stage calibration", str(DEFAULT_CALIBRATION_PATH), "JSON (*.json)"
        )
        if path:
            self._load_path(Path(path))

    def _load_if_present(self):
        if DEFAULT_CALIBRATION_PATH.is_file():
            self._load_path(DEFAULT_CALIBRATION_PATH)

    def _load_path(self, path: Path):
        try:
            calibration = StageScanCalibration.load(path)
        except (StageMappingError, OSError, ValueError, KeyError) as exc:
            self._say(f"<b>Could not load {path}: {exc}</b>")
            return
        self._backlash_box.setValue(calibration.backlash_steps)
        self._calibration = calibration
        self._say(
            f"Loaded {path} ({calibration.timestamp})<pre>{calibration.describe_axes()}</pre>"
        )
