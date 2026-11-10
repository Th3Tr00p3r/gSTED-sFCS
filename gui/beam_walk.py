"""
Beam walk assistant - a dialog that turns beam walking into two measurements and one
calculation, instead of an iteration.

The procedure:

  1. Put the reference beam (normally the excitation) on both cameras, fit it, and press
     'Remember Center' under each display. Those two crosshairs are the target.
  2. Switch to the beam being aligned (the depletion) and fit it on both cameras.
  3. Press 'Read baseline' here, then turn one knob at a time: type how much you turned
     it and press 'Record move'. Four moves (one per knob) measure the full response,
     including the cross terms; two moves (one per mirror, both axes) are enough if the
     mirrors' axes are aligned with the cameras' - the 'cross-talk' figure says whether
     they are. Do not turn anything back; the moves accumulate and the final answer
     accounts for them.
  4. Press 'Build calibration', then either:

     'Compute correction'  - how much to turn each knob, in the units you recorded. Only
                             useful if your knobs can be read.
     'Plan two moves'      - no numbers needed. It puts a yellow waypoint crosshair on
                             one camera: steer the FIRST mirror by eye until the spot
                             sits on it, with however many turns it takes. The other
                             camera then arrives where it must be on its own, so the
                             SECOND mirror alone finishes the job, and both cameras land
                             on their remembered centres together. 'Check step 1' then
                             compares that camera against the prediction, which is the
                             test of whether the calibration is trustworthy.

  The four 'Record <knob>' buttons record a move without an amount, which is all the
  waypoint route needs: the direction a knob moves the spots is enough, the size is not.

The calibration can be saved and reloaded, in which case steps 3 is skipped entirely and
an alignment costs one move per knob - as long as nothing in the beam path has changed.
Nothing here moves any hardware.
"""

import logging
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from utilities.beam_walk import (
    KNOB_NAMES,
    OBSERVABLE_NAMES,
    BeamWalkCalibration,
    BeamWalkError,
    WalkMove,
    calibrate,
)

DEFAULT_CALIBRATION_PATH = Path("./settings/beam_walk_calibration.json")
KNOB_LABELS = ("Mirror 1 X", "Mirror 1 Y", "Mirror 2 X", "Mirror 2 Y")


class BeamWalkDialog(QDialog):
    """Doc."""

    def __init__(self, impl, parent=None):
        super().__init__(parent)
        self._impl = impl
        self._baseline = None
        self._cumulative_knobs = np.zeros(4)
        self._moves = []
        self._calibration = None

        self.setWindowTitle("Beam Walk Assistant")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(__doc__.strip().splitlines()[2].strip() or "Beam walk"))
        layout.addWidget(self._positions_group())
        layout.addWidget(self._calibration_group())
        layout.addWidget(self._correction_group())
        self._refresh_positions()

    # ------------------------------------------------------------------ layout

    def _positions_group(self) -> QGroupBox:
        box = QGroupBox("Spot positions (pixels)")
        grid = QGridLayout(box)
        self._current_labels, self._target_boxes = [], []
        grid.addWidget(QLabel("<b>now</b>"), 0, 1)
        grid.addWidget(QLabel("<b>target</b>"), 0, 2)
        for row, name in enumerate(OBSERVABLE_NAMES, start=1):
            grid.addWidget(QLabel(name.replace("_", " ")), row, 0)
            current = QLabel("-")
            spin = QDoubleSpinBox()
            spin.setRange(-1e5, 1e5)
            spin.setDecimals(1)
            grid.addWidget(current, row, 1)
            grid.addWidget(spin, row, 2)
            self._current_labels.append(current)
            self._target_boxes.append(spin)

        read_btn = QPushButton("Read cameras")
        read_btn.clicked.connect(self._refresh_positions)
        target_btn = QPushButton("Target = remembered centres")
        target_btn.setToolTip(
            "Use the red 'Remember Center' crosshairs of both displays as the target"
        )
        target_btn.clicked.connect(self._target_from_references)
        row_layout = QHBoxLayout()
        row_layout.addWidget(read_btn)
        row_layout.addWidget(target_btn)
        grid.addLayout(row_layout, len(OBSERVABLE_NAMES) + 1, 0, 1, 3)
        return box

    def _calibration_group(self) -> QGroupBox:
        box = QGroupBox("Calibration")
        grid = QGridLayout(box)

        grid.addWidget(QLabel("Knob units"), 0, 0)
        self._unit_edit = QLineEdit("turns")
        grid.addWidget(self._unit_edit, 0, 1)
        grid.addWidget(QLabel("Mode"), 0, 2)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["full (4 moves)", "decoupled (2 moves)"])
        grid.addWidget(self._mode_combo, 0, 3)

        self._amount_boxes = []
        for col, label in enumerate(KNOB_LABELS):
            grid.addWidget(QLabel(label), 1, col)
            spin = QDoubleSpinBox()
            spin.setRange(-100, 100)
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
            grid.addWidget(spin, 2, col)
            self._amount_boxes.append(spin)
        grid.addWidget(
            QLabel("<i>type how much you turned, then record - amounts accumulate</i>"), 3, 0, 1, 4
        )

        baseline_btn = QPushButton("Read baseline")
        baseline_btn.clicked.connect(self._read_baseline)
        record_btn = QPushButton("Record move")
        record_btn.clicked.connect(self._record_move)
        build_btn = QPushButton("Build calibration")
        build_btn.clicked.connect(self._build_calibration)
        clear_btn = QPushButton("Clear moves")
        clear_btn.clicked.connect(self._clear_moves)
        buttons = QHBoxLayout()
        for btn in (baseline_btn, record_btn, build_btn, clear_btn):
            buttons.addWidget(btn)
        grid.addLayout(buttons, 4, 0, 1, 4)

        unit_buttons = QHBoxLayout()
        unit_buttons.addWidget(QLabel("no amounts:"))
        for knob_idx, label in enumerate(KNOB_LABELS):
            btn = QPushButton(f"Record {label}")
            btn.setToolTip(
                f"Turn {label} by any convenient amount, then press this - the direction "
                "is all the waypoint route needs"
            )
            btn.clicked.connect(lambda _, idx=knob_idx: self._record_unit_move(idx))
            unit_buttons.addWidget(btn)
        grid.addLayout(unit_buttons, 7, 0, 1, 4)

        self._moves_list = QListWidget()
        self._moves_list.setMaximumHeight(90)
        grid.addWidget(self._moves_list, 5, 0, 1, 4)

        save_btn = QPushButton("Save calibration")
        save_btn.clicked.connect(self._save_calibration)
        load_btn = QPushButton("Load calibration")
        load_btn.clicked.connect(self._load_calibration)
        io_buttons = QHBoxLayout()
        io_buttons.addWidget(save_btn)
        io_buttons.addWidget(load_btn)
        grid.addLayout(io_buttons, 6, 0, 1, 4)
        return box

    def _correction_group(self) -> QGroupBox:
        box = QGroupBox("Correction")
        layout = QVBoxLayout(box)
        buttons = QHBoxLayout()
        compute_btn = QPushButton("Compute correction")
        compute_btn.clicked.connect(self._compute_correction)
        plan_btn = QPushButton("Plan two moves")
        plan_btn.setToolTip(
            "Steer by eye: shows a waypoint crosshair for the first mirror, no knob "
            "amounts needed"
        )
        plan_btn.clicked.connect(self._plan_steps)
        check_btn = QPushButton("Check step 1")
        check_btn.clicked.connect(self._check_step)
        for btn in (compute_btn, plan_btn, check_btn):
            buttons.addWidget(btn)
        layout.addLayout(buttons)
        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setMinimumHeight(150)
        layout.addWidget(self._output)
        return box

    # ------------------------------------------------------------------ actions

    def _say(self, text: str) -> None:
        self._output.append(text)
        logging.debug(f"Beam walk: {text}")

    def _refresh_positions(self):
        """Doc."""

        try:
            positions = self._impl.read_beam_positions()
        except Exception as exc:
            self._say(f"Could not read the cameras: {exc}")
            return None
        for label, value in zip(self._current_labels, positions):
            label.setText(f"{value:.1f}")
        return np.asarray(positions, dtype=float)

    def _target_from_references(self):
        """Doc."""

        try:
            references = self._impl.read_beam_positions(use_reference=True)
        except Exception as exc:
            self._say(
                f"Could not read the remembered centres: {exc}\n"
                "Fit the reference beam on both cameras and press 'Remember Center' first."
            )
            return
        for spin, value in zip(self._target_boxes, references):
            spin.setValue(float(value))
        self._say("Target set from the remembered centres.")

    def _read_baseline(self):
        """Doc."""

        positions = self._refresh_positions()
        if positions is None:
            return
        self._baseline = positions
        self._cumulative_knobs = np.zeros(4)
        self._moves = []
        self._moves_list.clear()
        self._say(f"Baseline: {np.round(positions, 1).tolist()}")

    def _record_move(self):
        """Doc."""

        if self._baseline is None:
            self._say("Press 'Read baseline' first.")
            return
        amounts = np.array([spin.value() for spin in self._amount_boxes], dtype=float)
        if np.allclose(amounts, 0):
            self._say("All four amounts are zero - type how much you turned.")
            return
        positions = self._refresh_positions()
        if positions is None:
            return
        self._cumulative_knobs = self._cumulative_knobs + amounts
        self._moves.append(WalkMove(self._cumulative_knobs.copy(), positions))
        moved = ", ".join(
            f"{label} {amount:+.3g}"
            for label, amount in zip(KNOB_LABELS, amounts)
            if not np.isclose(amount, 0)
        )
        shift = np.linalg.norm(positions - self._baseline)
        self._moves_list.addItem(f"{moved}   ->  {shift:.1f} px from baseline")
        [spin.setValue(0) for spin in self._amount_boxes]

    def _clear_moves(self):
        """Doc."""

        self._moves, self._baseline = [], None
        self._cumulative_knobs = np.zeros(4)
        self._moves_list.clear()
        self._say("Moves cleared.")

    def _build_calibration(self):
        """Doc."""

        mode = "full" if self._mode_combo.currentIndex() == 0 else "decoupled"
        try:
            self._calibration = calibrate(
                self._baseline if self._baseline is not None else [],
                self._moves,
                mode=mode,
                knob_unit=self._unit_edit.text() or "turns",
            )
        except BeamWalkError as exc:
            self._say(f"Calibration failed: {exc}")
            return

        cal = self._calibration
        sensitivity = ", ".join(f"{k} {v:.0f}" for k, v in cal.sensitivity().items())
        self._say(
            f"Calibration built ({mode}, {len(self._moves)} moves).\n"
            f"  sensitivity (px per {cal.knob_unit}): {sensitivity}\n"
            f"  condition number: {cal.condition_number:.1f}"
            f"{'' if cal.condition_number <= 50 else '  <- the two cameras are nearly redundant'}\n"
            f"  cross-talk: {cal.crosstalk():.2f}"
            f"{' (decoupled mode is safe)' if cal.crosstalk() < 0.1 else ' (use full mode)'}"
        )

    def _compute_correction(self):
        """Doc."""

        if self._calibration is None:
            self._say("No calibration - build or load one first.")
            return
        current = self._refresh_positions()
        if current is None:
            return
        target = np.array([spin.value() for spin in self._target_boxes], dtype=float)
        try:
            result = self._calibration.solve(current, target)
        except BeamWalkError as exc:
            self._say(f"Could not solve: {exc}")
            return

        lines = [f"Misalignment: {result['error_norm_px']:.1f} px. Turn:"]
        lines += [
            f"   {label:12s} {result['knob_moves'][name]:+8.3f} {result['knob_unit']}"
            for label, name in zip(KNOB_LABELS, KNOB_NAMES)
        ]
        lines.append(f"Predicted residual: {result['predicted_residual_px']:.2f} px")
        if not result["is_well_conditioned"]:
            lines.append(
                "WARNING: condition number "
                f"{result['condition_number']:.0f} - the cameras barely separate position "
                "from angle. Move them further apart, or focus the far one."
            )
        self._say("\n".join(lines))

    def _record_unit_move(self, knob_idx: int):
        """Record a move of one knob without an amount - direction only."""

        if self._unit_edit.text() in ("turns", ""):
            self._unit_edit.setText("recorded moves")
        for idx, spin in enumerate(self._amount_boxes):
            spin.setValue(1.0 if idx == knob_idx else 0.0)
        self._record_move()

    def _plan_steps(self):
        """Doc."""

        if self._calibration is None:
            self._say("No calibration - build or load one first.")
            return
        current = self._refresh_positions()
        if current is None:
            return
        target = np.array([spin.value() for spin in self._target_boxes], dtype=float)
        try:
            plan = self._calibration.plan_steps(current, target)
        except BeamWalkError as exc:
            self._say(f"Could not plan: {exc}")
            return
        self._plan = plan

        steer, other = plan["steer_camera"], plan["other_camera"]
        waypoint = plan["waypoint"][steer]
        try:
            self._impl.show_beam_waypoint(steer, waypoint)
        except Exception as exc:
            self._say(f"(could not draw the waypoint on camera {steer}: {exc})")

        predicted_other = plan[f"predicted_after_mirror1"]["cam1" if other == 1 else "cam2"]
        final_other = plan["final_target"]["cam1" if other == 1 else "cam2"]
        lines = [
            f"Misalignment: {plan['error_norm_px']:.1f} px. Two moves, no amounts needed:",
            f"  STEP 1 - turn MIRROR 1 until the spot on camera {steer} sits on the "
            f"yellow crosshair at ({waypoint[0]:.1f}, {waypoint[1]:.1f}).",
            f"           Camera {other} should then read about "
            f"({predicted_other[0]:.1f}, {predicted_other[1]:.1f}) by itself.",
            "           Then press 'Check step 1'.",
            f"  STEP 2 - turn MIRROR 2 until camera {other} reaches its remembered centre "
            f"at ({final_other[0]:.1f}, {final_other[1]:.1f}).",
            f"           Camera {steer} should stay on target while you do.",
            f"Predicted residual: {plan['predicted_residual_px']:.2f} px",
        ]
        if not plan["is_well_conditioned"]:
            lines.append(
                f"WARNING: condition number {self._calibration.condition_number:.0f} - the "
                "two cameras barely separate position from angle; the split into two moves "
                "is poorly defined. Focus the far camera, or separate them further."
            )
        self._say("\n".join(lines))

    def _check_step(self):
        """Doc."""

        plan = getattr(self, "_plan", None)
        if plan is None:
            self._say("Press 'Plan two moves' first.")
            return
        current = self._refresh_positions()
        if current is None:
            return

        steer, other = plan["steer_camera"], plan["other_camera"]
        reached = current[:2] if steer == 1 else current[2:]
        waypoint = np.asarray(plan["waypoint"][steer], dtype=float)
        other_now = np.asarray(current[:2] if other == 1 else current[2:], dtype=float)
        other_predicted = np.asarray(
            plan["predicted_after_mirror1"]["cam1" if other == 1 else "cam2"], dtype=float
        )
        final_other = np.asarray(
            plan["final_target"]["cam1" if other == 1 else "cam2"], dtype=float
        )

        to_waypoint = float(np.linalg.norm(np.asarray(reached, dtype=float) - waypoint))
        mismatch = float(np.linalg.norm(other_now - other_predicted))
        lines = [
            f"Camera {steer} is {to_waypoint:.1f} px from the waypoint.",
            f"Camera {other} is {mismatch:.1f} px from where step 1 predicted "
            f"(predicted ({other_predicted[0]:.1f}, {other_predicted[1]:.1f}), "
            f"actual ({other_now[0]:.1f}, {other_now[1]:.1f})).",
        ]
        if to_waypoint > 15:
            lines.append("Step 1 is not finished - keep steering mirror 1 to the yellow mark.")
        elif mismatch > max(3 * to_waypoint, 15):
            lines.append(
                "That mismatch is larger than step 1's own error, so the calibration no "
                "longer describes the mirrors: re-record the four moves."
            )
        else:
            lines.append(
                f"Step 1 done. Now turn MIRROR 2 until camera {other} reaches "
                f"({final_other[0]:.1f}, {final_other[1]:.1f}) - its red crosshair. "
                f"Camera {steer} should hold still. Remaining: "
                f"{float(np.linalg.norm(final_other - other_now)):.1f} px."
            )
            try:
                self._impl.show_beam_waypoint(steer, None)
            except Exception:
                pass
        self._say("\n".join(lines))

    def _save_calibration(self):
        """Doc."""

        if self._calibration is None:
            self._say("Nothing to save - build a calibration first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save calibration", str(DEFAULT_CALIBRATION_PATH), "JSON (*.json)"
        )
        if path:
            self._say(f"Saved to {self._calibration.save(path)}")

    def _load_calibration(self):
        """Doc."""

        path, _ = QFileDialog.getOpenFileName(
            self, "Load calibration", str(DEFAULT_CALIBRATION_PATH), "JSON (*.json)"
        )
        if not path:
            return
        try:
            self._calibration = BeamWalkCalibration.load(path)
        except (BeamWalkError, OSError, ValueError) as exc:
            self._say(f"Could not load: {exc}")
            return
        cal = self._calibration
        self._say(
            f"Loaded {Path(path).name} ({cal.mode}, measured {cal.timestamp}, "
            f"units of {cal.knob_unit}, condition number {cal.condition_number:.1f}). "
            "Valid only if nothing in the beam path has changed."
        )
