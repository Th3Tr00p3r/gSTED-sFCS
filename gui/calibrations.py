"""
Calibrations - tools for occasional alignment work, kept out of the everyday tabs.

Donut field scan
----------------
Steps the scanners over a grid around the current AO position and, at each point,
measures the donut's ring on one camera (the far-field / BFP camera, normally) over a
few frames. Then it splits each azimuthal harmonic into a part that is the same at every
scanner position and a part that grows linearly with the position (utilities.field_scan):

  constant one-sided brightening - the vortex plate's centring (or the camera lens's tilt)
  two-fold growing with position - a tilted/decentred scan or tube lens, or the scanners

To align the tube lens with its tilt knobs: scan with the knob as found, then a small
step each way (e.g. -0.25 and +0.25 turns), typing the knob setting before each scan.
'Best setting' then fits the field-dependent part linearly against the knob and gives the
setting where it is smallest. Repeat for the other knob, then re-centre the beam.

The camera's video must be on, and the donut unsaturated. The scanners are returned to
the starting position at the end, or on Stop. Each scan is saved as a CSV file in
'<data path>/<DD_MM_YYYY>/Calibrations'.
"""

import asyncio
import csv
import logging
from datetime import datetime as dt
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from utilities.field_scan import (
    FieldPoint,
    best_knob_setting,
    fit_field_dependence,
    grid_offsets,
)
from utilities.fit_tools import FitError, ring_balance

FRAME_TIMEOUT_S = 10.0

INTRO = (
    "Measures the donut on a grid of scanner positions around the current AO position and "
    "splits it into what is the same everywhere (one-sided: vortex plate centring or the "
    "camera lens's tilt) and what grows with scan position (two-fold: tube lens or "
    "scanners - the number to bring down).\n"
    "Video on, donut unsaturated. For a knob: scan as found (setting 0), then e.g. -0.25 "
    "and +0.25 turns, typing the setting before each scan, then 'Best setting for this "
    "knob'. Each scan is saved as a CSV in <data path>/<date>/Calibrations."
)  # give up on a grid point if no new frames arrive for this long


class CalibrationsDialog(QDialog):
    """A window with one tab per calibration tool."""

    def __init__(self, impl, parent=None):
        super().__init__(parent)
        self._impl = impl
        self.setWindowTitle("Calibrations")
        self.setMinimumWidth(620)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        self.field_scan = DonutFieldScanTab(impl)
        tabs.addTab(self.field_scan, "Donut field scan")
        layout.addWidget(tabs)


class DonutFieldScanTab(QWidget):
    """Doc."""

    def __init__(self, impl, parent=None):
        super().__init__(parent)
        self._impl = impl
        self._task = None
        self._should_stop = False
        self._runs = []  # dicts: label, knob, knob_value, fits, path

        layout = QVBoxLayout(self)
        layout.addWidget(self._settings_group())
        layout.addWidget(self._knob_group())

        buttons = QHBoxLayout()
        self._run_btn = QPushButton("Run field scan")
        self._run_btn.clicked.connect(self._start)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        best_btn = QPushButton("Best setting for this knob")
        best_btn.setToolTip(
            "Fit the field-dependent part of the two-fold against the knob setting, over "
            "all scans made with this knob name, and give the setting where it is smallest"
        )
        best_btn.clicked.connect(self._best_setting)
        clear_btn = QPushButton("Forget scans")
        clear_btn.clicked.connect(self._clear_runs)
        for btn in (self._run_btn, self._stop_btn, best_btn, clear_btn):
            buttons.addWidget(btn)
        layout.addLayout(buttons)

        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setMinimumHeight(260)
        layout.addWidget(self._output)
        self._say(INTRO)

    # ------------------------------------------------------------------ layout

    def _settings_group(self) -> QGroupBox:
        box = QGroupBox("Grid")
        grid = QGridLayout(box)

        self._camera_box = QComboBox()
        self._camera_box.addItems(["Camera 1", "Camera 2"])
        self._camera_box.setCurrentIndex(1)  # the far-field (BFP) camera
        grid.addWidget(QLabel("camera"), 0, 0)
        grid.addWidget(self._camera_box, 0, 1)

        self._half_x = self._spin(0.15, 0.0, 2.0, 3, " V")
        self._half_y = self._spin(0.10, 0.0, 2.0, 3, " V")
        grid.addWidget(QLabel("X half-range"), 0, 2)
        grid.addWidget(self._half_x, 0, 3)
        grid.addWidget(QLabel("Y half-range"), 1, 2)
        grid.addWidget(self._half_y, 1, 3)

        self._n_points = QSpinBox()
        self._n_points.setRange(2, 9)
        self._n_points.setValue(3)
        self._n_points.setSuffix(" per axis")
        grid.addWidget(QLabel("points"), 1, 0)
        grid.addWidget(self._n_points, 1, 1)

        self._settle = self._spin(0.5, 0.0, 10.0, 2, " s")
        self._settle.setToolTip("Wait after each scanner move, before taking frames")
        self._n_frames = QSpinBox()
        self._n_frames.setRange(1, 100)
        self._n_frames.setValue(5)
        grid.addWidget(QLabel("settle"), 2, 0)
        grid.addWidget(self._settle, 2, 1)
        grid.addWidget(QLabel("frames per point"), 2, 2)
        grid.addWidget(self._n_frames, 2, 3)
        return box

    def _knob_group(self) -> QGroupBox:
        box = QGroupBox("What is being adjusted (recorded with each scan)")
        grid = QGridLayout(box)
        self._knob_name = QLineEdit("tube lens knob A")
        self._knob_value = self._spin(0.0, -100.0, 100.0, 3, " turns")
        self._knob_value.setSingleStep(0.25)
        self._knob_value.setToolTip(
            "The knob's setting relative to where you started (e.g. -0.25, 0, +0.25)"
        )
        grid.addWidget(QLabel("knob"), 0, 0)
        grid.addWidget(self._knob_name, 0, 1)
        grid.addWidget(QLabel("setting"), 0, 2)
        grid.addWidget(self._knob_value, 0, 3)
        return box

    @staticmethod
    def _spin(value, lo, hi, decimals, suffix="") -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setSingleStep(10 ** -(decimals - 1) if decimals > 1 else 0.5)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

    # ------------------------------------------------------------------ actions

    def _say(self, text: str) -> None:
        self._output.append(text)
        logging.debug(f"Field scan: {text}")

    def _start(self):
        if self._task is not None and not self._task.done():
            return
        self._should_stop = False
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._task = self._impl._app.loop.create_task(self._run())

    def _stop(self):
        self._should_stop = True
        self._say("Stopping after this point...")

    def _clear_runs(self):
        self._runs = []
        self._say("Scans forgotten.")

    async def _run(self):
        try:
            await self._run_scan()
        except Exception as exc:  # report rather than leave the buttons stuck
            self._say(f"Field scan failed: {exc}")
            logging.exception("Field scan failed")
        finally:
            self._run_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)

    async def _run_scan(self):
        impl = self._impl
        cameras = impl.cameras or (impl._app.devices.camera_1, impl._app.devices.camera_2)
        cam_num = self._camera_box.currentIndex() + 1
        camera = cameras[cam_num - 1]
        if not getattr(camera, "is_in_video_mode", False):
            self._say(f"Switch camera {cam_num}'s video on first.")
            return

        main = impl.main_gui
        centre = tuple(getattr(main, f"{ax}AOV").value() for ax in "xyz")
        offsets = grid_offsets(self._half_x.value(), self._half_y.value(), self._n_points.value())
        label = f"{self._knob_name.text().strip()} = {self._knob_value.value():+g}"
        self._say(
            f"\nScanning {len(offsets)} points around AO ({centre[0]:.3f}, {centre[1]:.3f}) V "
            f"on camera {cam_num}  [{label}]"
        )

        points = []
        try:
            for idx, (dx, dy) in enumerate(offsets, start=1):
                if self._should_stop:
                    self._say("Stopped.")
                    break
                if getattr(camera, "error_dict", None):
                    self._say(
                        f"Camera {cam_num} stopped responding (red LED) - scan aborted. "
                        "Replug its USB cable, press Restart Camera, and run it again."
                    )
                    self._should_stop = True
                    break
                impl.move_scanners("XY", (centre[0] + dx, centre[1] + dy, centre[2]))
                await asyncio.sleep(self._settle.value())
                # each point starts with a search of the whole frame: the donut has moved
                point = await self._measure_point(camera, (dx, dy))
                points.append(point)
                if point.ok:
                    h1, h2 = point.harmonics[1], point.harmonics[2]
                    self._say(
                        f"  {idx:2d}/{len(offsets)}  AO offset ({dx:+.3f}, {dy:+.3f}) V  "
                        f"core ({point.core_px[0]:.1f}, {point.core_px[1]:.1f}) px  "
                        f"one-sided {200 * abs(h1):.1f} %  two-fold {200 * abs(h2):.1f} %"
                    )
                else:
                    self._say(
                        f"  {idx:2d}/{len(offsets)}  AO offset ({dx:+.3f}, {dy:+.3f}) V  "
                        "no donut found (broken ring, saturated or off the camera?)"
                    )
        finally:
            impl.move_scanners("XY", centre)  # back to where we started, whatever happened

        if self._should_stop:
            return
        self._report(points, label, cam_num)

    async def _measure_point(self, camera, offset):
        """Average the ring harmonics over a few new frames at one grid position."""

        near, ring_radius = None, None
        n_wanted = self._n_frames.value()
        harmonic_sums = {}
        cores = []
        # start from the frame count now, so only frames taken AFTER the move are used
        previous = getattr(camera, "frame_count", None)
        loop = asyncio.get_event_loop()
        deadline = loop.time() + FRAME_TIMEOUT_S
        n_seen = 0
        while n_seen < n_wanted and loop.time() < deadline:
            await asyncio.sleep(0.03)
            frame = getattr(camera, "last_snapshot", None)
            count = getattr(camera, "frame_count", None)
            if count is not None:
                # the camera counts its frames: a dark, unchanging image is still new
                if frame is None or count == previous:
                    continue
                previous = count
            else:
                if frame is None or (previous is not None and np.array_equal(frame, previous)):
                    continue  # no new frame yet
                previous = np.array(frame, copy=True)
            n_seen += 1
            try:
                grey = camera._as_grayscale_array(frame)
                res = (
                    ring_balance(grey, near=near, ring_radius=ring_radius)
                    if near is not None
                    else ring_balance(grey)
                )
            except FitError:
                near, ring_radius = None, None  # search the whole frame next time
                continue
            near, ring_radius = res["core"], res["ring_radius"]
            cores.append(res["core"])
            for m, c in res["harmonics"].items():
                harmonic_sums[m] = harmonic_sums.get(m, 0) + c

        point = FieldPoint(offset_v=offset)
        if cores:
            point.core_px = tuple(np.mean(cores, axis=0))
            point.harmonics = {m: s / len(cores) for m, s in harmonic_sums.items()}
            point.n_frames = len(cores)
        return point

    def _report(self, points, label, cam_num):
        good = [p for p in points if p.ok]
        if len(good) < 4:
            self._say(f"Only {len(good)} points with a donut - not enough to analyse.")
            return
        fits = {}
        for m in (1, 2, 3):
            try:
                fits[m] = fit_field_dependence(points, m)
            except ValueError as exc:
                self._say(f"Harmonic {m}: {exc}")
        if 2 not in fits:
            return

        centres = [p for p in good if p.offset_v == (0.0, 0.0)]
        drift = (
            np.hypot(*np.subtract(centres[-1].core_px, centres[0].core_px))
            if len(centres) >= 2
            else float("nan")
        )
        f1, f2 = fits.get(1), fits[2]
        lines = [f"Result [{label}], {len(good)} points:"]
        lines.append(
            f"  TWO-FOLD GROWING WITH SCAN POSITION: {f2.field_pct:.1f} %  at the grid corner "
            "(tube lens / scanners - minimise this)"
        )
        lines.append(
            f"  two-fold, same everywhere: {f2.constant_pct:.1f} %, axis {f2.constant_angle_deg:+.0f} deg"
        )
        if f1 is not None:
            lines.append(
                f"  one-sided, same everywhere: {f1.constant_pct:.1f} %, bright side "
                f"{f1.constant_angle_deg:+.0f} deg (plate centring / camera-lens tilt)"
            )
            lines.append(f"  one-sided, growing with position: {f1.field_pct:.1f} %")
        lines.append(
            f"  fit residual: {200 * f2.residual_rms:.1f} % (two-fold); centre drift during "
            f"the scan: {drift:.1f} px"
        )
        self._say("\n".join(lines))

        path = self._save_csv(points, fits, label, cam_num)
        self._runs.append(
            dict(
                label=label,
                knob=self._knob_name.text().strip(),
                knob_value=self._knob_value.value(),
                fits=fits,
                path=path,
            )
        )

    def _best_setting(self):
        knob = self._knob_name.text().strip()
        runs = [r for r in self._runs if r["knob"] == knob and 2 in r["fits"]]
        values = sorted({r["knob_value"] for r in runs})
        if len(values) < 2:
            self._say(
                f"Need scans at two or more settings of '{knob}' (have {len(values)}). "
                "Change the setting box before each scan."
            )
            return
        try:
            k_best, predicted, per_unit = best_knob_setting(
                [r["knob_value"] for r in runs], [r["fits"][2].field_vector for r in runs]
            )
        except ValueError as exc:
            self._say(str(exc))
            return
        measured = ", ".join(
            f"{r['knob_value']:+g}: {r['fits'][2].field_pct:.1f} %" for r in runs
        )
        lo, hi = min(values), max(values)
        if per_unit * (hi - lo) < 2.0:
            self._say(
                f"'{knob}': measured {measured}\n"
                f"  -> across the settings scanned this knob changes the field-dependent "
                f"two-fold by only {per_unit * (hi - lo):.1f} %, within the noise. Either it "
                "is not what causes it, or larger steps are needed to see the effect."
            )
            return
        caution = (
            "  (outside the range scanned - step there and scan again to confirm)"
            if not lo <= k_best <= hi
            else ""
        )
        self._say(
            f"'{knob}': measured {measured}\n"
            f"  -> best setting {k_best:+.3f}, where the field-dependent two-fold should drop "
            f"to ~{predicted:.1f} %{caution}\n"
            f"  (it changes by {per_unit:.1f} % per unit of this knob; what is left at the best "
            "setting is for the other knob)"
        )

    def _save_csv(self, points, fits, label, cam_num):
        try:
            folder = (
                Path(self._impl._gui.settings.dataPath.text())
                / dt.now().strftime("%d_%m_%Y")
                / "Calibrations"
            )
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"donut_field_scan_{dt.now().strftime('%H%M%S')}.csv"
            with open(path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([f"# {label}; camera {cam_num}"])
                for m, f in fits.items():
                    writer.writerow(
                        [
                            f"# harmonic {m}: constant {f.constant_pct:.2f} % at "
                            f"{f.constant_angle_deg:+.1f} deg, field-dependent "
                            f"{f.field_pct:.2f} % at the grid corner"
                        ]
                    )
                header = ["ao_dx_V", "ao_dy_V", "core_x_px", "core_y_px", "n_frames"]
                for m in (1, 2, 3):
                    header += [f"h{m}_re", f"h{m}_im"]
                writer.writerow(header)
                for p in points:
                    row = [p.offset_v[0], p.offset_v[1]]
                    row += list(p.core_px) if p.ok else ["", ""]
                    row.append(p.n_frames)
                    for m in (1, 2, 3):
                        c = p.harmonics.get(m)
                        row += [c.real, c.imag] if c is not None else ["", ""]
                    writer.writerow(row)
            self._say(f"  saved to {path}")
            return path
        except Exception as exc:
            self._say(f"  (could not save the CSV: {exc})")
            return None
