"""
The 'ImageFCS' tab: image scanning done with the scanning-FCS (angular) routines.

The scan is the angular sFCS pattern with image-like line spacing (nanometres rather than
micrometres) and, optionally, several sweeps of each line before stepping to the next. The
photons are recorded exactly as in a solution sFCS measurement, and the image is rebuilt
from their time-stamps (data_analysis/image_fcs.py), so one file holds both the image and
the time series for the correlations.

The tab is built in code and appended to the main window's tab widget. Its input widgets
get object names and become attributes of the main window, so loadouts save and restore
them like any widget drawn in Qt Designer.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

from data_analysis import image_fcs
from logic.scan_patterns import ScanPatternAO
from utilities.display import GuiDisplay

TAB_TITLE = "sFCS ImageFCS"
MEAS_TYPE = "SFCSImageFCS"
AO_LIMITS_V = (-5.0, 5.0)  # X and Y galvo command range
Z_AO_LIMITS_V = (0.0, 10.0)  # Z piezo command range
SLOW_AXES = {"In plane (XY image)": "xy", "Z (XZ section)": "z"}

# Standard scans, as in the image tab: choosing one fills the 'Scan' boxes, which can then
# be edited freely. Keys are the names of the spin-box attributes below.
PRESETS = {
    # a single nucleoid: fast axis 10 um, slow axis 5 um, lines 50 nm apart
    "Nucleoid scan": dict(
        field_len=10,
        field_width=5,
        line_spacing=50,
        line_repeats=2,
        speed=500,
        lin_frac=0.8,
        slow_axis="xy",
    ),
    # a section through the nucleoid, to find its mid-plane: the same line, stepped 5 um
    # along z in 0.25 um steps (the axial resolution does not justify finer ones)
    "Nucleoid XZ scan": dict(
        field_len=10,
        field_width=5,
        line_spacing=250,
        line_repeats=2,
        speed=500,
        lin_frac=0.8,
        slow_axis="z",
    ),
    # the field of the image tab's 'MFC - XY compartment' preset (70 x 70 um), with lines
    # 500 nm apart - at 50 nm it would be 1400 lines
    "Compartment scan": dict(
        field_len=70,
        field_width=70,
        line_spacing=500,
        line_repeats=1,
        speed=5000,
        lin_frac=0.9,
        slow_axis="xy",
    ),
}


class ImageFCSTab(QtWidgets.QWidget):
    """The tab's widgets and everything they do."""

    def __init__(self, main_gui, app):
        super().__init__(main_gui)
        self._main = main_gui
        self._app = app
        self.setObjectName("imgFcsTab")

        self.last_image: Optional[image_fcs.ScanImage] = None
        self.last_origin_ao_v: Optional[Tuple[float, ...]] = None
        self.last_um_v_ratio: Optional[Tuple[float, ...]] = None
        self.pending_origin_ao_v: Optional[Tuple[float, float, Optional[float]]] = None
        self.is_measuring = False

        self._build()
        self._connect()
        self.update_pattern()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _spin(self, name, value, low, high, step=1.0, decimals=None, suffix=""):
        """A spin box that is also an attribute of the main window (for loadouts)."""

        if decimals is None:
            box = QtWidgets.QSpinBox(self)
            low, high, value, step = int(low), int(high), int(value), int(step)
        else:
            box = QtWidgets.QDoubleSpinBox(self)
            box.setDecimals(decimals)
        box.setObjectName(name)
        box.setRange(low, high)
        box.setSingleStep(step)
        box.setValue(value)
        box.setKeyboardTracking(False)
        if suffix:
            box.setSuffix(suffix)
        setattr(self._main, name, box)
        return box

    def _named(self, widget, name):
        widget.setObjectName(name)
        setattr(self._main, name, widget)
        return widget

    def _build(self):  # noqa: C901
        # --- the scan
        scan_box = QtWidgets.QGroupBox("Scan", self)
        scan_form = QtWidgets.QFormLayout(scan_box)
        self.preset = self._named(QtWidgets.QComboBox(self), "imgFcsPreset")
        self.preset.addItems(list(PRESETS))
        self.preset.setToolTip(
            "Choosing a standard scan fills the boxes below; they stay editable. (Loading a "
            "loadout restores the boxes as they were saved, not the standard values.)"
        )
        scan_form.addRow("Standard scan", self.preset)
        self.field_len = self._spin("imgFcsFieldLen", 10, 0.2, 500, 1, 2, " um")
        self.field_len.setToolTip("Length of the linear part of each line (the image width).")
        self.field_width = self._spin("imgFcsFieldWidth", 5, 0.2, 500, 1, 2, " um")
        self.field_width.setToolTip("Extent across the lines (the image height).")
        self.line_spacing = self._spin("imgFcsLineSpacing", 50, 5, 10000, 10, 0, " nm")
        self.line_spacing.setToolTip(
            "Distance between neighbouring lines. Scanning FCS uses 1-10 um; an image needs "
            "less than half the focal spot, i.e. 50-100 nm."
        )
        self.line_repeats = self._spin("imgFcsLineRepeats", 2, 1, 200)
        self.line_repeats.setToolTip(
            "Sweeps over each line before stepping to the next one. More sweeps bring the "
            "shortest correlation lag at a spot down from the frame time to the line time."
        )
        # the same default as the scanning-FCS tab's angle
        self.angle = self._spin("imgFcsAngle", 145, 0, 180, 5, suffix=" deg")
        self.angle.setToolTip("Direction of the lines (the fast axis) in the XY plane.")
        self.slow_axis = self._named(QtWidgets.QComboBox(self), "imgFcsSlowAxis")
        self.slow_axis.addItems(list(SLOW_AXES))
        self.slow_axis.setToolTip(
            "Where the lines are stepped: sideways in the XY plane, or along z with the "
            "piezo - an X-Z section along the scan angle. 'Width' is then the z range."
        )
        self.speed = self._spin("imgFcsSpeed", 500, 1, 100000, 50, 1, " um/s")
        self.ao_freq = self._spin("imgFcsAoSampFreq", 10000, 100, 1000000, 1000, suffix=" Hz")
        self.lin_frac = self._spin("imgFcsLinFrac", 0.8, 0.1, 0.95, 0.05, 2)
        scan_form.addRow("Line length", self.field_len)
        scan_form.addRow("Slow axis", self.slow_axis)
        scan_form.addRow("Width across lines", self.field_width)
        scan_form.addRow("Line spacing", self.line_spacing)
        scan_form.addRow("Sweeps per line", self.line_repeats)
        scan_form.addRow("Angle", self.angle)
        scan_form.addRow("Speed", self.speed)
        scan_form.addRow("AO sampling", self.ao_freq)
        scan_form.addRow("Linear fraction", self.lin_frac)
        self.scan_info = QtWidgets.QLabel("", self)
        self.scan_info.setWordWrap(True)
        self.scan_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.scan_info.setMinimumHeight(self.scan_info.fontMetrics().lineSpacing() * 5)
        self.scan_info.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        scan_form.addRow(self.scan_info)

        # --- the acquisition
        acq_box = QtWidgets.QGroupBox("Acquisition", self)
        acq_form = QtWidgets.QFormLayout(acq_box)
        self.laser_mode = self._named(QtWidgets.QComboBox(self), "imgFcsLaserMode")
        self.laser_mode.addItems(["Exc", "Dep", "Sted"])
        self.live_frames = self._spin("imgFcsLiveFrames", 1, 1, 1000)
        self.live_frames.setToolTip("Frames collected for each refresh of the live image.")
        self.duration = self._spin("imgFcsDuration", 60, 1, 1e6, 10, 1, " s")
        self.max_file_size = self._spin("imgFcsMaxFileSize", 50, 1, 2000, 10, 0, " Mb")
        self.file_template = self._named(QtWidgets.QLineEdit("imgfcs", self), "imgFcsFileTemplate")
        acq_form.addRow("Lasers", self.laser_mode)
        acq_form.addRow("Live: frames per image", self.live_frames)
        acq_form.addRow("Acquire: duration", self.duration)
        acq_form.addRow("Acquire: max. file size", self.max_file_size)
        acq_form.addRow("File template", self.file_template)
        buttons = QtWidgets.QHBoxLayout()
        self.live_button = QtWidgets.QPushButton("Live", self)
        self.live_button.setToolTip("Scan and refresh the image until stopped. Nothing is saved.")
        self.acquire_button = QtWidgets.QPushButton("Acquire", self)
        self.acquire_button.setToolTip(
            "Scan for the set duration and save the raw files with the solution sFCS data."
        )
        buttons.addWidget(self.live_button)
        buttons.addWidget(self.acquire_button)
        acq_form.addRow(buttons)

        # --- the image
        img_box = QtWidgets.QGroupBox("Image", self)
        img_form = QtWidgets.QFormLayout(img_box)
        self.auto_offset = self._named(
            QtWidgets.QCheckBox("Find scan offset from the data", self), "imgFcsAutoOffset"
        )
        self.auto_offset.setChecked(True)
        self.auto_offset.setToolTip(
            "The photon clock starts at the first detected photon, not at the start of the "
            "scan, and the scanners lag the command. Both are found by matching forward and "
            "backward sweeps. Untick to use the value below as it is."
        )
        self.offset = self._spin("imgFcsOffset", 0, -1e5, 1e5, 0.5, 2, " samples")
        self.offset.setToolTip(
            "Offset added to the photon times, in AO samples. With the box above ticked this "
            "shows the offset last found, and is used when the data cannot decide."
        )
        self.saturation = self._spin("imgFcsSaturation", 99.9, 50, 100, 0.1, 1, " %")
        self.saturation.setToolTip("Percentile of the pixel values mapped to white.")
        self.centre_button = QtWidgets.QPushButton("Centre scan on cursor", self)
        self.centre_button.setToolTip(
            "Click in the image, then press: the scan centre moves to the crosshair. During a "
            "measurement the move is made at the start of the next file/refresh."
        )
        img_form.addRow(self.auto_offset)
        img_form.addRow("Scan offset", self.offset)
        img_form.addRow("Saturate at", self.saturation)
        img_form.addRow(self.centre_button)
        self.status = QtWidgets.QLabel("No image yet.", self)
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.status.setMinimumHeight(self.status.fontMetrics().lineSpacing() * 7)
        self.status.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        img_form.addRow(self.status)

        controls = QtWidgets.QVBoxLayout()
        controls.addWidget(scan_box)
        controls.addWidget(acq_box)
        controls.addWidget(img_box)
        controls.addStretch(1)
        controls_holder = QtWidgets.QWidget(self)
        controls_holder.setLayout(controls)
        controls_holder.setFixedWidth(350)

        # --- displays
        image_layout = QtWidgets.QVBoxLayout()
        self.image_display = GuiDisplay(image_layout, self._main)
        pattern_layout = QtWidgets.QVBoxLayout()
        self.pattern_display = GuiDisplay(pattern_layout)
        pattern_holder = QtWidgets.QWidget(self)
        pattern_holder.setLayout(pattern_layout)
        pattern_holder.setMaximumWidth(300)
        pattern_holder.setMinimumWidth(180)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(controls_holder)
        layout.addLayout(image_layout, 1)
        layout.addWidget(pattern_holder)

    def _connect(self):
        for box in (
            self.field_len,
            self.field_width,
            self.line_spacing,
            self.line_repeats,
            self.angle,
            self.speed,
            self.ao_freq,
            self.lin_frac,
        ):
            box.valueChanged.connect(self.update_pattern)
        self.slow_axis.currentTextChanged.connect(self.update_pattern)
        self.preset.activated[str].connect(self.apply_preset)
        self.live_button.released.connect(lambda: self.start_or_stop(live=True))
        self.acquire_button.released.connect(lambda: self.start_or_stop(live=False))
        self.centre_button.released.connect(self.centre_on_cursor)
        self.saturation.valueChanged.connect(self.redraw)

    # ------------------------------------------------------------------
    # scan parameters and the pattern preview
    # ------------------------------------------------------------------
    def apply_preset(self, name: str) -> None:
        """Fill the scan boxes with a standard scan (the angle is left as it is)."""

        for attr, value in PRESETS.get(name, {}).items():
            box = getattr(self, attr)
            with QtCore.QSignalBlocker(box):
                if attr == "slow_axis":
                    box.setCurrentIndex(list(SLOW_AXES.values()).index(value))
                else:
                    box.setValue(value)
        self.update_pattern()
        logging.debug(f"ImageFCS preset chosen: '{name}'")

    def scan_params(self) -> dict:
        """The tab's settings as 'scan_params' for the angular pattern."""

        max_scan_freq_hz = 1e9
        with suppress(Exception):
            max_scan_freq_hz = self._app.gui.settings.maxScanFreq.value()
        linear_fraction = self.lin_frac.value()
        return dict(
            pattern="angular",
            angle_deg=self.angle.value(),
            linear_fraction=linear_fraction,
            # the pattern is specified by the total line length, turnarounds included
            max_line_len_um=self.field_len.value() / linear_fraction,
            scan_width_um=self.field_width.value(),
            line_shift_um=self.line_spacing.value() * 1e-3,
            ao_sampling_freq_hz=self.ao_freq.value(),
            max_scan_freq_Hz=max_scan_freq_hz,
            speed_um_s=self.speed.value(),
            line_repeats=self.line_repeats.value(),
            slow_axis=SLOW_AXES.get(self.slow_axis.currentText(), "xy"),
        )

    def _scanner_calibration(self):
        """(um/V ratios, current AO voltages) - placeholders if there are no scanners."""

        um_v_ratio, origin_ao_v = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
        with suppress(Exception):
            um_v_ratio = tuple(self._app.devices.scanners.um_v_ratio)
        with suppress(Exception):
            origin_ao_v = tuple(self._app.devices.scanners.ao_int)
        return um_v_ratio, origin_ao_v

    def frame_time_s(self) -> Optional[float]:
        """Duration of one scan frame with the current settings (None if they are invalid)."""

        try:
            _, params = ScanPatternAO(
                "angular", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), self.scan_params()
            ).calculate_pattern()
        except Exception:
            return None
        frame_samples = (params["n_passes"] + 1) * params["samples_per_line"]
        return frame_samples / params["ao_sampling_freq_hz"]

    def update_pattern(self, *_):
        """Draw the pattern and report what the settings amount to."""

        um_v_ratio, origin_ao_v = self._scanner_calibration()
        try:
            ao, params = ScanPatternAO(
                "angular", um_v_ratio, origin_ao_v, self.scan_params()
            ).calculate_pattern()
        except Exception:
            self.scan_info.setText(
                "These settings do not make a valid scan (the line frequency is probably "
                "above the scanners' limit - lower the speed or lengthen the line)."
            )
            with suppress(Exception):
                self.pattern_display.display_patterns([], [])
            return

        samples_per_line = params["samples_per_line"]
        frame_samples = (params["n_passes"] + 1) * samples_per_line
        pixel_nm = params["eff_speed_um_s"] / params["ao_sampling_freq_hz"] * 1e3
        frame_s = frame_samples / params["ao_sampling_freq_hz"]
        self.scan_info.setText(
            f"{params['n_lines']} lines x {params['line_repeats']} sweeps, "
            f"{params['line_freq_hz'] * 2:.1f} sweeps/s "
            f"({samples_per_line / params['ao_sampling_freq_hz'] * 1e3:.1f} ms each)\n"
            f"{pixel_nm:.0f} nm per AO sample along the line\n"
            f"frame: {frame_s:.2f} s, {frame_samples:,} AO samples"
            + (
                f"\nz: {np.ptp(ao[2, :frame_samples]) * um_v_ratio[2]:.2f} um in "
                f"{params['line_shift_um'] * 1e3:.0f} nm steps"
                if params.get("slow_axis") == "z"
                else ""
            )
        )
        with suppress(Exception):
            # one frame only - the buffer may hold several
            self.pattern_display.display_patterns(ao[0, :frame_samples], ao[1, :frame_samples])

    # ------------------------------------------------------------------
    # running measurements
    # ------------------------------------------------------------------
    def measurement_kwargs(self, live: bool) -> dict:
        """What SolutionMeasurementProcedure needs beyond the scan parameters."""

        if live:
            frame_s = self.frame_time_s() or 1.0
            duration_s = max(frame_s * self.live_frames.value(), 0.2)
        else:
            duration_s = self.duration.value()
        return dict(
            file_template=self.file_template.text() or "imgfcs",
            scan_type="angular",
            repeat=live,
            final=False,
            duration=duration_s,
            duration_units="seconds",
            max_file_size_mb=self.max_file_size.value(),
            should_plot=True,  # here: show the image after each file
            should_fit=False,
            should_accumulate_corrfuncs=False,
            stage_pattern="None",
            floating_z_amplitude_um=0,
            image_fcs_tab=self,
        )

    def start_or_stop(self, live: bool):
        """Start a live or a saved measurement - or stop the one that is running."""

        if not self.is_measuring and self.frame_time_s() is None:
            self.status.setText("Cannot start: the scan settings are not valid.")
            return
        self._app.loop.create_task(
            self._main.impl.fire_single_meas(
                meas_type=MEAS_TYPE,
                laser_mode=self.laser_mode.currentText(),
                should_turn_off_dep=False,
                live=live,
            )
        )

    def measurement_started(self, live: bool):
        self.is_measuring = True
        self.pending_origin_ao_v = None
        (self.live_button if live else self.acquire_button).setText("Stop")
        (self.acquire_button if live else self.live_button).setEnabled(False)

    def measurement_stopped(self):
        self.is_measuring = False
        self.pending_origin_ao_v = None
        self.live_button.setText("Live")
        self.acquire_button.setText("Acquire")
        self.live_button.setEnabled(True)
        self.acquire_button.setEnabled(True)

    # ------------------------------------------------------------------
    # the image
    # ------------------------------------------------------------------
    def image_options(self) -> dict:
        """Keyword arguments for image_fcs.scan_image, from the 'Image' box."""

        if self.auto_offset.isChecked():
            return dict(offset_samples=None, fallback_offset_samples=self.offset.value())
        return dict(offset_samples=self.offset.value())

    def show_image(self, image: image_fcs.ScanImage, origin_ao_v, um_v_ratio):
        """Display a rebuilt image and remember what is needed to aim at a point in it."""

        self.last_image = image
        self.last_origin_ao_v = tuple(origin_ao_v)
        self.last_um_v_ratio = tuple(um_v_ratio)

        found = self.auto_offset.isChecked() and image.offset_quality >= 5
        if found:
            with QtCore.QSignalBlocker(self.offset):
                self.offset.setValue(image.offset_samples)
        if not self.auto_offset.isChecked():
            how = "set by hand"
        elif found:
            how = "found from the data"
        else:
            how = "NOT found (too little structure) - using the value in the box"
        rate = image.rate_khz
        self.status.setText(
            f"{image.counts.shape[0]} x {image.counts.shape[1]} pixels of "
            f"{image.pixel_um * 1e3:.0f} nm, {image.duration_s:.1f} s of data\n"
            f"scan offset {image.offset_samples:+.2f} samples, {how}\n"
            f"count rate: mean {rate.mean():.1f} kHz, max {rate.max():.1f} kHz"
        )
        self.redraw()

    def redraw(self, *_):
        if self.last_image is None:
            return
        rate = self.last_image.rate_khz
        v_max = float(np.percentile(rate, self.saturation.value())) or float(rate.max()) or 1.0
        self.image_display.nm_per_px = self.last_image.pixel_um * 1e3
        self.image_display.display_image(
            rate, cursor=True, imshow_kwargs=dict(cmap="bone", vmin=0, vmax=v_max)
        )
        with suppress(AttributeError, IndexError):
            # the pixels are square (as long as the line spacing), so show them square -
            # the display otherwise stretches every image to a square box
            self.image_display.axes[0].set_aspect("equal")
            self.image_display.canvas.draw_idle()

    def cursor_ao_v(self) -> Optional[Tuple[float, float, Optional[float]]]:
        """
        AO voltages (x, y, z) of the point under the crosshair of the displayed image. z is
        None for an XY image, which says nothing about z.
        """

        if self.last_image is None:
            return None
        try:
            col, row = self.image_display.axes[0].cursor.pos
        except (AttributeError, IndexError):
            return None
        x_um, y_um, z_um = self.last_image.pixel_to_xyz_um(col, row)
        x_v = self.last_origin_ao_v[0] + x_um / self.last_um_v_ratio[0]
        y_v = self.last_origin_ao_v[1] + y_um / self.last_um_v_ratio[1]
        z_v = None
        if self.last_image.geom.slow_axis == "z":
            z_v = self.last_origin_ao_v[2] + z_um / self.last_um_v_ratio[2]
            z_v = float(np.clip(z_v, *Z_AO_LIMITS_V))
        return (float(np.clip(x_v, *AO_LIMITS_V)), float(np.clip(y_v, *AO_LIMITS_V)), z_v)

    def centre_on_cursor(self):
        """Move the centre of the scan to the crosshair."""

        target = self.cursor_ao_v()
        if target is None:
            self.status.setText("Take an image and click in it first.")
            return

        where = f"({target[0]:.4f}, {target[1]:.4f}) V"
        if target[2] is not None:
            where = f"({target[0]:.4f}, {target[1]:.4f}, z {target[2]:.4f}) V"
        if self.is_measuring:
            # the scanners are busy with the pattern - the measurement picks this up
            # between files and rebuilds the pattern around the new centre
            self.pending_origin_ao_v = target
            self.status.setText(f"Scan centre moves to {where} with the next file/refresh.")
        else:
            self._main.xAOV.setValue(target[0])
            self._main.yAOV.setValue(target[1])
            if target[2] is not None:
                self._main.zAOV.setValue(target[2])
            self._main.impl.move_scanners("XY" if target[2] is None else "XYZ")
            self.status.setText(f"Scanners moved to {where} - the next scan is centred there.")
        logging.debug(f"ImageFCS: scan centre -> {target} V")

    def take_pending_origin(self) -> Optional[Tuple[float, float, Optional[float]]]:
        """Hand a requested new scan centre to the running measurement (once)."""

        target, self.pending_origin_ao_v = self.pending_origin_ao_v, None
        return target


def add_image_fcs_tab(main_gui, app) -> ImageFCSTab:
    """Create the tab and append it to the main window's tab widget."""

    tab = ImageFCSTab(main_gui, app)
    main_gui.imgFcsTab = tab
    main_gui.tabWidget.addTab(tab, TAB_TITLE)
    return tab
