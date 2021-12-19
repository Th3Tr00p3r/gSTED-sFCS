""" GUI windows implementations module. """

import logging
import re
import sys
import webbrowser
from collections import namedtuple
from collections.abc import Iterable
from contextlib import contextmanager, suppress
from datetime import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QWidget

import gui.dialog as dialog
import gui.widgets as wdgts
import logic.measurements as meas
from data_analysis.correlation_function import (
    ImageSFCSMeasurement,
    SFCSExperiment,
    SolutionSFCSMeasurement,
)
from logic.scan_patterns import ScanPatternAO
from utilities import file_utilities, fit_tools, helper
from utilities.display import GuiDisplay, auto_brightness_and_contrast
from utilities.errors import DeviceError, err_hndlr


class MainWin:
    """Doc."""

    ####################
    ## General
    ####################
    def __init__(self, gui, app):
        """Doc."""

        self._app = app
        self._gui = gui
        self.cameras = None

    def close(self, event):
        """Doc."""

        self._app.exit_app(event)

    def restart(self) -> None:
        """Restart all devices (except camera) and the timeout loop."""

        pressed = dialog.Question(txt="Are you sure?", title="Restarting Application").display()
        if pressed == dialog.YES:
            self._app.loop.create_task(self._app.clean_up_app(restart=True))

    def save(self) -> None:
        """Doc."""

        file_path, _ = QFileDialog.getSaveFileName(
            self._gui,
            "Save Loadout",
            str(self._app.LOADOUT_DIR_PATH),
        )
        if file_path != "":
            wdgts.write_gui_to_file(self._gui, wdgts.MAIN_TYPES, file_path)
            logging.debug(f"Loadout saved as: '{file_path}'")

    def load(self, file_path="") -> None:
        """Doc."""

        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self._gui,
                "Load Loadout",
                str(self._app.LOADOUT_DIR_PATH),
            )
        if file_path != "":
            wdgts.read_file_to_gui(file_path, self._gui)
            logging.debug(f"Loadout loaded: '{file_path}'")

    def device_toggle(
        self, nick, toggle_mthd="toggle", state_attr="state", leave_on=False, leave_off=False
    ) -> bool:
        """Returns False in case operation fails"""

        was_toggled = False  # assume didn't work
        dvc = getattr(self._app.devices, nick)  # get device

        try:  # try probing device state
            is_dvc_on = getattr(dvc, state_attr)

        except AttributeError:  # faulty device
            exc = DeviceError(f"{dvc.log_ref} was not properly initialized. Cannot toggle.")
            if nick == "stage":
                err_hndlr(exc, sys._getframe(), locals(), lvl="warning")
            else:
                raise exc

        # have device state, keep going
        else:
            # no need to do anything if already ON/OFF and meant to stay that way
            if (leave_on and is_dvc_on) or (leave_off and not is_dvc_on):
                was_toggled = True

            # otherwise, need to toggle!
            else:
                # switch ON
                if not is_dvc_on:
                    try:
                        getattr(dvc, toggle_mthd)(True)
                    except DeviceError as exc:
                        err_hndlr(exc, sys._getframe(), locals(), lvl="warning")
                    else:
                        # if managed to turn ON
                        if is_dvc_on := getattr(dvc, state_attr):
                            logging.debug(f"{dvc.log_ref} toggled ON")
                            if nick == "stage":
                                self._gui.stageButtonsGroup.setEnabled(True)
                            was_toggled = True

                # switch OFF
                else:
                    with suppress(DeviceError):
                        getattr(dvc, toggle_mthd)(False)

                    if not (is_dvc_on := getattr(dvc, state_attr)):
                        # if managed to turn OFF
                        logging.debug(f"{dvc.log_ref} toggled OFF")

                        if nick == "stage":
                            self._gui.stageButtonsGroup.setEnabled(False)

                        if nick == "dep_laser":
                            # set curr/pow values to zero when depletion is turned OFF
                            self._gui.depActualCurr.setValue(0)
                            self._gui.depActualPow.setValue(0)

                        was_toggled = True

        return was_toggled

    def led_clicked(self, led_obj_name) -> None:
        """Doc."""

        led_name_to_nick_dict = {
            "ledExc": "exc_laser",
            "ledTdc": "TDC",
            "ledDep": "dep_laser",
            "ledShutter": "dep_shutter",
            "ledStage": "stage",
            "ledUm232h": "UM232H",
            "ledScn": "scanners",
            "ledCounter": "photon_detector",
            "ledPxlClk": "pixel_clock",
            "ledCam1": "camera_1",
            "ledCam2": "camera_2",
        }

        dvc_nick = led_name_to_nick_dict[led_obj_name]
        error_dict = getattr(self._app.devices, dvc_nick).error_dict
        if error_dict is not None:
            dialog.Error(
                **error_dict,
                custom_title=getattr(self._app.devices, dvc_nick).log_ref,
            ).display()

    def dep_sett_apply(self):
        """Doc."""

        with suppress(DeviceError):
            if self._gui.currMode.isChecked():  # current mode
                val = self._gui.depCurr.value()
                self._app.devices.dep_laser.set_current(val)
            else:  # power mode
                val = self._gui.depPow.value()
                self._app.devices.dep_laser.set_power(val)

    def move_scanners(self, axes_used: str = "XYZ", destination=None) -> None:
        """Doc."""

        scanners_dvc = self._app.devices.scanners

        if destination is None:
            # if no destination is specified, use the AO from the GUI
            destination = tuple(getattr(self._gui, f"{ax}AOV").value() for ax in "xyz")

        data = []
        type_str = ""
        for ax, vltg_V in zip("XYZ", destination):
            if ax in axes_used:
                data.append([vltg_V])
                type_str += ax

        try:
            scanners_dvc.start_write_task(data, type_str)
            scanners_dvc.start_continuous_read_task()  # restart cont. reading
        except DeviceError as exc:
            err_hndlr(exc, sys._getframe(), locals())

        logging.debug(f"{self._app.devices.scanners.log_ref} were moved to {str(destination)} V")

    # TODO: implement a 'go_to_last_position' - same as origin, just need to save last location each time (if it's not the origin)
    def go_to_origin(self, which_axes: str) -> None:
        """Doc."""

        scanners_dvc = self._app.devices.scanners

        for axis, is_chosen, org_axis_vltg in zip(
            "xyz",
            scanners_dvc.AXES_TO_BOOL_TUPLE_DICT[which_axes],
            scanners_dvc.ORIGIN,
        ):
            if is_chosen:
                getattr(self._gui, f"{axis}AOV").setValue(org_axis_vltg)

        self.move_scanners(which_axes)

        logging.debug(f"{self._app.devices.scanners.log_ref} sent to {which_axes} origin")

    def displace_scanner_axis(self, sign: int) -> None:
        """Doc."""

        um_disp = sign * self._gui.axisMoveUm.value()

        if um_disp != 0.0:
            scanners_dvc = self._app.devices.scanners
            axis = self._gui.axesGroup.checkedButton().text()
            current_vltg = scanners_dvc.ai_buffer[-1][3:][scanners_dvc.AXIS_INDEX[axis]]
            um_V_RATIO = dict(zip("XYZ", scanners_dvc.um_v_ratio))[axis]
            delta_vltg = um_disp / um_V_RATIO

            new_vltg = getattr(scanners_dvc, f"{axis.upper()}_AO_LIMITS").clamp(
                (current_vltg + delta_vltg)
            )

            getattr(self._gui, f"{axis.lower()}AOV").setValue(new_vltg)
            self.move_scanners(axis)

            logging.debug(
                f"{self._app.devices.scanners.log_ref}({axis}) was displaced {str(um_disp)} um"
            )

    def move_stage(self, dir: str, steps: int):
        """Doc."""

        self._app.loop.create_task(self._app.devices.stage.move(dir=dir, steps=steps))
        logging.info(f"{self._app.devices.stage.log_ref} moved {str(steps)} steps {str(dir)}")

    def show_laser_dock(self):
        """Make the laser dock visible (convenience)."""

        if not self._gui.laserDock.isVisible():
            self._gui.laserDock.setVisible(True)
            self._gui.actionLaser_Control.setChecked(True)

    def show_stage_dock(self):
        """Make the laser dock visible (convenience)."""

        if not self._gui.stepperDock.isVisible():
            self._gui.stepperDock.setVisible(True)
            self._gui.actionStepper_Stage_Control.setChecked(True)

    async def toggle_meas(self, meas_type, laser_mode):
        """Doc."""

        if meas_type != (current_type := self._app.meas.type):

            if not self._app.meas.is_running:
                # no meas running
                if meas_type == "SFCSSolution":
                    pattern = self._gui.solScanType.currentText()
                    if pattern == "angular":
                        scan_params = wdgts.SOL_ANG_SCAN_COLL.read_gui_to_obj(self._app)
                    elif pattern == "circle":
                        scan_params = wdgts.SOL_CIRC_SCAN_COLL.read_gui_to_obj(self._app)
                    elif pattern == "static":
                        scan_params = SimpleNamespace()

                    scan_params.pattern = pattern

                    kwargs = wdgts.SOL_MEAS_COLL.read_gui_to_obj(self._app, "dict")

                    self._app.meas = meas.SolutionMeasurementProcedure(
                        app=self._app,
                        scan_params=scan_params,
                        laser_mode=laser_mode.lower(),
                        **kwargs,
                    )

                    self._gui.startSolScanExc.setEnabled(False)
                    self._gui.startSolScanDep.setEnabled(False)
                    self._gui.startSolScanSted.setEnabled(False)
                    getattr(self._gui, f"startSolScan{laser_mode}").setEnabled(True)
                    getattr(self._gui, f"startSolScan{laser_mode}").setText("Stop \nScan")

                    self._gui.solScanMaxFileSize.setEnabled(False)
                    self._gui.solScanDur.setEnabled(self._gui.repeatSolMeas.isChecked())
                    self._gui.solScanDurUnits.setEnabled(False)
                    self._gui.solScanFileTemplate.setEnabled(False)

                elif meas_type == "SFCSImage":

                    kwargs = wdgts.IMG_MEAS_COLL.read_gui_to_obj(self._app, "dict")

                    self._app.meas = meas.ImageMeasurementProcedure(
                        app=self._app,
                        scan_params=wdgts.IMG_SCAN_COLL.read_gui_to_obj(self._app),
                        laser_mode=laser_mode.lower(),
                        **kwargs,
                    )

                    self._gui.startImgScanExc.setEnabled(False)
                    self._gui.startImgScanDep.setEnabled(False)
                    self._gui.startImgScanSted.setEnabled(False)
                    getattr(self._gui, f"startImgScan{laser_mode}").setEnabled(True)
                    getattr(self._gui, f"startImgScan{laser_mode}").setText("Stop \nScan")

                self._app.loop.create_task(self._app.meas.run())
            else:
                # other meas running
                logging.warning(
                    f"Another type of measurement " f"({current_type}) is currently running."
                )

        else:  # current_type == meas_type
            # manual shutdown
            if meas_type == "SFCSSolution":
                self._gui.startSolScanExc.setEnabled(True)
                self._gui.startSolScanDep.setEnabled(True)
                self._gui.startSolScanSted.setEnabled(True)
                getattr(self._gui, f"startSolScan{laser_mode}").setText(f"{laser_mode} \nScan")
                self._gui.impl.go_to_origin("XY")
                self._gui.solScanMaxFileSize.setEnabled(True)
                self._gui.solScanDur.setEnabled(True)
                self._gui.solScanDurUnits.setEnabled(True)
                self._gui.solScanFileTemplate.setEnabled(True)

            elif meas_type == "SFCSImage":
                self._gui.startImgScanExc.setEnabled(True)
                self._gui.startImgScanDep.setEnabled(True)
                self._gui.startImgScanSted.setEnabled(True)
                getattr(self._gui, f"startImgScan{laser_mode}").setText(f"{laser_mode} \nScan")

            if self._app.meas.is_running:
                # manual stop
                await self._app.meas.stop()

    def disp_scn_pttrn(self, pattern: str):
        """Doc."""

        if pattern == "image":
            scan_params_coll = wdgts.IMG_SCAN_COLL
            plt_wdgt = self._gui.imgScanPattern
        elif pattern == "angular":
            scan_params_coll = wdgts.SOL_ANG_SCAN_COLL
            plt_wdgt = self._gui.solScanPattern
        elif pattern == "circle":
            scan_params_coll = wdgts.SOL_CIRC_SCAN_COLL
            plt_wdgt = self._gui.solScanPattern
        elif pattern == "static":
            scan_params_coll = None
            plt_wdgt = self._gui.solScanPattern

        if scan_params_coll:
            scan_params = scan_params_coll.read_gui_to_obj(self._app)

            with suppress(AttributeError, ZeroDivisionError, ValueError):
                # AttributeError - devices not yet initialized
                # ZeroDivisionError - loadout has bad values
                um_v_ratio = self._app.devices.scanners.um_v_ratio
                ao, *_ = ScanPatternAO(pattern, scan_params, um_v_ratio).calculate_pattern()
                x_data, y_data = ao[0, :], ao[1, :]
                plt_wdgt.display_pattern(x_data, y_data)

        else:
            # no scan
            plt_wdgt.display_pattern([], [])

    def open_settwin(self):
        """Doc."""

        self._app.gui.settings.show()
        self._app.gui.settings.activateWindow()

    def counts_avg_interval_changed(self, val: int) -> None:
        """Doc."""

        with suppress(AttributeError):
            # AttributeError - devices not yet initialized
            self._app.timeout_loop.cntr_avg_interval_s = val * 1e-3  # convert to seconds

    ####################
    ## Image Tab
    ####################
    def roi_to_scan(self):
        """Doc"""

        plane_idx = self._gui.numPlaneShown.value()
        with suppress(AttributeError):
            # IndexError - 'App' object has no attribute 'curr_img_idx'
            image_tdc = ImageSFCSMeasurement()
            image_tdc.process_data_file(
                file_dict=self._app.last_image_scans[self._app.curr_img_idx]
            )
            image_data = image_tdc.image_data
            line_ticks_v = image_data.line_ticks_v
            row_ticks_v = image_data.row_ticks_v
            plane_ticks_v = image_data.plane_ticks_v

            coord_1, coord_2 = (round(pos_i) for pos_i in self._gui.imgScanPlot.axes[0].cursor.pos)

            dim1_vltg = line_ticks_v[coord_1]
            dim2_vltg = row_ticks_v[coord_2]
            dim3_vltg = plane_ticks_v[plane_idx]

            plane_orientation = image_data.plane_orientation

            if plane_orientation == "XY":
                vltgs = (dim1_vltg, dim2_vltg, dim3_vltg)
            elif plane_orientation == "XZ":
                vltgs = (dim1_vltg, dim3_vltg, dim2_vltg)
            elif plane_orientation == "YZ":
                vltgs = (dim3_vltg, dim1_vltg, dim2_vltg)

            [
                getattr(self._gui, f"{axis.lower()}AOV").setValue(vltg)
                for axis, vltg in zip("XYZ", vltgs)
            ]

            self.move_scanners(plane_orientation)

            logging.debug(f"{self._app.devices.scanners.log_ref} moved to ROI ({vltgs})")

    def auto_scale_image(self, percent_factor: int):
        """Doc."""

        with suppress(AttributeError):
            # AttributeError - no last image yet
            image = self._app.curr_img
            try:
                image = auto_brightness_and_contrast(image, percent_factor)
            except (ZeroDivisionError, IndexError) as exc:
                err_hndlr(exc, sys._getframe(), locals(), lvl="warning")

            self._gui.imgScanPlot.display_image(image, cursor=True, imshow_kwargs=dict(cmap="bone"))

    def disp_plane_img(self, img_idx=None, plane_idx=None, auto_cross=False):
        """Doc."""

        method_dict = {
            "Forward scan - actual counts per pixel": "forward",
            "Forward scan - points per pixel": "forward normalization",
            "Forward scan - normalized": "forward normalized",
            "Backwards scan - actual counts per pixel": "backward",
            "Backwards scan - points per pixel": "backward normalization",
            "Backwards scan - normalized": "backward normalized",
            "Both scans - interlaced": "interlaced",
            "Both scans - averaged": "averaged",
        }

        def auto_crosshair_position(image: np.ndarray, thresholding=False) -> Tuple[float, float]:
            """
            Beam co-alignment aid. Attempts to fit a 2D Gaussian to an image and returns the fitted center.
            If the fit fails, or if the fitted Gaussian is centered outside the image limits, it instead
            returns the image's center of mass.
            """

            try:
                fit_params = fit_tools.fit_2d_gaussian_to_image(image)
            except fit_tools.FitError:
                # Gaussian fit failed, using COM
                return helper.center_of_mass(image)
            else:
                _, x0, y0, sigma_x, sigma_y, *_ = fit_params.beta
                height, width = image.shape
                if (
                    (0 < x0 < width)
                    and (0 < y0 < height)
                    and (sigma_x < width)
                    and (sigma_y < height)
                ):
                    # x0 and y0 are within image boundaries. Using the Gaussian fit
                    return (x0, y0)
                else:
                    # using COM
                    return helper.center_of_mass(image)

        if img_idx is None:
            try:
                img_idx = self._app.curr_img_idx
            except AttributeError:
                # .curr_img_idx not yet set
                img_idx = 0

        img_meas_wdgts = wdgts.IMG_MEAS_COLL.read_gui_to_obj(self._app)
        disp_mthd = img_meas_wdgts.image_method
        with suppress(IndexError):
            # IndexError - No last_image_scans appended yet
            image_tdc = ImageSFCSMeasurement()
            image_tdc.process_data_file(file_dict=self._app.last_image_scans[img_idx])
            image_data = image_tdc.image_data
            if plane_idx is None:
                # use center plane if not supplied
                plane_idx = int(image_data.n_planes / 2)
            image = image_data.get_image(method_dict[disp_mthd], plane_idx)
            self._app.curr_img_idx = img_idx
            self._app.curr_img = image
            img_meas_wdgts.image_wdgt.obj.display_image(
                image, cursor=True, imshow_kwargs=dict(cmap="bone")
            )
            if auto_cross and image.any():
                img_meas_wdgts.image_wdgt.obj.axes[0].cursor.move_to_pos(
                    auto_crosshair_position(image)
                )

    def plane_choice_changed(self, plane_idx):
        """Doc."""

        with suppress(AttributeError):
            # AttributeError - no scan performed since app init
            self._app.meas.plane_shown.set(plane_idx)
            self.disp_plane_img(plane_idx=plane_idx)

    def fill_img_scan_preset_gui(self, curr_text: str) -> None:
        """Doc."""

        img_scn_wdgt_fillout_dict = {
            "Locate Plane - YZ Coarse": ["YZ", 15, 15, 10, 80, 1000, 20, 0.9, 1],
            "MFC - XY compartment": ["XY", 70, 70, 0, 80, 1000, 20, 0.9, 1],
            "GB -  XY Coarse": ["XY", 15, 15, 0, 80, 1000, 20, 0.9, 1],
            "GB - XY bead area": ["XY", 5, 5, 0, 80, 1000, 20, 0.9, 1],
            "GB - XY single bead": ["XY", 1, 1, 0, 80, 1000, 20, 0.9, 1],
            "GB - YZ single bead": ["YZ", 2.5, 2.5, 0, 80, 1000, 20, 0.9, 1],
        }

        wdgts.IMG_SCAN_COLL.write_to_gui(self._app, img_scn_wdgt_fillout_dict[curr_text])
        logging.debug(f"Image scan preset configuration chosen: '{curr_text}'")

    def cycle_through_image_scans(self, dir: str) -> None:
        """Doc."""

        with suppress(AttributeError):
            n_stored_images = len(self._app.last_image_scans)
            if dir == "next":
                if self._app.curr_img_idx > 0:
                    self.disp_plane_img(img_idx=self._app.curr_img_idx - 1, auto_cross=False)
            elif dir == "prev":
                if self._app.curr_img_idx < n_stored_images - 1:
                    self.disp_plane_img(img_idx=self._app.curr_img_idx + 1, auto_cross=False)

    def save_current_image(self) -> None:
        """Doc."""

        wdgt_coll = wdgts.IMG_MEAS_COLL.read_gui_to_obj(self._app)

        with suppress(AttributeError):
            file_dict = self._app.last_image_scans[self._app.curr_img_idx]
            file_name = f"{wdgt_coll.file_template}_{file_dict['laser_mode']}_{file_dict['scan_params']['plane_orientation']}_{dt.now().strftime('%H%M%S')}"
            today_dir = Path(wdgt_coll.save_path) / dt.now().strftime("%d_%m_%Y")
            dir_path = today_dir / "image"
            file_path = dir_path / (re.sub("\\s", "_", file_name) + ".pkl")
            file_utilities.save_object_to_disk(file_dict, file_path, compression_method="gzip")
            logging.debug(f"Saved measurement file: '{file_path}'.")

    ####################
    ## Solution Tab
    ####################
    def change_meas_duration(self, new_dur: float):
        """Allow duration change during run only for alignment measurements"""

        if self._app.meas.type == "SFCSSolution" and self._app.meas.repeat is True:
            self._app.meas.duration_s = new_dur * self._app.meas.duration_multiplier

    def fill_sol_meas_preset_gui(self, curr_text: str) -> None:
        """Doc."""

        sol_meas_wdgt_fillout_dict = {
            "Standard Alignment": {
                "scan_type": "static",
                "repeat": True,
            },
            "Standard Angular": {
                "scan_type": "angular",
                "regular": True,
            },
            "Standard Circular": {
                "scan_type": "circle",
                "regular": True,
            },
        }

        wdgts.SOL_MEAS_COLL.write_to_gui(self._app, sol_meas_wdgt_fillout_dict[curr_text])
        logging.debug(f"Solution measurement preset configuration chosen: '{curr_text}'")

    ####################
    ## Camera Dock
    ####################

    def toggle_camera_dock(self, is_toggled_on: bool) -> None:
        """Doc."""

        self._gui.cameraDock.setVisible(is_toggled_on)
        if is_toggled_on:
            if self.cameras is None:
                slider_const = 1e2
                self.cameras = (self._app.devices.camera_1, self._app.devices.camera_2)
                for idx, camera in enumerate(self.cameras):
                    self.update_slider_range(cam_num=idx + 1)
                    [
                        getattr(self._gui, f"{name}{idx+1}").setValue(val * slider_const)
                        for name, val in camera.DEFAULT_PARAMETERS.items()
                    ]
            self._gui.move(100, 30)
            self._gui.setFixedSize(1661, 950)
        else:
            self._gui.move(300, 30)
            self._gui.setFixedSize(1211, 950)
            [
                self.device_toggle(
                    f"camera_{cam_num}", "toggle_video", "is_in_video_mode", leave_off=True
                )
                for cam_num in (1, 2)
            ]
        self._gui.setMaximumSize(int(1e5), int(1e5))

    def update_slider_range(self, cam_num: int) -> None:
        """Doc."""

        slider_const = 1e2

        camera = self.cameras[cam_num - 1]

        with suppress(AttributeError):
            # AttributeError - camera not properly initialized
            for name in ("pixel_clock", "framerate", "exposure"):
                range = tuple(limit * slider_const for limit in getattr(camera, f"{name}_range"))
                getattr(self._gui, f"{name}{cam_num}").setRange(*range)

    def set_parameter(self, cam_num: int, param_name: str, value) -> None:
        """Doc."""

        if self.cameras is None:
            return

        slider_const = 1e2

        # convert from slider
        value /= slider_const

        getattr(self._gui, f"{param_name}_val{cam_num}").setValue(value)

        camera = self.cameras[cam_num - 1]
        with suppress(DeviceError):
            # DeviceError - camera not properly initialized
            camera.set_parameters({param_name: value})
            getattr(self._gui, f"autoExp{cam_num}").setChecked(False)
        self.update_slider_range(cam_num)

    def display_image(self, cam_num: int):
        """Doc."""

        camera = self.cameras[cam_num - 1]

        with suppress(DeviceError, ValueError):
            # TODO: not sure if 'suppress' is necessary
            # ValueError - new_image is None
            # DeviceError - error in camera
            camera.get_image()
            if not camera.is_in_video_mode:  # snapshot
                logging.debug(f"Camera {cam_num} photo taken")

    def set_auto_exposure(self, cam_num: int, is_checked: bool):
        """Doc."""

        if self.cameras is None:
            return

        camera = self.cameras[cam_num - 1]
        camera.set_auto_exposure(is_checked)
        if not is_checked:
            parameter_names = ("pixel_clock", "framerate", "exposure")
            parameter_values = (
                getattr(self._gui, f"{param_name}_val{cam_num}").value()
                for param_name in parameter_names
            )
            parameters = ((name, value) for name, value in zip(parameter_names, parameter_values))
            with suppress(DeviceError):
                camera.set_parameters(parameters)

    ####################
    ## Analysis Tab - Raw Data
    ####################

    def switch_data_type(self) -> None:
        """Doc."""

        # TODO: switch thses names e.g. DATA_IMPORT_COLL -> data_import_wdgts
        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL.read_gui_to_obj(self._app)

        if DATA_IMPORT_COLL.is_image_type:
            DATA_IMPORT_COLL.import_stacked.set(0)
            DATA_IMPORT_COLL.analysis_stacked.set(0)
            type_ = "image"
        elif DATA_IMPORT_COLL.is_solution_type:
            DATA_IMPORT_COLL.import_stacked.set(1)
            DATA_IMPORT_COLL.analysis_stacked.set(1)
            type_ = "solution"

        self.populate_all_data_dates(type_)

    def populate_all_data_dates(self, type_) -> None:
        """Doc."""

        # TODO: switch thses names e.g. DATA_IMPORT_COLL -> data_import_wdgts
        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL.read_gui_to_obj(self._app)

        if type_ == "image":
            save_path = wdgts.IMG_MEAS_COLL.read_gui_to_obj(self._app).save_path
        elif type_ == "solution":
            save_path = wdgts.SOL_MEAS_COLL.read_gui_to_obj(self._app).save_path
        else:
            raise ValueError(
                f"Data type '{type_}'' is not supported; use either 'image' or 'solution'."
            )

        wdgts.DATA_IMPORT_COLL.clear_all_objects()

        with suppress(TypeError, IndexError, FileNotFoundError):
            # no directories found... (dir_years is None or [])
            # FileNotFoundError - data directory deleted while app running!
            dir_years = helper.dir_date_parts(save_path, type_)
            DATA_IMPORT_COLL.data_years.obj.addItems(dir_years)

    def populate_data_dates_from_year(self, year: str) -> None:
        """Doc."""

        if not year:
            # ignore if combobox was just cleared
            return

        # define widgets
        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL.read_gui_to_obj(self._app)

        if DATA_IMPORT_COLL.is_image_type:
            meas_sett = wdgts.IMG_MEAS_COLL.read_gui_to_obj(self._app)
        elif DATA_IMPORT_COLL.is_solution_type:
            meas_sett = wdgts.SOL_MEAS_COLL.read_gui_to_obj(self._app)
        save_path = meas_sett.save_path
        sub_dir = meas_sett.sub_dir_name

        DATA_IMPORT_COLL.data_months.obj.clear()

        with suppress(TypeError, IndexError):
            # no directories found... (dir_years is None or [])
            dir_months = helper.dir_date_parts(save_path, sub_dir, year=year)
            DATA_IMPORT_COLL.data_months.obj.addItems(dir_months)

    def populate_data_dates_from_month(self, month: str) -> None:
        """Doc."""

        if not month:
            # ignore if combobox was just cleared
            return

        # define widgets
        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL.read_gui_to_obj(self._app)
        year = DATA_IMPORT_COLL.data_years.get()
        days_combobox = DATA_IMPORT_COLL.data_days.obj

        if DATA_IMPORT_COLL.is_image_type:
            meas_sett = wdgts.IMG_MEAS_COLL.read_gui_to_obj(self._app)
        elif DATA_IMPORT_COLL.is_solution_type:
            meas_sett = wdgts.SOL_MEAS_COLL.read_gui_to_obj(self._app)
        save_path = meas_sett.save_path
        sub_dir = meas_sett.sub_dir_name

        days_combobox.clear()

        with suppress(TypeError, IndexError):
            # no directories found... (dir_years is None or [])
            dir_days = helper.dir_date_parts(save_path, sub_dir, year=year, month=month)
            days_combobox.addItems(dir_days)

    def pkl_and_mat_templates(self, dir_path: Path) -> List[str]:
        """
        Accepts a directory path and returns a list of file templates
        ending in the form '*.pkl' or '*.mat' where * is any number.
        The templates are sorted by their time stamp of the form "HHMMSS".
        In case the folder contains legacy templates, they are sorted without a key.
        """

        is_solution_type = "solution" in dir_path.parts

        pkl_template_set = {
            (
                re.sub("[0-9]+.pkl", "*.pkl", str(file_path.name))
                if is_solution_type
                else file_path.name
            )
            for file_path in dir_path.iterdir()
            if file_path.suffix == ".pkl"
        }
        mat_template_set = {
            (
                re.sub("[0-9]+.mat", "*.mat", str(file_path.name))
                if is_solution_type
                else file_path.name
            )
            for file_path in dir_path.iterdir()
            if file_path.suffix == ".mat"
        }
        try:
            sorted_templates = sorted(
                pkl_template_set.union(mat_template_set),
                key=lambda tmpl: int(re.split("(\\d{6})_*", tmpl)[1]),
            )
        except IndexError:
            # Dealing with legacy templates which do not contain the appropriate time format
            sorted_templates = sorted(pkl_template_set.union(mat_template_set))
            logging.warning(
                "Folder contains irregular file templates! Templates may not be properly sorted."
            )
        finally:
            return sorted_templates

    def populate_data_templates_from_day(self, day: str) -> None:
        """Doc."""

        if not day:
            # ignore if combobox was just cleared
            return

        # define widgets
        templates_combobox = wdgts.DATA_IMPORT_COLL.data_templates.obj

        templates_combobox.clear()

        with suppress(TypeError, IndexError):
            # no directories found... (dir_years is None or [])
            templates = self.pkl_and_mat_templates(self.current_date_type_dir_path())
            templates_combobox.addItems(templates)

    def show_num_files(self, template: str) -> None:
        """Doc."""

        if template:
            n_files_wdgt = wdgts.DATA_IMPORT_COLL.n_files
            dir_path = Path(self.current_date_type_dir_path())
            n_files = sum(1 for file_path in dir_path.glob(template))
            n_files_wdgt.set(f"({n_files} Files)")

    def cycle_through_data_templates(self, dir: str) -> None:
        """Cycle through the daily data templates in order (next or previous)"""

        data_templates_combobox = wdgts.DATA_IMPORT_COLL.data_templates.obj
        curr_idx = data_templates_combobox.currentIndex()
        n_items = data_templates_combobox.count()

        if dir == "next":
            if curr_idx + 1 < n_items:
                data_templates_combobox.setCurrentIndex(curr_idx + 1)
            else:  # cycle to beginning
                data_templates_combobox.setCurrentIndex(0)
        elif dir == "prev":
            if curr_idx - 1 >= 0:
                data_templates_combobox.setCurrentIndex(curr_idx - 1)
            else:  # cycle to end
                data_templates_combobox.setCurrentIndex(n_items - 1)

    def rename_template(self) -> None:
        """Doc."""

        dir_path = self.current_date_type_dir_path()
        data_import_wdgts = wdgts.DATA_IMPORT_COLL.read_gui_to_obj(self._app)
        curr_template = data_import_wdgts.data_templates.get()
        new_template_prefix = data_import_wdgts.new_template

        # cancel if no new prefix supplied
        if not new_template_prefix:
            return

        try:
            # get the current prefix - anything before the timestamp
            curr_template_prefix = re.findall("(^.*)(?=_[0-9]{6})", curr_template)[0]
        except IndexError:
            # legacy template which has no timestamp
            curr_template_prefix = re.findall("(^.*)(?=_\\*\\.[a-z]{3})", curr_template)[0]

        if new_template_prefix == curr_template_prefix:
            logging.warning(
                f"rename_template(): Requested template and current template are identical ('{curr_template}'). Operation canceled."
            )
            return

        new_template = re.sub(curr_template_prefix, new_template_prefix, curr_template, count=1)

        # check if new template already exists (unlikely)
        if list(dir_path.glob(new_template)):
            logging.warning(
                f"New template '{new_template}' already exists in '{dir_path}'. Operation canceled."
            )
            return
        # check if current template doesn't exists (can only happen if deleted manually between discovering the template and running this function)
        if not (curr_filepaths := [str(filepath) for filepath in dir_path.glob(curr_template)]):
            logging.warning("Current template is missing! (Probably manually deleted)")
            return

        pressed = dialog.Question(
            txt=f"Change current template from:\n{curr_template}\nto:\n{new_template}\n?",
            title="Edit File Template",
        ).display()
        if pressed == dialog.NO:
            return

        # generate new filanames
        new_filepaths = [
            re.sub(curr_template_prefix, new_template_prefix, curr_filepath)
            for curr_filepath in curr_filepaths
        ]
        # rename the files
        [
            Path(curr_filepath).rename(new_filepath)
            for curr_filepath, new_filepath in zip(curr_filepaths, new_filepaths)
        ]
        # rename the log file, if applicable
        with suppress(FileNotFoundError):
            pattern = re.sub("\\*?", "[0-9]*", curr_template)
            replacement = re.sub("_\\*", "", curr_template)
            current_log_filepath = re.sub(pattern, replacement, curr_filepaths[0])
            current_log_filepath = re.sub("\\.pkl", ".log", current_log_filepath)
            new_log_filepath = re.sub(
                curr_template_prefix, new_template_prefix, current_log_filepath
            )
            Path(current_log_filepath).rename(new_log_filepath)

        # refresh templates
        day = data_import_wdgts.data_days
        self.populate_data_templates_from_day(day)

    def current_date_type_dir_path(self) -> Path:
        """Returns path to directory of currently selected date and measurement type"""

        import_wdgts = wdgts.DATA_IMPORT_COLL.read_gui_to_obj(self._app)

        day = import_wdgts.data_days.get()
        year = import_wdgts.data_years.get()
        month = import_wdgts.data_months.get()

        if import_wdgts.is_image_type:
            meas_settings = wdgts.IMG_MEAS_COLL.read_gui_to_obj(self._app)
        elif import_wdgts.is_solution_type:
            meas_settings = wdgts.SOL_MEAS_COLL.read_gui_to_obj(self._app)
        save_path = Path(meas_settings.save_path)
        sub_dir = meas_settings.sub_dir_name
        return save_path / f"{day.rjust(2, '0')}_{month.rjust(2, '0')}_{year}" / sub_dir

    def open_data_dir(self) -> None:
        """Doc."""

        if (dir_path := self.current_date_type_dir_path()).is_dir():
            webbrowser.open(str(dir_path))
        else:
            # dir was deleted, refresh all dates
            self.switch_data_type()

    def update_dir_log_file(self) -> None:
        """Doc."""

        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL
        text_lines = DATA_IMPORT_COLL.log_text.get().split("\n")
        curr_template = DATA_IMPORT_COLL.data_templates.get()
        log_filename = re.sub("_\\*.\\w{3}", ".log", curr_template)
        with suppress(AttributeError, TypeError, FileNotFoundError):
            # no directories found
            file_path = self.current_date_type_dir_path() / log_filename
            helper.write_list_to_file(file_path, text_lines)
            self.update_dir_log_wdgt(curr_template)

    def get_daily_alignment(self):
        """Doc."""

        date_dir_path = Path(re.sub("(solution|image)", "", str(self.current_date_type_dir_path())))

        try:
            # TODO: make this work for multiple templates (exc and sted), add both to log file and also the ratios
            template = self.pkl_and_mat_templates(date_dir_path)[0]
        except IndexError:
            # no alignment file found
            g0, tau = (None, None)
        else:
            # TODO: loading properly (err hndling) should be a seperate function

            # Inferring data_dype from template
            data_type = self.infer_data_type_from_template(template)

            measurement = SolutionSFCSMeasurement()
            try:
                measurement.read_fpga_data(
                    date_dir_path / template,
                    should_plot=False,
                )
                measurement.correlate_and_average(cf_name=data_type)

            except AttributeError:
                # No directories found
                g0, tau = (None, None)

            except (NotImplementedError, RuntimeError, ValueError, FileNotFoundError) as exc:
                err_hndlr(exc, sys._getframe(), locals())

            cf = measurement.cf[data_type]

            try:
                cf.fit_correlation_function()
            except fit_tools.FitError as exc:
                err_hndlr(exc, sys._getframe(), locals())
                g0, tau = (None, None)
            else:
                fit_params = cf.fit_params["diffusion_3d_fit"]
                g0, tau, _ = fit_params.beta

        return g0, tau

    def update_dir_log_wdgt(self, template: str) -> None:
        """Doc."""

        def initialize_dir_log_file(file_path, g0, tau) -> None:
            """Doc."""
            # TODO: change arguments to exc_params, sted_params which would be tuples (g0, tau) and set the free atto sted alignment too

            basic_header = []
            basic_header.append("-" * 40)
            basic_header.append("Measurement Log File")
            basic_header.append("-" * 40)
            basic_header.append("Excitation Power: 20 uW @ BFP")
            basic_header.append("Depletion Power: 200 mW @ BFP (set to 260 mW)")
            if g0 is not None:
                basic_header.append(
                    f"Free Atto FCS @ 12 uW: G0 = {g0/1e3:.2f} k/ tau = {tau*1e3:.2f} us"
                )
            else:
                basic_header.append("Free Atto FCS @ 12 uW: Not Available")
            basic_header.append("-" * 40)
            basic_header.append("EDIT_HERE")

            helper.write_list_to_file(file_path, basic_header)

        if not template:
            return

        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL.read_gui_to_obj(self._app)
        DATA_IMPORT_COLL.log_text.set("")  # clear first

        # get the log file path
        if DATA_IMPORT_COLL.is_solution_type:
            log_filename = re.sub("_?\\*\\.\\w{3}", ".log", template)
        elif DATA_IMPORT_COLL.is_image_type:
            log_filename = re.sub("\\.\\w{3}", ".log", template)
        file_path = self.current_date_type_dir_path() / log_filename

        try:  # load the log file in path
            text_lines = helper.read_file_to_list(file_path)
        except (FileNotFoundError, OSError):  # initialize a new log file if no existing file
            with suppress(OSError, IndexError):
                # OSError - missing file/folder (deleted during operation)
                # IndexError - alignment file does not exist
                initialize_dir_log_file(file_path, *self.get_daily_alignment())
                text_lines = helper.read_file_to_list(file_path)
        finally:  # write file to widget
            with suppress(UnboundLocalError):
                # UnboundLocalError
                DATA_IMPORT_COLL.log_text.set("\n".join(text_lines))

    def preview_img_scan(self, template: str) -> None:
        """Doc."""

        if not template:
            return

        data_import_wdgts = wdgts.DATA_IMPORT_COLL.read_gui_to_obj(self._app)

        if data_import_wdgts.is_image_type:
            # import the data
            try:
                image_tdc = ImageSFCSMeasurement()
                image_tdc.read_image_data(self.current_date_type_dir_path() / template)
            except FileNotFoundError:
                self.switch_data_type()
                return
            # get the center plane image, in "forward"
            image = image_tdc.image_data.get_image("forward")
            # plot it (below)
            data_import_wdgts.img_preview_disp.obj.display_image(
                image, imshow_kwargs=dict(cmap="bone"), scroll_zoom=False
            )

    def save_processed_data(self):
        """Doc."""

        with self.get_measurement_from_template(
            should_load=True, method_name="save_processed_data"
        ) as measurement:
            curr_dir = self.current_date_type_dir_path()
            file_utilities.save_processed_solution_meas(measurement, curr_dir)
            logging.info("Saved the processed data.")

    def toggle_save_processed_enabled(self):
        """Doc."""

        with self.get_measurement_from_template() as measurement:
            if measurement is None:
                self._gui.solImportSaveProcessed.setEnabled(False)
            else:
                self._gui.solImportSaveProcessed.setEnabled(True)

    def toggle_load_processed_enabled(self, current_template: str):
        """Doc."""

        curr_dir = self.current_date_type_dir_path()
        file_path = curr_dir / "processed" / re.sub("_[*]", "", current_template)
        self._gui.solImportLoadProcessed.setEnabled(file_path.is_file())

    def convert_files_to_matlab_format(self) -> None:
        """
        Convert files of current template to '.mat',
        after translating their dictionaries to the legacy matlab format.
        """

        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL
        current_template = DATA_IMPORT_COLL.data_templates.get()
        current_dir_path = self.current_date_type_dir_path()

        if not current_template or current_template.endswith(".mat"):
            return

        pressed = dialog.Question(
            txt=f"Are you sure you wish to convert '{current_template}'?",
            title="Conversion to .mat Format",
        ).display()
        if pressed == dialog.NO:
            return

        unsorted_paths = list(current_dir_path.glob(current_template))
        file_paths = file_utilities.sort_file_paths_by_file_number(unsorted_paths)

        print(f"Converting {len(file_paths)} files to '.mat' in legacy MATLAB format...", end=" ")

        for idx, file_path in enumerate(file_paths):
            file_dict = file_utilities.load_file_dict(file_path)
            mat_file_path_str = re.sub("\\.pkl", ".mat", str(file_path))
            if "solution" in mat_file_path_str:
                mat_file_path = Path(re.sub("solution", r"solution\\matlab", mat_file_path_str))
            elif "image" in mat_file_path_str:
                mat_file_path = Path(re.sub("image", r"image\\matlab", mat_file_path_str))
            Path.mkdir(current_dir_path / "matlab", parents=True, exist_ok=True)
            file_utilities.save_mat(file_dict, mat_file_path)
            print(f"({idx+1})", end=" ")

        print("Done.")
        logging.info(f"{current_template} converted to MATLAB format")

    ##########################
    ## Analysis Tab - Single Measurement
    ##########################

    def import_sol_data(self, should_load_processed=False) -> None:
        """Doc."""

        import_wdgts = wdgts.DATA_IMPORT_COLL.read_gui_to_obj(self._app)
        current_template = import_wdgts.data_templates.get()
        curr_dir = self.current_date_type_dir_path()

        if self._app.analysis.loaded_measurements.get(current_template) is not None:
            logging.info(f"Data '{current_template}' already loaded - ignoring.")
            return

        with self._app.pause_ai_ci():

            if should_load_processed:
                file_path = curr_dir / "processed" / re.sub("_[*]", "", current_template)
                logging.info(f"Loading processed data '{current_template}' from hard drive...")
                measurement = file_utilities.load_processed_solution_measurement(file_path)
                print("Done.")

            else:  # process data
                options_dict = self.get_loading_options_as_dict()

                # Inferring data_dype from template
                data_type = self.infer_data_type_from_template(current_template)

                # loading and correlating
                try:
                    with suppress(AttributeError):
                        # AttributeError - No directories found
                        measurement = SolutionSFCSMeasurement()
                        measurement.read_fpga_data(
                            curr_dir / current_template,
                            **options_dict,
                        )
                    measurement.correlate_data(
                        cf_name=data_type,
                        verbose=True,
                        **options_dict,
                    )

                except (NotImplementedError, RuntimeError, ValueError, FileNotFoundError) as exc:
                    err_hndlr(exc, sys._getframe(), locals())
                    return

            # save data and populate combobox
            imported_combobox1 = wdgts.SOL_MEAS_ANALYSIS_COLL.imported_templates
            imported_combobox2 = wdgts.SOL_EXP_ANALYSIS_COLL.imported_templates

            self._app.analysis.loaded_measurements[current_template] = measurement

            imported_combobox1.obj.addItem(current_template)
            imported_combobox2.obj.addItem(current_template)

            imported_combobox1.set(current_template)
            imported_combobox2.set(current_template)

            logging.info(f"Data '{current_template}' ready for analysis.")

            self.toggle_save_processed_enabled()  # refresh save option
            self.toggle_load_processed_enabled(current_template)  # refresh load option

    def get_loading_options_as_dict(self) -> dict:
        """Doc."""

        loading_options = dict()
        import_wdgts = wdgts.DATA_IMPORT_COLL.read_gui_to_obj(self._app)

        # file selection
        if import_wdgts.sol_file_dicrimination.objectName() == "solImportUse":
            loading_options[
                "file_selection"
            ] = f"{import_wdgts.sol_file_use_or_dont} {import_wdgts.sol_file_selection}"
        else:
            loading_options["file_selection"] = None

        loading_options["should_fix_shift"] = import_wdgts.fix_shift
        loading_options["should_subtract_afterpulse"] = import_wdgts.should_subtract_afterpulse
        loading_options["subtract_bg_corr"] = import_wdgts.subtract_bg_corr

        return loading_options

    @contextmanager
    def get_measurement_from_template(
        self,
        template: str = None,
        should_load=False,
        **kwargs,
    ) -> SolutionSFCSMeasurement:
        """Doc."""

        if template is None:
            template = wdgts.SOL_MEAS_ANALYSIS_COLL.imported_templates.get()
        curr_data_type, *_ = re.split(" -", template)
        measurement = self._app.analysis.loaded_measurements.get(curr_data_type)

        if should_load:
            with suppress(AttributeError):
                measurement.dump_or_load_data(should_load=True, **kwargs)

        try:
            yield measurement

        finally:
            if should_load:
                with suppress(AttributeError):
                    measurement.dump_or_load_data(should_load=False, **kwargs)

    def infer_data_type_from_template(self, template: str) -> str:
        """Doc."""

        if ((data_type_hint := "_exc_") in template) or ((data_type_hint := "_sted_") in template):
            data_type = "Confocal" if (data_type_hint == "_exc_") else "CW STED"
            return data_type
        else:
            print(
                f"Data type could not be inferred from template ({template}). Consider changing the template."
            )
            return "Unknown"

    def populate_sol_meas_analysis(self, imported_template):
        """Doc."""

        sol_data_analysis_wdgts = wdgts.SOL_MEAS_ANALYSIS_COLL.read_gui_to_obj(self._app)

        with self.get_measurement_from_template(imported_template) as measurement:
            if measurement is None:
                # no imported templates (deleted)
                wdgts.SOL_MEAS_ANALYSIS_COLL.clear_all_objects()
                sol_data_analysis_wdgts.scan_img_file_num.obj.setRange(1, 1)
                sol_data_analysis_wdgts.scan_img_file_num.set(1)
            else:
                num_files = measurement.n_files
                logging.debug("Populating analysis GUI...")

                # populate general measurement properties
                sol_data_analysis_wdgts.n_files.set(num_files)
                sol_data_analysis_wdgts.scan_duration_min.set(measurement.duration_min)
                sol_data_analysis_wdgts.avg_cnt_rate_khz.set(measurement.avg_cnt_rate_khz)

                if measurement.scan_type == "angular_scan":
                    # populate scan images tab
                    logging.debug("Displaying scan images...")
                    sol_data_analysis_wdgts.scan_img_file_num.obj.setRange(1, num_files)
                    sol_data_analysis_wdgts.scan_img_file_num.set(1)
                    self.display_scan_image(1, imported_template)

                    # calculate average and display
                    logging.debug("Averaging and plotting...")
                    self.calculate_and_show_sol_mean_acf(imported_template)

                    scan_settings_text = "\n\n".join(
                        [
                            f"{key}: {(', '.join([f'{ele:.2f}' for ele in val[:5]]) if isinstance(val, Iterable) else f'{val:.2f}')}"
                            for key, val in measurement.angular_scan_settings.items()
                        ]
                    )

                elif measurement.scan_type == "static":
                    logging.debug("Averaging, plotting and fitting...")
                    self.calculate_and_show_sol_mean_acf(imported_template)
                    scan_settings_text = "no scan."

                sol_data_analysis_wdgts.scan_settings.set(scan_settings_text)

                logging.debug("Done.")

    def display_scan_image(self, file_num, imported_template: str = None):
        """Doc."""

        with suppress(IndexError, KeyError, AttributeError):
            # IndexError - data import failed
            # KeyError, AttributeError - data deleted
            with self.get_measurement_from_template(imported_template) as measurement:
                img = measurement.scan_images_dstack[:, :, file_num - 1]
                roi = measurement.roi_list[file_num - 1]

                scan_image_disp = wdgts.SOL_MEAS_ANALYSIS_COLL.scan_image_disp.obj
                scan_image_disp.display_image(img)
                scan_image_disp.plot(roi["col"], roi["row"], color="white")
                scan_image_disp.entitle_and_label("Pixel Number", "Line Number")

    def calculate_and_show_sol_mean_acf(self, imported_template: str = None) -> None:
        """Doc."""

        sol_data_analysis_wdgts = wdgts.SOL_MEAS_ANALYSIS_COLL.read_gui_to_obj(self._app)

        with self.get_measurement_from_template(imported_template) as measurement:
            if measurement is None:
                return

            if measurement.scan_type == "angular_scan":
                row_disc_method = sol_data_analysis_wdgts.row_dicrimination.objectName()
                if row_disc_method == "solAnalysisRemoveOver":
                    avg_corr_kwargs = dict(rejection=sol_data_analysis_wdgts.remove_over)
                elif row_disc_method == "solAnalysisRemoveWorst":
                    avg_corr_kwargs = dict(
                        rejection=None, reject_n_worst=sol_data_analysis_wdgts.remove_worst.get()
                    )
                else:  # use all rows
                    avg_corr_kwargs = dict(rejection=None)

                with suppress(AttributeError, RuntimeError):
                    # AttributeError - no data loaded
                    data_type = self.infer_data_type_from_template(measurement.template)
                    cf = measurement.cf[data_type]
                    cf.average_correlation(**avg_corr_kwargs)

                    if sol_data_analysis_wdgts.plot_spatial:
                        x = (cf.vt_um, "disp")
                        x_label = r"squared displacement ($um^2$)"
                    else:
                        x = (cf.lag, "lag")
                        x_label = "lag (ms)"

                    sol_data_analysis_wdgts.row_acf_disp.obj.plot_acfs(
                        x,
                        cf.avg_cf_cr,
                        cf.g0,
                        cf.cf_cr[cf.j_good, :],
                    )
                    sol_data_analysis_wdgts.row_acf_disp.obj.entitle_and_label(x_label, "G0")

                    sol_data_analysis_wdgts.mean_g0.set(cf.g0 / 1e3)  # shown in thousands
                    sol_data_analysis_wdgts.mean_tau.set(0)

                    sol_data_analysis_wdgts.n_good_rows.set(n_good := len(cf.j_good))
                    sol_data_analysis_wdgts.n_bad_rows.set(n_bad := len(cf.j_bad))
                    sol_data_analysis_wdgts.remove_worst.obj.setMaximum(n_good + n_bad - 2)

            elif measurement.scan_type == "static":
                data_type = self.infer_data_type_from_template(measurement.template)
                cf = measurement.cf[data_type]
                cf.average_correlation()
                try:
                    cf.fit_correlation_function()
                except fit_tools.FitError as exc:
                    # fit failed, use g0 calculated in 'average_correlation()'
                    err_hndlr(exc, sys._getframe(), locals(), lvl="warning")
                    sol_data_analysis_wdgts.mean_g0.set(cf.g0 / 1e3)  # shown in thousands
                    sol_data_analysis_wdgts.mean_tau.set(0)
                else:  # fit succeeded
                    fit_params = cf.fit_params["diffusion_3d_fit"]
                    g0, tau, _ = fit_params.beta
                    fit_func = getattr(fit_tools, fit_params.func_name)
                    sol_data_analysis_wdgts.mean_g0.set(g0 / 1e3)  # shown in thousands
                    sol_data_analysis_wdgts.mean_tau.set(tau * 1e3)
                    y_fit = fit_func(fit_params.x, *fit_params.beta)
                    sol_data_analysis_wdgts.row_acf_disp.obj.clear()
                    sol_data_analysis_wdgts.row_acf_disp.obj.plot_acfs(
                        (cf.lag, "lag"),
                        cf.avg_cf_cr,
                        cf.g0,
                    )
                    sol_data_analysis_wdgts.row_acf_disp.obj.plot(fit_params.x, y_fit, color="red")

    def assign_template(self, type) -> None:
        """Doc."""

        curr_template = wdgts.SOL_MEAS_ANALYSIS_COLL.imported_templates.get()
        assigned_wdgt = getattr(wdgts.SOL_MEAS_ANALYSIS_COLL, type)
        assigned_wdgt.set(curr_template)

    def remove_imported_template(self) -> None:
        """Doc."""

        imported_templates1 = wdgts.SOL_MEAS_ANALYSIS_COLL.imported_templates
        imported_templates2 = wdgts.SOL_EXP_ANALYSIS_COLL.imported_templates

        template = imported_templates1.get()
        self._app.analysis.loaded_measurements[template] = None

        imported_templates1.obj.removeItem(imported_templates1.obj.currentIndex())
        imported_templates2.obj.removeItem(imported_templates2.obj.currentIndex())

        self.toggle_save_processed_enabled()  # refresh save option

        # TODO: clear image properties!

    #########################
    ## Analysis Tab - Single Experiment
    #########################

    def assign_measurement(self, meas_type: str) -> None:
        """Doc."""

        MeasAssignParams = namedtuple("MeasAssignParams", "template method options")
        wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.read_gui_to_obj(self._app)
        assigned_dict = self._app.analysis.assigned_to_experiment
        options = self.get_loading_options_as_dict()

        if wdgt_coll.should_assign_loaded:
            template = wdgt_coll.imported_templates.get()
            method = "loaded"
        elif wdgt_coll.should_assign_raw:
            template = wdgt_coll.data_templates.get()
            method = "raw"

        assigned_dict[meas_type] = MeasAssignParams(template, method, options)

        wdgt_to_assign_to = getattr(wdgt_coll, f"assigned_{meas_type}_template")
        wdgt_to_assign_to.set(template)

    def load_experiment(self) -> None:
        """Doc."""

        wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.read_gui_to_obj(self._app)

        if not (experiment_name := wdgt_coll.experiment_name.get()):
            logging.info("Can't load unnamed experiment!")
            return

        kwargs = dict()
        with suppress(KeyError):
            for meas_type, assignment_params in self._app.analysis.assigned_to_experiment.items():
                if assignment_params.method == "loaded":
                    with self.get_measurement_from_template(
                        getattr(wdgt_coll, f"assigned_{meas_type}_template").get(),
                        should_load=True,
                        method_name="load_experiment",
                    ) as measurement:
                        kwargs[meas_type] = measurement
                elif assignment_params.method == "raw":
                    curr_dir = self.current_date_type_dir_path()
                    kwargs[f"{meas_type}_template"] = (
                        curr_dir / getattr(wdgt_coll, f"assigned_{meas_type}_template").get()
                    )
                # get loading options as kwargs
                kwargs[f"{meas_type}_kwargs"] = assignment_params.options

        # plotting properties
        kwargs["gui_display"] = wdgt_coll.gui_display_loading.obj
        kwargs["gui_options"] = GuiDisplay.GuiDisplayOptions(show_axis=True)
        kwargs["fontsize"] = 10

        experiment = SFCSExperiment(experiment_name)
        try:
            experiment.load_experiment(**kwargs)
        except RuntimeError as exc:
            logging.info(str(exc))
        else:
            # save experiment in memory and GUI
            self._app.analysis.loaded_experiments[experiment_name] = experiment
            wdgt_coll.loaded_experiments.obj.addItem(experiment_name)

            # save measurements seperately in memory and GUI
            for meas_type in ("confocal", "sted"):
                with suppress(AttributeError):
                    # AttributeError - measurement does not exist
                    meas = getattr(experiment, meas_type)
                    self._app.analysis.loaded_measurements[meas.template] = meas
                    wdgts.SOL_MEAS_ANALYSIS_COLL.imported_templates.obj.addItem(meas.template)
                    wdgts.SOL_EXP_ANALYSIS_COLL.imported_templates.obj.addItem(meas.template)

            # reset the dict and clear relevant widgets
            self._app.analysis.assigned_to_experiment = dict()
            wdgt_coll.assigned_confocal_template.obj.clear()
            wdgt_coll.assigned_sted_template.obj.clear()
            wdgt_coll.experiment_name.obj.clear()

            with suppress(IndexError):
                # IndexError - either confocal or STED weren't loaded
                conf_g0 = list(experiment.confocal.cf.values())[0].g0
                sted_g0 = list(experiment.sted.cf.values())[0].g0
                wdgt_coll.g0_ratio.set(conf_g0 / sted_g0)
            logging.info(f"Experiment '{experiment_name}' loaded successfully.")

    def remove_experiment(self) -> None:
        """Doc."""

        loaded_templates_combobox = wdgts.SOL_EXP_ANALYSIS_COLL.loaded_experiments
        # delete from memory
        experiment_name = loaded_templates_combobox.get()
        self._app.analysis.loaded_experiments[experiment_name] = None
        # delete from GUI
        loaded_templates_combobox.obj.removeItem(loaded_templates_combobox.obj.currentIndex())
        logging.info(f"Experiment '{experiment_name}' removed.")

    def get_experiment(
        self, experiment_name: str = None, should_load=False
    ) -> SolutionSFCSMeasurement:
        """Doc."""

        if experiment_name is None:
            experiment_name = wdgts.SOL_EXP_ANALYSIS_COLL.loaded_experiments.get()
        return self._app.analysis.loaded_experiments.get(experiment_name)

    def calibrate_tdc(self) -> None:
        """Doc."""

        wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.read_gui_to_obj(self._app)

        kwargs = dict(gui_options=GuiDisplay.GuiDisplayOptions(show_axis=True), fontsize=10)
        experiment = self.get_experiment()
        try:
            kwargs["gui_display"] = wdgt_coll.gui_display_tdc_cal.obj
            experiment.calibrate_tdc(calib_time_ns=wdgt_coll.calibration_gating, **kwargs)
        except RuntimeError:  # sted or confocal not loaded
            logging.info(
                "One of the measurements (confocal or STED) was not loaded - cannot calibrate TDC."
            )
        except AttributeError:
            logging.info("Can't calibrate TDC, no experiment is loaded!")
        else:
            kwargs["gui_display"] = wdgt_coll.gui_display_comp_lifetimes.obj
            experiment.compare_lifetimes(**kwargs)
            lt_params = experiment.get_lifetime_parameters()

            # display parameters in GUI
            wdgt_coll.fluoresence_lifetime.set(lt_params.lifetime_ns)
            wdgt_coll.laser_pulse_delay.set(lt_params.laser_pulse_delay_ns)

    def assign_gate(self) -> None:
        """Doc."""

        wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.read_gui_to_obj(self._app)

        gate_lt = wdgt_coll.custom_gate_to_assign_lt.get()

        experiment = self.get_experiment()
        try:
            laser_pulse_delay_ns = experiment.lifetime_params.laser_pulse_delay_ns
            lifetime_ns = experiment.lifetime_params.lifetime_ns
        except AttributeError:  # TDC not calibrated!
            logging.info("Cannot gate if TDC isn't calibrated!")
        else:
            lower_gate_ns = gate_lt * lifetime_ns + laser_pulse_delay_ns
            upper_gate_ns = experiment.UPPERֹ_ֹGATE_NS
            gate_ns = helper.Limits(lower_gate_ns, upper_gate_ns)
            wdgt_coll.assigned_gates.obj.addItem(str(gate_ns))

    def remove_assigned_gate(self) -> None:
        """Doc."""

        wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.read_gui_to_obj(self._app)
        wdgt = wdgt_coll.assigned_gates
        gate_to_remove = wdgt.get()
        # delete from GUI
        wdgt.obj.removeItem(wdgt.obj.currentIndex())
        logging.info(f"Gate '{gate_to_remove}' was unassigned.")

    def gate(self) -> None:
        """Doc."""

        wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.read_gui_to_obj(self._app)

        gates_combobox = wdgt_coll.assigned_gates.obj
        gate_list = [
            helper.Limits(gates_combobox.itemText(i), from_string=True)
            for i in range(gates_combobox.count())
        ]

        experiment = self.get_experiment()
        kwargs = dict(
            gui_display=wdgt_coll.gui_display_sted_gating.obj,
            gui_options=GuiDisplay.GuiDisplayOptions(show_axis=True),
            fontsize=10,
        )
        try:
            experiment.add_gates(gate_list, **kwargs)
        except AttributeError:
            logging.info("Can't gate, no experiment is loaded!")
        else:
            wdgt_coll.available_gates.obj.addItems([str(gate) for gate in gate_list])

    def remove_available_gate(self) -> None:
        """Doc."""

        wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.read_gui_to_obj(self._app)
        wdgt = wdgt_coll.available_gates
        gate_to_remove = wdgt.get()
        # delete from object
        experiment = self.get_experiment()
        try:
            experiment.sted.cf.pop(f"gSTED {gate_to_remove}")
        except AttributeError:  # no experiment loaded or no STED
            logging.info("Can't remove gate.")
        except KeyError:  # no assigned gate
            # TODO: clear the available_gates when removing experiment!
            pass
        else:
            # re-plot
            kwargs = dict(
                gui_display=wdgt_coll.gui_display_sted_gating.obj,
                gui_options=GuiDisplay.GuiDisplayOptions(show_axis=True),
                fontsize=10,
            )
            experiment.plot_correlation_functions(**kwargs)
            # delete from GUI
            wdgt.obj.removeItem(wdgt.obj.currentIndex())
            logging.info(f"Gate '{gate_to_remove}' removed from '{experiment.name}' experiment.")

    def calculate_structure_factors(self) -> None:
        """Doc."""

        wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.read_gui_to_obj(self._app)
        experiment = self.get_experiment()
        try:
            experiment.calculate_structure_factors(
                g_min=wdgt_coll.g_min, n_rbst_intrp_points=wdgt_coll.n_rbst_intrp_points
            )
        except AttributeError:
            logging.info("Can't calculate structure factors, no experiment is loaded!")
        else:
            logging.info(f"Calculated all structure factors for '{experiment.name}' experiment.")


class SettWin:
    """Doc."""

    def __init__(self, gui, app):
        """Doc."""

        self._app = app
        self._gui = gui
        self.check_on_close = True

    def clean_up(self):
        """Doc."""

        if self.check_on_close is True:
            curr_file_path = self._gui.settingsFileName.text()

            current_state = set(wdgts.wdgt_items_to_text_lines(self._gui, wdgts.SETTINGS_TYPES))
            last_loaded_state = set(helper.read_file_to_list(curr_file_path))

            if not len(current_state) == len(last_loaded_state):
                err_hndlr(
                    RuntimeError(
                        "Something was changed in the GUI. This probably means that the default settings need to be overwritten"
                    ),
                    sys._getframe(),
                    locals(),
                )

            if current_state != last_loaded_state:
                pressed = dialog.Question(
                    "Keep changes if made? " "(otherwise, revert to last loaded settings file.)"
                ).display()
                if pressed == dialog.NO:
                    self.load(self._gui.settingsFileName.text())

        else:
            dialog.Notification("Using Current settings.").display()

    def save(self) -> None:
        """
        Write all QLineEdit, QspinBox and QdoubleSpinBox
        of settings window to 'file_path'.
        Show 'file_path' in 'settingsFileName' QLineEdit.
        """

        file_path, _ = QFileDialog.getSaveFileName(
            self._gui,
            "Save Settings",
            str(self._app.SETTINGS_DIR_PATH),
        )
        if file_path != "":
            self._gui.frame.findChild(QWidget, "settingsFileName").setText(file_path)
            wdgts.write_gui_to_file(self._gui.frame, wdgts.SETTINGS_TYPES, file_path)
            logging.debug(f"Settings file saved as: '{file_path}'")

    def load(self, file_path=""):
        """
        Read 'file_path' and write to matching widgets of settings window.
        Show 'file_path' in 'settingsFileName' QLineEdit.
        Default settings is chosen according to './settings/default_settings_choice'.
        """

        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self._gui,
                "Load Settings",
                str(self._app.SETTINGS_DIR_PATH),
            )
        if file_path != "":
            self._gui.frame.findChild(QWidget, "settingsFileName").setText(str(file_path))
            wdgts.read_file_to_gui(file_path, self._gui.frame)
            logging.debug(f"Settings file loaded: '{file_path}'")
