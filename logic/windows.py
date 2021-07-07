""" GUI windows implementations module. """

import logging
from types import SimpleNamespace
from typing import Tuple

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget

import logic.devices as dvcs
import logic.measurements as meas
import utilities.errors as errors
import utilities.helper as helper
import utilities.widget_collections as wdgt_colls
from logic.scan_patterns import ScanPatternAO
from utilities.dialog import Error, Notification, Question

SETTINGS_DIR_PATH = "./settings/"
LOADOUT_DIR_PATH = "./settings/loadouts/"


class MainWin:
    """Doc."""

    def __init__(self, gui, app):
        """Doc."""

        self._app = app
        self._gui = gui

    def close(self, event):
        """Doc."""

        self._app.exit_app(event)

    def restart(self) -> None:
        """Restart all devices (except camera) and the timeout loop."""

        pressed = Question(txt="Are you sure?", title="Restarting Application").display()
        if pressed == QMessageBox.Yes:
            self._app.loop.create_task(self._app.clean_up_app(restart=True))

    def save(self) -> None:
        """Doc."""

        file_path, _ = QFileDialog.getSaveFileName(
            self._gui,
            "Save Loadout",
            LOADOUT_DIR_PATH,
            "CSV Files(*.csv *.txt)",
        )
        if file_path != "":
            helper.gui_to_csv(self._gui, file_path)
            logging.debug(f"Loadout saved as: '{file_path}'")

    def load(self, file_path="") -> None:
        """Doc."""

        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self._gui,
                "Load Loadout",
                LOADOUT_DIR_PATH,
                "CSV Files(*.csv *.txt)",
            )
        if file_path != "":
            helper.csv_to_gui(file_path, self._gui)
            logging.debug(f"Loadout loaded: '{file_path}'")

    def dvc_toggle(
        self, nick, toggle_mthd="toggle", state_attr="state", leave_on=False, leave_off=False
    ) -> bool:
        """Returns False in case operation fails"""

        dvc = getattr(self._app.devices, nick)
        is_dvc_on = getattr(dvc, state_attr)

        if (leave_on and is_dvc_on) or (leave_off and not is_dvc_on):
            return True

        if not is_dvc_on:
            # switch ON
            try:
                getattr(dvc, toggle_mthd)(True)
            except errors.DeviceError:
                return False

            if (is_dvc_on := getattr(dvc, state_attr)) :
                # if managed to turn ON
                logging.debug(f"{dvc.log_ref} toggled ON")
                if nick == "stage":
                    self._gui.stageButtonsGroup.setEnabled(True)
                return True

        else:
            # switch OFF
            try:
                getattr(dvc, toggle_mthd)(False)
            except errors.DeviceError:
                return False

            if not (is_dvc_on := getattr(dvc, state_attr)):
                # if managed to turn OFF
                logging.debug(f"{dvc.log_ref} toggled OFF")

                if nick == "stage":
                    self._gui.stageButtonsGroup.setEnabled(False)

                if nick == "dep_laser":
                    # set curr/pow values to zero when depletion is turned OFF
                    self._gui.depActualCurr.setValue(0)
                    self._gui.depActualPow.setValue(0)

                return True

    def led_clicked(self, led_obj_name) -> None:
        """Doc."""

        led_name_to_nick_dict = {
            "ledExc": "exc_laser",
            "ledTdc": "TDC",
            "ledDep": "dep_laser",
            "ledShutter": "dep_shutter",
            "ledStage": "stage",
            "ledUm232h": "UM232H",
            "ledCam": "camera",
            "ledScn": "scanners",
            "ledCounter": "photon_detector",
            "ledPxlClk": "pixel_clock",
        }

        dvc_nick = led_name_to_nick_dict[led_obj_name]
        error_dict = getattr(self._app.devices, dvc_nick).error_dict
        if error_dict is not None:
            Error(**error_dict, custom_title=dvcs.DEVICE_ATTR_DICT[dvc_nick].log_ref).display()

    def dep_sett_apply(self):
        """Doc."""

        try:
            if self._gui.currMode.isChecked():  # current mode
                val = self._gui.depCurr.value()
                self._app.devices.dep_laser.set_current(val)
            else:  # power mode
                val = self._gui.depPow.value()
                self._app.devices.dep_laser.set_power(val)
        except errors.DeviceError:
            pass

    # TODO: only turn on LED while actually moving (also during scan)
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

        scanners_dvc.start_write_task(data, type_str)
        scanners_dvc.toggle(True)  # restart cont. reading

        logging.debug(
            f"{dvcs.DEVICE_ATTR_DICT['scanners'].log_ref} were moved to {str(destination)} V"
        )

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

        logging.debug(f"{dvcs.DEVICE_ATTR_DICT['scanners'].log_ref} sent to {which_axes} origin")

    def displace_scanner_axis(self, sign: int) -> None:
        """Doc."""

        def limit(val: float, min: float, max: float) -> float:
            if min <= val <= max:
                return val
            elif val < max:
                return min
            else:
                return max

        um_disp = sign * self._gui.axisMoveUm.value()

        if um_disp != 0.0:
            scanners_dvc = self._app.devices.scanners
            axis = self._gui.axesGroup.checkedButton().text()
            current_vltg = scanners_dvc.ai_buffer[-1][3:][scanners_dvc.AXIS_INDEX[axis]]
            um_V_RATIO = dict(zip("XYZ", scanners_dvc.um_v_ratio))[axis]
            delta_vltg = um_disp / um_V_RATIO

            axis_ao_limits = getattr(scanners_dvc, f"{axis.upper()}_AO_LIMITS")
            new_vltg = limit(
                (current_vltg + delta_vltg),
                axis_ao_limits["min_val"],
                axis_ao_limits["max_val"],
            )

            getattr(self._gui, f"{axis.lower()}AOV").setValue(new_vltg)
            self.move_scanners(axis)

            logging.debug(
                f"{dvcs.DEVICE_ATTR_DICT['scanners'].log_ref}({axis}) was displaced {str(um_disp)} um"
            )

    def roi_to_scan(self):
        """Doc"""

        plane_idx = self._gui.numPlaneShown.value()
        try:
            line_ticks_V = self._app.last_img_scn.plane_images_data[plane_idx].line_ticks_V
            row_ticks_V = self._app.last_img_scn.plane_images_data[plane_idx].row_ticks_V
            plane_ticks = self._app.last_img_scn.set_pnts_planes

            coord_1, coord_2 = (
                round(self._gui.imgScanPlot.vLine.value()) - 1,
                round(self._gui.imgScanPlot.hLine.value()) - 1,
            )

            dim1_vltg = line_ticks_V[coord_1]
            dim2_vltg = row_ticks_V[coord_2]
            dim3_vltg = plane_ticks[plane_idx]

            plane_type = self._app.last_img_scn.plane_type

            if plane_type == "XY":
                vltgs = (dim1_vltg, dim2_vltg, dim3_vltg)
            elif plane_type == "XZ":
                vltgs = (dim1_vltg, dim3_vltg, dim2_vltg)
            elif plane_type == "YZ":
                vltgs = (dim3_vltg, dim1_vltg, dim2_vltg)

            [
                getattr(self._gui, f"{axis.lower()}AOV").setValue(vltg)
                for axis, vltg in zip("XYZ", vltgs)
            ]

            self.move_scanners(plane_type)

            logging.debug(f"{dvcs.DEVICE_ATTR_DICT['scanners'].log_ref} moved to ROI ({vltgs})")

        except AttributeError:
            pass

    # TODO: only turn on LED while actually moving
    def move_stage(self, dir: str, steps: int):
        """Doc."""

        self._app.loop.create_task(self._app.devices.stage.move(dir=dir, steps=steps))
        logging.info(
            f"{dvcs.DEVICE_ATTR_DICT['stage'].log_ref} moved {str(steps)} steps {str(dir)}"
        )

    def release_stage(self):
        """Doc."""

        self._app.devices.stage.release()
        logging.info(f"{dvcs.DEVICE_ATTR_DICT['stage'].log_ref} released")

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

        if not (current_type := self._app.meas.type):
            # no meas running
            if meas_type == "SFCSSolution":
                pattern = self._gui.solScanType.currentText()
                if pattern == "angular":
                    scan_params = wdgt_colls.sol_ang_scan_wdgts.read_namespace_from_gui(self._app)
                elif pattern == "circle":
                    scan_params = wdgt_colls.sol_circ_scan_wdgts.read_namespace_from_gui(self._app)
                elif pattern == "static":
                    scan_params = SimpleNamespace()

                scan_params.pattern = pattern

                self._app.meas = meas.SFCSSolutionMeasurement(
                    app=self._app,
                    scan_params=scan_params,
                    laser_mode=laser_mode.lower(),
                    **wdgt_colls.sol_meas_wdgts.hold_objects(
                        self._app,
                        [
                            "prog_bar_wdgt",
                            "start_time_wdgt",
                            "end_time_wdgt",
                            "calibrated_save_intrvl_mins_wdgt",
                            "total_files_wdgt",
                            "file_num_wdgt",
                            "g0_wdgt",
                            "tau_wdgt",
                            "plot_wdgt",
                            "fit_led",
                        ],
                    ).read_dict_from_gui(self._app),
                )

                self._gui.startSolScanExc.setEnabled(False)
                self._gui.startSolScanDep.setEnabled(False)
                self._gui.startSolScanSted.setEnabled(False)
                helper.deep_getattr(self._gui, f"startSolScan{laser_mode}.setEnabled")(True)
                helper.deep_getattr(self._gui, f"startSolScan{laser_mode}.setText")("Stop \nScan")

                self._gui.solScanMaxFileSize.setEnabled(False)
                self._gui.solScanCalDur.setEnabled(False)
                self._gui.solScanTotalDur.setEnabled(self._gui.repeatSolMeas.isChecked())
                self._gui.solScanDurUnits.setEnabled(False)
                self._gui.solScanFileTemplate.setEnabled(False)

            elif meas_type == "SFCSImage":
                initial_pos = tuple(getattr(self._gui, f"{ax}AOVint").value() for ax in "xyz")
                [
                    getattr(self._gui, f"{ax}AOVint").setValue(vltg)
                    for ax, vltg in zip("xyz", initial_pos)
                ]
                self._app.meas = meas.SFCSImageMeasurement(
                    app=self._app,
                    scan_params=wdgt_colls.img_scan_wdgts.read_namespace_from_gui(self._app),
                    laser_mode=laser_mode.lower(),
                    initial_pos=initial_pos,
                    **wdgt_colls.img_meas_wdgts.hold_objects(
                        self._app,
                        [
                            "prog_bar_wdgt",
                            "curr_plane_wdgt",
                            "plane_shown",
                            "plane_choice",
                            "image_wdgt",
                            "pattern_wdgt",
                        ],
                    ).read_dict_from_gui(self._app),
                )

                self._gui.startImgScanExc.setEnabled(False)
                self._gui.startImgScanDep.setEnabled(False)
                self._gui.startImgScanSted.setEnabled(False)
                helper.deep_getattr(self._gui, f"startImgScan{laser_mode}.setEnabled")(True)
                helper.deep_getattr(self._gui, f"startImgScan{laser_mode}.setText")("Stop \nScan")

            self._app.loop.create_task(self._app.meas.run())

        elif current_type == meas_type:
            # manual shutdown
            if meas_type == "SFCSSolution":
                self._gui.startSolScanExc.setEnabled(True)
                self._gui.startSolScanDep.setEnabled(True)
                self._gui.startSolScanSted.setEnabled(True)
                helper.deep_getattr(self._gui, f"startSolScan{laser_mode}.setText")(
                    f"{laser_mode} \nScan"
                )
                self._gui.imp.go_to_origin("XY")
                # TODO: add all of the following to a QButtonsGroup and en/disable them together (see gui.py)
                self._gui.solScanMaxFileSize.setEnabled(True)
                self._gui.solScanCalDur.setEnabled(True)
                self._gui.solScanTotalDur.setEnabled(True)
                self._gui.solScanDurUnits.setEnabled(True)
                self._gui.solScanFileTemplate.setEnabled(True)

            elif meas_type == "SFCSImage":
                self._gui.startImgScanExc.setEnabled(True)
                self._gui.startImgScanDep.setEnabled(True)
                self._gui.startImgScanSted.setEnabled(True)
                helper.deep_getattr(self._gui, f"startImgScan{laser_mode}.setText")(
                    f"{laser_mode} \nScan"
                )

            if self._app.meas.is_running:
                # manual stop
                await self._app.meas.stop()

        else:
            # other meas running
            logging.warning(
                f"Another type of measurement " f"({current_type}) is currently running."
            )

    def disp_scn_pttrn(self, pattern: str):
        """Doc."""

        if pattern == "image":
            scan_params_coll = wdgt_colls.img_scan_wdgts
            plt_wdgt = self._gui.imgScanPattern
        elif pattern == "angular":
            scan_params_coll = wdgt_colls.sol_ang_scan_wdgts
            plt_wdgt = self._gui.solScanPattern
        elif pattern == "circle":
            scan_params_coll = wdgt_colls.sol_circ_scan_wdgts
            plt_wdgt = self._gui.solScanPattern
        elif pattern == "static":
            scan_params_coll = None
            plt_wdgt = self._gui.solScanPattern

        if scan_params_coll:
            scan_params = scan_params_coll.read_namespace_from_gui(self._app)

            try:
                um_v_ratio = self._app.devices.scanners.um_v_ratio
                ao, *_ = ScanPatternAO(pattern, scan_params, um_v_ratio).calculate_pattern()
                x_data, y_data = ao[0, :], ao[1, :]
                plt_wdgt.plot(x_data, y_data, clear=True)

            except (AttributeError, ZeroDivisionError):
                # AttributeError - devices not yet initialized
                # ZeroDivisionError - loadout has bad values
                pass

        else:
            # no scan
            plt_wdgt.plot([], [], clear=True)

    def disp_plane_img(self, plane_idx, auto_cross=False):
        """Doc."""

        def build_image(img_data, method):
            """Doc."""

            if method == "Forward scan - actual counts per pixel":
                return img_data.pic1

            elif method == "Forward scan - points per pixel":
                return img_data.norm1

            elif method == "Forward scan - normalized":
                return img_data.pic1 / img_data.norm1

            elif method == "Backwards scan - actual counts per pixel":
                return img_data.pic2

            elif method == "Backwards scan - points per pixel":
                return img_data.norm2

            elif method == "Backwards scan - normalized":
                return img_data.pic2 / img_data.norm2

            elif method == "Both scans - interlaced":
                p1 = img_data.pic1 / img_data.norm1
                p2 = img_data.pic2 / img_data.norm2
                n_lines = p1.shape[0] + p2.shape[0]
                p = np.zeros(p1.shape)
                p[:n_lines:2, :] = p1[:n_lines:2, :]
                p[1:n_lines:2, :] = p2[1:n_lines:2, :]
                return p

            elif method == "Both scans - averaged":
                return (img_data.pic1 + img_data.pic2) / (img_data.norm1 + img_data.norm2)

        def auto_crosshair_position(image: np.ndarray) -> Tuple[float, float]:
            """
            Gets the current image, calculates its weighted mean
            location and returns the position to which to move the crosshair.
            (later perhaps using gaussian fit?)

            """

            # Use some good-old heuristic thresholding
            n, bin_edges = np.histogram(image.ravel())
            T = 1
            for i in range(1, len(n)):
                if n[i] <= n.max() * 0.1:
                    if n[i + 1] >= n[i] * 10:
                        continue
                    else:
                        T = i
                        break
                else:
                    continue
            thresh = (bin_edges[T] - bin_edges[T - 1]) / 2
            image[image < thresh] = 0

            # calculate the weighted mean
            x = 1 / image.sum() * np.dot(np.arange(image.shape[1]), image.sum(axis=0))
            y = 1 / image.sum() * np.dot(np.arange(image.shape[0]), image.sum(axis=1))
            return (x, y)

        disp_mthd = self._gui.imgShowMethod.currentText()
        try:
            image_data = self._app.last_img_scn.plane_images_data[plane_idx]
        except AttributeError:
            # No last_img_scn yet
            pass
        else:
            image = build_image(image_data, disp_mthd)
            self._gui.imgScanPlot.add_image(image)
            if auto_cross:
                self._gui.imgScanPlot.move_crosshair(auto_crosshair_position(image))

    def plane_choice_changed(self, plane_idx):
        """Doc."""

        try:
            self._app.meas.plane_shown.set(plane_idx)
            self.disp_plane_img(plane_idx)
        except AttributeError:
            # no scan performed since app init
            pass

    def change_meas_duration(self, new_dur: float):
        """Allow duration change during run only for alignment measurements"""

        if self._app.meas.type == "SFCSSolution" and self._app.meas.repeat is True:
            self._app.meas.total_duration = new_dur
            self._app.meas.duration = new_dur

    def open_settwin(self):
        """Doc."""

        self._app.gui.settings.show()
        self._app.gui.settings.activateWindow()

    async def open_camwin(self):
        # TODO: simply making this func async doesn't help. the blocking function here is 'UC480_Camera(reopen_policy="new")'
        # from 'drivers.py', and I can't yet see a way to make it async (since I don't want to touch the API) I should try threading for this.
        """Doc."""

        self._gui.actionCamera_Control.setEnabled(False)
        self._app.gui.camera.show()
        self._app.gui.camera.activateWindow()
        self._app.gui.camera.imp.init_cam()

    def counts_avg_interval_changed(self, val: int) -> None:
        """Doc."""

        self._app.timeout_loop.updt_intrvl["cntr_avg"] = val / 1000  # convert to seconds

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

        wdgt_colls.img_scan_wdgts.write_to_gui(self._app, img_scn_wdgt_fillout_dict[curr_text])
        logging.debug(f"Image scan preset configuration chosen: '{curr_text}'")

    def fill_sol_meas_preset_gui(self, curr_text: str) -> None:
        """Doc."""

        sol_meas_wdgt_fillout_dict = {
            "Standard Alignment": {
                "scan_type": "static",
                "repeat": True,
            },
            "Standard Angular": {
                "scan_type": "angular",
                "repeat": False,
            },
            "Standard Circular": {
                "scan_type": "circle",
                "repeat": False,
            },
        }

        wdgt_colls.sol_meas_wdgts.write_to_gui(self._app, sol_meas_wdgt_fillout_dict[curr_text])
        logging.debug(f"Solution measurement preset configuration chosen: '{curr_text}'")


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

            current_state = helper.wdgt_children_as_row_list(self._gui)
            last_loaded_state = helper.csv_rows_as_list(curr_file_path)

            if set(current_state) != set(last_loaded_state):
                pressed = Question(
                    "Keep changes if made? " "(otherwise, revert to last loaded settings file.)"
                ).display()
                if pressed == QMessageBox.No:
                    self.load(self._gui.settingsFileName.text())

        else:
            Notification("Using Current settings.").display()

    def save(self):
        """
        Write all QLineEdit, QspinBox and QdoubleSpinBox
        of settings window to 'file_path' (csv).
        Show 'file_path' in 'settingsFileName' QLineEdit.
        """

        file_path, _ = QFileDialog.getSaveFileName(
            self._gui,
            "Save Settings",
            SETTINGS_DIR_PATH,
            "CSV Files(*.csv *.txt)",
        )
        if file_path != "":
            self._gui.frame.findChild(QWidget, "settingsFileName").setText(file_path)
            helper.gui_to_csv(self._gui.frame, file_path)
            logging.debug(f"Settings file saved as: '{file_path}'")

    def load(self, file_path=""):
        """
        Read 'file_path' (csv) and write to matching QLineEdit,
        QspinBox and QdoubleSpinBox of settings window.
        Show 'file_path' in 'settingsFileName' QLineEdit.
        """

        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self._gui,
                "Load Settings",
                SETTINGS_DIR_PATH,
                "CSV Files(*.csv *.txt)",
            )
        if file_path != "":
            self._gui.frame.findChild(QWidget, "settingsFileName").setText(file_path)
            helper.csv_to_gui(file_path, self._gui.frame)
            logging.debug(f"Settings file loaded: '{file_path}'")

    def confirm(self):
        """Doc."""

        self.check_on_close = False


class CamWin:
    """Doc."""

    # TODO:  Try to have the whole camera window operate in a thread

    def __init__(self, gui, app):
        """Doc."""

        self._app = app
        self._gui = gui
        self._cam = None

    def init_cam(self):
        """Doc."""

        self._cam = self._app.devices.camera
        self._app.gui.main.imp.dvc_toggle("camera")
        #        self._cam.video_timer.timeout.connect(self._video_timeout)
        logging.debug("Camera connection opened")

    def clean_up(self):
        """clean up before closing window"""

        if self._cam is not None:
            self.toggle_video()
            self._app.gui.main.imp.dvc_toggle("camera")
            self._app.gui.main.actionCamera_Control.setEnabled(True)
            self._cam = None
            logging.debug("Camera connection closed")

    def toggle_video(self):
        """Doc."""

        if self._cam.vid_state is False:  # turn On
            self._gui.videoButton.setStyleSheet(
                "background-color: " "rgb(225, 245, 225); " "color: black;"
            )
            self._gui.videoButton.setText("Video ON")

            self._cam.toggle_video(True)

            logging.debug("Camera video mode ON")

        elif self._cam is not None:  # turn Off
            self._gui.videoButton.setStyleSheet(
                "background-color: " "rgb(225, 225, 225); " "color: black;"
            )
            self._gui.videoButton.setText("Start Video")

            self._cam.toggle_video(False)

            logging.debug("Camera video mode OFF")

    def shoot(self):
        """Doc."""

        self._app.loop.create_task(self._cam.shoot())
        logging.info("Camera photo taken")
