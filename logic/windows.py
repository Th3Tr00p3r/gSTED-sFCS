# -*- coding: utf-8 -*-
""" GUI windows implementations module. """

import logging
from typing import NoReturn

import numpy as np
import PyQt5.QtWidgets as QtWidgets
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QIcon

import gui.icons.icon_paths as icon
import logic.measurements as meas
import utilities.constants as consts
import utilities.helper as helper
from logic.scan_patterns import ScanPatternAO
from utilities.dialog import Error, Notification, Question
from utilities.errors import dvc_err_chckr as err_chckr

# TODO: None of the following should be classes. Notice that they don't have any attributes, therefore their state cannot change and thus Objects have no meaning. Instead, the methods should be functions in seperate modules, perhaps under a


class MainWin:
    """Doc."""

    def __init__(self, gui, app):
        """Doc."""

        self._app = app
        self._gui = gui

        # status bar
        self.statusBar = QtWidgets.QStatusBar()
        self._gui.setStatusBar(self.statusBar)

        # intialize gui
        self._gui.actionLaser_Control.setChecked(True)
        self._gui.actionStepper_Stage_Control.setChecked(True)
        self._gui.stageButtonsGroup.setEnabled(False)

        self._gui.countsAvg.setValue(self._gui.countsAvgSlider.value())

    def close(self, event):
        """Doc."""

        self._app.exit_app(event)

    def restart(self) -> NoReturn:
        """Restart all devices (except camera) and the timeout loop."""

        pressed = Question(
            txt="Are you sure?", title="Restarting Application"
        ).display()
        if pressed == QtWidgets.QMessageBox.Yes:
            self._app.clean_up_app(restart=True)

    def save(self) -> NoReturn:
        """Doc."""

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._gui,
            "Save Loadout",
            consts.LOADOUT_FOLDER_PATH,
            "CSV Files(*.csv *.txt)",
        )
        if file_path != "":
            helper.gui_to_csv(self._gui, file_path)
            logging.debug(f"Loadout saved as: '{file_path}'")

    def load(self, file_path="") -> NoReturn:
        """Doc."""

        if not file_path:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self._gui,
                "Load Loadout",
                consts.LOADOUT_FOLDER_PATH,
                "CSV Files(*.csv *.txt)",
            )
        if file_path != "":
            helper.csv_to_gui(file_path, self._gui)
            logging.debug(f"Loadout loaded: '{file_path}'")

    @err_chckr()
    def dvc_toggle(
        self, nick, toggle_mthd="toggle", leave_on=False, leave_off=False
    ) -> NoReturn:
        """Doc."""

        dvc = getattr(self._app.devices, nick)

        if (leave_on and dvc.state is True) or (leave_off and dvc.state is False):
            return

        if dvc.state is False:  # switch ON
            getattr(dvc, toggle_mthd)(True)

            if dvc.state:  # if managed to turn ON
                try:
                    dvc.switch_widget.set(QIcon(icon.SWITCH_ON))
                except AttributeError:
                    # no switch
                    pass

                on_icon = QIcon(dvc.led_icon_path)
                dvc.led_widget.set(on_icon)

                logging.debug(f"{dvc.log_ref} toggled ON")

                if nick == "STAGE":
                    self._gui.stageButtonsGroup.setEnabled(True)

        else:  # switch OFF
            getattr(dvc, toggle_mthd)(False)

            if not dvc.state:  # if managed to turn OFF
                try:
                    dvc.switch_widget.set(QIcon(icon.SWITCH_OFF))
                except AttributeError:
                    # no switch
                    pass

                dvc.led_widget.set(QIcon(icon.LED_OFF))

                logging.debug(f"{dvc.log_ref} toggled OFF")

                if nick == "STAGE":
                    self._gui.stageButtonsGroup.setEnabled(False)

                # set curr/pow values to zero when depletion is turned OFF
                if nick == "DEP_LASER":
                    self._gui.depActualCurr.setValue(0)
                    self._gui.depActualPow.setValue(0)

    def led_clicked(self, led_obj_name) -> NoReturn:
        """Doc."""

        LED_NAME_DVC_NICK_DICT = helper.inv_dict(consts.DVC_NICK_LED_NAME_DICT)
        dvc_nick = LED_NAME_DVC_NICK_DICT[led_obj_name]
        err_dict = getattr(self._app.devices, dvc_nick).error_dict
        if err_dict is not None:
            Error(**err_dict, custom_title=getattr(consts, dvc_nick).log_ref).display()

    @err_chckr({"DEP_LASER"})
    def dep_sett_apply(self):
        """Doc."""

        if self._gui.currMode.isChecked():  # current mode
            val = self._gui.depCurr.value()
            self._app.devices.DEP_LASER.set_current(val)
        else:  # power mode
            val = self._gui.depPow.value()
            self._app.devices.DEP_LASER.set_power(val)

    @err_chckr({"SCANNERS"})
    def move_scanners(self, axes_used: str = "XYZ") -> NoReturn:
        """Doc."""

        scanners_dvc = self._app.devices.SCANNERS

        curr_pos = tuple(getattr(self._gui, f"{ax}AOV").value() for ax in "xyz")
        data = []
        type_str = ""
        for ax, curr_ax_val in zip("XYZ", curr_pos):
            if ax in axes_used:
                data.append([curr_ax_val])
                type_str += ax

        scanners_dvc.start_write_task(data, type_str)
        scanners_dvc.toggle(True)  # restart cont. reading

        logging.debug(
            f"{getattr(consts, 'SCANNERS').log_ref} were moved to {str(curr_pos)} V"
        )

    @err_chckr({"SCANNERS"})
    def go_to_origin(self, which_axes: str) -> NoReturn:
        """Doc."""

        scanners_dvc = self._app.devices.SCANNERS

        for axis, is_chosen, org_axis_vltg in zip(
            "xyz",
            consts.AXES_TO_BOOL_TUPLE_DICT[which_axes],
            scanners_dvc.origin,
        ):
            if is_chosen:
                getattr(self._gui, f"{axis}AOV").setValue(org_axis_vltg)

        self.move_scanners(which_axes)

        logging.debug(
            f"{getattr(consts, 'SCANNERS').log_ref} sent to {which_axes} origin"
        )

    @err_chckr({"SCANNERS"})
    def displace_scanner_axis(self, sign: int) -> NoReturn:
        """Doc."""

        um_disp = sign * self._gui.axisMoveUm.value()

        if um_disp != 0.0:
            scanners_dvc = self._app.devices.SCANNERS
            axis = self._gui.axis.currentText()
            current_vltg = scanners_dvc.ai_buffer[3:, -1][consts.AX_IDX[axis]]
            um_V_RATIO = dict(zip("XYZ", scanners_dvc.um_V_ratio))[axis]
            delta_vltg = um_disp / um_V_RATIO

            axis_ao_limits = getattr(scanners_dvc, f"{axis.lower()}_ao_limits")
            new_vltg = helper.limit(
                (current_vltg + delta_vltg),
                axis_ao_limits["min_val"],
                axis_ao_limits["max_val"],
            )

            getattr(self._gui, f"{axis.lower()}AOV").setValue(new_vltg)
            self.move_scanners(axis)

            logging.debug(
                f"{getattr(consts, 'SCANNERS').log_ref}({axis}) was displaced {str(um_disp)} um"
            )

    @err_chckr({"SCANNERS"})
    def roi_to_scan(self):
        """Doc"""

        plane_idx = self._gui.numPlaneShown.value()
        try:
            line_ticks_V = self._app.last_img_scn.plane_images_data[
                plane_idx
            ].line_ticks_V
            row_ticks_V = self._app.last_img_scn.plane_images_data[
                plane_idx
            ].row_ticks_V
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

            for axis, vltg in zip("XYZ", vltgs):
                getattr(self._gui, f"{axis.lower()}AOV").setValue(vltg)

            self.move_scanners(plane_type)

            logging.debug(
                f"{getattr(consts, 'SCANNERS').log_ref} moved to ROI ({vltgs})"
            )

        except AttributeError:
            pass

    @err_chckr({"STAGE"})
    def move_stage(self, dir: str, steps: int):
        """Doc."""

        self._app.devices.STAGE.move(dir=dir, steps=steps)
        logging.info(
            f"{getattr(consts, 'STAGE').log_ref} moved {str(steps)} steps {str(dir)}"
        )

    @err_chckr({"STAGE"})
    def release_stage(self):
        """Doc."""

        self._app.devices.STAGE.release()
        logging.info(f"{getattr(consts, 'STAGE').log_ref} released")

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

    @err_chckr({"TDC", "UM232H", "SCANNERS"})
    def toggle_meas(self, type):
        """Doc."""

        current_type = self._app.meas.type
        if current_type is None:  # no meas running

            if type == "SFCSSolution":
                self._app.gui.main.imp.go_to_origin("XY")
                pattern = self._gui.solScanType.currentText()
                if pattern == "angular":
                    scan_params = consts.SOL_ANG_SCN_WDGT_COLL.read_namespace_from_gui(
                        self._app
                    )
                elif pattern == "circle":
                    scan_params = consts.SOL_CIRC_SCN_WDGT_COLL.read_namespace_from_gui(
                        self._app
                    )
                elif pattern == "static":
                    scan_params = consts.SOL_NO_SCN_WDGT_COLL.read_namespace_from_gui(
                        self._app
                    )

                scan_params.pattern = pattern

                self._app.meas = meas.SFCSSolutionMeasurement(
                    app=self._app,
                    scan_params=scan_params,
                    **consts.SOL_MEAS_WDGT_COLL.hold_objects(
                        self._app,
                        [
                            "prog_bar_wdgt",
                            "start_time_wdgt",
                            "end_time_wdgt",
                            "cal_save_intrvl_wdgt",
                            "total_files_wdgt",
                            "file_num_wdgt",
                            "g0_wdgt",
                            "decay_time_wdgt",
                            "plot_wdgt",
                        ],
                    ).read_dict_from_gui(self._app),
                )

                self._gui.startSolScan.setText("Stop \nScan")
                # TODO: add all of the following to a QButtonsGroup and en/disable them together
                self._gui.solScanMaxFileSize.setEnabled(False)
                self._gui.solScanCalDur.setEnabled(False)
                self._gui.solScanTotalDur.setEnabled(
                    self._gui.repeatSolMeas.isChecked()
                )
                self._gui.solScanDurUnits.setEnabled(False)
                self._gui.solScanFileTemplate.setEnabled(False)

            elif type == "SFCSImage":
                self._app.meas = meas.SFCSImageMeasurement(
                    app=self._app,
                    scan_params=consts.IMG_SCN_WDGT_COLL.read_namespace_from_gui(
                        self._app
                    ),
                    **consts.IMG_MEAS_WDGT_COLL.hold_objects(
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
                self._gui.startImgScan.setText("Stop \nScan")

            self._app.loop.create_task(self._app.meas.start())

        elif current_type == type:  # manual shutdown

            if type == "SFCSSolution":
                self._gui.imp.go_to_origin("XY")
                self._gui.startSolScan.setText("Start \nScan")
                # TODO: add all of the following to a QButtonsGroup and en/disable them together
                self._gui.solScanMaxFileSize.setEnabled(True)
                self._gui.solScanCalDur.setEnabled(True)
                self._gui.solScanTotalDur.setEnabled(True)
                self._gui.solScanDurUnits.setEnabled(True)
                self._gui.solScanFileTemplate.setEnabled(True)

            elif type == "SFCSImage":
                self._gui.startImgScan.setText("Start \nScan")

            self._app.meas.stop()

        else:  # other meas running
            txt = (
                f"Another type of measurement "
                f"({current_type}) is currently running."
            )
            Notification(txt).display()

    def disp_scn_pttrn(self, pattern: str):
        """Doc."""

        if pattern == "image":
            scan_params_coll = consts.IMG_SCN_WDGT_COLL
            plt_wdgt = self._gui.imgScanPattern
        elif pattern == "angular":
            scan_params_coll = consts.SOL_ANG_SCN_WDGT_COLL
            plt_wdgt = self._gui.solScanPattern
        elif pattern == "circle":
            scan_params_coll = consts.SOL_CIRC_SCN_WDGT_COLL
            plt_wdgt = self._gui.solScanPattern
        elif pattern == "static":
            scan_params_coll = None
            plt_wdgt = self._gui.solScanPattern

        if scan_params_coll:
            scan_params = scan_params_coll.read_namespace_from_gui(self._app)

            try:
                um_V_ratio = self._app.devices.SCANNERS.um_V_ratio
                ao, *_ = ScanPatternAO(pattern, scan_params, um_V_ratio).calculate_ao()
                x_data, y_data = ao[0, :], ao[1, :]
                plt_wdgt.plot(x_data, y_data, clear=True)

            except AttributeError:
                # devices not yet initialized
                pass

        # no scan
        else:
            plt_wdgt.plot([], [], clear=True)

    def disp_plane_img(
        self,
        plane_idx,
    ):
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
                return (img_data.pic1 + img_data.pic2) / (
                    img_data.norm1 + img_data.norm2
                )

        disp_mthd = self._gui.imgShowMethod.currentText()
        try:
            image_data = self._app.last_img_scn.plane_images_data[plane_idx]
        except AttributeError:
            # No last_img_scn yet
            pass
        else:
            image = build_image(image_data, disp_mthd)
            self._gui.imgScanPlot.add_image(image)

    def plane_choice_changed(self, plane_idx):
        """Doc."""

        try:
            self._app.meas.plane_shown.set(plane_idx)
            self.disp_plane_img(plane_idx)
        except AttributeError:
            # no scan performed since app init
            pass

    def change_meas_dur(self, new_dur: float):
        """Allow duration change during run only for alignment measurements"""

        try:
            if self._app.meas.type == "SFCSSolution" and self._app.meas.repeat is True:
                self._app.meas.total_duration = new_dur
                self._app.meas.duration = new_dur
        except AttributeError:
            # meas no yet defined (this is due to loading preset on startup)
            pass

    def open_settwin(self):
        """Doc."""

        self._app.gui.settings.show()
        self._app.gui.settings.activateWindow()

    @err_chckr({"CAMERA"})
    async def open_camwin(self):
        # TODO: simply making this func async doesn't help. the blocking function here is 'UC480_Camera(reopen_policy="new")'
        # from 'drivers.py', and I can't yet see a way to make it async (since I don't want to touch the API) I should try threading for this.
        """Doc."""

        self._gui.actionCamera_Control.setEnabled(False)
        self._app.gui.camera.show()
        self._app.gui.camera.activateWindow()
        self._app.gui.camera.imp.init_cam()

    @err_chckr({"COUNTER"})
    def cnts_avg_sldr_changed(self, val):
        """Doc."""

        self._gui.countsAvg.setValue(val)
        self._app.timeout_loop.updt_intrvl["cntr_avg"] = (
            val / 1000
        )  # convert to seconds

    def fill_img_scan_preset_gui(self, curr_text: str) -> NoReturn:
        """Doc."""

        consts.IMG_SCN_WDGT_COLL.write_to_gui(
            self._app, consts.IMG_SCN_WDGT_FILLOUT_DICT[curr_text]
        )
        logging.debug(f"Image scan preset configuration chosen: '{curr_text}'")

    def fill_sol_meas_preset_gui(self, curr_text: str) -> NoReturn:
        """Doc."""

        consts.SOL_MEAS_WDGT_COLL.write_to_gui(
            self._app, consts.SOL_MEAS_WDGT_FILLOUT_DICT[curr_text]
        )
        logging.debug(
            f"Solution measurement preset configuration chosen: '{curr_text}'"
        )


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
                    "Keep changes if made? "
                    "(otherwise, revert to last loaded settings file.)"
                ).display()
                if pressed == QtWidgets.QMessageBox.No:
                    self.load(self._gui.settingsFileName.text())

        else:
            Notification("Using Current settings.").display()

    def save(self):
        """
        Write all QLineEdit, QspinBox and QdoubleSpinBox
        of settings window to 'file_path' (csv).
        Show 'file_path' in 'settingsFileName' QLineEdit.
        """

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._gui,
            "Save Settings",
            consts.SETTINGS_FOLDER_PATH,
            "CSV Files(*.csv *.txt)",
        )
        if file_path != "":
            self._gui.frame.findChild(QtWidgets.QWidget, "settingsFileName").setText(
                file_path
            )
            helper.gui_to_csv(self._gui.frame, file_path)
            logging.debug(f"Settings file saved as: '{file_path}'")

    def load(self, file_path=""):
        """
        Read 'file_path' (csv) and write to matching QLineEdit,
        QspinBox and QdoubleSpinBox of settings window.
        Show 'file_path' in 'settingsFileName' QLineEdit.
        """

        if not file_path:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self._gui,
                "Load Settings",
                consts.SETTINGS_FOLDER_PATH,
                "CSV Files(*.csv *.txt)",
            )
        if file_path != "":
            self._gui.frame.findChild(QtWidgets.QWidget, "settingsFileName").setText(
                file_path
            )
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

        # add matplotlib-ready widget (canvas) for showing camera output
        self._gui.figure = plt.figure()
        self._gui.canvas = FigureCanvas(self._gui.figure)
        self._gui.gridLayout.addWidget(self._gui.canvas, 0, 1)

    def init_cam(self):
        """Doc."""

        self._cam = self._app.devices.CAMERA
        self._app.gui.main.imp.dvc_toggle("CAMERA")
        #        self._cam.video_timer.timeout.connect(self._video_timeout)
        logging.debug("Camera connection opened")

    def clean_up(self):
        """clean up before closing window"""

        if self._cam is not None:
            self.toggle_video()
            self._app.gui.main.imp.dvc_toggle("CAMERA")
            self._app.gui.main.actionCamera_Control.setEnabled(True)
            self._cam = None
            logging.debug("Camera connection closed")

    @err_chckr({"CAMERA"})
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

    @err_chckr({"CAMERA"})
    def shoot(self):
        """Doc."""

        self._app.loop.create_task(self._cam.shoot())
        logging.info("Camera photo taken")
