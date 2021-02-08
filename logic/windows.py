# -*- coding: utf-8 -*-
""" GUI windows implementations module. """

import logging
from typing import NoReturn

import PyQt5.QtWidgets as QtWidgets
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QIcon

import gui.icons.icon_paths as icon
import logic.measurements as meas
import utilities.constants as consts
import utilities.helper as helper
from utilities.dialog import Error, Question
from utilities.errors import error_checker as err_chck
from utilities.errors import logic_error_handler as err_hndlr

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

    def restart(self):
        """Restart all devices (except camera) and the timeout loop."""

        pressed = Question(
            q_txt="Are you sure?", q_title="Restarting Program"
        ).display()
        if pressed == QtWidgets.QMessageBox.Yes:
            self._app.clean_up_app(restart=True)

    @err_hndlr
    def save(self):
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

    @err_hndlr
    def load(self, file_path=""):
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

    @err_chck()
    def dvc_toggle(self, nick):
        """Doc."""

        dvc = getattr(self._app.devices, nick)
        DVC_CONSTS = getattr(consts, nick)

        if dvc.state is False:  # switch ON
            dvc.toggle(True)

            if dvc.state:  # if managed to turn ON
                if DVC_CONSTS.switch_widget is not None:
                    DVC_CONSTS.switch_widget.set(QIcon(icon.SWITCH_ON))

                on_icon = QIcon(DVC_CONSTS.led_icon_path)
                DVC_CONSTS.led_widget.set(on_icon)

                logging.debug(f"{DVC_CONSTS.log_ref} toggled ON")

                if nick == "STAGE":
                    self._gui.stageButtonsGroup.setEnabled(True)

        else:  # switch OFF
            dvc.toggle(False)

            if not dvc.state:  # if managed to turn OFF
                if DVC_CONSTS.switch_widget is not None:
                    DVC_CONSTS.switch_widget.set(QIcon(icon.SWITCH_OFF))

                DVC_CONSTS.led_widget.set(QIcon(icon.LED_OFF))

                logging.debug(f"{DVC_CONSTS.log_ref} toggled OFF")

                if nick == "STAGE":
                    self._gui.stageButtonsGroup.setEnabled(False)

                # set curr/pow values to zero when depletion is turned OFF
                if nick == "DEP_LASER":
                    self._gui.depActualCurr.setValue(0)
                    self._gui.depActualPow.setValue(0)

    def led_clicked(self, led_obj_name):
        """Doc."""

        def dvc_nick_from_led_name(led_obj_name) -> str:
            """Doc."""

            led_to_nick_dict = {
                getattr(consts, dvc_nick).led_widget.obj_name: dvc_nick
                for dvc_nick in consts.DVC_NICKS_TUPLE
            }
            return led_to_nick_dict[led_obj_name]

        dvc_nick = dvc_nick_from_led_name(led_obj_name)
        err_dict = getattr(self._app.devices, dvc_nick).error_dict
        if err_dict is not None:
            Error(**err_dict, custom_title=getattr(consts, dvc_nick).log_ref).display()

    @err_chck({"DEP_LASER"})
    def dep_sett_apply(self):
        """Doc."""

        if self._gui.currMode.isChecked():  # current mode
            val = self._gui.depCurr.value()
            self._app.devices.DEP_LASER.set_current(val)
        else:  # power mode
            val = self._gui.depPow.value()
            self._app.devices.DEP_LASER.set_power(val)

    @err_chck({"SCANNERS"})
    def move_scanners(self) -> NoReturn:
        """Doc."""

        pos_vltgs = (
            self._gui.xAoV.value(),
            self._gui.yAoV.value(),
            self._gui.zAoV.value(),
        )

        self._app.loop.create_task(self._app.devices.SCANNERS.move_to_pos(pos_vltgs))
        logging.debug(
            f"{getattr(consts, 'SCANNERS').log_ref} were moved to {str(pos_vltgs)} V"
        )

    def go_to_origin(self, which_axes: dict) -> NoReturn:
        """Doc."""

        for axis, is_chosen, org_axis_vltg in zip(
            which_axes.keys(), which_axes.values(), self._app.devices.SCANNERS.origin
        ):
            if is_chosen:
                getattr(self._gui, f"{axis}AoV").setValue(org_axis_vltg)

        self.move_scanners()

        logging.debug(
            f"{getattr(consts, 'SCANNERS').log_ref} sent to origin {which_axes}"
        )

    def displace_scanner_axis(self, sign: int) -> NoReturn:
        """Doc."""

        axis = self._gui.axis.currentText()
        current_vltg = getattr(self._gui, f"{axis}AoV").value()
        um_disp = sign * self._gui.axisMove.value()

        um_V_RATIO = dict(zip(("x", "y", "z"), self._app.devices.SCANNERS.um_V_ratio))[
            axis
        ]

        delta_vltg = um_disp / um_V_RATIO

        getattr(self._gui, f"{axis}AoV").setValue(current_vltg + delta_vltg)

        self.move_scanners()

        logging.debug(
            f"{getattr(consts, 'SCANNERS').log_ref}({axis}) was displaced {str(um_disp)} um"
        )

    @err_chck({"STAGE"})
    def move_stage(self, dir: str, steps: int):
        """Doc."""

        self._app.devices.STAGE.move(dir=dir, steps=steps)
        logging.info(
            f"{getattr(consts, 'STAGE').log_ref} moved {str(steps)} steps {str(dir)}"
        )

    @err_chck({"STAGE"})
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

    @err_chck({"TDC", "UM232H"})
    def toggle_meas(self, type):
        """Doc."""

        current_type = self._app.meas.type
        if current_type is None:  # no meas running
            if type == "FCS":
                self._app.meas = meas.FCSMeasurement(
                    self._app,
                    **consts.FCS_MEAS_WDGT_COLL.hold_objects(
                        self._app, ["prog_bar_wdgt", "g0_wdgt", "decay_time_wdgt"]
                    ).read_dict_from_gui(self._app),
                )

                self._gui.startFcsMeasurementButton.setText("Stop \nMeasurement")

            elif type == "SFCSSolution":
                self._app.meas = meas.SFCSSolutionMeasurement(
                    app=self._app,
                    scn_params=consts.SOL_ANG_SCN_WDGT_COLL.read_dict_from_gui(
                        self._app
                    ),
                    **consts.SOL_MEAS_WDGT_COLL.hold_objects(
                        self._app,
                        [
                            "prog_bar_wdgt",
                            "start_time_wdgt",
                            "end_time_wdgt",
                            "cal_save_intrvl_wdgt",
                            "total_files_wdgt",
                            "file_num_wdgt",
                        ],
                    ).read_dict_from_gui(self._app),
                )

                self._gui.startSolScan.setText("Stop \nScan")
                # TODO: add all of the following to a QButtonsGroup and en/disable them together
                self._gui.solScanMaxFileSize.setEnabled(False)
                self._gui.solScanCalDur.setEnabled(False)
                self._gui.solScanTotalDur.setEnabled(False)
                self._gui.solScanFileTemplate.setEnabled(False)

            elif type == "SFCSImage":
                self._app.meas = meas.SFCSImageMeasurement(
                    app=self._app,
                    scn_params=consts.IMG_SCN_WDGT_COLL.read_dict_from_gui(self._app),
                    **consts.IMG_MEAS_WDGT_COLL.hold_objects(
                        self._app,
                        ["prog_bar_wdgt", "curr_line_wdgt", "curr_plane_wdgt"],
                    ).read_dict_from_gui(self._app),
                )
                self._gui.startImgScan.setText("Stop \nScan")

            self._app.loop.create_task(self._app.meas.start())

        elif current_type == type:  # manual shutdown
            if type == "FCS":
                self._gui.startFcsMeasurementButton.setText("Start \nMeasurement")

            elif type == "SFCSSolution":
                self._gui.startSolScan.setText("Start \nScan")
                # TODO: add all of the following to a QButtonsGroup and en/disable them together
                self._gui.solScanMaxFileSize.setEnabled(True)
                self._gui.solScanCalDur.setEnabled(True)
                self._gui.solScanTotalDur.setEnabled(True)
                self._gui.solScanFileTemplate.setEnabled(True)

            elif type == "SFCSImage":
                self._gui.startImgScan.setText("Start \nScan")

            self._app.meas.stop()

        else:  # other meas running
            txt = (
                f"Another type of measurement "
                f"({current_type}) is currently running."
            )
            Error(custom_txt=txt).display()

    def change_FCS_meas_dur(self, new_dur):
        """Doc."""

        if self._app.meas.type == "FCS":  # if FCS meas running
            self._app.meas.duration = new_dur

    def open_settwin(self):
        """Doc."""

        self._app.gui.settings.show()
        self._app.gui.settings.activateWindow()

    @err_chck({"CAMERA"})
    async def open_camwin(self):
        # TODO: simply making this func async doesn't help. the blocking function here is 'UC480_Camera(reopen_policy="new")'
        # from 'drivers.py', and I can't yet see a way to make it async (since I don't want to touch the API) I should try threading for this.
        """Doc."""

        self._gui.actionCamera_Control.setEnabled(False)
        self._app.gui.camera.show()
        self._app.gui.camera.activateWindow()
        self._app.gui.camera.imp.init_cam()

    @err_chck({"COUNTER"})
    def cnts_avg_sldr_changed(self, val):
        """Doc."""

        self._gui.countsAvg.setValue(val)
        self._app.timeout_loop.updt_intrvl["cntr_avg"] = (
            val / 1000
        )  # convert to seconds

    def reset(self):
        """Doc."""

        self._app.devices.UM232H.reset()

    def fill_img_scan_preset_gui(self, curr_text: str) -> NoReturn:
        """Doc."""

        # TODO: use this function at app •init to have the default value loaded
        # TODO: make laser modes into a QButtonGroup in gui.py, then can treat as a single parameter
        if curr_text == "Locate Plane - YZ Coarse":
            wdgt_vals = [1, 0, 0, "YZ", 15, 15, 80, 1000, 20, 0.9, 1, 10]
        elif curr_text == "MFC - XY compartment":
            wdgt_vals = [""]
        elif curr_text == "GB -  XY Coarse":
            wdgt_vals = [0, 1, 0, "XY", 15, 15, 80, 1000, 20, 0.9, 1, 10]
        elif curr_text == "GB - XY bead area":
            wdgt_vals = [1, 0, 0, "XY", 5, 5, 80, 1000, 20, 0.9, 1, 10]
        elif curr_text == "GB - XY single bead":
            wdgt_vals = [""]
        elif curr_text == "GB - YZ single bead":
            wdgt_vals = [""]

        consts.IMG_SCN_WDGT_COLL.write_to_gui(self._app, wdgt_vals)

        logging.info(f"Image scan preset configuration chosen: '{curr_text}'")


class SettWin:
    """Doc."""

    def __init__(self, gui, app):
        """Doc."""

        self._app = app
        self._gui = gui

    @err_hndlr
    def clean_up(self):
        """Doc."""

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

    @err_hndlr
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

    @err_hndlr
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

    @err_chck({"CAMERA"})
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

    @err_chck({"CAMERA"})
    def shoot(self):
        """Doc."""

        self._app.loop.create_task(self._cam.shoot())
        logging.info("Camera photo taken")
