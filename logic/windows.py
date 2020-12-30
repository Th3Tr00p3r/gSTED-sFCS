# -*- coding: utf-8 -*-
""" GUI windows implementations module. """

import logging
from typing import NoReturn

import PyQt5.QtWidgets as QtWidgets
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QIcon

import gui.icons.icon_paths as icon
import utilities.constants as const
from logic.measurements import Measurement
from utilities.dialog import Error, Question
from utilities.errors import error_checker as err_chck
from utilities.errors import logic_error_handler as err_hndlr


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

    @err_chck()
    def dvc_toggle(self, nick):
        """Doc."""

        if "SWITCH" in const.ICON_DICT[nick]:  # if device has a switch
            gui_switch_object = getattr(self._gui, const.ICON_DICT[nick]["SWITCH"])
        else:
            gui_switch_object = None

        if not self._app.dvc_dict[nick].state:  # switch ON
            self._app.dvc_dict[nick].toggle(True)

            if self._app.dvc_dict[nick].state:  # if managed to turn ON
                if gui_switch_object is not None:
                    gui_switch_object.setIcon(QIcon(icon.SWITCH_ON))

                gui_led_object = getattr(self._gui, const.ICON_DICT[nick]["LED"])
                on_icon = QIcon(const.ICON_DICT[nick]["ICON"])
                gui_led_object.setIcon(on_icon)

                logging.debug(f"{const.DVC_LOG_DICT[nick]} toggled ON")

                if nick == "STAGE":
                    self._gui.stageButtonsGroup.setEnabled(True)

                return True

            else:
                return False

        else:  # switch OFF
            self._app.dvc_dict[nick].toggle(False)

            if not self._app.dvc_dict[nick].state:  # if managed to turn OFF
                if gui_switch_object is not None:
                    gui_switch_object.setIcon(QIcon(icon.SWITCH_OFF))

                gui_led_object = getattr(self._gui, const.ICON_DICT[nick]["LED"])
                gui_led_object.setIcon(QIcon(icon.LED_OFF))

                logging.debug(f"{const.DVC_LOG_DICT[nick]} toggled OFF")

                if nick in {
                    "DEP_LASER"
                }:  # set curr/pow values to zero when depletion is turned OFF
                    self._gui.depActualCurrSpinner.setValue(0)
                    self._gui.depActualPowerSpinner.setValue(0)

                return False

    def led_clicked(self, led_obj_name):
        """Doc."""

        def get_key(dct, val):
            """
            Return (first) matching key
            if value is found in dict, else return None

            """

            for key, value in dct.items():
                if value == val:
                    return key
            return None

        nick = get_key(const.DVC_LED_NAME, led_obj_name)
        err_dict = self._app.error_dict[nick]
        if err_dict is not None:
            Error(**err_dict, custom_title=const.DVC_LOG_DICT[nick]).display()

    @err_chck({"DEP_LASER"})
    def dep_sett_apply(self):
        """Doc."""

        nick = "DEP_LASER"
        if self._gui.currModeRadio.isChecked():  # current mode
            val = self._gui.depCurrSpinner.value()
            self._app.dvc_dict[nick].set_current(val)
        else:  # power mode
            val = self._gui.depPowSpinner.value()
            self._app.dvc_dict[nick].set_power(val)

    @err_chck({"SCANNERS"})
    def move_scanners(self) -> NoReturn:
        """Doc."""

        pos_vltgs = (
            self._gui.xAoV.value(),
            self._gui.yAoV.value(),
            self._gui.zAoV.value(),
        )

        nick = "SCANNERS"
        self._app.loop.create_task(self._app.dvc_dict[nick].move_to_pos(pos_vltgs))
        logging.debug(f"{const.DVC_LOG_DICT[nick]} were moved to {str(pos_vltgs)} V")

    def go_to_origin(self, which_axes: dict) -> NoReturn:
        """Doc."""

        for axis, is_chosen, org_axis_vltg in zip(
            which_axes.keys(), which_axes.values(), const.ORIGIN
        ):
            if is_chosen:
                getattr(self._gui, f"{axis}AoV").setValue(org_axis_vltg)

        self.move_scanners()

        logging.debug(f"{const.DVC_LOG_DICT['SCANNERS']} sent to origin {which_axes}")

    def displace_scanner_axis(self, sign: int) -> NoReturn:
        """Doc."""

        axis = self._gui.axisCombox.currentText()
        current_vltg = getattr(self._gui, f"{axis}AoV").value()
        um_disp = sign * self._gui.axisMoveSpinner.value()
        # test - self._app.dvc_dict["SCANNERS"].um_V_ratio

        um_V_RATIO = dict(
            zip(("x", "y", "z"), self._app.dvc_dict["SCANNERS"].um_V_ratio)
        )[axis]

        delta_vltg = um_disp / um_V_RATIO

        getattr(self._gui, f"{axis}AoV").setValue(current_vltg + delta_vltg)

        self.move_scanners()

        logging.debug(
            f"{const.DVC_LOG_DICT['SCANNERS']}({axis}) was displaced {str(um_disp)} um"
        )

    @err_chck({"STAGE"})
    def move_stage(self, dir: str, steps: int):
        """Doc."""

        nick = "STAGE"
        self._app.dvc_dict[nick].move(dir=dir, steps=steps)
        logging.info(f"{const.DVC_LOG_DICT[nick]} moved {str(steps)} steps {str(dir)}")

    @err_chck({"STAGE"})
    def release_stage(self):
        """Doc."""

        nick = "STAGE"
        self._app.dvc_dict[nick].release()
        logging.info(f"{const.DVC_LOG_DICT[nick]} released")

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

    @err_chck({"TDC", "UM232"})
    def toggle_FCS_meas(self):
        """Doc."""

        if self._app.meas.type is None:
            self._app.meas = Measurement(
                self._app,
                type="FCS",
                duration_spinner=self._gui.measFCSDuration,
                prog_bar=self._gui.FCSprogressBar,
            )
            self._app.meas.start()
            self._gui.startFcsMeasurementButton.setText("Stop \nMeasurement")
        elif self._app.meas.type == "FCS":
            self._app.meas.stop()
            self._gui.startFcsMeasurementButton.setText("Start \nMeasurement")
        else:
            error_txt = (
                f"Another type of measurement "
                f"({self._app.meas.type}) is currently running."
            )
            Error(error_txt=error_txt).display()

    def open_settwin(self):
        """Doc."""

        self._app.win_dict["settings"].show()
        self._app.win_dict["settings"].activateWindow()

    @err_chck({"CAMERA"})
    async def open_camwin(self):
        # TODO: simply making this func async doesn't help. the blocking function here is really 'UC480_Camera(reopen_policy="new")'
        # from 'drivers.py', and I can't yet see a way to make it async (since I don't wan't to touch the API) I should try threading for this.
        """Doc."""

        self._gui.actionCamera_Control.setEnabled(False)
        self._app.win_dict["camera"].show()
        self._app.win_dict["camera"].activateWindow()
        self._app.win_dict["camera"].imp.init_cam()

    @err_chck({"COUNTER"})
    def cnts_avg_sldr_changed(self, val):
        """Doc."""

        self._gui.countsAvg.setValue(val)
        self._app.timeout_loop.updt_intrvl["cntr_avg"] = (
            val / 1000
        )  # convert to seconds


class SettWin:
    """Doc."""

    def __init__(self, gui, app):
        """Doc."""

        self._app = app
        self._gui = gui

    def clean_up(self):
        """Doc."""
        # TODO: add check to see if changes were made, if not, don't ask user

        pressed = Question(
            "Keep changes if made? " "(otherwise, revert to last loaded settings file.)"
        ).display()
        if pressed == QtWidgets.QMessageBox.No:
            self.read_csv(self._gui.settingsFileName.text())

    # public methods
    @err_hndlr
    def write_csv(self):
        """
        Write all QLineEdit, QspinBox and QdoubleSpinBox
        of settings window to 'filepath' (csv).
        Show 'filepath' in 'settingsFileName' QLineEdit.

        """
        # TODO: add support for combo box index

        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._gui,
            "Save Settings",
            const.SETTINGS_FOLDER_PATH,
            "CSV Files(*.csv *.txt)",
        )
        import csv

        self._gui.frame.findChild(QtWidgets.QWidget, "settingsFileName").setText(
            filepath
        )
        with open(filepath, "w") as stream:
            # print("saving", filepath)
            writer = csv.writer(stream)
            # get all names of fields in settings window (for file saving/loading)
            l1 = self._gui.frame.findChildren(QtWidgets.QLineEdit)
            l2 = self._gui.frame.findChildren(QtWidgets.QSpinBox)
            l3 = self._gui.frame.findChildren(QtWidgets.QDoubleSpinBox)
            field_names = [
                w.objectName()
                for w in (l1 + l2 + l3)
                if (not w.objectName() == "qt_spinbox_lineedit")
                and (not w.objectName() == "settingsFileName")
            ]  # perhaps better as for loop for readability
            # print(fieldNames)
            for i in range(len(field_names)):
                widget = self._gui.frame.findChild(QtWidgets.QWidget, field_names[i])
                if hasattr(widget, "value"):  # spinner
                    rowdata = [
                        field_names[i],
                        self._gui.frame.findChild(
                            QtWidgets.QWidget, field_names[i]
                        ).value(),
                    ]
                else:  # line edit
                    rowdata = [
                        field_names[i],
                        self._gui.frame.findChild(
                            QtWidgets.QWidget, field_names[i]
                        ).text(),
                    ]
                writer.writerow(rowdata)

        logging.debug(f"Settings file saved as: '{filepath}'")

    @err_hndlr
    def read_csv(self, filepath=""):
        """
        Read 'filepath' (csv) and write to matching QLineEdit,
        QspinBox and QdoubleSpinBox of settings window.
        Show 'filepath' in 'settingsFileName' QLineEdit.

        """
        # TODO: add support for combo box index

        if not filepath:
            filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
                self._gui,
                "Load Settings",
                const.SETTINGS_FOLDER_PATH,
                "CSV Files(*.csv *.txt)",
            )
        import pandas as pd

        df = pd.read_csv(
            filepath,
            header=None,
            delimiter=",",
            keep_default_na=False,
            error_bad_lines=False,
        )
        self._gui.frame.findChild(QtWidgets.QWidget, "settingsFileName").setText(
            filepath
        )
        for i in range(len(df)):
            widget = self._gui.frame.findChild(QtWidgets.QWidget, df.iloc[i, 0])
            if not widget == "nullptr":
                if hasattr(widget, "value"):  # spinner
                    widget.setValue(float(df.iloc[i, 1]))
                elif hasattr(widget, "text"):  # line edit
                    widget.setText(df.iloc[i, 1])

        logging.debug(f"Settings file loaded: '{filepath}'")


class CamWin:
    """Doc."""

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

        self._cam = self._app.dvc_dict["CAMERA"]
        self._app.win_dict["main"].imp.dvc_toggle("CAMERA")
        #        self._cam.video_timer.timeout.connect(self._video_timeout)
        logging.debug("Camera connection opened")

    def clean_up(self):
        """clean up before closing window"""

        if self._cam is not None:
            self.toggle_video(False)
            self._app.win_dict["main"].imp.dvc_toggle("CAMERA")
            self._app.win_dict["main"].actionCamera_Control.setEnabled(True)
            self._cam = None
            logging.debug("Camera connection closed")

    @err_chck({"CAMERA"})
    def toggle_video(self, bool):
        """Doc."""

        if bool:  # turn On
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
