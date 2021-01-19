# -*- coding: utf-8 -*-
""" GUI windows implementations module. """

import csv
import logging
from typing import NoReturn

import pandas as pd
import PyQt5.QtWidgets as QtWidgets
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QIcon

import gui.icons.icon_paths as icon
import utilities.constants as const
from logic.measurements import FCSMeasurement, SFCSSolutionMeasurement
from utilities.dialog import Error, Question
from utilities.errors import error_checker as err_chck
from utilities.errors import logic_error_handler as err_hndlr


def gui_to_csv(gui_parent, file_path):
    """Doc."""

    with open(file_path, "w") as f:
        # get all names of fields in settings window as lists (for file saving/loading)
        l1 = gui_parent.findChildren(QtWidgets.QLineEdit)
        l2 = gui_parent.findChildren(QtWidgets.QSpinBox)
        l3 = gui_parent.findChildren(QtWidgets.QDoubleSpinBox)
        l4 = gui_parent.findChildren(QtWidgets.QComboBox)
        children_list = l1 + l2 + l3 + l4

        obj_names = []
        for child in children_list:
            if not child.objectName() == "qt_spinbox_lineedit":
                if hasattr(child, "currentIndex"):  # QComboBox
                    obj_names.append(child.objectName())
                elif not child.isReadOnly():  # QSpinBox, QLineEdit
                    obj_names.append(child.objectName())

        writer = csv.writer(f)
        for i in range(len(obj_names)):
            child = gui_parent.findChild(QtWidgets.QWidget, obj_names[i])
            if hasattr(child, "value"):  # QSpinBox
                val = child.value()
            elif hasattr(child, "currentIndex"):  # QComboBox
                val = child.currentIndex()
            else:  # QLineEdit
                val = child.text()
            writer.writerow([obj_names[i], val])


def csv_to_gui(file_path, gui_parent):
    """Doc."""

    df = pd.read_csv(
        file_path,
        header=None,
        delimiter=",",
        keep_default_na=False,
        error_bad_lines=False,
    )

    for i in range(len(df)):
        obj_name, obj_val = df.iloc[i, 0], df.iloc[i, 1]
        child = gui_parent.findChild(QtWidgets.QWidget, obj_name)
        if not child == "nullptr":
            if hasattr(child, "value"):  # QSpinBox
                child.setValue(float(obj_val))
            elif hasattr(child, "currentIndex"):  # QComboBox
                child.setCurrentIndex(int(obj_val))
            elif hasattr(child, "text"):  # QLineEdit
                child.setText(obj_val)


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
            "Save Settings",
            const.LOADOUT_FOLDER_PATH,
            "CSV Files(*.csv *.txt)",
        )

        gui_to_csv(self._gui, file_path)
        logging.debug(f"Loadout saved as: '{file_path}'")

    @err_hndlr
    def load(self, file_path=""):
        """Doc."""

        if not file_path:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self._gui,
                "Load Settings",
                const.LOADOUT_FOLDER_PATH,
                "CSV Files(*.csv *.txt)",
            )

        csv_to_gui(file_path, self._gui)
        logging.debug(f"Loadout loaded: '{file_path}'")

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
            """ Return first matching 'key' to value 'val' """

            for key, value in dct.items():
                if value == val:
                    return key

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

    @err_chck({"TDC", "UM232H"})
    def toggle_meas(self, type):
        """Doc."""

        current_type = self._app.meas.type
        if current_type is None:  # no meas running

            if type == "FCS":
                self._app.meas = FCSMeasurement(
                    self._app,
                    duration_gui=self._gui.measFCSDuration,
                    prog_bar=self._gui.FCSprogressBar,
                )
                self._gui.startFcsMeasurementButton.setText("Stop \nMeasurement")

            elif type == "SFCSSolution":
                self._app.meas = SFCSSolutionMeasurement(
                    self._app,
                    duration_gui=self._gui.solScanCalIntrvl,  # TODO: is this clear enough? using file duration here instead of total
                    prog_bar=self._gui.solScanProgressBar,
                )
                self._gui.startSolScan.setText("Stop \nScan")
                self._gui.solScanMaxFileSize.setEnabled(False)
                self._gui.solScanCalTime.setEnabled(False)
                self._gui.solScanDuration.setEnabled(False)
                self._gui.solScanFileTemplate.setEnabled(False)

            self._app.loop.create_task(self._app.meas.start())

        elif current_type == type:  # manual shutdown

            if type == "FCS":
                self._gui.startFcsMeasurementButton.setText("Start \nMeasurement")

            elif type == "SFCSSolution":
                self._gui.startSolScan.setText("Start \nScan")
                self._gui.solScanMaxFileSize.setEnabled(True)
                self._gui.solScanCalTime.setEnabled(True)
                self._gui.solScanDuration.setEnabled(True)
                self._gui.solScanFileTemplate.setEnabled(True)

            self._app.meas.stop()

        else:  # other meas running
            txt = (
                f"Another type of measurement "
                f"({current_type}) is currently running."
            )
            Error(custom_txt=txt).display()

    def open_settwin(self):
        """Doc."""

        self._app.gui_dict["settings"].show()
        self._app.gui_dict["settings"].activateWindow()

    @err_chck({"CAMERA"})
    async def open_camwin(self):
        # TODO: simply making this func async doesn't help. the blocking function here is 'UC480_Camera(reopen_policy="new")'
        # from 'drivers.py', and I can't yet see a way to make it async (since I don't want to touch the API) I should try threading for this.
        """Doc."""

        self._gui.actionCamera_Control.setEnabled(False)
        self._app.gui_dict["camera"].show()
        self._app.gui_dict["camera"].activateWindow()
        self._app.gui_dict["camera"].imp.init_cam()

    @err_chck({"COUNTER"})
    def cnts_avg_sldr_changed(self, val):
        """Doc."""

        self._gui.countsAvg.setValue(val)
        self._app.timeout_loop.updt_intrvl["cntr_avg"] = (
            val / 1000
        )  # convert to seconds

    def reset(self):
        """Doc."""

        self._app.dvc_dict["UM232H"].reset()


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
            const.SETTINGS_FOLDER_PATH,
            "CSV Files(*.csv *.txt)",
        )
        self._gui.frame.findChild(QtWidgets.QWidget, "settingsFileName").setText(
            file_path
        )
        gui_to_csv(self._gui.frame, file_path)
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
                const.SETTINGS_FOLDER_PATH,
                "CSV Files(*.csv *.txt)",
            )

        self._gui.frame.findChild(QtWidgets.QWidget, "settingsFileName").setText(
            file_path
        )

        csv_to_gui(file_path, self._gui.frame)

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

        self._cam = self._app.dvc_dict["CAMERA"]
        self._app.gui_dict["main"].imp.dvc_toggle("CAMERA")
        #        self._cam.video_timer.timeout.connect(self._video_timeout)
        logging.debug("Camera connection opened")

    def clean_up(self):
        """clean up before closing window"""

        if self._cam is not None:
            self.toggle_video()
            self._app.gui_dict["main"].imp.dvc_toggle("CAMERA")
            self._app.gui_dict["main"].actionCamera_Control.setEnabled(True)
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
