# -*- coding: utf-8 -*-
""" GUI windows implementations module. """

import gui.gui as gui_module
import gui.icons.icon_paths as icon
import PyQt5.QtWidgets as QtWidgets
import utilities.constants as const
from logic.measurements import Measurement
from PyQt5.QtGui import QIcon
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
        self.statusBar = QtWidgets.QSpinBox()
        self._gui.setStatusBar(self.statusBar)

        # intialize gui
        self._gui.actionLaser_Control.setChecked(True)
        self._gui.actionStepper_Stage_Control.setChecked(True)
        self._gui.stageButtonsGroup.setEnabled(False)
        self._gui.actionLog.setChecked(True)

        self._gui.countsAvg.setValue(self._gui.countsAvgSlider.value())

        # connect signals and slots
        # TODO: (low priority) make single function to control
        # what these buttons can do (show docks, open errors if red)
        self._gui.ledExc.clicked.connect(self.show_laser_dock)
        self._gui.ledDep.clicked.connect(self.show_laser_dock)
        self._gui.ledShutter.clicked.connect(self.show_laser_dock)
        self._gui.ledStage.clicked.connect(self.show_stage_dock)
        self._gui.actionRestart.triggered.connect(self.restart)

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

        gui_switch_object = getattr(self._gui, const.ICON_DICT[nick]["SWITCH"])

        if not self._app.dvc_dict[nick].state:  # switch ON
            self._app.dvc_dict[nick].toggle(True)

            if self._app.dvc_dict[nick].state:  # if managed to turn ON
                gui_switch_object.setIcon(QIcon(icon.SWITCH_ON))

                gui_led_object = getattr(
                    self._gui, const.ICON_DICT[nick]["LED"]
                )
                on_icon = QIcon(const.ICON_DICT[nick]["ICON"])
                gui_led_object.setIcon(on_icon)

                self._app.log.update(
                    f"{const.LOG_DICT[nick]} toggled ON", tag="verbose"
                )

                if nick == "STAGE":
                    self._gui.stageButtonsGroup.setEnabled(True)

                return True

            else:
                return False

        else:  # switch OFF
            self._app.dvc_dict[nick].toggle(False)

            if not self._app.dvc_dict[nick].state:  # if managed to turn OFF
                gui_switch_object.setIcon(QIcon(icon.SWITCH_OFF))

                gui_led_object = getattr(
                    self._gui, const.ICON_DICT[nick]["LED"]
                )
                gui_led_object.setIcon(QIcon(icon.LED_OFF))

                self._app.log.update(
                    f"{const.LOG_DICT[nick]} toggled OFF", tag="verbose"
                )

                if nick in {
                    "DEP_LASER"
                }:  # set curr/pow values to zero when depletion is turned OFF
                    self._gui.depActualCurrSpinner.setValue(0)
                    self._gui.depActualPowerSpinner.setValue(0)

                return False

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

    @err_chck({"STAGE"})
    def move_stage(self, dir, steps):
        """Doc."""

        nick = "STAGE"
        self._app.dvc_dict[nick].move(dir=dir, steps=steps)

        self._app.log.update(
            f"{const.LOG_DICT[nick]} " f"moved {str(steps)} steps {str(dir)}",
            tag="verbose",
        )

    @err_chck({"STAGE"})
    def release_stage(self):
        """Doc."""

        nick = "STAGE"
        self._app.dvc_dict[nick].release()

        self._app.log.update(f"{const.LOG_DICT[nick]} released", tag="verbose")

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
        # TODO: somehow (not neccesarily here) allow stopping
        # measurement or stop automatically if errors.

        if self._app.meas.type is None:
            self._app.meas = Measurement(
                self._app,
                type="FCS",
                duration_spinner=self._gui.measFCSDuration,
                prog_bar=self._gui.FCSprogressBar,
                log=self._app.log,
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
    def open_camwin(self):
        """Doc."""

        self._gui.actionCamera_Control.setEnabled(False)
        self._app.win_dict["camera"] = gui_module.CamWin(app=self._app)
        self._app.win_dict["camera"].show()
        self._app.win_dict["camera"].activateWindow()

    def open_errwin(self):
        """Doc."""

        self._app.win_dict["errors"].show()
        self._app.win_dict["errors"].activateWindow()

    def cnts_avg_sldr_changed(self, val):
        """
        Set the spinner to show the value of the slider,
        and change the counts display frequency to as low as needed
        (lower then the averaging frequency)

        """

        self._gui.countsAvg.setValue(val)

        val = val / 1000  # convert to seconds

        curr_intrvl = self._app.dvc_dict["COUNTER"].update_time

        if val > curr_intrvl:
            self._app.timeout_loop._cntr_up.intrvl = val

        else:
            self._app.timeout_loop._cntr_up.intrvl = curr_intrvl


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
            "Keep changes if made? "
            "(otherwise, revert to last loaded settings file.)"
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

        self._gui.frame.findChild(
            QtWidgets.QWidget, "settingsFileName"
        ).setText(filepath)
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
                widget = self._gui.frame.findChild(
                    QtWidgets.QWidget, field_names[i]
                )
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

        self._app.log.update(f"Settings file saved as: '{filepath}'")

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
        self._gui.frame.findChild(
            QtWidgets.QWidget, "settingsFileName"
        ).setText(filepath)
        for i in range(len(df)):
            widget = self._gui.frame.findChild(
                QtWidgets.QWidget, df.iloc[i, 0]
            )
            if not widget == "nullptr":
                if hasattr(widget, "value"):  # spinner
                    widget.setValue(float(df.iloc[i, 1]))
                elif hasattr(widget, "text"):  # line edit
                    widget.setText(df.iloc[i, 1])

        self._app.log.update(f"Settings file loaded: '{filepath}'")


class CamWin:
    """Doc."""

    def __init__(self, gui, app):
        """Doc."""

        self._app = app
        self._gui = gui

        # add matplotlib-ready widget (canvas) for showing camera output
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg as FigureCanvas,
        )

        self._gui.figure = plt.figure()
        self._gui.canvas = FigureCanvas(self._gui.figure)
        self._gui.gridLayout.addWidget(self._gui.canvas, 0, 1)

        self.init_cam()

    def init_cam(self):
        """Doc."""

        self._cam = self._app.dvc_dict["CAMERA"]
        self._cam.toggle(True)
        self._cam.video_timer.timeout.connect(self._video_timeout)

        self._app.log.update("Camera connection opened", tag="verbose")

    def clean_up(self):
        """clean up before closing window"""

        self._cam.toggle(False)
        self._app.win_dict["main"].actionCamera_Control.setEnabled(True)
        # for restarting main loop in case camwin closed while video ON
        self._app.timeout_loop.start()

        self._app.log.update("Camera connection closed", tag="verbose")

        return None

    def toggle_video(self, bool):
        """Doc."""

        if bool:  # turn On
            self._app.timeout_loop.stop()
            self._cam.toggle_video(True)

            self._gui.videoButton.setStyleSheet(
                "background-color: " "rgb(225, 245, 225); " "color: black;"
            )
            self._gui.videoButton.setText("Video ON")

            self._app.log.update("Camera video mode ON", tag="verbose")

        else:  # turn Off
            self._cam.toggle_video(False)
            self._app.timeout_loop.start()

            self._gui.videoButton.setStyleSheet(
                "background-color: " "rgb(225, 225, 225); " "color: black;"
            )
            self._gui.videoButton.setText("Start Video")

            self._app.log.update("Camera video mode OFF", tag="verbose")

    def shoot(self):
        """Doc."""

        img = self._cam.shoot()
        self._imshow(img)

        self._app.log.update("Camera photo taken", tag="verbose")

    # private methods
    def _imshow(self, img):
        """Plot image"""

        self._gui.figure.clear()
        ax = self._gui.figure.add_subplot(111)
        ax.imshow(img)
        self._gui.canvas.draw()

    def _video_timeout(self):
        """Doc."""

        img = self._cam.latest_frame()
        self._imshow(img)


class ErrWin:
    """Doc."""

    # TODO:

    def __init__(self, gui, app):
        """Doc."""

        self._app = app
        self._gui = gui
