# -*- coding: utf-8 -*-
""" GUI Module. """

from typing import NoReturn  # , Any

from PyQt5 import uic
from PyQt5.QtCore import QEvent, Qt, pyqtSlot
from PyQt5.QtWidgets import QDialog, QMainWindow, QWidget

import logic.windows as wins_imp
import utilities.constants as const
from gui.icons import icons_rc  # for initial icons loadout # NOQA


class MainWin(QMainWindow):
    """Doc."""

    def __init__(self, app, parent: None = None) -> NoReturn:
        """Doc."""

        super(MainWin, self).__init__(parent)
        uic.loadUi(const.MAINWINDOW_UI_PATH, self)
        self.imp = wins_imp.MainWin(self, app)

    def closeEvent(self, event: QEvent) -> NoReturn:
        """Doc."""

        self.imp.close(event)

    @pyqtSlot()
    def on_actionRestart_triggered(self) -> NoReturn:
        """Doc."""

        self.imp.restart()

    @pyqtSlot()
    def on_startFcsMeasurementButton_released(self) -> NoReturn:
        """Begin FCS Measurement."""

        self.imp.toggle_FCS_meas()

    @pyqtSlot()
    def on_excOnButton_released(self) -> NoReturn:
        """Turn excitation laser On/Off."""

        self.imp.dvc_toggle("EXC_LASER")

    @pyqtSlot()
    def on_depEmissionOn_released(self) -> NoReturn:
        """Turn depletion laser On/Off"""

        self.imp.dvc_toggle("DEP_LASER")

    @pyqtSlot()
    def on_depShutterOn_released(self) -> NoReturn:
        """Turn depletion physical shutter On/Off"""

        self.imp.dvc_toggle("DEP_SHUTTER")

    @pyqtSlot()
    def on_powModeRadio_released(self) -> NoReturn:
        """Switch between power/current depletion laser settings"""

        self.depModeStacked.setCurrentIndex(1)

    @pyqtSlot()
    def on_currModeRadio_released(self) -> NoReturn:
        """Switch between power/current depletion laser settings"""

        self.depModeStacked.setCurrentIndex(0)

    @pyqtSlot()
    def on_depApplySettings_released(self) -> NoReturn:
        """Apply current/power mode and value"""
        # TODO: causes fatal error when pressed if there is DEP ERROR, fix it

        self.imp.dep_sett_apply()

    @pyqtSlot(int)
    def on_solScanTypeCombox_currentIndexChanged(self, index: int) -> NoReturn:
        """
        Change stacked widget 'solScanParamsStacked' index
        according to index of the combo box 'solScanTypeCombox'.

        """

        self.solScanParamsStacked.setCurrentIndex(index)

    @pyqtSlot()
    def on_actionSettings_triggered(self) -> NoReturn:
        """Show settings window"""

        self.imp.open_settwin()

    @pyqtSlot(bool)
    def on_actionLaser_Control_toggled(self, p0: bool) -> NoReturn:
        """Show/hide stepper laser control dock"""

        self.laserDock.setVisible(p0)

    @pyqtSlot(bool)
    def on_actionStepper_Stage_Control_toggled(self, p0: bool) -> NoReturn:
        """Show/hide stepper stage control dock"""

        self.stepperDock.setVisible(p0)

    @pyqtSlot()
    def on_actionCamera_Control_triggered(self) -> NoReturn:
        """Instantiate 'CameraWindow' object and show it"""
        # TODO: add support for 2nd camera

        self.imp.open_camwin()

    @pyqtSlot(bool)
    def on_actionLog_toggled(self, p0: bool) -> NoReturn:
        """Show/hide log dock"""

        self.logDock.setVisible(p0)

    @pyqtSlot(int)
    def on_countsAvgSlider_valueChanged(self, val: int) -> NoReturn:
        """Doc."""

        self.imp.cnts_avg_sldr_changed(val)

    # -----------------------------------------------------------------------
    # LEDS
    # -----------------------------------------------------------------------

    @pyqtSlot()
    def on_ledExc_clicked(self) -> NoReturn:
        """Doc."""

        self.imp.led_clicked(str(self.sender().objectName()))

    @pyqtSlot()
    def on_ledDep_clicked(self) -> NoReturn:
        """Doc."""

        self.imp.led_clicked(str(self.sender().objectName()))

    @pyqtSlot()
    def on_ledShutter_clicked(self) -> NoReturn:
        """Doc."""

        self.imp.led_clicked(str(self.sender().objectName()))

    @pyqtSlot()
    def on_ledStage_clicked(self) -> NoReturn:
        """Doc."""

        self.imp.led_clicked(str(self.sender().objectName()))

    @pyqtSlot()
    def on_ledCounter_clicked(self) -> NoReturn:
        """Doc."""

        self.imp.led_clicked(str(self.sender().objectName()))

    @pyqtSlot()
    def on_ledUm232_clicked(self) -> NoReturn:
        """Doc."""

        self.imp.led_clicked(str(self.sender().objectName()))

    @pyqtSlot()
    def on_ledTdc_clicked(self) -> NoReturn:
        """Doc."""

        self.imp.led_clicked(str(self.sender().objectName()))

    @pyqtSlot()
    def on_ledCam_clicked(self) -> NoReturn:
        """Doc."""

        self.imp.led_clicked(str(self.sender().objectName()))

    # -----------------------------------------------------------------------
    # Stepper Stage Dock
    # -----------------------------------------------------------------------

    @pyqtSlot()
    def on_stageOn_released(self) -> NoReturn:
        """Doc."""

        self.imp.dvc_toggle("STAGE")

    @pyqtSlot()
    def on_stageUp_released(self) -> NoReturn:
        """Doc."""

        self.imp.move_stage(dir="UP", steps=self.stageSteps.value())

    @pyqtSlot()
    def on_stageDown_released(self) -> NoReturn:
        """Doc."""

        self.imp.move_stage(dir="DOWN", steps=self.stageSteps.value())

    @pyqtSlot()
    def on_stageLeft_released(self) -> NoReturn:
        """Doc."""

        self.imp.move_stage(dir="LEFT", steps=self.stageSteps.value())

    @pyqtSlot()
    def on_stageRight_released(self) -> NoReturn:
        """Doc."""

        self.imp.move_stage(dir="RIGHT", steps=self.stageSteps.value())

    @pyqtSlot()
    def on_stageRelease_released(self) -> NoReturn:
        """Doc."""

        self.imp.release_stage()


class SettWin(QDialog):
    """ Documentation."""

    def __init__(self, app, parent=None) -> NoReturn:
        """Doc."""

        super(SettWin, self).__init__(parent)
        uic.loadUi(const.SETTINGSWINDOW_UI_PATH, self)
        self.imp = wins_imp.SettWin(self, app)

    def closeEvent(self, event: QEvent) -> NoReturn:
        """Doc."""

        self.imp.clean_up()

    @pyqtSlot()
    def on_saveButton_released(self) -> NoReturn:
        """Save settings as .csv"""
        # TODO: add all forms in main window too

        self.imp.write_csv()

    @pyqtSlot()
    def on_loadButton_released(self) -> NoReturn:
        """load settings .csv file"""
        # TODO: add all forms in main window too

        self.imp.read_csv()


class CamWin(QWidget):
    """ Documentation."""

    def __init__(self, app, parent=None) -> NoReturn:
        """Doc."""

        super(CamWin, self).__init__(parent, Qt.WindowStaysOnTopHint)
        uic.loadUi(const.CAMERAWINDOW_UI_PATH, self)
        self.move(30, 180)
        self.imp = wins_imp.CamWin(self, app)

    def closeEvent(self, event: QEvent) -> NoReturn:
        """Doc."""

        self.imp.clean_up()

    @pyqtSlot()
    def on_shootButton_released(self) -> NoReturn:
        """Doc."""

        self.imp.shoot()

    @pyqtSlot()
    def on_videoButton_released(self) -> NoReturn:
        """Doc."""

        if self.videoButton.text() == "Start Video":
            self.imp.toggle_video(True)
        else:
            self.imp.toggle_video(False)
