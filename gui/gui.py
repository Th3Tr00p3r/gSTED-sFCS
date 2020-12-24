# -*- coding: utf-8 -*-
""" GUI Module. """

from typing import NoReturn

from PyQt5 import uic
from PyQt5.QtCore import QEvent, Qt, pyqtSlot
from PyQt5.QtWidgets import QDialog, QMainWindow, QWidget

import logic.windows as wins_imp
import utilities.constants as const
from gui.icons import icons_rc  # for initial icons loadout # NOQA

# from asyncqt import asyncSlot


class MainWin(QMainWindow):
    """Doc."""

    def __init__(self, app, parent: None = None) -> NoReturn:
        """Doc."""

        super(MainWin, self).__init__(parent)
        uic.loadUi(const.MAINWINDOW_UI_PATH, self)
        self.imp = wins_imp.MainWin(self, app)

        # connecting signals & slots
        self.xAoSpinner.valueChanged.connect(self.AoSpinners_value_changed)
        self.yAoSpinner.valueChanged.connect(self.AoSpinners_value_changed)
        self.zAoSpinner.valueChanged.connect(self.AoSpinners_value_changed)

        self.axisMoveUp.released.connect(lambda: self.axisMove_released(1))
        self.axisMoveDown.released.connect(lambda: self.axisMove_released(-1))

        self.ledExc.clicked.connect(self.leds_clicked)
        self.ledDep.clicked.connect(self.leds_clicked)
        self.ledShutter.clicked.connect(self.leds_clicked)
        self.ledStage.clicked.connect(self.leds_clicked)
        self.ledCounter.clicked.connect(self.leds_clicked)
        self.ledUm232.clicked.connect(self.leds_clicked)
        self.ledTdc.clicked.connect(self.leds_clicked)
        self.ledCam.clicked.connect(self.leds_clicked)
        self.ledScn.clicked.connect(self.leds_clicked)

        self.stageUp.released.connect(lambda: self.stageMove_released("UP"))
        self.stageDown.released.connect(lambda: self.stageMove_released("DOWN"))
        self.stageLeft.released.connect(lambda: self.stageMove_released("LEFT"))
        self.stageRight.released.connect(lambda: self.stageMove_released("RIGHT"))

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

    @pyqtSlot(int)
    def on_countsAvgSlider_valueChanged(self, val: int) -> NoReturn:
        """Doc."""

        self.imp.cnts_avg_sldr_changed(val)

    # -----------------------------------------------------------------------
    # LEDS
    # -----------------------------------------------------------------------

    def leds_clicked(self) -> NoReturn:
        """Doc."""

        self.imp.led_clicked(str(self.sender().objectName()))

    # -----------------------------------------------------------------------
    # Position Control
    # -----------------------------------------------------------------------

    @pyqtSlot()
    def on_goToOrgButton_released(self) -> NoReturn:
        """Doc."""

        self.imp.go_to_origin()

    def AoSpinners_value_changed(self, _) -> NoReturn:
        """Doc."""

        self.imp.move_scanners_to()

    def axisMove_released(self, sign: int) -> NoReturn:
        """Doc."""

        self.imp.displace_scanner_axis(sign)

    # -----------------------------------------------------------------------
    # Stepper Stage Dock
    # -----------------------------------------------------------------------

    @pyqtSlot()
    def on_stageOn_released(self) -> NoReturn:
        """Doc."""

        self.imp.dvc_toggle("STAGE")

    def stageMove_released(self, dir: str) -> NoReturn:
        """Doc."""

        self.imp.move_stage(dir=dir, steps=self.stageSteps.value())

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
