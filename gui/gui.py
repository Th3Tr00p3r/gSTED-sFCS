# -*- coding: utf-8 -*-
""" GUI Module. """

from typing import NoReturn

from PyQt5 import uic
from PyQt5.QtCore import QEvent, Qt, pyqtSlot
from PyQt5.QtWidgets import QDialog, QMainWindow, QWidget

# TODO: see if 'import gui.icons' can work from here instead of in main.py (test by switching some icon in QDesigner)
import gui.icons  # this generates an icon resource file (see gui.py) as well as paths to all icons (see logic.py) # NOQA
import logic.windows as wins_imp
import utilities.constants as const
from gui.icons import icons_rc  # for initial icons loadout # NOQA


class MainWin(QMainWindow):
    """Doc."""

    def __init__(self, app, parent: None = None) -> NoReturn:
        super(MainWin, self).__init__(parent)
        uic.loadUi(const.MAINWINDOW_UI_PATH, self)
        self.move(600, 30)
        self.imp = wins_imp.MainWin(self, app)
        self._loop = app.loop

        # Positioning/Scanners
        self.axisMoveUp.released.connect(lambda: self.axisMove_released(1))
        self.axisMoveDown.released.connect(lambda: self.axisMove_released(-1))
        self.goToOrg.released.connect(
            lambda: self.origin_released({"x": True, "y": True, "z": True})
        )
        self.goToOrgXY.released.connect(
            lambda: self.origin_released({"x": True, "y": True, "z": False})
        )
        self.goToOrgZ.released.connect(
            lambda: self.origin_released({"x": False, "y": False, "z": True})
        )

        # Device LEDs
        self.ledExc.clicked.connect(self.leds_clicked)
        self.ledDep.clicked.connect(self.leds_clicked)
        self.ledShutter.clicked.connect(self.leds_clicked)
        self.ledStage.clicked.connect(self.leds_clicked)
        self.ledCounter.clicked.connect(self.leds_clicked)
        self.ledUm232h.clicked.connect(self.leds_clicked)
        self.ledTdc.clicked.connect(self.leds_clicked)
        self.ledCam.clicked.connect(self.leds_clicked)
        self.ledScn.clicked.connect(self.leds_clicked)

        # Stage
        self.stageUp.released.connect(lambda: self.stageMove_released("UP"))
        self.stageDown.released.connect(lambda: self.stageMove_released("DOWN"))
        self.stageLeft.released.connect(lambda: self.stageMove_released("LEFT"))
        self.stageRight.released.connect(lambda: self.stageMove_released("RIGHT"))

        # Device Toggling
        self.excOnButton.released.connect(
            lambda: self.device_toggle_button_released("EXC_LASER")
        )
        self.depEmissionOn.released.connect(
            lambda: self.device_toggle_button_released("DEP_LASER")
        )
        self.depShutterOn.released.connect(
            lambda: self.device_toggle_button_released("DEP_SHUTTER")
        )

    def closeEvent(self, event: QEvent) -> NoReturn:
        """Doc."""

        self.imp.close(event)

    @pyqtSlot()
    def on_actionRestart_triggered(self) -> NoReturn:
        """Doc."""

        self.imp.restart()

    @pyqtSlot()
    def on_actionLoadLoadout_triggered(self) -> NoReturn:
        """Doc."""

        self.imp.load()

    @pyqtSlot()
    def on_actionSaveLoadout_triggered(self) -> NoReturn:
        """Doc."""

        self.imp.save()

    @pyqtSlot()
    def on_startFcsMeasurementButton_released(self) -> NoReturn:
        """Begin/end FCS measurement."""

        self.imp.toggle_meas("FCS")

    @pyqtSlot()
    def on_startSolScan_released(self) -> NoReturn:
        """Begin/end SFCS measurement."""

        self.imp.toggle_meas("SFCSSolution")

    def device_toggle_button_released(self, dvc_nick: str) -> NoReturn:
        """Turn devices On/Off."""

        self.imp.dvc_toggle(dvc_nick)

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

        self._loop.create_task(self.imp.open_camwin())

    @pyqtSlot(int)
    def on_countsAvgSlider_valueChanged(self, val: int) -> NoReturn:
        """Doc."""

        self.imp.cnts_avg_sldr_changed(val)

    @pyqtSlot()
    def on_resetUm232_released(self) -> NoReturn:
        """Doc."""

        self.imp.reset()

    # -----------------------------------------------------------------------
    # LEDS
    # -----------------------------------------------------------------------

    def leds_clicked(self) -> NoReturn:
        """Doc."""

        self.imp.led_clicked(str(self.sender().objectName()))

    # -----------------------------------------------------------------------
    # Position Control
    # -----------------------------------------------------------------------

    def origin_released(self, which_axes: dict) -> NoReturn:
        """Doc."""

        self.imp.go_to_origin(which_axes)

    @pyqtSlot()
    def on_goTo_released(self) -> NoReturn:
        """Doc."""

        self.imp.move_scanners()

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

        self.imp.save()

    @pyqtSlot()
    def on_loadButton_released(self) -> NoReturn:
        """load settings .csv file"""

        self.imp.load()


# TODO: Decide what this button does
#    @pyqtSlot()
#    def on_confirmButton_released(self) -> NoReturn:
#        """Doc."""
#


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

        self.imp.toggle_video()
