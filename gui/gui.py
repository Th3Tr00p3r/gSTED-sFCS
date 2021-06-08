""" GUI Module. """

from typing import NoReturn

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import uic
from PyQt5.QtCore import QEvent, Qt, pyqtSlot
from PyQt5.QtWidgets import QDialog, QMainWindow, QStatusBar, QWidget

import gui.icons  # this generates an icon resource file (see gui.py) as well as paths to all icons (see logic.py) # NOQA
import logic.windows as wins_imp
import utilities.constants as consts
from gui.icons import icons_rc  # for initial icons loadout # NOQA
from utilities.helper import ImageDisplay


class MainWin(QMainWindow):
    """Doc."""

    def __init__(self, app, parent: None = None) -> NoReturn:
        super(MainWin, self).__init__(parent)
        uic.loadUi(consts.MAINWINDOW_UI_PATH, self)
        self.move(600, 30)
        self.imp = wins_imp.MainWin(self, app)
        self._loop = app.loop

        # graphics
        self.imgScanPlot = ImageDisplay(layout=self.imageLayout)

        # scan patterns
        # image
        self.imgScanDim1.valueChanged.connect(lambda: self.display_scan_pattern("image"))
        self.imgScanDim2.valueChanged.connect(lambda: self.display_scan_pattern("image"))
        self.imgScanNumLines.valueChanged.connect(lambda: self.display_scan_pattern("image"))
        self.imgScanPPL.valueChanged.connect(lambda: self.display_scan_pattern("image"))
        self.imgScanLinFrac.valueChanged.connect(lambda: self.display_scan_pattern("image"))
        self.imgScanType.currentTextChanged.connect(lambda: self.display_scan_pattern("image"))
        # solution (angular)
        self.maxLineLen.valueChanged.connect(lambda: self.display_scan_pattern("angular"))
        self.angle.valueChanged.connect(lambda: self.display_scan_pattern("angular"))
        self.solLinFrac.valueChanged.connect(lambda: self.display_scan_pattern("angular"))
        self.lineShift.valueChanged.connect(lambda: self.display_scan_pattern("angular"))
        self.minNumLines.valueChanged.connect(lambda: self.display_scan_pattern("angular"))
        # solution (circle)
        self.circDiameter.valueChanged.connect(lambda: self.display_scan_pattern("circle"))

        # Positioning/Scanners
        self.axisMoveUp.released.connect(lambda: self.axisMoveUm_released(1))
        self.axisMoveDown.released.connect(lambda: self.axisMoveUm_released(-1))
        self.goToOrg.released.connect(lambda: self.origin_released("XYZ"))
        self.goToOrgXY.released.connect(lambda: self.origin_released("XY"))
        self.goToOrgZ.released.connect(lambda: self.origin_released("Z"))

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
            lambda: self.device_toggle_button_released("EXC_LASER", "toggle")
        )
        self.depEmissionOn.released.connect(
            lambda: self.device_toggle_button_released("DEP_LASER", "laser_toggle")
        )
        self.depShutterOn.released.connect(
            lambda: self.device_toggle_button_released("DEP_SHUTTER", "toggle")
        )

        # status bar
        self.setStatusBar(QStatusBar())

        # intialize gui
        self.actionLaser_Control.setChecked(True)
        self.actionStepper_Stage_Control.setChecked(True)
        self.stageButtonsGroup.setEnabled(False)
        self.acf.setLogMode(x=True)
        self.acf.setLimits(xMin=-5, xMax=5, yMin=-1e7, yMax=1e7)

        self.countsAvg.setValue(self.countsAvgSlider.value())

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

    @pyqtSlot(int)
    def on_minNumLines_valueChanged(self, val: int) -> NoReturn:
        """Allow only even values"""

        if val % 2:
            self.minNumLines.setValue(val - 1)

    def display_scan_pattern(self, pattern: str) -> NoReturn:
        """Doc."""

        self.imp.disp_scn_pttrn(pattern)

    @pyqtSlot()
    def on_roiImgScn_released(self) -> NoReturn:
        """Doc."""

        self.imp.roi_to_scan()

    @pyqtSlot(int)
    def on_imgShowMethod_currentIndexChanged(self) -> NoReturn:
        """Doc."""

        plane_idx = self.numPlaneShownChoice.value()
        self.imp.disp_plane_img(plane_idx)

    @pyqtSlot(int)
    def on_numPlaneShownChoice_sliderMoved(self, val: int) -> NoReturn:
        """Doc."""

        self.imp.plane_choice_changed(val)

    @pyqtSlot(float)
    def on_solScanTotalDur_valueChanged(self, float) -> NoReturn:
        """Doc."""

        self.imp.change_meas_dur(float)

    @pyqtSlot()
    def on_startImgScan_released(self) -> NoReturn:
        """Begin/end Image SFCS measurement"""

        self.imp.toggle_meas("SFCSImage")

    @pyqtSlot()
    def on_startSolScan_released(self) -> NoReturn:
        """Begin/end SFCS measurement."""

        self.imp.toggle_meas("SFCSSolution")

    def device_toggle_button_released(self, dvc_nick: str, toggle_mthd: str) -> NoReturn:
        """Turn devices On/Off."""

        self.imp.dvc_toggle(dvc_nick, toggle_mthd)

    @pyqtSlot()
    def on_powMode_released(self) -> NoReturn:
        """Switch between power/current depletion laser settings"""

        self.depModeStacked.setCurrentIndex(1)

    @pyqtSlot()
    def on_currMode_released(self) -> NoReturn:
        """Switch between power/current depletion laser settings"""

        self.depModeStacked.setCurrentIndex(0)

    @pyqtSlot()
    def on_depApplySettings_released(self) -> NoReturn:
        """Apply current/power mode and value"""

        self.imp.dep_sett_apply()

    @pyqtSlot(int)
    def on_solScanType_currentIndexChanged(self, index: int) -> NoReturn:
        """
        Change stacked widget 'solScanParamsStacked' index
        according to index of the combo box 'solScanTypeCombox'.
        """

        self.solScanParamsStacked.setCurrentIndex(index)

    @pyqtSlot(str)
    def on_solScanType_currentTextChanged(self, txt: str) -> NoReturn:
        """Doc."""

        self.imp.disp_scn_pttrn(txt)

    @pyqtSlot(str)
    def on_imgScanPreset_currentTextChanged(self, curr_txt: str) -> NoReturn:
        """Doc."""

        self.imp.fill_img_scan_preset_gui(curr_txt)

    @pyqtSlot(str)
    def on_solMeasPreset_currentTextChanged(self, curr_txt: str) -> NoReturn:
        """Doc."""

        self.imp.fill_sol_meas_preset_gui(curr_txt)

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

        self._loop.create_task(self.imp.open_camwin())

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

    def origin_released(self, axes: str) -> NoReturn:
        """Doc."""

        self.imp.go_to_origin(axes)

    @pyqtSlot()
    def on_goTo_released(self) -> NoReturn:
        """Doc."""

        self.imp.move_scanners()

    def axisMoveUm_released(self, sign: int) -> NoReturn:
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
        uic.loadUi(consts.SETTINGSWINDOW_UI_PATH, self)
        self.imp = wins_imp.SettWin(self, app)

    def closeEvent(self, event: QEvent) -> NoReturn:
        """Doc."""

        self.imp.clean_up()
        self.imp.check_on_close = True

    @pyqtSlot()
    def on_saveButton_released(self) -> NoReturn:
        """Save settings as .csv"""

        self.imp.save()

    @pyqtSlot()
    def on_loadButton_released(self) -> NoReturn:
        """load settings .csv file"""

        self.imp.load()

    @pyqtSlot()
    def on_confirmButton_released(self) -> NoReturn:
        """Doc."""

        self.imp.confirm()
        self.close()


class CamWin(QWidget):
    """Doc."""

    def __init__(self, app, parent=None) -> NoReturn:
        """Doc."""

        super(CamWin, self).__init__(parent, Qt.WindowStaysOnTopHint)
        uic.loadUi(consts.CAMERAWINDOW_UI_PATH, self)
        self.move(30, 180)
        self.imp = wins_imp.CamWin(self, app)

        # add matplotlib-ready widget (canvas) for showing camera output
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.gridLayout.addWidget(self.canvas, 0, 1)

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
