""" GUI Module. """

from typing import Tuple

import numpy as np
import pyqtgraph as pg
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import uic
from PyQt5.QtCore import QEvent, Qt, pyqtSlot
from PyQt5.QtWidgets import QButtonGroup, QDialog, QMainWindow, QStatusBar, QWidget

import gui.icons  # for initial icons loadout # NOQA
import logic.windows as wins_imp
from gui.icons import icons_rc  # for initial icons loadout # NOQA

MAINWINDOW_UI_PATH = "./gui/mainwindow.ui"
SETTINGSWINDOW_UI_PATH = "./gui/settingswindow.ui"
CAMERAWINDOW_UI_PATH = "./gui/camerawindow.ui"

# MEAS_COMPLETE_SOUND = "./sounds/meas_complete.wav"
#                        from PyQt5.QtMultimedia import QSound
#                                if self.time_passed == self.duration_spinbox.value():
#                                    QSound.play(MEAS_COMPLETE_SOUND);


class MainWin(QMainWindow):
    """Doc."""

    def __init__(self, app, parent: None = None) -> None:

        super(MainWin, self).__init__(parent)
        uic.loadUi(MAINWINDOW_UI_PATH, self)
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
        self.axesGroup = QButtonGroup()
        self.axesGroup.addButton(self.posAxisX)
        self.axesGroup.addButton(self.posAxisY)
        self.axesGroup.addButton(self.posAxisZ)
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
        self.excOnButton.released.connect(lambda: self.device_toggle_button_released("exc_laser"))
        self.depEmissionOn.released.connect(
            lambda: self.device_toggle_button_released(
                "dep_laser", "laser_toggle", "emission_state"
            )
        )
        self.depShutterOn.released.connect(
            lambda: self.device_toggle_button_released("dep_shutter")
        )

        # Image Scan
        self.startImgScanExc.released.connect(lambda: self.scan_button_released("SFCSImage", "Exc"))
        self.startImgScanDep.released.connect(lambda: self.scan_button_released("SFCSImage", "Dep"))
        self.startImgScanSted.released.connect(
            lambda: self.scan_button_released("SFCSImage", "Sted")
        )

        # Solution Scan
        self.startSolScanExc.released.connect(
            lambda: self.scan_button_released("SFCSSolution", "Exc")
        )
        self.startSolScanDep.released.connect(
            lambda: self.scan_button_released("SFCSSolution", "Dep")
        )
        self.startSolScanSted.released.connect(
            lambda: self.scan_button_released("SFCSSolution", "Sted")
        )

        # status bar
        self.setStatusBar(QStatusBar())

        # intialize gui
        self.actionLaser_Control.setChecked(True)
        self.actionStepper_Stage_Control.setChecked(True)
        self.stageButtonsGroup.setEnabled(False)
        self.acf.setLogMode(x=True)
        self.acf.setLimits(xMin=-5, xMax=5, yMin=-1e7, yMax=1e7)

    def closeEvent(self, event: QEvent) -> None:
        """Doc."""

        self.imp.close(event)

    @pyqtSlot()
    def on_actionRestart_triggered(self) -> None:
        """Doc."""

        self.imp.restart()

    @pyqtSlot()
    def on_actionLoadLoadout_triggered(self) -> None:
        """Doc."""

        self.imp.load()

    @pyqtSlot()
    def on_actionSaveLoadout_triggered(self) -> None:
        """Doc."""

        self.imp.save()

    @pyqtSlot(int)
    def on_minNumLines_valueChanged(self, val: int) -> None:
        """Allow only even values"""

        if val % 2:
            self.minNumLines.setValue(val - 1)

    def display_scan_pattern(self, pattern: str) -> None:
        """Doc."""

        self.imp.disp_scn_pttrn(pattern)

    @pyqtSlot()
    def on_roiImgScn_released(self) -> None:
        """Doc."""

        self.imp.roi_to_scan()

    @pyqtSlot(int)
    def on_imgShowMethod_currentIndexChanged(self) -> None:
        """Doc."""

        plane_idx = self.numPlaneShownChoice.value()
        self.imp.disp_plane_img(plane_idx)

    @pyqtSlot(int)
    def on_numPlaneShownChoice_sliderMoved(self, val: int) -> None:
        """Doc."""

        self.imp.plane_choice_changed(val)

    @pyqtSlot(float)
    def on_solScanTotalDur_valueChanged(self, float) -> None:
        """Doc."""

        self.imp.change_meas_dur(float)

    def scan_button_released(self, meas_type: str, laser_mode: str) -> None:
        """Begin/end SFCS measurement"""

        self._loop.create_task(self.imp.toggle_meas(meas_type, laser_mode))

    def device_toggle_button_released(
        self, dvc_nick: str, toggle_mthd: str = "toggle", state_attr="state"
    ) -> None:
        """Turn devices On/Off."""

        self.imp.dvc_toggle(dvc_nick, toggle_mthd, state_attr)

    @pyqtSlot()
    def on_powMode_released(self) -> None:
        """Switch between power/current depletion laser settings"""

        self.depModeStacked.setCurrentIndex(1)

    @pyqtSlot()
    def on_currMode_released(self) -> None:
        """Switch between power/current depletion laser settings"""

        self.depModeStacked.setCurrentIndex(0)

    @pyqtSlot()
    def on_depApplySettings_released(self) -> None:
        """Apply current/power mode and value"""

        self.imp.dep_sett_apply()

    @pyqtSlot(int)
    def on_solScanType_currentIndexChanged(self, index: int) -> None:
        """
        Change stacked widget 'solScanParamsStacked' index
        according to index of the combo box 'solScanTypeCombox'.
        """

        self.solScanParamsStacked.setCurrentIndex(index)

    @pyqtSlot(str)
    def on_solScanType_currentTextChanged(self, txt: str) -> None:
        """Doc."""

        self.imp.disp_scn_pttrn(txt)

    @pyqtSlot(str)
    def on_imgScanPreset_currentTextChanged(self, curr_txt: str) -> None:
        """Doc."""

        self.imp.fill_img_scan_preset_gui(curr_txt)

    @pyqtSlot(str)
    def on_solMeasPreset_currentTextChanged(self, curr_txt: str) -> None:
        """Doc."""

        self.imp.fill_sol_meas_preset_gui(curr_txt)

    @pyqtSlot()
    def on_actionSettings_triggered(self) -> None:
        """Show settings window"""

        self.imp.open_settwin()

    @pyqtSlot(bool)
    def on_actionLaser_Control_toggled(self, p0: bool) -> None:
        """Show/hide stepper laser control dock"""

        self.laserDock.setVisible(p0)

    @pyqtSlot(bool)
    def on_actionStepper_Stage_Control_toggled(self, p0: bool) -> None:
        """Show/hide stepper stage control dock"""

        self.stepperDock.setVisible(p0)

    @pyqtSlot()
    def on_actionCamera_Control_triggered(self) -> None:
        """Instantiate 'CameraWindow' object and show it"""

        self._loop.create_task(self.imp.open_camwin())

    @pyqtSlot(str)
    def on_avgInterval_currentTextChanged(self, val: str) -> None:
        """Doc."""

        self.imp.counts_avg_interval_changed(int(val))

    # -----------------------------------------------------------------------
    # LEDS
    # -----------------------------------------------------------------------

    def leds_clicked(self) -> None:
        """Doc."""

        self.imp.led_clicked(str(self.sender().objectName()))

    # -----------------------------------------------------------------------
    # Position Control
    # -----------------------------------------------------------------------

    def origin_released(self, axes: str) -> None:
        """Doc."""

        self.imp.go_to_origin(axes)

    @pyqtSlot()
    def on_goTo_released(self) -> None:
        """Doc."""

        self.imp.move_scanners()

    def axisMoveUm_released(self, sign: int) -> None:
        """Doc."""

        self.imp.displace_scanner_axis(sign)

    # -----------------------------------------------------------------------
    # Stepper Stage Dock
    # -----------------------------------------------------------------------

    @pyqtSlot()
    def on_stageOn_released(self) -> None:
        """Doc."""

        self.imp.dvc_toggle("stage")

    def stageMove_released(self, dir: str) -> None:
        """Doc."""

        self.imp.move_stage(dir=dir, steps=self.stageSteps.value())

    @pyqtSlot()
    def on_stageRelease_released(self) -> None:
        """Doc."""

        self.imp.release_stage()


class SettWin(QDialog):
    """ Documentation."""

    def __init__(self, app, parent=None) -> None:
        """Doc."""

        super(SettWin, self).__init__(parent)
        uic.loadUi(SETTINGSWINDOW_UI_PATH, self)
        self.imp = wins_imp.SettWin(self, app)

    def closeEvent(self, event: QEvent) -> None:
        """Doc."""

        self.imp.clean_up()
        self.imp.check_on_close = True

    @pyqtSlot()
    def on_saveButton_released(self) -> None:
        """Save settings as .csv"""

        self.imp.save()

    @pyqtSlot()
    def on_loadButton_released(self) -> None:
        """load settings .csv file"""

        self.imp.load()

    @pyqtSlot()
    def on_confirmButton_released(self) -> None:
        """Doc."""

        self.imp.confirm()
        self.close()


class CamWin(QWidget):
    """Doc."""

    def __init__(self, app, parent=None) -> None:
        """Doc."""

        super(CamWin, self).__init__(parent, Qt.WindowStaysOnTopHint)
        uic.loadUi(CAMERAWINDOW_UI_PATH, self)
        self.move(30, 180)
        self.imp = wins_imp.CamWin(self, app)

        # add matplotlib-ready widget (canvas) for showing camera output
        # TODO: replace this with ImageDisplay
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.gridLayout.addWidget(self.canvas, 0, 1)

    def closeEvent(self, event: QEvent) -> None:
        """Doc."""

        self.imp.clean_up()

    @pyqtSlot()
    def on_shootButton_released(self) -> None:
        """Doc."""

        self.imp.shoot()

    @pyqtSlot()
    def on_videoButton_released(self) -> None:
        """Doc."""

        self.imp.toggle_video()


class ImageDisplay:
    """Doc."""

    def __init__(self, layout):
        glw = pg.GraphicsLayoutWidget()
        self.vb = glw.addViewBox()
        self.hist = pg.HistogramLUTItem()
        glw.addItem(self.hist)
        layout.addWidget(glw)

    def add_image(self, image: np.ndarray, limit_zoomout=True, crosshair=True):
        """Doc."""

        self.vb.clear()
        image_item = pg.ImageItem(image)
        self.vb.addItem(image_item)
        self.hist.setImageItem(image_item)

        if limit_zoomout:
            self.vb.setLimits(
                xMin=0,
                xMax=image.shape[0],
                minXRange=0,
                maxXRange=image.shape[0],
                yMin=0,
                yMax=image.shape[1],
                minYRange=0,
                maxYRange=image.shape[1],
            )
        if crosshair:
            self.vLine = pg.InfiniteLine(angle=90, movable=True)
            self.hLine = pg.InfiniteLine(angle=0, movable=True)
            self.vb.addItem(self.vLine)
            self.vb.addItem(self.hLine)
            try:
                # keep crosshair at last position
                self.move_crosshair(self.last_roi)
            except AttributeError:
                # case first image since application loaded (no last_roi)
                pass
            self.vb.scene().sigMouseClicked.connect(self.mouseClicked)

        self.vb.autoRange()

    def move_crosshair(self, loc: Tuple[float, float]):
        """Doc."""

        self.vLine.setPos(loc[0])
        self.hLine.setPos(loc[1])
        self.last_roi = loc

    def mouseClicked(self, evt):
        """Doc."""
        # TODO: selected position is not accurate for some reason.

        try:
            pos = evt.pos()
        except AttributeError:
            # outside image
            pass
        else:
            #            if self.vb.sceneBoundingRect().contains(pos):
            mousePoint = self.vb.mapSceneToView(pos)
            self.move_crosshair(loc=(mousePoint.x(), mousePoint.y()))
