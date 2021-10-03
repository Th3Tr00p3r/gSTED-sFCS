""" GUI - signals and slots"""

import PyQt5
import PyQt5.uic

import gui.icons  # for initial icons loadout # NOQA
import logic.windows
from utilities.display import Display

try:
    from gui.icons import icons_rc  # for initial icons loadout # NOQA
except ImportError:
    print("icons_rc.py was not found - icons will not initialize.", end=" ")


class MainWin(PyQt5.QtWidgets.QMainWindow):
    """Doc."""

    UI_PATH = "./gui/mainwindow.ui"

    def __init__(self, app, parent: None = None) -> None:

        super(MainWin, self).__init__(parent)
        PyQt5.uic.loadUi(self.UI_PATH, self)
        self.move(600, 30)
        self.impl = logic.windows.MainWin(self, app)
        self._loop = app.loop

        # graphics
        self.imgScanPlot = Display(layout=self.imageLayout, parent=self)
        self.solScanAcf = Display(layout=self.solScanAcfLayout, parent=self)
        self.imgScanPattern = Display(layout=self.imgScanPatternLayout)
        self.solScanPattern = Display(layout=self.solScanPatternLayout)

        # scan patterns
        # image
        self.imgScanDim1.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("image"))
        self.imgScanDim2.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("image"))
        self.imgScanNumLines.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("image"))
        self.imgScanPPL.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("image"))
        self.imgScanLinFrac.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("image"))
        self.imgScanType.currentTextChanged.connect(lambda: self.impl.disp_scn_pttrn("image"))
        # solution (angular)
        self.maxLineLen.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("angular"))
        self.angle.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("angular"))
        self.solLinFrac.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("angular"))
        self.lineShift.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("angular"))
        self.minNumLines.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("angular"))
        # solution (circle)
        self.circDiameter.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("circle"))
        self.circAoSampFreq.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("circle"))

        # Positioning/Scanners
        self.axesGroup = PyQt5.QtWidgets.QButtonGroup()
        self.axesGroup.addButton(self.posAxisX)
        self.axesGroup.addButton(self.posAxisY)
        self.axesGroup.addButton(self.posAxisZ)
        self.axisMoveUp.released.connect(lambda: self.impl.displace_scanner_axis(1))
        self.axisMoveDown.released.connect(lambda: self.impl.displace_scanner_axis(-1))
        self.goToOrg.released.connect(lambda: self.impl.go_to_origin("XYZ"))
        self.goToOrgXY.released.connect(lambda: self.impl.go_to_origin("XY"))
        self.goToOrgZ.released.connect(lambda: self.impl.go_to_origin("Z"))

        # Analysis GUI
        self.analysisDataTypeGroup = PyQt5.QtWidgets.QButtonGroup()
        self.analysisDataTypeGroup.addButton(self.imageDataImport)
        self.analysisDataTypeGroup.addButton(self.solDataImport)
        self.analysisDataTypeGroup.buttonReleased.connect(self.impl.switch_data_type)

        self.rowDiscriminationGroup = PyQt5.QtWidgets.QButtonGroup()
        self.rowDiscriminationGroup.addButton(self.solAnalysisRemoveOver)
        self.rowDiscriminationGroup.addButton(self.solAnalysisRemoveWorst)

        self.fileSelectionGroup = PyQt5.QtWidgets.QButtonGroup()
        self.fileSelectionGroup.addButton(self.solImportUseAll)
        self.fileSelectionGroup.addButton(self.solImportUse)

        self.solScanImgDisp = Display(self.solAnalysisScanImageLayout, self)
        self.solScanAcfDisp = Display(self.solAnalysisAveragingLayout, self)
        self.solScanTdcDisp = Display(self.solAnalysisTDCLayout, self)
        self.solScanGstedDisp = Display(self.solAnalysisGSTEDLayout, self)

        self.imgScanPreviewDisp = Display(self.importImgPreviewLayout)

        self.nextTemplate.released.connect(lambda: self.impl.cycle_through_data_templates("next"))
        self.prevTemplate.released.connect(lambda: self.impl.cycle_through_data_templates("prev"))

        self.solAveragingPlotSpatial.released.connect(self.impl.calculate_and_show_sol_mean_acf)
        self.solAveragingPlotTemporal.released.connect(self.impl.calculate_and_show_sol_mean_acf)

        # TDC calibration
        self.assignExcCal.released.connect(lambda: self.impl.assign_template("exc_cal"))
        self.assignStedCal.released.connect(lambda: self.impl.assign_template("sted_cal"))
        self.assignExcSamp.released.connect(lambda: self.impl.assign_template("exc_samp"))
        self.assignStedSamp.released.connect(lambda: self.impl.assign_template("sted_samp"))

        # Device LEDs
        def led_clicked(wdgt):
            self.impl.led_clicked(str(wdgt.sender().objectName()))

        self.ledExc.clicked.connect(lambda: led_clicked(self))
        self.ledDep.clicked.connect(lambda: led_clicked(self))
        self.ledShutter.clicked.connect(lambda: led_clicked(self))
        self.ledStage.clicked.connect(lambda: led_clicked(self))
        self.ledCounter.clicked.connect(lambda: led_clicked(self))
        self.ledUm232h.clicked.connect(lambda: led_clicked(self))
        self.ledTdc.clicked.connect(lambda: led_clicked(self))
        self.ledCam.clicked.connect(lambda: led_clicked(self))
        self.ledScn.clicked.connect(lambda: led_clicked(self))

        # Stage
        self.stageUp.released.connect(
            lambda: self.impl.move_stage(dir="UP", steps=self.stageSteps.value())
        )
        self.stageDown.released.connect(
            lambda: self.impl.move_stage(dir="DOWN", steps=self.stageSteps.value())
        )
        self.stageLeft.released.connect(
            lambda: self.impl.move_stage(dir="LEFT", steps=self.stageSteps.value())
        )
        self.stageRight.released.connect(
            lambda: self.impl.move_stage(dir="RIGHT", steps=self.stageSteps.value())
        )

        # Device Toggling
        self.excOnButton.released.connect(lambda: self.impl.dvc_toggle("exc_laser"))
        self.depEmissionOn.released.connect(
            lambda: self.impl.dvc_toggle("dep_laser", "laser_toggle", "emission_state")
        )
        self.depShutterOn.released.connect(lambda: self.impl.dvc_toggle("dep_shutter"))

        # Image Scan
        self.startImgScanExc.released.connect(
            lambda: self._loop.create_task(self.impl.toggle_meas("SFCSImage", "Exc"))
        )
        self.startImgScanDep.released.connect(
            lambda: self._loop.create_task(self.impl.toggle_meas("SFCSImage", "Dep"))
        )
        self.startImgScanSted.released.connect(
            lambda: self._loop.create_task(self.impl.toggle_meas("SFCSImage", "Sted"))
        )

        # Solution Scan
        self.startSolScanExc.released.connect(
            lambda: self._loop.create_task(self.impl.toggle_meas("SFCSSolution", "Exc"))
        )
        self.startSolScanDep.released.connect(
            lambda: self._loop.create_task(self.impl.toggle_meas("SFCSSolution", "Dep"))
        )
        self.startSolScanSted.released.connect(
            lambda: self._loop.create_task(self.impl.toggle_meas("SFCSSolution", "Sted"))
        )

        # status bar
        self.setStatusBar(PyQt5.QtWidgets.QStatusBar())

        # intialize gui
        self.actionLaser_Control.setChecked(True)
        self.actionStepper_Stage_Control.setChecked(True)
        self.stageButtonsGroup.setEnabled(False)

    def on_calTdc_released(self) -> None:
        """Doc."""

        self.impl.calibrate_tdc_all()

    @PyQt5.QtCore.pyqtSlot()
    def on_renameTemplate_released(self) -> None:
        """Doc."""

        self.impl.rename_template()

    @PyQt5.QtCore.pyqtSlot()
    def on_convertToMatlab_released(self) -> None:
        """Doc."""

        self.impl.convert_files_to_matlab_format()

    @PyQt5.QtCore.pyqtSlot()
    def on_solAnalysisRecalMeanAcf_released(self) -> None:
        """Doc"""

        self.impl.calculate_and_show_sol_mean_acf()

    @PyQt5.QtCore.pyqtSlot(int)
    def on_scanImgFileNum_valueChanged(self, val: int) -> None:
        """Doc."""

        self.impl.display_scan_image(val)

    @PyQt5.QtCore.pyqtSlot()
    def on_removeImportedSolData_released(self) -> None:
        """Doc."""

        self.impl.remove_imported_template()

    @PyQt5.QtCore.pyqtSlot()
    def on_dataDirLogUpdate_released(self) -> None:
        """Doc."""

        self.impl.update_dir_log_file()

    @PyQt5.QtCore.pyqtSlot()
    def on_importSolData_released(self) -> None:
        """Doc."""

        self.impl.import_sol_data()

    @PyQt5.QtCore.pyqtSlot()
    def on_openDir_released(self) -> None:
        """Doc."""

        self.impl.open_data_dir()

    @PyQt5.QtCore.pyqtSlot(str)
    def on_dataYear_currentTextChanged(self, year: str) -> None:
        """Doc."""

        self.impl.populate_data_dates_from_year(year)

    @PyQt5.QtCore.pyqtSlot(str)
    def on_dataMonth_currentTextChanged(self, month: str) -> None:
        """Doc."""

        self.impl.populate_data_dates_from_month(month)

    @PyQt5.QtCore.pyqtSlot(str)
    def on_dataDay_currentTextChanged(self, day: str) -> None:
        """Doc."""

        self.impl.populate_data_templates_from_day(day)

    @PyQt5.QtCore.pyqtSlot(str)
    def on_dataTemplate_currentTextChanged(self, template: str) -> None:
        """Doc."""

        self.impl.show_num_files(template)
        self.impl.update_dir_log_wdgt(template)
        self.impl.preview_img_scan(template)

    @PyQt5.QtCore.pyqtSlot(str)
    def on_importedSolDataTemplates_currentTextChanged(self, template: str) -> None:
        """Doc."""

        self.impl.populate_sol_meas_analysis(template)

    def closeEvent(self, event: PyQt5.QtCore.QEvent) -> None:
        """Doc."""

        self.impl.close(event)

    @PyQt5.QtCore.pyqtSlot()
    def on_actionRestart_triggered(self) -> None:
        """Doc."""

        self.impl.restart()

    @PyQt5.QtCore.pyqtSlot()
    def on_actionLoadLoadout_triggered(self) -> None:
        """Doc."""

        self.impl.load()

    @PyQt5.QtCore.pyqtSlot()
    def on_actionSaveLoadout_triggered(self) -> None:
        """Doc."""

        self.impl.save()

    @PyQt5.QtCore.pyqtSlot(int)
    def on_minNumLines_valueChanged(self, val: int) -> None:
        """Allow only even values"""

        if val % 2:
            self.minNumLines.setValue(val - 1)

    @PyQt5.QtCore.pyqtSlot()
    def on_roiImgScn_released(self) -> None:
        """Doc."""

        self.impl.roi_to_scan()

    @PyQt5.QtCore.pyqtSlot(int)
    def on_scaleImgScan_valueChanged(self, percent_factor: int) -> None:
        """Doc."""

        self.impl.auto_scale_image(percent_factor)

    @PyQt5.QtCore.pyqtSlot(int)
    def on_imgShowMethod_currentIndexChanged(self) -> None:
        """Doc."""

        plane_idx = self.numPlaneShownChoice.value()
        self.impl.disp_plane_img(plane_idx)

    @PyQt5.QtCore.pyqtSlot(int)
    def on_numPlaneShownChoice_sliderMoved(self, val: int) -> None:
        """Doc."""

        self.impl.plane_choice_changed(val)

    @PyQt5.QtCore.pyqtSlot(float)
    def on_solScanDur_valueChanged(self, float) -> None:
        """Doc."""

        self.impl.change_meas_duration(float)

    @PyQt5.QtCore.pyqtSlot()
    def on_powMode_released(self) -> None:
        """Switch between power/current depletion laser settings"""

        self.depModeStacked.setCurrentIndex(1)

    @PyQt5.QtCore.pyqtSlot()
    def on_currMode_released(self) -> None:
        """Switch between power/current depletion laser settings"""

        self.depModeStacked.setCurrentIndex(0)

    @PyQt5.QtCore.pyqtSlot()
    def on_depApplySettings_released(self) -> None:
        """Apply current/power mode and value"""

        self.impl.dep_sett_apply()

    @PyQt5.QtCore.pyqtSlot(int)
    def on_solScanType_currentIndexChanged(self, index: int) -> None:
        """
        Change stacked widget 'solScanParamsStacked' index
        according to index of the combo box 'solScanTypeCombox'.
        """

        self.solScanParamsStacked.setCurrentIndex(index)

    @PyQt5.QtCore.pyqtSlot(str)
    def on_solScanType_currentTextChanged(self, txt: str) -> None:
        """Doc."""

        self.impl.disp_scn_pttrn(txt)

    @PyQt5.QtCore.pyqtSlot(str)
    def on_imgScanPreset_currentTextChanged(self, curr_txt: str) -> None:
        """Doc."""

        self.impl.fill_img_scan_preset_gui(curr_txt)

    @PyQt5.QtCore.pyqtSlot(str)
    def on_solMeasPreset_currentTextChanged(self, curr_txt: str) -> None:
        """Doc."""

        self.impl.fill_sol_meas_preset_gui(curr_txt)

    @PyQt5.QtCore.pyqtSlot()
    def on_actionSettings_triggered(self) -> None:
        """Show settings window"""

        self.impl.open_settwin()

    @PyQt5.QtCore.pyqtSlot(bool)
    def on_actionLaser_Control_toggled(self, p0: bool) -> None:
        """Show/hide stepper laser control dock"""

        self.laserDock.setVisible(p0)

    @PyQt5.QtCore.pyqtSlot(bool)
    def on_actionStepper_Stage_Control_toggled(self, p0: bool) -> None:
        """Show/hide stepper stage control dock"""

        self.stepperDock.setVisible(p0)

    @PyQt5.QtCore.pyqtSlot()
    def on_actionCamera_Control_triggered(self) -> None:
        """Instantiate 'CameraWindow' object and show it"""

        self._loop.create_task(self.impl.open_camwin())

    @PyQt5.QtCore.pyqtSlot(str)
    def on_avgInterval_currentTextChanged(self, val: str) -> None:
        """Doc."""

        self.impl.counts_avg_interval_changed(int(val))

    # -----------------------------------------------------------------------
    # Position Control
    # -----------------------------------------------------------------------

    @PyQt5.QtCore.pyqtSlot()
    def on_goTo_released(self) -> None:
        """Doc."""

        self.impl.move_scanners()

    # -----------------------------------------------------------------------
    # Stepper Stage Dock
    # -----------------------------------------------------------------------

    @PyQt5.QtCore.pyqtSlot()
    def on_stageOn_released(self) -> None:
        """Doc."""

        self.impl.dvc_toggle("stage")


class SettWin(PyQt5.QtWidgets.QDialog):
    """Documentation."""

    UI_PATH = "./gui/settingswindow.ui"

    def __init__(self, app, parent=None) -> None:
        """Doc."""

        super(SettWin, self).__init__(parent)
        PyQt5.uic.loadUi(self.UI_PATH, self)
        self.impl = logic.windows.SettWin(self, app)

    def closeEvent(self, event: PyQt5.QtCore.QEvent) -> None:
        """Doc."""

        self.impl.clean_up()
        self.impl.check_on_close = True

    @PyQt5.QtCore.pyqtSlot()
    def on_saveButton_released(self) -> None:
        """Save settings."""

        self.impl.save()

    @PyQt5.QtCore.pyqtSlot()
    def on_loadButton_released(self) -> None:
        """load settings."""

        self.impl.load()

    @PyQt5.QtCore.pyqtSlot()
    def on_confirmButton_released(self) -> None:
        """Doc."""

        self.impl.confirm()
        self.close()


class CamWin(PyQt5.QtWidgets.QWidget):
    """Doc."""

    UI_PATH = "./gui/camerawindow.ui"

    def __init__(self, app, parent=None) -> None:
        """Doc."""

        super(CamWin, self).__init__(parent, PyQt5.QtCore.Qt.WindowStaysOnTopHint)
        PyQt5.uic.loadUi(self.UI_PATH, self)
        self.move(30, 180)
        self.impl = logic.windows.CamWin(self, app)
        self._loop = app.loop

        # add matplotlib-ready widget (canvas) for showing camera output
        self.ImgDisp = Display(self.imageDisplayLayout, self)

    def closeEvent(self, event: PyQt5.QtCore.QEvent) -> None:
        """Doc."""

        self._loop.create_task(self.impl.clean_up())

    @PyQt5.QtCore.pyqtSlot()
    def on_shootButton_released(self) -> None:
        """Doc."""

        self.impl.shoot()

    @PyQt5.QtCore.pyqtSlot()
    def on_videoButton_released(self) -> None:
        """Doc."""

        self._loop.create_task(self.impl.toggle_video())
