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
        self.imp = logic.windows.MainWin(self, app)
        self._loop = app.loop

        # graphics
        self.imgScanPlot = Display(layout=self.imageLayout, parent=self)
        self.solScanAcf = Display(layout=self.solScanAcfLayout, parent=self)
        self.imgScanPattern = Display(layout=self.imgScanPatternLayout)
        self.solScanPattern = Display(layout=self.solScanPatternLayout)

        # scan patterns
        # image
        self.imgScanDim1.valueChanged.connect(lambda: self.imp.disp_scn_pttrn("image"))
        self.imgScanDim2.valueChanged.connect(lambda: self.imp.disp_scn_pttrn("image"))
        self.imgScanNumLines.valueChanged.connect(lambda: self.imp.disp_scn_pttrn("image"))
        self.imgScanPPL.valueChanged.connect(lambda: self.imp.disp_scn_pttrn("image"))
        self.imgScanLinFrac.valueChanged.connect(lambda: self.imp.disp_scn_pttrn("image"))
        self.imgScanType.currentTextChanged.connect(lambda: self.imp.disp_scn_pttrn("image"))
        # solution (angular)
        self.maxLineLen.valueChanged.connect(lambda: self.imp.disp_scn_pttrn("angular"))
        self.angle.valueChanged.connect(lambda: self.imp.disp_scn_pttrn("angular"))
        self.solLinFrac.valueChanged.connect(lambda: self.imp.disp_scn_pttrn("angular"))
        self.lineShift.valueChanged.connect(lambda: self.imp.disp_scn_pttrn("angular"))
        self.minNumLines.valueChanged.connect(lambda: self.imp.disp_scn_pttrn("angular"))
        # solution (circle)
        self.circDiameter.valueChanged.connect(lambda: self.imp.disp_scn_pttrn("circle"))
        self.circAoSampFreq.valueChanged.connect(lambda: self.imp.disp_scn_pttrn("circle"))

        # Positioning/Scanners
        self.axesGroup = PyQt5.QtWidgets.QButtonGroup()
        self.axesGroup.addButton(self.posAxisX)
        self.axesGroup.addButton(self.posAxisY)
        self.axesGroup.addButton(self.posAxisZ)
        self.axisMoveUp.released.connect(lambda: self.imp.displace_scanner_axis(1))
        self.axisMoveDown.released.connect(lambda: self.imp.displace_scanner_axis(-1))
        self.goToOrg.released.connect(lambda: self.imp.go_to_origin("XYZ"))
        self.goToOrgXY.released.connect(lambda: self.imp.go_to_origin("XY"))
        self.goToOrgZ.released.connect(lambda: self.imp.go_to_origin("Z"))

        # Analysis GUI
        self.analysisDataTypeGroup = PyQt5.QtWidgets.QButtonGroup()
        self.analysisDataTypeGroup.addButton(self.imageDataImport)
        self.analysisDataTypeGroup.addButton(self.solDataImport)
        self.analysisDataTypeGroup.buttonReleased.connect(self.imp.populate_all_data_dates)

        self.rowDiscriminationGroup = PyQt5.QtWidgets.QButtonGroup()
        self.rowDiscriminationGroup.addButton(self.solAnalysisRemoveOver)
        self.rowDiscriminationGroup.addButton(self.solAnalysisRemoveWorst)

        self.fileSelectionGroup = PyQt5.QtWidgets.QButtonGroup()
        self.fileSelectionGroup.addButton(self.solImportUseAll)
        self.fileSelectionGroup.addButton(self.solImportUse)

        self.solScanImgDisp = Display(self.solAnalysisScanImageLayout, self)
        self.solScanAcfDisp = Display(self.solAnalysisAveragingLayout, self)
        self.solScanGstedDisp = Display(self.solAnalysisGSTEDLayout, self)

        self.imgScanPreviewDisp = Display(self.importImgPreviewLayout)

        self.nextTemplate.released.connect(lambda: self.imp.cycle_through_data_templates("next"))
        self.prevTemplate.released.connect(lambda: self.imp.cycle_through_data_templates("prev"))

        self.solAveragingPlotSpatial.released.connect(self.imp.calculate_and_show_sol_mean_acf)
        self.solAveragingPlotTemporal.released.connect(self.imp.calculate_and_show_sol_mean_acf)

        # Device LEDs
        def led_clicked(wdgt):
            self.imp.led_clicked(str(wdgt.sender().objectName()))

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
            lambda: self.imp.move_stage(dir="UP", steps=self.stageSteps.value())
        )
        self.stageDown.released.connect(
            lambda: self.imp.move_stage(dir="DOWN", steps=self.stageSteps.value())
        )
        self.stageLeft.released.connect(
            lambda: self.imp.move_stage(dir="LEFT", steps=self.stageSteps.value())
        )
        self.stageRight.released.connect(
            lambda: self.imp.move_stage(dir="RIGHT", steps=self.stageSteps.value())
        )

        # Device Toggling
        self.excOnButton.released.connect(lambda: self.imp.dvc_toggle("exc_laser"))
        self.depEmissionOn.released.connect(
            lambda: self.imp.dvc_toggle("dep_laser", "laser_toggle", "emission_state")
        )
        self.depShutterOn.released.connect(lambda: self.imp.dvc_toggle("dep_shutter"))

        # Image Scan
        self.startImgScanExc.released.connect(
            lambda: self._loop.create_task(self.imp.toggle_meas("SFCSImage", "Exc"))
        )
        self.startImgScanDep.released.connect(
            lambda: self._loop.create_task(self.imp.toggle_meas("SFCSImage", "Dep"))
        )
        self.startImgScanSted.released.connect(
            lambda: self._loop.create_task(self.imp.toggle_meas("SFCSImage", "Sted"))
        )

        # Solution Scan
        self.startSolScanExc.released.connect(
            lambda: self._loop.create_task(self.imp.toggle_meas("SFCSSolution", "Exc"))
        )
        self.startSolScanDep.released.connect(
            lambda: self._loop.create_task(self.imp.toggle_meas("SFCSSolution", "Dep"))
        )
        self.startSolScanSted.released.connect(
            lambda: self._loop.create_task(self.imp.toggle_meas("SFCSSolution", "Sted"))
        )

        # status bar
        self.setStatusBar(PyQt5.QtWidgets.QStatusBar())

        # intialize gui
        self.actionLaser_Control.setChecked(True)
        self.actionStepper_Stage_Control.setChecked(True)
        self.stageButtonsGroup.setEnabled(False)

    @PyQt5.QtCore.pyqtSlot()
    def on_renameTemplate_released(self) -> None:
        """Doc."""

        self.imp.rename_template()

    @PyQt5.QtCore.pyqtSlot()
    def on_convertToMatlab_released(self) -> None:
        """Doc."""

        self.imp.convert_files_to_matlab_format()

    @PyQt5.QtCore.pyqtSlot()
    def on_solAnalysisRecalMeanAcf_released(self) -> None:
        """Doc"""

        self.imp.calculate_and_show_sol_mean_acf()

    @PyQt5.QtCore.pyqtSlot(int)
    def on_scanImgFileNum_valueChanged(self, val: int) -> None:
        """Doc."""

        self.imp.display_scan_image(val)

    @PyQt5.QtCore.pyqtSlot()
    def on_removeImportedSolData_released(self) -> None:
        """Doc."""

        self.imp.remove_imported_template()

    @PyQt5.QtCore.pyqtSlot()
    def on_dataDirLogUpdate_released(self) -> None:
        """Doc."""

        self.imp.update_dir_log_file()

    @PyQt5.QtCore.pyqtSlot()
    def on_importSolData_released(self) -> None:
        """Doc."""

        self.imp.import_sol_data()

    @PyQt5.QtCore.pyqtSlot()
    def on_openDir_released(self) -> None:
        """Doc."""

        self.imp.open_data_dir()

    @PyQt5.QtCore.pyqtSlot(str)
    def on_dataYear_currentTextChanged(self, year: str) -> None:
        """Doc."""

        self.imp.populate_data_dates_from_year(year)

    @PyQt5.QtCore.pyqtSlot(str)
    def on_dataMonth_currentTextChanged(self, month: str) -> None:
        """Doc."""

        self.imp.populate_data_dates_from_month(month)

    @PyQt5.QtCore.pyqtSlot(str)
    def on_dataDay_currentTextChanged(self, day: str) -> None:
        """Doc."""

        self.imp.populate_data_templates_from_day(day)

    @PyQt5.QtCore.pyqtSlot(str)
    def on_dataTemplate_currentTextChanged(self, template: str) -> None:
        """Doc."""

        self.imp.show_num_files(template)
        self.imp.update_dir_log_wdgt(template)
        self.imp.preview_img_scan(template)

    @PyQt5.QtCore.pyqtSlot(str)
    def on_importedSolDataTemplates_currentTextChanged(self, template: str) -> None:
        """Doc."""

        self.imp.populate_sol_meas_analysis(template)

    def closeEvent(self, event: PyQt5.QtCore.QEvent) -> None:
        """Doc."""

        self.imp.close(event)

    @PyQt5.QtCore.pyqtSlot()
    def on_actionRestart_triggered(self) -> None:
        """Doc."""

        self.imp.restart()

    @PyQt5.QtCore.pyqtSlot()
    def on_actionLoadLoadout_triggered(self) -> None:
        """Doc."""

        self.imp.load()

    @PyQt5.QtCore.pyqtSlot()
    def on_actionSaveLoadout_triggered(self) -> None:
        """Doc."""

        self.imp.save()

    @PyQt5.QtCore.pyqtSlot(int)
    def on_minNumLines_valueChanged(self, val: int) -> None:
        """Allow only even values"""

        if val % 2:
            self.minNumLines.setValue(val - 1)

    @PyQt5.QtCore.pyqtSlot()
    def on_roiImgScn_released(self) -> None:
        """Doc."""

        self.imp.roi_to_scan()

    @PyQt5.QtCore.pyqtSlot(int)
    def on_scaleImgScan_valueChanged(self, clip_hist_percent) -> None:
        """Doc."""

        self.imp.auto_scale_image(clip_hist_percent)

    @PyQt5.QtCore.pyqtSlot(int)
    def on_imgShowMethod_currentIndexChanged(self) -> None:
        """Doc."""

        plane_idx = self.numPlaneShownChoice.value()
        self.imp.disp_plane_img(plane_idx)

    @PyQt5.QtCore.pyqtSlot(int)
    def on_numPlaneShownChoice_sliderMoved(self, val: int) -> None:
        """Doc."""

        self.imp.plane_choice_changed(val)

    @PyQt5.QtCore.pyqtSlot(float)
    def on_solScanDur_valueChanged(self, float) -> None:
        """Doc."""

        self.imp.change_meas_duration(float)

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

        self.imp.dep_sett_apply()

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

        self.imp.disp_scn_pttrn(txt)

    @PyQt5.QtCore.pyqtSlot(str)
    def on_imgScanPreset_currentTextChanged(self, curr_txt: str) -> None:
        """Doc."""

        self.imp.fill_img_scan_preset_gui(curr_txt)

    @PyQt5.QtCore.pyqtSlot(str)
    def on_solMeasPreset_currentTextChanged(self, curr_txt: str) -> None:
        """Doc."""

        self.imp.fill_sol_meas_preset_gui(curr_txt)

    @PyQt5.QtCore.pyqtSlot()
    def on_actionSettings_triggered(self) -> None:
        """Show settings window"""

        self.imp.open_settwin()

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

        self._loop.create_task(self.imp.open_camwin())

    @PyQt5.QtCore.pyqtSlot(str)
    def on_avgInterval_currentTextChanged(self, val: str) -> None:
        """Doc."""

        self.imp.counts_avg_interval_changed(int(val))

    # -----------------------------------------------------------------------
    # Position Control
    # -----------------------------------------------------------------------

    @PyQt5.QtCore.pyqtSlot()
    def on_goTo_released(self) -> None:
        """Doc."""

        self.imp.move_scanners()

    # -----------------------------------------------------------------------
    # Stepper Stage Dock
    # -----------------------------------------------------------------------

    @PyQt5.QtCore.pyqtSlot()
    def on_stageOn_released(self) -> None:
        """Doc."""

        self.imp.dvc_toggle("stage")


class SettWin(PyQt5.QtWidgets.QDialog):
    """ Documentation."""

    UI_PATH = "./gui/settingswindow.ui"

    def __init__(self, app, parent=None) -> None:
        """Doc."""

        super(SettWin, self).__init__(parent)
        PyQt5.uic.loadUi(self.UI_PATH, self)
        self.imp = logic.windows.SettWin(self, app)

    def closeEvent(self, event: PyQt5.QtCore.QEvent) -> None:
        """Doc."""

        self.imp.clean_up()
        self.imp.check_on_close = True

    @PyQt5.QtCore.pyqtSlot()
    def on_saveButton_released(self) -> None:
        """Save settings."""

        self.imp.save()

    @PyQt5.QtCore.pyqtSlot()
    def on_loadButton_released(self) -> None:
        """load settings."""

        self.imp.load()

    @PyQt5.QtCore.pyqtSlot()
    def on_confirmButton_released(self) -> None:
        """Doc."""

        self.imp.confirm()
        self.close()


class CamWin(PyQt5.QtWidgets.QWidget):
    """Doc."""

    UI_PATH = "./gui/camerawindow.ui"

    def __init__(self, app, parent=None) -> None:
        """Doc."""

        super(CamWin, self).__init__(parent, PyQt5.QtCore.Qt.WindowStaysOnTopHint)
        PyQt5.uic.loadUi(self.UI_PATH, self)
        self.move(30, 180)
        self.imp = logic.windows.CamWin(self, app)
        self._loop = app.loop

        # add matplotlib-ready widget (canvas) for showing camera output
        self.ImgDisp = Display(self.imageDisplayLayout, self)

    def closeEvent(self, event: PyQt5.QtCore.QEvent) -> None:
        """Doc."""

        self._loop.create_task(self.imp.clean_up())

    @PyQt5.QtCore.pyqtSlot()
    def on_shootButton_released(self) -> None:
        """Doc."""

        self.imp.shoot()

    @PyQt5.QtCore.pyqtSlot()
    def on_videoButton_released(self) -> None:
        """Doc."""

        self._loop.create_task(self.imp.toggle_video())
