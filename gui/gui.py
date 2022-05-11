""" GUI - signals and slots"""

import PyQt5.uic
from PyQt5 import QtCore, QtWidgets

# import gui.icons  # for initial icons loadout # NOQA
import logic.slot_implementations as impl
from utilities.display import GuiDisplay

try:
    from gui.icons import icons_rc  # for initial icons loadout # NOQA
except ImportError:
    print("icons_rc.py was not found - icons will not initialize.", end=" ")


class MainWin(QtWidgets.QMainWindow):
    """Doc."""

    UI_PATH = "./gui/mainwindow.ui"

    def __init__(self, app) -> None:

        super(MainWin, self).__init__()
        PyQt5.uic.loadUi(self.UI_PATH, self)
        self.impl = impl.MainWin(self, app)
        self._loop = app.loop

        # graphics
        self.imgScanPlot = GuiDisplay(self.imageLayout, self)
        self.solScanAcf = GuiDisplay(self.solScanAcfLayout, self)
        self.imgScanPattern = GuiDisplay(self.imgScanPatternLayout)
        self.solScanPattern = GuiDisplay(self.solScanPatternLayout)
        self.solScanAnalysisPattern = GuiDisplay(self.solAnalysisScanTestLayout)
        self.solExpLoading = GuiDisplay(self.solExpLoadingLayout, self)
        self.solExpTDC1 = GuiDisplay(self.solAnalysisTDCLayout1, self)
        self.solExpTDC2 = GuiDisplay(self.solAnalysisTDCLayout2, self)
        self.solExpGSTED = GuiDisplay(self.solExpGSTEDLayout, self)

        # scan patterns
        # image
        self.imgScanDim1.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("image"))
        self.imgScanDim2.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("image"))
        self.imgScanNumLines.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("image"))
        self.imgScanPPL.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("image"))
        self.imgScanLinFrac.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("image"))
        self.imgScanType.currentTextChanged.connect(lambda: self.impl.disp_scn_pttrn("image"))
        # solution (angular) # TODO - place in button group?
        self.maxLineLen.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("angular"))
        self.angle.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("angular"))
        self.solLinFrac.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("angular"))
        self.lineShift.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("angular"))
        self.minNumLines.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("angular"))
        self.solAngScanSpeed.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("angular"))
        # solution (circle)
        self.circDiameter.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("circle"))
        self.circAoSampFreq.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("circle"))
        self.circSpeed.valueChanged.connect(lambda: self.impl.disp_scn_pttrn("circle"))

        # Positioning/Scanners
        self.axisMoveUp.released.connect(lambda: self.impl.displace_scanner_axis(1))
        self.axisMoveDown.released.connect(lambda: self.impl.displace_scanner_axis(-1))
        self.goToOrg.released.connect(lambda: self.impl.go_to_origin("XYZ"))
        self.goToOrgXY.released.connect(lambda: self.impl.go_to_origin("XY"))
        self.goToOrgZ.released.connect(lambda: self.impl.go_to_origin("Z"))

        # Image Scan Tab
        # TODO: add a label with the image name which would be displayed above the image
        self.nextImg.released.connect(lambda: self.impl.cycle_through_image_scans("next"))
        self.prevImg.released.connect(lambda: self.impl.cycle_through_image_scans("prev"))

        self.saveImg.released.connect(self.impl.save_current_image)

        # Analysis - Data and single measurements
        self.analysisDataTypeGroup = QtWidgets.QButtonGroup()
        self.analysisDataTypeGroup.addButton(self.imageDataImport)
        self.analysisDataTypeGroup.addButton(self.solDataImport)
        self.analysisDataTypeGroup.buttonReleased.connect(self.impl.switch_data_type)
        self.solImportSaveProcessed.released.connect(self.impl.save_processed_data)
        self.solImportDeleteProcessed.released.connect(self.impl.delete_all_processed_data)
        self.solImportLoadProcessed.released.connect(
            lambda: self.impl.import_sol_data(should_load_processed=True)
        )

        self.rowDiscriminationGroup = QtWidgets.QButtonGroup()
        self.rowDiscriminationGroup.addButton(self.solAnalysisRemoveOver)
        self.rowDiscriminationGroup.addButton(self.solAnalysisRemoveWorst)

        self.fileSelectionGroup = QtWidgets.QButtonGroup()
        self.fileSelectionGroup.addButton(self.solImportUseAll)
        self.fileSelectionGroup.addButton(self.solImportUse)

        self.solScanImgDisp = GuiDisplay(self.solAnalysisScanImageLayout, self)
        self.solScanAcfDisp = GuiDisplay(self.solAnalysisAveragingLayout, self)

        self.imgScanPreviewDisp = GuiDisplay(self.importImgPreviewLayout)

        self.nextTemplate.released.connect(lambda: self.impl.cycle_through_data_templates("next"))
        self.prevTemplate.released.connect(lambda: self.impl.cycle_through_data_templates("prev"))

        self.solAveragingPlotSpatial.released.connect(self.impl.calculate_and_show_sol_mean_acf)
        self.solAveragingPlotTemporal.released.connect(self.impl.calculate_and_show_sol_mean_acf)

        # analysis - experiment
        self.assignExpConfMeas.released.connect(lambda: self.impl.assign_measurement("confocal"))
        self.assignExpStedMeas.released.connect(lambda: self.impl.assign_measurement("sted"))
        self.loadExperiment.released.connect(self.impl.load_experiment)
        self.calibrateTdc.released.connect(self.impl.calibrate_tdc)
        self.AddCustomGate.released.connect(self.impl.assign_gate)
        self.removeAssignedGate.released.connect(self.impl.remove_assigned_gate)
        self.addGates.released.connect(self.impl.gate)
        self.removeAvailableGate.released.connect(self.impl.remove_available_gate)
        self.calculateStructureFactors.released.connect(self.impl.calculate_structure_factors)

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
        self.ledPxlClk.clicked.connect(lambda: led_clicked(self))
        self.ledScn.clicked.connect(lambda: led_clicked(self))
        self.ledPSD.clicked.connect(lambda: led_clicked(self))
        self.ledSPAD.clicked.connect(lambda: led_clicked(self))
        self.ledCam1.clicked.connect(lambda: led_clicked(self))
        self.ledCam2.clicked.connect(lambda: led_clicked(self))

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

        # Delayer
        self.psdDelay.valueChanged.connect(self.impl.set_detector_gate)

        # SPAD
        self.spadGateWidth.valueChanged.connect(self.impl.set_spad_gatewidth)

        # Device Toggling
        self.excOnButton.released.connect(lambda: self.impl.device_toggle("exc_laser"))
        self.depEmissionOn.released.connect(
            lambda: self.impl.device_toggle("dep_laser", "laser_toggle", "is_emission_on")
        )
        self.depShutterOn.released.connect(lambda: self.impl.device_toggle("dep_shutter"))
        self.stageOn.released.connect(lambda: self.impl.device_toggle("stage"))
        self.psdSwitch.released.connect(lambda: self.impl.device_toggle("delayer"))
        self.openSpadInterface.released.connect(self.impl.open_spad_interface)

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
        self.setStatusBar(QtWidgets.QStatusBar())

        # Camera Dock
        self.CAMERA_DOCK_MIN_SIZE = self.cameraDock.minimumSize()
        self.ImgDisp1 = GuiDisplay(self.imageDisplayLayout1, self)
        self.ImgDisp2 = GuiDisplay(self.imageDisplayLayout2, self)

        self.shootButton1.released.connect(lambda: self.impl.display_image(1))
        self.shootButton2.released.connect(lambda: self.impl.display_image(2))
        self.videoSwitch1.released.connect(
            lambda: self.impl.device_toggle("camera_1", "toggle_video", "is_in_video_mode")
        )
        self.videoSwitch2.released.connect(
            lambda: self.impl.device_toggle("camera_2", "toggle_video", "is_in_video_mode")
        )
        self.autoExp1.toggled.connect(
            lambda: self.impl.set_auto_exposure(1, self.autoExp1.isChecked())
        )
        self.autoExp2.toggled.connect(
            lambda: self.impl.set_auto_exposure(2, self.autoExp2.isChecked())
        )

        self.pixel_clock1.valueChanged.connect(
            lambda: self.impl.set_parameter(1, "pixel_clock", self.pixel_clock1.value())
        )
        self.pixel_clock2.valueChanged.connect(
            lambda: self.impl.set_parameter(2, "pixel_clock", self.pixel_clock2.value())
        )
        self.framerate1.valueChanged.connect(
            lambda: self.impl.set_parameter(1, "framerate", self.framerate1.value())
        )
        self.framerate2.valueChanged.connect(
            lambda: self.impl.set_parameter(2, "framerate", self.framerate2.value())
        )
        self.exposure1.valueChanged.connect(
            lambda: self.impl.set_parameter(1, "exposure", self.exposure1.value())
        )
        self.exposure2.valueChanged.connect(
            lambda: self.impl.set_parameter(2, "exposure", self.exposure2.value())
        )

        # intialize gui
        self.actionCamera_Control.setChecked(False)
        self.cameraDock.setVisible(False)
        self.stageButtonsGroup.setEnabled(False)

        self.move(300, 30)
        self.setFixedSize(1251, 950)
        self.setMaximumSize(int(1e5), int(1e5))

    @QtCore.pyqtSlot(bool)
    def on_cameraDock_topLevelChanged(self, is_floating: bool) -> None:
        """Doc."""

        if is_floating:
            self.cameraDock.setMaximumSize(int(1e5), int(1e5))
        else:
            self.cameraDock.setFixedSize(self.CAMERA_DOCK_MIN_SIZE)

    @QtCore.pyqtSlot()
    def on_renameTemplate_released(self) -> None:
        """Doc."""

        self.impl.rename_template()

    @QtCore.pyqtSlot()
    def on_convertToMatlab_released(self) -> None:
        """Doc."""

        self.impl.convert_files_to_matlab_format()

    @QtCore.pyqtSlot()
    def on_solAnalysisRecalMeanAcf_released(self) -> None:
        """Doc"""

        self.impl.calculate_and_show_sol_mean_acf()

    @QtCore.pyqtSlot(int)
    def on_scanImgFileNum_valueChanged(self, val: int) -> None:
        """Doc."""

        self.impl.display_angular_scan_image(val)

    @QtCore.pyqtSlot()
    def on_removeImportedSolData_released(self) -> None:
        """Doc."""

        self.impl.remove_imported_template()

    def on_removeLoadedExperiment_released(self) -> None:
        """Doc."""

        self.impl.remove_experiment()

    @QtCore.pyqtSlot()
    def on_dataDirLogUpdate_released(self) -> None:
        """Doc."""

        self.impl.update_dir_log_file()

    @QtCore.pyqtSlot()
    def on_importSolData_released(self) -> None:
        """Doc."""

        self.impl.import_sol_data()

    @QtCore.pyqtSlot()
    def on_openDir_released(self) -> None:
        """Doc."""

        self.impl.open_data_dir()

    @QtCore.pyqtSlot(str)
    def on_dataYear_currentTextChanged(self, year: str) -> None:
        """Doc."""

        self.impl.populate_data_dates_from_year(year)

    @QtCore.pyqtSlot(str)
    def on_dataMonth_currentTextChanged(self, month: str) -> None:
        """Doc."""

        self.impl.populate_data_dates_from_month(month)

    @QtCore.pyqtSlot(str)
    def on_dataDay_currentTextChanged(self, day: str) -> None:
        """Doc."""

        self.impl.populate_data_templates_from_day(day)

    @QtCore.pyqtSlot(str)
    def on_dataTemplate1_currentTextChanged(self, template: str) -> None:
        """Doc."""

        self.impl.show_num_files(template)
        self.impl.update_dir_log_wdgt(template)
        self.impl.preview_img_scan(template)
        self.impl.toggle_save_processed_enabled()
        self.impl.toggle_load_processed_enabled(template)

    @QtCore.pyqtSlot(str)
    def on_importedSolDataTemplates1_currentTextChanged(self, template: str) -> None:
        """Doc."""

        self.impl.populate_sol_meas_analysis(template)

    def closeEvent(self, event: QtCore.QEvent) -> None:
        """Doc."""

        self.impl.close(event)

    @QtCore.pyqtSlot()
    def on_actionRestart_triggered(self) -> None:
        """Doc."""

        self.impl.restart()

    @QtCore.pyqtSlot()
    def on_actionLoadLoadout_triggered(self) -> None:
        """Doc."""

        self.impl.load()

    @QtCore.pyqtSlot()
    def on_actionSaveLoadout_triggered(self) -> None:
        """Doc."""

        self.impl.save()

    @QtCore.pyqtSlot(int)
    def on_minNumLines_valueChanged(self, val: int) -> None:
        """Allow only even values"""

        if val % 2:
            self.minNumLines.setValue(val - 1)

    @QtCore.pyqtSlot()
    def on_roiImgScn_released(self) -> None:
        """Doc."""

        self.impl.roi_to_scan()

    @QtCore.pyqtSlot(int)
    def on_scaleImgScan_valueChanged(self, percent_factor: int) -> None:
        """Doc."""

        self.impl.auto_scale_image(percent_factor)

    @QtCore.pyqtSlot(int)
    def on_imgShowMethod_currentIndexChanged(self) -> None:
        """Doc."""

        plane_idx = self.numPlaneShownChoice.value()
        self.impl.disp_plane_img(plane_idx)

    @QtCore.pyqtSlot(int)
    def on_numPlaneShownChoice_sliderMoved(self, val: int) -> None:
        """Doc."""

        self.impl.plane_choice_changed(val)

    @QtCore.pyqtSlot(float)
    def on_solScanDur_valueChanged(self, float) -> None:
        """Doc."""

        self.impl.change_meas_duration(float)

    @QtCore.pyqtSlot()
    def on_powMode_released(self) -> None:
        """Switch between power/current depletion laser settings"""

        self.depModeStacked.setCurrentIndex(1)

    @QtCore.pyqtSlot()
    def on_currMode_released(self) -> None:
        """Switch between power/current depletion laser settings"""

        self.depModeStacked.setCurrentIndex(0)

    @QtCore.pyqtSlot()
    def on_depApplySettings_released(self) -> None:
        """Apply current/power mode and value"""

        self.impl.dep_sett_apply()

    @QtCore.pyqtSlot(int)
    def on_solScanType_currentIndexChanged(self, index: int) -> None:
        """
        Change stacked widget 'solScanParamsStacked' index
        according to index of the combo box 'solScanTypeCombox'.
        """

        self.solScanParamsStacked.setCurrentIndex(index)

    @QtCore.pyqtSlot(str)
    def on_solScanType_currentTextChanged(self, txt: str) -> None:
        """Doc."""

        self.impl.disp_scn_pttrn(txt)

    @QtCore.pyqtSlot(str)
    def on_imgScanPreset_currentTextChanged(self, curr_txt: str) -> None:
        """Doc."""

        self.impl.fill_img_scan_preset_gui(curr_txt)

    @QtCore.pyqtSlot(str)
    def on_solMeasPreset_currentTextChanged(self, curr_txt: str) -> None:
        """Doc."""

        self.impl.fill_sol_meas_preset_gui(curr_txt)

    @QtCore.pyqtSlot()
    def on_actionSettings_triggered(self) -> None:
        """Show settings window"""

        self.impl.open_settwin()

    @QtCore.pyqtSlot(bool)
    def on_actionLaser_Control_toggled(self, is_toggled_on: bool) -> None:
        """Show/hide stepper laser control dock"""

        self.laserDock.setVisible(is_toggled_on)

    @QtCore.pyqtSlot(bool)
    def on_actionStepper_Stage_Control_toggled(self, is_toggled_on: bool) -> None:
        """Show/hide stepper stage control dock (switches with delayer, no more room)"""

        self.stepperDock.setVisible(is_toggled_on)
        if is_toggled_on:
            self.delayerDock.setVisible(False)
            self.actionDelayer_Control.setChecked(False)

    @QtCore.pyqtSlot(bool)
    def on_actionDelayer_Control_toggled(self, is_toggled_on: bool) -> None:
        """Show/hide delayer control dock (switches with stage, no more room)"""

        self.delayerDock.setVisible(is_toggled_on)
        if is_toggled_on:
            self.stepperDock.setVisible(False)
            self.actionStepper_Stage_Control.setChecked(False)

    @QtCore.pyqtSlot(bool)
    def on_actionCamera_Control_toggled(self, is_toggled_on: bool) -> None:
        """Show/hide camera control dock"""

        self.impl.toggle_camera_dock(is_toggled_on)

    @QtCore.pyqtSlot(str)
    def on_avgInterval_currentTextChanged(self, val: str) -> None:
        """Doc."""

        self.impl.counts_avg_interval_changed(int(val))

    # -----------------------------------------------------------------------
    # Position Control
    # -----------------------------------------------------------------------

    @QtCore.pyqtSlot()
    def on_goTo_released(self) -> None:
        """Doc."""

        self.impl.move_scanners()


class SettWin(QtWidgets.QDialog):
    """Documentation."""

    UI_PATH = "./gui/settingswindow.ui"

    def __init__(self, app) -> None:
        """Doc."""

        super(SettWin, self).__init__()
        PyQt5.uic.loadUi(self.UI_PATH, self)
        self.impl = impl.SettWin(self, app)

        self.getDeviceDetails.released.connect(self.impl.get_all_device_details)

    def closeEvent(self, event: QtCore.QEvent) -> None:
        """Doc."""

        self.impl.clean_up()
        self.impl.check_on_close = True

    @QtCore.pyqtSlot()
    def on_saveButton_released(self) -> None:
        """Save settings."""

        self.impl.save()

    @QtCore.pyqtSlot()
    def on_loadButton_released(self) -> None:
        """load settings."""

        self.impl.load()

    @QtCore.pyqtSlot()
    def on_confirmButton_released(self) -> None:
        """Doc."""

        self.impl.check_on_close = False
        self.close()
