""" Widget Collections """

from utilities.helper import QtWidgetCollection

# ------------------------------
# Devices
# ------------------------------

led_wdgts = QtWidgetCollection(
    exc=("ledExc", "icon", "main", True),
    dep=("ledDep", "icon", "main", True),
    shutter=("ledShutter", "icon", "main", True),
    stage=("ledStage", "icon", "main", True),
    tdc=("ledTdc", "icon", "main", True),
    cam=("ledCam", "icon", "main", True),
    scnrs=("ledScn", "icon", "main", True),
    cntr=("ledCounter", "icon", "main", True),
    um232=("ledUm232h", "icon", "main", True),
    pxl_clk=("ledPxlClk", "icon", "main", True),
)

switch_wdgts = QtWidgetCollection(
    exc=("excOnButton", "icon", "main", True),
    dep=("depEmissionOn", "icon", "main", True),
    shutter=("depShutterOn", "icon", "main", True),
    stage=("stageOn", "icon", "main", True),
)

# ----------------------------------------------
# Analysis Widget Collections
# ----------------------------------------------

data_import_wdgts = QtWidgetCollection(
    is_image_type=("imageDataImport", "isChecked", "main", True),
    is_solution_type=("solDataImport", "isChecked", "main", True),
    data_days=("dataDay", "currentText", "main", True),
    data_months=("dataMonth", "currentText", "main", True),
    data_years=("dataYear", "currentText", "main", True),
    data_templates=("dataTemplate", "currentText", "main", True),
    import_stacked=("importStacked", "currentIndex", "main", True),
    is_calibration=("isDataCalibration", "isChecked", "main", True),
    log_text=("dataDirLog", "toPlainText", "main", True),
)

sol_data_analysis_wdgts = QtWidgetCollection(
    fix_shift=("solDataFixShift", "isChecked", "main", True),
    scan_image_disp=("solScanImgDisp", None, "main", True),
    imported_templates=("importedSolDataTemplates", "currentText", "main", True),
    scan_img_file_num=("scanImgFileNum", "value", "main", True),
    scan_settings=("solAnalysisScanSettings", "toPlainText", "main", True),
    scan_duration_min=("solAnalysisDur", "value", "main", True),
    n_files=("solAnalysisNumFiles", "value", "main", True),
)
# ----------------------------------------------
# Measurement Widget Collections
# ----------------------------------------------

sol_ang_scan_wdgts = QtWidgetCollection(
    max_line_len_um=("maxLineLen", "value", "main", False),
    ao_samp_freq_Hz=("angAoSampFreq", "value", "main", False),
    angle_deg=("angle", "value", "main", False),
    lin_frac=("solLinFrac", "value", "main", False),
    line_shift_um=("lineShift", "value", "main", False),
    speed_um_s=("solAngScanSpeed", "value", "main", False),
    min_lines=("minNumLines", "value", "main", False),
    max_scan_freq_Hz=("maxScanFreq", "value", "main", False),
)

sol_circ_scan_wdgts = QtWidgetCollection(
    ao_samp_freq_Hz=("circAoSampFreq", "value", "main", False),
    diameter_um=("circDiameter", "value", "main", False),
    speed_um_s=("circSpeed", "value", "main", False),
)

sol_meas_wdgts = QtWidgetCollection(
    scan_type=("solScanType", "currentText", "main", False),
    file_template=("solScanFileTemplate", "text", "main", False),
    repeat=("repeatSolMeas", "isChecked", "main", False),
    save_path=("dataPath", "text", "settings", False),
    sub_dir_name=("solSubdirName", "text", "settings", False),
    save_frmt=("saveFormat", "currentText", "settings", False),
    max_file_size_mb=("solScanMaxFileSize", "value", "main", False),
    duration=("solScanDur", "value", "main", False),
    duration_units=("solScanDurUnits", "currentText", "main", False),
    prog_bar_wdgt=("solScanProgressBar", "value", "main", True),
    start_time_wdgt=("solScanStartTime", "time", "main", True),
    end_time_wdgt=("solScanEndTime", "time", "main", True),
    file_num_wdgt=("solScanFileNo", "value", "main", True),
    pattern_wdgt=("solScanPattern", None, "main", True),
    g0_wdgt=("g0", "value", "main", True),
    tau_wdgt=("decayTime", "value", "main", True),
    plot_wdgt=("acf", None, "main", True),
    fit_led=("ledFit", "icon", "main", True),
)

img_scan_wdgts = QtWidgetCollection(
    scan_plane=("imgScanType", "currentText", "main", False),
    dim1_um=("imgScanDim1", "value", "main", False),
    dim2_um=("imgScanDim2", "value", "main", False),
    dim3_um=("imgScanDim3", "value", "main", False),
    n_lines=("imgScanNumLines", "value", "main", False),
    ppl=("imgScanPPL", "value", "main", False),
    line_freq_Hz=("imgScanLineFreq", "value", "main", False),
    lin_frac=("imgScanLinFrac", "value", "main", False),
    n_planes=("imgScanNumPlanes", "value", "main", False),
    curr_ao_x=("xAOVint", "value", "main", False),
    curr_ao_y=("yAOVint", "value", "main", False),
    curr_ao_z=("zAOVint", "value", "main", False),
    auto_cross=("autoCrosshair", "isChecked", "main", False),
)

img_meas_wdgts = QtWidgetCollection(
    file_template=("imgScanFileTemplate", "text", "main", False),
    save_path=("dataPath", "text", "settings", False),
    sub_dir_name=("imgSubdirName", "text", "settings", False),
    save_frmt=("saveFormat", "currentText", "settings", False),
    prog_bar_wdgt=("imgScanProgressBar", "value", "main", True),
    curr_plane_wdgt=("currPlane", "value", "main", True),
    plane_shown=("numPlaneShown", "value", "main", True),
    plane_choice=("numPlaneShownChoice", "value", "main", True),
    image_wdgt=("imgScanPlot", None, "main", True),
    pattern_wdgt=("imgScanPattern", None, "main", True),
)
