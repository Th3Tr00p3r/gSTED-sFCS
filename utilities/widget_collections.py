""" Widget Collections """

from utilities.helper import QtWidgetCollection

# ------------------------------
# Devices
# ------------------------------

led_wdgts = QtWidgetCollection(
    exc=("ledExc", "QIcon", "main", True),
    dep=("ledDep", "QIcon", "main", True),
    shutter=("ledShutter", "QIcon", "main", True),
    stage=("ledStage", "QIcon", "main", True),
    tdc=("ledTdc", "QIcon", "main", True),
    cam=("ledCam", "QIcon", "main", True),
    scnrs=("ledScn", "QIcon", "main", True),
    cntr=("ledCounter", "QIcon", "main", True),
    um232=("ledUm232h", "QIcon", "main", True),
    pxl_clk=("ledPxlClk", "QIcon", "main", True),
)

switch_wdgts = QtWidgetCollection(
    exc=("excOnButton", "QIcon", "main", True),
    dep=("depEmissionOn", "QIcon", "main", True),
    shutter=("depShutterOn", "QIcon", "main", True),
    stage=("stageOn", "QIcon", "main", True),
)

# ----------------------------------------------
# Analysis Widget Collections
# ----------------------------------------------

data_import_wdgts = QtWidgetCollection(
    is_image_type=("imageDataImport", "QRadioButton", "main", True),
    is_solution_type=("solDataImport", "QRadioButton", "main", True),
    data_days=("dataDay", "QComboBox", "main", True),
    data_months=("dataMonth", "QComboBox", "main", True),
    data_years=("dataYear", "QComboBox", "main", True),
    data_templates=("dataTemplate", "QComboBox", "main", True),
    import_stacked=("importStacked", "QStackedWidget", "main", True),
    is_calibration=("isDataCalibration", "QCheckBox", "main", True),
    log_text=("dataDirLog", "QPlainTextEdit", "main", True),
)

sol_data_analysis_wdgts = QtWidgetCollection(
    fix_shift=("solDataFixShift", "QCheckBox", "main", True),
    scan_image_disp=("solScanImgDisp", None, "main", True),
    row_dicrimination=("rowDiscriminationGroup", "QButtonGroup", "main", True),
    remove_over=("solAnalysisRemoveOverSpinner", "QDoubleSpinBox", "main", True),
    remove_worst=("solAnalysisRemoveWorstSpinner", "QDoubleSpinBox", "main", True),
    is_show_only_mean=("showOnlyMeanAcf", "QCheckBox", "main", True),
    row_acf_disp=("solScanAcfDisp", None, "main", True),
    imported_templates=("importedSolDataTemplates", "QComboBox", "main", True),
    scan_img_file_num=("scanImgFileNum", "QSpinBox", "main", True),
    scan_settings=("solAnalysisScanSettings", "QPlainTextEdit", "main", True),
    scan_duration_min=("solAnalysisDur", "QDoubleSpinBox", "main", True),
    n_files=("solAnalysisNumFiles", "QSpinBox", "main", True),
)
# ----------------------------------------------
# Measurement Widget Collections
# ----------------------------------------------

sol_ang_scan_wdgts = QtWidgetCollection(
    max_line_len_um=("maxLineLen", "QDoubleSpinBox", "main", False),
    ao_samp_freq_Hz=("angAoSampFreq", "QDoubleSpinBox", "main", False),
    angle_deg=("angle", "QSpinBox", "main", False),
    lin_frac=("solLinFrac", "QDoubleSpinBox", "main", False),
    line_shift_um=("lineShift", "QSpinBox", "main", False),
    speed_um_s=("solAngScanSpeed", "QSpinBox", "main", False),
    min_lines=("minNumLines", "QSpinBox", "main", False),
    max_scan_freq_Hz=("maxScanFreq", "QSpinBox", "main", False),
)

sol_circ_scan_wdgts = QtWidgetCollection(
    ao_samp_freq_Hz=("circAoSampFreq", "QSpinBox", "main", False),
    diameter_um=("circDiameter", "QDoubleSpinBox", "main", False),
    speed_um_s=("circSpeed", "QSpinBox", "main", False),
)

sol_meas_wdgts = QtWidgetCollection(
    scan_type=("solScanType", "QComboBox", "main", False),
    file_template=("solScanFileTemplate", "QLineEdit", "main", False),
    repeat=("repeatSolMeas", "QCheckBox", "main", False),
    save_path=("dataPath", "QLineEdit", "settings", False),
    sub_dir_name=("solSubdirName", "QLineEdit", "settings", False),
    max_file_size_mb=("solScanMaxFileSize", "QDoubleSpinBox", "main", False),
    duration=("solScanDur", "QDoubleSpinBox", "main", False),
    duration_units=("solScanDurUnits", "QComboBox", "main", False),
    prog_bar_wdgt=("solScanProgressBar", "QSlider", "main", True),
    start_time_wdgt=("solScanStartTime", "QTimeEdit", "main", True),
    end_time_wdgt=("solScanEndTime", "QTimeEdit", "main", True),
    file_num_wdgt=("solScanFileNo", "QSpinBox", "main", True),
    pattern_wdgt=("solScanPattern", None, "main", True),
    g0_wdgt=("g0", "QDoubleSpinBox", "main", True),
    tau_wdgt=("decayTime", "QDoubleSpinBox", "main", True),
    plot_wdgt=("acf", None, "main", True),
    fit_led=("ledFit", "QIcon", "main", True),
)

img_scan_wdgts = QtWidgetCollection(
    scan_plane=("imgScanType", "QComboBox", "main", False),
    dim1_um=("imgScanDim1", "QDoubleSpinBox", "main", False),
    dim2_um=("imgScanDim2", "QDoubleSpinBox", "main", False),
    dim3_um=("imgScanDim3", "QDoubleSpinBox", "main", False),
    n_lines=("imgScanNumLines", "QSpinBox", "main", False),
    ppl=("imgScanPPL", "QSpinBox", "main", False),
    line_freq_Hz=("imgScanLineFreq", "QSpinBox", "main", False),
    lin_frac=("imgScanLinFrac", "QDoubleSpinBox", "main", False),
    n_planes=("imgScanNumPlanes", "QSpinBox", "main", False),
    curr_ao_x=("xAOVint", "QDoubleSpinBox", "main", False),
    curr_ao_y=("yAOVint", "QDoubleSpinBox", "main", False),
    curr_ao_z=("zAOVint", "QDoubleSpinBox", "main", False),
    auto_cross=("autoCrosshair", "QCheckBox", "main", False),
)

img_meas_wdgts = QtWidgetCollection(
    file_template=("imgScanFileTemplate", "QLineEdit", "main", False),
    save_path=("dataPath", "QLineEdit", "settings", False),
    sub_dir_name=("imgSubdirName", "QLineEdit", "settings", False),
    prog_bar_wdgt=("imgScanProgressBar", "QSlider", "main", True),
    curr_plane_wdgt=("currPlane", "QSpinBox", "main", True),
    plane_shown=("numPlaneShown", "QSpinBox", "main", True),
    plane_choice=("numPlaneShownChoice", "QSlider", "main", True),
    image_wdgt=("imgScanPlot", None, "main", True),
    pattern_wdgt=("imgScanPattern", None, "main", True),
)
