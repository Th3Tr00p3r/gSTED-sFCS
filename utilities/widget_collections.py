""" Widget Collections """

from utilities.helper import QtWidgetCollection

# ------------------------------
# Devices
# ------------------------------

LED_WDGT_COLL = QtWidgetCollection(
    exc=("ledExc", "icon", "main"),
    dep=("ledDep", "icon", "main"),
    shutter=("ledShutter", "icon", "main"),
    stage=("ledStage", "icon", "main"),
    tdc=("ledTdc", "icon", "main"),
    cam=("ledCam", "icon", "main"),
    scnrs=("ledScn", "icon", "main"),
    cntr=("ledCounter", "icon", "main"),
    um232=("ledUm232h", "icon", "main"),
    pxl_clk=("ledPxlClk", "icon", "main"),
)

SWITCH_WDGT_COLL = QtWidgetCollection(
    exc=("excOnButton", "icon", "main"),
    dep=("depEmissionOn", "icon", "main"),
    shutter=("depShutterOn", "icon", "main"),
    stage=("stageOn", "icon", "main"),
)

# ----------------------------------------------
# Measurement Widget Collections
# ----------------------------------------------

SOL_ANG_SCN_WDGT_COLL = QtWidgetCollection(
    exc_mode=("solScanModeExc", "isChecked", "main"),
    dep_mode=("solScanModeDep", "isChecked", "main"),
    sted_mode=("solScanModeSted", "isChecked", "main"),
    max_line_len_um=("maxLineLen", "value", "main"),
    ao_samp_freq_Hz=("angAoSampFreq", "value", "main"),
    angle_deg=("angle", "value", "main"),
    lin_frac=("solLinFrac", "value", "main"),
    line_shift_um=("lineShift", "value", "main"),
    speed_um_s=("solAngScanSpeed", "value", "main"),
    min_lines=("minNumLines", "value", "main"),
    max_scan_freq_Hz=("maxScanFreq", "value", "main"),
)

SOL_CIRC_SCN_WDGT_COLL = QtWidgetCollection(
    exc_mode=("solScanModeExc", "isChecked", "main"),
    dep_mode=("solScanModeDep", "isChecked", "main"),
    sted_mode=("solScanModeSted", "isChecked", "main"),
    ao_samp_freq_Hz=("circAoSampFreq", "value", "main"),
    diameter_um=("circDiameter", "value", "main"),
    speed_um_s=("circSpeed", "value", "main"),
)

SOL_NO_SCN_WDGT_COLL = QtWidgetCollection(
    # TODO: move laser modes to SOL_MEAS_WDGT_COLL (also from 2 collections above) - or switch to buttons, same as image scan
    exc_mode=("solScanModeExc", "isChecked", "main"),
    dep_mode=("solScanModeDep", "isChecked", "main"),
    sted_mode=("solScanModeSted", "isChecked", "main"),
)

SOL_MEAS_WDGT_COLL = QtWidgetCollection(
    scan_type=("solScanType", "currentText", "main"),
    file_template=("solScanFileTemplate", "text", "main"),
    repeat=("repeatSolMeas", "isChecked", "main"),
    save_path=("solDataPath", "text", "settings"),
    save_frmt=("saveFormat", "currentText", "settings"),
    max_file_size=("solScanMaxFileSize", "value", "main"),
    cal_duration=("solScanCalDur", "value", "main"),
    total_duration=("solScanTotalDur", "value", "main"),
    duration_units=("solScanDurUnits", "currentText", "main"),
    cal_save_intrvl_wdgt=("solScanCalSaveIntrvl", "value", "main"),
    prog_bar_wdgt=("solScanProgressBar", "value", "main"),
    start_time_wdgt=("solScanStartTime", "time", "main"),
    end_time_wdgt=("solScanEndTime", "time", "main"),
    file_num_wdgt=("solScanFileNo", "value", "main"),
    total_files_wdgt=("solScanTotalFiles", "value", "main"),
    pattern_wdgt=("solScanPattern", None, "main"),
    g0_wdgt=("g0", "value", "main"),
    tau_wdgt=("decayTime", "value", "main"),
    plot_wdgt=("acf", None, "main"),
    fit_led=("ledFit", "icon", "main"),
)

IMG_SCN_WDGT_COLL = QtWidgetCollection(
    scan_plane=("imgScanType", "currentText", "main"),
    dim1_um=("imgScanDim1", "value", "main"),
    dim2_um=("imgScanDim2", "value", "main"),
    dim3_um=("imgScanDim3", "value", "main"),
    n_lines=("imgScanNumLines", "value", "main"),
    ppl=("imgScanPPL", "value", "main"),
    line_freq_Hz=("imgScanLineFreq", "value", "main"),
    lin_frac=("imgScanLinFrac", "value", "main"),
    n_planes=("imgScanNumPlanes", "value", "main"),
    curr_ao_x=("xAOVint", "value", "main"),
    curr_ao_y=("yAOVint", "value", "main"),
    curr_ao_z=("zAOVint", "value", "main"),
    auto_cross=("autoCrosshair", "isChecked", "main"),
)

IMG_MEAS_WDGT_COLL = QtWidgetCollection(
    file_template=("imgScanFileTemplate", "text", "main"),
    save_path=("imgDataPath", "text", "settings"),
    save_frmt=("saveFormat", "currentText", "settings"),
    prog_bar_wdgt=("imgScanProgressBar", "value", "main"),
    curr_plane_wdgt=("currPlane", "value", "main"),
    plane_shown=("numPlaneShown", "value", "main"),
    plane_choice=("numPlaneShownChoice", "value", "main"),
    image_wdgt=("imgScanPlot", None, "main"),
    pattern_wdgt=("imgScanPattern", None, "main"),
)
