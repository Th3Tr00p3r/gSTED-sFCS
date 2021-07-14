""" Widget Collections """

from utilities.helper import QtWidgetCollection

# TODO: something to think about - all of the following collection objects are instantiated here.
# When I call, e.g., utilities.helper.led_wdgts from some other module, I'm actually using the existing object,
# and not a newly instantiated one. This may be good since this way I'm reusing stuff which does not change throughout
# the lifetime of the application - pehaps I should just hold_all objects by default and behind the scenes, to make the use of this class clearer

# ------------------------------
# Devices
# ------------------------------

led_wdgts = QtWidgetCollection(
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

switch_wdgts = QtWidgetCollection(
    exc=("excOnButton", "icon", "main"),
    dep=("depEmissionOn", "icon", "main"),
    shutter=("depShutterOn", "icon", "main"),
    stage=("stageOn", "icon", "main"),
)

# ----------------------------------------------
# Analysis Widget Collections
# ----------------------------------------------

data_import_wdgts = QtWidgetCollection(
    is_image_type=("imageDataImport", "isChecked", "main"),
    is_solution_type=("solDataImport", "isChecked", "main"),
    data_days=("dataDay", "currentText", "main"),
    data_months=("dataMonth", "currentText", "main"),
    data_years=("dataYear", "currentText", "main"),
    data_templates=("dataTemplate", "currentText", "main"),
    import_stacked=("importStacked", "currentIndex", "main"),
    is_calibration=("isDataCalibration", "isChecked", "main"),
)

# ----------------------------------------------
# Measurement Widget Collections
# ----------------------------------------------

sol_ang_scan_wdgts = QtWidgetCollection(
    max_line_len_um=("maxLineLen", "value", "main"),
    ao_samp_freq_Hz=("angAoSampFreq", "value", "main"),
    angle_deg=("angle", "value", "main"),
    lin_frac=("solLinFrac", "value", "main"),
    line_shift_um=("lineShift", "value", "main"),
    speed_um_s=("solAngScanSpeed", "value", "main"),
    min_lines=("minNumLines", "value", "main"),
    max_scan_freq_Hz=("maxScanFreq", "value", "main"),
)

sol_circ_scan_wdgts = QtWidgetCollection(
    ao_samp_freq_Hz=("circAoSampFreq", "value", "main"),
    diameter_um=("circDiameter", "value", "main"),
    speed_um_s=("circSpeed", "value", "main"),
)

sol_meas_wdgts = QtWidgetCollection(
    scan_type=("solScanType", "currentText", "main"),
    file_template=("solScanFileTemplate", "text", "main"),
    repeat=("repeatSolMeas", "isChecked", "main"),
    save_path=("solDataPath", "text", "settings"),
    save_frmt=("saveFormat", "currentText", "settings"),
    max_file_size_mb=("solScanMaxFileSize", "value", "main"),
    duration=("solScanDur", "value", "main"),
    duration_units=("solScanDurUnits", "currentText", "main"),
    prog_bar_wdgt=("solScanProgressBar", "value", "main"),
    start_time_wdgt=("solScanStartTime", "time", "main"),
    end_time_wdgt=("solScanEndTime", "time", "main"),
    file_num_wdgt=("solScanFileNo", "value", "main"),
    pattern_wdgt=("solScanPattern", None, "main"),
    g0_wdgt=("g0", "value", "main"),
    tau_wdgt=("decayTime", "value", "main"),
    plot_wdgt=("acf", None, "main"),
    fit_led=("ledFit", "icon", "main"),
)

img_scan_wdgts = QtWidgetCollection(
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

img_meas_wdgts = QtWidgetCollection(
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
