""" Global constants. """

import nidaqmx.constants as NI  # NOQA

from utilities.helper import QtWidgetCollection

# ------------------------------
# Paths
# ------------------------------

MAINWINDOW_UI_PATH = "./gui/mainwindow.ui"
SETTINGSWINDOW_UI_PATH = "./gui/settingswindow.ui"
CAMERAWINDOW_UI_PATH = "./gui/camerawindow.ui"
ERRORSWINDOW_UI_PATH = "./gui/errorswindow.ui"
LOGWINDOW_UI_PATH = "./gui/logwindow.ui"

SETTINGS_DIR_PATH = "./settings/"
DEFAULT_SETTINGS_FILE_PATH = "./settings/default_settings.csv"
LOADOUT_DIR_PATH = "./settings/loadouts/"
DEFAULT_LOADOUT_FILE_PATH = "./settings/loadouts/default_loadout.csv"

LOG_DIR_PATH = "./log/"

# ------------------------------
# Sounds
# ------------------------------

MEAS_COMPLETE_SOUND = "./sounds/meas_complete.wav"
#                        from PyQt5.QtMultimedia import QSound
#                                if self.time_passed == self.duration_spinbox.value():
#                                    QSound.play(consts.MEAS_COMPLETE_SOUND);

# ------------------------------
# Devices
# ------------------------------

DVC_NICKS_TUPLE = (
    "exc_laser",
    "dep_shutter",
    "TDC",
    "dep_laser",
    "stage",
    "UM232H",
    "camera",
    "scanners",
    "photon_detector",
    "pixel_clock",
)

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

DVC_NICK_LED_NAME_DICT = {
    "exc_laser": "ledExc",
    "dep_shutter": "ledShutter",
    "TDC": "ledTdc",
    "dep_laser": "ledDep",
    "stage": "ledStage",
    "UM232H": "ledUm232h",
    "camera": "ledCam",
    "scanners": "ledScn",
    "photon_detector": "ledCounter",
    "pixel_clock": "ledPxlClk",
}

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
    # TODO: move laser modes to SOL_MEAS_WDGT_COLL (also from 2 collections above)
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
    exc_mode=("imgScanModeExc", "isChecked", "main"),
    dep_mode=("imgScanModeDep", "isChecked", "main"),
    sted_mode=("imgScanModeSted", "isChecked", "main"),
    scan_plane=("imgScanType", "currentText", "main"),
    dim1_um=("imgScanDim1", "value", "main"),
    dim2_um=("imgScanDim2", "value", "main"),
    dim3_um=("imgScanDim3", "value", "main"),
    n_lines=("imgScanNumLines", "value", "main"),
    ppl=("imgScanPPL", "value", "main"),
    line_freq_Hz=("imgScanLineFreq", "value", "main"),
    lin_frac=("imgScanLinFrac", "value", "main"),
    n_planes=("imgScanNumPlanes", "value", "main"),
    curr_ao_x=("xAOV", "value", "main"),
    curr_ao_y=("yAOV", "value", "main"),
    curr_ao_z=("zAOV", "value", "main"),
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

# ----------------------------------------------
# Presets
# ----------------------------------------------

IMG_SCN_WDGT_FILLOUT_DICT = {
    "Locate Plane - YZ Coarse": [1, 0, 0, "YZ", 15, 15, 10, 80, 1000, 20, 0.9, 1],
    "MFC - XY compartment": [1, 0, 0, "XY", 70, 70, 0, 80, 1000, 20, 0.9, 1],
    "GB -  XY Coarse": [0, 1, 0, "XY", 15, 15, 0, 80, 1000, 20, 0.9, 1],
    "GB - XY bead area": [1, 0, 0, "XY", 5, 5, 0, 80, 1000, 20, 0.9, 1],
    "GB - XY single bead": [1, 0, 0, "XY", 1, 1, 0, 80, 1000, 20, 0.9, 1],
    "GB - YZ single bead": [1, 0, 0, "YZ", 2.5, 2.5, 0, 80, 1000, 20, 0.9, 1],
}

SOL_MEAS_WDGT_FILLOUT_DICT = {
    "Standard Alignment": {
        "scan_type": "static",
        "repeat": True,
        "duration_units": "seconds",
        "total_duration": 10,
    },
    "Standard Angular": {
        "scan_type": "angular",
        "repeat": False,
        "duration_units": "hours",
        "total_duration": 1,
    },
    "Standard Circular": {
        "scan_type": "circle",
        "repeat": False,
        "duration_units": "hours",
        "total_duration": 1,
    },
}
