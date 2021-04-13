# -*- coding: utf-8 -*-
""" Global constants. """

import nidaqmx.constants as NI  # NOQA

import gui.icons.icon_paths as icon_path
from utilities.helper import DeviceAttrs, QtWidgetCollection

# ------------------------------
# General
# ------------------------------

TIMEOUT = 0.010  # seconds (10 ms)

AX_IDX = {"x": 0, "y": 1, "z": 2, "X": 0, "Y": 1, "Z": 2}

AXES_TO_BOOL_TUPLE_DICT = {
    "X": (True, False, False),
    "Y": (False, True, False),
    "Z": (False, False, True),
    "XY": (True, True, False),
    "XZ": (True, False, True),
    "YZ": (False, True, True),
    "XYZ": (True, True, True),
}

# ------------------------------
# Paths
# ------------------------------

MAINWINDOW_UI_PATH = "./gui/mainwindow.ui"
SETTINGSWINDOW_UI_PATH = "./gui/settingswindow.ui"
CAMERAWINDOW_UI_PATH = "./gui/camerawindow.ui"
ERRORSWINDOW_UI_PATH = "./gui/errorswindow.ui"
LOGWINDOW_UI_PATH = "./gui/logwindow.ui"

SETTINGS_FOLDER_PATH = "./settings/"
DEFAULT_SETTINGS_FILE_PATH = "./settings/default_settings.csv"
LOADOUT_FOLDER_PATH = "./settings/loadouts/"
DEFAULT_LOADOUT_FILE_PATH = "./settings/loadouts/default_loadout.csv"

LOG_FOLDER_PATH = "./log/"

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
    "EXC_LASER",
    "DEP_SHUTTER",
    "TDC",
    "DEP_LASER",
    "STAGE",
    "UM232H",
    "CAMERA",
    "SCANNERS",
    "COUNTER",
    "PXL_CLK",
)

LED_COLL = QtWidgetCollection(
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
    "EXC_LASER": "ledExc",
    "DEP_SHUTTER": "ledShutter",
    "TDC": "ledTdc",
    "DEP_LASER": "ledDep",
    "STAGE": "ledStage",
    "UM232H": "ledUm232h",
    "CAMERA": "ledCam",
    "SCANNERS": "ledScn",
    "COUNTER": "ledCounter",
    "PXL_CLK": "ledPxlClk",
}

SWITCH_COLL = QtWidgetCollection(
    exc=("excOnButton", "icon", "main"),
    dep=("depEmissionOn", "icon", "main"),
    shutter=("depShutterOn", "icon", "main"),
    stage=("stageOn", "icon", "main"),
)

EXC_LASER = DeviceAttrs(
    class_name="SimpleDO",
    log_ref="Excitation Laser",
    led_icon_path=icon_path.LED_BLUE,
    param_widgets=QtWidgetCollection(
        led_widget=("ledExc", "icon", "main"),
        switch_widget=("excOnButton", "icon", "main"),
        model=("excMod", "text"),
        trg_src=("excTriggerSrc", "currentText"),
        ext_trg_addr=("excTriggerExtAddr", "text"),
        int_trg_addr=("excTriggerIntAddr", "text"),
        address=("excAddr", "text"),
    ),
)

DEP_LASER = DeviceAttrs(
    class_name="DepletionLaser",
    log_ref="Depletion Laser",
    led_icon_path=icon_path.LED_ORANGE,
    param_widgets=QtWidgetCollection(
        led_widget=("ledDep", "icon", "main"),
        switch_widget=("depEmissionOn", "icon", "main"),
        model=("depMod", "text"),
        address=("depAddr", "text"),
    ),
)

DEP_SHUTTER = DeviceAttrs(
    class_name="SimpleDO",
    log_ref="Shutter",
    param_widgets=QtWidgetCollection(
        led_widget=("ledShutter", "icon", "main"),
        switch_widget=("depShutterOn", "icon", "main"),
        address=("depShutterAddr", "text"),
    ),
)

STAGE = DeviceAttrs(
    class_name="StepperStage",
    log_ref="Stage",
    param_widgets=QtWidgetCollection(
        led_widget=("ledStage", "icon", "main"),
        switch_widget=("stageOn", "icon", "main"),
        address=("arduinoAddr", "text"),
    ),
)

COUNTER = DeviceAttrs(
    class_name="Counter",
    cls_xtra_args=["devices.SCANNERS.tasks.ai"],
    log_ref="Counter",
    param_widgets=QtWidgetCollection(
        led_widget=("ledCounter", "icon", "main"),
        pxl_clk=("counterPixelClockAddress", "text"),
        pxl_clk_output=("pixelClockCounterIntOutputAddress", "text"),
        trggr=("counterTriggerAddress", "text"),
        trggr_armstart_digedge=("counterTriggerArmStartDigEdgeSrc", "text"),
        trggr_edge=("counterTriggerEdge", "currentText"),
        address=("counterAddress", "text"),
        CI_cnt_edges_term=("counterCIcountEdgesTerm", "text"),
        CI_dup_prvnt=("counterCIdupCountPrevention", "isChecked"),
    ),
)

UM232H = DeviceAttrs(
    class_name="UM232H",
    log_ref="UM232H",
    param_widgets=QtWidgetCollection(
        led_widget=("ledUm232h", "icon", "main"),
        vend_id=("um232VendID", "value"),
        prod_id=("um232ProdID", "value"),
        ltncy_tmr_val=("um232latencyTimerVal", "value"),
        flow_ctrl=("um232FlowControl", "text"),
        bit_mode=("um232BitMode", "text"),
        n_bytes=("um232nBytes", "value"),
    ),
)

TDC = DeviceAttrs(
    class_name="SimpleDO",
    log_ref="TDC",
    param_widgets=QtWidgetCollection(
        led_widget=("ledTdc", "icon", "main"),
        address=("TDCaddress", "text"),
        data_vrsn=("TDCdataVersion", "text"),
        laser_freq_MHz=("TDClaserFreq", "value"),
        fpga_freq_MHz=("TDCFPGAFreq", "value"),
        tdc_vrsn=("TDCversion", "value"),
    ),
)

CAMERA = DeviceAttrs(
    class_name="Camera",
    cls_xtra_args=["loop", "gui.camera"],
    log_ref="Camera",
    param_widgets=QtWidgetCollection(
        led_widget=("ledCam", "icon", "main"),
        model=("uc480PlaceHolder", "value"),
    ),
)

SCANNERS = DeviceAttrs(
    class_name="Scanners",
    log_ref="Scanners",
    param_widgets=QtWidgetCollection(
        led_widget=("ledScn", "icon", "main"),
        ao_x_init_vltg=("xAOV", "value", "main"),
        ao_y_init_vltg=("yAOV", "value", "main"),
        ao_z_init_vltg=("zAOV", "value", "main"),
        x_um2V_const=("xConv", "value"),
        y_um2V_const=("yConv", "value"),
        z_um2V_const=("zConv", "value"),
        ai_x_addr=("AIXaddr", "text"),
        ai_y_addr=("AIYaddr", "text"),
        ai_z_addr=("AIZaddr", "text"),
        ai_laser_mon_addr=("AIlaserMonAddr", "text"),
        ai_clk_div=("AIclkDiv", "value"),
        ai_trg_src=("AItrigSrc", "text"),
        ao_x_addr=("AOXaddr", "text"),
        ao_y_addr=("AOYaddr", "text"),
        ao_z_addr=("AOZaddr", "text"),
        ao_int_x_addr=("AOXintAddr", "text"),
        ao_int_y_addr=("AOYintAddr", "text"),
        ao_int_z_addr=("AOZintAddr", "text"),
        ao_dig_trg_src=("AOdigTrigSrc", "text"),
        ao_trg_edge=("AOtriggerEdge", "currentText"),
        ao_wf_type=("AOwfType", "currentText"),
    ),
)

PXL_CLK = DeviceAttrs(
    class_name="PixelClock",
    log_ref="Pixel Clock",
    param_widgets=QtWidgetCollection(
        led_widget=("ledPxlClk", "icon", "main"),
        low_ticks=("pixelClockLowTicks", "value"),
        high_ticks=("pixelClockHighTicks", "value"),
        cntr_addr=("pixelClockCounterAddress", "text"),
        tick_src=("pixelClockSrcOfTicks", "text"),
        out_term=("pixelClockOutput", "text"),
        out_ext_term=("pixelClockOutputExt", "text"),
        freq_MHz=("pixelClockFreq", "value"),
    ),
)

# ----------------------------------------------
# Measurements
# ----------------------------------------------

FCS_MEAS_WDGT_COLL = QtWidgetCollection(
    duration=("measFCSDur", "value", "main"),
    g0_wdgt=("fcsG0", "value", "main"),
    decay_time_wdgt=("fcsDecayTime", "value", "main"),
    prog_bar_wdgt=("fcsProgressBar", "value", "main"),
    save=("fcsSave", "isChecked", "main"),
    save_path=("fcsDataPath", "text", "settings"),
    save_frmt=("saveFormat", "currentText", "settings"),
    plot_wdgt=("fcsPlot", None, "main"),
)

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
    exc_mode=("solScanModeExc", "isChecked", "main"),
    dep_mode=("solScanModeDep", "isChecked", "main"),
    sted_mode=("solScanModeSted", "isChecked", "main"),
)

SOL_MEAS_WDGT_COLL = QtWidgetCollection(
    file_template=("solScanFileTemplate", "text", "main"),
    save_path=("solDataPath", "text", "settings"),
    save_frmt=("saveFormat", "currentText", "settings"),
    max_file_size=("solScanMaxFileSize", "value", "main"),
    cal_duration=("solScanCalDur", "value", "main"),
    total_duration=("solScanTotalDur", "value", "main"),
    cal_save_intrvl_wdgt=("solScanCalSaveIntrvl", "value", "main"),
    prog_bar_wdgt=("solScanProgressBar", "value", "main"),
    start_time_wdgt=("solScanStartTime", "time", "main"),
    end_time_wdgt=("solScanEndTime", "time", "main"),
    file_num_wdgt=("solScanFileNo", "value", "main"),
    total_files_wdgt=("solScanTotalFiles", "value", "main"),
    pattern_wdgt=("solScanPattern", None, "main"),
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

# TODO: make laser modes into a QButtonGroup in gui.py, then can treat as a single parameter
IMG_SCN_WDGT_FILLOUT_DICT = {
    "Locate Plane - YZ Coarse": [1, 0, 0, "YZ", 15, 15, 10, 80, 1000, 20, 0.9, 1],
    "MFC - XY compartment": [""],  # TODO
    "GB -  XY Coarse": [0, 1, 0, "XY", 15, 15, 0, 80, 1000, 20, 0.9, 1],
    "GB - XY bead area": [1, 0, 0, "XY", 5, 5, 0, 80, 1000, 20, 0.9, 1],
    "GB - XY single bead": [""],  # TODO
    "GB - YZ single bead": [""],  # TODO
}

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
