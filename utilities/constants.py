# -*- coding: utf-8 -*-
""" Global constants. """

import gui.icons.icon_paths as icon_path
from utilities.helper import DeviceAttrs, QtWidgetAccess, QtWidgetCollection

# ------------------------------
# General
# ------------------------------

TIMEOUT = 0.010  # seconds (10 ms)
ORIGIN = (0.0, 0.0, 5.0)  # TODO: move to settings (scanners)

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
)

EXC_LASER = DeviceAttrs(
    cls_name="SimpleDO",
    log_ref="Excitation Laser",
    led_widget=QtWidgetAccess("ledExc", "icon", "main"),
    led_icon_path=icon_path.LED_BLUE,
    switch_widget=QtWidgetAccess("excOnButton", "icon", "main"),
    param_widgets=QtWidgetCollection(
        model=QtWidgetAccess("excMod", "text"),
        trg_src=QtWidgetAccess("excTriggerSrc", "currentText"),
        ext_trg_addr=QtWidgetAccess("excTriggerExtAddr", "text"),
        int_trg_addr=QtWidgetAccess("excTriggerIntAddr", "text"),
        address=QtWidgetAccess("excAddr", "text"),
    ),
)

DEP_LASER = DeviceAttrs(
    cls_name="DepletionLaser",
    log_ref="Depletion Laser",
    led_widget=QtWidgetAccess("ledDep", "icon", "main"),
    led_icon_path=icon_path.LED_ORANGE,
    switch_widget=QtWidgetAccess("depEmissionOn", "icon", "main"),
    param_widgets=QtWidgetCollection(
        model=QtWidgetAccess("depMod", "text"),
        address=QtWidgetAccess("depAddr", "text"),
    ),
)

DEP_SHUTTER = DeviceAttrs(
    cls_name="SimpleDO",
    log_ref="Shutter",
    led_widget=QtWidgetAccess("ledShutter", "icon", "main"),
    switch_widget=QtWidgetAccess("depShutterOn", "icon", "main"),
    param_widgets=QtWidgetCollection(
        address=QtWidgetAccess("depShutterAddr", "text"),
    ),
)

STAGE = DeviceAttrs(
    cls_name="StepperStage",
    log_ref="Stage",
    led_widget=QtWidgetAccess("ledStage", "icon", "main"),
    switch_widget=QtWidgetAccess("stageOn", "icon", "main"),
    param_widgets=QtWidgetCollection(address=QtWidgetAccess("arduinoAddr", "text")),
)

COUNTER = DeviceAttrs(
    cls_name="Counter",
    cls_xtra_args=["devices.SCANNERS.ai_task"],
    log_ref="Counter",
    led_widget=QtWidgetAccess("ledCounter", "icon", "main"),
    param_widgets=QtWidgetCollection(
        pxl_clk=QtWidgetAccess("counterPixelClockAddress", "text"),
        pxl_clk_output=QtWidgetAccess("pixelClockCounterIntOutputAddress", "text"),
        trggr=QtWidgetAccess("counterTriggerAddress", "text"),
        trggr_armstart_digedge=QtWidgetAccess(
            "counterTriggerArmStartDigEdgeSrc", "text"
        ),
        trggr_edge=QtWidgetAccess("counterTriggerEdge", "currentText"),
        address=QtWidgetAccess("counterAddress", "text"),
        CI_cnt_edges_term=QtWidgetAccess("counterCIcountEdgesTerm", "text"),
        CI_dup_prvnt=QtWidgetAccess("counterCIdupCountPrevention", "isChecked"),
    ),
)

UM232H = DeviceAttrs(
    cls_name="UM232H",
    log_ref="UM232H",
    led_widget=QtWidgetAccess("ledUm232h", "icon", "main"),
    param_widgets=QtWidgetCollection(
        vend_id=QtWidgetAccess("um232VendID", "value"),
        prod_id=QtWidgetAccess("um232ProdID", "value"),
        ltncy_tmr_val=QtWidgetAccess("um232latencyTimerVal", "value"),
        flow_ctrl=QtWidgetAccess("um232FlowControl", "text"),
        bit_mode=QtWidgetAccess("um232BitMode", "text"),
        n_bytes=QtWidgetAccess("um232nBytes", "value"),
    ),
)

TDC = DeviceAttrs(
    cls_name="SimpleDO",
    log_ref="TDC",
    led_widget=QtWidgetAccess("ledTdc", "icon", "main"),
    param_widgets=QtWidgetCollection(
        address=QtWidgetAccess("TDCaddress", "text"),
        data_vrsn=QtWidgetAccess("TDCdataVersion", "text"),
        laser_freq=QtWidgetAccess("TDClaserFreq", "value"),
        fpga_freq=QtWidgetAccess("TDCFPGAFreq", "value"),
        pxl_clk_freq=QtWidgetAccess("TDCpixelClockFreq", "value"),
        tdc_vrsn=QtWidgetAccess("TDCversion", "value"),
    ),
)

CAMERA = DeviceAttrs(
    cls_name="Camera",
    cls_xtra_args=["loop", "gui.camera"],
    log_ref="Camera",
    led_widget=QtWidgetAccess("ledCam", "icon", "main"),
    param_widgets=QtWidgetCollection(
        model=QtWidgetAccess("uc480PlaceHolder", "value"),
    ),
)

SCANNERS = DeviceAttrs(
    cls_name="Scanners",
    log_ref="Scanners",
    led_widget=QtWidgetAccess("ledScn", "icon", "main"),
    param_widgets=QtWidgetCollection(
        ao_x_init_vltg=QtWidgetAccess("xAoV", "value", "main"),
        ao_y_init_vltg=QtWidgetAccess("yAoV", "value", "main"),
        ao_z_init_vltg=QtWidgetAccess("zAoV", "value", "main"),
        x_conv_const=QtWidgetAccess("xConv", "value"),
        y_conv_const=QtWidgetAccess("yConv", "value"),
        z_conv_const=QtWidgetAccess("zConv", "value"),
        ai_x_addr=QtWidgetAccess("AIXaddr", "text"),
        ai_y_addr=QtWidgetAccess("AIYaddr", "text"),
        ai_z_addr=QtWidgetAccess("AIZaddr", "text"),
        ai_laser_mon_addr=QtWidgetAccess("AIlaserMonAddr", "text"),
        ai_clk_div=QtWidgetAccess("AIclkDiv", "value"),
        ai_trg_src=QtWidgetAccess("AItrigSrc", "text"),
        ao_x_addr=QtWidgetAccess("AOXaddr", "text"),
        ao_y_addr=QtWidgetAccess("AOYaddr", "text"),
        ao_z_addr=QtWidgetAccess("AOZaddr", "text"),
        ao_int_x_addr=QtWidgetAccess("AOXintAddr", "text"),
        ao_int_y_addr=QtWidgetAccess("AOYintAddr", "text"),
        ao_int_z_addr=QtWidgetAccess("AOZintAddr", "text"),
        ao_dig_trg_src=QtWidgetAccess("AOdigTrigSrc", "text"),
        ao_trg_edge=QtWidgetAccess("AOtriggerEdge", "currentText"),
        ao_wf_type=QtWidgetAccess("AOwfType", "currentText"),
    ),
)

# PXL_CLK = DeviceAttrs(
#    cls_name="PixelClock",
#    log_ref="Pixel Clock",
#    led_widget=QtWidgetAccess("ledPxlClk", "icon", "main"),
#    param_widgets=QtWidgetCollection(
#        low_ticks=QtWidgetAccess("pixelClockLowTicks", "value"),
#        high_ticks=QtWidgetAccess("pixelClockHighTicks", "value"),
#        cntr_addr=QtWidgetAccess("pixelClockCounterAddress", "text"),
#        tick_src=QtWidgetAccess("pixelClockSrcOfTicks", "text"),
#        freq=QtWidgetAccess("pixelClockFreq", "value")
#    )
# )

# ----------------------------------------------
# Measurements
# ----------------------------------------------

FCS_MEAS_WDGT_COLL = QtWidgetCollection(
    duration=QtWidgetAccess("measFCSDur", "value", "main"),
    g0_wdgt=QtWidgetAccess("fcsG0", "value", "main"),
    decay_time_wdgt=QtWidgetAccess("fcsDecayTime", "value", "main"),
    prog_bar_wdgt=QtWidgetAccess("fcsProgressBar", "value", "main"),
)

SOL_ANG_SCN_WDGT_COLL = QtWidgetCollection(
    line_length=QtWidgetAccess("lineLen", "value", "main"),
    ao_smplng_freq=QtWidgetAccess("aoSampFreq", "value", "main"),
    angle=QtWidgetAccess("angle", "value", "main"),
    lin_frac=QtWidgetAccess("solLinFrac", "value", "main"),
    line_shift=QtWidgetAccess("lineShift", "value", "main"),
    speed=QtWidgetAccess("solAngScanSpeed", "value", "main"),
    min_lines=QtWidgetAccess("minNumLines", "value", "main"),
    max_scn_freq=QtWidgetAccess("maxScanFreq", "value", "main"),
)

SOL_MEAS_WDGT_COLL = QtWidgetCollection(
    file_template=QtWidgetAccess("solScanFileTemplate", "text", "main"),
    save_path=QtWidgetAccess("solDataPath", "text", "settings"),
    max_file_size=QtWidgetAccess("solScanMaxFileSize", "value", "main"),
    cal_duration=QtWidgetAccess("solScanCalDur", "value", "main"),
    total_duration=QtWidgetAccess("solScanTotalDur", "value", "main"),
    cal_save_intrvl_wdgt=QtWidgetAccess("solScanCalSaveIntrvl", "value", "main"),
    prog_bar_wdgt=QtWidgetAccess("solScanProgressBar", "value", "main"),
    start_time_wdgt=QtWidgetAccess("solScanStartTime", "time", "main"),
    end_time_wdgt=QtWidgetAccess("solScanEndTime", "time", "main"),
    file_num_wdgt=QtWidgetAccess("solScanFileNo", "value", "main"),
    total_files_wdgt=QtWidgetAccess("solScanTotalFiles", "value", "main"),
)

IMG_SCN_WDGT_COLL = QtWidgetCollection(
    exc_mode=QtWidgetAccess("imgScanModeExc", "checked", "main"),
    dep_mode=QtWidgetAccess("imgScanModeDep", "checked", "main"),
    sted_mode=QtWidgetAccess("imgScanModeSted", "checked", "main"),
    type=QtWidgetAccess("imgScanType", "currentText", "main"),
    dim1=QtWidgetAccess("imgScanDim1", "value", "main"),
    dim2=QtWidgetAccess("imgScanDim2", "value", "main"),
    n_lines=QtWidgetAccess("imgScanNumLines", "value", "main"),
    pnts_per_line=QtWidgetAccess("imgScanPPLine", "value", "main"),
    line_freq=QtWidgetAccess("imgScanLineFreq", "value", "main"),
    lin_frac=QtWidgetAccess("imgScanLinFrac", "value", "main"),
    n_planes=QtWidgetAccess("imgScanNumPlanes", "value", "main"),
    z_step=QtWidgetAccess("imgScanZstep", "value", "main"),
)

IMG_MEAS_WDGT_COLL = QtWidgetCollection(
    file_template=QtWidgetAccess("imgScanFileTemplate", "text", "main"),
    prog_bar_wdgt=QtWidgetAccess("imgScanProgressBar", "value", "main"),
    curr_line_wdgt=QtWidgetAccess("currLine", "value", "main"),
    curr_plane_wdgt=QtWidgetAccess("currPlane", "value", "main"),
)
