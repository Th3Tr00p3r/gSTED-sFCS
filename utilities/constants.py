# -*- coding: utf-8 -*-
""" Global constants. """

import gui.icons.icon_paths as icon_path
from utilities.helper import DeviceAttrs, QtWidgetAccess, QtWidgetCollection

# ------------------------------
# general
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
    led_widget=QtWidgetAccess("ledExc", "setIcon", "main"),
    led_icon_path=icon_path.LED_BLUE,
    switch_widget=QtWidgetAccess("excOnButton", "setIcon", "main"),
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
    led_widget=QtWidgetAccess("ledDep", "setIcon", "main"),
    led_icon_path=icon_path.LED_ORANGE,
    switch_widget=QtWidgetAccess("depEmissionOn", "setIcon", "main"),
    param_widgets=QtWidgetCollection(
        model=QtWidgetAccess("depMod", "text"),
        address=QtWidgetAccess("depAddr", "text"),
    ),
)

DEP_SHUTTER = DeviceAttrs(
    cls_name="SimpleDO",
    log_ref="Shutter",
    led_widget=QtWidgetAccess("ledShutter", "setIcon", "main"),
    switch_widget=QtWidgetAccess("depShutterOn", "setIcon", "main"),
    param_widgets=QtWidgetCollection(
        address=QtWidgetAccess("depShutterAddr", "text"),
    ),
)

STAGE = DeviceAttrs(
    cls_name="StepperStage",
    log_ref="Stage",
    led_widget=QtWidgetAccess("ledStage", "setIcon", "main"),
    switch_widget=QtWidgetAccess("stageOn", "setIcon", "main"),
    param_widgets=QtWidgetCollection(address=QtWidgetAccess("arduinoAddr", "text")),
)

COUNTER = DeviceAttrs(
    cls_name="Counter",
    cls_xtra_args=["app.devices.SCANNERS.ai_task"],
    log_ref="Counter",
    led_widget=QtWidgetAccess("ledCounter", "setIcon", "main"),
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
    led_widget=QtWidgetAccess("ledUm232h", "setIcon", "main"),
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
    led_widget=QtWidgetAccess("ledTdc", "setIcon", "main"),
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
    cls_xtra_args=["app.loop", "app.gui.camera"],
    log_ref="Camera",
    led_widget=QtWidgetAccess("ledCam", "setIcon", "main"),
    param_widgets=QtWidgetCollection(
        model=QtWidgetAccess("uc480PlaceHolder", "value"),
    ),
)

SCANNERS = DeviceAttrs(
    cls_name="Scanners",
    log_ref="Scanners",
    led_widget=QtWidgetAccess("ledScn", "setIcon", "main"),
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
#    led_widget=QtWidgetAccess("ledPxlClk", "setIcon", "main"),
#    param_widgets=QtWidgetCollection(
#        low_ticks=QtWidgetAccess("pixelClockLowTicks", "value"),
#        high_ticks=QtWidgetAccess("pixelClockHighTicks", "value"),
#        cntr_addr=QtWidgetAccess("pixelClockCounterAddress", "text"),
#        tick_src=QtWidgetAccess("pixelClockSrcOfTicks", "text"),
#        freq=QtWidgetAccess("pixelClockFreq", "value")
#    )
# )

# ----------------------------------------------
# GUI presets
# ----------------------------------------------

IMG_SCN_WDGT_COLLCTN = QtWidgetCollection(
    type=QtWidgetAccess("imgScanType", "setCurrentText", "main"),
    dim1=QtWidgetAccess("imgScanDim1", "setValue", "main"),
    dim2=QtWidgetAccess("imgScanDim2", "setValue", "main"),
    n_lines=QtWidgetAccess("imgScanNumLines", "setValue", "main"),
    pnts_per_line=QtWidgetAccess("imgScanPPLine", "setValue", "main"),
    line_freq=QtWidgetAccess("imgScanLineFreq", "setValue", "main"),
    lin_frac=QtWidgetAccess("imgScanLinFrac", "setValue", "main"),
    n_planes=QtWidgetAccess("imgScanNumPlanes", "setValue", "main"),
    z_step=QtWidgetAccess("imgScanZstep", "setValue", "main"),
)
