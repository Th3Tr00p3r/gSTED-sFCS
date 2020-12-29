# -*- coding: utf-8 -*-
""" Global constants. """

import gui.icons.icon_paths as icon

# general
TIMEOUT = 0.010  # seconds (10 ms)
ORIGIN = (0.0, 0.0, 5.0)

# paths
MAINWINDOW_UI_PATH = "./gui/mainwindow.ui"
SETTINGSWINDOW_UI_PATH = "./gui/settingswindow.ui"
CAMERAWINDOW_UI_PATH = "./gui/camerawindow.ui"
ERRORSWINDOW_UI_PATH = "./gui/errorswindow.ui"
LOGWINDOW_UI_PATH = "./gui/logwindow.ui"

SETTINGS_FOLDER_PATH = "./settings/"
DEFAULT_SETTINGS_FILE_PATH = "./settings/default_settings.csv"

# sounds
MEAS_COMPLETE_SOUND = "./sounds/meas_complete.wav"
#                        from PyQt5.QtMultimedia import QSound
#                                if self.time_passed == self.duration_spinbox.value():
#                                    QSound.play(const.MEAS_COMPLETE_SOUND);

# devices
DEP_CMND_DICT = {
    "tmp": "SHGtemp",
    "curr": "LDcurrent 1",
    "pow": "Power 0",
}

DVC_LED_NAME = {
    "EXC_LASER": "ledExc",
    "DEP_LASER": "ledDep",
    "DEP_SHUTTER": "ledShutter",
    "STAGE": "ledStage",
    "COUNTER": "ledCounter",
    "UM232": "ledUm232",
    "TDC": "ledTdc",
    "CAMERA": "ledCam",
    "SCANNERS": "ledScn",
}

ICON_DICT = {
    "EXC_LASER": {
        "LED": DVC_LED_NAME["EXC_LASER"],
        "SWITCH": "excOnButton",
        "ICON": icon.LED_BLUE,
    },
    "DEP_LASER": {
        "LED": DVC_LED_NAME["DEP_LASER"],
        "SWITCH": "depEmissionOn",
        "ICON": icon.LED_ORANGE,
    },
    "DEP_SHUTTER": {
        "LED": DVC_LED_NAME["DEP_SHUTTER"],
        "SWITCH": "depShutterOn",
        "ICON": icon.LED_GREEN,
    },
    "STAGE": {
        "LED": DVC_LED_NAME["STAGE"],
        "SWITCH": "stageOn",
        "ICON": icon.LED_GREEN,
    },
    "COUNTER": {"LED": DVC_LED_NAME["COUNTER"], "ICON": icon.LED_GREEN},
    "UM232": {"LED": DVC_LED_NAME["UM232"], "ICON": icon.LED_GREEN},
    "TDC": {"LED": DVC_LED_NAME["TDC"], "ICON": icon.LED_GREEN},
    "CAMERA": {"LED": DVC_LED_NAME["CAMERA"], "ICON": icon.LED_GREEN},
    "SCANNERS": {"LED": DVC_LED_NAME["SCANNERS"], "ICON": icon.LED_GREEN},
}

DVC_LOG_DICT = {
    "EXC_LASER": "Excitation Laser",
    "DEP_LASER": "Depletion Laser",
    "DEP_SHUTTER": "Shutter",
    "STAGE": "Stage",
    "CAMERA": "Camera",
    "TDC": "TDC",
    "COUNTER": "Counter",
    "UM232": "UM232",
    "SCANNERS": "Scanners",
}

DEVICE_NICKS = [
    "EXC_LASER",
    "DEP_SHUTTER",
    "TDC",
    "DEP_LASER",
    "STAGE",
    "UM232",
    "CAMERA",
    "SCANNERS",
    "COUNTER",
]

DVC_CLASS_NAMES = {
    "EXC_LASER": "SimpleDO",
    "TDC": "SimpleDO",
    "DEP_SHUTTER": "SimpleDO",
    "DEP_LASER": "DepletionLaser",
    "STAGE": "StepperStage",
    "COUNTER": "Counter",
    "UM232": "UM232",
    "CAMERA": "Camera",
    "SCANNERS": "Scanners",
}

EXC_LASER_PARAM_GUI_DICT = {
    "model": {"field": "excMod", "access": "text"},
    "trg_src": {"field": "excTriggerSrc", "access": "currentText"},
    "ext_trg_addr": {"field": "excTriggerExtAddr", "access": "text"},
    "int_trg_addr": {"field": "excTriggerIntAddr", "access": "text"},
    "addr": {"field": "excAddr", "access": "text"},
}

DEP_LASER_PARAM_GUI_DICT = {
    "model": {"field": "depMod", "access": "text"},
    "update_time": {"field": "depUpdateTime", "access": "value"},
    "addr": {"field": "depAddr", "access": "text"},
}

DEP_SHUTTER_PARAM_GUI_DICT = {"addr": {"field": "depShutterAddr", "access": "text"}}

STAGE_PARAM_GUI_DICT = {"addr": {"field": "arduinoAddr", "access": "text"}}

COUNTER_PARAM_GUI_DICT = {
    "buff_sz": {"field": "counterBufferSizeSpinner", "access": "value"},
    "update_time": {"field": "counterUpdateTime", "access": "value"},
    "pxl_clk": {"field": "counterPixelClockAddress", "access": "text"},
    "pxl_clk_output": {
        "field": "pixelClockCounterIntOutputAddress",
        "access": "text",
    },
    "trggr": {"field": "counterTriggerAddress", "access": "text"},
    "trggr_armstart_digedge": {
        "field": "counterTriggerArmStartDigEdgeSrc",
        "access": "text",
    },
    "trggr_edge": {"field": "counterTriggerEdge", "access": "currentText"},
    "photon_cntr": {"field": "counterPhotonCounter", "access": "text"},
    "CI_cnt_edges_term": {
        "field": "counterCIcountEdgesTerm",
        "access": "text",
    },
    "CI_dup_prvnt": {
        "field": "counterCIdupCountPrevention",
        "access": "isChecked",
    },
}

UM232_PARAM_GUI_DICT = {
    "vend_id": {"field": "um232VendID", "access": "value"},
    "prod_id": {"field": "um232ProdID", "access": "value"},
    "dvc_dscrp": {"field": "um232DeviceDescription", "access": "text"},
    "ltncy_tmr_val": {"field": "um232latencyTimerVal", "access": "value"},
    "baud_rate": {"field": "um232BaudRate", "access": "value"},
    "read_timeout": {"field": "um232ReadTimeout", "access": "value"},
    "flow_ctrl": {"field": "um232FlowControl", "access": "text"},
    "bit_mode": {"field": "um232BitMode", "access": "text"},
    "n_bytes": {"field": "um232nBytes", "access": "value"},
}

TDC_PARAM_GUI_DICT = {
    "addr": {"field": "TDCaddress", "access": "text"},
    "data_vrsn": {"field": "TDCdataVersion", "access": "text"},
    "laser_freq": {"field": "TDClaserFreq", "access": "value"},
    "fpga_freq": {"field": "TDCFPGAFreq", "access": "value"},
    "pxl_clk_freq": {"field": "TDCpixelClockFreq", "access": "value"},
    "tdc_vrsn": {"field": "TDCversion", "access": "value"},
}

PXL_CLK_PARAM_GUI_DICT = {
    "low_ticks": {"field": "pixelClockLowTicks", "access": "value"},
    "high_ticks": {"field": "pixelClockHighTicks", "access": "value"},
    "cntr_addr": {"field": "pixelClockCounterAddress", "access": "text"},
    "tick_src": {"field": "pixelClockSrcOfTicks", "access": "text"},
    "freq": {"field": "pixelClockFreq", "access": "value"},
}

SCANNERS_PARAM_GUI_DICT = {
    "ai_x_addr": {"field": "AIXaddr", "access": "text"},
    "ai_y_addr": {"field": "AIYaddr", "access": "text"},
    "ai_z_addr": {"field": "AIZaddr", "access": "text"},
    "ai_laser_mon_addr": {"field": "AIlaserMonAddr", "access": "text"},
    "ai_clk_div": {"field": "AIclkDivSpinner", "access": "value"},
    "ai_trg_src": {"field": "AItrigSrc", "access": "text"},
    "ao_x_p_addr": {"field": "AOXaddrP", "access": "text"},
    "ao_x_n_addr": {"field": "AOXaddrN", "access": "text"},
    "ao_y_p_addr": {"field": "AOYaddrP", "access": "text"},
    "ao_y_n_addr": {"field": "AOYaddrN", "access": "text"},
    "ao_z_addr": {"field": "AOZaddr", "access": "text"},
    "ao_int_x_p_addr": {"field": "AOXintAddrP", "access": "text"},
    "ao_int_x_n_addr": {"field": "AOXintAddrN", "access": "text"},
    "ao_int_y_p_addr": {"field": "AOYintAddrP", "access": "text"},
    "ao_int_y_n_addr": {"field": "AOYintAddrN", "access": "text"},
    "ao_int_z_addr": {"field": "AOZintAddr", "access": "text"},
    "ao_dig_trg_src": {"field": "AOdigTrigSrc", "access": "text"},
    "ao_trg_edge": {"field": "AOtriggerEdge", "access": "currentText"},
    "ao_wf_type": {"field": "AOwfType", "access": "currentText"},
}

CAMERA_PARAM_GUI_DICT = {"vid_intrvl": {"field": "uc480VidIntrvl", "access": "value"}}

# TODO: There has to be a more elegant way to get the parameters from the settings window into the device classes.
DVC_NICK_PARAMS_DICT = {
    "EXC_LASER": EXC_LASER_PARAM_GUI_DICT,
    "DEP_SHUTTER": DEP_SHUTTER_PARAM_GUI_DICT,
    "DEP_LASER": DEP_LASER_PARAM_GUI_DICT,
    "STAGE": STAGE_PARAM_GUI_DICT,
    "COUNTER": COUNTER_PARAM_GUI_DICT,
    "UM232": UM232_PARAM_GUI_DICT,
    "TDC": TDC_PARAM_GUI_DICT,
    "PXL_CLK": PXL_CLK_PARAM_GUI_DICT,
    "SCANNERS": SCANNERS_PARAM_GUI_DICT,
    "CAMERA": CAMERA_PARAM_GUI_DICT,
}

DVC_X_ARGS_DICT = {
    "EXC_LASER": [],
    "DEP_SHUTTER": [],
    "DEP_LASER": [],
    "STAGE": [],
    "COUNTER": ['app.dvc_dict["SCANNERS"].ai_task'],
    "UM232": [],
    "TDC": [],
    "PXL_CLK": [],
    "SCANNERS": [
        '(app.win_dict["main"].xAoV.value(), app.win_dict["main"].yAoV.value(), app.win_dict["main"].zAoV.value())'
    ],
    "CAMERA": ["app.loop", 'app.win_dict["camera"]'],
}
