'''
Constants
'''
import gui.icons.icon_paths as icon

# general
EXIT_CODE_REBOOT = -123456789 # for restarting the application
MIN_SHG_TEMP = 52
COUNTER_BUFFER_ALLOC_SIZE = (1, )

# icons
ICON_DICT = {'EXC_LASER': {'LED': 'ledExc',
                                          'SWITCH': 'excOnButton',
                                          'ICON': icon.LED_BLUE
                                          }, 
                   'DEP_LASER': {'LED': 'ledDep',
                                         'SWITCH': 'depEmissionOn',
                                         'ICON': icon.LED_ORANGE
                                         },
                   'DEP_SHUTTER': {'LED': 'ledShutter',
                                         'SWITCH': 'depShutterOn',
                                         'ICON': icon.LED_GREEN
                                         },
                   'STAGE': {'SWITCH': 'stageOn'}
                    }

# log
LOG_VERBOSITY = {'always', 'verbose'}
LOG_DICT = {'EXC_LASER': 'excitation laser',
                   'DEP_LASER': 'depletion laser',
                   'DEP_SHUTTER': 'depletion shutter',
                   'STAGE': 'stepper stage', 
                   }

# paths
MAINWINDOW_UI_PATH = './gui/mainwindow.ui'
SETTINGSWINDOW_UI_PATH = './gui/settingswindow.ui'
CAMERAWINDOW_UI_PATH = './gui/camerawindow.ui'
ERRORSWINDOW_UI_PATH = './gui/errorswindow.ui'
LOGWINDOW_UI_PATH = './gui/logwindow.ui'

SETTINGS_FOLDER_PATH = './settings/'
DEFAULT_SETTINGS_FILE_PATH = './settings/default_settings.csv'

# sounds
MEAS_COMPLETE_SOUND = './sounds/meas_complete.wav'

# devices
DEVICE_NICKS = ['COUNTER', 'EXC_LASER', 'DEP_LASER', 'DEP_SHUTTER', 'STAGE']

DEVICE_ADDRESS_GUI_DICT = {'EXC_LASER': 'excTriggerExtChan',
                                            'DEP_LASER': 'depPort',
                                            'DEP_SHUTTER': 'depShutterChan',
                                            'STAGE': 'arduinoChan', 
                                            }
# device classes
DEVICE_CLASS_NAMES = {'EXC_LASER': 'ExcitationLaser',
                                     'DEP_LASER': 'DepletionLaser',
                                     'DEP_SHUTTER': 'DepletionShutter',
                                     'STAGE': 'StepperStage',
                                     'COUNTER': 'Counter',
                                     'UM232': 'UM232'
                                     }
                            
COUNTER_PARAM_GUI_DICT = {'buff_sz': {'field': 'counterBufferSizeSpinner',
                                                               'access': 'value'},
                                           'update_time': {'field': 'counterUpdateTimeSpinner',
                                                                     'access': 'value'},
                                           'pxl_clk': {'field': 'counterPixelClockAddress',
                                                              'access': 'text'},
                                           'pxl_clk_output': {'field': 'pixelClockCounterIntOutputAddress',
                                                                         'access': 'text'},
                                           'trggr': {'field': 'counterTriggerAddress',
                                                          'access': 'text'},
                                           'trggr_armstart_digedge': {'field': 'counterTriggerArmStartDigEdgeSrc',
                                                                                     'access': 'text'},
                                           'trggr_edge': {'field': 'counterTriggerEdge',
                                                                  'access': 'currentText'},
                                           'photon_cntr': {'field': 'counterPhotonCounter',
                                                                    'access': 'text'},
                                           'CI_cnt_edges_term': {'field': 'counterCIcountEdgesTerm',
                                                                             'access': 'text'},
                                           'CI_dup_prvnt': {'field': 'counterCIdupCountPrevention',
                                                                          'access': 'isChecked'}
                                            }
                                            
UM232_PARAM_GUI_DICT = {'dvc_idx': {'field': 'um232DvcIdx',
                                                           'access': 'value'},
                                        'ltncy_tmr_val': {'field': 'um232latencyTimerVal',
                                                                    'access': 'value'},
                                        'in_trnsfr_rate': {'field': 'um232InTransferRate',
                                                                     'access': 'value'},
                                        'read_timeout': {'field': 'um232ReadTimeout',
                                                                  'access': 'value'},
                                        'flow_ctrl': {'field': 'um232FlowControl',
                                                              'access': 'currentText'},
                                        'bit_mode': {'field': 'um232BitMode',
                                                            'access': 'currentText'},
                                        'dvc_dscrp': {'field': 'um232DeviceDescription',
                                                             'access': 'currentText'}
                                        }
                                            
TDC_PARAM_GUI_DICT = {'start_addr': {'field': 'TDCstartAddress',
                                                           'access': 'text'},
                                    'data_vrsn': {'field': 'TDCdataVersion',
                                                          'access': 'text'},
                                    'laser_freq': {'field': 'TDClaserFreq',
                                                           'access': 'value'},
                                    'fpga_freq': {'field': 'TDCFPGAFreq',
                                                         'access': 'value'},
                                    'pxl_clk_freq': {'field': 'TDCpixelClockFreq',
                                                              'access': 'value'},
                                    'tdc_vrsn': {'field': 'TDCversion',
                                                        'access': 'value'},
                                    }

PXL_CLK_PARAM_GUI_DICT = {'low_ticks': {'field': 'pixelClockLowTicks',
                                                           'access': 'value'},
                                         'high_ticks': {'field': 'pixelClockHighTicks',
                                                              'access': 'value'},
                                         'cntr_addr': {'field': 'pixelClockCounterAddress',
                                                              'access': 'text'},
                                         'tick_src': {'field': 'pixelClockSrcOfTicks',
                                                             'access': 'text'},
                                         'freq': {'field': 'pixelClockFreq',
                                                       'access': 'value'},
                                        }

DVC_NICK_PARAMS_DICT = {'COUNTER': COUNTER_PARAM_GUI_DICT,
                                        'UM232': UM232_PARAM_GUI_DICT,
                                        'TDC': TDC_PARAM_GUI_DICT,
                                        'PXL_CLK': PXL_CLK_PARAM_GUI_DICT, 
                                        }
