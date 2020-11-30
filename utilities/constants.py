'''
Constants
'''
import gui.icons.icon_paths as icon

# general
TIMEOUT = 10

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
                   'STAGE': {'LED': 'ledStage',
                                  'SWITCH': 'stageOn',
                                  'ICON': icon.LED_GREEN
                                  },
                    'COUNTER': {'LED': 'ledCounter',
                                      'ICON': icon.LED_GREEN
                                  },
                    'UM232': {'LED': 'ledUm232',
                                   'ICON': icon.LED_GREEN
                                  },
                    'TDC': {'LED': 'ledTdc',
                                'ICON': icon.LED_GREEN
                                  },
                    'CAMERA': {'LED': 'ledCam',
                                  'ICON': icon.LED_GREEN
                                  },
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
#                        from PyQt5.QtMultimedia import QSound
#                                if self.time_passed == self.duration_spinbox.value():
#                                    QSound.play(const.MEAS_COMPLETE_SOUND);

# devices
DEVICE_NICKS = {'COUNTER', 'UM232', 'EXC_LASER', 'DEP_LASER', 'DEP_SHUTTER', 'STAGE', 'CAMERA',  'TDC'}

DEVICE_ADDRESS_GUI_DICT = {'EXC_LASER': 'excTriggerExtChan',
                                            'DEP_LASER': 'depPort',
                                            'DEP_SHUTTER': 'depShutterChan',
                                            'STAGE': 'arduinoChan', 
                                            }
# device classes
DEVICE_CLASS_NAMES = {'EXC_LASER': 'SimpleDO',
                                     'TDC': 'SimpleDO',
                                     'DEP_SHUTTER': 'SimpleDO',
                                     'DEP_LASER': 'DepletionLaser',
                                     'STAGE': 'StepperStage',
                                     'COUNTER': 'Counter',
                                     'UM232': 'UM232',
                                     'CAMERA': 'Camera'
                                     }

EXC_LASER_PARAM_GUI_DICT = {'model': {'field': 'excMod',
                                                             'access': 'text'},
                                              'trg_src': {'field': 'excTriggerSrc',
                                                                'access': 'currentText'},
                                              'ext_trg_addr': {'field': 'excTriggerExtAddr',
                                                                        'access': 'text'},
                                              'int_trg_addr': {'field': 'excTriggerIntAddr',
                                                                        'access': 'text'},
                                              'addr': {'field': 'excAddr',
                                                           'access': 'text'}
                                            }

DEP_LASER_PARAM_GUI_DICT = {'model': {'field': 'depMod',
                                                             'access': 'text'},
                                              'update_time': {'field': 'depUpdateTime',
                                                                       'access': 'value'},
                                              'addr': {'field': 'depAddr',
                                                           'access': 'text'}
                                            }

DEP_SHUTTER_PARAM_GUI_DICT = {'addr': {'field': 'depShutterAddr',
                                                               'access': 'text'}
                                                }

STAGE_PARAM_GUI_DICT = {'addr': {'field': 'arduinoAddr',
                                                     'access': 'text'}
                                      }

COUNTER_PARAM_GUI_DICT = {'buff_sz': {'field': 'counterBufferSizeSpinner',
                                                               'access': 'value'},
                                           'update_time': {'field': 'counterUpdateTime',
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

UM232_PARAM_GUI_DICT = {'vend_id': {'field': 'um232VendID',
                                                           'access': 'value'},
                                        'prod_id': {'field': 'um232ProdID',
                                                           'access': 'value'},
                                        'dvc_dscrp': {'field': 'um232DeviceDescription',
                                                             'access': 'text'}, 
                                        'ltncy_tmr_val': {'field': 'um232latencyTimerVal',
                                                                    'access': 'value'},
                                        'baud_rate': {'field': 'um232BaudRate',
                                                                     'access': 'value'},
                                        'read_timeout': {'field': 'um232ReadTimeout',
                                                                  'access': 'value'},
                                        'flow_ctrl': {'field': 'um232FlowControl',
                                                              'access': 'text'},
                                        'bit_mode': {'field': 'um232BitMode',
                                                            'access': 'text'},
                                        'n_bytes': {'field': 'um232nBytes',
                                                          'access': 'value'}
                                        }

TDC_PARAM_GUI_DICT = {'addr': {'field': 'TDCaddress',
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

DVC_NICK_PARAMS_DICT = {'EXC_LASER': EXC_LASER_PARAM_GUI_DICT,
                                        'DEP_SHUTTER': DEP_SHUTTER_PARAM_GUI_DICT,
                                        'DEP_LASER': DEP_LASER_PARAM_GUI_DICT,
                                        'STAGE': STAGE_PARAM_GUI_DICT,
                                        'COUNTER': COUNTER_PARAM_GUI_DICT,
                                        'UM232': UM232_PARAM_GUI_DICT,
                                        'TDC': TDC_PARAM_GUI_DICT,
                                        'PXL_CLK': PXL_CLK_PARAM_GUI_DICT, 
                                        }
