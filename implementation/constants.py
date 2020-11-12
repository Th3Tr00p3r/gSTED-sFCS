'''
Constants
'''
import gui.icons.icon_paths as icon

# general
EXIT_CODE_REBOOT = -123456789 # for restarting the application
MIN_SHG_TEMP = 52

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

# device nicks
DEVICE_NICKS = ['EXC_LASER', 'DEP_LASER', 'DEP_SHUTTER', 'STAGE']

# device address fields in settings
DEVICE_ADDRSS_FIELD_NAMES = {'EXC_LASER': 'excTriggerExtChan',
                                                'DEP_LASER': 'depPort',
                                                'DEP_SHUTTER': 'depShutterChan',
                                                'STAGE': 'arduinoChan', 
                                                }
# device classes
DEVICE_CLASS_NAMES = {'EXC_LASER': 'ExcitationLaser',
                                     'DEP_LASER': 'DepletionLaser',
                                     'DEP_SHUTTER': 'DepletionShutter',
                                     'STAGE': 'StepperStage', 
                                     }
