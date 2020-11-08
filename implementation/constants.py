'''
Constants
'''

# general
EXIT_CODE_REBOOT = -123456789
min_SHG_temp = 52

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
DEV_NICKS = ['EXC_LASER', 'DEP_LASER', 'DEP_SHUTTER', 'STAGE']

# device address fields in settings
DEV_ADDRSS_FLDS = {'EXC_LASER': 'excTriggerExtChan',
                                  'DEP_LASER': 'depPort',
                                  'DEP_SHUTTER': 'depShutterChan',
                                  'STAGE': 'arduinoChan', 
                                  }
# device classes
DEV_DRIVERS = {'EXC_LASER': 'ExcitationLaser',
                            'DEP_LASER': 'DepletionLaser',
                            'DEP_SHUTTER': 'DepletionShutter',
                            'STAGE': 'StepperStage', 
                            }
