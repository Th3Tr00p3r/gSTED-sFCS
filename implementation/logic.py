import os
import sys
#import time
from PyQt5.QtWidgets import (QWidget, QLineEdit, QSpinBox,
                                              QDoubleSpinBox, QMessageBox, QApplication, 
                                              QFileDialog
                                              )
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QTimer
import implementation.constants as const
import implementation.drivers as drivers
import gui.icons.icon_paths as icon

class App():
    
    actv_dvcs = {}
    meas_state = {}
    meas = None
    
    def clean_up_app(self):
        '''
        Close all secondary windows
        before closing/restarting application
        '''
        [window.reject() for window in self.gui.windows.values()]
            
    def exit_app(self, event):
        '''
        Clean up and exit, ask first.
        '''
        pressed = Question(q_txt='Are you sure you want to quit?', q_title='Quitting Program').display()
        if pressed == QMessageBox.Yes:
            self.clean_up_app()
            self.gui.close()
        else:
            event.ignore()
            
    def restart_app(self):
        '''
        Clean up and restart, ask first.
        '''
        # TODO: restart fails if camera window open (creates second main)
        pressed = Question(q_title='Restarting Program',
                                   q_txt=('Are you sure you want ' +
                                             'to restart the program?')).display()
        if pressed == QMessageBox.Yes:
            self.clean_up_app()
            QApplication.exit(const.EXIT_CODE_REBOOT);

class MainWin(App):
    
    def __init__(self,  main_gui):
        
        self.gui = main_gui
        # set up main timeout event
        self.timer = QTimer()
        self.timer.timeout.connect(self.timeout)
        self.timer.start(10)
        # intialize buttons
        self.gui.actionLaser_Control.setChecked(True)
        self.gui.actionStepper_Stage_Control.setChecked(True)
        self.gui.stageButtonsGroup.setEnabled(False)
        self.gui.actionLog.setChecked(True)
        #connect signals and slots
        self.gui.ledExc.clicked.connect(self.show_laser_dock)
        self.gui.ledDep.clicked.connect(self.show_laser_dock)
        self.gui.ledShutter.clicked.connect(self.show_laser_dock)
        self.gui.actionRestart.triggered.connect(self.restart_app)
        #initialize active devices
        self.actv_dvcs['exc_laser_emisson'] = 0
        self.actv_dvcs['dep_laser_emisson'] = 0
        self.actv_dvcs['dep_shutter'] = 0
        self.actv_dvcs['stage_control'] = 0
        self.actv_dvcs['camera_control'] = 0
        # initialize measurement states
        self.meas_state['FCS'] = ''
        self.meas_state['sol_sFCS'] = ''
        self.meas_state['img_sFCS'] = ''

    def timeout(self):
        '''
        Continuous updates
        '''
        
        def check_SHG_temp(self):
            if self.gui.depTemp.value() < 52:
                self.gui.depTemp.setStyleSheet("background-color: rgb(255, 0, 0); color: white;")
            else:
                self.gui.depTemp.setStyleSheet("background-color: white; color: black;")
        #MAIN
        check_SHG_temp(self)
    
    def exc_emission_toggle(self):
        
        if not self.actv_dvcs['exc_laser_emisson']:
            self.actv_dvcs['exc_laser_emisson'] = 1
            self.gui.excOnButton.setIcon(QIcon(icon.SWITCH_ON))
            self.gui.ledExc.setIcon(QIcon(icon.LED_BLUE))
        else:
            self.actv_dvcs['exc_laser_emisson'] = 0
            self.gui.excOnButton.setIcon(QIcon(icon.SWITCH_OFF))
            self.gui.ledExc.setIcon(QIcon(icon.LED_OFF)) 

    def dep_emission_toggle(self):
        
        if not self.actv_dvcs['dep_laser_emisson']:
            self.actv_dvcs['dep_laser_emisson'] = 1
            self.gui.depEmissionOn.setIcon(QIcon(icon.SWITCH_ON))
            self.gui.ledDep.setIcon(QIcon(icon.LED_ORANGE)) 
        else:
            self.actv_dvcs['dep_laser_emisson'] = 0
            self.gui.depEmissionOn.setIcon(QIcon(icon.SWITCH_OFF))
            self.gui.ledDep.setIcon(QIcon(icon.LED_OFF))

    def dep_shutter_toggle(self):
        
        if not self.actv_dvcs['dep_shutter']:
            self.actv_dvcs['dep_shutter'] = 1
            self.gui.depShutterOn.setIcon(QIcon(icon.SWITCH_ON))
            self.gui.ledShutter.setIcon(QIcon(icon.LED_GREEN)) 
        else:
            self.actv_dvcs['dep_shutter'] = 0
            self.gui.depShutterOn.setIcon(QIcon(icon.SWITCH_OFF))
            self.gui.ledShutter.setIcon(QIcon(icon.LED_OFF))

    def stage_toggle(self):
        
        if not self.actv_dvcs['stage_control']:
            self.actv_dvcs['stage_control'] = 1
            self.gui.stageOn.setIcon(QIcon(icon.SWITCH_ON))
            self.gui.stageButtonsGroup.setEnabled(True)
            self.stage = drivers.StepperStage(rsrc_alias=self.gui.windows['settings'].arduinoChan.text())

        else:
            self.actv_dvcs['stage_control'] = 0
            self.gui.stageOn.setIcon(QIcon(icon.SWITCH_OFF))
            self.gui.stageButtonsGroup.setEnabled(False)
            self.stage = self.stage.clean_up()

    def show_laser_dock(self):
        '''
        Make the laser dock visible (convenience)
        '''
        if not self.gui.laserDock.isVisible():
            self.gui.laserDock.setVisible(True)
            self.gui.actionLaser_Control.setChecked(True)

    def start_FCS_meas(self):
        
        if not self.meas_state['FCS']:
            self.meas = Measurement(type='FCS', 
                                                               duration_spinner=self.gui.measFCSDurationSpinBox,
                                                               prog_bar=self.gui.FCSprogressBar)
            self.meas.start()
            self.gui.startFcsMeasurementButton.setText('Stop \nMeasurement')
        else: # off state
            self.meas.stop()
            self.gui.startFcsMeasurementButton.setText('Start \nMeasurement')

class SettingsWin():
    
    def __init__(self, settings_gui):
        self.gui = settings_gui
    
    def clean_up(self):
        # TODO: add check to see if changes were made, if not, don't ask user
        pressed = Question('Keep changes if made? ' +
                                  '(otherwise, revert to last loaded settings file.)'
                                  ).display()
        if pressed == QMessageBox.No:
            self.read_csv(self.gui.settingsFileName.text())

    # public methods
    def write_csv(self):
        '''
        Write all QLineEdit, QspinBox and QdoubleSpinBox of settings window to 'filepath' (csv).
        Show 'filepath' in 'settingsFileName' QLineEdit.
        '''
        filepath, _ = QFileDialog.getSaveFileName(self.gui,
                                                                 'Save Settings',
                                                                 const.SETTINGS_FOLDER_PATH,
                                                                 "CSV Files(*.csv *.txt)")
        import csv
        if filepath:
            self.gui.frame.findChild(QWidget, 'settingsFileName').setText(filepath)
            with open(filepath, 'w') as stream:
                #print("saving", filepath)
                writer = csv.writer(stream)
                # get all names of fields in settings window (for file saving/loading)
                l1 = self.gui.frame.findChildren(QLineEdit)
                l2 = self.gui.frame.findChildren(QSpinBox)
                l3 = self.gui.frame.findChildren(QDoubleSpinBox)
                field_names = [w.objectName() for w in (l1 + l2 + l3) if (not w.objectName() == 'qt_spinbox_lineedit') and (not w.objectName() == 'settingsFileName')]
                #print(fieldNames)
                for i in range(len(field_names)):
                    widget = self.gui.frame.findChild(QWidget, field_names[i])
                    if hasattr(widget, 'value'): # spinner
                        rowdata = [field_names[i],  self.gui.frame.findChild(QWidget, field_names[i]).value()]
                    else: # line edit
                        rowdata = [field_names[i],  self.gui.frame.findChild(QWidget, field_names[i]).text()]
                    writer.writerow(rowdata)

    def read_csv(self, filepath=''):
        '''
        Read 'filepath' (csv) and write to matching QLineEdit, QspinBox and QdoubleSpinBox of settings window.
        Show 'filepath' in 'settingsFileName' QLineEdit.
        '''
        if not filepath:
            filepath, _ = QFileDialog.getOpenFileName(self.gui,
                                                                     "Load Settings",
                                                                     const.SETTINGS_FOLDER_PATH,
                                                                     "CSV Files(*.csv *.txt)")
        import pandas as pd
        if filepath:
            try:
                 df = pd.read_csv(filepath, header=None, delimiter=',', keep_default_na=False, error_bad_lines=False)
                 self.gui.frame.findChild(QWidget, 'settingsFileName').setText(filepath)
                 for i in range(len(df)):
                    widget = self.gui.frame.findChild(QWidget, df.iloc[i, 0])
                    if not widget == 'nullptr':
                        if hasattr(widget, 'value'): # spinner
                            widget.setValue(float(df.iloc[i, 1]))
                        else: # line edit
                            widget.setText(df.iloc[i, 1])
            except: # handeling missing default settings file
                error_txt = ('Default settings file "default_settings.csv" '
                                   'not found in' + '\n'
                                   '\'' + filepath + '\''
                                   '\nUsing standard settings.' + '\n'
                                   'To avoid this error, please save '
                                   'some settings as "default_settings.csv".')
                Error(sys.exc_info(), error_txt=error_txt).display()

        else:
            error_txt = ('File path not supplied.')
            Error(error_txt=error_txt).display()

class CamWin():
    
    __video_timer = QTimer()
    state = {}
    
    def __init__(self, cam_gui):
        
        self.gui = cam_gui
        # add matplotlib-ready widget (canvas) for showing camera output
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        self.gui.figure = plt.figure()
        self.gui.canvas = FigureCanvas(self.gui.figure)
        self.gui.gridLayout.addWidget(self.gui.canvas, 0, 1)
        # initialize states
        self.state['video'] = 0

    def clean_up(self):
        '''
        clean up before closing window
        '''
        # turn off video if On
        if self.state['video']:
            self.toggle_video()
        # disconnect camera
        self.cam.close()
        return None

    def init_cam(self):
        
        try:
            self.cam = drivers.Camera(reopen_policy='new') # instantiate camera object
            self.cam.open() # connect to first available camera
        except:
            error_txt = ('No cameras appear to be connected.')
            Error(error_txt=error_txt).display()
    
    def toggle_video(self):
        #TODO: return the try/except
        #Turn On
        if not self.state['video']:
            self.state['video'] = 1
            self.gui.videoButton.setStyleSheet("background-color: rgb(225, 245, 225); color: black;")
            self.gui.videoButton.setText('Video ON')
            self.cam.start_live_video()
            self.__video_timer.timeout.connect(self.__video_timeout)
            self.__video_timer.start(50) # video refresh period (ms)
        #Turn Off
        else:
            self.state['video'] = 0
            self.gui.videoButton.setStyleSheet("background-color: rgb(225, 225, 225); color: black;")
            self.gui.videoButton.setText('Start Video')
            self.cam.stop_live_video()
            self.__video_timer.stop()

    def shoot(self):
        #TODO: return the try/except
        img = self.cam.grab_image()
        self.__imshow(img)
    
    # private methods
    def __imshow(self, img):
        '''
        Plot image
        '''
        self.gui.figure.clear()
        ax = self.gui.figure.add_subplot(111)
        ax.imshow(img)
        self.gui.canvas.draw()
    
    def __video_timeout(self):
        #TODO: return the try/except
        img = self.cam.grab_image()
        self.__imshow(img)

class UserDialog():
    
    def __init__(self,
                       msg_icon=QMessageBox.NoIcon,
                       msg_title=None,
                       msg_text=None,
                       msg_inf=None,
                       msg_det=None):
        self._msg_box = QMessageBox()
        self._msg_box.setIcon(msg_icon)
        self._msg_box.setWindowTitle(msg_title)
        self._msg_box.setText(msg_text)
        self._msg_box.setInformativeText(msg_inf)
        self._msg_box.setDetailedText(msg_det)
    
    def set_buttons(self, std_bttn_list):
        for std_bttn in std_bttn_list:
            self._msg_box.addButton(getattr(QMessageBox, std_bttn))
        
    def display(self):
        return self._msg_box.exec_()
    
class Error(UserDialog):
    
    def __init__(self, error_info='', error_txt='Error occured.'):
        import traceback
        if error_info:
            self.exc_type, _, self.tb = error_info
            self.error_type = self.exc_type.__name__
            self.fname = os.path.split(self.tb.tb_frame.f_code.co_filename)[1]
            self.lineno = self.tb.tb_lineno
            super().__init__(msg_icon=QMessageBox.Critical,
                                   msg_title='Error',
                                   msg_text=error_txt,
                                   msg_inf=('{}, {}' +
                                                    ', line: {}').format(self.error_type,
                                                                                      self.fname,
                                                                                      self.lineno),
                                    msg_det='\n'.join(traceback.format_tb(self.tb))
                                  )
        else:
            super().__init__(msg_icon=QMessageBox.Critical,
                                   msg_title='Error',
                                   msg_text=error_txt,
                                  )

class Question(UserDialog):
    def __init__(self, q_txt,  q_title='User Input Needed'):
        super().__init__(msg_icon=QMessageBox.Question,
                               msg_title=q_title,
                               msg_text=q_txt
                               )
        self.set_buttons(['Yes', 'No'])
        self._msg_box.setDefaultButton(QMessageBox.No)

class Measurement():
    
    # class attributes
    __timer = QTimer()
    
    def __init__(self, type, duration_spinner, prog_bar=None):
        # instance attributes
        self.type = type
        self.duration_spinner = duration_spinner
        self.prog_bar = prog_bar
        self.__timer.timeout.connect(self.__timer_timeout)
        self.time_passed = 0
        
    # public methods
    def start(self):
        self.__timer.start(1000)
        
    def stop(self):
        self.__timer.stop()
        if self.prog_bar:
            self.prog_bar.setValue(0)
    
    # private methods
    def __timer_timeout(self):
#        from PyQt5.QtMultimedia import QSound
        if self.time_passed < self.duration_spinner.value():
            self.time_passed += 1
#            if self.time_passed == self.duration_spinbox.value():
#                QSound.play(const.MEAS_COMPLETE_SOUND);
        else:
            self.time_passed = 1
        if self.prog_bar:
            prog = self.time_passed / self.duration_spinner.value() * 100
            self.prog_bar.setValue(prog)
class LogDock():
    
    def __init__(self):
        '''
        ready and show the log window
        '''
        try:
            # load today's log file, add "program started" etc.
            print('Log Dock Initialized') # TEST
            pass
        except:
            error_txt = ('')
            Error(sys.exc_info(), error_txt=error_txt).display()
