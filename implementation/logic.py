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
from datetime import datetime

class App():
    
    meas_state = {}
    meas = None
    
    def clean_up_app(self):
        '''
        Close all secondary windows
        before closing/restarting application
        '''
        [window.reject() for window in self.gui.windows.values()]
            
    def exit_app(self, event):
        
        if not self.restart_flag: # closing
            pressed = Question(q_txt='Are you sure you want to quit?', q_title='Quitting Program').display()
            if pressed == QMessageBox.Yes:
                self.log.update('Quitting Application')
                self.clean_up_app()
                event.ignore()
                QApplication.exit(0)
            else:
                event.ignore()
        else: # restarting
            # TODO: can only restart once for some reason
            pressed = Question(q_title='Restarting Program',
                                   q_txt=('Are you sure you want ' +
                                             'to restart the program?')).display()
            if pressed == QMessageBox.Yes:
                self.log.update('Restarting Application')
                self.clean_up_app()
                event.ignore()
                QApplication.exit(const.EXIT_CODE_REBOOT)
            else:
                event.ignore()

class MainWin(App):
    
    def __init__(self,  main_gui):
        
        self.gui = main_gui
        # set restart flag
        self.restart_flag = False
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
        self.gui.actionRestart.triggered.connect(self.restart)
        #initialize active devices
        self.actv_dvcs = {}
        # TODO: organize in a single method - make a list of all devices that need initialization/testing (get channels/addresses from settings) and make a 'for' loop
        # exc laser
#        dvc = drivers.ExcitationLaser(address=self.gui.windows['settings'].excTriggerExtChan.text())
#        self.actv_dvcs[dvc.name] = dvc
#        # dep shutter
#        dvc = drivers.DepletionShutter(address=self.gui.windows['settings'].depShutterChan.text())
#        self.actv_dvcs[dvc.name] = dvc
        
        self.actv_dvcs['dep_laser'] = 'READY'
        self.actv_dvcs['stage'] = 'OFF'
        self.actv_dvcs['camera'] = 'OFF'
        # initialize measurement states
        self.meas_state['FCS'] = ''
        self.meas_state['sol_sFCS'] = ''
        self.meas_state['img_sFCS'] = ''
        # initialize log dock
        self.log = Log(self.gui)
    
    def restart(self):
        self.restart_flag = True
        self.gui.close()

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
        
        self.actv_dvcs[const.EXC_LASER_NAME].toggle()
        
        # switch ON
        if self.actv_dvcs[const.EXC_LASER_NAME].state == 'ON':
            self.gui.excOnButton.setIcon(QIcon(icon.SWITCH_ON))
            self.gui.ledExc.setIcon(QIcon(icon.LED_BLUE))
            self.log.update('Excitation ON')
        # switch OFF
        else:
            self.gui.excOnButton.setIcon(QIcon(icon.SWITCH_OFF))
            self.gui.ledExc.setIcon(QIcon(icon.LED_OFF)) 
            self.log.update('Excitation OFF')

    def dep_emission_toggle(self):
        
        # switch ON
        if self.actv_dvcs['dep_laser'] == 'READY':
            self.gui.depEmissionOn.setIcon(QIcon(icon.SWITCH_ON))
            self.gui.ledDep.setIcon(QIcon(icon.LED_ORANGE)) 
            self.actv_dvcs['dep_laser_emisson'] = 'ON'
            self.log.update('Depletion ON')
        # switch OFF
        else:
            self.gui.depEmissionOn.setIcon(QIcon(icon.SWITCH_OFF))
            self.gui.ledDep.setIcon(QIcon(icon.LED_OFF))
            self.actv_dvcs['dep_laser'] = 'READY'
            self.log.update('Depletion OFF')

    def dep_shutter_toggle(self):
        
        self.actv_dvcs[const.DEP_SHUTTER_NAME].toggle()
        
        # switch ON
        if self.actv_dvcs['dep_shutter'] == 'OPEN':
            self.gui.depShutterOn.setIcon(QIcon(icon.SWITCH_ON))
            self.gui.ledShutter.setIcon(QIcon(icon.LED_GREEN)) 
            self.log.update('Depletion shutter OPEN')
        # switch OFF
        else:
            self.gui.depShutterOn.setIcon(QIcon(icon.SWITCH_OFF))
            self.gui.ledShutter.setIcon(QIcon(icon.LED_OFF))
            self.log.update('Depletion shutter CLOSED')

    def stage_toggle(self):
        
        # switch ON
        if self.actv_dvcs['stage_control'] == 'OFF':
            self.stage = drivers.StepperStage(rsrc_alias=self.gui.windows['settings'].arduinoChan.text())
            self.gui.stageOn.setIcon(QIcon(icon.SWITCH_ON))
            self.gui.stageButtonsGroup.setEnabled(True)
            self.actv_dvcs['stage_control'] = 'ON'
            self.log.update('Stage Control ON')
        
        # switch OFF
        else:
            self.stage = self.stage.clean_up()
            self.gui.stageOn.setIcon(QIcon(icon.SWITCH_OFF))
            self.gui.stageButtonsGroup.setEnabled(False)
            self.actv_dvcs['stage_control'] = 0
            self.log.update('Stage Control OFF')

    def show_laser_dock(self):
        '''
        Make the laser dock visible (convenience)
        '''
        if not self.gui.laserDock.isVisible():
            self.gui.laserDock.setVisible(True)
            self.gui.actionLaser_Control.setChecked(True)

    def start_FCS_meas(self):
        
        if not self.meas_state['FCS']:
            self.meas_state['FCS'] = True
            self.meas = Measurement(type='FCS', 
                                                duration_spinner=self.gui.measFCSDurationSpinBox,
                                                prog_bar=self.gui.FCSprogressBar)
            self.meas.start()
            self.gui.startFcsMeasurementButton.setText('Stop \nMeasurement')
            self.log.update('FCS measurement started.')
        else: # off state
            self.meas_state['FCS'] = False
            self.meas.stop()
            self.gui.startFcsMeasurementButton.setText('Start \nMeasurement')
            self.log.update('FCS measurement stopped.')

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
        # TODO: add support for combo box index
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
        # TODO: add support for combo box index
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
                        elif hasattr(widget, 'text'): # line edit
                            widget.setText(df.iloc[i, 1])
            except:
#                error_txt = ('Default settings file "default_settings.csv" '
#                                   'not found in' + '\n'
#                                   '\'' + filepath + '\''
#                                   '\nUsing standard settings.' + '\n'
#                                   'To avoid this error, please save '
#                                   'some settings as "default_settings.csv".')
                Error(sys.exc_info()).display()

        else:
            error_txt = ('File path not supplied.')
            Error(error_txt=error_txt).display()

class CamWin():
    
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
        self.cam.driver.close()
        return None

    def init_cam(self):
        
        try:
            self.cam = drivers.Camera() # instantiate camera object
            self.cam.driver.set_auto_exposure()
        except:
            Error(sys.exc_info()).display()
    
    def toggle_video(self):
        #TODO: return the try/except
        # TODO: allow control of refresh period from GUI
        #Turn On
        if not self.state['video']:
            self.state['video'] = 1
            self.gui.videoButton.setStyleSheet("background-color: rgb(225, 245, 225); color: black;")
            self.gui.videoButton.setText('Video ON')
            self._video_timer = QTimer()
            self._video_timer.timeout.connect(self._wait_for_frame)
            self.cam.driver.start_live_video()
            self._video_timer.start(0)  # Run full throttle
            
        #Turn Off
        else:
            self.state['video'] = 0
            self.gui.videoButton.setStyleSheet("background-color: rgb(225, 225, 225); color: black;")
            self.gui.videoButton.setText('Start Video')
            self.cam.driver.stop_live_video()
            self._video_timer.stop()

    def shoot(self):
        #TODO: return the try/except
        if self.state['video']:
            self.toggle_video()
            img = self.cam.driver.grab_image()
            self._imshow(img)
            self.toggle_video()
        else:
            img = self.cam.driver.grab_image()
            self._imshow(img)
    
    # private methods
    def _imshow(self, img):
        '''
        Plot image
        '''
        self.gui.figure.clear()
        ax = self.gui.figure.add_subplot(111)
        ax.imshow(img)
        self.gui.canvas.draw()
    
    def _wait_for_frame(self):
        frame_ready = self.cam.driver.wait_for_frame(timeout='0 ms')
        if frame_ready:
            img = self.cam.driver.latest_frame(copy=False)
#            self._set_pixmap_from_array(arr)
            self._imshow(img)

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
    _timer = QTimer()
    
    def __init__(self, type, duration_spinner, prog_bar=None):
        # instance attributes
        self.type = type
        self.duration_spinner = duration_spinner
        self.prog_bar = prog_bar
        self._timer.timeout.connect(self._timer_timeout)
        self.time_passed = 0
        
    # public methods
    def start(self):
        self._timer.start(1000)
        
    def stop(self):
        self._timer.stop()
        if self.prog_bar:
            self.prog_bar.setValue(0)
    
    # private methods
    def _timer_timeout(self):
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
class Log():
    
    def __init__(self, gui):
        
        self.gui = gui
        self.dir_path = self.gui.windows['settings'].logDataPath.text()
        try:
            os.mkdir(self.dir_path)
        except:
            pass
        date_str = datetime.now().strftime("%d_%m_%Y")
        self.file_path = self.dir_path + date_str + '.csv'
        self.update('Application Started')
    
    def update(self, log_line):
        
        # add line to log file
        with open(self.file_path, 'a+') as file:
            time_str = datetime.now().strftime("%H:%M:%S")
            file.write(time_str + ' ' + log_line + '\n')
        # read log file to log dock
        self.read_log()
        # read last line
        self.gui.lastActionLineEdit.setText(self.get_last_line())
    
    def read_log(self):
        
        with open(self.file_path, 'r') as file:
            log = []
            for line in reversed(list(file)):
                log.append(line.rstrip())
            log = '\n'.join(log)
        self.gui.logText.setPlainText(log)
    
    def get_last_line(self):
        
        with open(self.file_path, 'r') as file:
            for line in reversed(list(file)):
                line = line.rstrip()
                break
        return line
