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
import gui.gui as gui
from datetime import datetime

class App():
    
    def __init__(self):
        
        #init windows
        self.win = {}
        self.win['main'] = gui.MainWin(self)
        self.win['settings'] = gui.SettWin(self)
        self.win['errors'] = gui.ErrWin(self)
        self.win['camera'] = gui.CamWin(self)
        
        # set restart flag
        self.restart_flag = False
        #initialize active devices
        self.dvcs = {}
        self.init_devices()
        # initialize log
        self.log = Log(self.win['main'], dir_path='./log/')
        
        #FINALLY
        self.win['main'].show()
        self.win['settings'].imp.read_csv(const.DEFAULT_SETTINGS_FILE_PATH)

        # set up main timeout event
        self.timing = self.timing_dict()
        self.timeout_loop = self.timeout(self)
#        self.timeout_loop.timer.start(50)
    
    def timing_dict(self):
        
        # TODO: add here all the different timings for timeout()
        # TODO: should all times be in seconds? or include a 'unit' property (ms, s, etc.)
        d ={}
        d['DEP_LASER'] = self.win['settings'].depUpdateTimeSpinner.value()
        return d
    
    class timeout():
    
        def __init__(self, app):
            self.app = app
            self.timer = QTimer()
            self.timer.timeout.connect(self.main)
        
        def check_SHG_temp(self):
            SHG_temp = self.app.dvcs['DEP_LASER'].get_SHG_temp()
            self.app.win['main'].depTemp.setValue(SHG_temp)
            if SHG_temp < const.MIN_SHG_TEMP:
                self.app.win['main'].depTemp.setStyleSheet("background-color: rgb(255, 0, 0); color: white;")
            else:
                self.app.win['main'].depTemp.setStyleSheet("background-color: white; color: black;")
        
        def check_power(self):
            power = self.app.dvcs['DEP_LASER'].get_power()
            self.app.win['main'].depActualPowerSpinner.setValue(power)
        
        def check_current(self):
            current = self.app.dvcs['DEP_LASER'].get_current()
            self.app.win['main'].depActualCurrSpinner.setValue(current)
        
        def update_dep(self):
            
            self.check_SHG_temp()
            self.check_power()
            self.check_current()
        
        #MAIN
        def main(self):
            self.update_dep()
        
    def init_devices(self):
        '''
        goes through a list of device nicknames, instantiating a driver object for each device
        '''
        for nick in const.DEV_NICKS:
            dev_driver = getattr(drivers,
                                           const.DEV_DRIVERS[nick])
            dev_address = getattr(self.win['settings'],
                                              const.DEV_ADDRSS_FLDS[nick]).text()
            self.dvcs[nick] = dev_driver(nick=nick, address=dev_address)
    
    def clean_up_app(self):
        '''
        Close all devices and secondary windows
        before closing/restarting application
        '''
        self.init_devices() # TODO: just being lazy for now, need to create a new function which only closes all devices, no init needed
        [self.win[win_key].reject() for win_key in self.win.keys() if win_key not in {'main'}]
            
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

class MainWin():
    
    def __init__(self, gui, app):
        
        self.app = app
        self.gui = gui
        
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
        # initialize measurement states
        self.meas_state = {}
        self.meas = None
        self.meas_state['FCS'] = ''
        self.meas_state['sol_sFCS'] = ''
        self.meas_state['img_sFCS'] = ''
    
    def close(self, event):
        
        self.app.exit_app(event)
        
    def restart(self):
        
        self.app.restart_flag = True
        self.gui.close()
    
    def exc_emission_toggle(self):
        
        nick = 'EXC_LASER'
        self.app.dvcs[nick].toggle()
        
        # switch ON
        if self.app.dvcs[nick].state:
            self.gui.excOnButton.setIcon(QIcon(icon.SWITCH_ON))
            self.gui.ledExc.setIcon(QIcon(icon.LED_BLUE))
            self.app.log.update('Excitation ON', tag='verbose')
        # switch OFF
        else:
            self.gui.excOnButton.setIcon(QIcon(icon.SWITCH_OFF))
            self.gui.ledExc.setIcon(QIcon(icon.LED_OFF)) 
            self.app.log.update('Excitation OFF', tag='verbose')

    def dep_emission_toggle(self):
        
        nick = 'DEP_LASER'
        
        # switch ON
        if not self.app.dvcs[nick].state:
            self.app.dvcs[nick].toggle(True)
            self.gui.depEmissionOn.setIcon(QIcon(icon.SWITCH_ON))
            self.gui.ledDep.setIcon(QIcon(icon.LED_ORANGE)) 
            self.app.log.update('Depletion ON', tag='verbose')
        # switch OFF
        else:
            self.app.dvcs[nick].toggle(False)
            self.gui.depEmissionOn.setIcon(QIcon(icon.SWITCH_OFF))
            self.gui.ledDep.setIcon(QIcon(icon.LED_OFF))
            self.app.log.update('Depletion OFF', tag='verbose')

    def dep_shutter_toggle(self):
        
        nick = 'DEP_SHUTTER'
        
        # switch ON
        if not self.app.dvcs[nick].state:
            self.app.dvcs[nick].toggle(True)
            self.app.dvcs[nick].state = True
            self.gui.depShutterOn.setIcon(QIcon(icon.SWITCH_ON))
            self.gui.ledShutter.setIcon(QIcon(icon.LED_GREEN)) 
            self.app.log.update('Depletion shutter OPEN', tag='verbose')
        # switch OFF
        else:
            self.app.dvcs[nick].toggle(False)
            self.app.dvcs[nick].state = False
            self.gui.depShutterOn.setIcon(QIcon(icon.SWITCH_OFF))
            self.gui.ledShutter.setIcon(QIcon(icon.LED_OFF))
            self.app.log.update('Depletion shutter CLOSED', tag='verbose')
    
    def dep_sett_apply(self):
        
        if self.gui.currModeRadio.isChecked(): # current mode
            val = self.gui.depCurrSpinner.value()
            self.app.dvcs['DEP_LASER'].set_current(val)
        else: # power mode
            val = self.gui.depPowSpinner.value()
            self.app.dvcs['DEP_LASER'].set_power(val)
    
    def stage_toggle(self):
        
        nick = 'STAGE'
        
        # switch ON
        if not self.app.dvcs[nick].state:
            self.app.dvcs[nick].open()
            self.gui.stageOn.setIcon(QIcon(icon.SWITCH_ON))
            self.gui.stageButtonsGroup.setEnabled(True)
            self.app.log.update('Stage Control ON', tag='verbose')
        
        # switch OFF
        else:
            self.app.dvcs[nick].close()
            self.gui.stageOn.setIcon(QIcon(icon.SWITCH_OFF))
            self.gui.stageButtonsGroup.setEnabled(False)
            self.app.log.update('Stage Control OFF', tag='verbose')

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
                                                prog_bar=self.gui.FCSprogressBar, log=self.app.log)
            self.meas.start()
            self.gui.startFcsMeasurementButton.setText('Stop \nMeasurement')
        else: # off state
            self.meas_state['FCS'] = False
            self.meas.stop()
            self.gui.startFcsMeasurementButton.setText('Start \nMeasurement')
    
    def open_settwin(self):
        
        self.app.win['settings'].show()
        self.app.win['settings'].activateWindow()
    
    def open_camwin(self):
        
        self.app.win['camera'].show()
        self.app.win['camera'].activateWindow()
        self.app.win['camera'].imp.init_cam()
    
    def open_errwin(self):
        
        self.app.win['errors'].show()
        self.app.win['errors'].activateWindow()
    
class SettWin():
    
    def __init__(self, gui, app):
        self.app = app
        self.gui = gui
    
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
            self.app.log.update('Settings file save as: ' + filepath)

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
                 df = pd.read_csv(filepath, header=None, delimiter=',',
                                         keep_default_na=False, error_bad_lines=False)
                 self.gui.frame.findChild(QWidget, 'settingsFileName').setText(filepath)
                 for i in range(len(df)):
                    widget = self.gui.frame.findChild(QWidget, df.iloc[i, 0])
                    if not widget == 'nullptr':
                        if hasattr(widget, 'value'): # spinner
                            widget.setValue(float(df.iloc[i, 1]))
                        elif hasattr(widget, 'text'): # line edit
                            widget.setText(df.iloc[i, 1])
            except:
                Error(sys.exc_info()).display()
            
            self.app.log.update('Settings file loaded: ' + filepath)

        else:
            error_txt = ('File path not supplied.')
            Error(error_txt=error_txt).display()

class CamWin():
    
    state = {}
    
    def __init__(self, gui, app):
        
        self.app = app
        self.gui = gui
        
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
        self.app.log.update('Camera connection closed', tag='verbose')
        return None

    def init_cam(self):
        
        try:
            self.cam = drivers.Camera() # instantiate camera object
            self.cam.driver.set_auto_exposure()
            self.app.log.update('Camera connection opened', tag='verbose')
        except:
            Error(sys.exc_info()).display()
    
    def toggle_video(self):
        
        #TODO: return the try/except
        # TODO: allow control of refresh period from GUI
        #Turn On
        if not self.state['video']:
            self.state['video'] = 1
            self.gui.videoButton.setStyleSheet("background-color: "
                                                                                 "rgb(225, 245, 225); "
                                                                                 "color: black;")
            self.gui.videoButton.setText('Video ON')
            self._video_timer = QTimer()
            self._video_timer.timeout.connect(self._wait_for_frame)
            self.cam.driver.start_live_video()
            self._video_timer.start(0)  # Run full throttle
            self.app.log.update('Camera video mode ON', tag='verbose')
            
        #Turn Off
        else:
            self.state['video'] = 0
            self.gui.videoButton.setStyleSheet("background-color: "
                                                                                 "rgb(225, 225, 225); "
                                                                                 "color: black;")
            self.gui.videoButton.setText('Start Video')
            self.cam.driver.stop_live_video()
            self._video_timer.stop()
            self.app.log.update('Camera video mode OFF', tag='verbose')

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
        self.app.log.update('Camera photo taken', tag='verbose')
    
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
            self._imshow(img)
            
class ErrWin():
    # TODO: 
    
    def __init__(self, gui, app):
        
        self.app = app

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
    
    def __init__(self, type, duration_spinner, prog_bar=None, log=None):
        # instance attributes
        self.type = type
        self.duration_spinner = duration_spinner
        self.prog_bar = prog_bar
        self.log = log
        self._timer.timeout.connect(self._timer_timeout)
        self.time_passed = 0
        
    # public methods
    def start(self):
        self._timer.start(1000)
        if self.log:
            self.log.update('FCS measurement started')
        
    def stop(self):
        self._timer.stop()
        if self.log:
            self.log.update('FCS measurement stopped')
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
    
    def __init__(self, gui, dir_path):
        
        self.gui = gui
        self.dir_path = dir_path
        try:
            os.mkdir(self.dir_path)
        except:
            pass
        date_str = datetime.now().strftime("%d_%m_%Y")
        self.file_path = self.dir_path + date_str + '.csv'
        self.update('Application Started')
    
    def update(self, log_line, tag='always'):
        
        # add line to log file
        if tag in const.VERBOSITY:
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
