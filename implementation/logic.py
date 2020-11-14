'''
Logic Module.
'''

# PyQt5 imports
from PyQt5.QtWidgets import (QWidget, QLineEdit, QSpinBox,
                                              QDoubleSpinBox, QMessageBox, QFileDialog)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QTimer
#implementation imports
import implementation.constants as const
import implementation.devices as devices
from implementation.dialog import Error, Question
from implementation.log import Log
# GUI imports
import gui.icons.icon_paths as icon
import gui.gui as gui_module
# camera error
from instrumental.drivers.cameras.uc480 import UC480Error

class App():
    
    def __init__(self):
        
        #init windows
        self.win = {}
        self.win['main'] = gui_module.MainWin(self)
        self.win['settings'] = gui_module.SettWin(self)
        self.win['errors'] = gui_module.ErrWin(self)
        self.win['camera'] = None # instantiated on pressing camera button
        # initialize error dict
        self.init_errors()
        #initialize active devices
        self.init_devices()
        # initialize log
        self.log = Log(self.win['main'], dir_path='./log/')
        
        #FINALLY
        self.win['main'].show()
        self.win['settings'].imp.read_csv(const.DEFAULT_SETTINGS_FILE_PATH)

        # set up main timeout event
        self.timeout_loop = self.Timeout(self)
    
    class Timeout():
        
        def __init__(self, app):
            self._app = app            
            self._timers = self._get_timers()
            self._main_timer = QTimer()
            self._main_timer.timeout.connect(self._main)
            
            self._start_all_timers()
        
        def _get_timings_dict(self):
                
                '''Associate an (interval (ms), update_function) tuple with each device'''
                # TODO: possibly have a single or a few longer interval timers, not one for each (could be resource heavy)
                
                sett_gui = self._app.win['settings']
                t_dict ={}
                t_dict['DEP_LASER'] = (sett_gui.depUpdateTimeSpinner.value(),
                                                 self._update_dep)
                t_dict['COUNTER'] = (sett_gui.counterUpdateTimeSpinner.value(),
                                              self._update_counter)
                t_dict['ERRORS'] = (1, self._update_errorGUI)
                return t_dict
        
        def _get_timers(self):
            
            '''Instantiate specific timers for devices'''
            
            timers_dict = {}
            for key, value in self._get_timings_dict().items():
                timers_dict[key] = QTimer()
                timers_dict[key].setInterval(1000 * value[0]) # values are is seconds
                timers_dict[key].timeout.connect(value[1])
                
            return timers_dict
        
        def _start_all_timers(self):
            
            # initiate all update functions once
            update_func_dict = self._get_timings_dict()
            for key, value in update_func_dict.items():
                update_func_dict[key][1]()
                
            # then start individual timers
            self._main_timer.start(50)
            for key in self._timers.keys():
                self._timers[key].start()
        
        def _update_counter(self):
            
            pass
        
        def _update_dep(self):
            
            '''Update depletion laser GUI'''
            
            nick = 'DEP_LASER'
            
            def check_SHG_temp(self):
                self._app.dvcs[nick].get_SHG_temp()
                self._app.win['main'].depTemp.setValue(self._app.dvcs[nick].temp)
                if self._app.dvcs[nick].temp < const.MIN_SHG_TEMP:
                    self._app.win['main'].depTemp.setStyleSheet("background-color: rgb(255, 0, 0); color: white;")
                else:
                    self._app.win['main'].depTemp.setStyleSheet("background-color: white; color: black;")
            
            def check_power(self):
                self._app.dvcs[nick].get_power()
                self._app.win['main'].depActualPowerSpinner.setValue(self._app.dvcs[nick].power)
            
            def check_current(self):
                self._app.dvcs[nick].get_current()
                self._app.win['main'].depActualCurrSpinner.setValue(self._app.dvcs[nick].current)
            
            if not self._app.error_dict[nick]: # if there's no errors
            
                check_SHG_temp(self)
                check_power(self)
                check_current(self)
        
        def _update_errorGUI(self):
            # TODO: update the error GUI according to errors in self._app.error_dict
            pass
        
        #MAIN
        def _main(self):
            pass
    
    def init_errors(self):
        
        self.error_dict = {}
        for nick in const.DEVICE_NICKS:
            self.error_dict[nick] = None
    
    def init_devices(self):
        
        '''
        goes through a list of device nicknames,
        instantiating a driver object for each device
        '''
        
        def get_counter_params(app):
            
            '''
            Get counter parameters from settings GUI
            using a dictionary predefined in constants.py
            '''
            
            gui_dict = const.COUNTER_PARAM_GUI_DICT
            param_dict = {}
            
            for key, val_dict in gui_dict.items():
                gui_field = getattr(app.win['settings'], val_dict['field'])
                gui_field_value = getattr(gui_field, val_dict['access'])()
                param_dict[key] = gui_field_value
            return param_dict
        
        self.dvcs = {}
        for nick in const.DEVICE_NICKS:
            if nick in {'COUNTER'}:
                dvc_class = getattr(devices,
                                            const.DEVICE_CLASS_NAMES[nick])
                self.dvcs[nick] = dvc_class(param_dict=get_counter_params(self),
                                                      error_dict=self.error_dict)
            elif nick in {'EXC_LASER', 'DEP_LASER', 'DEP_SHUTTER', 'STAGE'}:
                dvc_class = getattr(devices,
                                            const.DEVICE_CLASS_NAMES[nick])
                dvc_address = getattr(self.win['settings'],
                                                  const.DEVICE_ADDRSS_FIELD_NAMES[nick]).text()
                self.dvcs[nick] = dvc_class(address=dvc_address, error_dict=self.error_dict)
    
    def clean_up_app(self):
        
        '''
        Close all devices and secondary windows
        before closing/restarting application
        '''
        
        def close_all_dvcs(app):
            
            for nick in const.DEVICE_NICKS:
                if not self.error_dict[nick]:
                    app.dvcs[nick].toggle(False)
                
        def close_all_wins(app):
            
            for win_key in self.win.keys():
                if self.win[win_key]: # if not None (can happen for camwin
                    if win_key not in {'main', 'camera'}: # dialogs close with reject()
                        self.win[win_key].reject()
                    else:
                        self.win[win_key].close() # mainwindows and widgets close with close()
        
        close_all_dvcs(self)
        close_all_wins(self)
        self.log.update('Quitting Application')
    
    def exit_app(self, event):
        
        pressed = Question(q_txt='Are you sure you want to quit?', q_title='Quitting Program').display()
        if pressed == QMessageBox.Yes:
            self.clean_up_app()
        else:
            event.ignore()

class MainWin():
    
    def __init__(self, gui, app):
        
        self._app = app
        self._gui = gui
        
        # intialize buttons
        self._gui.actionLaser_Control.setChecked(True)
        self._gui.actionStepper_Stage_Control.setChecked(True)
        self._gui.stageButtonsGroup.setEnabled(False)
        self._gui.actionLog.setChecked(True)
        #connect signals and slots
        self._gui.ledExc.clicked.connect(self.show_laser_dock)
        self._gui.ledDep.clicked.connect(self.show_laser_dock)
        self._gui.ledShutter.clicked.connect(self.show_laser_dock)
        self._gui.actionRestart.triggered.connect(self.restart)
        # initialize measurement states
        self.meas_state = {}
        self.meas = None
        self.meas_state['FCS'] = ''
        self.meas_state['sol_sFCS'] = ''
        self.meas_state['img_sFCS'] = ''
    
    def close(self, event):
        
        self._app.exit_app(event)
        
    def restart(self):
        
        '''Restart all devices (except camera)'''
        # TODO: no need to restart the application, only re-initiate the devices (I think).
        # this function was changed, clean up the application restart stuff from the rest of the code.
        
        def lights_out(self):
            
            '''turn OFF all device switches and LED icons'''
            
            self._gui.excOnButton.setIcon(QIcon(icon.SWITCH_OFF))
            self._gui.ledExc.setIcon(QIcon(icon.LED_OFF))
            self._gui.depEmissionOn.setIcon(QIcon(icon.SWITCH_OFF))
            self._gui.ledDep.setIcon(QIcon(icon.LED_OFF))
            self._gui.depShutterOn.setIcon(QIcon(icon.SWITCH_OFF))
            self._gui.ledShutter.setIcon(QIcon(icon.LED_OFF))
            self._gui.stageOn.setIcon(QIcon(icon.SWITCH_OFF))
            self._gui.stageButtonsGroup.setEnabled(False)
        
        self._app.init_devices()
        self.timeout_loop = self._app.Timeout(self._app)
        self._app.log.update('Restarting...', tag='verbose')
    
    def dvc_toggle(self, nick):
        
        gui_switch_object = getattr(self._gui, const.ICON_DICT[nick]['SWITCH'])
        # switch ON
        if not self._app.dvcs[nick].state:
            self._app.dvcs[nick].toggle(True)
            if self._app.dvcs[nick].state: # if managed to turn ON
                gui_switch_object.setIcon(QIcon(icon.SWITCH_ON))
                if 'LED' in const.ICON_DICT[nick].keys():
                    gui_led_object = getattr(self._gui, const.ICON_DICT[nick]['LED'])
                    on_icon = QIcon(const.ICON_DICT[nick]['ICON'])
                    gui_led_object.setIcon(on_icon)
                self._app.log.update(const.LOG_DICT[nick] +
                                            ' toggled ON', tag='verbose')
            return True
        # switch OFF
        else:
            self._app.dvcs[nick].toggle(False)
            if not self._app.dvcs[nick].state: # if managed to turn OFF
                gui_switch_object.setIcon(QIcon(icon.SWITCH_OFF))
                if 'LED' in const.ICON_DICT[nick].keys():
                    gui_led_object = getattr(self._gui, const.ICON_DICT[nick]['LED'])
                    gui_led_object.setIcon(QIcon(icon.LED_OFF)) 
                self._app.log.update(const.LOG_DICT[nick] +
                                        ' toggled OFF', tag='verbose')
            return False
    
    def dep_sett_apply(self):
        
        if self._gui.currModeRadio.isChecked(): # current mode
            val = self._gui.depCurrSpinner.value()
            self._app.dvcs['DEP_LASER'].set_current(val)
        else: # power mode
            val = self._gui.depPowSpinner.value()
            self._app.dvcs['DEP_LASER'].set_power(val)
    
    def move_stage(self, dir, steps):
        
        nick = 'STAGE'
        self._app.dvcs[nick].move(dir=dir, steps=steps)
        self._app.log.update(const.LOG_DICT[nick] +
                                    ' moved ' + str(steps) +
                                    ' steps ' + str(dir), tag='verbose')
    
    def release_stage(self):
        
        nick = 'STAGE'
        self._app.dvcs[nick].release()
        self._app.log.update(const.LOG_DICT[nick] +
                                    ' released', tag='verbose')

    def show_laser_dock(self):
        '''
        Make the laser dock visible (convenience)
        '''
        if not self._gui.laserDock.isVisible():
            self._gui.laserDock.setVisible(True)
            self._gui.actionLaser_Control.setChecked(True)

    def start_FCS_meas(self):
        
        if not self.meas_state['FCS']:
            self.meas_state['FCS'] = True
            self.meas = Measurement(type='FCS', 
                                                duration_spinner=self._gui.measFCSDurationSpinBox,
                                                prog_bar=self._gui.FCSprogressBar, log=self.app.log)
            self.meas.start()
            self._gui.startFcsMeasurementButton.setText('Stop \nMeasurement')
        else: # off state
            self.meas_state['FCS'] = False
            self.meas.stop()
            self._gui.startFcsMeasurementButton.setText('Start \nMeasurement')
    
    def open_settwin(self):
        
        self._app.win['settings'].show()
        self._app.win['settings'].activateWindow()
    
    def open_camwin(self):
        
        self._gui.actionCamera_Control.setEnabled(False)
        self._app.win['camera'] = gui_module.CamWin(app=self._app)
        self._app.win['camera'].show()
        self._app.win['camera'].activateWindow()
        self._app.win['camera'].imp.init_cam()
    
    def open_errwin(self):
        
        self._app.win['errors'].show()
        self._app.win['errors'].activateWindow()
    
class SettWin():
    
    def __init__(self, gui, app):
        self._app = app
        self._gui = gui
    
    def clean_up(self):
        # TODO: add check to see if changes were made, if not, don't ask user
        pressed = Question('Keep changes if made? ' +
                                  '(otherwise, revert to last loaded settings file.)'
                                  ).display()
        if pressed == QMessageBox.No:
            self.read_csv(self._gui.settingsFileName.text())

    # public methods
    def write_csv(self):
        '''
        Write all QLineEdit, QspinBox and QdoubleSpinBox of settings window to 'filepath' (csv).
        Show 'filepath' in 'settingsFileName' QLineEdit.
        '''
        # TODO: add support for combo box index

        filepath, _ = QFileDialog.getSaveFileName(self._gui,
                                                                 'Save Settings',
                                                                 const.SETTINGS_FOLDER_PATH,
                                                                 "CSV Files(*.csv *.txt)")
        import csv
        if filepath:
            self._gui.frame.findChild(QWidget, 'settingsFileName').setText(filepath)
            with open(filepath, 'w') as stream:
                #print("saving", filepath)
                writer = csv.writer(stream)
                # get all names of fields in settings window (for file saving/loading)
                l1 = self._gui.frame.findChildren(QLineEdit)
                l2 = self._gui.frame.findChildren(QSpinBox)
                l3 = self._gui.frame.findChildren(QDoubleSpinBox)
                field_names = [w.objectName() for w in (l1 + l2 + l3) if \
                                     (not w.objectName() == 'qt_spinbox_lineedit') and \
                                     (not w.objectName() == 'settingsFileName')] # perhaps better as for loop for readability
                #print(fieldNames)
                for i in range(len(field_names)):
                    widget = self._gui.frame.findChild(QWidget, field_names[i])
                    if hasattr(widget, 'value'): # spinner
                        rowdata = [field_names[i],  self._gui.frame.findChild(QWidget, field_names[i]).value()]
                    else: # line edit
                        rowdata = [field_names[i],  self._gui.frame.findChild(QWidget, field_names[i]).text()]
                    writer.writerow(rowdata)
            self._app.log.update('Settings file saved as: ' + filepath)

    def read_csv(self, filepath=''):
        '''
        Read 'filepath' (csv) and write to matching QLineEdit, QspinBox and QdoubleSpinBox of settings window.
        Show 'filepath' in 'settingsFileName' QLineEdit.
        '''
        # TODO: add support for combo box index
        
        if not filepath:
            filepath, _ = QFileDialog.getOpenFileName(self._gui,
                                                                     "Load Settings",
                                                                     const.SETTINGS_FOLDER_PATH,
                                                                     "CSV Files(*.csv *.txt)")
        import pandas as pd
        if filepath:
            try:
                 df = pd.read_csv(filepath, header=None, delimiter=',',
                                         keep_default_na=False, error_bad_lines=False)
                 self._gui.frame.findChild(QWidget, 'settingsFileName').setText(filepath)
                 for i in range(len(df)):
                    widget = self._gui.frame.findChild(QWidget, df.iloc[i, 0])
                    if not widget == 'nullptr':
                        if hasattr(widget, 'value'): # spinner
                            widget.setValue(float(df.iloc[i, 1]))
                        elif hasattr(widget, 'text'): # line edit
                            widget.setText(df.iloc[i, 1])
            except Exception as exc:
                txt = 'Error during read_csv.'
                Error(exc=exc, error_txt=txt).display()
            
            self._app.log.update('Settings file loaded: ' + filepath)

        else:
            error_txt = ('File path not supplied.')
            Error(error_txt=error_txt).display()

class CamWin():
    
    state = {}
    
    def __init__(self, gui, app):
        
        self._app = app
        self._gui = gui
        
        # add matplotlib-ready widget (canvas) for showing camera output
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        self._gui.figure = plt.figure()
        self._gui.canvas = FigureCanvas(self._gui.figure)
        self._gui.gridLayout.addWidget(self._gui.canvas, 0, 1)

    def clean_up(self):
        
        '''clean up before closing window'''
        
        # turn off video if On
        if self._cam.video_state:
            self.toggle_video(False)
        # disconnect camera
        self._cam.close()
        self._app.win['main'].actionCamera_Control.setEnabled(True)
        self._app.win['camera'] = None
        self._app.log.update('Camera connection closed', tag='verbose')
        return None

    def init_cam(self):
        
        try:
            self._cam = devices.Camera() # instantiate camera object
            self._cam.set_auto_exposure(True) #TEST?
            self._app.log.update('Camera connection opened', tag='verbose')
        except UC480Error as exc:
            Error(exc=exc).display()
    
    def toggle_video(self, bool):
        
        #Turn On
        if bool:
            self._gui.videoButton.setStyleSheet("background-color: "
                                                             "rgb(225, 245, 225); "
                                                             "color: black;")
            self._gui.videoButton.setText('Video ON')
            self._video_timer = QTimer()
            self._video_timer.timeout.connect(self._wait_for_frame)
            self._cam.toggle_video(True)
            self._video_timer.start(0)  # Run full throttle
            self._app.log.update('Camera video mode ON', tag='verbose')
            
        #Turn Off
        else:
            self._gui.videoButton.setStyleSheet("background-color: "
                                                             "rgb(225, 225, 225); "
                                                             "color: black;")
            self._gui.videoButton.setText('Start Video')
            self._cam.toggle_video(False)
            self._video_timer.stop()
            self._app.log.update('Camera video mode OFF', tag='verbose')

    def shoot(self):

        if self._cam.video_state:
            self.toggle_video(False)
            img = self._cam.shoot()
            self._imshow(img)
            self.toggle_video(True)
        else:
            img = self._cam.shoot()
            self._imshow(img)
        self._app.log.update('Camera photo taken', tag='verbose')
    
    # private methods
    def _imshow(self, img):
        
        '''Plot image'''
        
        self._gui.figure.clear()
        ax = self._gui.figure.add_subplot(111)
        ax.imshow(img)
        self._gui.canvas.draw()
    
    def _wait_for_frame(self):

        img = self._cam.latest_frame()
        self._imshow(img)
        
            
class ErrWin():
    # TODO: 
    
    def __init__(self, gui, app):
        
        self._app = app
        self._gui = gui

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
