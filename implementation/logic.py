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

import time

from implementation.error_handler import logic_error_handler as err_hndlr
from implementation.error_handler import error_checker as err_chck

class Timeout():
        
        def __init__(self, app):
            
            self._app = app
            
            self.timer_dict = self._get_timers()
            self.start_timers()
        
        def _get_timings_dict(self):
                
                '''Associate an (interval (ms), update_function) tuple with each device'''
                # TODO: possibly have a single or a few longer interval timers, not one for each (could be resource heavy)
                
                t_dict ={}
                t_dict['MAIN'] = (const.TIMEOUT, self._main)
                t_dict['DEP_LASER'] = (self._app.dvc_dict['DEP_LASER'].update_time * 1000,
                                                 lambda: self._app.dvc_dict['DEP_LASER'].toggle_update_ready(True))
                t_dict['COUNTER'] = (self._app.dvc_dict['COUNTER'].update_time * 1000,
                                              lambda: self._app.dvc_dict['COUNTER'].toggle_update_ready(True))
                t_dict['ERRORS'] = (2 * 1000, self._update_errorGUI)
                return t_dict
        
        def _get_timers(self):
            
            '''Instantiate timers for specific updates devices'''
            
            timers_dict = {}
            for key, value in self._get_timings_dict().items():
                timers_dict[key] = QTimer()
                timers_dict[key].setInterval(value[0])
                timers_dict[key].timeout.connect(value[1])
                
            return timers_dict
        
        def start_timers(self):
            
            # initiate all update functions once
            update_func_dict = self._get_timings_dict()
            for key, value in update_func_dict.items():
                update_func_dict[key][1]()
                
            # then start individual timers
            for key in self.timer_dict.keys():
                self.timer_dict[key].start()
        
        def stop_main(self):
            
            self.timer_dict['MAIN'].stop()
            self._app.log.update('stopping main timer.',
                                        tag='verbose')
            
        def start_main(self):
            
            self.timer_dict['MAIN'].start()
            self._app.log.update('starting main timer.',
                                        tag='verbose')
        
        def _update_dep(self):
            
            '''Update depletion laser GUI'''
            
            nick = 'DEP_LASER'
            
            @err_hndlr
            def update_SHG_temp(dep, main_gui):
                
                dep.get_SHG_temp()
                main_gui.depTemp.setValue(dep.temp)
                if dep.temp < const.MIN_SHG_TEMP:
                    main_gui.depTemp.setStyleSheet("background-color: rgb(255, 0, 0); color: white;")
                else:
                    main_gui.depTemp.setStyleSheet("background-color: white; color: black;")
            
            def update_power(dep, main_gui):
                dep.get_power()
                main_gui.depActualPowerSpinner.setValue(dep.power)
            
            def update_current(dep, main_gui):
                dep.get_current()
                main_gui.depActualCurrSpinner.setValue(dep.current)
            
            if (self._app.error_dict[nick] is None) and (self._app.dvc_dict[nick].update_ready):
                update_SHG_temp(self._app.dvc_dict[nick], self._app.win_dict['main'])
                
                if self._app.dvc_dict[nick].state is True: # check current/power only if laser is ON
                    update_power(self._app.dvc_dict[nick], self._app.win_dict['main'])
                    update_current(self._app.dvc_dict[nick], self._app.win_dict['main'])
                    
                self._app.dvc_dict[nick].toggle_update_ready(False)
        
        def _update_errorGUI(self):
            # TODO: update the error GUI according to errors in self._app.error_dict
            pass
            
        def _update_counter(self):
            
            nick = 'COUNTER'
            
            if self._app.error_dict[nick] is None:
                
                self._app.dvc_dict[nick].count() # read new counts
                
                if self._app.dvc_dict[nick].update_ready:
                    avg_interval = self._app.win_dict['main'].countsAvg.value()
                    self._app.win_dict['main'].countsSpinner.setValue( # update avg counts in main GUI
                        self._app.dvc_dict[nick].average_counts(avg_interval)
                        )
                        
                self._app.dvc_dict[nick].dump_buff_overflow() # dump old counts beyond buffer size
                
                self._app.dvc_dict[nick].toggle_update_ready(False)
        
        #MAIN
        def _main(self):
            
            self._update_dep()
            self._update_counter()
            
            if self._app.meas.type == 'FCS':
                self._app.dvc_dict['UM232'].read_TDC_data()

class App():
    
    def __init__(self):
        
        #init windows
        self.win_dict = {}
        self.win_dict['main'] = gui_module.MainWin(self)
        self.log = Log(self.win_dict['main'], dir_path='./log/')
        
        self.win_dict['settings'] = gui_module.SettWin(self)
        self.win_dict['settings'].imp.read_csv(const.DEFAULT_SETTINGS_FILE_PATH)
        
        self.win_dict['main'].imp.change_from_settings()
        
        self.win_dict['errors'] = gui_module.ErrWin(self)
        self.win_dict['camera'] = None # instantiated on pressing camera button
        
        # initialize error dict
        self.init_errors()
        #initialize active devices
        self.init_devices()
        # initialize measurement
        self.meas = Measurement(self)
        
        #FINALLY
        self.win_dict['main'].show()

        # set up main timeout event
        self.timeout_loop = Timeout(self)
    
    def init_devices(self):
        
        '''
        goes through a list of device nicknames,
        instantiating a driver object for each device
        '''
        
        def params_from_GUI(app, gui_dict):
            
            '''
            Get counter parameters from settings GUI
            using a dictionary predefined in constants.py
            '''
        
            param_dict = {}
            
            for key, val_dict in gui_dict.items():
                gui_field = getattr(app.win_dict['settings'], val_dict['field'])
                gui_field_value = getattr(gui_field, val_dict['access'])()
                param_dict[key] = gui_field_value
            return param_dict
        
        self.dvc_dict = {}
        for nick in const.DEVICE_NICKS:
            dvc_class = getattr(devices,
                                        const.DEVICE_CLASS_NAMES[nick])
            
            if nick in {'CAMERA'}:
                self.dvc_dict[nick] = dvc_class(nick=nick, error_dict=self.error_dict)
            
            else:
                param_dict = params_from_GUI(self, const.DVC_NICK_PARAMS_DICT[nick])
                self.dvc_dict[nick] = dvc_class(nick=nick,
                                                           param_dict=param_dict,
                                                           error_dict=self.error_dict)
    
    def init_errors(self):
        
        self.error_dict = {}
        for nick in const.DEVICE_NICKS:
            self.error_dict[nick] = None
    
    def clean_up_app(self, restart=False):
        
        '''
        Close all devices and secondary windows
        before closing/restarting application
        '''
        
        def close_all_dvcs(app):
            
            for nick in const.DEVICE_NICKS:
                if not self.error_dict[nick]:
                    app.dvc_dict[nick].toggle(False)
                
        def close_all_wins(app):
            
            for win_key in self.win_dict.keys():
                if self.win_dict[win_key] is not None: # can happen for camwin
                    if win_key not in {'main', 'camera'}: # dialogs close with reject()
                        self.win_dict[win_key].reject()
                    else:
                        self.win_dict[win_key].close() # mainwindows and widgets close with close()
        
        def lights_out(gui):
            
            '''turn OFF all device switch/LED icons'''
            
            gui.excOnButton.setIcon(QIcon(icon.SWITCH_OFF))
            gui.ledExc.setIcon(QIcon(icon.LED_OFF))
            gui.depEmissionOn.setIcon(QIcon(icon.SWITCH_OFF))
            gui.ledDep.setIcon(QIcon(icon.LED_OFF))
            gui.depShutterOn.setIcon(QIcon(icon.SWITCH_OFF))
            gui.ledShutter.setIcon(QIcon(icon.LED_OFF))
            gui.stageOn.setIcon(QIcon(icon.SWITCH_OFF))
            gui.stageButtonsGroup.setEnabled(False)
        
        if self.meas.type is not None:
            if self.meas.type == 'FCS':
                self.win_dict['main'].imp.toggle_FCS_meas()
            
        close_all_dvcs(self)
        
        if restart:
            
            if self.win_dict['camera'] is not None:
                self.win_dict['camera'].close()
            
            self.timeout_loop.stop_main()
            
            lights_out(self.win_dict['main'])
            self.win_dict['main'].depActualCurrSpinner.setValue(0)
            self.win_dict['main'].depActualPowerSpinner.setValue(0)
            
            self.init_errors()
            self.init_devices()
            time.sleep(0.1) # needed to avoid error with main timeout
            self.timeout_loop = Timeout(self)
            self.log.update('restarting application.',
                                        tag='verbose')
        
        else:
            close_all_wins(self)
            self.log.update('Quitting Application.')
    
    def exit_app(self, event):
        
        pressed = Question(q_txt='Are you sure you want to quit?',
                                   q_title='Quitting Program').display()
        if pressed == QMessageBox.Yes:
            self.timeout_loop.stop_main()
            self.clean_up_app()
        else:
            event.ignore()

class MainWin():
    
    def __init__(self, gui, app):
        
        self._app = app
        self._gui = gui
        
        # intialize gui
        self._gui.actionLaser_Control.setChecked(True)
        self._gui.actionStepper_Stage_Control.setChecked(True)
        self._gui.stageButtonsGroup.setEnabled(False)
        self._gui.actionLog.setChecked(True)
        
        self._gui.countsAvg.setValue(self._gui.countsAvgSlider.value())
        
        #connect signals and slots
        self._gui.ledExc.clicked.connect(self.show_laser_dock)
        self._gui.ledDep.clicked.connect(self.show_laser_dock)
        self._gui.ledShutter.clicked.connect(self.show_laser_dock)
        self._gui.actionRestart.triggered.connect(self.restart)
    
    def close(self, event):
        
        self._app.exit_app(event)
        
    def restart(self):
        
        '''Restart all devices (except camera) and the timeout loop'''
        
        pressed = Question(q_txt='Are you sure?',
                                   q_title='Restarting Program').display()
        if pressed == QMessageBox.Yes:
            self._app.clean_up_app(restart=True)
    
    @err_chck()
    def dvc_toggle(self, nick):
        
        gui_switch_object = getattr(self._gui, const.ICON_DICT[nick]['SWITCH'])
        
        if not self._app.dvc_dict[nick].state: # switch ON
            self._app.dvc_dict[nick].toggle(True)
            
            if self._app.dvc_dict[nick].state and (not self._app.error_dict[nick]): # if managed to turn ON
                gui_switch_object.setIcon(QIcon(icon.SWITCH_ON))
                
                if 'LED' in const.ICON_DICT[nick].keys():
                    gui_led_object = getattr(self._gui, const.ICON_DICT[nick]['LED'])
                    on_icon = QIcon(const.ICON_DICT[nick]['ICON'])
                    gui_led_object.setIcon(on_icon)
                    
                self._app.log.update(F"{const.LOG_DICT[nick]} toggled ON",
                                            tag='verbose')
                
                if nick == 'STAGE':
                    self._gui.stageButtonsGroup.setEnabled(True)
                
                return True
            
            else:
                return False
            
        else: # switch OFF
            self._app.dvc_dict[nick].toggle(False)
            
            if not self._app.dvc_dict[nick].state: # if managed to turn OFF
                gui_switch_object.setIcon(QIcon(icon.SWITCH_OFF))
                
                if 'LED' in const.ICON_DICT[nick].keys():
                    gui_led_object = getattr(self._gui, const.ICON_DICT[nick]['LED'])
                    gui_led_object.setIcon(QIcon(icon.LED_OFF)) 
                    
                self._app.log.update(F"{const.LOG_DICT[nick]} toggled OFF",
                                            tag='verbose')
                
                if nick in {'DEP_LASER'}: # set curr/pow values to zero when depletion is turned OFF
                    self._gui.depActualCurrSpinner.setValue(0)
                    self._gui.depActualPowerSpinner.setValue(0)
                    
                return False
    
    @err_chck({'DEP_LASER'})
    def dep_sett_apply(self):
        
        nick = 'DEP_LASER'
        if self._gui.currModeRadio.isChecked(): # current mode
            val = self._gui.depCurrSpinner.value()
            self._app.dvc_dict[nick].set_current(val)
        else: # power mode
            val = self._gui.depPowSpinner.value()
            self._app.dvc_dict[nick].set_power(val)
    
    @err_chck({'STAGE'})
    def move_stage(self, dir, steps):
        
        nick = 'STAGE'
        self._app.dvc_dict[nick].move(dir=dir, steps=steps)
        self._app.log.update(F"{const.LOG_DICT[nick]} "
                                    F"moved {str(steps)} steps {str(dir)}",
                                    tag='verbose')
    
    @err_chck({'STAGE'})
    def release_stage(self):
        
        nick = 'STAGE'
        self._app.dvc_dict[nick].release()
        self._app.log.update(F"{const.LOG_DICT[nick]} released",
                                    tag='verbose')

    def show_laser_dock(self):
        '''
        Make the laser dock visible (convenience)
        '''
        if not self._gui.laserDock.isVisible():
            self._gui.laserDock.setVisible(True)
            self._gui.actionLaser_Control.setChecked(True)
    
    @err_chck({'TDC', 'UM232'})
    def toggle_FCS_meas(self):
        
        if self._app.meas.type is None:
            self._app.meas = Measurement(self._app, type='FCS',
                                                       duration_spinner=self._gui.measFCSDuration,
                                                       prog_bar=self._gui.FCSprogressBar,
                                                       log=self._app.log)
            self._app.meas.start()
            self._gui.startFcsMeasurementButton.setText('Stop \nMeasurement')
        elif self._app.meas.type == 'FCS':
            self._app.meas.stop()
            self._app.meas = Measurement(self._app)
            self._gui.startFcsMeasurementButton.setText('Start \nMeasurement')
        else:
            error_txt = (F"Another type of measurement "
                             F"({self._app.meas.type}) is currently running.")
            Error(error_txt=error_txt).display()
    
    def open_settwin(self):
        
        self._app.win_dict['settings'].show()
        self._app.win_dict['settings'].activateWindow()
    
    @err_chck({'CAMERA'})
    def open_camwin(self):
        
        self._gui.actionCamera_Control.setEnabled(False)
        self._app.win_dict['camera'] = gui_module.CamWin(app=self._app)
        self._app.win_dict['camera'].show()
        self._app.win_dict['camera'].activateWindow()
    
    def open_errwin(self):
        
        self._app.win_dict['errors'].show()
        self._app.win_dict['errors'].activateWindow()
    
    def cnts_avg_sldr_changed(self, val):
        '''
        Set the spinner to show the value of the slider,
        and change the counts display frequency to as low as needed
        (lower then the averaging frequency)
        '''
        self._gui.countsAvg.setValue(val)
        
        curr_intrvl = self._app.dvc_dict['COUNTER'].update_time * 1000
        
        if val > curr_intrvl:
            self._app.timeout_loop.timer_dict['COUNTER'].setInterval(val)
        else:
            self._app.timeout_loop.timer_dict['COUNTER'].setInterval(curr_intrvl)
    
    def change_from_settings(self):
        
        pass
#        self._gui.countsAvgSlider.setMinimum(
#            self._app.win_dict['settings'].counterUpdateTime.value() * 1000)
    
class SettWin():
    
    def __init__(self, gui, app):
        self._app = app
        self._gui = gui
    
    def clean_up(self):
        # TODO: add check to see if changes were made, if not, don't ask user
        pressed = Question('Keep changes if made? '
                                   '(otherwise, revert to last loaded settings file.)'
                                  ).display()
        if pressed == QMessageBox.No:
            self.read_csv(self._gui.settingsFileName.text())

    # public methods
    @err_hndlr
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
                
        self._app.log.update(F"Settings file saved as: '{filepath}'")
    
    @err_hndlr
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
        
        self._app.log.update(F"Settings file loaded: '{filepath}'")

class CamWin():
    
    def __init__(self, gui, app):
        
        self._app = app
        self._gui = gui
        
        # add matplotlib-ready widget (canvas) for showing camera output
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        self._gui.figure = plt.figure()
        self._gui.canvas = FigureCanvas(self._gui.figure)
        self._gui.gridLayout.addWidget(self._gui.canvas, 0, 1)
        
        self.init_cam()
    
    def init_cam(self):
        
        self._cam = self._app.dvc_dict['CAMERA']
        self._cam.toggle(True)
        self._cam.video_timer.timeout.connect(self._video_timeout)
        
        self._app.log.update('Camera connection opened',
                                    tag='verbose')
    
    def clean_up(self):
        
        '''clean up before closing window'''

        self._cam.toggle(False)
        self._app.win_dict['main'].actionCamera_Control.setEnabled(True) # enable camera button again
        self._app.timeout_loop.start_main() # for restarting main loop in case camwin closed while video ON
        
        self._app.log.update('Camera connection closed',
                                    tag='verbose')
                                    
        return None
    
    def toggle_video(self, bool):
        
        if bool: #turn On
            self._app.timeout_loop.stop_main()
            self._cam.toggle_video(True)
            
            self._gui.videoButton.setStyleSheet("background-color: "
                                                             "rgb(225, 245, 225); "
                                                             "color: black;")
            self._gui.videoButton.setText('Video ON')
            
            self._app.log.update('Camera video mode ON',
                                        tag='verbose')
            
        else: #turn Off
            self._cam.toggle_video(False)
            self._app.timeout_loop.start_main()
            
            self._gui.videoButton.setStyleSheet("background-color: "
                                                             "rgb(225, 225, 225); "
                                                             "color: black;")
            self._gui.videoButton.setText('Start Video')
            
            self._app.log.update('Camera video mode OFF',
                                        tag='verbose')

    def shoot(self):
        
        img = self._cam.shoot()
        self._imshow(img)
        
        self._app.log.update('Camera photo taken',
                                    tag='verbose')
    
    # private methods
    def _imshow(self, img):
        
        '''Plot image'''
        
        self._gui.figure.clear()
        ax = self._gui.figure.add_subplot(111)
        ax.imshow(img)
        self._gui.canvas.draw()
    
    def _video_timeout(self):

        img = self._cam.latest_frame()
        self._imshow(img)
    
class ErrWin():
    # TODO: 
    
    def __init__(self, gui, app):
        
        self._app = app
        self._gui = gui

class Measurement():
    
    # TODO: move timer to timeout_loop. this would mean all updates will be made from there too (progress bar etc.)
    
    def __init__(self, app, type=None,
                       duration_spinner=None,
                       prog_bar=None,
                       log=None):
                           
        # instance attributes
        self._app = app
        self.type = type
        self.duration_spinner = duration_spinner
        self.prog_bar = prog_bar
        self.log = log
        self._timer = QTimer()
        self._timer.timeout.connect(self._timer_timeout)
        self.time_passed = 0
        
    # public methods
    def start(self):
        self._timer.start(1000)
        self._app.dvc_dict['TDC'].toggle(True)
        
        self.log.update(F"{self.type} measurement started")
        
    def stop(self):
        self._timer.stop()
        self._app.dvc_dict['TDC'].toggle(False)
            
        if self.prog_bar:
            self.prog_bar.setValue(0)
            
        self.log.update(F"{self.type} measurement stopped")
    
    # private methods
    def _timer_timeout(self):
#        from PyQt5.QtMultimedia import QSound
        if self.time_passed < self.duration_spinner.value():
            self.time_passed += 1
#            if self.time_passed == self.duration_spinbox.value():
#                QSound.play(const.MEAS_COMPLETE_SOUND);
        else: # timer finished
            self.disp_ACF()
            self._app.dvc_dict['UM232'].init_data()
            self._app.dvc_dict['UM232'].purge()
            self.time_passed = 1
        if self.prog_bar:
            prog = self.time_passed / self.duration_spinner.value() * 100
            self.prog_bar.setValue(prog)
    
    def disp_ACF(self):
        
        print(F"Measurement Finished:\n"
                F"Full Data = {self._app.dvc_dict['UM232'].data}\n"
                F"Total Bytes = {self._app.dvc_dict['UM232'].tot_bytes}\n")
