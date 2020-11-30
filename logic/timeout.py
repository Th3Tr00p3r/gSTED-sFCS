'''
Timeout module
'''

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon

import gui.icons.icon_paths as icon
import utilities.constants as const

class Timeout():
        
        def __init__(self, app):
            
            self._app = app
            self._gui_update_ready = False
            
            self.timer_dict = self._get_timers()
            self.start_timers()
        
        def _get_timings_dict(self):
                
                '''Associate an (interval (ms), update_function) tuple with each device'''
                
                t_dict ={}
                t_dict['MAIN'] = (const.TIMEOUT, self._main)
                t_dict['DEP_LASER'] = (self._app.dvc_dict['DEP_LASER'].update_time * 1000,
                                                 lambda: self._app.dvc_dict['DEP_LASER'].toggle_update_ready(True))
                t_dict['COUNTER'] = (self._app.dvc_dict['COUNTER'].update_time * 1000,
                                              lambda: self._app.dvc_dict['COUNTER'].toggle_update_ready(True))
                t_dict['GUI'] = (2 * 1000,
                                        self._set_gui_update_ready)
                t_dict['MEAS'] = (1 * 1000,
                                          lambda: self._app.meas.toggle_update_ready(True))
                return t_dict
        
        def _get_timers(self):
            
            '''Instantiate timers for specific updates devices'''
            
            timers_dict = {}
            for key, value in self._get_timings_dict().items():
                timers_dict[key] = QTimer()
                timers_dict[key].setInterval(value[0])
                timers_dict[key].timeout.connect(value[1])
                
            return timers_dict
        
        def _set_gui_update_ready(self):
            
            self._gui_update_ready = True
        
        def start_timers(self):
            
            # initiate all update functions once
            update_func_dict = self._get_timings_dict()
            for key, value in update_func_dict.items():
                update_func_dict[key][1]()
                
            # then start individual timers
            for key in self.timer_dict.keys():
                self.timer_dict[key].start()
        
        def stop_timers(self):
                
            for key in self.timer_dict.keys():
                self.timer_dict[key].stop()
            self._app.log.update('stopping all timers.',
                                        tag='verbose')
        
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
            
            def update_SHG_temp(dep, main_gui):
                
                # TODO: fix Nnone type when turning off dep laser while app running○
                
                dep.get_SHG_temp()
                main_gui.depTemp.setValue(dep.temp)
                if dep.temp < dep.min_SHG_temp:
                    main_gui.depTemp.setStyleSheet("background-color: red; color: white;")
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
        
        def _update_gui(self):
            # TODO: update the error GUI according to errors in self._app.error_dict
            
            if self._gui_update_ready:
            
                for nick in self._app.dvc_dict.keys():
                    if self._app.error_dict[nick] is not None: # case ERROR
                        gui_led_object = getattr(self._app.win_dict['main'], const.ICON_DICT[nick]['LED'])
                        red_led = QIcon(icon.LED_RED)
                        gui_led_object.setIcon(red_led)

                    elif self._app.dvc_dict[nick].state is True: # case ON
                        gui_led_object = getattr(self._app.win_dict['main'], const.ICON_DICT[nick]['LED'])
                        on_icon = QIcon(const.ICON_DICT[nick]['ICON'])
                        gui_led_object.setIcon(on_icon)
                    
                    else: # case OFF
                        gui_led_object = getattr(self._app.win_dict['main'], const.ICON_DICT[nick]['LED'])
                        off_led = QIcon(icon.LED_OFF)
                        gui_led_object.setIcon(off_led)
                
                self._gui_update_ready = False
        
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
        
        def _update_measurement(self):
            
            if self._app.error_dict['UM232'] is None:
                meas = self._app.meas
                
                if meas.type == 'FCS':
                    self._app.dvc_dict['UM232'].read_TDC_data()
                
                    if meas.update_ready:
                        
                        if meas.time_passed < meas.duration_spinner.value():
                            meas.time_passed += 1
                            
                        else: # timer finished
                            meas.disp_ACF()
                            self._app.dvc_dict['UM232'].init_data()
                            self._app.dvc_dict['UM232'].purge()
                            meas.time_passed = 1
                            
                        if meas.prog_bar:
                            prog = meas.time_passed / meas.duration_spinner.value() * 100
                            meas.prog_bar.setValue(prog)
                            
                        meas.update_ready = False
        
        #MAIN
        def _main(self):
            
            self._update_dep()
            self._update_counter()
            self._update_gui()
            self._update_measurement()
