'''
Devices Module.
'''

import implementation.drivers as drivers
import implementation.logic as logic
import numpy as np
from implementation.error_handler import driver_error_handler as err_hndlr
from PyQt5.QtCore import QTimer

class UM232(drivers.FTDI_Instrument):
    
    def __init__(self, nick, param_dict, error_dict):
        
        super().__init__(nick=nick,
                               param_dict=param_dict,
                               error_dict=error_dict
                               )
        self.toggle(True)
    
    def toggle(self, bool):
        
        if bool:
            self.open()
        else:
            self.close()
        self.state = bool
    
class Counter(drivers.DAQmxInstrumentCI):
    
    def __init__(self, nick, param_dict, error_dict):
        
        super().__init__(nick=nick,
                               param_dict=param_dict,
                               error_dict=error_dict
                               )
        self.cont_count_buff = np.zeros(1, )
        self.counts = None
        
        self.toggle(True) # turn ON right from the start
    
    def toggle(self, bool):
        
        if bool:
            self.start()
        else:
            self.stop()
        self.state = bool
    
    def count(self):
        
        counts_np = self.read()
        self.cont_count_buff = np.append(self.cont_count_buff, counts_np)
        
    def average_counts(self, avg_intrvl): # in ms (range 10-1000)
        
        # TODO: this seems to average over about X10 the interval...
        
        dt = 1 # in ms (TODO: why is this value chosen? Ask Oleg)
        
        intrvl_time_unts = int(avg_intrvl/dt)
        start_idx = len(self.cont_count_buff) - intrvl_time_unts
        if start_idx > 0:
            return (self.cont_count_buff[-1] - self.cont_count_buff[-intrvl_time_unts]) /   \
                avg_intrvl # to have KHz
        else:
            return 0
    
    def dump_buff_overflow(self):
        
        buff_sz = self._param_dict['buff_sz']
        cnts_arr1D = self.cont_count_buff
        
        if len(cnts_arr1D) > buff_sz:
            self.cont_count_buff = cnts_arr1D[-buff_sz : ]
            
class Camera():
    
#    idealy 'Camera' would inherit 'UC480_Camera',
#    but this is problematic because of the 'Instrumental' API
    
    def __init__(self, nick, error_dict):
        
        self.nick = nick
        self.error_dict = error_dict
        self.video_timer = QTimer()
        self.video_timer.setInterval(100) # set to 100 ms
    
    @err_hndlr
    def toggle(self, bool):
        
        if bool:
            self._driver = drivers.UC480_Camera(reopen_policy='new')
        elif hasattr(self, '_driver'):
            self.video_timer.stop() # in case video is ON
            # TODO: possibly need to turn off video before closing?
            self._driver.close()
        self.state = bool
    
    @err_hndlr
    def set_auto_exposure(self, bool):
        
        self._driver.set_auto_exposure(bool)
    
    @err_hndlr
    def shoot(self):
        
        if self.video_timer.isActive():
            self.toggle_video(False)
            img = self._driver.grab_image()
            self.toggle_video(True)
            
        else:
            img = self._driver.grab_image()
            
        return img
    
    @err_hndlr
    def toggle_video(self, bool):
        
        if bool:
            self._driver.start_live_video()
            self.video_timer.start()
            
        else:
            self._driver.stop_live_video()
            self.video_timer.stop()
            
        self.video_state = bool
    
    @err_hndlr
    def latest_frame(self):
        
        frame_ready = self._driver.wait_for_frame(timeout='0 ms')
        if frame_ready:
           return self._driver.latest_frame(copy=False)            
    
class ExcitationLaser(drivers.DAQmxInstrumentDO):
        
    '''Excitation Laser Control'''
        
    def __init__(self, nick, address, error_dict):
    
        super().__init__(nick=nick,
                               address=address,
                               error_dict=error_dict)

class DepletionLaser(drivers.VISAInstrument):
    '''
    Control depletion laser through pyVISA
    '''
    
    def __init__(self, nick, address, error_dict):
        
        super().__init__(nick=nick,
                               address=address,
                               error_dict=error_dict,
                               read_termination = '\r', 
                               write_termination = '\r')
                               
        self.state = None
        
        self.toggle(False)
        self.set_current(1500)
        self.get_SHG_temp()
    
    def toggle(self, bool):
        
            if bool:
                if self.temp > 52 :
                    self.write('setLDenable 1')
                    self.state = bool
                else:
                    logic.Error(error_txt='SHG temperature too low.').display()
            else:
                self.write('setLDenable 0')
                self.state = bool
    
    def get_SHG_temp(self):
        
        self.temp = self.query('SHGtemp')
    
    def get_current(self):
        
        self.current = self.query('LDcurrent 1')
    
    def get_power(self):
        
        self.power = self.query('Power 0')
    
    def set_power(self, value):
        
        # check that current value is within range
        if (value <= 1000) and (value >= 99):
            # change the mode to current
            self.write('Powerenable 1')
            # then set the power
            self.write('Setpower 0 ' + str(value))
        else:
            logic.Error(error_txt='Power out of range').display()
    
    def set_current(self, value):
        
        # check that current value is within range
        if (value <= 2500) and (value >= 1500):
            # change the mode to current
            self.write('Powerenable 0')
            # then set the current
            self.write('setLDcur 1 ' + str(value))
        else:
            logic.Error(error_txt='Current out of range').display()
            

class DepletionShutter(drivers.DAQmxInstrumentDO):
    
    '''Depletion Shutter Control'''
    
    def __init__(self, nick, address, error_dict):
    
        super().__init__(nick=nick,
                               address=address,
                               error_dict=error_dict)
    
class StepperStage():
    
    '''
    Control stepper stage through Arduino chip using PyVISA.
    This device operates slowly and needs special care,
    and so its driver is within its own class (not inherited)
    '''
    
    def __init__(self, nick, address, error_dict):
        
        self.nick = nick
        self.address = address
        self.error_dict = error_dict
        self.rm = drivers.visa.ResourceManager()
        
        self.toggle(False)
    
    def toggle(self, bool):
        
        if bool:
            self.rsrc = self.rm.open_resource(self.address)
        else:
            if hasattr(self, 'rsrc'):
                self.rsrc.close()
        self.state = bool
    
    def move(self, dir=None,  steps=None):
        
        cmd_dict = {'UP': (lambda steps: 'my ' + str(-steps)),
                          'DOWN': (lambda steps: 'my ' + str(steps)),
                          'LEFT': (lambda steps: 'mx ' + str(steps)),
                          'RIGHT': (lambda steps: 'mx ' + str(-steps))
                        }
        self.rsrc.write(cmd_dict[dir](steps))
    
    def release(self):

        self.rsrc.write('ryx ')
