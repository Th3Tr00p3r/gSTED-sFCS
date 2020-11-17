'''
Devices Module.
'''

import implementation.drivers as drivers
import implementation.logic as logic
import numpy as np
from implementation.error_handler import dvc_error_handler as err_hndlr

class UM232(drivers.FTDI_Instrument):
    
    def __init__(self):
        
        pass
    
class Counter(drivers.DAQmxInstrumentCI):
    
    def __init__(self, param_dict, error_dict):
        
        self.nick = 'COUNTER' # for errors?
        super().__init__(param_dict=param_dict)
        
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
        
        self.cont_count_buff = np.append(self.cont_count_buff, self.read())
        
    def average_counts(self, avg_intrvl): # in ms (range 10-1000)
                
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
    
    def __init__(self):
        
        # idealy 'Camera' would inherit 'UC480_Camera', but this is problematic because of the 'Instrumental' API
        self._driver = drivers.UC480_Camera(reopen_policy='new')
        
        self.video_state = False
        
    def close(self):
        
        self._driver.close()
    
    def set_auto_exposure(self, bool):
        
        self._driver.set_auto_exposure(bool)
    
    def shoot(self):
        
        return self._driver.grab_image()
    
    def toggle_video(self, bool):
        
        if bool:
            self._driver.start_live_video()
        else:
            self._driver.stop_live_video()
        self.video_state = bool
    
    def latest_frame(self):
        
        frame_ready = self._driver.wait_for_frame(timeout='0 ms')
        if frame_ready:
           return self._driver.latest_frame(copy=False)            
    
class ExcitationLaser(drivers.DAQmxInstrumentDO):
        
    '''Excitation Laser Control'''
        
    def __init__(self, address):
    
        super().__init__(address=address)

class DepletionLaser(drivers.VISAInstrument):
    '''
    Control depletion laser through pyVISA
    '''
    
    def __init__(self, address, error_dict):
        
        self.nick = 'DEP_LASER'
        self.address = address
        self.error_dict = error_dict
        self.error_dict[self.nick] = None
        
        self.current = None
        self.power = None
        self.temp = -999
        self.state = None
        super().__init__(address=address,
                               read_termination = '\r', 
                               write_termination = '\r')
                               
        self.toggle(False)
        self.set_current(1500)
        self.get_SHG_temp()
    
    @err_hndlr
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
    
    @err_hndlr
    def get_SHG_temp(self):
        
        while True:
            self.temp = self.query('SHGtemp')
            if self.temp != -999:
                break
    
    @err_hndlr
    def get_power(self):
        
        while True:
            self.power = self.query('Power 0')
            if self.power != -999:
                break
                
    @err_hndlr
    def set_power(self, value):
        
        # check that current value is within range
        if (value <= 1000) and (value >= 99):
            # change the mode to current
            self.write('Powerenable 1')
            # then set the power
            self.write('Setpower 0 ' + str(value))
        else:
            logic.Error(error_txt='Power out of range').display()
    
    @err_hndlr
    def get_current(self):
        while True:
            self.current = self.query('LDcurrent 1')
            if self.current != -999:
                break
    
    @err_hndlr
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
    
    def __init__(self, address):
    
        super().__init__(address=address)
    
class StepperStage():
    
    '''
    Control stepper stage through Arduino chip using PyVISA.
    This device operates slowly and needs special care,
    and so its driver is within its own class (not inherited)
    '''
    
    def __init__(self, address):
        
        self.nick = 'STAGE'
        self.address = address
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
