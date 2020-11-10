import pyvisa as visa
import nidaqmx
from instrumental.drivers.cameras.uc480 import UC480_Camera # NOQA
import implementation.logic as logic
#import time #testing

class Camera():
    
    def __init__(self):
        
        self.driver = UC480_Camera(reopen_policy='new')

class VISAInstrument():
    
    def __init__(self, nick, address,
                       read_termination='',
                       write_termination=''):
        
        self.nick = nick
        self.address = address
        self.state = False
        self.read_termination = read_termination
        self.write_termination = write_termination
        self.rm = visa.ResourceManager()
    
    class Task():
        
        def __init__(self, inst):

            self.inst = inst
            
    
        def __enter__(self):
            self.rsrc = self.inst.rm.open_resource(self.inst.address,
                                                               read_termination=self.inst.read_termination,
                                                               write_termination=self.inst.write_termination)
            return self.rsrc
        
        def __exit__(self, exc_type, exc_value, exc_tb):
            
            if hasattr(self, 'rsrc'):
                self.rsrc.close()
    
    def write(self, cmnd):
        
        with VISAInstrument.Task(self) as task:
            task.write(cmnd)
    
    def query(self, cmnd):
        
        with VISAInstrument.Task(self) as task:
            reply = task.query(cmnd)
            try:
                return float(reply)
            except:
                return -999                
    
class ExcitationLaser():
    
    def __init__(self, nick, address, error_dict):
        
        self.nick = nick
        self.address = address
        self.comm_type = 'DO' # to be used for a general definition of daqmx class
        # Turn OFF if somehow ON
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(self.address)
            task.write(False)
        self.state = False
    
    def toggle(self):
        
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(self.address)
            self.state = not self.state
            task.write(self.state)

class DepletionLaser(VISAInstrument):
    '''
    Control depletion laser through pyVISA
    '''
    
    def __init__(self, nick, address, error_dict):
        
        self.current = None
        self.power = None
        self.state = False
        self.temp = -999
        super().__init__(nick=nick, address=address,
                               read_termination = '\r', 
                               write_termination = '\r')
        try:
            self.toggle(False)
            self.set_current(1500)
        except visa.errors.VisaIOError:
            error_dict[nick] = 'VISA Error'
            logic.Error(error_txt='VISA can\'t access ' + address +
                                         ' (depletion laser)').display()
    
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
        return self.temp
    
    def get_power(self):
        
        return self.query('Power 0')
    
    def set_power(self, value):
        
        # check that current value is within range
        if (value <= 1000) and (value >= 99):
            # change the mode to current
            self.write('Powerenable 1')
            # then set the power
            self.write('Setpower 0 ' + str(value))
        else:
            logic.Error(error_txt='Power out of range').display()
    
    def get_current(self):
        
        return self.query('LDcurrent 1')
    
    def set_current(self, value):
        
        # check that current value is within range
        if (value <= 2500) and (value >= 1500):
            # change the mode to current
            self.write('Powerenable 0')
            # then set the current
            self.write('setLDcur 1 ' + str(value))
        else:
            logic.Error(error_txt='Current out of range').display()
            

class DepletionShutter():
    '''
    Depletion Shutter Control
    '''
    def __init__(self, nick, address, error_dict):
        '''
        Instatiate object and close the shutter
        '''
        self.nick = nick
        self.address = address
        self.comm_type = 'DO'
        self.toggle(False)
        self.state = False
    
    def toggle(self,  bool):
        
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(self.address)
            task.write(bool)
            self.state = bool
    
class StepperStage():
    '''
    Control stepper stage through arduino chip using PyVISA.
    This device operates slowly and needs special care.
    '''
    
    def __init__(self, nick, address, error_dict):
        
        self.nick = nick
        self.address = address
        self.rm = visa.ResourceManager()
        self.state = False
    
    def open(self):
        
        self.rsrc = self.rm.open_resource(self.address)
        self.state = True
        
    def close(self):
        
        self.rsrc.close()
        self.state = False
    
    def move(self, dir=None,  steps=None):
        
        cmd_dict = {'UP': (lambda steps: 'my ' + str(-steps)),
                          'DOWN': (lambda steps: 'my ' + str(steps)),
                          'LEFT': (lambda steps: 'mx ' + str(steps)),
                          'RIGHT': (lambda steps: 'mx ' + str(-steps))
                        }
        self.rsrc.write(cmd_dict[dir](steps))
    
    def release(self):

        self.rsrc.write('ryx ')

        ####TESTING####
        
##        print('settings channel input: ',  chan)
##        local_sys = nidaqmx.system.System.local()
##        local_driver_v = local_sys.driver_version
##        
##        print('DAQmx {0}.{1}.{2}'.format(local_driver_v.major_version, local_driver_v.minor_version,
##                                                         local_driver_v.update_version))
##        for device in local_sys.devices:
##            print('Device Name: {0}, Product Category: {1}, Product Type: {2}'.format(
##                    device.name, device.product_category, device.product_type))
##            device.self_test_device()
##            print('digital ports: ',  device.do_ports.channel_names)
##            print('digital lines: ',  device.di_lines.channel_names)
            
        ####TESTING####
