import pyvisa as visa
import nidaqmx
from instrumental.drivers.cameras.uc480 import UC480_Camera as Camera # NOQA
import implementation.logic as logic
#import time #testing
# TODO: add parent classes for general visa and daqmx control

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
            
            self.rsrc.close()
    
class ExcitationLaser():
    
    def __init__(self, nick, address):
        
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
    
    def __init__(self, nick, address):
        
        self.SHG_temp = None
        self.mode = None
        self.current = None
        self.power = None
        super().__init__(nick=nick, address=address,
                               read_termination = '\r', 
                               write_termination = '\r')
        with VISAInstrument.Task(self) as task:
            task.write('setLDenable 0')
    
    def toggle(self):
        
        with VISAInstrument.Task(self) as task:
    #        if self.SHG_temp > 52 :
            if 1:
                if not self.state:
                    task.write('setLDenable 1')
                else:
                    task.write('setLDenable 0')
            else:
                logic.Error(error_txt='SHG temperature too low.')
            self.state = not self.state
        
    def get_SHG_temp(self):
        
        rsrc = self.rm.open_resource(self.address)
        rsrc.write('SHGtemp\r')
        ans = rsrc.read_raw(size=1)
        print(ans)
        rsrc.close()
        return ans

class DepletionShutter():
    
    def __init__(self, nick, address):
        
        self.nick = nick
        self.address = address
        self.comm_type = 'DO'
        # CLOSE if somehow OPEN
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(self.address)
            task.write(False)
        self.state = False
    
    def toggle(self):
        
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(self.address)
            self.state = not self.state
            task.write(self.state)
    
class StepperStage():
    '''
    Control stepper stage through arduino chip using PyVISA.
    This device operates slowly and needs special care.
    '''
    
    def __init__(self, nick, address):
        
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
