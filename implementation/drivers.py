import pyvisa as visa
import nidaqmx
import implementation.constants as const
from instrumental.drivers.cameras.uc480 import UC480_Camera # NOQA
#import time #testing

class Camera():
    
    def __init__(self):
        
        self.driver = UC480_Camera(reopen_policy='new')
    
class ExcitationLaser():
    
    def __init__(self, address):
        
        self.name = const.EXC_LASER_NAME
        self.address = address
        self.comm_type = 'DO'
        # Turn OFF if somehow ON
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(self.address)
            task.write(False)
        self.state = 'OFF'
    
    def toggle(self):
        
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(self.address)
            task.write(self.state == 'OFF')
        self.state = 'ON' if self.state == 'OFF' else 'OFF'
        
class DepletionShutter():
    
    def __init__(self, address):
        
        self.name = const.DEP_SHUTTER_NAME
        self.address = address
        self.comm_type = 'DO'
        # CLOSE if somehow OPEN
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(self.address)
            task.write(False)
        self.state = 'CLOSED'
    
    def toggle(self):
        
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(self.address)
            task.write(self.state == 'CLOSED')
        self.state = 'OPEN' if self.state == 'CLOSED' else 'CLOSED'
    
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
    
    

class StepperStage():
    '''
    Control stepper stage through pyVISA
    '''
    
    def __init__(self,  rsrc_alias):
        
        self.name = const.STAGE_NAME
        self.state = 'ON'
        self.rm = visa.ResourceManager()
        self.rsrc = self.rm.open_resource(rsrc_alias)

    def clean_up(self):
        
        self.rsrc.close()
        self.state = 'OFF'
        return None
    
    def move(self, dir=None,  steps=None):
        cmd_dict = {'UP': (lambda steps: 'my ' + str(-steps)),
                          'DOWN': (lambda steps: 'my ' + str(steps)),
                          'LEFT': (lambda steps: 'mx ' + str(steps)),
                          'RIGHT': (lambda steps: 'mx ' + str(-steps))
                        }
        self.rsrc.write(cmd_dict[dir](steps))
    
    def release(self):
        cmnd = 'ryx '
        self.rsrc.write(cmnd)
