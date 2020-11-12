from instrumental.drivers.cameras.uc480 import UC480_Camera # NOQA
import pyvisa as visa
import nidaqmx

class DAQmxInstrument():
    
    def __init__(self, address, type): # types : AI/O, DI/O, CI/O
        self.address = address
        self.type = type
    
    def write(self, cmnd):
        with nidaqmx.Task() as task:
            if self.type == 'D':
                task.do_channels.add_do_chan(self.address)
            elif self.type == 'C':
                print('Implement me!')
            task.write(cmnd)

class VISAInstrument():
    
    def __init__(self, address,
                       read_termination='',
                       write_termination=''):
        
        self.address = address
        self.read_termination = read_termination
        self.write_termination = write_termination
        self.rm = visa.ResourceManager()
    
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
