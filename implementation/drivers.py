from instrumental.drivers.cameras.uc480 import UC480_Camera # NOQA
import pyvisa as visa
import nidaqmx

class DAQmxInstrumentDO():
    
    def __init__(self, address):
        self.address = address
    
    def write(self, cmnd):
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(self.address)
            task.write(cmnd)

class DAQmxInstrumentCI():
    
    def __init__(self, param_dict):
        
         self.params = param_dict
        
    def start(self):
        
        with nidaqmx.Task() as task:
            chan = task.ci_channels. \
                       add_ci_count_edges_chan(counter=self.params['photon_cntr'],
                                                           edge=nidaqmx.constants.Edge.RISING,
                                                           initial_count=0,
                                                           count_direction=nidaqmx.constants.CountDirection.COUNT_UP)
            chan.ci_count_edges_term = self.params['CI_cnt_edges_term']
            chan.ci_dup_count_prevention = self.params['CI_dup_prvnt']
            task.start()

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
