'''
Drivers Module.
'''

from instrumental.drivers.cameras.uc480 import UC480_Camera # NOQA
import pyvisa as visa
import nidaqmx
import numpy as np

class DAQmxInstrumentDO():
    
    def __init__(self, address):
        self._address = address
    
    def write(self, cmnd):
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(self._address)
            task.write(cmnd)

class DAQmxInstrumentCI():
    
    def __init__(self, param_dict):
        
        self._param_dict = param_dict
        self._task = nidaqmx.Task()
        self._init_chan()
        
    def _init_chan(self):
    
        chan = self._task.ci_channels. \
                   add_ci_count_edges_chan(counter=self._param_dict['photon_cntr'],
                                                       edge=nidaqmx.constants.Edge.RISING,
                                                       initial_count=0,
                                                       count_direction=nidaqmx.constants.CountDirection.COUNT_UP)
        chan.ci_count_edges_term = self._param_dict['CI_cnt_edges_term']
#        chan.ci_dup_count_prevention = self._params['CI_dup_prvnt']
    
    def start(self):
        
        self._task.start()
    
    def read(self):
    
        return np.array(self._task.read(number_of_samples_per_channel=-1)[0])
        
    def stop(self):
        
        self._task.stop()

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
            except ValueError:
                return -999
                
    class Task():
    
        def __init__(self, inst):
            self._inst = inst
            
        def __enter__(self):
            self._rsrc = self._inst.rm.open_resource(self._inst.address,
                                                               read_termination=self._inst.read_termination,
                                                               write_termination=self._inst.write_termination)
            return self._rsrc
        
        def __exit__(self, exc_type, exc_value, exc_tb):
            
            if hasattr(self, '_rsrc'):
                self._rsrc.close()

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
