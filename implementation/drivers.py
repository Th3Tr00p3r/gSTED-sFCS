'''
Drivers Module.
'''

from instrumental.drivers.cameras.uc480 import UC480_Camera # NOQA
import pyvisa as visa
import nidaqmx
from pyftdi.ftdi import Ftdi
import numpy as np
from implementation.error_handler import driver_error_handler as err_hndlr

class FTDI_Instrument():
    
    def __init__(self, nick, param_dict, error_dict):
        
        self.nick = nick
        self._param_dict = param_dict
        self.error_dict = error_dict
        self.inst = Ftdi()
    
    def open(self):
        
        self.inst.open(self._param_dict['vend_id'],
                           self._param_dict['prod_id']
                           )
        self.inst.set_bitmode(0, getattr(Ftdi.BitMode,
                                                   self._param_dict['bit_mode']
                                                   ))
        self.inst._usb_read_timeout = self._param_dict['read_timeout']
        self.inst._usb_write_timeout = self._param_dict['read_timeout']
        self.inst.set_latency_timer(self._param_dict['ltncy_tmr_val'])
        self.inst.set_flowctrl(self._param_dict['flow_ctrl'])
        self.eff_baud_rate = self.inst.set_baudrate(self._param_dict['baud_rate'])
        
        self.inst.purge_buffers()
    
    def close(self):
        
        self.inst.close()
        

class DAQmxInstrumentDO():
    
    def __init__(self, nick, address, error_dict):
        
        self.nick = nick
        self._address = address
        self.error_dict = error_dict
        
        self.toggle(False)
        
    @err_hndlr
    def _write(self, cmnd):
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(self._address)
            task.write(cmnd)
            
    @err_hndlr        
    def toggle(self, bool):
        
        self._write(bool)
        self.state = bool

class DAQmxInstrumentCI():
    
    def __init__(self, nick, param_dict, error_dict):
        
        self.nick = nick
        self._param_dict = param_dict
        self.error_dict = error_dict
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
    
    @err_hndlr
    def start(self):
        
        self._task.start()
    
    @err_hndlr
    def read(self):
        
        counts = self._task.read(number_of_samples_per_channel=-1)[0]
        return np.array(counts)
    
    @err_hndlr
    def stop(self):
        
        self._task.stop()

class VISAInstrument():
    
    def __init__(self, nick, address, error_dict,
                       read_termination='',
                       write_termination=''):
        
        self.nick = nick
        self.address = address
        self.error_dict = error_dict
        self.read_termination = read_termination
        self.write_termination = write_termination
        self.rm = visa.ResourceManager()
    
    @err_hndlr
    def write(self, cmnd):
        
        with VISAInstrument.Task(self) as task:
            task.write(cmnd)
    
    @err_hndlr
    def query(self, cmnd):
        
        with VISAInstrument.Task(self) as task:
            return float(task.query(cmnd))
                
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