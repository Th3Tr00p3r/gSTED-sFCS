'''
Error Handeling
'''
from implementation.dialog import Error
from pyvisa.errors import VisaIOError
from instrumental.drivers.cameras.uc480 import UC480Error
from nidaqmx.errors import DaqError
from pyftdi.ftdi import FtdiError
import functools

def driver_error_handler(func):
    
    # TODO: possibly turn this into a class, and create subclasses as needed. also create one for logging
    @functools.wraps(func)
    def wrapper_error_handler(dvc, *args, **kwargs):
        
        if dvc.error_dict[dvc.nick] is None: # if there's no errors
            try:
                return func(dvc, *args, **kwargs)
            
            except ValueError:
                return -999
            
            except DaqError as exc:
                Error(exc).display()
                
            except VisaIOError as exc:
                dvc.error_dict[dvc.nick] = 'VISA Error'
                # TODO: instead of showing error, show it in dedicated the ErrWin and set the LED to red
                Error(exc,
                    error_txt=F"VISA can't access {dvc.address} (depletion laser)").display()
                    
            except ValueError as exc:
                Error(exc).display()
                
            except FtdiError as exc:
                Error(exc).display()
                dvc.error_dict[dvc.nick] = 'FTDI Error'
                # TODO: STOP REPEATING ERRORS!
                
            except UC480Error as exc:
                Error(exc).display()
            
        else:
            print(F"'{dvc.nick}' error. Ignoring '{func.__name__}()' call.")
            return -999
                                               
    return wrapper_error_handler

def logic_error_handler(func):
    
    @functools.wraps(func)
    def wrapper_error_handler(*args, **kwargs):
        
        try:
            return func(*args, **kwargs)
            
        except FileNotFoundError as exc:
            Error(exc).display()
        
        except TypeError as exc:
            Error(exc).display()
            if hasattr(args[0], 'temp'):
                args[0].temp = -999
        
    return wrapper_error_handler
