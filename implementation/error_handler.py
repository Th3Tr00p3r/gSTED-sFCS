'''
Error Handeling
'''
from implementation.dialog import Error
from pyvisa.errors import VisaIOError
from instrumental.drivers.cameras.uc480 import UC480Error
from nidaqmx.errors import DaqError
import functools

def driver_error_handler(func):
    
    # TODO: possibly turn this into a class, and create subclasses as needed. also create one for logging
    @functools.wraps(func)
    def wrapper_error_handler(dvc, *args, **kwargs):
        
        if not dvc.error_dict[dvc.nick]: # if there's no errors
            try:
                return func(dvc, *args, **kwargs)
            
            except ValueError:
                return -999
            
            except DaqError as exc:
                Error(exc).display()
                
            except VisaIOError as exc:
                dvc.error_dict[dvc.nick] = 'VISA Error'
                Error(exc, error_txt='VISA can\'t access ' + dvc.address +
                                                   ' (depletion laser)').display()
            
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
            func(*args, **kwargs)
            
        except FileNotFoundError as exc:
            Error(exc).display()
                                               
    return wrapper_error_handler
