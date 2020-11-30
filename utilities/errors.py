'''
Error Handeling
'''
from utilities.dialog import Error
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
                if dvc.nick == 'DEP_LASER':
                    return -999
                
                elif dvc.nick == 'DEP_SHUTTER':
                    return False
                
                else:
                    raise
            
            except DaqError as exc:
                dvc.error_dict[dvc.nick] = exc
                
                if dvc.nick in {'EXC_LASER', 'DEP_SHUTTER', 'TDC'}:
                    return False
                
            except VisaIOError as exc:
                dvc.error_dict[dvc.nick] = exc
                
                if dvc.nick == 'DEP_LASER':
                    return -999
                
                if dvc.nick == 'STAGE':
                    return False
                    
#            except ValueError as exc:
#                Error(exc).display()
                
            except FtdiError as exc:
                dvc.error_dict[dvc.nick] = exc
                txt = F"{dvc.nick} cable disconnected."
                Error(error_txt=txt, error_title='Error').display()
            
            except AttributeError as exc:
                if dvc.nick == 'UM232':
                    dvc.error_dict[dvc.nick] = exc
                else:
                    raise
                
            except OSError as exc:
                
                if dvc.nick == 'UM232':
                    dvc.error_dict[dvc.nick] = exc
                    txt = F"Make sure {dvc.nick} cable is connected and restart"
                    Error(error_txt=txt, error_title='Error').display()
                    
                else:
                    raise
            
            except UC480Error as exc:
                dvc.error_dict[dvc.nick] = exc
            
        else:
            print(F"'{dvc.nick}' error. Ignoring '{func.__name__}()' call.")
            # TODO: flash error LED
            return False
                                               
    return wrapper_error_handler

def error_checker(nick_set=None):
    
    def outer_wrapper(func):
    
        @functools.wraps(func)
        def inner_wrapper(self, *args, **kwargs):
            
            if nick_set is not None:
                count = 0
                txt = ''
                for nick in nick_set:
                    exc = self._app.error_dict[nick]
                    
                    if exc is not None:
                        txt += F"{nick} error.\n"
                        count += 1
                
                if count > 0:
                    txt += '\nSee "error window" for details.'
                    Error(error_txt=txt, error_title=F"Errors ({count})").display()
                    
                else:
                    return func(self, *args, **kwargs)
                    
            else:
                nick = args[0]
                exc = self._app.error_dict[nick]
                
                if exc is not None:
                    txt = F"{nick} error.\n\nSee \"error window\" for details."
                    Error(error_txt=txt, error_title='Error').display()
                else:
                    return func(self, *args, **kwargs)
                
        return inner_wrapper
    return outer_wrapper


def logic_error_handler(func):
    
    @functools.wraps(func)
    def wrapper_error_handler(*args, **kwargs):
        
        try:
            return func(*args, **kwargs)
            
        except FileNotFoundError as exc:
            Error(exc).display()
        
    return wrapper_error_handler
