# -*- coding: utf-8 -*-
"""Error handeling."""

import functools

from instrumental.drivers.cameras.uc480 import UC480Error
from nidaqmx.errors import DaqError
from pyftdi.ftdi import FtdiError
from pyvisa.errors import VisaIOError

from utilities.dialog import Error


def driver_error_handler(func):
    """decorator for clean handling of various known errors occuring in drivers.py."""
    # TODO: decide what to do with multiple errors - make a list (could explode?) or leave only the first?
    
    @functools.wraps(func)
    def wrapper_error_handler(dvc, *args, **kwargs):
        """Doc."""

        #        if dvc.error_dict[dvc.nick] is None:  # if there's no errors
        try:
            return func(dvc, *args, **kwargs)

        except ValueError as exc:
            if dvc.nick == "DEP_LASER":
                return -999

            elif dvc.nick == "DEP_SHUTTER":
                return False
            
            elif dvc.nick == "UM232":
                dvc.error_dict[dvc.nick] = exc

            else:
                raise

        except DaqError as exc:
            dvc.error_dict[dvc.nick] = exc

            if dvc.nick in {"EXC_LASER", "DEP_SHUTTER", "TDC"}:
                return False

        except VisaIOError as exc:
            dvc.error_dict[dvc.nick] = exc

            if dvc.nick == "DEP_LASER":
                return -999

            if dvc.nick == "STAGE":
                return False

        except FtdiError as exc:
            dvc.error_dict[dvc.nick] = exc
            txt = f"{dvc.nick} cable disconnected."
            Error(error_txt=txt, error_title="Error").display()

            return False

        except AttributeError as exc:
            if dvc.nick == "UM232":
                dvc.error_dict[dvc.nick] = exc
            else:
                raise

        except OSError as exc:

            if dvc.nick == "UM232":
                dvc.error_dict[dvc.nick] = exc
                txt = f"Make sure {dvc.nick} cable is connected and restart"
                Error(error_txt=txt, error_title="Error").display()

            else:
                raise

        except UC480Error as exc:
            dvc.error_dict[dvc.nick] = exc

    #        else:
    #            print(f"'{dvc.nick}' error. Ignoring '{func.__name__}()' call.")
    #            # TODO: (low priority) this should not happen -
    #            # calls should be avoided if device is in error
    #            return False

    return wrapper_error_handler


def error_checker(nick_set=None):
    """
    decorator for clean handeling of GUI interactions with errorneous devices.

    nick_set - a set of all device nicks to check for errors
        before attempting the decorated func()

    """

    def outer_wrapper(func):
        """Doc."""

        @functools.wraps(func)
        def inner_wrapper(self, *args, **kwargs):
            """Doc."""

            if nick_set is not None:
                count = 0
                txt = ""
                for nick in nick_set:
                    exc = self._app.error_dict[nick]

                    if exc is not None:
                        txt += f"{nick} error.\n"
                        count += 1

                if count > 0:
                    txt += '\nSee "error window" for details.'
                    Error(error_txt=txt, error_title=f"Errors ({count})").display()

                else:
                    return func(self, *args, **kwargs)

            else:
                nick = args[0]
                exc = self._app.error_dict[nick]

                if exc is not None:
                    txt = f'{nick} error.\n\nSee "error window" for details.'
                    Error(error_txt=txt, error_title="Error").display()
                else:
                    return func(self, *args, **kwargs)

        return inner_wrapper

    return outer_wrapper


def logic_error_handler(func):
    """Doc."""

    @functools.wraps(func)
    def wrapper_error_handler(*args, **kwargs):
        """Doc."""

        try:
            return func(*args, **kwargs)

        except FileNotFoundError as exc:
            Error(exc).display()

    return wrapper_error_handler
