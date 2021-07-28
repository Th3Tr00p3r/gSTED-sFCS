"""Error handeling."""

import asyncio
import functools
import logging
import os
import sys
import traceback
from types import FunctionType
from typing import Callable

from PyQt5.QtGui import QIcon

import gui
import logic.devices as dvcs
from utilities.dialog import Error


def build_error_dict(exc: Exception) -> str:
    """Doc."""

    exc_type = exc.__class__.__name__
    _, _, tb = sys.exc_info()
    frmtd_tb = "\n".join(traceback.format_tb(tb))
    # show the first 'n' existing levels of traceback for module and line number
    exc_loc = []
    while tb is not None:
        _, filename = os.path.split(tb.tb_frame.f_code.co_filename)
        lineno = tb.tb_lineno
        exc_loc.append((filename, lineno))
        tb = tb.tb_next
    return dict(type=exc_type, loc=exc_loc, msg=str(exc), tb=frmtd_tb)


def err_hndlr(exc, func_locals, func_frame, lvl="error", dvc=None, disp=False) -> str:
    """Doc."""

    def get_frame_details(frame, locals):
        """
        Create a set of all argument names except 'self', and use them to filter the 'locals()' dictionary,
        leaving only external arguments and their values in the final string. Also get the function name.
        """

        frame_code = frame.f_code
        func_name = frame_code.co_name
        func_arg_names = set(frame_code.co_varnames[: frame_code.co_argcount]) - {"self"}
        arg_string = ", ".join(
            [f"{key}={str(val)}" for key, val in locals.items() if key in func_arg_names]
        )
        return f"{func_name}({arg_string})"

    error_dict = build_error_dict(exc)
    func_string = get_frame_details(func_frame, func_locals)
    location_string = " -> ".join(
        [f"{filename}, {lineno}" for filename, lineno in error_dict["loc"]]
    )

    if dvc is not None:  # device error
        dvc_log_ref = dvcs.DEVICE_ATTR_DICT[dvc.nick].log_ref
        log_str = (
            f"{dvc_log_ref} didn't respond to '{func_string}' ({location_string}). "
            f"[{error_dict['type']}: {error_dict['msg']}]"
        )
        if lvl == "error":
            if not dvc.error_dict:  # keep only first error
                dvc.error_dict = error_dict
            dvc.led_widget.set(QIcon(gui.icons.icon_paths_dict["led_red"]))

    else:  # logic eror
        log_str = f"{error_dict['type']}: {error_dict['msg']} ({func_string}, {location_string})"
        if disp:
            Error(**error_dict).display()

    getattr(logging, lvl)(log_str, exc_info=False)

    return log_str


def device_error_checker(func) -> Callable:
    """
    Decorator for clean handeling of GUI interactions with errorneous devices.
    Checks for errors in devices associated with 'func' and shows error box
    if exist.

    """

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):

            try:
                if not self.error_dict:
                    return await func(self, *args, **kwargs)
                else:
                    if (func.__name__ == "_toggle") and (args[0] is False):
                        # if toggling off
                        pass
                    else:
                        self.error_display.set(
                            f"{self.log_ref} error. Click relevant LED for details."
                        )
                        raise DeviceError(self.error_dict["msg"])
            except AttributeError:
                # if not hasattr(self, "error_dict")
                return await func(self, *args, **kwargs)

    else:

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            try:
                if not self.error_dict:
                    return func(self, *args, **kwargs)
                else:
                    if (func.__name__ == "_toggle") and (args[0] is False):
                        # if toggling off
                        pass
                    else:
                        self.error_display.set(f"{self.log_ref} error. Click LED for details.")
                        raise DeviceError(self.error_dict["msg"])
            except AttributeError:
                # if not hasattr(self, "error_dict")
                return func(self, *args, **kwargs)

    return wrapper


class DeviceCheckerMetaClass(type):
    """
    meta-class which silently wraps every method
    with 'device_error_checker()'
    """

    def __new__(meta, classname, bases, classDict):
        newClassDict = {}
        for attributeName, attribute in classDict.items():
            if isinstance(attribute, FunctionType):
                # replace it with a wrapped version
                attribute = device_error_checker(attribute)
            newClassDict[attributeName] = attribute
        return type.__new__(meta, classname, bases, newClassDict)


class DeviceError(Exception):
    """Represents any error in the device"""

    pass


class IOError(Exception):
    """Represents an error in communication with the device"""

    pass
