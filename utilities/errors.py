"""Error handeling."""

import asyncio
import functools
import logging
import sys
import traceback
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Dict

from utilities.dialog import ErrorDialog


def build_error_dict(exc: Exception) -> Dict[str, Any]:
    """Doc."""

    exc_type = exc.__class__.__name__
    _, _, tb = sys.exc_info()
    frmtd_tb = "\n".join(traceback.format_tb(tb))
    # show the first 'n' existing levels of traceback for module and line number
    exc_loc = []
    while tb is not None:
        *_, filename = Path(tb.tb_frame.f_code.co_filename).parts
        lineno = tb.tb_lineno
        exc_loc.append((filename, lineno))
        tb = tb.tb_next
    return dict(type=exc_type, loc=exc_loc, msg=str(exc), tb=frmtd_tb)


def get_frame_details(frame, locals):
    """
    Create a set of all argument names except 'self', and use them to filter the 'locals()' dictionary,
    leaving only external arguments and their values in the final string. Also get the function name.
    """

    frame_code = frame.f_code
    func_name = frame_code.co_name
    func_arg_names = set(frame_code.co_varnames[: frame_code.co_argcount]) - {"self"}
    if locals is not None:
        arg_string = ", ".join(
            [f"{key}={str(val)}" for key, val in locals.items() if key in func_arg_names]
        )
        return f"{func_name}({arg_string})"
    else:
        return f"{func_name}()"


def err_hndlr(exc, func_frame, func_locals, dvc=None, lvl="error", disp=False) -> str:
    """Doc."""

    error_dict = build_error_dict(exc)
    func_string = get_frame_details(func_frame, func_locals)
    location_string = " -> ".join(
        [f"{filename}, {lineno}" for filename, lineno in error_dict["loc"]]
    )

    if dvc is not None:  # device error
        log_str = (
            f"{dvc.log_ref} didn't respond to '{func_string}' ({location_string}). "
            f"[{error_dict['type']}: {error_dict['msg']}]"
        )
        if lvl == "error":
            if not dvc.error_dict:  # keep only first error
                dvc.error_dict = error_dict
            dvc.change_icons("error")
            dvc.disable_switch()

    else:  # logic eror
        # TODO: this module shouldn't import the gui module...
        log_str = f"{error_dict['type']}: {error_dict['msg']} ({func_string}, {location_string})"
        if disp:
            ErrorDialog(**error_dict).display()

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

            if not self.error_dict:
                return await func(self, *args, **kwargs)
            else:
                self.error_display.set(f"{self.log_ref} error. Click LED for details.")
                raise DeviceError(self.error_dict["msg"])

    else:

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            if not self.error_dict:
                return func(self, *args, **kwargs)
            else:
                self.error_display.set(f"{self.log_ref} error. Click LED for details.")
                raise DeviceError(self.error_dict["msg"])

    return wrapper


class DeviceCheckerMetaClass(type):
    """
    meta-class which silently wraps every method
    with 'device_error_checker()'
    """

    ignored_attributes = {
        "__init__",
        "__module__",
        "__doc__",
        "__qualname__",
        "__classcell__",
        "close",
    }

    def __new__(self, classname, bases, classDict):
        newClassDict = {}
        for attribute_name, attribute in classDict.items():
            if (
                isinstance(attribute, FunctionType)
                and attribute_name not in self.ignored_attributes
            ):
                # replace it with a wrapped version
                attribute = device_error_checker(attribute)
            newClassDict[attribute_name] = attribute
        return type.__new__(self, classname, bases, newClassDict)


class DeviceError(Exception):
    """Represents any error in the device"""

    pass


class IOError(Exception):
    """Represents an error in communication with the device"""

    pass
