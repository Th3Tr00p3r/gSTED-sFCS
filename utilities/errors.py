"""Error handeling."""

import asyncio
import functools
import logging
import os
import sys
import traceback
from types import SimpleNamespace
from typing import Callable

from PyQt5.QtGui import QIcon

import gui
import logic.devices as dvcs
from utilities.dialog import Error


def build_err_dict(exc: Exception) -> str:
    """Doc."""

    exc_type, _, tb = sys.exc_info()
    exc_type = exc_type.__name__
    frmtd_tb = "\n".join(traceback.format_tb(tb))
    # show the first 'n' existing levels of traceback for module and line number
    exc_loc = []
    while tb is not None:
        filename = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        lineno = tb.tb_lineno
        exc_loc.append((filename, lineno))
        tb = tb.tb_next
    return dict(type=exc_type, loc=exc_loc, msg=str(exc), tb=frmtd_tb)


def err_hndlr(exc, func_locals, func_frame, lvl="error", dvc=None, disp=False):
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

    err_dict = build_err_dict(exc)
    func_string = get_frame_details(func_frame, func_locals)
    location_string = " -> ".join([f"{filename}, {lineno}" for filename, lineno in err_dict["loc"]])

    if dvc is not None:  # device error
        dvc_log_ref = dvcs.DEVICE_ATTR_DICT[dvc.nick].log_ref
        log_str = (
            f"{dvc_log_ref} didn't respond to '{func_string}' ({location_string}). "
            f"[{err_dict['type']}: {err_dict['msg']}]"
        )
        if lvl == "error":
            if not dvc.error_dict:  # keep only first error
                dvc.error_dict = err_dict
            dvc.led_widget.set(QIcon(gui.icons.ICON_PATHS_DICT["led_red"]))

    else:  # logic eror
        log_str = f"{err_dict['type']}: {err_dict['msg']} ({func_string}, {location_string})"
        if disp:
            Error(**err_dict).display()

    getattr(logging, lvl)(log_str, exc_info=False)


# TODO: allow toggle(off) on error!
def dvc_err_chckr(nick_set: set = None) -> Callable:
    """
    Decorator for clean handeling of GUI interactions with errorneous devices.
    Checks for errors in devices associated with 'func' and shows error box
    if exist.

    nick_set - a set of all device nicks to check for errors
        before attempting the decorated func()
    """

    def outer_wrapper(func) -> Callable:
        def check(nick_set, devices: SimpleNamespace, func_args) -> bool:
            """
            Checks for error in specified devices
            and displays informative error messages to user.
            """

            if not nick_set:
                nick_set = {func_args[0]}

            txt = [f"{nick} error.\n" for nick in nick_set if getattr(devices, nick).error_dict]

            if txt:
                # if any errors found for specified devices
                txt.append("\nClick relevant LED for details.")
                Error(custom_txt="".join(txt)).display()
                return False
            else:
                # no errors found
                return True

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def inner_wrapper(self, *args, **kwargs):

                if check(nick_set, self._app.devices, args) is True:
                    return await func(self, *args, **kwargs)

        else:

            @functools.wraps(func)
            def inner_wrapper(self, *args, **kwargs):

                if check(nick_set, self._app.devices, args) is True:
                    return func(self, *args, **kwargs)

        return inner_wrapper

    return outer_wrapper
