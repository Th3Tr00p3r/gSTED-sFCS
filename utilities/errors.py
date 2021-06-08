"""Error handeling."""

import asyncio
import functools
import logging
import os
import sys
import traceback
from types import SimpleNamespace
from typing import Callable

import utilities.constants as consts
from utilities.dialog import Error


def build_err_dict(exc: Exception) -> str:
    """Doc."""

    exc_type, _, tb = sys.exc_info()
    exc_type = exc_type.__name__
    frmtd_tb = "\n".join(traceback.format_tb(tb))
    fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
    lineno = tb.tb_lineno
    return dict(type=exc_type, msg=str(exc), tb=frmtd_tb, module=fname, line=lineno)


def err_hndlr(exc, func, lvl="error", dvc=None, disp=False):
    """Doc."""

    err_dict = build_err_dict(exc)

    if dvc is not None:  # device error
        dvc_log_ref = getattr(consts, dvc.nick).log_ref
        log_str = (
            f"{dvc_log_ref} didn't respond to '{func}' ({err_dict['module']}, {err_dict['line']}). "
            f"[{err_dict['type']}: {err_dict['msg']}]"
        )
        if lvl == "error":
            if dvc.error_dict is None:  # keep only first error
                dvc.error_dict = err_dict
            dvc.led_widget.set(consts.LED_ERROR_ICON)

    else:  # logic eror
        log_str = f"{err_dict['type']}: {err_dict['msg']} ({func}, {err_dict['module']}, {err_dict['line']})"
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

            if nick_set is None:
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
