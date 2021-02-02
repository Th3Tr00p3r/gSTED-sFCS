# -*- coding: utf-8 -*-
"""Error handeling."""

import asyncio
import functools
import logging
import sys
import traceback
from typing import Callable

from instrumental.drivers.cameras.uc480 import UC480Error
from nidaqmx.errors import DaqError
from pyftdi.ftdi import FtdiError
from PyQt5.QtGui import QIcon
from pyvisa.errors import VisaIOError

import gui.icons.icon_paths as icon
import utilities.constants as consts
from utilities.dialog import Error


class CounterError(IOError):
    """Doc."""


def build_err_dict(exc: Exception) -> str:
    """Doc."""

    exc_type, _, tb = sys.exc_info()
    exc_type = exc_type.__name__
    frmtd_tb = "\n".join(traceback.format_tb(tb))

    return dict(exc_type=exc_type, exc_msg=str(exc), exc_tb=frmtd_tb)


def parse_args(args: tuple) -> str:
    """Doc."""

    if args == ():
        cmnd = ""
    else:
        cmnd, *_ = args
        cmnd = "'" + str(cmnd) + "'"
    return cmnd


def resolve_dvc_exc(exc: Exception, func_name: str, cmnd: str, dvc) -> int:
    """Decides what to do with caught, device-related exceptions"""

    dvc_log_ref = getattr(consts, dvc.nick).log_ref
    log_str = f"{dvc_log_ref} didn't respond to {func_name}({cmnd}) call"
    lvl = "ERROR"

    if isinstance(exc, ValueError):
        if dvc.nick == "DEP_LASER":
            if dvc.state is None:  # initial toggle error
                result = 0
            else:
                lvl = "WARNING"
                result = -999

        elif dvc.nick in {"DEP_SHUTTER", "UM232H"}:
            result = 0
        else:
            raise exc

    elif isinstance(exc, DaqError) and dvc.nick in {
        "EXC_LASER",
        "DEP_SHUTTER",
        "TDC",
        "COUNTER",
        "SCANNERS",
    }:
        result = 0

    elif isinstance(exc, VisaIOError):
        if dvc.nick == "DEP_LASER":
            result = -999

        elif dvc.nick == "STAGE":
            result = 0
        else:
            raise exc

    elif isinstance(exc, (AttributeError, OSError, FtdiError)) and dvc.nick == "UM232H":
        result = 0

    elif isinstance(exc, UC480Error) and dvc.nick == "CAMERA":
        result = 0

    elif isinstance(exc, TypeError) and dvc.nick == "CAMERA":
        lvl = "WARNING"
        result = 0

    else:
        raise exc

    if lvl == "ERROR":

        if dvc.error_dict is None:  # keep only first error
            dvc.error_dict = build_err_dict(exc)

        dvc.led_widget.set(QIcon(icon.LED_RED))
        logging.error(log_str, exc_info=False)

    else:  # "WARNING"
        logging.warning(log_str)

    return result


def dvc_err_hndlr(func) -> Callable:
    """Decorator for clean handling of various known device errors."""

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(dvc, *args, **kwargs):
            """Doc."""

            try:
                return await func(dvc, *args, **kwargs)

            except Exception as exc:
                cmnd = parse_args(args)
                return resolve_dvc_exc(exc, func.__name__, cmnd, dvc)

    else:

        @functools.wraps(func)
        def wrapper(dvc, *args, **kwargs):
            """Doc."""

            try:
                return func(dvc, *args, **kwargs)

            except Exception as exc:
                cmnd = parse_args(args)
                return resolve_dvc_exc(exc, func.__name__, cmnd, dvc)

    return wrapper


def error_checker(nick_set: set = None) -> Callable:
    """
    Decorator for clean handeling of GUI interactions with errorneous devices.
    Checks for errors in devices associated with 'func' and shows error box
    if exist.

    nick_set - a set of all device nicks to check for errors
        before attempting the decorated func()

    """

    def outer_wrapper(func) -> Callable:
        """Doc."""

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def inner_wrapper(self, *args, **kwargs):
                """Doc."""

                if nick_set is not None:
                    count = 0
                    txt = ""
                    for nick in nick_set:

                        if getattr(self._app.devices, nick).error_dict is not None:
                            txt += f"{nick} error.\n"
                            count += 1

                    if count > 0:
                        txt += "\nClick relevant LED for details."
                        Error(
                            custom_txt=txt, custom_title=f"Errors ({count})"
                        ).display()

                    else:
                        return await func(self, *args, **kwargs)

                else:
                    nick = args[0]
                    err_msg = getattr(self._app.devices, nick).error_dict

                    if err_msg is not None:
                        txt = f"{nick} error.\n\nClick relevant LED for details."
                        Error(custom_txt=txt).display()
                    else:
                        return await func(self, *args, **kwargs)

        else:

            @functools.wraps(func)
            def inner_wrapper(self, *args, **kwargs):
                """Doc."""

                if nick_set is not None:
                    count = 0
                    txt = ""
                    for nick in nick_set:

                        if getattr(self._app.devices, nick).error_dict is not None:
                            txt += f"{nick} error.\n"
                            count += 1

                    if count > 0:
                        txt += "\nClick relevant LED for details."
                        Error(
                            custom_txt=txt, custom_title=f"Errors ({count})"
                        ).display()

                    else:
                        return func(self, *args, **kwargs)

                else:
                    nick = args[0]
                    err_msg = getattr(self._app.devices, nick).error_dict

                    if err_msg is not None:
                        txt = f"{nick} error.\n\nClick relevant LED for details."
                        Error(custom_txt=txt).display()
                    else:
                        return func(self, *args, **kwargs)

        return inner_wrapper

    return outer_wrapper


def logic_error_handler(func: Callable) -> Callable:
    """Doc."""

    @functools.wraps(func)
    def wrapper_error_handler(*args, **kwargs):
        """Doc."""

        try:
            return func(*args, **kwargs)

        except FileNotFoundError as exc:
            Error(**build_err_dict(exc)).display()

    return wrapper_error_handler
