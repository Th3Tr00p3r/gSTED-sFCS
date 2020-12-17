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
from pyvisa.errors import VisaIOError

import utilities.constants as const
from utilities.dialog import Error


class CounterError(Exception):
    """Doc."""

    def __init__(self, msg):
        super().__init__(msg)


def build_err_dict(exc: Exception) -> str:
    """Doc."""

    exc_type, _, tb = sys.exc_info()
    exc_type = exc_type.__name__
    frmtd_tb = ("\n".join(traceback.format_tb(tb)),)

    return dict(exc_type=exc_type, exc_msg=str(exc), exc_tb=frmtd_tb[0])


def parse_args(args: tuple) -> str:
    """Doc."""

    if args == ():
        cmnd = ""
    else:
        cmnd, *_ = args
        cmnd = '"' + str(cmnd) + '"'
    return cmnd


def resolve_dvc_exc(exc: Exception, func_name: str, cmnd: str, dvc) -> int:
    """Decides what to do with caught, device-related exceptions"""

    log_str = (
        f"{const.DVC_LOG_DICT[dvc.nick]} didn't respond to {func_name}({cmnd}) call"
    )

    if isinstance(exc, ValueError):
        if dvc.nick == "DEP_LASER":
            if not hasattr(dvc, "state"):  # initial toggle error
                dvc.error_dict[dvc.nick] = build_err_dict(exc)
                logging.error(log_str, exc_info=False)
            else:
                logging.warning(log_str)
                return -999

        elif dvc.nick in {"DEP_SHUTTER", "UM232"}:
            dvc.error_dict[dvc.nick] = build_err_dict(exc)
            logging.error(log_str, exc_info=False)
            return 0

        else:
            raise exc

    elif isinstance(exc, DaqError):
        dvc.error_dict[dvc.nick] = build_err_dict(exc)
        logging.error(log_str, exc_info=False)

        if dvc.nick in {"EXC_LASER", "DEP_SHUTTER", "TDC"}:
            return 0

    elif isinstance(exc, VisaIOError):
        if dvc.nick == "DEP_LASER":
            dvc.error_dict[dvc.nick] = build_err_dict(exc)
            logging.error(log_str, exc_info=False)
            return -999

        if dvc.nick == "STAGE":
            dvc.error_dict[dvc.nick] = build_err_dict(exc)
            logging.error(log_str, exc_info=False)
            return 0

        else:
            raise

    elif isinstance(exc, (AttributeError, OSError, FtdiError)):
        if dvc.nick == "UM232":
            dvc.error_dict[dvc.nick] = build_err_dict(exc)
            logging.error(log_str, exc_info=False)
            return 0

        else:
            raise exc

    elif isinstance(exc, UC480Error):
        dvc.error_dict[dvc.nick] = build_err_dict(exc)
        logging.error(log_str, exc_info=False)

    elif isinstance(exc, TypeError):
        if dvc.nick == "CAMERA":
            logging.warning(log_str)

    else:
        raise exc


def dvc_err_hndlr(func) -> Callable:
    """Decorator for clean handling of various known device errors."""
    # TODO: (low priority) decide what to do with multiple errors - make a list (could explode?) or leave only the first?

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

        @functools.wraps(func)
        def inner_wrapper(self, *args, **kwargs):
            """Doc."""

            if nick_set is not None:
                count = 0
                txt = ""
                for nick in nick_set:
                    err_msg = self._app.error_dict[nick]

                    if err_msg is not None:
                        txt += f"{nick} error.\n"
                        count += 1

                if count > 0:
                    txt += "\nClick relevant LED for details."
                    Error(custom_txt=txt, custom_title=f"Errors ({count})").display()

                else:
                    return func(self, *args, **kwargs)

            else:
                nick = args[0]
                err_msg = self._app.error_dict[nick]

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
