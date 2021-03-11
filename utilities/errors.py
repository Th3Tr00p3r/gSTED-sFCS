# -*- coding: utf-8 -*-
"""Error handeling."""

import asyncio
import functools
import logging
import os
import sys
import traceback
from typing import Callable

from PyQt5.QtGui import QIcon

import gui.icons.icon_paths as icon
import utilities.constants as consts
from utilities.dialog import Error


def build_err_dict(exc: Exception) -> str:
    """Doc."""

    exc_type, _, tb = sys.exc_info()
    exc_type = exc_type.__name__
    frmtd_tb = "\n".join(traceback.format_tb(tb))

    return dict(exc_type=exc_type, exc_msg=str(exc), exc_tb=frmtd_tb)


def hndl_dvc_err(exc, dvc, func, lvl="ERROR"):
    """Doc."""

    exc_type, _, exc_tb = sys.exc_info()
    exc_name = exc_type.__name__
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    lineno = exc_tb.tb_lineno

    dvc_log_ref = getattr(consts, dvc.nick).log_ref
    log_str = (
        f"{exc_name}: {dvc_log_ref} didn't respond to '{func}' ({fname}, {lineno})"
    )

    if lvl == "ERROR":
        if dvc.error_dict is None:  # keep only first error
            dvc.error_dict = build_err_dict(exc)
        dvc.led_widget.set(QIcon(icon.LED_RED))
        logging.error(log_str, exc_info=False)

    else:  # "WARNING"
        logging.warning(log_str)


def dvc_error_checker(nick_set: set = None) -> Callable:
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

        except Exception as exc:
            Error(**build_err_dict(exc)).display()

    return wrapper_error_handler


def meas_err_hndlr(func: Callable) -> Callable:
    """Doc."""

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            """Doc."""

            try:
                return await func(*args, **kwargs)

            except Exception as exc:
                Error(**build_err_dict(exc)).display()

        return wrapper

    else:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Doc."""

            try:
                return func(*args, **kwargs)

            except Exception as exc:
                Error(**build_err_dict(exc)).display()

        return wrapper
