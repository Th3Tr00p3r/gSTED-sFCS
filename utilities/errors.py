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
    fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
    lineno = tb.tb_lineno

    return dict(type=exc_type, msg=str(exc), tb=frmtd_tb, module=fname, line=lineno)


def err_hndlr(exc, func, lvl="ERROR", dvc=None):
    """Doc."""

    err_dict = build_err_dict(exc)

    if dvc is not None:
        dvc_log_ref = getattr(consts, dvc.nick).log_ref
        log_str = f"{err_dict['type']}: {dvc_log_ref} didn't respond to '{func}' ({err_dict['module']}, {err_dict['line']})"
        if lvl == "ERROR":
            if dvc.error_dict is None:  # keep only first error
                dvc.error_dict = err_dict
            dvc.led_widget.set(QIcon(icon.LED_RED))

    else:  # logic eror
        log_str = (
            f"{func}: {err_dict['msg']} ({err_dict['module']}, {err_dict['line']})"
        )
        Error(err_dict).display()

    if lvl == "ERROR":
        logging.error(log_str, exc_info=False)
    else:  # "WARNING"
        logging.warning(log_str)


def dvc_err_chckr(nick_set: set = None) -> Callable:
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
