# -*- coding: utf-8 -*-
"""Error handeling."""

import functools
import logging
import sys
import traceback

from instrumental.drivers.cameras.uc480 import UC480Error
from nidaqmx.errors import DaqError
from pyftdi.ftdi import FtdiError
from pyvisa.errors import VisaIOError

import utilities.constants as const
from utilities.dialog import Error


def build_err_msg(exc: Exception) -> str:
    """Doc."""

    exc_type, _, tb = sys.exc_info()
    exc_type = exc_type.__name__
    frmtd_tb = ("\n".join(traceback.format_tb(tb)),)

    return dict(exc_type=exc_type, exc_msg=str(exc), exc_tb=frmtd_tb[0])


def driver_error_handler(func):
    """decorator for clean handling of various known errors occuring in drivers.py."""
    # TODO: decide what to do with multiple errors - make a list (could explode?) or leave only the first?

    @functools.wraps(func)
    def wrapper_error_handler(dvc, *args, **kwargs):
        """Doc."""

        try:
            return func(dvc, *args, **kwargs)

        except ValueError as exc:
            if dvc.nick == "DEP_LASER":
                if not hasattr(dvc, "state"):  # initial toggle error
                    dvc.error_dict[dvc.nick] = build_err_msg(exc)
                    logging.error(
                        f'{const.DVC_LOG_DICT[dvc.nick]} didn\'t respond to {func.__name__}({args[0] if args != () else ""})',
                        exc_info=False,
                    )
                else:
                    logging.warning(
                        f'{const.DVC_LOG_DICT[dvc.nick]} didn\'t respond to {func.__name__}({args[0] if args != () else ""})',
                    )
                    return -999

            elif dvc.nick == "DEP_SHUTTER":
                logging.error(
                    f'{const.DVC_LOG_DICT[dvc.nick]} didn\'t respond to {func.__name__}({args[0] if args != () else ""})',
                    exc_info=False,
                )
                return False

            elif dvc.nick == "UM232":
                dvc.error_dict[dvc.nick] = build_err_msg(exc)
                logging.error(
                    f'{const.DVC_LOG_DICT[dvc.nick]} didn\'t respond to {func.__name__}({args[0] if args != () else ""})',
                    exc_info=False,
                )
            else:
                raise exc

        except DaqError as exc:
            dvc.error_dict[dvc.nick] = build_err_msg(exc)
            logging.error(
                f'{const.DVC_LOG_DICT[dvc.nick]} didn\'t respond to {func.__name__}({args[0] if args != () else ""})',
                exc_info=False,
            )

            if dvc.nick in {"EXC_LASER", "DEP_SHUTTER", "TDC"}:
                return False

        except VisaIOError as exc:
            dvc.error_dict[dvc.nick] = build_err_msg(exc)
            logging.error(
                f'{const.DVC_LOG_DICT[dvc.nick]} didn\'t respond to {func.__name__}({args[0] if args != () else ""})',
                exc_info=False,
            )

            if dvc.nick == "DEP_LASER":
                return -999

            if dvc.nick == "STAGE":
                return False

        except (AttributeError, OSError, FtdiError) as exc:
            if dvc.nick == "UM232":
                dvc.error_dict[dvc.nick] = build_err_msg(exc)
                logging.error(
                    f'{const.DVC_LOG_DICT[dvc.nick]} didn\'t respond to {func.__name__}({args[0] if args != () else ""})',
                    exc_info=False,
                )
                return False

            else:
                raise exc

        except UC480Error as exc:
            dvc.error_dict[dvc.nick] = build_err_msg(exc)
            logging.error(
                f'{const.DVC_LOG_DICT[dvc.nick]} didn\'t respond to {func.__name__}({args[0] if args != () else ""})',
                exc_info=False,
            )

    return wrapper_error_handler


def error_checker(nick_set=None):
    """
    Decorator for clean handeling of GUI interactions with errorneous devices.
    Checks for errors in devices associated with 'func' and shows error box
    if exist.

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


def logic_error_handler(func):
    """Doc."""

    @functools.wraps(func)
    def wrapper_error_handler(*args, **kwargs):
        """Doc."""

        try:
            return func(*args, **kwargs)

        except FileNotFoundError as exc:
            Error(**build_err_msg(exc)).display()

    return wrapper_error_handler
