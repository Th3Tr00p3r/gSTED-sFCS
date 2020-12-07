# -*- coding: utf-8 -*-
"""Logic Module."""

import time

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox

import gui.gui as gui_module
import gui.icons.icon_paths as icon
import logic.devices as devices
import utilities.constants as const
from logic.measurements import Measurement
from logic.timeout import Timeout
from utilities.dialog import Question
from utilities.log import Log


class App:
    """Doc."""

    def __init__(self):
        """Doc."""

        # init windows
        self.win_dict = {}
        self.win_dict["main"] = gui_module.MainWin(self)
        self.log = Log(self.win_dict["main"], dir_path="./log/")

        self.win_dict["settings"] = gui_module.SettWin(self)
        self.win_dict["settings"].imp.read_csv(const.DEFAULT_SETTINGS_FILE_PATH)

        self.win_dict["errors"] = gui_module.ErrWin(self)
        self.win_dict["camera"] = None  # instantiated on pressing camera button

        self.win_dict["main"].ledUm232.setIcon(
            QIcon(icon.LED_GREEN)
        )  # either error or ON
        self.win_dict["main"].ledCounter.setIcon(
            QIcon(icon.LED_GREEN)
        )  # either error or ON

        # initialize error dict
        self.init_errors()
        # initialize active devices
        self.init_devices()
        # initialize measurement
        self.meas = Measurement(self)

        # FINALLY
        self.win_dict["main"].show()
        # set up main timeout event
        self.timeout_loop = Timeout(self)

    def init_devices(self):
        """
        goes through a list of device nicknames,
        instantiating a driver object for each device.

        """

        def params_from_GUI(app, gui_dict):
            """
            Get counter parameters from settings GUI
            using a dictionary predefined in constants.py

            """

            param_dict = {}

            for key, val_dict in gui_dict.items():
                gui_field = getattr(app.win_dict["settings"], val_dict["field"])
                gui_field_value = getattr(gui_field, val_dict["access"])()
                param_dict[key] = gui_field_value

            return param_dict

        self.dvc_dict = {}
        for nick in const.DEVICE_NICKS:
            dvc_class = getattr(devices, const.DEVICE_CLASS_NAMES[nick])

            if nick in {"CAMERA"}:
                self.dvc_dict[nick] = dvc_class(nick=nick, error_dict=self.error_dict)

            else:
                param_dict = params_from_GUI(self, const.DVC_NICK_PARAMS_DICT[nick])
                self.dvc_dict[nick] = dvc_class(
                    nick=nick,
                    param_dict=param_dict,
                    error_dict=self.error_dict,
                )

    def init_errors(self):
        """Doc."""

        self.error_dict = {}
        for nick in const.DEVICE_NICKS:
            self.error_dict[nick] = None

    def clean_up_app(self, restart=False):
        """Close all devices and secondary windows before closing/restarting application."""

        def close_all_dvcs(app):
            """Doc."""

            for nick in const.DEVICE_NICKS:
                #                if not self.error_dict[nick]:
                app.dvc_dict[nick].toggle(False)

        def close_all_wins(app):
            """Doc."""

            for win_key in self.win_dict.keys():
                if self.win_dict[win_key] is not None:  # can happen for camwin
                    if win_key not in {
                        "main",
                        "camera",
                    }:  # dialogs close with reject()
                        self.win_dict[win_key].reject()
                    else:  # mainwindows and widgets close with close()
                        self.win_dict[win_key].close()

        def lights_out(gui):
            """turn OFF all device switch/LED icons"""

            gui.excOnButton.setIcon(QIcon(icon.SWITCH_OFF))
            gui.ledExc.setIcon(QIcon(icon.LED_OFF))
            gui.depEmissionOn.setIcon(QIcon(icon.SWITCH_OFF))
            gui.ledDep.setIcon(QIcon(icon.LED_OFF))
            gui.depShutterOn.setIcon(QIcon(icon.SWITCH_OFF))
            gui.ledShutter.setIcon(QIcon(icon.LED_OFF))
            gui.stageOn.setIcon(QIcon(icon.SWITCH_OFF))
            gui.ledStage.setIcon(QIcon(icon.LED_OFF))
            gui.stageButtonsGroup.setEnabled(False)
            gui.ledUm232.setIcon(QIcon(icon.LED_GREEN))  # either error or ON
            gui.ledTdc.setIcon(QIcon(icon.LED_OFF))
            gui.ledCounter.setIcon(QIcon(icon.LED_GREEN))  # either error or ON
            gui.ledCam.setIcon(QIcon(icon.LED_OFF))

        if self.meas.type is not None:
            if self.meas.type == "FCS":
                self.win_dict["main"].imp.toggle_FCS_meas()

        close_all_dvcs(self)

        if restart:

            if self.win_dict["camera"] is not None:
                self.win_dict["camera"].close()

            self.timeout_loop.stop()

            lights_out(self.win_dict["main"])
            self.win_dict["main"].depActualCurrSpinner.setValue(0)
            self.win_dict["main"].depActualPowerSpinner.setValue(0)

            self.init_errors()
            self.init_devices()
            time.sleep(0.2)  # needed to avoid error with main timeout
            self.timeout_loop = Timeout(self)
            self.log.update("restarting application.", tag="verbose")

        else:
            close_all_wins(self)
            self.log.update("Quitting Application.")

    def exit_app(self, event):
        """Doc."""

        pressed = Question(
            q_txt="Are you sure you want to quit?", q_title="Quitting Program"
        ).display()
        if pressed == QMessageBox.Yes:
            self.timeout_loop.stop()
            self.clean_up_app()
        else:
            event.ignore()
