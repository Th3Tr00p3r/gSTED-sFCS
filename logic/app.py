# -*- coding: utf-8 -*-
"""Logic Module."""

import logging
import logging.config

import yaml
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox

import gui.gui as gui_module
import gui.icons.icon_paths as icon
import logic.devices as devices
import utilities.constants as const
from logic.measurements import Measurement
from logic.timeout import Timeout
from utilities.dialog import Question


class App:
    """Doc."""

    def __init__(self, loop):
        """Doc."""

        self.loop = loop

        # init logging
        self.config_logging()
        logging.info("Application Started")

        # init windows
        self.win_dict = {}
        self.win_dict["main"] = gui_module.MainWin(self)

        self.win_dict["settings"] = gui_module.SettWin(self)
        self.win_dict["settings"].imp.read_csv(const.DEFAULT_SETTINGS_FILE_PATH)

        self.win_dict["camera"] = gui_module.CamWin(
            self
        )  # instantiated on pressing camera button

        # either error or ON
        self.win_dict["main"].ledUm232.setIcon(QIcon(icon.LED_GREEN))
        self.win_dict["main"].ledCounter.setIcon(QIcon(icon.LED_GREEN))

        self.init_errors()
        self.init_devices()
        self.meas = Measurement(self)

        # FINALLY
        self.win_dict["main"].show()

        # set up main timeout event
        self.timeout_loop = Timeout(self)
        self.timeout_loop.start()

    def config_logging(self):
        """Doc."""

        with open("logging_config.yaml", "r") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)

    def init_devices(self):
        """
        Goes through a list of device nicknames,
        instantiating a driver object for each device.
        """

        def params_from_GUI(app, gui_dict):
            """
            Get counter parameters from settings GUI
            using a dictionary predefined in constants.py.
            """

            param_dict = {}
            for key, val_dict in gui_dict.items():
                gui_field = getattr(app.win_dict["settings"], val_dict["field"])
                gui_field_value = getattr(gui_field, val_dict["access"])()
                param_dict[key] = gui_field_value
            return param_dict

        def extra_args(app, x_args: list) -> list:
            """
            Add additional parameters to device using a dictionary
            predefined in constants.py.
            """
            # TODO: this function uses eval(), there should be a better way to refactor this code

            args = []
            if x_args:
                for var_str in x_args:
                    args.append(eval(var_str))

            return args

        self.dvc_dict = {}
        for nick in const.DEVICE_NICKS:
            dvc_class = getattr(devices, const.DVC_CLASS_NAMES[nick])
            param_dict = params_from_GUI(self, const.DVC_NICK_PARAMS_DICT[nick])
            led = getattr(self.win_dict["main"], const.ICON_DICT[nick]["LED"])
            x_args = extra_args(self, const.DVC_X_ARGS_DICT[nick])
            if x_args:
                self.dvc_dict[nick] = dvc_class(
                    nick, param_dict, self.error_dict, led, *x_args
                )
            else:
                self.dvc_dict[nick] = dvc_class(nick, param_dict, self.error_dict, led)

    def init_errors(self):
        """Doc."""

        self.error_dict = {}
        for nick in const.DEVICE_NICKS:
            self.error_dict[nick] = None

    def clean_up_app(self, restart=False):
        """Doc."""

        def close_all_dvcs(app):
            """Doc."""

            for nick in const.DEVICE_NICKS:
                if nick in {"DEP_LASER"}:
                    if not self.error_dict[nick]:
                        app.dvc_dict[nick].toggle(False)
                else:
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
            gui.ledScn.setIcon(QIcon(icon.LED_OFF))

        if restart:  # restarting
            if self.meas.type is not None:
                if self.meas.type == "FCS":
                    self.win_dict["main"].imp.toggle_FCS_meas()

            close_all_dvcs(self)

            if self.win_dict["camera"] is not None:
                self.win_dict["camera"].close()

            self.timeout_loop.pause()

            lights_out(self.win_dict["main"])
            self.win_dict["main"].depActualCurrSpinner.setValue(0)
            self.win_dict["main"].depActualPowerSpinner.setValue(0)

            self.init_errors()
            self.init_devices()

            self.timeout_loop.resume()

            logging.info("Restarting application.")

        else:  # exiting
            self.loop.create_task(
                self.timeout_loop.finish()
            )  # TODO: shouldn't this be 'await'ed instead?
            if self.meas.type is not None:
                if self.meas.type == "FCS":
                    self.win_dict["main"].imp.toggle_FCS_meas()

            close_all_wins(self)
            close_all_dvcs(self)
            logging.info("Quitting application.")

    def exit_app(self, event):
        """Doc."""

        pressed = Question(
            q_txt="Are you sure you want to quit?", q_title="Quitting Program"
        ).display()
        if pressed == QMessageBox.Yes:
            self.clean_up_app()
        else:
            event.ignore()
