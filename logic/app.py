# -*- coding: utf-8 -*-
"""Logic Module."""

import logging
import logging.config
import os

import yaml
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox

import gui.gui as gui_module
import gui.icons.icon_paths as icon
import logic.devices as devices
import utilities.constants as consts
from logic.measurements import Measurement
from logic.timeout import Timeout
from utilities.dialog import Question


class App:
    """Doc."""

    def __init__(self, loop):
        """Doc."""
        # refactor this init so that it uses clean_up app method - split it into a once-occuring setup, and the rest in clean_up can be repeated in restsrts

        self.loop = loop

        # init logging
        self.config_logging()
        logging.info("Application Started")

        # init windows
        self.gui_dict = {}
        self.gui_dict["main"] = gui_module.MainWin(self)
        self.gui_dict["main"].imp.load(consts.DEFAULT_LOADOUT_FILE_PATH)

        self.gui_dict["settings"] = gui_module.SettWin(self)
        self.gui_dict["settings"].imp.load(consts.DEFAULT_SETTINGS_FILE_PATH)

        self.gui_dict["camera"] = gui_module.CamWin(
            self
        )  # instantiated on pressing camera button

        # create neccessary data folders based on settings paths
        self.create_data_folders()

        # either error or ON
        self.gui_dict["main"].ledUm232h.setIcon(QIcon(icon.LED_GREEN))
        self.gui_dict["main"].ledCounter.setIcon(QIcon(icon.LED_GREEN))

        self.init_errors()
        self.init_devices()
        self.meas = Measurement(app=self, type=None)

        # FINALLY
        self.gui_dict["main"].show()

        # set up main timeout event
        self.timeout_loop = Timeout(self)
        self.timeout_loop.start()

    def config_logging(self):
        """Doc."""

        with open("logging_config.yaml", "r") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)

    def create_data_folders(self):
        """Doc."""

        for gui_object_name in {"solDataPath", "imgDataPath", "camDataPath"}:
            rel_path = getattr(self.gui_dict["settings"], gui_object_name).text()
            os.makedirs(rel_path, exist_ok=True)

    def init_devices(self):
        """
        Goes through a list of device nicknames,
        instantiating a driver object for each device.
        """

        def params_from_GUI(app: App, param_widgets: consts.ParamWidgets):
            """
            Get constant device parameters from GUI
            using ParamWidgets objects defined in constants.py.
            """

            param_dict = {}
            for param_name, widget_access in vars(param_widgets).items():
                parent_gui = app.gui_dict[widget_access.gui_parent_name]
                param_dict[param_name] = widget_access.access(parent_gui=parent_gui)
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
        for nick in consts.DVC_NICKS_TUPLE:
            DVC_CONSTS = getattr(consts, nick)
            dvc_class = getattr(devices, DVC_CONSTS.cls_name)
            param_dict = params_from_GUI(self, DVC_CONSTS.param_widgets)

            gui_parent_name = DVC_CONSTS.led_widget.gui_parent_name
            led_widget = DVC_CONSTS.led_widget.hold_obj(
                parent_gui=self.gui_dict[gui_parent_name]
            )

            if DVC_CONSTS.switch_widget is not None:
                gui_parent_name = DVC_CONSTS.switch_widget.gui_parent_name
                switch_widget = DVC_CONSTS.switch_widget.hold_obj(
                    parent_gui=self.gui_dict[gui_parent_name]
                )
            else:
                switch_widget = None
            x_args = extra_args(self, DVC_CONSTS.cls_xtra_args)
            if x_args:
                self.dvc_dict[nick] = dvc_class(
                    nick,
                    param_dict,
                    self.error_dict,
                    led_widget,
                    switch_widget,
                    *x_args,
                )
            else:
                self.dvc_dict[nick] = dvc_class(
                    nick, param_dict, self.error_dict, led_widget, switch_widget
                )

    def init_errors(self):
        """Doc."""

        self.error_dict = {}
        for nick in consts.DVC_NICKS_TUPLE:
            self.error_dict[nick] = None

    def clean_up_app(self, restart=False):
        """Doc."""

        def close_all_dvcs(app):
            """Doc."""

            for nick in consts.DVC_NICKS_TUPLE:
                if not self.error_dict[nick]:
                    app.dvc_dict[nick].toggle(False)

        def close_all_wins(app):
            """Doc."""

            for win_key in self.gui_dict.keys():
                if self.gui_dict[win_key] is not None:  # can happen for camwin
                    if win_key not in {
                        "main",
                        "camera",
                    }:  # dialogs close with reject()
                        self.gui_dict[win_key].reject()
                    else:  # mainwindows and widgets close with close()
                        self.gui_dict[win_key].close()

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
            gui.ledUm232h.setIcon(QIcon(icon.LED_GREEN))  # either error or ON
            gui.ledTdc.setIcon(QIcon(icon.LED_OFF))
            gui.ledCounter.setIcon(QIcon(icon.LED_GREEN))  # either error or ON
            gui.ledCam.setIcon(QIcon(icon.LED_OFF))
            gui.ledScn.setIcon(QIcon(icon.LED_OFF))

        if restart:  # restarting

            if self.gui_dict["camera"] is not None:
                self.gui_dict["camera"].close()

            if self.meas.type is not None:
                self.gui_dict["main"].imp.toggle_meas(self.meas.type)

            close_all_dvcs(self)

            self.timeout_loop.pause()

            lights_out(self.gui_dict["main"])
            self.gui_dict["main"].depActualCurrSpinner.setValue(0)
            self.gui_dict["main"].depActualPowerSpinner.setValue(0)

            self.init_errors()
            self.init_devices()

            self.timeout_loop.resume()

            logging.info("Restarting application.")

        else:  # exiting
            self.timeout_loop.finish()

            if self.meas.type is not None:
                self.gui_dict["main"].imp.toggle_meas(self.meas.type)

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
