# -*- coding: utf-8 -*-
"""Logic Module."""

import logging
import logging.config
import os
from types import SimpleNamespace

import yaml
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox

import gui.gui as gui_module
import gui.icons.icon_paths as icon
import logic.devices as devices
import utilities.constants as consts
import utilities.helper as helper
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
        self.gui = SimpleNamespace()

        self.gui.main = gui_module.MainWin(self)
        self.gui.main.imp.load(consts.DEFAULT_LOADOUT_FILE_PATH)

        self.gui.settings = gui_module.SettWin(self)
        self.gui.settings.imp.load(consts.DEFAULT_SETTINGS_FILE_PATH)

        self.gui.camera = gui_module.CamWin(self)
        # (instantiated on pressing camera button)

        # create neccessary data folders based on settings paths
        self.create_data_folders()

        # either error or ON
        self.gui.main.ledUm232h.setIcon(QIcon(icon.LED_GREEN))
        self.gui.main.ledCounter.setIcon(QIcon(icon.LED_GREEN))

        self.init_devices()
        self.meas = Measurement(app=self, type=None)

        # FINALLY
        self.gui.main.show()

        # set up main timeout event
        self.timeout_loop = Timeout(self)
        self.timeout_loop.start()

        self.devices.SCANNERS.last_ao = (
            self.gui.main.xAOV.value(),
            self.gui.main.yAOV.value(),
            self.gui.main.zAOV.value(),
        )

    def config_logging(self):
        """Doc."""

        with open("logging_config.yaml", "r") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)

    def create_data_folders(self):
        """Doc."""

        for gui_object_name in {"solDataPath", "imgDataPath", "camDataPath"}:
            rel_path = getattr(self.gui.settings, gui_object_name).text()
            os.makedirs(rel_path, exist_ok=True)

    def init_devices(self):
        """
        Goes through a list of device nicknames,
        instantiating a driver object for each device.
        """

        self.devices = SimpleNamespace()
        for nick in consts.DVC_NICKS_TUPLE:
            DVC_CONSTS = getattr(consts, nick)
            dvc_class = getattr(devices, DVC_CONSTS.cls_name)
            param_dict = DVC_CONSTS.param_widgets.read_dict_from_gui(self)

            gui_parent_name = DVC_CONSTS.led_widget.gui_parent_name
            led_widget = DVC_CONSTS.led_widget.hold_obj(
                parent_gui=getattr(self.gui, gui_parent_name)
            )

            if DVC_CONSTS.switch_widget is not None:
                gui_parent_name = DVC_CONSTS.switch_widget.gui_parent_name
                switch_widget = DVC_CONSTS.switch_widget.hold_obj(
                    parent_gui=getattr(self.gui, gui_parent_name)
                )
            else:
                switch_widget = None
            if DVC_CONSTS.cls_xtra_args is not None:
                x_args = [
                    helper.deep_getattr(self, deep_attr)
                    for deep_attr in DVC_CONSTS.cls_xtra_args
                ]
                setattr(
                    self.devices,
                    nick,
                    dvc_class(
                        nick,
                        param_dict,
                        led_widget,
                        switch_widget,
                        *x_args,
                    ),
                )
            else:
                setattr(
                    self.devices,
                    nick,
                    dvc_class(nick, param_dict, led_widget, switch_widget),
                )

    def clean_up_app(self, restart=False):
        """Doc."""

        def close_all_dvcs(app):
            """Doc."""

            for nick in consts.DVC_NICKS_TUPLE:
                dvc = getattr(app.devices, nick)
                if not dvc.error_dict:
                    dvc.toggle(False)

        def close_all_wins(app):
            """Doc."""

            for win_key in vars(self.gui).keys():
                if win_key == "settings":
                    # dialogs close with reject()
                    getattr(self.gui, win_key).reject()
                else:
                    # mainwindows and widgets close with close()
                    getattr(self.gui, win_key).close()

        def lights_out(gui):
            """turn OFF all device switch/LED icons"""

            led_list = [QIcon(icon.LED_OFF)] * 7 + [QIcon(icon.LED_GREEN)] * 2
            consts.LED_COLL.write_to_gui(self, led_list)
            consts.SWITCH_COLL.write_to_gui(self, QIcon(icon.SWITCH_OFF))
            gui.stageButtonsGroup.setEnabled(False)

        if restart:  # restarting

            if self.gui.camera is not None:
                self.gui.camera.close()

            if self.meas.type is not None:
                self.gui.main.imp.toggle_meas(self.meas.type)

            close_all_dvcs(self)

            self.timeout_loop.pause()

            lights_out(self.gui.main)
            self.gui.main.depActualCurr.setValue(0)
            self.gui.main.depActualPow.setValue(0)

            self.init_devices()

            self.timeout_loop.resume()

            logging.info("Restarting application.")

        else:  # exiting
            self.timeout_loop.finish()

            if self.meas.type is not None:
                self.gui.main.imp.toggle_meas(self.meas.type)

            close_all_wins(self)
            close_all_dvcs(self)
            logging.info("Quitting application.")

    def exit_app(self, event):
        """Doc."""

        pressed = Question(
            txt="Are you sure you want to quit?", title="Quitting Program"
        ).display()
        if pressed == QMessageBox.Yes:
            self.clean_up_app()
        else:
            event.ignore()
