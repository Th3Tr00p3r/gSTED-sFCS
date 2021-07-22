"""Logic Module."""

import logging
import logging.config
import os
from types import SimpleNamespace

import yaml
from PyQt5.QtWidgets import QMessageBox

import gui.gui
import logic.devices as dvcs
import utilities.helper as helper
import utilities.widget_collections as wdgt_colls
from logic.timeout import Timeout
from utilities.dialog import Question

DEFAULT_LOADOUT_FILE_PATH = "./settings/loadouts/default_loadout.csv"
DEFAULT_SETTINGS_FILE_PATH = "./settings/default_settings.csv"
DVC_NICKS = (
    "exc_laser",
    "dep_shutter",
    "TDC",
    "dep_laser",
    "stage",
    "UM232H",
    "camera",
    "scanners",
    "photon_detector",
    "pixel_clock",
)


class App:
    """Doc."""

    def __init__(self, loop):
        """Doc."""

        self.config_logging()

        self.loop = loop

        self.meas = SimpleNamespace(type=None, is_running=False)

        self.analysis = SimpleNamespace()
        self.analysis.loaded_data = dict()

        # get icons
        self.icon_dict = helper.paths_to_icons(gui.icons.icon_paths_dict)

        # init windows
        print("Initializing GUI...", end=" ")
        self.gui = SimpleNamespace()
        self.gui.main = gui.gui.MainWin(self)
        self.gui.main.imp.load(
            DEFAULT_LOADOUT_FILE_PATH
        )  # self.gui.main.isDataCalibration.isChecked() TESTESTEST
        self.gui.settings = gui.gui.SettWin(self)
        self.gui.settings.imp.load(DEFAULT_SETTINGS_FILE_PATH)
        self.gui.camera = gui.gui.CamWin(self)  # instantiated on pressing camera button

        # populate all widget collections in 'utilities.widget_collections' with objects
        [
            val.hold_objects(app=self)
            for val in wdgt_colls.__dict__.values()
            if isinstance(val, helper.QtWidgetCollection)
        ]

        # create neccessary data folders based on settings paths
        self.create_data_folders()

        # either error or ON
        self.gui.main.ledScn.setIcon(self.icon_dict["led_green"])
        self.gui.main.ledCounter.setIcon(self.icon_dict["led_green"])
        self.gui.main.ledUm232h.setIcon(self.icon_dict["led_green"])

        print("Done.")

        print("Initializing Devices...", end=" ")
        self.init_devices()
        print("Done.")

        # init AO as origin (actual AO is measured in internal AO if last position is needed)
        [
            getattr(self.gui.main, f"{axis}AOV").setValue(org_vltg)
            for axis, org_vltg in zip("xyz", self.devices.scanners.ORIGIN)
        ]

        # init scan patterns
        print(
            "Displaying patterns for first time (Numba first compilation might be slow).", end=" "
        )
        self.gui.main.imp.disp_scn_pttrn("image")
        sol_pattern = wdgt_colls.sol_meas_wdgts.read_namespace_from_gui(self).scan_type
        self.gui.main.imp.disp_scn_pttrn(sol_pattern)
        print("Done.")

        # init existing data folders (solution by default)
        self.gui.main.solDataImport.setChecked(True)
        self.gui.main.imp.populate_all_data_dates()

        # show the GUI
        self.gui.main.show()

        # set up main timeout event
        print("Initializing timeout loop...", end=" ")
        self.timeout_loop = Timeout(self)
        print("Done.")

        print("Application Initialized.")
        logging.info("Application Started")

    def config_logging(self):
        """Configure the logging package for the whole application."""

        with open("logging_config.yaml", "r") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)

    def create_data_folders(self):
        """Doc."""

        for gui_object_name in {"dataPath", "camDataPath"}:
            rel_path = getattr(self.gui.settings, gui_object_name).text()
            os.makedirs(rel_path, exist_ok=True)

    def init_devices(self):
        """
        Goes through a list of device nicknames,
        instantiating a driver object for each device.
        """

        self.devices = SimpleNamespace()
        for nick in DVC_NICKS:
            dvc_attrs = dvcs.DEVICE_ATTR_DICT[nick]
            dvc_class = getattr(dvcs, dvc_attrs.class_name)
            param_dict = dvc_attrs.param_widgets.hold_objects(app=self).read_dict_from_gui(self)
            param_dict["nick"] = nick
            param_dict["log_ref"] = dvc_attrs.log_ref
            param_dict["led_icon"] = self.icon_dict[f"led_{dvc_attrs.led_color}"]
            param_dict["error_display"] = helper.QtWidgetAccess(
                "deviceErrorDisplay", "text", "main", True
            ).hold_obj(self.gui.main)

            if dvc_attrs.cls_xtra_args:
                x_args = [
                    helper.deep_getattr(self, deep_attr) for deep_attr in dvc_attrs.cls_xtra_args
                ]
                setattr(
                    self.devices,
                    nick,
                    dvc_class(
                        param_dict,
                        *x_args,
                    ),
                )
            else:
                setattr(
                    self.devices,
                    nick,
                    dvc_class(param_dict),
                )

    async def clean_up_app(self, restart=False):
        """Doc."""

        def close_all_dvcs(app):
            """Doc."""

            for nick in DVC_NICKS:
                dvc = getattr(app.devices, nick)
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

        def lights_out(gui_wdgt):
            """turn OFF all device switch/LED icons"""

            led_list = [self.icon_dict["led_off"]] * 6 + [self.icon_dict["led_green"]] * 3
            wdgt_colls.led_wdgts.write_to_gui(self, led_list)
            wdgt_colls.switch_wdgts.write_to_gui(self, self.icon_dict["switch_off"])
            gui_wdgt.stageButtonsGroup.setEnabled(False)

        if restart:
            # restarting

            if self.gui.camera is not None:
                self.gui.camera.close()

            if self.meas.type is not None:
                await self.gui.main.imp.toggle_meas(
                    self.meas.type, self.meas.laser_mode.capitalize()
                )

            close_all_dvcs(self)

            self.gui.main.deviceErrorDisplay.setText("")

            # finish current timeout loop
            self.timeout_loop.not_finished = False

            lights_out(self.gui.main)
            self.gui.main.depActualCurr.setValue(0)
            self.gui.main.depActualPow.setValue(0)
            self.gui.main.imp.load(DEFAULT_LOADOUT_FILE_PATH)
            #            self.gui.settings.imp.load(DEFAULT_SETTINGS_FILE_PATH)

            self.init_devices()

            # restart timeout loop
            self.timeout_loop = Timeout(self)

            logging.info("Restarting application.")

        else:  # exiting
            self.timeout_loop.not_finished = False

            if self.meas.type is not None:
                await self.gui.main.imp.toggle_meas(
                    self.meas.type, self.meas.laser_mode.capitalize()
                )

            close_all_wins(self)
            close_all_dvcs(self)
            logging.info("Quitting application.")
            print("Application closed.")

    def exit_app(self, event):
        """Doc."""

        try:
            if not self.exiting:
                pressed = Question(
                    txt="Are you sure you want to quit?", title="Quitting Program"
                ).display()
                if pressed == QMessageBox.Yes:
                    self.exiting = True
                    self.loop.create_task(self.clean_up_app())
                else:
                    event.ignore()
        except AttributeError:
            # this is to save defining the 'self.exiting' flag at __init__
            self.exiting = False
            self.exit_app(event)
