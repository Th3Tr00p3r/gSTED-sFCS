"""Logic Module."""

import logging
import logging.config
import os
from contextlib import contextmanager, suppress
from types import SimpleNamespace

import yaml
from PyQt5.QtWidgets import QMessageBox

import gui.gui
import logic.devices as dvcs
import utilities.helper as helper
import utilities.widget_collections as wdgt_colls
from logic.timeout import Timeout
from utilities.dialog import Question
from utilities.errors import DeviceError


class App:
    """Doc."""

    SETTINGS_DIR_PATH = "./settings/"
    LOADOUT_DIR_PATH = os.path.join(SETTINGS_DIR_PATH, "loadouts/")
    DEFAULT_LOADOUT_FILE_PATH = os.path.join(LOADOUT_DIR_PATH, "default_loadout")
    DEFAULT_SETTINGS_FILE_PATH = os.path.join(SETTINGS_DIR_PATH, "default_settings")
    DEFAULT_LOG_PATH = "./log/"
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
        self.gui.main.imp.load(self.DEFAULT_LOADOUT_FILE_PATH)
        self.gui.settings = gui.gui.SettWin(self)
        self.gui.settings.imp.load(self.default_settings_path())
        self.gui.camera = gui.gui.CamWin(self)  # instantiated on pressing camera button

        # populate all widget collections in 'utilities.widget_collections' with objects
        [
            val.hold_objects(app=self)
            for val in wdgt_colls.__dict__.values()
            if isinstance(val, wdgt_colls.QtWidgetCollection)
        ]

        # create neccessary data folders based on settings paths
        self.create_data_folders()

        # either error or ON
        self.gui.main.ledScn.setIcon(self.icon_dict["led_green"])
        self.gui.main.ledCounter.setIcon(self.icon_dict["led_green"])
        self.gui.main.ledUm232h.setIcon(self.icon_dict["led_green"])

        print("Done.")

        print("Initializing Devices:")
        self.init_devices()
        print("        Done.")

        # init AO as origin (actual AO is measured in internal AO if last position is needed)
        [
            getattr(self.gui.main, f"{axis}AOV").setValue(org_vltg)
            for axis, org_vltg in zip("xyz", self.devices.scanners.ORIGIN)
        ]

        # init scan patterns
        self.gui.main.imp.disp_scn_pttrn("image")
        sol_pattern = wdgt_colls.sol_meas_wdgts.read_gui(self).scan_type
        self.gui.main.imp.disp_scn_pttrn(sol_pattern)

        # init existing data folders (solution by default)
        self.gui.main.solDataImport.setChecked(True)
        self.gui.main.imp.populate_all_data_dates()

        # show the GUI
        self.gui.main.show()

        # set up main timeout event
        print("Initializing timeout loop...", end=" ")
        self.timeout_loop = Timeout(self)
        print("Done.")

        print("Application initialized.")
        logging.info("Application started")

    def config_logging(self):
        """
        Configure the logging package for the whole application,
        and ensure folder and initial files exist.
        """

        os.makedirs(self.DEFAULT_LOG_PATH, exist_ok=True)
        init_log_file_list = ["debug", "log"]
        for init_log_file in init_log_file_list:
            file_path = os.path.join(self.DEFAULT_LOG_PATH, init_log_file)
            open(file_path, "a").close()

        with open("logging_config.yaml", "r") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)

    def create_data_folders(self):
        """Doc."""

        for gui_object_name in {"dataPath", "camDataPath"}:
            rel_path = getattr(self.gui.settings, gui_object_name).text()
            os.makedirs(rel_path, exist_ok=True)

    def default_settings_path(self) -> str:
        """Doc."""
        try:
            with open(os.path.join(self.SETTINGS_DIR_PATH, "default_settings_choice"), "r") as f:
                return os.path.join(self.SETTINGS_DIR_PATH, f.readline())
        except FileNotFoundError:
            print(
                "Warning - default settings choice file not found! using 'default_settings_lab'.",
                end=" ",
            )
            return os.path.join(self.SETTINGS_DIR_PATH, "default_settings_lab")

    def init_devices(self):
        """
        Goes through a list of device nicknames,
        instantiating a driver object for each device.
        """

        self.devices = SimpleNamespace()
        for nick in self.DVC_NICKS:
            dvc_attrs = dvcs.DEVICE_ATTR_DICT[nick]
            print(f"        Initializing {dvc_attrs.log_ref}...")
            dvc_class = getattr(dvcs, dvc_attrs.class_name)
            param_dict = dvc_attrs.param_widgets.hold_objects(app=self).read_gui(self, "dict")
            param_dict["nick"] = nick
            param_dict["log_ref"] = dvc_attrs.log_ref
            param_dict["led_icon"] = self.icon_dict[f"led_{dvc_attrs.led_color}"]
            param_dict["error_display"] = wdgt_colls.QtWidgetAccess(
                "deviceErrorDisplay", "QLineEdit", "main", True
            ).hold_obj(self.gui.main)

            if dvc_attrs.cls_xtra_args:
                x_args = [
                    helper.deep_getattr(self, deep_attr) for deep_attr in dvc_attrs.cls_xtra_args
                ]
                setattr(self.devices, nick, dvc_class(param_dict, *x_args))
            else:
                setattr(self.devices, nick, dvc_class(param_dict))

    @contextmanager
    def pause_ai_ci(self):
        """
        Context manager for elegantly pausing the countinuous ai/ci reading
        while executing long, blocking functions (such as during data processing
        for on-board analysis)
        """

        logging.debug("Pausing 'ai' and 'ci' tasks")
        with suppress(DeviceError):
            self.devices.scanners.pause_tasks("ai")
            self.devices.photon_detector.pause_tasks("ci")

        try:
            yield
        finally:
            logging.debug("Resuming 'ai' and 'ci' tasks")
            with suppress(DeviceError):
                # devices not initialized
                self.devices.scanners.init_ai_buffer()
                self.devices.scanners.start_tasks("ai")
                self.devices.photon_detector.init_ci_buffer()
                self.devices.photon_detector.start_tasks("ci")

    async def clean_up_app(self, restart=False):
        """Doc."""

        def close_all_dvcs(app):
            """Doc."""

            for nick in self.DVC_NICKS:
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
            logging.info("Restarting application.")

            if self.gui.camera is not None:
                self.gui.camera.close()

            if self.meas.type is not None:
                await self.gui.main.imp.toggle_meas(
                    self.meas.type, self.meas.laser_mode.capitalize()
                )

            close_all_dvcs(self)

            self.gui.main.deviceErrorDisplay.setText("")

            self.timeout_loop.not_finished = False  # finish current timeout loop

            lights_out(self.gui.main)
            self.gui.main.depActualCurr.setValue(0)
            self.gui.main.depActualPow.setValue(0)
            self.gui.main.imp.load(self.DEFAULT_LOADOUT_FILE_PATH)

            print("Initializing Devices:")
            self.init_devices()
            print("        Done.")

            # restart timeout loop
            self.timeout_loop = Timeout(self)

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
