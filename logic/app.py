"""Logic Module."""

import logging
import logging.config
import shutil
from collections import deque
from contextlib import contextmanager, suppress
from pathlib import Path
from types import SimpleNamespace

import yaml

import gui.gui
import gui.widgets as wdgts
import utilities.dialog as dialog
from logic.devices import (
    TDC,
    UM232H,
    Camera1,
    Camera2,
    DepletionLaser,
    ExcitationLaser,
    FastGatedSPAD,
    PhotonCounter,
    PicoSecondDelayer,
    PixelClock,
    Scanners,
    Shutter,
    StepperStage,
)
from logic.timeout import Timeout
from utilities.errors import DeviceError


class App:
    """Doc."""

    SETTINGS_DIR_PATH = Path("./settings/")
    PROCESSING_OPTIONS_DIR_PATH = SETTINGS_DIR_PATH / "processing options"
    LOADOUT_DIR_PATH = SETTINGS_DIR_PATH / "loadouts/"
    DEFAULT_LOADOUT_FILE_PATH = LOADOUT_DIR_PATH / "default_loadout"
    DEFAULT_SETTINGS_FILE_PATH = SETTINGS_DIR_PATH / "default_settings"
    DEFAULT_LOG_PATH = Path("./log/")
    device_nick_class_dict = {
        "exc_laser": ExcitationLaser,
        "dep_shutter": Shutter,
        "TDC": TDC,
        "dep_laser": DepletionLaser,
        "stage": StepperStage,
        "UM232H": UM232H,
        "scanners": Scanners,
        "photon_counter": PhotonCounter,
        "delayer": PicoSecondDelayer,
        "spad": FastGatedSPAD,
        "pixel_clock": PixelClock,
        "camera_1": Camera1,
        "camera_2": Camera2,
    }

    def __init__(self, loop):
        """Doc."""

        self.config_logging()

        self.loop = loop
        self.meas = SimpleNamespace(type=None, is_running=False)
        self.analysis = SimpleNamespace(
            loaded_measurements=dict(),
            assigned_to_experiment=dict(),
            loaded_experiments=dict(),
        )

        # get icons
        self.icon_dict = gui.gui.get_icon_paths()

        # init windows
        print("Initializing GUI...", end=" ")
        self.gui = SimpleNamespace()
        # TODO: widget collections from widget module should be part of the gui objects somehow - it's weird that they need to be supplied the gui object to read it - it should be a class method
        self.gui.main = gui.gui.MainWin(self)
        self.gui.settings = gui.gui.SettWin(self)
        self.gui.options = gui.gui.ProcessingOptionsWindow(self)
        self.gui.main.impl.load(self.DEFAULT_LOADOUT_FILE_PATH)
        self.gui.settings.impl.load(self.default_settings_path())
        self.gui.options.impl.load()

        # populate all widget collections in 'gui.widgets' with objects
        [
            val.hold_widgets(self.gui)
            for val in wdgts.__dict__.values()
            if isinstance(val, wdgts.QtWidgetCollection)
        ]

        # create neccessary data folders based on settings paths
        self.create_data_folders()

        # either error or ON
        self.gui.main.ledScn.setIcon(self.icon_dict["led_green"])
        self.gui.main.ledCounter.setIcon(self.icon_dict["led_green"])
        self.gui.main.ledUm232h.setIcon(self.icon_dict["led_green"])

        print("Done.")

        # Initialize all devices
        print("Initializing Devices:")
        self.init_devices()
        print("        Done.")

        # init AO as origin (actual AO is measured in internal AO if last position is needed)
        [
            getattr(self.gui.main, f"{axis}AOV").setValue(org_vltg)
            for axis, org_vltg in zip("xyz", self.devices.scanners.origin)
        ]

        # init scan patterns
        self.gui.main.impl.disp_scn_pttrn("image")
        sol_pattern = wdgts.SOL_MEAS_COLL.gui_to_dict(self.gui)["scan_type"]
        self.gui.main.impl.disp_scn_pttrn(sol_pattern)

        # init last images deque
        self.last_image_scans = deque([], maxlen=10)

        # init existing data folders (solution by default)
        self.gui.main.solDataImport.setChecked(True)
        self.gui.main.impl.switch_data_type()

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

        Path.mkdir(self.DEFAULT_LOG_PATH, parents=True, exist_ok=True)
        init_log_file_list = ["debug.txt", "log.txt"]
        for init_log_file in init_log_file_list:
            file_path = self.DEFAULT_LOG_PATH / init_log_file
            open(file_path, "a").close()

        with open("logging_config.yaml", "r") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)

    def create_data_folders(self):
        """Doc."""

        for gui_object_name in {"dataPath", "camDataPath"}:
            rel_path = Path(getattr(self.gui.settings, gui_object_name).text())
            Path.mkdir(rel_path, parents=True, exist_ok=True)

    def default_settings_path(self) -> Path:
        """Doc."""
        try:
            with open(self.SETTINGS_DIR_PATH / "default_settings_choice", "r") as f:
                return self.SETTINGS_DIR_PATH / f.readline()
        except FileNotFoundError:
            print(
                "Warning - default settings choice file not found! using 'default_settings_lab'.",
                end=" ",
            )
            return self.SETTINGS_DIR_PATH / "default_settings_lab"

    def init_devices(self):
        """
        Goes through a list of device nicknames,
        instantiating a driver object for each device.
        """

        self.devices = SimpleNamespace()
        for nick in self.device_nick_class_dict.keys():
            dvc_class = self.device_nick_class_dict[nick]
            print(f"        Initializing {dvc_class.attrs.log_ref}...", end=" ")
            setattr(self.devices, nick, dvc_class(self))

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
            self.devices.photon_counter.pause_tasks("ci")

        try:
            yield

        finally:
            logging.debug("Resuming 'ai' and 'ci' tasks")
            with suppress(DeviceError):
                # devices not initialized
                self.devices.scanners.init_ai_buffer()
                self.devices.scanners.start_tasks("ai")
                self.devices.photon_counter.init_ci_buffer()
                self.devices.photon_counter.start_tasks("ci")

    async def clean_up_app(self):
        """Doc."""

        def close_all_dvcs(app):
            """Doc."""

            for nick in self.device_nick_class_dict.keys():
                dvc = getattr(app.devices, nick)
                with suppress(DeviceError):
                    dvc.close()

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

            led_list = [self.icon_dict["led_off"]] * 9 + [self.icon_dict["led_green"]] * 2
            wdgts.LED_COLL.obj_to_gui(self, led_list)
            wdgts.SWITCH_COLL.obj_to_gui(self, self.icon_dict["switch_off"])
            gui_wdgt.stageButtonsGroup.setEnabled(False)

        # exiting
        self.timeout_loop.not_finished = False

        if self.meas.type is not None:
            await self.gui.main.impl.toggle_meas(self.meas.type, self.meas.laser_mode.capitalize())

        close_all_wins(self)
        close_all_dvcs(self)

        # clear temp folder
        shutil.rmtree("C:/temp_sfcs_data/", ignore_errors=True)  # TODO: make optional

        logging.info("Quitting application.")
        print("Application closed.")

    def exit_app(self, event):
        """Doc."""

        try:
            if not self.exiting:
                pressed = dialog.QuestionDialog(
                    txt="Are you sure you want to quit?", title="Quitting Program"
                ).display()
                if pressed == dialog.YES:
                    self.exiting = True
                    self.loop.create_task(self.clean_up_app())
                else:
                    event.ignore()
        except AttributeError:
            # this is to save defining the 'self.exiting' flag at __init__
            self.exiting = False
            self.exit_app(event)
