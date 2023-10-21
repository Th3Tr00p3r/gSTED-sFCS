"""Logic Module."""

import asyncio
import logging
import logging.config
import shutil
from collections import deque
from contextlib import contextmanager, suppress
from pathlib import Path
from types import SimpleNamespace

import yaml  # type: ignore # mypy issue with 'stub'?

import gui.gui
import gui.widgets as wdgts
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
from utilities.dialog import QuestionDialog
from utilities.errors import DeviceError
from utilities.file_utilities import DUMP_PATH


class App:
    """Doc."""

    SETTINGS_DIR_PATH = Path("./settings/")
    DEFAULT_LOADOUT_FILE_PATH = SETTINGS_DIR_PATH / "default_loadout.txt"
    DEFAULT_SETTINGS_FILE_PATH = SETTINGS_DIR_PATH / "default_settings.txt"
    DEFAULT_PROCESSING_OPTIONS_FILE_PATH = SETTINGS_DIR_PATH / "default_processing_options.txt"
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
        self.meas_queue = []
        self.analysis = SimpleNamespace(
            loaded_measurements=dict(),
            assigned_to_experiment=dict(),
            loaded_experiments=dict(),
        )
        self.are_ai_ci_paused = False

        # get icons
        self.icon_dict = gui.gui.get_icon_paths()

        # init last images deque
        self.last_image_scans = deque([], maxlen=10)

        # init windows
        print("    Initializing GUI...", end=" ")
        self.gui = SimpleNamespace()
        # TODO: widget collections from widget module should be part of the gui objects somehow - it's weird that they need to be supplied the gui object to read it - it should be a class method
        self.gui.main = gui.gui.MainWin(self)
        self.gui.settings = gui.gui.SettWin(self)
        self.gui.options = gui.gui.ProcessingOptionsWindow(self)
        self.gui.main.impl.load()
        self.gui.settings.impl.load(self.DEFAULT_SETTINGS_FILE_PATH)
        self.gui.options.impl.load()

        # populate all widget collections in 'gui.widgets' with objects
        [
            val.hold_widgets(self.gui)
            for val in wdgts.__dict__.values()
            if isinstance(val, wdgts.QtWidgetCollection)
        ]

        # clear measurement queue
        self.gui.main.measQueue.clear()

        # create neccessary data folders based on settings paths
        self.create_data_folders()

        # either error or ON
        self.gui.main.ledScn.setIcon(self.icon_dict["led_green"])
        self.gui.main.ledCounter.setIcon(self.icon_dict["led_green"])
        self.gui.main.ledUm232h.setIcon(self.icon_dict["led_green"])

        print("Done.")

        # Initialize all devices
        print("    Initializing Devices:")
        self.init_devices()
        print("        Done.")

        # init AO as origin (actual AO is measured in internal AO if last position is needed)
        [
            getattr(self.gui.main, f"{axis}AOV").setValue(org_vltg)
            for axis, org_vltg in zip("xyz", self.devices.scanners.origin)
        ]

        # init scan patterns
        print("    Calculating initial scan patterns...", end=" ")
        self.gui.main.impl.disp_scn_pttrn("image")
        sol_pattern = wdgts.SOL_MEAS_COLL.gui_to_dict(self.gui)["scan_type"]
        self.gui.main.impl.disp_scn_pttrn(sol_pattern)
        print("Done.")

        # init existing data folders (solution scan data by default)
        print("    Fetching lastest measurement data...", end=" ")
        self.gui.main.solDataImport.setChecked(True)
        self.gui.main.impl.switch_data_type()
        print("Done.")

        # set up main timeout event
        print("    Initializing timeout loop...", end=" ")
        self.timeout_loop = Timeout(self)
        print("Done.")

        # show the GUI
        print("    Displaying GUI...", end=" ")
        self.gui.main.show()
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
            self.log_file_path = self.DEFAULT_LOG_PATH / init_log_file
            open(self.log_file_path, "a").close()

        with open("logging_config.yaml", "r") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)

    def create_data_folders(self):
        """Doc."""

        for gui_object_name in {"dataPath", "camDataPath"}:
            rel_path = Path(getattr(self.gui.settings, gui_object_name).text())
            Path.mkdir(rel_path, parents=True, exist_ok=True)

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
            self.devices.scanners.stop_tasks("ai")
            self.devices.photon_counter.stop_tasks("ci")
            self.are_ai_ci_paused = True

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
                self.are_ai_ci_paused = False

    async def clean_up_app(self, should_clear_dump_path: bool):
        """Doc."""

        def close_all_dvcs(app):
            """Doc."""

            for nick in self.device_nick_class_dict.keys():
                dvc = getattr(app.devices, nick)
                with suppress(DeviceError):
                    if asyncio.iscoroutinefunction(dvc.close):
                        self.loop.create_task(dvc.close())
                    else:
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

        if self.meas.is_running is not None:
            await self.gui.main.impl.cancel_queue()

        close_all_wins(self)
        close_all_dvcs(self)

        # stopping the event loop
        self.loop.stop()

        # clear temp folder
        if should_clear_dump_path:
            shutil.rmtree(DUMP_PATH, ignore_errors=True)
            logging.info("Dump path cleared.")

        logging.info("Quitting application.")
        print("Application closed.")

    def exit_app(self, event):
        """Doc."""

        try:
            if not self.exiting:
                pressed = QuestionDialog(
                    txt=f"Would you like to clear the temprary dump path?\n({DUMP_PATH})",
                    title="Quitting Program",
                    should_include_cancel=True,
                ).display()
                if pressed is not None:
                    self.exiting = True
                    self.loop.create_task(self.clean_up_app(should_clear_dump_path=pressed is True))
                else:
                    event.ignore()
        except AttributeError:
            # this is to save defining the 'self.exiting' flag at __init__ ('exiting' attribute doesn't exist)
            self.exiting = False
            self.exit_app(event)
