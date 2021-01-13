# -*- coding: utf-8 -*-
"""Measurements Module."""

import datetime
import logging
import time
from typing import NoReturn, Tuple, Union

import numpy as np


class Measurement:
    """Base class for measurements"""

    def __init__(
        self,
        app,
        type,
        duration_gui=None,
        duration_multiplier=1,
    ):
        def get_laser_config(app):
            """Doc."""

            exc_state = app.dvc_dict["EXC_LASER"].state
            dep_state = app.dvc_dict["DEP_LASER"].state

            if exc_state and dep_state:
                laser_config = "sted"
            elif exc_state:
                laser_config = "exc"
            elif dep_state:
                laser_config = "dep"
            else:
                laser_config = "nolaser"

            return laser_config

        self._app = app
        self.type = type
        self.laser_config = get_laser_config(app)
        self.duration_gui = duration_gui
        self.duration_multiplier = duration_multiplier
        self.start_time = None
        self.is_running = False

    async def start(self):
        """Doc."""

        self._app.dvc_dict["UM232H"].purge()  # TODO: is this needed?
        self._app.gui_dict["main"].imp.dvc_toggle("TDC")
        self.is_running = True

        logging.info(f"{self.type} measurement started")

        await self.run()

    def stop(self):
        """Doc."""

        self.is_running = False
        self._app.meas.type = None
        self._app.gui_dict["main"].imp.dvc_toggle("TDC")

        if hasattr(self, "prog_bar"):
            self.prog_bar.setValue(0)

        logging.info(f"{self.type} measurement stopped")


class SFCSSolutionMeasurement(Measurement):
    """Doc."""

    def __init__(self, app, duration_gui, prog_bar, start_time_gui, end_time_gui):
        super().__init__(
            app=app,
            type="SFCSSolution",
            duration_gui=duration_gui,
            duration_multiplier=60,
        )
        self.prog_bar = prog_bar
        self.start_time_gui = start_time_gui
        self.end_time_gui = end_time_gui

        self.file_num_gui = app.gui_dict["main"].solScanFileNo
        self.data_dvc = app.dvc_dict["UM232H"]
        self.save_path = app.gui_dict["settings"].solDataPath.text()
        self.file_template = app.gui_dict["main"].solScanFileTemplate.text()

    async def run(self):
        """Doc."""

        def get_current_and_end_times(
            duration_in_seconds: Union[int, float]
        ) -> Tuple[datetime.time, datetime.time]:
            """
            Given a duration in seconds, returns a tuple (current_time, end_time)
            in datetime.time format, where end_time is current_time + duration_in_seconds.
            """

            duration_in_seconds = int(duration_in_seconds)
            curr_datetime = datetime.datetime.now()
            curr_time = datetime.datetime.now().time()
            end_time = (
                curr_datetime + datetime.timedelta(seconds=duration_in_seconds)
            ).time()
            return (curr_time, end_time)

        def disp_ACF(data_dvc):
            """Doc."""

            print(
                f"Measurement Finished:\n"
                f"Full Data = {data_dvc.data}\n"
                f"Total Bytes = {data_dvc.tot_bytes_read}\n"
            )

        def save_data(
            np_data, dir_path: str, file_template: str, file_no: int, laser_config: str
        ) -> NoReturn:
            """Doc."""

            file_name = f"{file_template}_{laser_config}_{file_no}"
            file_format = ".npy"
            file_path = dir_path + file_name + file_format

            with open(file_path, "wb") as f:
                np.save(f, np_data)

        self.start_time = time.perf_counter()

        # initialize gui start/end times
        start_time, end_time = get_current_and_end_times(
            self.duration_gui.value() * self.duration_multiplier
        )
        self.start_time_gui.setTime(start_time)
        self.end_time_gui.setTime(end_time)

        self.time_passed = 0

        await self._app.loop.run_in_executor(
            None, self._app.dvc_dict["UM232H"].stream_read_TDC, self
        )

        disp_ACF(self.data_dvc)

        save_data(
            self.data_dvc.data,
            self.save_path,
            self.file_template,
            self.file_num_gui.value(),
            self.laser_config,
        )

        self._app.dvc_dict["UM232H"].init_data()

        if self.is_running:  # if not manually stopped
            self._app.gui_dict["main"].imp.toggle_meas(self.type)


class FCSMeasurement(Measurement):
    """Repeated static FCS measurement, intended fo system calibration"""

    def __init__(self, app, duration_gui, prog_bar):
        super().__init__(app=app, type="FCS", duration_gui=duration_gui)
        self.prog_bar = prog_bar

    async def run(self):
        """Doc."""

        def disp_ACF(meas_dvc):
            """Doc."""

            print(
                f"Measurement Finished:\n"
                f"Full Data = {meas_dvc.data}\n"
                f"Total Bytes = {meas_dvc.tot_bytes_read}\n"
            )

        while self.is_running:
            #            await asyncio.to_thread(mock_io, self.duration_gui) # TODO: Try when upgrade to Python 3.9 is feasible

            self.intrvl_done = False
            self.start_time = time.perf_counter()
            self.time_passed = 0

            await self._app.loop.run_in_executor(
                None, self._app.dvc_dict["UM232H"].stream_read_TDC, self
            )

            disp_ACF(meas_dvc=self._app.dvc_dict["UM232H"])
            self._app.dvc_dict["UM232H"].init_data()
