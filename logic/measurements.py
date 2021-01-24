# -*- coding: utf-8 -*-
"""Measurements Module."""

import datetime
import logging
import math
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

            exc_state = app.devices.EXC_LASER.state
            dep_state = app.devices.DEP_LASER.state

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
        self.data_dvc = app.devices.UM232H
        self.laser_config = get_laser_config(app)
        self.duration_gui = duration_gui
        self.duration_multiplier = duration_multiplier
        self.start_time = None
        self.is_running = False

    async def start(self):
        """Doc."""

        self._app.devices.UM232H.purge()
        self._app.gui_dict["main"].imp.dvc_toggle("TDC")
        self.is_running = True

        logging.info(f"{self.type} measurement started")

        await self.run()

    def stop(self):
        """Doc."""

        self.is_running = False
        self._app.gui_dict["main"].imp.dvc_toggle("TDC")

        if hasattr(self, "prog_bar"):
            self.prog_bar.setValue(0)

        logging.info(f"{self.type} measurement stopped")

        self._app.meas.type = None


class SFCSSolutionMeasurement(Measurement):
    """Doc."""

    def __init__(self, app, duration_gui, prog_bar):
        super().__init__(
            app=app,
            type="SFCSSolution",
            duration_gui=duration_gui,
            duration_multiplier=60,
        )
        self.prog_bar = prog_bar
        self.start_time_gui = app.gui_dict["main"].solScanStartTime
        self.end_time_gui = app.gui_dict["main"].solScanEndTime
        self.total_duration_gui = app.gui_dict["main"].solScanDuration

        self.max_file_size = app.gui_dict["main"].solScanMaxFileSize.value()
        self.cal_time = app.gui_dict["main"].solScanCalTime.value()

        self.total_files_gui = app.gui_dict["main"].solScanTotalFiles
        self.file_num_gui = app.gui_dict["main"].solScanFileNo
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
            """Placeholder for calculating and presenting the ACF"""

            print(
                f"Measurement Finished:\n"
                f"Full Data[:100] = {data_dvc.data[:100]}\n"
                f"Total Bytes = {data_dvc.tot_bytes_read}\n"
            )

        def save_data(
            array_data,
            dir_path: str,
            file_template: str,
            file_no: int,
            laser_config: str,
        ) -> NoReturn:
            """Doc."""

            file_name = f"{file_template}_{laser_config}_{file_no}"
            file_format = ".npy"
            file_path = dir_path + file_name + file_format

            np_data = np.frombuffer(array_data, dtype=np.uint8)

            with open(file_path, "wb") as f:
                np.save(f, np_data)

        # initialize gui start/end times
        start_time, end_time = get_current_and_end_times(
            self.total_duration_gui.value() * self.duration_multiplier
        )
        self.start_time_gui.setTime(start_time)
        self.end_time_gui.setTime(end_time)

        # calibrating save-intervals
        saved_dur_mul = self.duration_multiplier
        self.duration_gui = self._app.gui_dict["main"].solScanCalTime
        self.duration_multiplier = 1  # in seconds
        self.start_time = time.perf_counter()
        self.time_passed = 0
        self.cal = True
        logging.info(f"Calibrating file intervals for {self.type} measurement")
        await self._app.loop.run_in_executor(None, self.data_dvc.stream_read_TDC, self)
        self.cal = False
        bps = self.data_dvc.tot_bytes_read / self.time_passed
        self.save_intrvl = self.max_file_size * 10 ** 6 / bps / saved_dur_mul

        if self.save_intrvl > self.total_duration_gui.value():
            self.save_intrvl = self.total_duration_gui.value()

        self.duration_gui = self._app.gui_dict["main"].solScanCalIntrvl
        self.duration_gui.setValue(self.save_intrvl)
        self.duration_multiplier = saved_dur_mul

        # determining number of files
        num_files = int(math.ceil(self.total_duration_gui.value() / self.save_intrvl))
        self.total_files_gui.setValue(num_files)

        self.total_time_passed = 0
        logging.info(f"Running {self.type} measurement")
        for file_num in range(1, num_files + 1):

            if self.is_running:

                self.file_num_gui.setValue(file_num)

                self.start_time = time.perf_counter()
                self.time_passed = 0
                await self._app.loop.run_in_executor(
                    None, self.data_dvc.stream_read_TDC, self
                )

                self.total_time_passed += self.time_passed

                disp_ACF(self.data_dvc)

                save_data(
                    self.data_dvc.data,
                    self.save_path,
                    self.file_template,
                    file_num,
                    self.laser_config,
                )

                self.data_dvc.init_data()

            else:
                break

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
                f"Full Data[:100] = {meas_dvc.data[:100]}\n"
                f"Total Bytes = {meas_dvc.tot_bytes_read}\n"
            )

        while self.is_running:
            #            await asyncio.to_thread(mock_io, self.duration_gui) # TODO: Try when upgrade to Python 3.9 is feasible

            self.start_time = time.perf_counter()
            self.time_passed = 0
            await self._app.loop.run_in_executor(
                None, self.data_dvc.stream_read_TDC, self
            )

            disp_ACF(meas_dvc=self.data_dvc)
            self.data_dvc.init_data()
