# -*- coding: utf-8 -*-
"""Measurements Module."""

import datetime
import logging
import math
import time
from typing import NoReturn, Tuple, Union

import numpy as np

import utilities.helper as helper


class Measurement:
    """Base class for measurements"""

    def __init__(self, app, type, **kwargs):

        self._app = app
        self.type = type
        [setattr(self, key, val) for key, val in kwargs.items()]
        self.data_dvc = app.devices.UM232H
        self.is_running = False

    async def start(self):
        """Doc."""

        self.data_dvc.purge()
        self._app.gui.main.imp.dvc_toggle("TDC")
        self.is_running = True

        logging.info(f"{self.type} measurement started")

        await self.run()

    def stop(self):
        """Doc."""

        self.is_running = False
        self._app.gui.main.imp.dvc_toggle("TDC")
        self.prog_bar_wdgt.set(0)

        # TODO: need to distinguish stopped from finished - a finished flag?
        logging.info(f"{self.type} measurement stopped")

        self._app.meas.type = None

    def save_data(self, file_name: str) -> NoReturn:
        """Save measurement data as NumPy array (.npy), given filename"""

        file_path = self.save_path + file_name + ".npy"
        np_data = np.frombuffer(self.data_dvc.data, dtype=np.uint8)
        with open(file_path, "wb") as f:
            np.save(f, np_data)

    def get_laser_config(self):
        """Doc."""

        exc_state = self._app.devices.EXC_LASER.state
        dep_state = self._app.devices.DEP_LASER.state

        if exc_state and dep_state:
            laser_config = "sted"
        elif exc_state:
            laser_config = "exc"
        elif dep_state:
            laser_config = "dep"
        else:
            laser_config = "nolaser"

        return laser_config


class SFCSImageMeasurement(Measurement):
    """Doc."""

    def __init__(self, app, **kwargs):
        super().__init__(app=app, type="SFCSImage", **kwargs)
        self.pxl_clk_dvc = app.devices.PixelClock
        self.scanners_dvc = app.devices.Scanners
        self.counter_dvc = app.devices.Scanners

    def build_filename(self, file_no: int) -> str:
        return f"{self.file_template}_{self.laser_config}_{self.scn_type}_{file_no}"

    def setup_scan(self):
        """Doc."""

        def sync_line_freq(line_freq: int, ppl: int, pxl_clk_freq: int) -> (int, int):
            """Doc."""

            point_freq = line_freq * ppl
            clk_div = round(pxl_clk_freq / point_freq)
            syncd_line_freq = pxl_clk_freq / (clk_div * ppl)
            return syncd_line_freq, clk_div

        def fix_ppl(min_freq: int, line_freq: int, ppl: int) -> int:
            """Doc."""

            ratio = round(min_freq / line_freq)
            if ratio > ppl:
                return helper.div_ceil(ratio, 2) * 2
            else:
                return helper.div_ceil(ppl, 2) * 2

        def bld_scn_addrs_str(scn_type: str, scanners_dvc) -> str:
            """Doc."""

            if scn_type == "XY":
                return ", ".join(scanners_dvc.ao_x_addr, scanners_dvc.ao_y_addr)
            elif scn_type == "YZ":
                return ", ".join(scanners_dvc.ao_y_addr, scanners_dvc.ao_z_addr)
            elif scn_type == "ZX":
                return ", ".join(scanners_dvc.ao_z_addr, scanners_dvc.ao_x_addr)

        self.line_freq, clk_div = sync_line_freq(
            self.line_freq_Hz, self.ppl, self.pxl_clk_dvc.freq_MHz * 1e6
        )
        self.pxl_clk_dvc.low_ticks = clk_div - 2
        self.ppl = fix_ppl(
            self.scanners_dvc.MIN_OUTPUT_RATE_Hz, self.line_freq_Hz, self.ppl
        )
        self.scn_addrs = bld_scn_addrs_str(self.scn_type, self.scanners_dvc)
        self.scanners_dvc.init_buffers()
        self.counter_dvc.init_buffer()
        self.curr_line_wdgt.set(0)
        self.curr_plane_wdgt.set(0)

    async def run(self):
        """Doc."""
        # TODO: Add check if file templeate exists in save dir, and if so confirm overwrite or cancel

        pass


class SFCSSolutionMeasurement(Measurement):
    """Doc."""

    def __init__(self, app, **kwargs):
        super().__init__(app=app, type="SFCSSolution", **kwargs)
        self.duration_multiplier = 60
        self.laser_config = self.get_laser_config()

    def build_filename(self, file_no: int) -> str:
        return f"{self.file_template}_{self.laser_config}_{file_no}"

    async def run(self):
        """Doc."""
        # TODO: this needs refactoring - add more functions and clean up code
        # TODO: Add check if file templeate exists in save dir, and if so confirm overwrite or cancel

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

        # initialize gui start/end times
        start_time, end_time = get_current_and_end_times(
            self.total_duration * self.duration_multiplier
        )
        self.start_time_wdgt.set(start_time)
        self.end_time_wdgt.set(end_time)

        # calibrating save-intervals
        saved_dur_mul = self.duration_multiplier
        self.duration = self.cal_duration
        self.duration_multiplier = 1  # in seconds
        self.start_time = time.perf_counter()
        self.time_passed = 0
        self.cal = True
        logging.info(f"Calibrating file intervals for {self.type} measurement")
        await self._app.loop.run_in_executor(None, self.data_dvc.stream_read_TDC, self)
        self.cal = False
        bps = self.data_dvc.tot_bytes_read / self.time_passed
        self.save_intrvl = self.max_file_size * 10 ** 6 / bps / saved_dur_mul

        if self.save_intrvl > self.total_duration:
            self.save_intrvl = self.total_duration

        self.cal_save_intrvl_wdgt.set(self.save_intrvl)
        self.duration = self.save_intrvl
        self.duration_multiplier = saved_dur_mul

        # determining number of files
        num_files = int(math.ceil(self.total_duration / self.save_intrvl))
        self.total_files_wdgt.set(num_files)

        self.total_time_passed = 0
        logging.info(f"Running {self.type} measurement")
        for file_num in range(1, num_files + 1):

            if self.is_running:

                self.file_num_wdgt.set(file_num)

                self.start_time = time.perf_counter()
                self.time_passed = 0
                await self._app.loop.run_in_executor(
                    None, self.data_dvc.stream_read_TDC, self
                )

                self.total_time_passed += self.time_passed

                disp_ACF(self.data_dvc)

                file_name = self.build_filename(file_num)
                self.save_data(file_name)

                self.data_dvc.init_data()

            else:
                break

        if self.is_running:  # if not manually stopped
            self._app.gui.main.imp.toggle_meas(self.type)


class FCSMeasurement(Measurement):
    """Repeated static FCS measurement, intended fo system calibration"""

    def __init__(self, app, **kwargs):
        super().__init__(app=app, type="FCS", **kwargs)
        self.duration_multiplier = 1

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
            #            await asyncio.to_thread(mock_io, self.duration) # TODO: Try when upgrade to Python 3.9 is feasible

            self.start_time = time.perf_counter()
            self.time_passed = 0
            await self._app.loop.run_in_executor(
                None, self.data_dvc.stream_read_TDC, self
            )

            disp_ACF(meas_dvc=self.data_dvc)
            self.data_dvc.init_data()
