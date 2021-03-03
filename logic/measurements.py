# -*- coding: utf-8 -*-
"""Measurements Module."""

import asyncio
import datetime
import logging
import math
import time
from types import SimpleNamespace
from typing import NoReturn, Tuple, Union

import nidaqmx.constants as ni_consts  # TODO: move to consts.py
import numpy as np

import utilities.constants as consts
from logic.scan_patterns import ScanPatternAO
from utilities.dialog import Error
from utilities.helper import div_ceil


class Measurement:
    """Base class for measurements"""

    def __init__(self, app, type: str, scan_params: dict = {}, **kwargs):

        self._app = app
        self.type = type
        self.scan_params = SimpleNamespace()
        [setattr(self.scan_params, key, val) for key, val in scan_params.items()]
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

    def __init__(self, app, scan_params, **kwargs):
        super().__init__(app=app, type="SFCSImage", scan_params=scan_params, **kwargs)
        self.pxl_clk_dvc = app.devices.PXL_CLK
        self.scanners_dvc = app.devices.SCANNERS
        self.counter_dvc = app.devices.COUNTER
        self.laser_dvcs = SimpleNamespace()
        self.laser_dvcs.exc = app.devices.EXC_LASER
        self.laser_dvcs.dep = app.devices.DEP_LASER
        self.laser_dvcs.dep_shutter = app.devices.DEP_SHUTTER
        self.um_V_ratio = tuple(
            getattr(self.scanners_dvc, f"{ax}_um2V_const") for ax in "xyz"
        )

    def build_filename(self, file_no: int) -> str:
        return f"{self.file_template}_{self.laser_config}_{self.scan_params.scn_type}_{file_no}"

    async def toggle_lasers(self, finish=False) -> NoReturn:
        """Doc."""

        if self.scan_params.sted_mode:
            self.scan_params.exc_mode = True
            self.scan_params.dep_mode = True

        if finish:
            if self.scan_params.exc_mode:
                self._app.gui.main.imp.dvc_toggle("EXC_LASER", leave_off=True)
            if self.scan_params.dep_mode:
                self._app.gui.main.imp.dvc_toggle("DEP_SHUTTER", leave_off=True)
        else:
            if self.scan_params.exc_mode:
                self._app.gui.main.imp.dvc_toggle("EXC_LASER", leave_on=True)
            if self.scan_params.dep_mode:
                if self.laser_dvcs.dep.state is False:
                    logging.info(
                        f"{consts.DEP_LASER.log_ref} isn't on. Turnning on and waiting 5 s before measurement."
                    )
                    # TODO: add measurement error decorator to cleanly handle device errors before/during measurements (such as error in dep)
                    self._app.gui.main.imp.dvc_toggle("DEP_LASER")
                    await asyncio.sleep(5)
                self._app.gui.main.imp.dvc_toggle("DEP_SHUTTER", leave_on=True)

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
                return div_ceil(ratio, 2) * 2
            else:
                return div_ceil(ppl, 2) * 2

        def bld_scn_addrs_str(scn_type: str, scanners_dvc) -> str:
            """Doc."""

            if scn_type == "XY":
                return ", ".join([scanners_dvc.ao_x_addr, scanners_dvc.ao_y_addr])
            elif scn_type == "YZ":
                return ", ".join([scanners_dvc.ao_y_addr, scanners_dvc.ao_z_addr])
            elif scn_type == "XZ":
                return ", ".join([scanners_dvc.ao_x_addr, scanners_dvc.ao_z_addr])

        # fix line freq, pxl clk low ticks, and ppl
        self.line_freq, clk_div = sync_line_freq(
            self.scan_params.line_freq_Hz,
            self.scan_params.ppl,
            self.pxl_clk_dvc.freq_MHz * 1e6,
        )
        self.pxl_clk_dvc.low_ticks = clk_div - 2
        self.ppl = fix_ppl(
            self.scanners_dvc.MIN_OUTPUT_RATE_Hz,
            self.scan_params.line_freq_Hz,
            self.scan_params.ppl,
        )
        # unite relevant physical addresses
        self.scn_addrs = bld_scn_addrs_str(self.scan_params.scn_type, self.scanners_dvc)
        # init device buffers
        self.scanners_dvc.init_ai_buffer()
        self.counter_dvc.init_ci_buffer()
        # create ao_buffer
        (
            self.ao_buffer,
            self.dt,
            self.set_pnts_lines_odd,
            self.set_pnts_lines_even,
            self.set_pnts_planes,
            self.total_pnts,
        ) = ScanPatternAO("image", self.scan_params, self.um_V_ratio).calculate_ao()
        # TODO: why is the next line correct? explain and use a constant for 1.5E-7
        # TODO: create a scan arguments/parameters object to send to scanners_dvc (or simply for more clarity)
        self.n_ao_samps = len(self.ao_buffer[0])
        self.ai_conv_rate = 1 / (
            (self.dt - 1.5e-7) / self.scanners_dvc.ai_buffer.shape[0]
        )

    def init_scan_tasks(self):
        """Doc."""

        self.scanners_dvc.start_write_task(
            ao_data=self.ao_buffer,
            type=self.scan_params.scn_type,
            samp_clk_cnfg_xy={
                "source": self.pxl_clk_dvc.out_term,
                "sample_mode": ni_consts.AcquisitionType.FINITE,
                "samps_per_chan": self.n_ao_samps,
                "rate": 100000,
            },
            samp_clk_cnfg_z={
                "source": self.pxl_clk_dvc.out_ext_term,
                "sample_mode": ni_consts.AcquisitionType.FINITE,
                "samps_per_chan": self.n_ao_samps,
                "rate": 100000,
            },
        )

        self.scanners_dvc.start_scan_read_task(
            samp_clk_cnfg={},
            timing_params={
                "samp_quant_samp_mode": ni_consts.AcquisitionType.CONTINUOUS,
                "samp_quant_samp_per_chan": int(self.total_pnts * 1.2),
                "samp_timing_type": ni_consts.SampleTimingType.SAMPLE_CLOCK,
                "samp_clk_src": self.scanners_dvc.tasks.ao[
                    "AO XY"
                ].timing.samp_clk_term,
                "ai_conv_rate": self.ai_conv_rate,
            },
        )

        self.counter_dvc.start_scan_read_task(
            samp_clk_cnfg={},
            timing_params={
                "samp_quant_samp_mode": ni_consts.AcquisitionType.CONTINUOUS,
                "samp_quant_samp_per_chan": int(self.total_pnts * 1.2),
                "samp_timing_type": ni_consts.SampleTimingType.SAMPLE_CLOCK,
                "samp_clk_src": self.scanners_dvc.tasks.ao[
                    "AO XY"
                ].timing.samp_clk_term,
            },
        )

    def change_plane(self, plane_idx):
        """Doc."""

        for axis in "XYZ":
            if axis not in self.scan_params.scn_type:
                plane_axis = axis
                break

        self.scanners_dvc.start_write_task(
            ao_data=[[self.set_pnts_planes[plane_idx]]],
            type=plane_axis,
        )

    def move_to_start(self):
        """Doc."""
        # TODO: perhaps move back to current AO when done?

        self.scanners_dvc.start_write_task(
            # TODO: clearer way do get the following line?
            ao_data=[[self.ao_buffer[0][0]], [self.ao_buffer[1][0]]],
            type=self.scan_params.scn_type,
        )

    async def run(self):
        """Doc."""
        # TODO: Add check if file templeate exists in save dir, and if so confirm overwrite or cancel

        await self.toggle_lasers()
        self.setup_scan()
        self._app.gui.main.imp.dvc_toggle("PXL_CLK")
        self.move_to_start()

        plane_data = []
        logging.info(f"Running {self.type} measurement")
        for plane_idx in range(len(self.set_pnts_planes)):

            if self.is_running:

                self.change_plane(plane_idx)

                self.curr_line_wdgt.set(plane_idx)
                self.curr_plane_wdgt.set(plane_idx)

                self.data_dvc.purge()

                self.init_scan_tasks()

                await asyncio.to_thread(self.data_dvc.ao_task_sync_read_TDC, self)

                # TODO: add progbar support (calculate time for each plane, and mulyiply by number of planes)

                print(
                    f"Plane {plane_idx} scanned. First 10 bytes of data: {self.data_dvc.data[:10]}"
                )
                plane_data.append(self.data_dvc.data)

                self.data_dvc.init_data()

            else:
                break

            # TODO: prepare images, save, and present the middle plane
        #        plane_images = [self.build_image(plane) for plane in plane_data]
        #        self.data = np.dstack(plane_images)
        #        filename = self.build_filename()
        #        self.save_data(filename)

        self._app.gui.main.numPlaneShown.setValue(
            1
        )  # TODO: show middle plane? or lowest?

        self._app.gui.main.imp.dvc_toggle("PXL_CLK")
        await self.toggle_lasers(finish=True)
        self.scanners_dvc.start_continuous_read_task()
        self.counter_dvc.start_continuous_read_task()

        if self.is_running:  # if not manually stopped
            self._app.gui.main.imp.toggle_meas(self.type)


class SFCSSolutionMeasurement(Measurement):
    """Doc."""

    def __init__(self, app, scan_params, **kwargs):
        super().__init__(
            app=app, type="SFCSSolution", scan_params=scan_params, **kwargs
        )
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

        await asyncio.to_thread(self.data_dvc.stream_read_TDC, self)

        self.cal = False
        bps = self.data_dvc.tot_bytes_read / self.time_passed
        try:
            self.save_intrvl = self.max_file_size * 10 ** 6 / bps / saved_dur_mul
        except ZeroDivisionError:
            # TODO: create a decorator for measurement errors - better yet, add a counter error check for when counts are zero/unchanging
            Error(custom_txt="Counter is probably unplugged.").display()
            return

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

                await asyncio.to_thread(self.data_dvc.stream_read_TDC, self)

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

            self.start_time = time.perf_counter()
            self.time_passed = 0

            await asyncio.to_thread(self.data_dvc.stream_read_TDC, self)

            disp_ACF(meas_dvc=self.data_dvc)
            self.data_dvc.init_data()
