"""Measurements."""

import asyncio
import datetime
import logging
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime as dt
from types import SimpleNamespace

import nidaqmx.constants as ni_consts
import numba as nb
import numpy as np
import scipy.io as sio

import gui.gui
import logic.devices as dvcs
from data_analysis import fit_tools
from data_analysis.correlation_function import CorrFuncTDC
from data_analysis.photon_data import PhotonData
from logic.scan_patterns import ScanPatternAO
from utilities.errors import DeviceError, err_hndlr
from utilities.helper import div_ceil, paths_to_icons


class Measurement:
    """Base class for measurements"""

    def __init__(self, app, type: str, scan_params, **kwargs):

        self._app = app
        self.type = type
        self.tdc_dvc = app.devices.TDC
        self.data_dvc = app.devices.UM232H
        self.icon_dict = paths_to_icons(gui.icons.icon_paths_dict)  # get icons
        [setattr(self, key, val) for key, val in kwargs.items()]
        self.counter_dvc = app.devices.photon_detector

        self.pxl_clk_dvc = app.devices.pixel_clock
        self.scanners_dvc = app.devices.scanners
        self.scan_params = scan_params
        self.um_v_ratio = tuple(getattr(self.scanners_dvc, f"{ax}_um2V_const") for ax in "xyz")
        self.sys_info = {
            "setup": "STED with galvos",
            "after_pulse_param": (
                "multi_exponent_fit",
                1e5
                * np.array(
                    [
                        0.183161051158731,
                        0.021980256326163,
                        6.882763042785681,
                        0.154790280034295,
                        0.026417532300439,
                        0.004282749744374,
                        0.001418363840077,
                        0.000221275818533,
                    ]
                ),
            ),
            "ai_scaling_xyz": [1.243, 1.239, 1],  # TODO: check if matches today's ratio
            "xyz_um_to_v": self.um_v_ratio,
        }

        self.laser_dvcs = SimpleNamespace(
            exc=app.devices.exc_laser,
            dep=app.devices.dep_laser,
            dep_shutter=app.devices.dep_shutter,
        )
        self.is_running = False

    async def stop(self):
        """Doc."""

        try:
            await self.toggle_lasers(finish=True)
            self._app.gui.main.imp.dvc_toggle("TDC", leave_off=True)

            if self.scanning:
                self.return_to_regular_tasks()
                self._app.gui.main.imp.dvc_toggle("pixel_clock", leave_off=True)

                if self.type == "SFCSSolution":
                    self._app.gui.main.imp.go_to_origin("XY")
                elif self.type == "SFCSImage":
                    self._app.gui.main.imp.move_scanners(destination=self.initial_pos)

        except DeviceError:
            pass

        self.is_running = False
        await self._app.gui.main.imp.toggle_meas(self.type, self.laser_mode.capitalize())
        self.prog_bar_wdgt.set(0)
        self._app.gui.main.imp.populate_all_data_dates()  # refresh saved measurements
        logging.info(f"{self.type} measurement stopped")
        self._app.meas.type = None

    async def record_data(self, timed: bool = False, size_limited: bool = False) -> float:
        """
        Turn ON the TDC (FPGA), read while conditions are met,
        turn OFF TDC and read leftover data,
        return the time it took (seconds).
        """

        self._app.gui.main.imp.dvc_toggle("TDC", leave_on=True)

        if timed:
            # Solution
            while self.is_running:
                await self.data_dvc.read_TDC()
                self.time_passed_s = time.perf_counter() - self.start_time
                if self.time_passed_s >= self.duration_s:
                    break
                if size_limited and ((self.data_dvc.tot_bytes_read / 1e6) >= self.max_file_size_mb):
                    break

        else:
            # Image
            while (
                self.is_running
                and not self.scanners_dvc.are_tasks_done("ao")
                and not (overdue := self.time_passed_s > (self.est_plane_duration * 1.1))
            ):
                await self.data_dvc.read_TDC()
                self.time_passed_s = time.perf_counter() - self.start_time
            if overdue:
                raise MeasurementError(
                    "Tasks are overdue. Check that all relevant devices are turned ON"
                )

        self._app.gui.main.imp.dvc_toggle("TDC", leave_off=True)
        await self.data_dvc.read_TDC()  # read leftovers

    def save_data(self, data_dict: dict, file_name: str) -> None:
        """
        Create a directory of today's date, and there
        save measurement data as a .mat file or a
        .pkl file. .mat files can be analyzed in MATLAB using our
        current MATLAB-based analysis (or later in Python using sio.loadmat()).
        Note: saving as .mat takes longer
        """
        # NOTE: does not handle overnight measurements during which the date changes (who cares)

        today_dir = os.path.join(self.save_path, dt.now().strftime("%d_%m_%Y"))

        if self.type == "SFCSSolution":
            save_path = os.path.join(today_dir, "solution")
        elif self.type == "SFCSImage":
            save_path = os.path.join(today_dir, "image")

        os.makedirs(save_path, exist_ok=True)

        file_path = os.path.join(save_path, file_name)

        if self.save_frmt == "MATLAB":
            # .mat
            sio.savemat(file_path + ".mat", data_dict)

        else:
            # .pkl
            with open(file_path + ".pkl", "wb") as f:
                pickle.dump(data_dict, f)

    async def toggle_lasers(self, finish=False) -> None:
        """Doc."""

        def current_emission_state() -> str:
            """Doc."""

            exc_state = self._app.devices.exc_laser.state
            try:
                dep_state = (
                    self._app.devices.dep_laser.emission_state
                    and self._app.devices.dep_shutter.state
                )
            except AttributeError:
                # dep error on startup, emission should be off
                dep_state = False

            if exc_state and dep_state:
                return "sted"
            elif exc_state:
                return "exc"
            elif dep_state:
                return "dep"
            else:
                return "nolaser"

        if finish:
            # measurement finishing
            if self.laser_mode == "exc":
                self._app.gui.main.imp.dvc_toggle("exc_laser", leave_off=True)
            elif self.laser_mode == "dep":
                self._app.gui.main.imp.dvc_toggle("dep_shutter", leave_off=True)
            elif self.laser_mode == "sted":
                self._app.gui.main.imp.dvc_toggle("exc_laser", leave_off=True)
                self._app.gui.main.imp.dvc_toggle("dep_shutter", leave_off=True)
        else:
            # measurement begins
            if self.laser_mode == "exc":
                # turn excitation ON and depletion shutter OFF
                self._app.gui.main.imp.dvc_toggle("exc_laser", leave_on=True)
                self._app.gui.main.imp.dvc_toggle("dep_shutter", leave_off=True)
            elif self.laser_mode == "dep":
                await self.prep_dep() if not self.laser_dvcs.dep.emission_state else None
                # turn depletion shutter ON and excitation OFF
                self._app.gui.main.imp.dvc_toggle("dep_shutter", leave_on=True)
                self._app.gui.main.imp.dvc_toggle("exc_laser", leave_off=True)
            elif self.laser_mode == "sted":
                await self.prep_dep() if not self.laser_dvcs.dep.emission_state else None
                # turn both depletion shutter and excitation ON
                self._app.gui.main.imp.dvc_toggle("exc_laser", leave_on=True)
                self._app.gui.main.imp.dvc_toggle("dep_shutter", leave_on=True)

            if current_emission_state() != self.laser_mode:
                # cancel measurement if relevant lasers are not ON,
                raise MeasurementError(
                    f"Requested laser mode ({self.laser_mode}) was not attained."
                )

    async def prep_dep(self):
        """Doc."""

        toggle_succeeded = self._app.gui.main.imp.dvc_toggle(
            "dep_laser", toggle_mthd="laser_toggle", state_attr="emission_state"
        )
        if toggle_succeeded:
            logging.info(
                f"{dvcs.DEVICE_ATTR_DICT['dep_laser'].log_ref} isn't on. Turning on and waiting 5 s before measurement."
            )
            if self.type == "SFCSImage" and self.laser_mode == "dep":
                button = self._app.gui.main.startImgScanDep
            elif self.type == "SFCSImage" and self.laser_mode == "sted":
                button = self._app.gui.main.startImgScanSted
            elif self.type == "SFCSSolution" and self.laser_mode == "dep":
                button = self._app.gui.main.startSolScanDep
            elif self.type == "SFCSSolution" and self.laser_mode == "sted":
                button = self._app.gui.main.startSolScanSted
            button.setEnabled(False)
            await asyncio.sleep(5)
            button.setEnabled(True)

    def init_scan_tasks(self, ao_sample_mode: str) -> None:
        """Doc."""

        ao_sample_mode = getattr(ni_consts.AcquisitionType, ao_sample_mode)

        self.scanners_dvc.start_write_task(
            ao_data=self.ao_buffer,
            type=self.scan_params.scan_plane,
            samp_clk_cnfg_xy={
                "source": self.pxl_clk_dvc.out_term,
                "sample_mode": ao_sample_mode,
                "samps_per_chan": self.n_ao_samps,
                "rate": 100000,
            },
            samp_clk_cnfg_z={
                "source": self.pxl_clk_dvc.out_ext_term,
                "sample_mode": ao_sample_mode,
                "samps_per_chan": self.n_ao_samps,
                "rate": 100000,
            },
            start=False,
        )

        xy_ao_task = [task for task in self.scanners_dvc.tasks.ao if (task.name == "AO XY")][0]
        ao_clk_src = xy_ao_task.timing.samp_clk_term

        self.scanners_dvc.start_scan_read_task(
            samp_clk_cnfg={},
            timing_params={
                "samp_quant_samp_mode": ni_consts.AcquisitionType.CONTINUOUS,
                "samp_quant_samp_per_chan": int(self.n_ao_samps * 1.2),
                "samp_timing_type": ni_consts.SampleTimingType.SAMPLE_CLOCK,
                "samp_clk_src": ao_clk_src,
                "ai_conv_rate": self.ai_conv_rate,
            },
        )

        self.counter_dvc.start_scan_read_task(
            samp_clk_cnfg={},
            timing_params={
                "samp_quant_samp_mode": ni_consts.AcquisitionType.CONTINUOUS,
                "samp_quant_samp_per_chan": int(self.n_ao_samps * 1.2),
                "samp_timing_type": ni_consts.SampleTimingType.SAMPLE_CLOCK,
                "samp_clk_src": ao_clk_src,
            },
        )

    def return_to_regular_tasks(self):
        """Close ao tasks and resume continuous ai/CI"""

        self.scanners_dvc.start_continuous_read_task()
        self.counter_dvc.start_continuous_read_task()
        self.scanners_dvc.close_tasks("ao")


class SFCSImageMeasurement(Measurement):
    """Doc."""

    def __init__(self, app, scan_params, **kwargs):
        super().__init__(app=app, type="SFCSImage", scan_params=scan_params, **kwargs)
        self.scanning = True

    def build_filename(self) -> str:
        return f"{self.file_template}_{self.laser_mode}_{self.scan_params.scan_plane}_{dt.now().strftime('%H%M%S')}"

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

        # fix line freq, pxl clk low ticks, and ppl
        self.scan_params.line_freq_Hz, clk_div = sync_line_freq(
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
        # create ao_buffer
        (
            self.ao_buffer,
            self.scan_params.set_pnts_lines_odd,
            self.scan_params.set_pnts_lines_even,
            self.scan_params.set_pnts_planes,
        ) = ScanPatternAO("image", self.scan_params, self.um_v_ratio).calculate_pattern()
        self.n_ao_samps = self.ao_buffer.shape[1]
        # TODO: why is the next line correct? explain and use a constant for 1.5E-7. ask Oleg
        self.ai_conv_rate = 6 * 2 * (1 / (self.scan_params.dt - 1.5e-7))
        self.est_plane_duration = self.n_ao_samps * self.scan_params.dt
        self.est_total_duration_s = self.est_plane_duration * len(self.scan_params.set_pnts_planes)
        self.plane_choice.obj.setMaximum(len(self.scan_params.set_pnts_planes) - 1)

    def change_plane(self, plane_idx):
        """Doc."""

        for axis in "XYZ":
            if axis not in self.scan_params.scan_plane:
                plane_axis = axis
                break

        self.scanners_dvc.start_write_task(
            ao_data=[[self.scan_params.set_pnts_planes[plane_idx]]],
            type=plane_axis,
        )

    def prepare_image_data(self, plane_idx):
        """Doc."""

        @dataclass
        class ImageData:

            pic1: np.ndarray
            norm1: np.ndarray
            pic2: np.ndarray
            norm2: np.ndarray
            line_ticks_V: np.ndarray
            row_ticks_V: np.ndarray

        @nb.njit(cache=True)
        def calc_pic(K, counts, x, x_min, n_lines, pxls_pl, pxl_sz_V, Ic):
            """Doc."""

            pic = np.zeros((n_lines, pxls_pl))
            norm = np.zeros((n_lines, pxls_pl))
            for j in K:
                temp_counts = np.zeros(pxls_pl)
                temp_num = np.zeros(pxls_pl)
                for i in range(Ic):
                    idx = int((x[i, j] - x_min) / pxl_sz_V) + 1
                    if 0 <= idx < pxls_pl:
                        temp_counts[idx] = temp_counts[idx] + counts[i, j]
                        temp_num[idx] = temp_num[idx] + 1
                pic[K[j], :] = temp_counts
                norm[K[j], :] = temp_num

            return pic.T, norm.T

        n_lines = self.scan_params.n_lines
        pxl_sz = self.scan_params.dim2_um / (n_lines - 1)
        scan_plane = self.scan_params.scan_plane
        ppl = self.scan_params.ppl
        ao = self.ao_buffer[0, :ppl]
        counts = np.array(self.counter_dvc.ci_buffer)

        if scan_plane in {"XY", "XZ"}:
            xc = self.scan_params.curr_ao_x
            um_per_V = self.um_v_ratio[0]

        elif scan_plane == "YZ":
            xc = self.scan_params.curr_ao_y
            um_per_V = self.um_v_ratio[1]

        pxl_sz_V = pxl_sz / um_per_V
        line_len_V = (self.scan_params.dim1_um) / um_per_V

        ppp = n_lines * ppl
        j0 = plane_idx * ppp
        J = np.arange(ppp) + j0

        if plane_idx == 0:  # if first plane
            counts = np.concatenate([[0], counts[J]])
            counts = np.diff(counts)
        else:
            counts = np.concatenate([[j0], counts[J]])
            counts = np.diff(counts)

        counts = counts.reshape(ppl, n_lines, order="F")

        x = np.tile(ao[:].T, (1, n_lines))
        x = x.reshape(ppl, n_lines, order="F")

        Ic = int(ppl / 2)
        x1 = x[:Ic, :]
        x2 = x[-1 : Ic - 1 : -1, :]
        counts1 = counts[:Ic, :]
        counts2 = counts[-1 : Ic - 1 : -1, :]

        x_min = xc - line_len_V / 2
        pxls_pl = int(line_len_V / pxl_sz_V) + 1

        # even and odd planes are scanned in the opposite directions
        if plane_idx % 2:
            # odd (backwards)
            K = np.arange(n_lines - 1, -1, -1)
        else:
            # even (forwards)
            K = np.arange(n_lines)

        # TODO: test if this fixes flipped photos on GB
        #        K = np.arange(n_lines) # TESTESTEST

        pic1, norm1 = calc_pic(K, counts1, x1, x_min, n_lines, pxls_pl, pxl_sz_V, Ic)
        pic2, norm2 = calc_pic(K, counts2, x2, x_min, n_lines, pxls_pl, pxl_sz_V, Ic)

        if pic1.shape[0] != pic1.shape[1]:
            logging.warning(f"image shape is not square ({pic1.shape}), figure this out.")

        line_scale_V = x_min + np.arange(pxls_pl) * pxl_sz_V
        row_scale_V = self.scan_params.set_pnts_lines_odd

        return ImageData(pic1, norm1, pic2, norm2, line_scale_V, row_scale_V)

    def keep_last_meas(self):
        """Doc."""

        try:
            self._app.last_img_scn.plane_type = self.scan_params.scan_plane
            self._app.last_img_scn.plane_images_data = self.plane_images_data
            self._app.last_img_scn.set_pnts_planes = self.scan_params.set_pnts_planes
        except AttributeError:
            # create a namespace if doesn't exist and restart function
            self._app.last_img_scn = SimpleNamespace()
            self.keep_last_meas()

    # TODO: generalize these and unite in base class (use basic dict and add specific, shorter dict from inheriting classes)
    def prep_data_dict(self) -> dict:
        """
        Prepare the full measurement data, in a way that
        matches current MATLAB analysis
        """

        def prep_scan_params() -> dict:

            return {
                "dim1_lines_um": self.scan_params.dim1_um,
                "dim2_col_um": self.scan_params.dim2_um,
                "dim3_um": self.scan_params.dim3_um,
                "lines": self.scan_params.n_lines,
                "planes": self.scan_params.n_planes,
                "line_freq_hz": self.scan_params.line_freq_Hz,
                "points_per_line": self.scan_params.ppl,
                "scan_type": self.scan_params.scan_plane + "scan",
                "offset_aox": self.scan_params.curr_ao_x,
                "offset_aoy": self.scan_params.curr_ao_y,
                "offset_aoz": self.scan_params.curr_ao_z,
                "offset_aix": self.scan_params.curr_ao_x,  # check
                "offset_aiy": self.scan_params.curr_ao_y,  # check
                "offset_aiz": self.scan_params.curr_ao_z,  # check
                "what_stage": "Galvanometers",
                "linear_frac": self.scan_params.lin_frac,
            }

        def prep_tdc_scan_data() -> dict:

            return {
                "plane": np.array([data for data in self.plane_data], dtype=np.object),
                "data_version": self.tdc_dvc.data_vrsn,
                "fpga_freq": self.tdc_dvc.fpga_freq_MHz,
                "pix_clk_freq": self.pxl_clk_dvc.freq_MHz,
                "laser_freq": self.tdc_dvc.laser_freq_MHz,
                "version": self.tdc_dvc.tdc_vrsn,
            }

        return {
            "tdc_scan_data": prep_tdc_scan_data(),
            "version": self.tdc_dvc.tdc_vrsn,
            "ai": np.array(self.scanners_dvc.ai_buffer, dtype=np.float),
            "cnt": np.array(self.counter_dvc.ci_buffer, dtype=np.int),
            "scan_param": prep_scan_params(),
            "pid": [],  # check
            "ao": self.ao_buffer,
            "sp": [],  # check
            "log": [],  # info in mat filename
            "lines_odd": self.scan_params.set_pnts_lines_odd,
            "is_fast_scan": True,
            "system_info": self.sys_info,
            "xyz_um_to_v": self.um_v_ratio,
        }

    async def run(self):
        """Doc."""

        self.setup_scan()

        try:
            await self.toggle_lasers()
            self.data_dvc.init_data()
            self.data_dvc.purge_buffers()
            self._app.gui.main.imp.dvc_toggle("pixel_clock", leave_on=True)
            self.scanners_dvc.init_ai_buffer(type="inf")
            self.counter_dvc.init_ci_buffer(type="inf")

        except (MeasurementError, DeviceError) as exc:
            await self.stop()
            err_hndlr(exc, locals(), sys._getframe())
            return

        else:
            n_planes = self.scan_params.n_planes
            self.plane_data = []
            self.start_time = time.perf_counter()
            self.time_passed_s = 0
            self.is_running = True
            logging.info(f"Running {self.type} measurement")

        try:  # TESTESTEST
            for plane_idx in range(n_planes):

                if self.is_running:

                    self.change_plane(plane_idx)
                    self.curr_plane_wdgt.set(plane_idx)

                    self.init_scan_tasks("FINITE")
                    self.scanners_dvc.start_tasks("ao")

                    # collect initial ai/CI to avoid possible overflow
                    self.counter_dvc.fill_ci_buffer()
                    self.scanners_dvc.fill_ai_buffer()

                    self.data_dvc.purge_buffers()

                    # recording
                    await self.record_data(timed=False)

                    # collect final ai/CI
                    self.counter_dvc.fill_ci_buffer()
                    self.scanners_dvc.fill_ai_buffer()

                    self.plane_data.append(self.data_dvc.data)
                    self.data_dvc.init_data()

                else:
                    break

        except MeasurementError as exc:
            await self.stop()
            err_hndlr(exc, locals(), sys._getframe())
            return

        except Exception as exc:  # TESTESTEST
            err_hndlr(exc, locals(), sys._getframe())  # TESTESTEST

        # finished measurement
        if self.is_running:
            # if not manually stopped
            # prepare data
            self.plane_images_data = [
                self.prepare_image_data(plane_idx) for plane_idx in range(self.scan_params.n_planes)
            ]

            # save data
            self.save_data(self.prep_data_dict(), self.build_filename())
            self.keep_last_meas()

            # show middle plane
            mid_plane = int(len(self.scan_params.set_pnts_planes) // 2)
            self.plane_shown.set(mid_plane)
            self.plane_choice.set(mid_plane)
            should_display_autocross = self.scan_params.auto_cross and (self.laser_mode == "exc")
            self._app.gui.main.imp.disp_plane_img(mid_plane, auto_cross=should_display_autocross)

        if self.is_running:  # if not manually before completion
            await self.stop()


class SFCSSolutionMeasurement(Measurement):
    """Doc."""

    dur_mul_dict = {
        "seconds": 1,
        "minutes": 60,
        "hours": 3600,
    }

    def __init__(self, app, scan_params, **kwargs):
        super().__init__(app=app, type="SFCSSolution", scan_params=scan_params, **kwargs)
        self.scan_params.scan_plane = "XY"
        self.duration_multiplier = self.dur_mul_dict[self.duration_units]
        self.duration_s = self.duration * self.duration_multiplier
        self.scanning = not (self.scan_params.pattern == "static")
        self._app.gui.main.imp.go_to_origin("XY")

    def build_filename(self, file_no: int) -> str:
        """Doc."""

        if self.repeat is True:
            return f"alignment_{self.scan_type}_{self.laser_mode}_0"

        else:
            if not self.file_template:
                self.file_template = "sol"
            return f"{self.file_template}_{self.scan_type}_{self.laser_mode}_{self.start_time_str}_{file_no}"

    def set_current_and_end_times(self) -> None:
        """
        Given a duration in seconds, returns a tuple (current_time, end_time)
        in datetime.time format, where end_time is current_time + duration_in_seconds.
        """

        curr_datetime = dt.now()
        self.start_time_str = curr_datetime.strftime("%H%M%S")
        end_datetime = curr_datetime + datetime.timedelta(seconds=int(self.duration_s))
        self.start_time_wdgt.set(curr_datetime.time())
        self.end_time_wdgt.set(end_datetime.time())

    def setup_scan(self):
        """Doc."""

        # make sure divisablility, then sync pixel clock with ao sample clock
        if (self.pxl_clk_dvc.freq_MHz * 1e6) % self.scan_params.ao_samp_freq_Hz != 0:
            raise ValueError(
                f"Pixel clock ({self.pxl_clk_dvc.freq_MHz} Hz) and ao samplng ({self.scan_params.ao_samp_freq_Hz} Hz) frequencies aren't divisible."
            )
        else:
            self.pxl_clk_dvc.low_ticks = (
                int((self.pxl_clk_dvc.freq_MHz * 1e6) / self.scan_params.ao_samp_freq_Hz) - 2
            )
        # create ao_buffer
        self.ao_buffer, self.scan_params = ScanPatternAO(
            self.scan_params.pattern, self.scan_params, self.um_v_ratio
        ).calculate_pattern()
        self.n_ao_samps = self.ao_buffer.shape[1]
        # TODO: why is the next line correct? explain and use a constant for 1.5E-7. ask Oleg
        self.ai_conv_rate = 6 * 2 * (1 / (self.scan_params.dt - 1.5e-7))

    def disp_ACF(self):
        """Doc."""

        def compute_acf(data):
            """Doc."""

            p = PhotonData()
            p.convert_fpga_data_to_photons(np.array(data, dtype=np.uint8))
            s = CorrFuncTDC()
            s.laser_freq = self.tdc_dvc.laser_freq_MHz * 1e6
            s.data["data"].append(p)
            s.correlate_regular_data()
            s.average_correlation(no_plot=True, use_numba=True)
            return s

        if self.repeat is True:
            # display ACF for alignments
            try:
                s = compute_acf(self.data_dvc.data)
            except Exception as exc:
                err_hndlr(exc, locals(), sys._getframe())
            else:
                try:
                    s.fit_correlation_function(no_plot=True)
                except fit_tools.FitError as exc:
                    # fit failed
                    err_hndlr(exc, locals(), sys._getframe(), lvl="debug")
                    self.fit_led.set(self.icon_dict["led_red"])
                    g0, tau = s.g0, 0.1
                    self.g0_wdgt.set(s.g0)
                    self.tau_wdgt.set(0)
                    self.plot_wdgt.obj.plot(s.lag, s.average_cf_cr, clear=True)
                    self.plot_wdgt.obj.plotItem.vb.setRange(
                        xRange=(math.log(0.05), math.log(5)), yRange=(-g0 * 0.1, g0 * 1.3)
                    )
                else:
                    # fit succeeded
                    self.fit_led.set(self.icon_dict["led_off"])
                    fit_params = s.fit_param["diffusion_3d_fit"]
                    g0, tau, _ = fit_params["beta"]
                    x, y = fit_params["x"], fit_params["y"]
                    fit_func = getattr(fit_tools, fit_params["fit_func"])
                    self.g0_wdgt.set(g0)
                    self.tau_wdgt.set(tau * 1e3)
                    self.plot_wdgt.obj.plot(x, y, clear=True)
                    y_fit = fit_func(x, *fit_params["beta"])
                    self.plot_wdgt.obj.plot(x, y_fit, pen="r")
                    self.plot_wdgt.obj.plotItem.autoRange()
                    logging.info(
                        f"Aligning ({self.laser_mode}): g0: {g0/1e3:.1f}K, tau: {tau*1e3:.1f} us."
                    )

    # TODO: generalize these and unite in base class (use basic dict and add specific, shorter dict from inheriting classes)
    def prep_data_dict(self) -> dict:
        """
        Prepare the full measurement data, in a way that
        matches current MATLAB analysis
        """

        def prep_ang_scan_sett_dict():
            return {
                "x": self.ao_buffer[0, :],
                "y": self.ao_buffer[1, :],
                "actual_speed": self.scan_params.eff_speed_um_s,
                "scan_freq": self.scan_params.scan_freq_Hz,
                "sample_freq": self.scan_params.ao_samp_freq_Hz,
                "points_per_line_total": self.scan_params.tot_ppl,
                "points_per_line": self.scan_params.ppl,
                "n_lines": self.scan_params.n_lines,
                "line_length": self.scan_params.lin_len,
                "total_length": self.scan_params.tot_len,
                "max_line_length": self.scan_params.max_line_len_um,
                "line_shift": self.scan_params.line_shift_um,
                "angle_degrees": self.scan_params.angle_deg,
                "linear_frac": self.scan_params.lin_frac,
                "linear_part": self.scan_params.lin_part,
                "x_lim": self.scan_params.x_lim,
                "y_lim": self.scan_params.y_lim,
            }

        if self.scanning:
            full_data = {
                "data": np.array(self.data_dvc.data, dtype=np.uint8),
                "data_version": self.tdc_dvc.data_vrsn,
                "fpga_freq": self.tdc_dvc.fpga_freq_MHz,
                "pix_clk_freq": self.pxl_clk_dvc.freq_MHz,
                "laser_freq": self.tdc_dvc.laser_freq_MHz,
                "version": self.tdc_dvc.tdc_vrsn,
                "ai": np.array(self.scanners_dvc.ai_buffer, dtype=np.float),
                "ao": self.ao_buffer,
                "avg_cnt_rate_khz": self.counter_dvc.avg_cnt_rate_khz,
            }
            if self.scan_params.pattern == "circle":
                full_data["circle_speed_um_sec"] = self.scan_params.speed_um_s
            else:
                full_data["angular_scan_settings"] = prep_ang_scan_sett_dict()

            return {"full_data": full_data, "system_info": self.sys_info}

        else:
            full_data = {
                "data": np.array(self.data_dvc.data, dtype=np.uint8),
                "data_version": self.tdc_dvc.data_vrsn,
                "fpga_freq": self.tdc_dvc.fpga_freq_MHz,
                "laser_freq": self.tdc_dvc.laser_freq_MHz,
                "version": self.tdc_dvc.tdc_vrsn,
                "avg_cnt_rate_khz": self.counter_dvc.avg_cnt_rate_khz,
            }

            return {"full_data": full_data, "system_info": self.sys_info}

    async def run(self):
        """Doc."""

        # initialize gui start/end times
        self.set_current_and_end_times()

        # estimate time per file
        apprx_byte_rate_hz = self.counter_dvc.avg_cnt_rate_khz * 1e3 * 7
        bytes_per_file = self.max_file_size_mb * 1e6
        apprx_file_time_min = bytes_per_file / apprx_byte_rate_hz / 60
        self._app.gui.main.solScanFileTime.setValue(apprx_file_time_min)

        # turn on lasers
        try:
            await self.toggle_lasers()
        except (MeasurementError, DeviceError) as exc:
            await self.stop()
            err_hndlr(exc, locals(), sys._getframe())
            return

        try:
            if self.scanning:
                self.setup_scan()
                self._app.gui.main.imp.dvc_toggle("pixel_clock", leave_on=True)
                # make the circular ai buffer clip as long as the ao buffer
                self.scanners_dvc.init_ai_buffer(type="circular", size=self.ao_buffer.shape[1])
            else:
                self.scanners_dvc.init_ai_buffer()

            if not self.repeat:
                # during alignment we don't change the counter_dvc tasks, do no need to initialize
                self.counter_dvc.init_ci_buffer()

            self.data_dvc.init_data()
            self.data_dvc.purge_buffers()

        except DeviceError as exc:
            await self.stop()
            err_hndlr(exc, locals(), sys._getframe())
            return

        else:
            self.is_running = True
            self.start_time = time.perf_counter()
            self.time_passed_s = 0
            file_num = 1
            logging.info(f"Running {self.type} measurement")

        try:  # TESTESTEST`
            while self.is_running and self.time_passed_s < self.duration_s:

                if not self.repeat:
                    # during alignment we don't change the counter_dvc tasks, do no need to initialize
                    self.counter_dvc.init_ci_buffer()
                if self.scanning:
                    # re-start scan for each file
                    self.scanners_dvc.init_ai_buffer(type="circular", size=self.ao_buffer.shape[1])
                    self.init_scan_tasks("CONTINUOUS")
                    self.scanners_dvc.start_tasks("ao")
                else:
                    self.scanners_dvc.init_ai_buffer()

                self.file_num_wdgt.set(file_num)

                # reading
                if self.repeat:
                    await self.record_data(timed=True)
                else:
                    await self.record_data(timed=True, size_limited=True)

                # collect final ai/CI
                self.counter_dvc.fill_ci_buffer()
                self.scanners_dvc.fill_ai_buffer()

                # case aligning and not manually stopped
                if self.repeat and self.is_running:
                    self.disp_ACF()
                    self.save_data(self.prep_data_dict(), self.build_filename(0))
                    # reset measurement
                    self.set_current_and_end_times()
                    self.start_time = time.perf_counter()
                    self.time_passed_s = 0

                # case measuring and finished file or measurement
                elif not self.repeat:
                    self.save_data(self.prep_data_dict(), self.build_filename(file_num))
                    if self.scanning:
                        self.scanners_dvc.init_ai_buffer(
                            type="circular", size=self.ao_buffer.shape[1]
                        )

                # initialize data buffers for next file
                self.data_dvc.init_data()
                self.data_dvc.purge_buffers()

                file_num += 1

        except Exception as exc:  # TESTESTEST
            err_hndlr(exc, locals(), sys._getframe())

        if self.is_running:  # if not manually stopped
            await self.stop()


class MeasurementError(Exception):
    pass
