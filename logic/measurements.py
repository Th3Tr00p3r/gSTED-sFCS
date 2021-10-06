"""Measurements."""

import asyncio
import datetime
import logging
import os
import pickle
import re
import sys
import time
from contextlib import suppress
from datetime import datetime as dt
from types import SimpleNamespace

import nidaqmx.constants as ni_consts
import numpy as np

import gui.widgets as wdgts
import logic.devices as dvcs
from data_analysis.correlation_function import CorrFuncTDC
from data_analysis.image import ImageScanData
from logic.scan_patterns import ScanPatternAO
from utilities import errors, fit_tools, helper


class Measurement:
    """Base class for measurements"""

    def __init__(self, app, type: str, scan_params, **kwargs):

        self._app = app
        self.type = type
        self.tdc_dvc = app.devices.TDC
        self.data_dvc = app.devices.UM232H
        self.icon_dict = wdgts.get_icon_paths()  # get icons
        [setattr(self, key, val) for key, val in kwargs.items()]
        self.counter_dvc = app.devices.photon_detector
        self.pxl_clk_dvc = app.devices.pixel_clock
        self.scanners_dvc = app.devices.scanners
        self.scan_params = scan_params
        self.um_v_ratio = tuple(getattr(self.scanners_dvc, f"{ax}_um2v_const") for ax in "xyz")
        # TODO: check if 'ai_scaling_xyz' matches today's ratio
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
            "ai_scaling_xyz": (1.243, 1.239, 1),
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

        # turn off devices and return to starting position if scanning
        with suppress(errors.DeviceError, MeasurementError):
            await self.toggle_lasers(finish=True)
            self._app.gui.main.impl.dvc_toggle("TDC", leave_off=True)

            if self.scanning:
                self.return_to_regular_tasks()
                self._app.gui.main.impl.dvc_toggle("pixel_clock", leave_off=True)

                if self.type == "SFCSSolution":
                    self._app.gui.main.impl.go_to_origin("XY")
                    type_ = "solution"
                elif self.type == "SFCSImage":
                    self._app.gui.main.impl.move_scanners(destination=self.scan_params.initial_ao)
                    type_ = "image"

            # TODO: make this more readable - the idea is that static measurements are also of type_ "solution"
            # An option is to change the type of measurement from "SFCSSolution" to "solution" ("SFCSImage" -> "image")
            # or the other way around.
            elif self.type == "SFCSSolution":
                type_ = "solution"

        self.is_running = False
        await self._app.gui.main.impl.toggle_meas(self.type, self.laser_mode.capitalize())
        self.prog_bar_wdgt.set(0)
        self._app.gui.main.impl.populate_all_data_dates(type_)  # refresh saved measurements
        logging.info(f"{self.type} measurement stopped")

    async def record_data(self, timed: bool = False, size_limited: bool = False):
        """
        Turn ON the TDC (FPGA), read while conditions are met,
        turn OFF TDC and read leftover data,
        return the time it took (seconds).
        """

        self._app.gui.main.impl.dvc_toggle("TDC", leave_on=True)

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
                and not (
                    overdue := self.time_passed_s
                    > (self.est_plane_duration * (self.curr_plane + 1) * 1.1)
                )
            ):
                await self.data_dvc.read_TDC()
                self.time_passed_s = time.perf_counter() - self.start_time
            if overdue:
                raise MeasurementError(
                    "Tasks are overdue. Check that all relevant devices are turned ON"
                )

        await self.data_dvc.read_TDC()  # read leftovers
        self._app.gui.main.impl.dvc_toggle("TDC", leave_off=True)

    def save_data(self, data_dict: dict, file_name: str) -> None:
        """
        Create a directory of today's date, and there
        save measurement data as a .pkl file.

        Note: does not handle overnight measurements during which the date changes
        """

        today_dir = os.path.join(self.save_path, dt.now().strftime("%d_%m_%Y"))

        if self.type == "SFCSSolution":
            if self.final:
                save_path = today_dir
            else:
                save_path = os.path.join(today_dir, "solution")
        elif self.type == "SFCSImage":
            save_path = os.path.join(today_dir, "image")
        else:
            raise NotImplementedError(f"Measurements of type '{self.type}' are not handled.")

        os.makedirs(save_path, exist_ok=True)

        file_path = os.path.join(save_path, re.sub("\\s", "_", file_name))

        logging.debug(f"Saving measurement file: '{file_name}'...")

        with open(file_path + ".pkl", "wb") as f:
            pickle.dump(data_dict, f)

        logging.debug("Saved Successfuly.")

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
                self._app.gui.main.impl.dvc_toggle("exc_laser", leave_off=True)
            elif self.laser_mode == "dep":
                self._app.gui.main.impl.dvc_toggle("dep_shutter", leave_off=True)
            elif self.laser_mode == "sted":
                self._app.gui.main.impl.dvc_toggle("exc_laser", leave_off=True)
                self._app.gui.main.impl.dvc_toggle("dep_shutter", leave_off=True)
        else:
            # measurement begins
            if self.laser_mode == "exc":
                # turn excitation ON and depletion shutter OFF
                self._app.gui.main.impl.dvc_toggle("exc_laser", leave_on=True)
                self._app.gui.main.impl.dvc_toggle("dep_shutter", leave_off=True)
            elif self.laser_mode == "dep":
                await self.prep_dep() if not self.laser_dvcs.dep.emission_state else None
                # turn depletion shutter ON and excitation OFF
                self._app.gui.main.impl.dvc_toggle("dep_shutter", leave_on=True)
                self._app.gui.main.impl.dvc_toggle("exc_laser", leave_off=True)
            elif self.laser_mode == "sted":
                await self.prep_dep() if not self.laser_dvcs.dep.emission_state else None
                # turn both depletion shutter and excitation ON
                self._app.gui.main.impl.dvc_toggle("exc_laser", leave_on=True)
                self._app.gui.main.impl.dvc_toggle("dep_shutter", leave_on=True)

            if current_emission_state() != self.laser_mode:
                # cancel measurement if relevant lasers are not ON,
                raise MeasurementError(
                    f"Requested laser mode ({self.laser_mode}) was not attained."
                )

    async def prep_dep(self):
        """Doc."""

        toggle_succeeded = self._app.gui.main.impl.dvc_toggle(
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

    def init_scan_tasks(self, ao_sample_mode: str, ao_only=False) -> None:
        """Doc."""

        # multiplication of the internal PC buffer size by this factor avoids buffer overflow
        buff_ovflw_const = 1.2  # could be redundant, but just to be safe...

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

        if not ao_only:
            xy_ao_task = [task for task in self.scanners_dvc.tasks.ao if (task.name == "AO XY")][0]
            ao_clk_src = xy_ao_task.timing.samp_clk_term

            # init continuous AI
            self.scanners_dvc.start_scan_read_task(
                samp_clk_cnfg={},
                timing_params={
                    "samp_quant_samp_mode": ni_consts.AcquisitionType.CONTINUOUS,
                    "samp_quant_samp_per_chan": int(self.n_ao_samps * buff_ovflw_const),
                    "samp_timing_type": ni_consts.SampleTimingType.SAMPLE_CLOCK,
                    "samp_clk_src": ao_clk_src,
                    "ai_conv_rate": self.ai_conv_rate,
                },
            )

            # init continuous CI
            self.counter_dvc.start_scan_read_task(
                samp_clk_cnfg={},
                timing_params={
                    "samp_quant_samp_mode": ni_consts.AcquisitionType.CONTINUOUS,
                    "samp_quant_samp_per_chan": int(self.n_ao_samps * buff_ovflw_const),
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
        self.scan_params.initial_ao = tuple(getattr(scan_params, f"curr_ao_{ax}") for ax in "xyz")

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
                return helper.div_ceil(ratio, 2) * 2
            else:
                return helper.div_ceil(ppl, 2) * 2

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
        # NOTE: why is the next line correct? explain and use a constant for 1.5E-7. ask Oleg
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
        """Doc."""

        return {
            "version": self.tdc_dvc.tdc_vrsn,
            "ai": np.array(self.scanners_dvc.ai_buffer, dtype=np.float),
            "ci": np.array(self.counter_dvc.ci_buffer, dtype=np.int),
            "pid": [],  # check
            "ao": self.ao_buffer,
            "is_fast_scan": True,
            "system_info": self.sys_info,
            "xyz_um_to_v": self.um_v_ratio,
            "tdc_scan_data": {
                # TODO: prepare a function to cut the data into planes, similar to how the counts are cut
                "data": np.array(self.data_dvc.data, dtype=np.uint8),
                "data_version": self.tdc_dvc.data_vrsn,  # already in 'version'
                "fpga_freq_mhz": self.tdc_dvc.fpga_freq_mhz,
                "pix_clk_freq_mhz": self.pxl_clk_dvc.freq_MHz,
                "laser_freq_mhz": self.tdc_dvc.laser_freq_mhz,
                "version": self.tdc_dvc.tdc_vrsn,
            },
            "scan_param": {
                "set_pnts_lines_odd": self.scan_params.set_pnts_lines_odd,
                "dim1_lines_um": self.scan_params.dim1_lines_um,
                "dim2_col_um": self.scan_params.dim2_col_um,
                "dim3_um": self.scan_params.dim3_um,
                "n_lines": self.scan_params.n_lines,
                "n_planes": self.scan_params.n_planes,
                "line_freq_hz": self.scan_params.line_freq_Hz,
                "ppl": self.scan_params.ppl,
                "scan_plane": self.scan_params.scan_plane,
                "initial_ao": self.scan_params.initial_ao,
                "what_stage": "Galvanometers",
                "linear_frac": self.scan_params.lin_frac,
            },
        }

    async def run(self):
        """Doc."""

        self.setup_scan()

        try:
            await self.toggle_lasers()
            self.data_dvc.init_data()
            self.data_dvc.purge_buffers()
            self._app.gui.main.impl.dvc_toggle("pixel_clock", leave_on=True)
            self.scanners_dvc.init_ai_buffer(type="inf")
            self.counter_dvc.init_ci_buffer(type="inf")

        except (MeasurementError, errors.DeviceError) as exc:
            await self.stop()
            self.type = None
            errors.err_hndlr(exc, sys._getframe(), locals())
            return

        else:
            n_planes = self.scan_params.n_planes
            self.plane_data = []
            self.start_time = time.perf_counter()
            self.time_passed_s = 0
            self.is_running = True
            logging.info(f"Running {self.type} measurement")

        # start all tasks (AO, AI, CI)
        self.init_scan_tasks("FINITE")
        self.scanners_dvc.start_tasks("ao")

        try:  # TESTESTEST
            for plane_idx in range(n_planes):

                if self.is_running:

                    self.change_plane(plane_idx)
                    self.curr_plane_wdgt.set(plane_idx)
                    self.curr_plane = plane_idx

                    # Re-start only AO
                    self.init_scan_tasks("FINITE", ao_only=True)
                    self.scanners_dvc.start_tasks("ao")

                    # recording
                    await self.record_data(timed=False)

                    # collect final ai/CI
                    self.counter_dvc.fill_ci_buffer()
                    self.scanners_dvc.fill_ai_buffer()

                else:
                    break

        except MeasurementError as exc:
            await self.stop()
            errors.err_hndlr(exc, sys._getframe(), locals())
            return

        except Exception as exc:  # TESTESTEST
            errors.err_hndlr(exc, sys._getframe(), locals())  # TESTESTEST

        # finished measurement
        if self.is_running:
            # if not manually stopped
            # prepare data
            counts = np.array(self.counter_dvc.ci_buffer, dtype=np.int)
            self.plane_images_data = ImageScanData(
                counts, self.ao_buffer, vars(self.scan_params).copy(), self.um_v_ratio
            )

            # save data
            self.save_data(self.prep_data_dict(), self.build_filename())
            self.keep_last_meas()

            # show middle plane
            mid_plane = len(self.scan_params.set_pnts_planes) // 2
            self.plane_shown.set(mid_plane)
            self.plane_choice.set(mid_plane)
            should_display_autocross = self.scan_params.auto_cross and (self.laser_mode == "exc")
            self._app.gui.main.impl.disp_plane_img(mid_plane, auto_cross=should_display_autocross)

        if self.is_running:  # if not manually before completion
            await self.stop()

        self.type = None


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
        self._app.gui.main.impl.go_to_origin("XY")

    def build_filename(self, file_no: int) -> str:
        """Doc."""

        if self.final is True:
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
        # NOTE: why is the next line correct? explain and use a constant for 1.5E-7. ask Oleg
        self.ai_conv_rate = 6 * 2 * (1 / (self.scan_params.dt - 1.5e-7))

    def disp_ACF(self):
        """Doc."""

        def compute_acf(data):
            """Doc."""

            s = CorrFuncTDC()
            s.after_pulse_param = self.sys_info["after_pulse_param"]
            s.laser_freq_hz = self.tdc_dvc.laser_freq_mhz * 1e6
            p = s.convert_fpga_data_to_photons(
                np.array(data, dtype=np.uint8), ignore_coarse_fine=True
            )
            s.data.append(p)
            s.correlate_and_average(self.laser_mode)
            return s

        if self.repeat is True:
            # display ACF for alignments
            try:
                s = compute_acf(self.data_dvc.data)
            except Exception as exc:
                errors.err_hndlr(exc, sys._getframe(), locals())
            else:
                cf = s.cf[self.laser_mode]
                try:
                    cf.fit_correlation_function()
                except fit_tools.FitError as exc:
                    # fit failed
                    errors.err_hndlr(exc, sys._getframe(), locals(), lvl="debug")
                    self.fit_led.set(self.icon_dict["led_red"])
                    g0, tau = cf.g0, 0.1
                    self.g0_wdgt.set(g0)
                    self.tau_wdgt.set(0)
                    self.plot_wdgt.obj.plot_acfs((cf.lag, "lag"), cf.avg_cf_cr, g0)
                else:
                    # fit succeeded
                    self.fit_led.set(self.icon_dict["led_off"])
                    fit_params = cf.fit_param["diffusion_3d_fit"]
                    g0, tau, _ = fit_params["beta"]
                    x, y = fit_params["x"], fit_params["y"]
                    fit_func = getattr(fit_tools, fit_params["func_name"])
                    self.g0_wdgt.set(g0)
                    self.tau_wdgt.set(tau * 1e3)
                    self.plot_wdgt.obj.plot_acfs((x, "lag"), y, g0)
                    y_fit = fit_func(x, *fit_params["beta"])
                    self.plot_wdgt.obj.plot(x, y_fit, "-.r")
                    self.plot_wdgt.obj.ax.autoscale()
                    logging.info(
                        f"Aligning ({self.laser_mode}): g0: {g0/1e3:.1f}K, tau: {tau*1e3:.1f} us."
                    )

    # TODO: generalize these and unite in base class (use basic dict and add specific, shorter dict from inheriting classes)
    def prep_data_dict(self) -> dict:
        """Doc."""

        full_data = {
            "duration_s": self.duration_s,
            "data": np.array(self.data_dvc.data, dtype=np.uint8),
            "version": self.tdc_dvc.tdc_vrsn,
            "data_version": self.tdc_dvc.data_vrsn,
            "fpga_freq_mhz": self.tdc_dvc.fpga_freq_mhz,
            "laser_freq_mhz": self.tdc_dvc.laser_freq_mhz,
            "avg_cnt_rate_khz": self.counter_dvc.avg_cnt_rate_khz,
        }

        if self.scanning:
            full_data["pix_clk_freq_mhz"] = self.pxl_clk_dvc.freq_MHz
            full_data["ai"] = np.array(self.scanners_dvc.ai_buffer, dtype=np.float32)
            full_data["ao"] = self.ao_buffer

            if self.scan_params.pattern == "circle":
                full_data["circle_speed_um_s"] = self.scan_params.speed_um_s
            elif self.scan_params.pattern == "angular":
                full_data["angular_scan_settings"] = {
                    "x": self.ao_buffer[0, :],
                    "y": self.ao_buffer[1, :],
                    "actual_speed_um_s": self.scan_params.eff_speed_um_s,
                    "scan_freq_hz": self.scan_params.scan_freq_Hz,
                    "sample_freq_hz": self.scan_params.ao_samp_freq_Hz,
                    "points_per_line_total": self.scan_params.tot_ppl,
                    "ppl": self.scan_params.ppl,
                    "n_lines": self.scan_params.n_lines,
                    "line_length": self.scan_params.lin_len,
                    "total_length": self.scan_params.tot_len,
                    "max_line_length_um": self.scan_params.max_line_len_um,
                    "line_shift_um": self.scan_params.line_shift_um,
                    "angle_degrees": self.scan_params.angle_deg,
                    "linear_frac": self.scan_params.lin_frac,
                    "linear_part": self.scan_params.lin_part,
                    "x_lim": self.scan_params.x_lim,
                    "y_lim": self.scan_params.y_lim,
                }

        return {"full_data": full_data, "system_info": self.sys_info}

    async def run(self):
        """Doc."""

        # initialize gui start/end times
        self.set_current_and_end_times()

        # turn on lasers
        try:
            await self.toggle_lasers()
        except (MeasurementError, errors.DeviceError) as exc:
            await self.stop()
            self.type = None
            errors.err_hndlr(exc, sys._getframe(), locals())
            return

        try:
            if self.scanning:
                self.setup_scan()
                self._app.gui.main.impl.dvc_toggle("pixel_clock", leave_on=True)
                # make the circular ai buffer clip as long as the ao buffer
                self.scanners_dvc.init_ai_buffer(type="circular", size=self.ao_buffer.shape[1])
            else:
                self.scanners_dvc.init_ai_buffer()

            if not self.repeat:
                # during alignment we don't change the counter_dvc tasks, do no need to initialize
                self.counter_dvc.init_ci_buffer()

        except errors.DeviceError as exc:
            await self.stop()
            errors.err_hndlr(exc, sys._getframe(), locals())
            return

        else:
            self.is_running = True
            self.start_time = time.perf_counter()
            self.time_passed_s = 0
            file_num = 1
            logging.info(f"Running {self.type} measurement")

        try:  # TESTESTEST`
            while self.is_running and self.time_passed_s < self.duration_s:

                # initialize data buffer
                self.data_dvc.init_data()
                self.data_dvc.purge_buffers()

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

                logging.debug("FPGA reading starts.")

                # reading
                if self.repeat:
                    await self.record_data(timed=True)
                else:
                    await self.record_data(timed=True, size_limited=True)

                logging.debug("FPGA reading finished.")

                # collect final ai/CI
                self.counter_dvc.fill_ci_buffer()
                self.scanners_dvc.fill_ai_buffer()

                # case aligning and not manually stopped
                if self.repeat and self.is_running:
                    self.disp_ACF()
                    # reset measurement
                    self.set_current_and_end_times()
                    self.start_time = time.perf_counter()
                    self.time_passed_s = 0

                # case final alignment and not manually stopped
                elif self.final and self.is_running:
                    self.disp_ACF()
                    self.save_data(self.prep_data_dict(), self.build_filename(0))

                # case regular measurement and finished file or measurement
                elif not self.repeat:
                    self.save_data(self.prep_data_dict(), self.build_filename(file_num))
                    if self.scanning:
                        self.scanners_dvc.init_ai_buffer(
                            type="circular", size=self.ao_buffer.shape[1]
                        )

                file_num += 1

        except Exception as exc:  # TESTESTEST
            errors.err_hndlr(exc, sys._getframe(), locals())

        if self.is_running:  # if not manually stopped
            await self.stop()

        self.type = None


class MeasurementError(Exception):
    pass
