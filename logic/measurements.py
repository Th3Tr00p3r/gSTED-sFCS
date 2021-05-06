# -*- coding: utf-8 -*-
"""Measurements Module."""

import asyncio
import datetime
import logging
import math
import pickle
import time
from multiprocessing import Process
from types import SimpleNamespace
from typing import NoReturn

import numpy as np
import scipy.io as sio
from PyQt5.QtGui import QIcon
from scipy.optimize import OptimizeWarning

import gui.icons.icon_paths as icon_path
import utilities.constants as consts
from data_analysis import FitTools
from data_analysis.CorrFuncTDCclass import CorrFuncTDCclass
from data_analysis.PhotonDataClass import PhotonDataClass
from logic.scan_patterns import ScanPatternAO
from utilities.errors import err_hndlr
from utilities.helper import ImageData, div_ceil, get_datetime_str


class Measurement:
    """Base class for measurements"""

    def __init__(self, app, type: str, scan_params=None, **kwargs):

        self._app = app
        self.type = type
        self.tdc_dvc = app.devices.TDC
        self.pxl_clk_dvc = app.devices.TDC
        self.data_dvc = app.devices.UM232H
        [setattr(self, key, val) for key, val in kwargs.items()]
        self.counter_dvc = app.devices.COUNTER
        if scan_params is not None:
            self.pxl_clk_dvc = app.devices.PXL_CLK
            self.scanners_dvc = app.devices.SCANNERS
            self.scan_params = scan_params
            self.um_V_ratio = tuple(
                getattr(self.scanners_dvc, f"{ax.lower()}_um2V_const") for ax in "XYZ"
            )
            self.sys_info = dict(
                Setup="STED with galvos",
                AfterPulseParam=[
                    -0.004057535648770,
                    -0.107704707102406,
                    -1.069455813887638,
                    -4.827204349438697,
                    -10.762333427569356,
                    -7.426041455313178,
                ],
                AI_ScalingXYZ=[1.243, 1.239, 1],
                XYZ_um_to_V=self.um_V_ratio,
            )
        self.laser_dvcs = SimpleNamespace()
        self.laser_dvcs.exc = app.devices.EXC_LASER
        self.laser_dvcs.dep = app.devices.DEP_LASER
        self.laser_dvcs.dep_shutter = app.devices.DEP_SHUTTER
        self.is_running = False

    async def start(self):
        """Doc."""

        self.data_dvc.purge()
        self.is_running = True
        logging.info(f"{self.type} measurement started")
        await self.run()

    def stop(self):
        """Doc."""

        self.is_running = False
        self.prog_bar_wdgt.set(0)
        logging.info(f"{self.type} measurement stopped")
        self._app.meas.type = None

    async def record_data(self, timed: bool) -> NoReturn:
        """
        Turn ON the TDC (FPGA), read while conditions are met,
        turn OFF TDC and read leftover data.
        """

        async def read_and_track_time():
            await self.data_dvc.read_TDC()
            self.time_passed = time.perf_counter() - self.start_time

        self._app.gui.main.imp.dvc_toggle("TDC")

        if timed:
            while (
                self.time_passed < self.duration * self.duration_multiplier
                and self.is_running
            ):
                await read_and_track_time()

        else:
            while self.is_running and not self.scanners_dvc.are_tasks_done("ao"):
                await read_and_track_time()

        self._app.gui.main.imp.dvc_toggle("TDC")
        await read_and_track_time()

    def save_data(self, data_dict: dict, file_name: str) -> NoReturn:
        """
        Save measurement data as a .mat (MATLAB) file or a
        .pkl file. .mat files can be analyzed in MATLAB using our
        current MATLAB-based analysis (or later in Python using sio.loadmat()).
        Note: saving as .mat takes longer
        """

        file_path = self.save_path + file_name

        if self.save_frmt == "MATLAB":
            # .mat
            sio.savemat(file_path + ".mat", data_dict)

        else:
            # .pkl
            with open(file_path + ".pkl", "wb") as f:
                pickle.dump(data_dict, f)

    async def toggle_lasers(self, finish=False) -> NoReturn:
        """Doc."""

        if self.scan_params.sted_mode:
            self.scan_params.exc_mode = True
            self.scan_params.dep_mode = True

        if finish:
            if self.scan_params.exc_mode:
                self._app.gui.main.imp.dvc_toggle("EXC_LASER", leave_off=True)
            if self.scan_params.dep_mode:
                self._app.gui.main.imp.dvc_toggle(
                    "DEP_SHUTTER", toggle_mthd="laser_toggle", leave_off=True
                )
        else:
            if self.scan_params.exc_mode:
                self._app.gui.main.imp.dvc_toggle("EXC_LASER", leave_on=True)
            if self.scan_params.dep_mode:
                if self.laser_dvcs.dep.state is False:
                    logging.info(
                        f"{consts.DEP_LASER.log_ref} isn't on. Turnning on and waiting 5 s before measurement."
                    )
                    self._app.gui.main.imp.dvc_toggle("DEP_LASER")
                    await asyncio.sleep(5)
                self._app.gui.main.imp.dvc_toggle(
                    "DEP_SHUTTER", toggle_mthd="laser_toggle", leave_on=True
                )
            self.get_laser_config()

    def get_laser_config(self) -> NoReturn:
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

        self.laser_config = laser_config

    def init_scan_tasks(self, ao_sample_mode: str) -> NoReturn:
        """Doc."""

        ao_sample_mode = getattr(consts.NI.AcquisitionType, ao_sample_mode)

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

        ao_clk_src = self.scanners_dvc.tasks.ao["AO XY"].timing.samp_clk_term

        self.scanners_dvc.start_scan_read_task(
            samp_clk_cnfg={},
            timing_params={
                "samp_quant_samp_mode": consts.NI.AcquisitionType.CONTINUOUS,
                "samp_quant_samp_per_chan": int(self.n_ao_samps * 1.2),
                "samp_timing_type": consts.NI.SampleTimingType.SAMPLE_CLOCK,
                "samp_clk_src": ao_clk_src,
                "ai_conv_rate": self.ai_conv_rate,
            },
        )

        self.counter_dvc.start_scan_read_task(
            samp_clk_cnfg={},
            timing_params={
                "samp_quant_samp_mode": consts.NI.AcquisitionType.CONTINUOUS,
                "samp_quant_samp_per_chan": int(self.n_ao_samps * 1.2),
                "samp_timing_type": consts.NI.SampleTimingType.SAMPLE_CLOCK,
                "samp_clk_src": ao_clk_src,
            },
        )

    def return_to_regular_tasks(self):
        """Close AO tasks and resume continuous AI/CI"""

        self.scanners_dvc.start_continuous_read_task()
        self.counter_dvc.start_continuous_read_task()
        self.scanners_dvc.close_tasks("ao")


class SFCSImageMeasurement(Measurement):
    """Doc."""

    def __init__(self, app, scan_params, **kwargs):
        super().__init__(app=app, type="SFCSImage", scan_params=scan_params, **kwargs)

    def build_filename(self) -> str:
        return f"{self.file_template}_{self.laser_config}_{self.scan_params.scan_plane}_{get_datetime_str()}"

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
        # create ao_buffer
        (
            self.ao_buffer,
            self.scan_params.set_pnts_lines_odd,
            self.scan_params.set_pnts_lines_even,
            self.scan_params.set_pnts_planes,
        ) = ScanPatternAO("image", self.scan_params, self.um_V_ratio).calculate_ao()
        self.n_ao_samps = self.ao_buffer.shape[1]
        # TODO: why is the next line correct? explain and use a constant for 1.5E-7. ask Oleg
        self.ai_conv_rate = (
            self.scanners_dvc.ai_buffer.shape[0]
            * 2
            * (1 / (self.scan_params.dt - 1.5e-7))
        )
        self.est_duration = (
            self.n_ao_samps
            * self.scan_params.dt
            * len(self.scan_params.set_pnts_planes)
        )
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

        n_lines = self.scan_params.n_lines
        pxl_sz = self.scan_params.dim2_um / (n_lines - 1)
        scan_plane = self.scan_params.scan_plane
        ppl = self.scan_params.ppl
        ao = self.ao_buffer[0, :ppl]
        counts = self.counter_dvc.ci_buffer

        if scan_plane in {"XY", "XZ"}:
            xc = self.scan_params.curr_ao_x
            um_per_V = self.um_V_ratio[0]

        elif scan_plane == "YZ":
            xc = self.scan_params.curr_ao_y
            um_per_V = self.um_V_ratio[1]

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

        pic1 = np.zeros((n_lines, pxls_pl))
        norm1 = np.zeros((n_lines, pxls_pl))
        for j in K:
            temp_counts = np.zeros(pxls_pl)
            temp_num = np.zeros(pxls_pl)
            for i in range(Ic):
                idx = int((x1[i, j] - x_min) / pxl_sz_V) + 1
                if 0 <= idx < pxls_pl:
                    temp_counts[idx] = temp_counts[idx] + counts1[i, j]
                    temp_num[idx] = temp_num[idx] + 1
            pic1[K[j], :] = temp_counts
            norm1[K[j], :] = temp_num

        pic2 = np.zeros((n_lines, pxls_pl))
        norm2 = np.zeros((n_lines, pxls_pl))
        for j in K:
            temp_counts = np.zeros(pxls_pl)
            temp_num = np.zeros(pxls_pl)
            for i in range(Ic):
                idx = int((x2[i, j] - x_min) / pxl_sz_V) + 1
                if 0 <= idx < pxls_pl:
                    temp_counts[idx] = temp_counts[idx] + counts2[i, j]
                    temp_num[idx] = temp_num[idx] + 1
            pic2[K[j], :] = temp_counts
            norm2[K[j], :] = temp_num

        line_scale_V = x_min + np.arange(pxls_pl) * pxl_sz_V
        row_scale_V = self.scan_params.set_pnts_lines_odd

        return ImageData(pic1, norm1, pic2, norm2, line_scale_V, row_scale_V)

    def keep_last_meas(self):
        """Doc."""

        self._app.last_img_scn = SimpleNamespace()
        self._app.last_img_scn.plane_type = self.scan_params.scan_plane
        self._app.last_img_scn.plane_images_data = self.plane_images_data
        self._app.last_img_scn.set_pnts_planes = self.scan_params.set_pnts_planes

    def prep_data_dict(self) -> dict:
        """
        Prepare the full measurement data, in a way that
        matches current MATLAB analysis
        """

        def prep_scan_params() -> dict:

            return {
                "Dimension1_lines_um": self.scan_params.dim1_um,
                "Dimension2_col_um": self.scan_params.dim2_um,
                "Dimension3_um": self.scan_params.dim3_um,
                "Lines": self.scan_params.n_lines,
                "Planes": self.scan_params.n_planes,
                "Line_freq_Hz": self.scan_params.line_freq_Hz,
                "Points_per_Line": self.scan_params.ppl,
                "ScanType": self.scan_params.scan_plane + "scan",
                "Offset_AOX": self.scan_params.curr_ao_x,
                "Offset_AOY": self.scan_params.curr_ao_y,
                "Offset_AOZ": self.scan_params.curr_ao_z,
                "Offset_AIX": self.scan_params.curr_ao_x,  # check
                "Offset_AIY": self.scan_params.curr_ao_y,  # check
                "Offset_AIZ": self.scan_params.curr_ao_z,  # check
                "whatStage": "Galvanometers",
                "LinFrac": self.scan_params.lin_frac,
            }

        def prep_tdc_scan_data() -> dict:

            return {
                "Plane": np.array([data for data in self.plane_data], dtype=np.object),
                "DataVersion": self.tdc_dvc.data_vrsn,
                "FpgaFreq": self.tdc_dvc.fpga_freq_MHz,
                "PixelFreq": self.pxl_clk_dvc.freq_MHz,
                "LaserFreq": self.tdc_dvc.laser_freq_MHz,
                "Version": self.tdc_dvc.tdc_vrsn,
            }

        return {
            "PixClkFreq": self.pxl_clk_dvc.freq_MHz,
            "TdcScanData": prep_tdc_scan_data(),
            "version": self.tdc_dvc.tdc_vrsn,
            "AI": self.scanners_dvc.ai_buffer,
            "Cnt": self.counter_dvc.ci_buffer,
            "ScanParam": prep_scan_params(),
            "PID": [],  # check
            "AO": self.ao_buffer,
            "SP": [],  # check
            "log": [],  # info in mat filename
            "LinesOdd": self.scan_params.set_pnts_lines_odd,
            "FastScan": True,
            "SystemInfo": self.sys_info,
            "XYZ_um_to_V": self.um_V_ratio,
        }

    async def run(self):
        """Doc."""

        await self.toggle_lasers()
        self.setup_scan()
        self._app.gui.main.imp.dvc_toggle("PXL_CLK")

        self.data_dvc.init_data()
        self.data_dvc.purge()
        self.scanners_dvc.init_ai_buffer()
        self.counter_dvc.init_ci_buffer()

        n_planes = self.scan_params.n_planes
        self.plane_data = []
        self.start_time = time.perf_counter()
        self.time_passed = 0
        logging.info(f"Running {self.type} measurement")
        for plane_idx in range(n_planes):

            if self.is_running:

                self.change_plane(plane_idx)
                self.curr_plane_wdgt.set(plane_idx)

                self.init_scan_tasks("FINITE")
                self.scanners_dvc.start_tasks("ao")

                # collect initial AI/CI to avoid possible overflow
                self.counter_dvc.fill_ci_buffer()
                self.scanners_dvc.fill_ai_buffer()

                self.data_dvc.purge()

                # recording
                await self.record_data(timed=False)

                # collect final AI/CI
                self.counter_dvc.fill_ci_buffer()
                self.scanners_dvc.fill_ai_buffer()

                self.plane_data.append(self.data_dvc.data)
                self.data_dvc.init_data()

            else:
                break

        # finished measurement
        if self.is_running:
            # prepare data
            self.plane_images_data = [
                self.prepare_image_data(plane_idx)
                for plane_idx in range(self.scan_params.n_planes)
            ]

            # save data
            self.save_data(self.prep_data_dict(), self.build_filename())
            self.keep_last_meas()

            # show middle plane
            mid_plane = int(len(self.scan_params.set_pnts_planes) // 2)
            self.plane_shown.set(mid_plane)
            self.plane_choice.set(mid_plane)
            self._app.gui.main.imp.disp_plane_img(mid_plane)

        # manually stopped
        else:
            pass

        # return to stand-by state
        await self.toggle_lasers(finish=True)
        self.return_to_regular_tasks()
        self._app.gui.main.imp.dvc_toggle("PXL_CLK")

        if self.is_running:  # if not manually stopped
            self._app.gui.main.imp.toggle_meas(self.type)


class SFCSSolutionMeasurement(Measurement):
    """Doc."""

    dur_mul_dict = {
        "seconds": 1,
        "minutes": 60,
        "hours": 3600,
    }

    def __init__(self, app, scan_params, **kwargs):
        super().__init__(
            app=app, type="SFCSSolution", scan_params=scan_params, **kwargs
        )
        self.scan_params.scan_plane = "XY"
        self.duration_multiplier = self.dur_mul_dict[self.duration_units]

        if self.scan_params.pattern == "static":
            self.scanning = False
        else:
            self.scanning = True

        self.cal = False

    def build_filename(self, file_no: int) -> str:

        if not self.file_template:
            # give a general template for a solution measuremnt
            self.file_template = "sol"

        if self.repeat is True:
            # repeated measurements are always file 0
            file_no = 0

        return f"{self.file_template}_{self.scan_type}_{self.laser_config}_{file_no}"

    def set_current_and_end_times(self) -> NoReturn:
        """
        Given a duration in seconds, returns a tuple (current_time, end_time)
        in datetime.time format, where end_time is current_time + duration_in_seconds.
        """

        duration_in_seconds = int(self.total_duration * self.duration_multiplier)
        curr_datetime = datetime.datetime.now()
        start_time = datetime.datetime.now().time()
        end_time = (
            curr_datetime + datetime.timedelta(seconds=duration_in_seconds)
        ).time()
        self.start_time_wdgt.set(start_time)
        self.end_time_wdgt.set(end_time)

    async def calibrate_num_files(self):
        """Doc."""

        saved_dur_mul = self.duration_multiplier
        self.duration = self.cal_duration
        self.duration_multiplier = 1  # in seconds
        self.start_time = time.perf_counter()
        self.time_passed = 0
        self.cal = True
        logging.info(f"Calibrating file intervals for {self.type} measurement")

        # reading
        await self.record_data(timed=True)

        self.cal = False
        bps = self.data_dvc.tot_bytes_read / self.time_passed
        try:
            self.save_intrvl = self.max_file_size * 10 ** 6 / bps / saved_dur_mul
        except ZeroDivisionError as exc:
            err_hndlr(exc, "calibrate_num_files()")
            return

        if self.save_intrvl > self.total_duration:
            self.save_intrvl = self.total_duration

        self.cal_save_intrvl_wdgt.set(self.save_intrvl)
        self.duration = self.save_intrvl
        self.duration_multiplier = saved_dur_mul

        # determining number of files
        num_files = int(math.ceil(self.total_duration / self.save_intrvl))
        self.total_files_wdgt.set(num_files)
        return num_files

    def setup_scan(self):
        """Doc."""

        # make sure divisablility, then sync pixel clock with AO sample clock
        if (self.pxl_clk_dvc.freq_MHz * 1e6) % self.scan_params.ao_samp_freq_Hz != 0:
            raise ValueError(
                f"Pixel clock ({self.pxl_clk_dvc.freq_MHz} Hz) and AO samplng ({self.scan_params.ao_samp_freq_Hz} Hz) frequencies aren't divisible."
            )
        else:
            self.pxl_clk_dvc.low_ticks = (
                int(
                    (self.pxl_clk_dvc.freq_MHz * 1e6) / self.scan_params.ao_samp_freq_Hz
                )
                - 2
            )
        # create ao_buffer
        self.ao_buffer, self.scan_params = ScanPatternAO(
            self.scan_params.pattern, self.scan_params, self.um_V_ratio
        ).calculate_ao()
        self.n_ao_samps = self.ao_buffer.shape[1]
        # TODO: why is the next line correct? explain and use a constant for 1.5E-7. ask Oleg
        self.ai_conv_rate = (
            self.scanners_dvc.ai_buffer.shape[0]
            * 2
            * (1 / (self.scan_params.dt - 1.5e-7))
        )

    def disp_ACF(self):
        """Doc."""

        #        def ACF(data):

        if self.repeat is True:
            # display ACF for alignments
            # TODO: (later perhaps for scans too)
            p = PhotonDataClass()
            p.DoConvertFPGAdataToPhotons(self.data_dvc.data)
            s = CorrFuncTDCclass()
            s.LaserFreq = self.tdc_dvc.laser_freq_MHz * 1e6
            s.data["Data"].append(p)
            s.DoCorrelateRegularData()
            s.DoAverageCorr(NoPlot=True)
            try:
                s.DoFit(NoPlot=True)
            except (RuntimeWarning, RuntimeError, OptimizeWarning) as exc:
                # fit failed
                err_hndlr(exc, "disp_ACF()", lvl="warning")
                self.fit_led.set(QIcon(icon_path.LED_RED))
                g0, tau = s.G0, 0.1
                self.g0_wdgt.set(s.G0)
                self.tau_wdgt.set(0)
                self.plot_wdgt.obj.plot(s.lag, s.AverageCF_CR, clear=True)
            else:
                # fit succeeded
                self.fit_led.set(QIcon(icon_path.LED_OFF))
                fit_params = s.FitParam["Diffusion3Dfit"]
                g0, tau, _ = fit_params["beta"]
                x, y = fit_params["x"], fit_params["y"]
                fit_func = getattr(FitTools, fit_params["FitFunc"])
                self.g0_wdgt.set(g0)
                self.tau_wdgt.set(tau * 1e3)
                self.plot_wdgt.obj.plot(x, y, clear=True)
                y_fit = fit_func(x, *fit_params["beta"])
                self.plot_wdgt.obj.plot(x, y_fit, pen="r")

            self.plot_wdgt.obj.plotItem.vb.setRange(
                xRange=(0, tau * 1.3), yRange=(-g0 * 0.1, g0 * 1.3)
            )

    def prep_data_dict(self) -> dict:
        """
        Prepare the full measurement data, in a way that
        matches current MATLAB analysis
        """

        def prep_ang_scan_sett_dict():
            return {
                "X": self.ao_buffer[0, :],
                "Y": self.ao_buffer[1, :],
                "ActualSpeed": self.scan_params.eff_speed_um_s,
                "ScanFreq": self.scan_params.scan_freq_Hz,
                "SampleFreq": self.scan_params.ao_samp_freq_Hz,
                "PointsPerLineTotal": self.scan_params.tot_ppl,
                "PointsPerLine": self.scan_params.ppl,
                "NofLines": self.scan_params.n_lines,
                "LineLength": self.scan_params.lin_len,
                "LengthTot": self.scan_params.tot_len,
                "LineLengthMax": self.scan_params.max_line_len_um,
                "LineShift": self.scan_params.line_shift_um,
                "AngleDegrees": self.scan_params.angle_deg,
                "LinFrac": self.scan_params.lin_frac,
                "PixClockFreq": self.pxl_clk_dvc.freq_MHz,  # already in full_data
                "LinearPart": self.scan_params.lin_part,
                "Xlim": self.scan_params.x_lim,
                "Ylim": self.scan_params.y_lim,
            }

        if self.scanning:
            full_data = {
                "Data": self.data_dvc.data,
                "DataVersion": self.tdc_dvc.data_vrsn,
                "FpgaFreq": self.tdc_dvc.fpga_freq_MHz,
                "PixelFreq": self.pxl_clk_dvc.freq_MHz,
                "LaserFreq": self.tdc_dvc.laser_freq_MHz,
                "Version": self.tdc_dvc.tdc_vrsn,
                "AI": self.scanners_dvc.ai_buffer,
                "AO": self.ao_buffer,
                "AvgCnt": self.counter_dvc.avg_cnt_rate,
            }
            if self.scan_params.pattern == "circle":
                full_data["CircleSpeed_um_sec"] = self.scan_params.speed_um_s
                full_data["AnglularScanSettings"] = []
            else:
                full_data["CircleSpeed_um_sec"] = 0
                full_data["AnglularScanSettings"] = prep_ang_scan_sett_dict()

            return {"FullData": full_data, "SystemInfo": self.sys_info}

        else:
            full_data = {
                "Data": self.data_dvc.data,
                "DataVersion": self.tdc_dvc.data_vrsn,
                "FpgaFreq": self.tdc_dvc.fpga_freq_MHz,
                "LaserFreq": self.tdc_dvc.laser_freq_MHz,
                "Version": self.tdc_dvc.tdc_vrsn,
                "AvgCnt": self.counter_dvc.avg_cnt_rate,
            }

            return {"FullData": full_data}

    async def run(self):
        """Doc."""

        # initialize gui start/end times
        self.set_current_and_end_times()

        # turn on lasers
        await self.toggle_lasers()

        # calibrating save-intervals/num_files
        # TODO: test if non-scanning calibration gives good enough approximation for file size, if not, consider scanning inside cal function
        if not self.repeat:
            num_files = await self.calibrate_num_files()
        else:
            # if alignment measurement
            num_files = 999
            self.duration = self.total_duration

        if self.scanning:
            self.setup_scan()
            self._app.gui.main.imp.dvc_toggle("PXL_CLK")

        self.data_dvc.init_data()
        self.data_dvc.purge()
        self.scanners_dvc.init_ai_buffer()
        self.counter_dvc.init_ci_buffer()

        if self.scanning:
            self.init_scan_tasks("CONTINUOUS")
            self.scanners_dvc.start_tasks("ao")

        # collect initial AI/CI to avoid possible overflow
        self.counter_dvc.fill_ci_buffer()
        self.scanners_dvc.fill_ai_buffer()

        self.total_time_passed = 0

        if self.is_running:
            logging.info(f"Running {self.type} measurement")

        for file_num in range(1, num_files + 1):

            if self.is_running:

                self.file_num_wdgt.set(file_num)

                self.start_time = time.perf_counter()
                self.time_passed = 0

                # reading
                await self.record_data(timed=True)

                self.total_time_passed += self.time_passed

                # collect final AI/CI
                self.counter_dvc.fill_ci_buffer()
                self.scanners_dvc.fill_ai_buffer()

                if self.scanning:
                    # save just one full pattern of AI
                    self.scanners_dvc.ai_buffer = self.scanners_dvc.ai_buffer[
                        :, : self.ao_buffer.shape[1]
                    ]

                # reset timers for alignment measurements
                if self.repeat:
                    self.total_time_passed = 0
                    self.set_current_and_end_times()

                # save and display data
                if self.is_running or self.repeat is False:
                    # if not manually stopped while aligning
                    self.save_data(self.prep_data_dict(), self.build_filename(file_num))
                    self.disp_ACF()

                # initialize data buffers for next file
                self.data_dvc.init_data()
                self.data_dvc.purge()
                self.scanners_dvc.init_ai_buffer()
                self.counter_dvc.init_ci_buffer()

            else:
                break

        await self.toggle_lasers(finish=True)

        if self.scanning:
            self.return_to_regular_tasks()
            self._app.gui.main.imp.dvc_toggle("PXL_CLK")

        if self.is_running:  # if not manually stopped
            self._app.gui.main.imp.toggle_meas(self.type)
