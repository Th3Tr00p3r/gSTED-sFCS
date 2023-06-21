"""Measurements."""

import asyncio
import datetime
import logging
import re
import sys
import time
from contextlib import suppress
from datetime import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import nidaqmx.constants as ni_consts
import numpy as np

from data_analysis.correlation_function import SolutionSFCSMeasurement
from logic.scan_patterns import ScanPatternAO, vector_snake_pattern
from utilities import errors, file_utilities, fit_tools, helper


class MeasurementProcedure:
    """Base class for measurement procedures"""

    scanning: bool

    def __init__(
        self,
        app,
        type: str,
        scan_params,
        laser_mode,
        file_template=None,
        save_path=None,
        sub_dir_name=None,
        prog_bar_wdgt=None,
        **kwargs,
    ):

        # devices
        self.counter_dvc = app.devices.photon_counter
        self.pxl_clk_dvc = app.devices.pixel_clock
        self.scanners_dvc = app.devices.scanners
        self.tdc_dvc = app.devices.TDC
        self.delayer_dvc = app.devices.delayer
        self.spad_dvc = app.devices.spad
        self.data_dvc = app.devices.UM232H
        self.laser_dvcs = SimpleNamespace(
            exc=app.devices.exc_laser,
            dep=app.devices.dep_laser,
            dep_shutter=app.devices.dep_shutter,
        )
        self.stage_dvc = app.devices.stage

        self._app = app
        self.type = type

        # keep the depletion state for setting up the measurement before running
        self.laser_mode = laser_mode
        if laser_mode in {"dep", "sted"}:
            self.dep_type = self.laser_dvcs.dep.mode
            if self.dep_type == "current":
                self.dep_value = app.gui.main.depCurr.value()
            else:  # power mode
                self.dep_value = app.gui.main.depPow.value()
            # keep laser power
            self.dep_power_mw = self.dep_value
        else:
            self.dep_power_mw = None
        # keep the detector/delayer state for setting up the measurement before running
        self.spad_settings = self.spad_dvc.settings

        self.file_template = file_template
        self.save_path = save_path
        self.sub_dir_name = sub_dir_name
        self.prog_bar_wdgt = prog_bar_wdgt
        self.icon_dict = app.icon_dict  # get icons
        self.scan_params = scan_params
        # TODO: check if 'ai_scaling_xyz' matches today's ratio
        self.sys_info = file_utilities.default_system_info
        self.sys_info["xyz_um_to_v"] = self.scanners_dvc.um_v_ratio
        self.sys_info["date"] = dt.now().strftime("%d-%m-%Y")

        # TODO: These are for mypy to be silent. Ultimately, I believe creating an ABC will
        # better suit this case. see this: https://github.com/python/mypy/issues/1996
        self.start_time: float
        self.duration_s: int
        self.max_file_size_mb: float
        self.est_plane_duration: float
        self.curr_plane: int
        self.final: bool
        self.ao_buffer: np.ndarray
        self.n_ao_samps: int
        self.ai_conv_rate: float

        self.is_running = False

    async def stop(self):
        """Doc."""

        # turn off devices and return to starting position if scanning
        with suppress(errors.DeviceError, MeasurementError):
            await self.toggle_lasers(finish=True)
            self._app.gui.main.impl.device_toggle("TDC", leave_off=True)

            if self.scanning:
                self.return_to_regular_tasks()
                self._app.gui.main.impl.device_toggle("pixel_clock", leave_off=True)

                if self.type == "SFCSSolution":
                    self._app.gui.main.impl.go_to_origin()
                    type_ = "solution"
                elif self.type == "SFCSImage":
                    self._app.gui.main.impl.move_scanners(
                        destination=self.scan_params["initial_ao"]
                    )
                    type_ = "image"

            # TODO: make this more readable - the idea is that static measurements are also of type_ "solution"
            # An option is to change the type of measurement from "SFCSSolution" to "solution" ("SFCSImage" -> "image")
            # or the other way around.
            elif self.type == "SFCSSolution":
                type_ = "solution"

        self.is_running = False
        await self._app.gui.main.impl.toggle_meas(self)
        with suppress(AttributeError):
            # AttributeError: no widget
            self.prog_bar_wdgt.set(0)
        self._app.gui.main.impl.populate_all_data_dates(type_)  # refresh saved measurements
        # define dummy None measreument for App object
        self._app.meas = SimpleNamespace(type=None, is_running=False)

    async def record_data(
        self, start_time: float, timed: bool = False, size_limited: bool = False
    ) -> None:
        """
        Turn ON the TDC (FPGA), read while conditions are met,
        turn OFF TDC and read leftover data.
        """

        # turn ON exc (if relevant) and TDC before reading FPGA (will toggle off both in the end)
        with self.laser_dvcs.exc.use(self.laser_mode in {"exc", "sted"}), self.tdc_dvc.use():

            # Solution - record data until time is up or ordered to stop
            if timed:
                dwell_start_s = time.perf_counter()
                while self.is_running:
                    await self.data_dvc.read_TDC()
                    self.time_passed_s = time.perf_counter() - self.start_time
                    # check if total time has passed
                    if self.time_passed_s >= self.duration_s:
                        break
                    # check if file size was reached
                    if size_limited and (
                        (self.data_dvc.tot_bytes_read / 1e6) >= self.max_file_size_mb
                    ):
                        break
                    # check if stage_dwelltime_s was reached
                    if (
                        stage_dwelltime_s := self.scan_params["stage_dwelltime_s"]
                    ) is not None and time.perf_counter() - dwell_start_s >= stage_dwelltime_s:
                        break

            # Image - record data until image scan task is done
            else:
                while self.is_running and not self.scanners_dvc.are_tasks_done("ao"):
                    await self.data_dvc.read_TDC()
                    self.time_passed_s = time.perf_counter() - self.start_time

            await self.data_dvc.read_TDC()  # read leftovers

    def _prep_meas_dict(self) -> dict:
        """Doc."""

        full_data = {
            "laser_mode": self.laser_mode,
            "dep_power_mw": self.dep_power_mw,
            "duration_s": self.duration_s if hasattr(self, "duration_s") else None,
            # TODO: prepare a function to cut img data into planes, similar to how the counts are cut
            "byte_data": np.asarray(self.data_dvc.data, dtype=np.uint8),
            "version": self.tdc_dvc.tdc_vrsn,
            "data_version": self.tdc_dvc.data_vrsn,
            "fpga_freq_mhz": self.tdc_dvc.fpga_freq_mhz,
            "laser_freq_mhz": self.tdc_dvc.laser_freq_mhz,
            "avg_cnt_rate_khz": self.counter_dvc.avg_cnt_rate_khz,
            "detector_settings": getattr(self.spad_dvc, "settings", {}),
        }
        full_data["delayer_settings"] = (
            self.delayer_dvc.settings if self.delayer_dvc.is_on else None
        )

        if self.scanning:
            full_data["pix_clk_freq_mhz"] = self.pxl_clk_dvc.freq_MHz
            full_data["scan_settings"] = dict(
                pattern=self.scan_params["pattern"],
                ai=np.array(self.scanners_dvc.ai_buffer, dtype=np.float32),
                ao=self.ao_buffer.T,
                speed_um_s=self.scan_params["eff_speed_um_s"]
                if hasattr(self, "duration_s")
                else None,
                ao_sampling_freq_hz=self.scan_params["ao_sampling_freq_hz"]
                if hasattr(self, "duration_s")
                else None,
            )
            if self.scan_params["pattern"] == "circle":
                full_data["scan_settings"].update(
                    diameter_um=self.scan_params["diameter_um"],
                    n_circles=self.scan_params["n_circles"],
                    circle_freq_hz=self.scan_params["circle_freq_hz"],
                )
            elif self.scan_params["pattern"] == "angular":
                full_data["scan_settings"].update(
                    line_freq_hz=self.scan_params["line_freq_hz"],
                    samples_per_line=self.scan_params["samples_per_line"],
                    ppl=self.scan_params["ppl"],
                    n_lines=self.scan_params["n_lines"],
                    linear_len_um=self.scan_params["linear_len_um"],
                    max_line_length_um=self.scan_params["max_line_len_um"],
                    line_shift_um=self.scan_params["line_shift_um"],
                    angle_degrees=self.scan_params["angle_deg"],
                    linear_frac=self.scan_params["linear_fraction"],
                    linear_part=self.scan_params["linear_part"],
                    x_lim=self.scan_params["x_lim"],
                    y_lim=self.scan_params["y_lim"],
                )
            elif self.scan_params["pattern"] == "image":
                full_data["scan_settings"].update(
                    ai=np.asarray(self.scanners_dvc.ai_buffer, dtype=np.float64),
                    set_pnts_lines_odd=self.scan_params["set_pnts_lines_odd"],
                    set_pnts_planes=self.scan_params["set_pnts_planes"],
                    dim1_um=self.scan_params["dim1_um"],
                    dim2_um=self.scan_params["dim2_um"],
                    dim3_um=self.scan_params["dim3_um"],
                    dim_order=self.scan_params["dim_order"],
                    n_lines=self.scan_params["n_lines"],
                    n_planes=self.scan_params["n_planes"],
                    line_freq_hz=self.scan_params["line_freq_hz"],
                    ppl=self.scan_params["ppl"],
                    plane_orientation=self.scan_params["plane_orientation"],
                    initial_ao=self.scan_params["initial_ao"],
                    what_stage="Galvanometers",
                    linear_frac=self.scan_params["linear_fraction"],
                )
                full_data["ci"] = np.asarray(self.counter_dvc.ci_buffer, dtype=np.int64)

        return {"full_data": full_data, "system_info": self.sys_info}

    def save_data(self, data_dict: dict, file_name: str) -> None:
        """
        Create a directory of today's date, and there
        save the raw data as a .npy file and compress the rest in a pickled .gzip file.
        """
        # TODO: does not handle overnight measurements during which the date changes.
        # To do that, one would need to keep the starting date and ensure overnight measurements
        # are still save in their original date folder

        today_dir = Path(self.save_path) / dt.now().strftime("%d_%m_%Y")

        if self.type == "SFCSSolution":
            if self.final:
                save_path = today_dir
            else:
                save_path = today_dir / "solution"
        elif self.type == "SFCSImage":
            save_path = today_dir / "image"
        else:
            # TODO: why do I get here sometimes? (related to y-galvo cal.?)
            raise NotImplementedError(f"Measurements of type '{self.type}' are not handled.")

        file_path = save_path / (re.sub("\\s", "_", file_name) + ".pkl")
        byte_data_path = file_path.with_name(file_path.name.replace(".pkl", "_byte_data.npy"))

        byte_data = data_dict["full_data"].pop("byte_data")
        Path.mkdir(byte_data_path.parent, parents=True, exist_ok=True)
        np.save(
            byte_data_path,
            byte_data,
            allow_pickle=False,
            fix_imports=False,
        )

        file_utilities.save_object(
            data_dict, file_path, compression_method="gzip", obj_name="raw data"
        )
        logging.debug(f"Saved measurement file: '{file_path}'.")

    async def toggle_lasers(self, finish=False) -> None:  # NOQA C901
        """Doc."""

        def current_emission_state() -> str:
            """Doc."""

            is_exc_on = self._app.devices.exc_laser.is_on
            try:
                is_dep_on = (
                    self._app.devices.dep_laser.is_emission_on
                    and self._app.devices.dep_shutter.is_on
                )
            except AttributeError:
                # dep error on startup, emission should be off
                is_dep_on = False

            if is_exc_on and is_dep_on:
                return "sted"
            elif is_exc_on:
                return "exc"
            elif is_dep_on:
                return "dep"
            else:
                return "nolaser"

        async def prep_depletion() -> None:
            """Doc."""
            # TODO: move this to the device itself - if turned on on power mode, it should prep itself.

            toggle_succeeded = self._app.gui.main.impl.device_toggle(
                "dep_laser", toggle_mthd="laser_toggle", state_attr="is_emission_on"
            )
            if toggle_succeeded:
                logging.info(
                    f"{self._app.devices.dep_laser.log_ref} isn't on. Turning on and waiting 5 s before measurement."
                )
                if self.type == "SFCSImage" and self.laser_mode == "dep":
                    button = self._app.gui.main.startImgScanDep
                elif self.type == "SFCSImage" and self.laser_mode == "sted":
                    button = self._app.gui.main.startImgScanSted
                elif self.type == "SFCSSolution" and self.laser_mode in {"dep", "sted"}:
                    button = self._app.gui.main.stopMeasurements
                button.setEnabled(False)
                await asyncio.sleep(5)
                button.setEnabled(True)

        # measurement ends
        if finish:
            # reset automatic shutdown for depletion laser
            if self.laser_mode in {"dep", "sted"}:
                self.laser_dvcs.dep.turn_on_time = time.perf_counter()
            # turn off lasers
            if self.laser_mode == "exc":
                self._app.gui.main.impl.device_toggle("exc_laser", leave_off=True)
            elif self.laser_mode == "dep":
                self._app.gui.main.impl.device_toggle("dep_shutter", leave_off=True)
            elif self.laser_mode == "sted":
                self._app.gui.main.impl.device_toggle("exc_laser", leave_off=True)
                self._app.gui.main.impl.device_toggle("dep_shutter", leave_off=True)

        # measurement begins
        else:
            # set dep laser power if necessary
            if self.laser_mode in {"dep", "sted"}:
                if self.dep_type == "current":
                    self.laser_dvcs.dep.set_current(self.dep_value)
                else:  # power mode
                    self.laser_dvcs.dep.set_power(self.dep_value)
            # turn on emissions
            if self.laser_mode == "exc":
                # turn excitation ON and depletion shutter OFF
                self._app.gui.main.impl.device_toggle("exc_laser", leave_on=True)
                self._app.gui.main.impl.device_toggle("dep_shutter", leave_off=True)
            elif self.laser_mode == "dep":
                await prep_depletion() if not self.laser_dvcs.dep.is_emission_on else None
                # turn depletion shutter ON and excitation OFF
                self._app.gui.main.impl.device_toggle("dep_shutter", leave_on=True)
                self._app.gui.main.impl.device_toggle("exc_laser", leave_off=True)
            elif self.laser_mode == "sted":
                await prep_depletion() if not self.laser_dvcs.dep.is_emission_on else None
                # turn both depletion shutter and excitation ON
                self._app.gui.main.impl.device_toggle("exc_laser", leave_on=True)
                self._app.gui.main.impl.device_toggle("dep_shutter", leave_on=True)
            elif self.laser_mode == "nolaser":
                # turn both excitation and depletion shutter OFF
                self._app.gui.main.impl.device_toggle("exc_laser", leave_off=True)
                self._app.gui.main.impl.device_toggle("dep_shutter", leave_off=True)

            if current_emission_state() != self.laser_mode:
                # cancel measurement if relevant lasers are not ON,
                raise MeasurementError(
                    f"Requested laser mode ({self.laser_mode}) was not attained."
                )

    def ready_detector(self) -> None:
        """
        Set the detector and pulse delayer to their selected values
        before starting the measurement.
        """

        # TODO: fix me!

    def init_scan_tasks(self, ao_sample_mode: str) -> None:
        """Doc."""

        # initialize write task

        # multiplication of the internal PC buffer size by this factor avoids buffer overflow
        buff_ovflw_const = 1.2  # could be redundant, but just to be safe...

        ao_sample_mode = getattr(ni_consts.AcquisitionType, ao_sample_mode)

        self.scanners_dvc.start_write_task(
            ao_data=self.ao_buffer,
            type=self.scan_params["plane_orientation"] if self.ao_buffer.ndim < 3 else "XYZ",
            samp_clk_cnfg_xy={
                "source": self.pxl_clk_dvc.out_term,
                "sample_mode": ao_sample_mode,
                "samps_per_chan": self.n_ao_samps,
                "rate": 100000,  # TODO: should be defined by some meaningful constant
            },
            samp_clk_cnfg_z={
                "source": self.pxl_clk_dvc.out_ext_term,
                "sample_mode": ao_sample_mode,
                "samps_per_chan": self.n_ao_samps,
                "rate": 100000,  # TODO: should be defined by some meaningful constant
            },
            start=False,
        )

        # initialize read tasks
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


class ImageMeasurementProcedure(MeasurementProcedure):
    """Doc."""

    def __init__(self, app, scan_params, **kwargs):
        super().__init__(app=app, type="SFCSImage", scan_params=scan_params, **kwargs)
        self.always_save = kwargs["always_save"]
        self.plane_shown = kwargs["plane_shown"]
        self.plane_choice = kwargs["plane_choice"]
        self.image_wdgt = kwargs["image_wdgt"]
        self.pattern_wdgt = kwargs["pattern_wdgt"]
        self.scale_image = kwargs["scale_image"]
        self.image_method = kwargs["image_method"]
        self.scan_type = "image"
        self.scanning = True
        self.scan_params["initial_ao"] = tuple(
            getattr(self._app.gui.main, f"{ax}AOVint").value() for ax in "xyz"
        )

    def build_filename(self) -> str:
        return f"{self.file_template}_{self.laser_mode}_{self.scan_params['plane_orientation']}{(f'_{self.dep_power_mw}mW' if self.dep_power_mw is not None else '')}_{dt.now().strftime('%H%M%S')}"

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
        self.scan_params["line_freq_hz"], clk_div = sync_line_freq(
            self.scan_params["line_freq_hz"],
            self.scan_params["ppl"],
            self.pxl_clk_dvc.freq_MHz * 1e6,
        )
        self.pxl_clk_dvc.low_ticks = clk_div - 2
        self.scan_params["ppl"] = fix_ppl(
            self.scanners_dvc.MIN_OUTPUT_RATE_Hz,
            self.scan_params["line_freq_hz"],
            self.scan_params["ppl"],
        )
        # create ao_buffer
        self.ao_buffer, self.scan_params = ScanPatternAO(
            "image",
            self.scanners_dvc.um_v_ratio,
            self.scan_params["initial_ao"],
            self.scan_params,
        ).calculate_pattern()
        self.n_ao_samps = self.ao_buffer.shape[1]
        # NOTE: why is the next line correct? explain and use a constant for 1.5E-7. ask Oleg
        self.ai_conv_rate = 6 * (1 / (self.scan_params["dt"] - 1.5e-7))
        self.est_total_duration_s = self.n_ao_samps * self.scan_params["dt"] / 2
        self.plane_choice.obj.setMaximum(len(self.scan_params["set_pnts_planes"]) - 1)

    async def run(self):
        """Doc."""

        self.setup_scan()

        try:
            await self.toggle_lasers()
            self.data_dvc.init_data()
            self.data_dvc.purge_buffers()
            self._app.gui.main.impl.device_toggle("pixel_clock", leave_on=True)
            self.scanners_dvc.init_ai_buffer(type="inf")
            self.counter_dvc.init_ci_buffer(type="inf")

        except (MeasurementError, errors.DeviceError) as exc:
            await self.stop()
            errors.err_hndlr(exc, sys._getframe(), locals())
            return

        else:
            n_planes = self.scan_params["n_planes"]
            self.plane_data = []
            self.start_time = time.perf_counter()
            self.time_passed_s = 0
            self.is_running = True
            logging.info(f"Running {self.type} measurement")

        # start all tasks (AO, AI, CI)
        try:
            self.init_scan_tasks("FINITE")
            self.scanners_dvc.start_tasks("ao")

            # record data during scan
            await self.record_data(self.start_time, timed=False)

            # collect final ai/CI
            self.counter_dvc.fill_ci_buffer()
            self.scanners_dvc.fill_ai_buffer()

        except MeasurementError as exc:
            await self.stop()
            errors.err_hndlr(exc, sys._getframe(), locals())
            return

        except Exception as exc:  # TESTESTEST
            print(f"ImageMeasurementProcedure: THIS SHOULD NOT HAPPEN! [{exc}]")
            errors.err_hndlr(exc, sys._getframe(), locals())

        # finished measurement
        if self.is_running:  # if not manually stopped
            # prepare data
            data_dict = self._prep_meas_dict()
            if self.always_save:  # save data
                self.save_data(data_dict, self.build_filename())
            # keep last measurement
            self._app.last_image_scans.appendleft(data_dict)
            # show middle plane
            mid_plane = int(len(self.scan_params["set_pnts_planes"]) / 2)
            self.plane_shown.set(mid_plane)
            self.plane_choice.set(mid_plane)
            should_display_autocross = self.scan_params["auto_cross"] and (self.laser_mode == "exc")
            self._app.gui.main.impl.disp_plane_img(auto_cross=should_display_autocross)

        if self.is_running:  # if not manually before completion
            await self.stop()


class SolutionMeasurementProcedure(MeasurementProcedure):
    """Doc."""

    dur_mul_dict = {
        "seconds": 1,
        "minutes": 60,
        "hours": 3600,
    }

    def __init__(self, app, scan_params, **kwargs):
        super().__init__(app=app, type="SFCSSolution", scan_params=scan_params, **kwargs)
        # TODO: would make more sense if these were in a specified dict rather than in the kwargs dict...
        self.scan_type = kwargs["scan_type"]
        self.repeat = kwargs["repeat"]
        self.final = kwargs["final"]
        self.should_disp_acf = kwargs.get("should_disp_acf", True)
        self.max_file_size_mb = kwargs.get("max_file_size_mb", 20)
        self.duration = kwargs["duration"]
        self.duration_units = kwargs["duration_units"]
        self.start_time_wdgt = kwargs.get("start_time_wdgt")
        self.end_time_wdgt = kwargs.get("end_time_wdgt")
        self.time_left_wdgt = kwargs.get("time_left_wdgt")
        self.file_num_wdgt = kwargs.get("file_num_wdgt")
        self.pattern_wdgt = kwargs.get("pattern_wdgt")
        self.g0_wdgt = kwargs.get("g0_wdgt")
        self.tau_wdgt = kwargs.get("tau_wdgt")
        self.plot_wdgt = kwargs.get("plot_wdgt")
        self.fit_led = kwargs.get("fit_led")
        self.should_accumulate_corrfuncs = kwargs.get("should_accumulate_corrfuncs")
        self.should_fit_acf = kwargs.get("should_fit")
        self.processing_options = kwargs.get("processing_options")

        if self.scan_params.get("floating_z_amplitude_um", 0) != 0:
            self.scan_params["plane_orientation"] = "XYZ"
        else:
            self.scan_params["plane_orientation"] = "XY"

        # stage pattern
        if self.scan_params["stage_pattern"] == "Snake":
            enc_dims = helper.get_encompassing_rectangle_dims(
                (self.scan_params["max_line_len_um"], self.scan_params["scan_width_um"]),
                self.scan_params["angle_deg"],
            )
            sample_size = self.stage_dvc.steps_per_sample_droplet / self.stage_dvc.steps_per_um
            self.stage_pattern = vector_snake_pattern(enc_dims, (sample_size, sample_size))
        else:
            self.stage_pattern = None

        self.duration_multiplier = self.dur_mul_dict[self.duration_units]
        self.duration_s = self.duration * self.duration_multiplier
        self.scanning = not (self.scan_params["pattern"] == "static")
        self._app.gui.main.impl.go_to_origin("XY")

    def build_filename(self, file_no: int) -> str:
        """Doc."""

        if self.final is True:
            return f"alignment_{self.scan_type}_{self.laser_mode}_0"

        else:
            if not self.file_template:
                self.file_template = "sol"
            return f"{self.file_template}_{self.scan_type}_{self.laser_mode}{(f'_{self.dep_power_mw}mW' if self.dep_power_mw is not None else '')}_{self.start_time_str}_{file_no}"

    def set_current_and_end_times(self) -> None:
        """Doc."""

        curr_datetime = dt.now()
        self.start_time_str = curr_datetime.strftime("%H%M%S")
        end_datetime = curr_datetime + datetime.timedelta(seconds=int(self.duration_s))
        with suppress(AttributeError):
            # AttributeError: no widgets
            self.start_time_wdgt.set(curr_datetime.time())
            self.end_time_wdgt.set(end_datetime.time())

    def setup_scan(self):
        """Doc."""

        # make sure divisablility, then sync pixel clock with ao sample clock
        if (self.pxl_clk_dvc.freq_MHz * 1e6) % self.scan_params["ao_sampling_freq_hz"] != 0:
            raise ValueError(
                f"Pixel clock ({self.pxl_clk_dvc.freq_MHz} Hz) and ao samplng ({self.scan_params['ao_sampling_freq_hz']} Hz) frequencies aren't divisible."
            )
        else:
            self.pxl_clk_dvc.low_ticks = (
                int((self.pxl_clk_dvc.freq_MHz * 1e6) / self.scan_params["ao_sampling_freq_hz"]) - 2
            )
        # create ao_buffer
        curr_ao_v = tuple(getattr(self._app.gui.main, f"{ax}AOVint").value() for ax in "xyz")
        self.ao_buffer, self.scan_params = ScanPatternAO(
            self.scan_params["pattern"],
            self.scanners_dvc.um_v_ratio,
            curr_ao_v,
            self.scan_params,
        ).calculate_pattern()
        self.n_ao_samps = self.ao_buffer.shape[1]
        # NOTE: why is the next line correct? explain and use a constant for 1.5E-7. ask Oleg
        self.ai_conv_rate = 6 * 2 * (1 / (self.scan_params["dt"] - 1.5e-7))

    def disp_ACF(self):
        """Doc."""

        def compute_acf(byte_data):
            """Doc."""

            meas_laser_type_dict = dict(
                exc="confocal", dep="depletion", sted="sted", nolaser="nolaser"
            )

            s = SolutionSFCSMeasurement(meas_laser_type_dict[self.laser_mode])
            # NOTE: using only up to first 10 Mb to avoid blocking measurements for too long
            p = s.process_data_file(
                file_dict=self._prep_meas_dict(),
                byte_data=byte_data,
                byte_data_slice=slice(0, int(10e6)),
                **self.processing_options,
            )
            s.data.append(p)

            # TESTESTEST - patch for background subtration - fix this
            if self.scan_type != "static":
                s.bg_line_corr_list = [
                    bg_line_corr
                    for bg_file_corr in [p.general.bg_line_corr for p in s.data]
                    for bg_line_corr in bg_file_corr
                ]
            # /TESTESTEST

            CF = s.correlate_and_average(
                cf_name=f"{self.laser_mode} alignment",
                **self.processing_options,
            )
            return CF

        try:
            CF = compute_acf(np.array(self.data_dvc.data, dtype=np.uint8))
            if self.should_accumulate_corrfuncs:
                try:
                    self.cf += CF
                except AttributeError:
                    self.cf = CF
            else:
                self.cf = CF
        except (RuntimeError, RuntimeWarning) as exc:
            # RuntimeWarning - some sort of zero-division in _calculate_weighted_avg
            # (due to invalid data during beam obstruction)
            # or in automatic ROI selection (same reason, for angular scans)
            # RuntimeError - detector disconected
            errors.err_hndlr(exc, sys._getframe(), locals())
        except Exception as exc:
            print(f"THIS SHOULD NOT HAPPEN, HANDLE THE EXCEPTION PROPERLY! [{exc}]")
            errors.err_hndlr(exc, sys._getframe(), locals())
        else:
            if self.should_fit_acf:
                try:
                    fp = self.cf.fit_correlation_function()
                except fit_tools.FitError as exc:
                    # fit failed
                    errors.err_hndlr(exc, sys._getframe(), locals(), lvl="debug")
                    self.fit_led.set(self.icon_dict["led_red"])
                    g0, tau = self.cf.g0, 0.1
                    self.g0_wdgt.set(g0)
                    self.tau_wdgt.set(0)
                    self.plot_wdgt.obj.plot_acfs((self.cf.lag, "lag"), self.cf.avg_cf_cr, g0)
                else:
                    # fit succeeded
                    self.fit_led.set(self.icon_dict["led_off"])
                    g0, tau = fp.beta["G0"], fp.beta["tau"]
                    fit_func = fp.fit_func
                    self.g0_wdgt.set(g0)
                    self.tau_wdgt.set(tau * 1e3)
                    self.plot_wdgt.obj.plot_acfs((self.cf.lag, "lag"), self.cf.avg_cf_cr, g0)
                    y_fit = fit_func(self.cf.lag, *fp.beta.values())
                    self.plot_wdgt.obj.plot(self.cf.lag, y_fit, "-.r")
                    logging.info(
                        f"Aligning ({self.laser_mode}): g0: {g0/1e3:.1f} K, tau: {tau*1e3:.1f} us."
                    )
            # don't fit
            else:
                g0, tau = self.cf.g0, 0.1
                self.g0_wdgt.set(g0)
                self.tau_wdgt.set(0)
                self.plot_wdgt.obj.plot_acfs((self.cf.lag, "lag"), self.cf.avg_cf_cr, g0)

    async def run(self, should_save=True):  # NOQA C901
        """Doc."""

        # initialize gui start/end times
        self.set_current_and_end_times()

        # turn on lasers
        try:
            await self.toggle_lasers()
        except (MeasurementError, errors.DeviceError) as exc:
            await self.stop()
            errors.err_hndlr(exc, sys._getframe(), locals())
            return

        try:
            if self.scanning:
                self.setup_scan()
                self._app.gui.main.impl.device_toggle("pixel_clock", leave_on=True)
                # make the circular ai buffer clip as long as the ao buffer
                self.scanners_dvc.init_ai_buffer(size=self.ao_buffer.shape[1])
                self.counter_dvc.init_ci_buffer(size=self.ao_buffer.shape[1])
            else:
                self.scanners_dvc.init_ai_buffer()

        except errors.DeviceError as exc:
            await self.stop()
            errors.err_hndlr(exc, sys._getframe(), locals())
            return

        else:
            # Move to first stage position (optional)
            if self.stage_pattern:
                await self.stage_dvc.move(next(self.stage_pattern), relative=False)
            self.is_running = True
            self.start_time = time.perf_counter()
            self.time_passed_s = 0
            file_num = 1
            logging.info(f"Running {self.type} measurement")

        try:
            while self.is_running and self.time_passed_s < self.duration_s:

                # initialize data buffer
                self.data_dvc.init_data()
                self.data_dvc.purge_buffers()

                if self.scanning:
                    self.counter_dvc.init_ci_buffer(size=self.ao_buffer.shape[1])
                    # re-start scan for each file

                    # if scanning measurement, ensure proper Y-galvo calibration during measurement and re-setup the scan if was recalibrated
                    # TODO: test this!
                    logging.info("Testing Y-galvo calibration before starting new file.")
                    if len(self.scanners_dvc.ai_buffer) and self.scanners_dvc.recalibrate_y_galvo(
                        self.scan_params,
                    ):
                        logging.info("Re-calibrating Y-galvo.")
                        self.setup_scan()

                    self.scanners_dvc.init_ai_buffer(size=self.ao_buffer.shape[1])
                    self.init_scan_tasks("CONTINUOUS")
                    self.scanners_dvc.start_tasks("ao")
                else:
                    self.scanners_dvc.init_ai_buffer()

                with suppress(AttributeError):
                    # AttributeError: no widget
                    self.file_num_wdgt.set(file_num)

                # FPGA Readout
                logging.debug("FPGA reading starts.")
                if self.repeat:
                    await self.record_data(self.start_time, timed=True)
                else:
                    await self.record_data(self.start_time, timed=True, size_limited=True)
                logging.debug("FPGA reading finished.")

                # collect final ai/CI
                self.counter_dvc.fill_ci_buffer()
                self.scanners_dvc.fill_ai_buffer()

                # Move stage (optional)
                if self.stage_pattern:
                    await self.stage_dvc.move(next(self.stage_pattern), relative=False)

                # show/add the ACF of the latest file in GUI (if not manually stopped)
                if self.should_disp_acf and self.is_running:
                    self.disp_ACF()

                # case aligning and not manually stopped
                if self.repeat and self.is_running:
                    # reset measurement
                    self.set_current_and_end_times()
                    self.start_time = time.perf_counter()
                    self.time_passed_s = 0

                # case final alignment and not manually stopped
                elif self.final and self.is_running:
                    if should_save:
                        self.save_data(self._prep_meas_dict(), self.build_filename(0))

                # case regular measurement and finished file or measurement
                elif not self.repeat:
                    if should_save:
                        self.save_data(self._prep_meas_dict(), self.build_filename(file_num))
                        if self.scanning:
                            self.scanners_dvc.init_ai_buffer(size=self.ao_buffer.shape[1])

                file_num += 1

        except errors.IOError as exc:
            errors.err_hndlr(exc, sys._getframe(), locals())

        except Exception as exc:  # TESTESTEST
            print(f"SolutionMeasurementProcedure: THIS SHOULD NOT HAPPEN! [{exc}]")
            errors.err_hndlr(exc, sys._getframe(), locals())

        if self.is_running:  # if not manually stopped
            if not should_save:
                # if not saving a file, keep the last measurement in memory
                self._app.last_meas_data = self._prep_meas_dict()
            await self.stop()

        # Move to stage origin (optional)
        if self.stage_pattern:
            await self.stage_dvc.move(helper.Vector(0, 0, "steps"))


class MeasurementError(Exception):
    pass
