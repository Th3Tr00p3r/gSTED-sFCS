# -*- coding: utf-8 -*-
"""Devices Module."""

import asyncio
import time
from array import array
from typing import NoReturn, Tuple

import nidaqmx.constants as ni_consts
import numpy as np

import logic.drivers as drivers
import utilities.constants as consts
import utilities.dialog as dialog
from utilities.errors import dvc_err_hndlr as err_hndlr
from utilities.helper import div_ceil, limit


class PixelClock(drivers.NIDAQmxInstrument):
    """
    The pixel clock is fed to the DAQ board from the FPGA.
    Base frequency is 4 MHz. Used for scans, where it is useful to
    have divisible frequencies for both the laser pulses and AI/AO.
    """

    def __init__(self, nick, param_dict):
        super().__init__(nick=nick, param_dict=param_dict)

    def toggle(self, bool):
        """Doc."""

        if bool:
            self._start_co_clock_sync()
        else:
            self.close_all_tasks()

    def _start_co_clock_sync(self) -> NoReturn:
        """Doc."""

        task_name = "Pixel Clock CO"

        self.close_tasks("co")

        self.create_co_task(
            name=task_name,
            chan_spec={
                "name_to_assign_to_channel": "pixel clock",
                "counter": self.cntr_addr,
                "source_terminal": self.tick_src,
                "low_ticks": self.low_ticks,
                "high_ticks": self.high_ticks,
            },
            clk_cnfg={"sample_mode": ni_consts.AcquisitionType.CONTINUOUS},
        )
        self.start_tasks("co")


class UM232H(drivers.FtdiInstrument):
    """
    Represents the FTDI chip used to transfer data from the FPGA
    to the PC.
    """

    def __init__(self, nick, param_dict, led_widget, switch_widget):
        self.led_widget = led_widget
        self.switch_widget = switch_widget
        super().__init__(nick=nick, param_dict=param_dict)
        self.init_data()
        self.toggle(True)

    def toggle(self, bool):
        """Doc."""

        if bool:
            self.open()
            self.purge()
        else:
            self.close()

    def stream_read_TDC(self, meas):
        """Doc."""

        while (
            meas.time_passed < meas.duration * meas.duration_multiplier
            and meas.is_running
        ):
            read_bytes = self.read()
            #            print(f"# Bytes read: {len(read_bytes)}")  # TEST
            self.data.extend(read_bytes)
            self.tot_bytes_read += len(read_bytes)

            meas.time_passed = time.perf_counter() - meas.start_time

    def init_data(self):
        """Doc."""

        self.data = array("B")
        self.tot_bytes_read = 0

    def reset(self):
        """Doc."""

        self.reset_dvc()
        self.close()
        time.sleep(0.3)
        self.open()


class Scanners(drivers.NIDAQmxInstrument):
    """
    Scanners encompasses all analog focal point positioning devices
    (X: x_galvo, Y: y_galvo, Z: z_piezo)
    """

    origin = (0.0, 0.0, 5.0)
    x_ao_limits = {"min_val": -5.0, "max_val": 5.0}
    y_ao_limits = {"min_val": -5.0, "max_val": 5.0}
    z_ao_limits = {"min_val": 0.0, "max_val": 10.0}

    def __init__(self, nick, param_dict, led_widget, switch_widget):
        self.led_widget = led_widget
        self.switch_widget = switch_widget
        super().__init__(
            nick=nick,
            param_dict=param_dict,
            ao_timeout=0.1,  # TODO: this should belong to the device and not the driver (as well as the param_dict)
        )  # , ai_buffer=np.zeros(shape=(3, 1000), dtype=np.double))

        self.ai_chan_specs = tuple(
            {
                "physical_channel": getattr(self, f"ai_{axis}_addr"),
                "name_to_assign_to_channel": f"{axis}-{inst} AI",
                "min_val": -10.0,
                "max_val": 10.0,
            }
            for axis, inst in zip("xyz", ("galvo", "galvo", "piezo"))
        )

        self.ao_int_chan_specs = tuple(
            {
                "physical_channel": getattr(self, f"ao_int_{axis}_addr"),
                "name_to_assign_to_channel": f"{axis}-{inst} internal AO",
                **getattr(self, f"{axis}_ao_limits"),
            }
            for axis, inst in zip("xyz", ("galvo", "galvo", "piezo"))
        )

        self.ao_chan_specs = tuple(
            {
                "physical_channel": getattr(self, f"ao_{axis}_addr"),
                "name_to_assign_to_channel": f"{axis}-{inst} AO",
                **getattr(self, f"{axis}_ao_limits"),
            }
            for axis, inst in zip("xyz", ("galvo", "galvo", "piezo"))
        )

        self.um_V_ratio = (self.x_um2V_const, self.y_um2V_const, self.z_um2V_const)

        self.toggle(True)

    def toggle(self, bool):
        """Doc."""

        if bool:
            self.start_continuous_read_task()
        else:
            self.close_all_tasks()

    def read_single_ao_internal(self) -> (float, float, float):
        """
        Return a single sample from each channel (x,y,z),
        for indicating the current AO position of the scanners.
        """

        def diff_to_rse(
            read_samples: [list, list, list, list, list]
        ) -> (float, float, float):
            """Doc."""

            rse_samples = []
            rse_samples.append((read_samples[0][0] - read_samples[1][0]) / 2)
            rse_samples.append((read_samples[2][0] - read_samples[3][0]) / 2)
            rse_samples.append(read_samples[4][0])
            return rse_samples

        task_name = "Single Sample AI"

        self.close_tasks("ai")

        self.create_ai_task(
            name=task_name,
            chan_specs=self.ao_int_chan_specs,
            samp_clk_cnfg={
                # TODO: decide if rate makes sense
                "rate": self.MIN_OUTPUT_RATE_Hz,
                "sample_mode": ni_consts.AcquisitionType.FINITE,
                "active_edge": ni_consts.Edge.RISING,
            },
        )
        self.start_tasks("ai")
        read_samples = self.analog_read(task_name, 1)
        return diff_to_rse(read_samples)

    def start_continuous_read_task(self) -> NoReturn:
        """Doc."""

        task_name = "Continuous AI"

        self.close_tasks("ai")

        self.create_ai_task(
            name=task_name,
            chan_specs=self.ai_chan_specs,
            samp_clk_cnfg={
                "rate": self.MIN_OUTPUT_RATE_Hz,
                "sample_mode": ni_consts.AcquisitionType.CONTINUOUS,
                "samps_per_chan": self.CONT_READ_BFFR_SZ,
                "active_edge": ni_consts.Edge.RISING,
            },
        )
        self.init_ai_buffer()
        self.start_tasks("ai")

    def start_scan_read_task(self, samp_clk_cnfg, ai_conv_rate) -> NoReturn:
        """Doc."""

        self.close_tasks("ai")

        self.create_ai_task(
            name="Image Scan AI",
            chan_specs=self.ai_chan_specs,
            samp_clk_cnfg=samp_clk_cnfg,
            clk_params={"ai_conv_rate": ai_conv_rate},
        )
        self.init_ai_buffer()
        self.start_tasks("ai")

    def create_scan_write_task(
        self,
        ao_data: Tuple[list],
        type: str,
        samp_clk_cnfg_xy: dict = {},
        samp_clk_cnfg_z: dict = {},
        scanning: bool = True,
    ) -> NoReturn:
        """Doc."""

        def diff_vltg_data(ao_data_row: list) -> [list, list]:
            return [ao_data_row, [(-1) * val for val in ao_data_row]]

        def smooth_start(
            axis: str, ao_chan_specs: dict, final_pos: float, step_sz: float = 0.25
        ) -> NoReturn:
            """Ask Oleg why we used 40 steps in LabVIEW (this is why I use a step size of 10/40 V)"""

            init_pos = limit(
                self.read_single_ao_internal()[consts.AX_IDX[axis]], 0.0, 10.0
            )

            total_dist = abs(final_pos - init_pos)
            n_steps = div_ceil(total_dist, step_sz)

            if n_steps < 2:
                return

            else:
                ao_data = np.linspace(init_pos, final_pos, n_steps)

                # move
                task_name = "Smooth AO"
                self.close_tasks("ao")
                self.create_ao_task(
                    name=task_name,
                    chan_specs=ao_chan_specs,
                    samp_clk_cnfg={
                        "rate": self.MIN_OUTPUT_RATE_Hz,  # WHY? see CreateAOTask.vi
                        "samps_per_chan": n_steps,
                        "sample_mode": ni_consts.AcquisitionType.FINITE,
                        "active_edge": ni_consts.Edge.RISING,
                    },
                )
                self.analog_write(task_name, ao_data, auto_start=False)
                self.start_tasks("ao")
                self.wait_for_task("ao", task_name)
                self.close_tasks("ao")

        axes_to_use = consts.AXES_TO_BOOL_TUPLE_DICT[type]

        xy_chan_spcs = []
        z_chan_spcs = []
        diff_ao_data_xy = []
        ao_data_z = []
        ao_data_row_idx = 0
        for ax, use_ax, ax_chn_spcs in zip("XYZ", axes_to_use, self.ao_chan_specs):
            if use_ax is True:
                if ax in "XY":
                    xy_chan_spcs.append(ax_chn_spcs)
                    diff_ao_data_xy += diff_vltg_data(ao_data[ao_data_row_idx])
                    ao_data_row_idx += 1
                else:  # "Z"
                    z_chan_spcs.append(ax_chn_spcs)
                    ao_data_z += ao_data[ao_data_row_idx]

        # start smooth
        if z_chan_spcs:
            smooth_start(axis="z", ao_chan_specs=z_chan_spcs, final_pos=ao_data_z[0])

        self.close_tasks("ao")

        if xy_chan_spcs:
            xy_task_name = "AO XY"
            self.create_ao_task(
                name=xy_task_name,
                chan_specs=xy_chan_spcs,
                samp_clk_cnfg=samp_clk_cnfg_xy,
            )
            self.analog_write(xy_task_name, diff_ao_data_xy)

        if z_chan_spcs:
            z_task_name = "AO Z"
            self.create_ao_task(
                name=z_task_name, chan_specs=z_chan_spcs, samp_clk_cnfg=samp_clk_cnfg_z
            )
            self.analog_write(z_task_name, ao_data_z)

        if not scanning:
            self.start_tasks("ao")

    def init_ai_buffer(self) -> NoReturn:
        """Doc."""

        self.ai_buffer = np.array([[-99], [-99], [-99]], dtype=np.float)

    def fill_ai_buff(
        self, task_name: str = "Continuous AI", n_samples=ni_consts.READ_ALL_AVAILABLE
    ) -> NoReturn:
        """Doc."""

        #        # TODO: stream reading currently not working for some reason - reading only one channel, the other two stay at zero
        #        num_samps_read = self.read()
        #        self.ai_buffer = np.concatenate(
        #        (self.ai_buffer, self.read_buffer[:, :num_samps_read]), axis=1
        #        )
        read_samples = self.analog_read("Continuous AI", n_samples)
        self.ai_buffer = np.concatenate((self.ai_buffer, read_samples), axis=1)

    def dump_ai_buff_overflow(self):
        """Doc."""

        ai_buffer_len = self.ai_buffer.shape[1]
        if ai_buffer_len > self.CONT_READ_BFFR_SZ:
            self.ai_buffer = self.ai_buffer[:, -self.CONT_READ_BFFR_SZ :]


class Counter(drivers.NIDAQmxInstrument):
    """
    Represents the detector which counts the green
    fluorescence photons coming from the sample.
    """

    # TODO: ADD CHECK FOR ERROR CAUSED BY INACTIVITY (SUCH AS WHEN DEBUGGING).

    updt_time = 0.2

    def __init__(self, nick, param_dict, led_widget, switch_widget, scanners_ai_tasks):
        self.led_widget = led_widget
        self.switch_widget = switch_widget
        super().__init__(nick=nick, param_dict=param_dict)
        self.counts = None  # this is for scans where the counts are actually used.
        self.last_avg_time = time.perf_counter()
        self.num_reads_since_avg = 0

        self.ai_cont_rate = scanners_ai_tasks["Continuous AI"].timing.samp_clk_rate
        self.ai_cont_src = scanners_ai_tasks["Continuous AI"].timing.samp_clk_term

        self.ci_chan_specs = {
            "name_to_assign_to_channel": "photon counter",
            "counter": self.address,
            "edge": ni_consts.Edge.RISING,
            "initial_count": 0,
            "count_direction": ni_consts.CountDirection.COUNT_UP,
        }

        self.toggle(True)

    def toggle(self, bool):
        """Doc."""

        if bool:
            self.start_continuous_read_task()
        else:
            self.close_all_tasks()

    def start_continuous_read_task(self) -> NoReturn:
        """Doc."""

        task_name = "Continuous CI"

        self.close_tasks("ci")

        self.create_ci_task(
            name=task_name,
            chan_spec=self.ci_chan_specs,
            chan_xtra_params={"ci_count_edges_term": self.CI_cnt_edges_term},
            samp_clk_cnfg={
                "rate": self.ai_cont_rate,
                "source": self.ai_cont_src,
                "sample_mode": ni_consts.AcquisitionType.CONTINUOUS,
                "samps_per_chan": self.CONT_READ_BFFR_SZ,
                "active_edge": ni_consts.Edge.RISING,
            },
        )
        self.init_ci_buffer()
        self.start_tasks("ci")

    def start_scan_read_task(self, samp_clk_cnfg) -> NoReturn:
        """Doc."""

        self.close_tasks("ci")

        self.start_ci_task(
            name="Image Scan CI",
            chan_specs=self.ci_chan_specs,
            chan_xtra_params={
                "ci_count_edges_term": self.CI_cnt_edges_term,
                "ci_data_xfer_mech": ni_consts.DataTransferActiveTransferMode.DMA,
            },
            samp_clk_cnfg=samp_clk_cnfg,
        )
        self.init_ci_buffer()
        self.start_tasks("ci")

    def count(self):
        """Doc."""

        num_samps_read = self.counter_stream_read()
        self.ci_buffer = np.append(
            self.ci_buffer, self.cont_read_buffer[:num_samps_read]
        )
        self.num_reads_since_avg += num_samps_read

    def average_counts(self):
        """Doc."""

        actual_intrvl = time.perf_counter() - self.last_avg_time
        start_idx = len(self.ci_buffer) - self.num_reads_since_avg

        if start_idx > 0:
            avg_cnt_rate = (
                self.ci_buffer[-1] - self.ci_buffer[-(self.num_reads_since_avg + 1)]
            ) / actual_intrvl
            avg_cnt_rate = avg_cnt_rate / 1000  # Hz -> KHz
        else:
            avg_cnt_rate = 0

        self.num_reads_since_avg = 0
        self.last_avg_time = time.perf_counter()
        return avg_cnt_rate

    def init_ci_buffer(self) -> NoReturn:
        """Doc."""

        self.ci_buffer = np.empty(shape=(0,))

    def dump_ci_buff_overflow(self):
        """Doc."""

        if len(self.ci_buffer) > self.CONT_READ_BFFR_SZ:
            self.ci_buffer = self.ci_buffer[-self.CONT_READ_BFFR_SZ :]


class Camera(drivers.UC480Instrument):
    """Doc."""

    vid_intrvl = 0.3

    def __init__(self, nick, param_dict, led_widget, switch_widget, loop, gui):
        self.led_widget = led_widget
        self.switch_widget = switch_widget
        super().__init__(nick=nick, param_dict=param_dict)
        self._loop = loop
        self._gui = gui
        self.state = False
        self.vid_state = False

    def toggle(self, bool):
        """Doc."""

        if bool:
            self.init_cam()
        else:
            self.close_cam()
        self.state = bool

    async def shoot(self):
        """Doc."""

        if self.vid_state is False:
            img = self.grab_image()
            self._imshow(img)
        else:
            self.toggle_video(False)
            await asyncio.sleep(0.2)
            img = self.grab_image()
            self._imshow(img)
            self.toggle_video(True)

    def toggle_video(self, new_state):
        """Doc."""

        async def helper(self):
            """
            This is a workaround -
            asyncio.to_thread() must be awaited, but toggle_video needs to be
            a regular function to keep the rest of the code as it is. by creating this
            async helper function I can make it work. A lambda would be better here
            but there's no async lambda yet.
            """

            await asyncio.to_thread(self._vidshow)

        if self.vid_state != new_state:
            self.toggle_vid(new_state)
            self.vid_state = new_state

        if new_state is True:
            self._loop.create_task(helper(self))

    def _vidshow(self):
        """Doc."""

        while self.vid_state is True:
            img = self.get_latest_frame()
            if img is not None:
                self._imshow(img)
            time.sleep(self.vid_intrvl)

    @err_hndlr
    def _imshow(self, img):
        """Plot image"""

        self._gui.figure.clear()
        ax = self._gui.figure.add_subplot(111)
        ax.imshow(img)
        self._gui.canvas.draw()


class SimpleDO(drivers.NIDAQmxInstrument):
    """ON/OFF device (excitation laser, depletion shutter, TDC)."""

    def __init__(self, nick, param_dict, led_widget, switch_widget):
        self.led_widget = led_widget
        self.switch_widget = switch_widget
        super().__init__(nick=nick, param_dict=param_dict)

        self.toggle(False)

    def toggle(self, bool):
        """Doc."""

        self.digital_write(bool)


class DepletionLaser(drivers.VisaInstrument):
    """Control depletion laser through pyVISA"""

    min_SHG_temp = 52

    def __init__(self, nick, param_dict, led_widget, switch_widget):
        self.led_widget = led_widget
        self.switch_widget = switch_widget
        super().__init__(
            nick=nick,
            param_dict=param_dict,
            read_termination="\r",
            write_termination="\r",
        )
        self.updt_time = 1
        self.state = None

        self.toggle(False)

        if self.state is False:
            self.set_current(1500)

    def toggle(self, bool):
        """Doc."""

        self._write(f"setLDenable {int(bool)}")

    def get_prop(self, prop):
        """Doc."""

        prop_dict = {
            "temp": "SHGtemp",
            "curr": "LDcurrent 1",
            "pow": "Power 0",
        }

        return self._query(prop_dict[prop])

    def set_power(self, value):
        """Doc."""

        # check that current value is within range
        if (value <= 1000) and (value >= 99):
            # change the mode to current
            self._write("Powerenable 1")
            # then set the power
            self._write("Setpower 0 " + str(value))
        else:
            dialog.Error(error_txt="Power out of range").display()

    def set_current(self, value):
        """Doc."""

        # check that current value is within range
        if (value <= 2500) and (value >= 1500):
            # change the mode to current
            self._write("Powerenable 0")
            # then set the current
            self._write("setLDcur 1 " + str(value))
        else:
            dialog.Error(error_txt="Current out of range").display()


class StepperStage:
    """
    Control stepper stage through Arduino chip using PyVISA.
    This device operates slowly and needs special care,
    and so its driver is within its own class (not inherited)
    """

    def __init__(self, nick, param_dict, led_widget, switch_widget):
        self.led_widget = led_widget
        self.switch_widget = switch_widget
        self.nick = nick
        [setattr(self, key, val) for key, val in param_dict.items()]
        self.error_dict = None
        self.rm = drivers.visa.ResourceManager()

        self.toggle(True)
        self.toggle(False)

    @err_hndlr
    def toggle(self, bool):
        """Doc."""

        if bool:
            self.rsrc = self.rm.open_resource(self.address)
        else:
            if hasattr(self, "rsrc"):
                self.rsrc.close()
        self.state = bool

    @err_hndlr
    def move(self, dir=None, steps=None):
        """Doc."""

        cmd_dict = {
            "UP": (lambda steps: "my " + str(-steps)),
            "DOWN": (lambda steps: "my " + str(steps)),
            "LEFT": (lambda steps: "mx " + str(steps)),
            "RIGHT": (lambda steps: "mx " + str(-steps)),
        }
        self.rsrc.write(cmd_dict[dir](steps))

    @err_hndlr
    def release(self):
        """Doc."""

        self.rsrc.write("ryx ")
