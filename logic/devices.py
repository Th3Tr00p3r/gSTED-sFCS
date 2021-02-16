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
            self.close_task("out")

    def _start_co_clock_sync(self) -> NoReturn:
        """Doc."""

        self.close_tasks("out")

        self.create_co_task(
            name="Pixel Clock CI",
            chan_spec={
                "name_to_assign_to_channel": "pixel clock",
                "counter": self.cntr_addr,
                "source_terminal": self.tick_src,
                "low_ticks": self.low_ticks,
                "high_ticks": self.high_ticks,
            },
            clk_cnfg={"sample_mode": ni_consts.AcquisitionType.CONTINUOUS},
        )
        self.start_tasks("out")


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

        # TODO: I accidently created a generator here where I was going for a tuple of dicts (should have used tuple()
        # if I really wanted that). This is why I had to use it as a method/function and not as an attribute - the iterator
        # is out of items after one go and leaves the attribute depleted. Perhaps I could use generators in other places?
        self.ai_chan_specs = tuple(
            {
                "physical_channel": getattr(self, f"ai_{axis}_addr"),
                "name_to_assign_to_channel": f"{axis}-{inst} ai",
                "min_val": -10.0,
                "max_val": 10.0,
            }
            for axis, inst in zip("xyz", ("galvo", "galvo", "piezo"))
        )

        self.ao_chan_specs = tuple(
            {
                "physical_channel": getattr(self, f"ao_{axis}_addr"),
                "name_to_assign_to_channel": f"{axis}-{inst} ao",
                **getattr(self, f"{axis}_ao_limits"),
            }
            for axis, inst in zip("xyz", ("galvo", "galvo", "piezo"))
        )

        self.um_V_ratio = (self.x_conv_const, self.y_conv_const, self.z_conv_const)

        self.toggle(True)

    def toggle(self, bool):
        """Doc."""

        if bool:
            self.start_continuous_read_task()
        else:
            self.close_tasks("in")

    def start_continuous_read_task(self) -> NoReturn:
        """Doc."""

        self.close_tasks("in")

        self.create_ai_task(
            name="Continuous AI",
            chan_specs=self.ai_chan_specs,
            samp_clk_cnfg={
                "rate": self.MIN_OUTPUT_RATE_Hz,
                "sample_mode": ni_consts.AcquisitionType.CONTINUOUS,
                "samps_per_chan": self.CONT_READ_BFFR_SZ,
                "active_edge": ni_consts.Edge.RISING,
            },
        )
        self.init_buffers()
        self.start_tasks("in")

    def start_scan_read_task(
        self, samps_per_chan, scn_clk_src, ai_conv_rate
    ) -> NoReturn:
        """Doc."""

        self.close_tasks("in")

        self.create_ai_task(
            name="Image Scan AI",
            chan_specs=self.ai_chan_specs,
            samp_clk_cnfg={
                "rate": self.MIN_OUTPUT_RATE_Hz * 100,  # WHY? see CreateAOTask.vi
                "source": scn_clk_src,
                "samps_per_chan": samps_per_chan,
                "sample_mode": ni_consts.AcquisitionType.CONTINUOUS,
                "active_edge": ni_consts.Edge.RISING,
            },
            clk_params={"ai_conv_rate": ai_conv_rate},
        )
        self.init_buffers()
        self.start_tasks("in")

    def create_scan_write_task(
        self,
        data: Tuple[list],
        type: str,
        samp_clk_cnfg: dict = {},
        scanning: bool = True,
    ) -> NoReturn:
        """Doc."""

        def diff_vltg_data(data_row: list) -> [list, list]:
            return [data_row, [(-1) * val for val in data_row]]

        def smooth_start(
            chan_specs: dict, final_pos: float, n_steps: int = 40
        ) -> NoReturn:
            """Doc."""

            # read init_pos
            *_, ai_chan_specs_z = self.ai_chan_specs
            self.close_tasks("in")
            self.create_ai_task(
                name="Single Sample AI",
                chan_specs=ai_chan_specs_z,
                samp_clk_cnfg={
                    # TODO: decide if rate makes sense
                    "rate": self.MIN_OUTPUT_RATE_Hz,
                    "samps_per_chan": 1,
                    "sample_mode": ni_consts.AcquisitionType.FINITE,
                    "active_edge": ni_consts.Edge.RISING,
                },
            )
            self.init_buffers()
            self.start_tasks("in")
            self.fill_ai_buff()
            #            init_pos = self.ai_buffer

            # create smooth data from init_pos to final_pos

            # move
            task = self.create_ao_task(name="Smooth AO", chan_specs=chan_specs)
            self.analog_write(task, data_z)

        self.close_tasks("out")

        if scanning:
            # TODO: why 100 KHz? see CreateAOTask.vi, ask Oleg
            # TODO: fix how arguments get here from measurements.py
            samp_clk_cnfg: dict = {
                **samp_clk_cnfg,
                "rate": 100000,
                "active_edge": ni_consts.Edge.RISING,
            }

        axes_to_use = consts.AXES_TO_BOOL_TUPLE_DICT[type]

        xy_chan_spcs = []
        z_chan_spcs = []
        diff_data_xy = []
        data_z = []
        data_row_idx = 0
        for ax, use_ax, ax_chn_spcs in zip("XYZ", axes_to_use, self.ao_chan_specs):
            if use_ax is True:
                if ax in "XY":
                    xy_chan_spcs.append(ax_chn_spcs)
                    diff_data_xy += diff_vltg_data(data[data_row_idx])
                    data_row_idx += 1
                else:  # "Z"
                    z_chan_spcs.append(ax_chn_spcs)
                    data_z += data[data_row_idx]

        if xy_chan_spcs:
            task = self.create_ao_task(
                name="AO XY",
                chan_specs=xy_chan_spcs,
                samp_clk_cnfg=samp_clk_cnfg,
            )
            self.analog_write(task, diff_data_xy)

        if z_chan_spcs:
            smooth_start(data_z)
            task = self.create_ao_task(
                name="AO Z", chan_specs=z_chan_spcs, samp_clk_cnfg=samp_clk_cnfg
            )
            self.analog_write(task, data_z)

        self.init_buffers()

        if not scanning:
            self.start_tasks("out")

    def init_buffers(self) -> NoReturn:
        """Doc."""

        self.ao_buffer = np.empty(shape=(3, 0), dtype=np.float)
        ai_task, *_ = self.in_tasks
        num_ai_chans = ai_task.number_of_channels
        self.ai_buffer = np.empty(shape=(num_ai_chans, 0), dtype=np.float)

    def fill_ai_buff(self) -> NoReturn:
        """Doc."""

        #        # TODO: stream reading currently not working for some reason - reading only one channel, the other two stay at zero
        #        num_samps_read = self.read()
        #        self.ai_buffer = np.concatenate(
        #        (self.ai_buffer, self.read_buffer[:, :num_samps_read]), axis=1
        #        )

        self.ai_buffer = np.concatenate((self.ai_buffer, self.analog_read()), axis=1)

    def dump_buff_overflow(self):
        """Doc."""

        _, ao_buffer_len = self.ao_buffer.shape
        _, ai_buffer_len = self.ai_buffer.shape
        if ao_buffer_len > self.CONT_READ_BFFR_SZ:
            self.ao_buffer = self.ao_buffer[:, -self.CONT_READ_BFFR_SZ :]
        if ai_buffer_len > self.CONT_READ_BFFR_SZ:
            self.ai_buffer = self.ai_buffer[:, -self.CONT_READ_BFFR_SZ :]


class Counter(drivers.NIDAQmxInstrument):
    """
    Represents the detector which counts the green
    fluorescence photons coming from the sample.
    """

    # TODO: ADD CHECK FOR ERROR CAUSED BY INACTIVITY (SUCH AS WHEN DEBUGGING).

    updt_time = 0.2

    def __init__(self, nick, param_dict, led_widget, switch_widget, scanners_in_tasks):
        self.led_widget = led_widget
        self.switch_widget = switch_widget
        super().__init__(nick=nick, param_dict=param_dict)
        self.counts = None  # this is for scans where the counts are actually used.
        self.last_avg_time = time.perf_counter()
        self.num_reads_since_avg = 0
        self.ai_task, *_ = scanners_in_tasks
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
            self.start_ci_continuous()
        else:
            self.close_tasks("in")

    def start_ci_continuous(self) -> NoReturn:
        """Doc."""

        self.close_tasks("in")

        self.create_ci_task(
            name="Continuous CI",
            chan_spec=self.ci_chan_specs,
            chan_xtra_params={"ci_count_edges_term": self.CI_cnt_edges_term},
            samp_clk_cnfg={
                "rate": self.ai_task.timing.samp_clk_rate,
                "source": self.ai_task.timing.samp_clk_term,
                "sample_mode": ni_consts.AcquisitionType.CONTINUOUS,
                "samps_per_chan": self.CONT_READ_BFFR_SZ,
                "active_edge": ni_consts.Edge.RISING,
            },
        )
        self.init_buffer()
        self.start_tasks("in")

    def start_ci_scan(self, samps_per_chan, scn_clk_src) -> NoReturn:
        """Doc."""

        self.close_tasks("in")

        self.start_ci_task(
            name="Image Scan CI",
            chan_specs=self.ci_chan_specs,
            chan_xtra_params={
                "ci_count_edges_term": self.CI_cnt_edges_term,
                "ci_data_xfer_mech": ni_consts.DataTransferActiveTransferMode.DMA,
            },
            samp_clk_cnfg={
                "rate": self.MIN_OUTPUT_RATE_Hz * 100,  # WHY? see CreateAOTask.vi
                "source": scn_clk_src,
                "samps_per_chan": samps_per_chan,
                "sample_mode": ni_consts.AcquisitionType.CONTINUOUS,
                "active_edge": ni_consts.Edge.RISING,
            },
        )
        self.init_buffer()
        self.start_tasks("in")

    def init_buffer(self) -> NoReturn:
        """Doc."""

        self.ci_buffer = np.empty(shape=(0,))

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

    def dump_buff_overflow(self):
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
