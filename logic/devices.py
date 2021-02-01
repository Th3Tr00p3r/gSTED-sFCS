# -*- coding: utf-8 -*-
"""Devices Module."""

import asyncio
import time
from array import array
from typing import NoReturn

import nidaqmx.constants as NI_CONSTS
import numpy as np

import logic.drivers as drivers
import utilities.dialog as dialog
from utilities.errors import dvc_err_hndlr as err_hndlr


class PixelClock(drivers.NIDAQmxInstrument):
    """Doc."""

    def __init__(self, nick, param_dict):
        super().__init__(nick=nick, param_dict=param_dict)

    def toggle(self, bool):
        """Doc."""

        if bool:
            if hasattr(self, "task"):
                self.close_task()
            self.start_co_task(
                type="Scan",
                chan_spec={
                    "name_to_assign_to_channel": "pixel clock",
                    "counter": self.cntr_addr,
                    "source_terminal": self.tick_src,
                    "low_ticks": self.low_ticks,
                    "high_ticks": self.high_ticks,
                },
                clk_cnfg={"sample_mode": NI_CONSTS.AcquisitionType.CONTINUOUS},
            )
        else:
            self.close_task()


class UM232H(drivers.FtdiInstrument):
    """Doc."""

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

    x_limits = {"min_val": -5.0, "max_val": 5.0}
    y_limits = {"min_val": -5.0, "max_val": 5.0}
    z_limits = {"min_val": 0.0, "max_val": 10.0}

    def __init__(self, nick, param_dict, led_widget, switch_widget):
        self.led_widget = led_widget
        self.switch_widget = switch_widget
        super().__init__(
            nick=nick,
            param_dict=param_dict,
            ao_timeout=0.1,  # TODO: these kwargs should belong to the device and not the driver (as well as the param_dict)
            ai_clk_rate=1000,
            ai_buff_sz=1000,
        )  # , ai_buffer=np.zeros(shape=(3, 1000), dtype=np.double))

        self.last_ai = None
        self.last_ao = (self.ao_x_init_vltg, self.ao_y_init_vltg, self.ao_z_init_vltg)
        self.um_V_ratio = (self.x_conv_const, self.y_conv_const, self.z_conv_const)

        # TODO: these buffers will be used for scans so the shape of the scan could be used/reviewed and compared for AO vs. AI
        self.ao_buffer = np.empty(shape=(3, 0), dtype=np.float)
        self.ai_buffer = np.empty(shape=(3, 0), dtype=np.float)

        self.toggle(True)

    def toggle(self, bool):
        """Doc."""

        if bool:
            if hasattr(self, "task"):
                self.close_task()
            self.start_ai_task(
                type="Continuous",
                chan_specs=[
                    {
                        "physical_channel": self.ai_x_addr,
                        "name_to_assign_to_channel": "x-galvo ai",
                        **self.x_limits,
                    },
                    {
                        "physical_channel": self.ai_y_addr,
                        "name_to_assign_to_channel": "y-galvo ai",
                        **self.y_limits,
                    },
                    {
                        "physical_channel": self.ai_z_addr,
                        "name_to_assign_to_channel": "z-piezo ai",
                        **self.z_limits,
                    },
                ],
                clk_cnfg={
                    "rate": self.ai_clk_rate,
                    "sample_mode": NI_CONSTS.AcquisitionType.CONTINUOUS,
                    "samps_per_chan": self.ai_buff_sz,
                },
            )
        else:
            self.close_task()

    def fill_ai_buff(self) -> NoReturn:
        """Doc."""

        #        # TODO: stream reading currently not working for some reason - reading only one channel, the other two stay at zero
        #        num_samps_read = self.read()
        #        self.ai_buffer = np.concatenate(
        #        (self.ai_buffer, self.read_buffer[:, :num_samps_read]), axis=1
        #        )

        self.ai_buffer = np.concatenate((self.ai_buffer, self.analog_read()), axis=1)

    async def move_to_pos(self, pos_vltgs: iter):
        """
        Finds out which AO voltages need to be changed,
        writes those voltages to the relevant scanners with
        the relevant limits, and saves the changed AO voltages.

        """

        ao_addresses = []
        limits = []
        vltgs = []
        axes = ("x", "y", "z")
        for axis, new_axis_vltg in zip(axes, pos_vltgs):
            if axis in {"x", "y"}:  # Differential Voltage [-5,5]
                ao_addresses.append(getattr(self, f"ao_{axis}_addr"))
                limits.append(getattr(self, f"{axis}_limits"))
                vltgs += [new_axis_vltg, -new_axis_vltg]
            else:  # RSE Voltage [0,10]
                ao_addresses.append(getattr(self, f"ao_{axis}_addr"))
                limits.append(getattr(self, f"{axis}_limits"))
                vltgs.append(new_axis_vltg)

        self.last_ao = pos_vltgs

        *ao_xy_addresses, ao_z_addr = ao_addresses
        *xy_vltgs, z_vltg = vltgs
        *xy_limits, z_limits = limits

        await self.analog_write(ao_xy_addresses, xy_vltgs, xy_limits)
        await self.analog_write([ao_z_addr], [z_vltg], [z_limits])

    def dump_buff_overflow(self):
        """Doc."""

        if max(self.ao_buffer.shape) > self.ai_buff_sz:
            self.ao_buffer = self.ao_buffer[:, -self.ai_buff_sz :]
        if max(self.ai_buffer.shape) > self.ai_buff_sz:
            self.ai_buffer = self.ai_buffer[:, -self.ai_buff_sz :]


class Counter(drivers.NIDAQmxInstrument):
    """Doc."""

    # TODO: ADD CHECK FOR ERROR CAUSED BY INACTIVITY (SUCH AS WHEN DEBUGGING).

    updt_time = 0.2
    ci_buff_sz = 10000
    ci_buffer = np.zeros(shape=(10000,), dtype=np.uint32)

    def __init__(self, nick, param_dict, led_widget, switch_widget, ai_task):
        self.led_widget = led_widget
        self.switch_widget = switch_widget
        super().__init__(nick=nick, param_dict=param_dict, ci_buffer=self.ci_buffer)

        self.cont_count_buff = np.empty(shape=(0,))
        self.counts = None  # this is for scans where the counts are actually used.
        self.last_avg_time = time.perf_counter()
        self.num_reads_since_avg = 0
        self.ai_task = ai_task

        self.toggle(True)

    def toggle(self, bool):
        """Doc."""

        if bool:
            if hasattr(self, "task"):
                self.close_task()
            self.start_ci_task(
                type="Continuous",
                chan_spec={
                    "name_to_assign_to_channel": "photon counter",
                    "counter": self.address,
                    "edge": NI_CONSTS.Edge.RISING,
                    "initial_count": 0,
                    "count_direction": NI_CONSTS.CountDirection.COUNT_UP,
                },
                ci_count_edges_term=self.CI_cnt_edges_term,
                clk_cnfg={
                    "rate": self.ai_task.timing.samp_clk_rate,
                    "source": self.ai_task.timing.samp_clk_term,
                    "sample_mode": NI_CONSTS.AcquisitionType.CONTINUOUS,
                    "samps_per_chan": len(self.ci_buffer),
                },
            )
        else:
            self.close_task()

    def count(self):
        """Doc."""

        num_samps_read = self.counter_read()
        self.cont_count_buff = np.append(
            self.cont_count_buff, self.ci_buffer[:num_samps_read]
        )
        self.num_reads_since_avg += num_samps_read

    def average_counts(self):
        """Doc."""

        actual_intrvl = time.perf_counter() - self.last_avg_time
        start_idx = len(self.cont_count_buff) - self.num_reads_since_avg

        if start_idx > 0:
            avg_cnt_rate = (
                self.cont_count_buff[-1]
                - self.cont_count_buff[-(self.num_reads_since_avg + 1)]
            ) / actual_intrvl

            self.num_reads_since_avg = 0
            self.last_avg_time = time.perf_counter()

            return avg_cnt_rate / 1000  # Hz -> KHz

        else:
            self.num_reads_since_avg = 0
            self.last_avg_time = time.perf_counter()
            return 0

    def dump_buff_overflow(self):
        """Doc."""

        if len(self.cont_count_buff) > self.ci_buff_sz:
            self.cont_count_buff = self.cont_count_buff[-self.ci_buff_sz :]


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
            loop.run_in_executor() must be awaited, but toggle_video needs to be
            a regular function to keep the rest of the code as it is. by creating this
            async helper function I can make it work. A lambda would be better here
            but there's no async lambda yet.
            """

            await self._loop.run_in_executor(None, self._vidshow)

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
