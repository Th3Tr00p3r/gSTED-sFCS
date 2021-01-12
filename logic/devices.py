# -*- coding: utf-8 -*-
"""Devices Module."""

import asyncio
import time
from typing import NoReturn

import numpy as np

import logic.drivers as drivers
import utilities.constants as const
import utilities.dialog as dialog
from utilities.errors import dvc_err_hndlr as err_hndlr


class UM232H(drivers.FTDI_Instrument):
    """Doc."""

    def __init__(self, nick, param_dict, error_dict, led):
        self.led = led
        super().__init__(nick=nick, param_dict=param_dict, error_dict=error_dict)
        self.init_data()
        self.toggle(True)

    def toggle(self, bool):
        """Doc."""

        if bool:
            self.open()
            self.purge()
        else:
            self.purge()
            self.close()

    def stream_read_TDC(self, meas):
        """Doc."""

        while (
            meas.time_passed < meas.duration_gui.value() * meas.duration_multiplier
            and meas.is_running
        ):
            read_bytes = self.read()
            #            print(f"# Bytes read: {len(read_bytes)}")  # TEST
            self.data = np.append(self.data, read_bytes)

            meas.time_passed = time.perf_counter() - meas.start_time

        self.tot_bytes_read = len(self.data)

    def init_data(self):
        """Doc."""

        self.data = np.empty(shape=(0,), dtype=np.uint8)
        self.tot_bytes_read = 0

    def reset(self):
        """Doc."""

        self.reset_dvc()
        self.close()
        time.sleep(0.3)
        self.open()


class Scanners(drivers.DAQmxInstrumentAIO):
    """
    Scanners encompasses all analog focal point positioning devices
    (X: x_galvo, Y: y_galvo, Z: z_piezo)

    """

    x_limits = {"min_val": -5.0, "max_val": 5.0}
    y_limits = {"min_val": -5.0, "max_val": 5.0}
    z_limits = {"min_val": 0.0, "max_val": 10.0}

    # TODO - move these to settings -> param_dict
    buff_sz = 1000
    ai_clk_rate = 1000

    def __init__(self, nick, param_dict, error_dict, led, init_pos_vltgs, um_V_ratio):
        self.led = led
        super().__init__(nick=nick, param_dict=param_dict, error_dict=error_dict)

        self.last_ai = None
        self.last_ao = init_pos_vltgs
        self.um_V_ratio = um_V_ratio
        # TODO: these buffers will be used for scans so the shape of the scan could be used/reviewed and compared for AO vs. AI
        self.ao_buffer = np.empty(shape=(3, 0), dtype=np.float)
        self.ai_buffer = np.empty(shape=(3, 0), dtype=np.float)

        self.toggle(True)  # turn ON right from the start

    def toggle(self, bool):
        """Doc."""

        if bool:
            self.start_ai()
        else:
            self.close_ai()

    def fill_ai_buff(self) -> NoReturn:
        """Doc."""

        #        # TODO: stream reading currently not working for some reason - reading only one channel, the other two stay at zero
        #        num_samps_read = self.read()
        #        self.ai_buffer = np.concatenate(
        #        (self.ai_buffer, self.read_buffer[:, :num_samps_read]), axis=1
        #        )

        self.ai_buffer = np.concatenate((self.ai_buffer, self.read()), axis=1)

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
        for axis, curr_axis_vltg, new_axis_vltg in zip(axes, self.last_ao, pos_vltgs):
            if axis in {"x", "y"}:  # Differential Voltage [-5,5]
                ao_addresses.append(
                    [
                        self._param_dict[f"ao_{axis}_p_addr"],
                        self._param_dict[f"ao_{axis}_n_addr"],
                    ]
                )
                limits.append(getattr(self, f"{axis}_limits"))
                vltgs += [new_axis_vltg, -new_axis_vltg]
            else:  # RSE Voltage [0,10]
                ao_addresses.append(self._param_dict[f"ao_{axis}_addr"])
                limits.append(getattr(self, f"{axis}_limits"))
                vltgs.append(new_axis_vltg)

        self.last_ao = pos_vltgs

        *ao_xy_addresses, ao_z_addr = ao_addresses
        *xy_vltgs, z_vltg = vltgs
        *xy_limits, z_limits = limits

        await self.write(ao_xy_addresses, xy_vltgs, xy_limits)
        await self.write([ao_z_addr], [z_vltg], [z_limits])

    def dump_buff_overflow(self):
        """Doc."""

        if max(self.ao_buffer.shape) > self.buff_sz:
            self.ao_buffer = self.ao_buffer[:, -self.buff_sz :]
        if max(self.ai_buffer.shape) > self.buff_sz:
            self.ai_buffer = self.ai_buffer[:, -self.buff_sz :]


class Counter(drivers.DAQmxInstrumentCI):
    """Doc."""

    # TODO: ADD CHECK FOR ERROR CAUSED BY INACTIVITY (SUCH AS WHEN DEBUGGING).
    # PREVIOUSLY DONE IN TIMEOUT

    def __init__(self, nick, param_dict, error_dict, led, ai_task):
        self.led = led
        super().__init__(
            nick=nick, param_dict=param_dict, error_dict=error_dict, ai_task=ai_task
        )
        #        self.cont_count_buff = []
        self.cont_count_buff = np.empty(shape=(0,))
        self.counts = None  # this is for scans where the counts are actually used.
        self.update_time = param_dict["update_time"]
        self.last_avg_time = time.perf_counter()
        self.num_reads_since_avg = 0

        self.toggle(True)  # turn ON right from the start

    def toggle(self, bool):
        """Doc."""

        if bool:
            self.start()
        else:
            self.close()

    def count(self):
        """Doc."""

        num_samps_read = self.read()
        self.cont_count_buff = np.append(
            self.cont_count_buff, self.read_buffer[:num_samps_read]
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

        if len(self.cont_count_buff) > self._param_dict["buff_sz"]:
            self.cont_count_buff = self.cont_count_buff[-self._param_dict["buff_sz"] :]


class Camera(drivers.UC480Instrument):
    """Doc."""

    # TODO: test during UM232H reading. could be that pyUSB/pyftdi interferes with the uc480

    def __init__(self, nick, param_dict, error_dict, led, loop, gui):
        self.led = led
        super().__init__(nick=nick, param_dict=param_dict, error_dict=error_dict)
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
            time.sleep(self.param_dict["vid_intrvl"])

    @err_hndlr
    def _imshow(self, img):
        """Plot image"""

        self._gui.figure.clear()
        ax = self._gui.figure.add_subplot(111)
        ax.imshow(img)
        self._gui.canvas.draw()


class SimpleDO(drivers.DAQmxInstrumentDO):
    """ON/OFF device (excitation laser, depletion shutter, TDC)."""

    def __init__(self, nick, param_dict, error_dict, led):
        self.led = led
        super().__init__(nick=nick, address=param_dict["addr"], error_dict=error_dict)

        self.toggle(False)

    def toggle(self, bool):
        """Doc."""

        self.write(bool)


class DepletionLaser(drivers.VISAInstrument):
    """Control depletion laser through pyVISA"""

    min_SHG_temp = 52

    def __init__(self, nick, param_dict, error_dict, led):
        self.led = led
        super().__init__(
            nick=nick,
            address=param_dict["addr"],
            error_dict=error_dict,
            read_termination="\r",
            write_termination="\r",
        )
        self.update_time = param_dict["update_time"]
        self.state = None

        self.toggle(False)

        if self.state is False:
            self.set_current(1500)

    def toggle(self, bool):
        """Doc."""

        self._write(f"setLDenable {int(bool)}")

    async def get_prop(self, prop):
        """Doc."""

        return await self._aquery(const.DEP_CMND_DICT[prop])

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

    # TODO: (low priority) try to fit with VISA driver - try adding longer response time as option to driver

    def __init__(self, nick, param_dict, error_dict, led):
        self.led = led
        self.nick = nick
        self.address = param_dict["addr"]
        self.error_dict = error_dict
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
