# -*- coding: utf-8 -*-
"""Devices Module."""

import asyncio
import time

import numpy as np

import logic.drivers as drivers
import utilities.constants as const
import utilities.dialog as dialog
from utilities.errors import dvc_err_hndlr as err_hndlr


class UM232(drivers.FTDI_Instrument):
    """Doc."""

    def __init__(self, nick, param_dict, error_dict):
        """Doc."""

        super().__init__(nick=nick, param_dict=param_dict, error_dict=error_dict)
        self.init_data()
        self.toggle(True)
        self.check_read_error()

    def toggle(self, bool):
        """Doc."""

        if bool:
            self.open()
            self.purge()
        else:
            self.purge()
            self.close()

    def check_read_error(self):
        """Doc."""

        if self.is_read_error():
            self.error_dict[self.nick] = "read error"

    def read_TDC_data(self):
        """Doc."""

        read_bytes = self.read_bytes(self._param_dict["n_bytes"])
        if isinstance(read_bytes, list):  # check for error
            self.tot_bytes += len(read_bytes)
            self.data = np.append(self.data, read_bytes)

        self.check_read_error()

    def init_data(self):
        """Doc."""

        self.data = np.empty(shape=(0,))
        self.tot_bytes = 0


class Counter(drivers.DAQmxInstrumentCI):
    """Doc."""

    def __init__(self, nick, param_dict, error_dict):
        """Doc."""

        super().__init__(nick=nick, param_dict=param_dict, error_dict=error_dict)
        self.cont_count_buff = []
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

        counts = self.read()
        self.cont_count_buff.append(counts)
        self.num_reads_since_avg += 1

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

            #            # TEST ----------------------------------------------------------------------------------------
            #            print(f'')
            #            # -----------------------------------------------------------------------------------------------

            return avg_cnt_rate / 1000  # Hz -> KHz

        else:
            self.num_reads_since_avg = 0
            self.last_avg_time = time.perf_counter()
            return 0

    def dump_buff_overflow(self):
        """Doc."""

        buff_sz = self._param_dict["buff_sz"]
        cnts_arr1D = self.cont_count_buff

        if len(cnts_arr1D) > buff_sz:
            self.cont_count_buff = cnts_arr1D[-buff_sz:]


class Camera(drivers.UC480Instrument):
    """Doc."""

    def __init__(self, nick, error_dict, app, gui):
        """Doc."""

        super().__init__(nick=nick, error_dict=error_dict)
        self._app = app
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
            await self.toggle_vid(False)
            img = self.grab_image()
            self._imshow(img)
            await asyncio.gather(self.toggle_vid(True), self._vidshow())

    def toggle_video(self, bool):
        """Doc."""

        self._app.loop.create_task(self.toggle_vid(bool))

        if bool:
            self._app.loop.create_task(self._vidshow())

    @err_hndlr
    def _imshow(self, img):
        """Plot image"""

        self._gui.figure.clear()
        ax = self._gui.figure.add_subplot(111)
        ax.imshow(img)
        self._gui.canvas.draw()

    async def _vidshow(self):
        """Doc."""

        while self.vid_state is True:
            img = self.get_latest_frame()
            self._imshow(img)
            await asyncio.sleep(const.CAM_VID_INTRVL)


class SimpleDO(drivers.DAQmxInstrumentDO):
    """ON/OFF device (excitation laser, depletion shutter, TDC)."""

    def __init__(self, nick, param_dict, error_dict):
        """Doc."""

        super().__init__(nick=nick, address=param_dict["addr"], error_dict=error_dict)


class DepletionLaser(drivers.VISAInstrument):
    """Control depletion laser through pyVISA"""

    min_SHG_temp = 52

    def __init__(self, nick, param_dict, error_dict):
        """Doc."""

        super().__init__(
            nick=nick,
            address=param_dict["addr"],
            error_dict=error_dict,
            read_termination="\r",
            write_termination="\r",
        )
        self.update_time = param_dict["update_time"]

        if self.toggle(False) is True:
            self.set_current(1500)
            self.get_SHG_temp()

    def toggle(self, bool):
        """Doc."""

        self._write(f"setLDenable {int(bool)}")

    async def get_SHG_temp(self):
        """Doc."""

        self.temp = await self._aquery("SHGtemp")

    async def get_current(self):
        """Doc."""

        self.current = await self._aquery("LDcurrent 1")

    async def get_power(self):
        """Doc."""

        self.power = await self._aquery("Power 0")

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

    def __init__(self, nick, param_dict, error_dict):
        """Doc."""

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
