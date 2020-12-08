# -*- coding: utf-8 -*-
"""Devices Module."""

import numpy as np
from instrumental.drivers.cameras.uc480 import UC480Error
from PyQt5.QtCore import QTimer

import logic.drivers as drivers
import utilities.constants as const
import utilities.dialog as dialog
from utilities.errors import driver_error_handler as err_hndlr


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
        self.counts = None
        self.update_time = param_dict["update_time"]

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

    def average_counts(self, avg_intrvl):
        """Doc."""

        intrvl_time_unts = int(avg_intrvl / const.TIMEOUT)
        start_idx = len(self.cont_count_buff) - intrvl_time_unts

        if start_idx > 0:
            return (
                self.cont_count_buff[-1] - self.cont_count_buff[-(intrvl_time_unts + 1)]
            ) / avg_intrvl  # to have KHz

        else:  # TODO: get the most averaging possible if requested fails
            print("TEST TEST TEST: start_idx < 0")
            return 0

    def dump_buff_overflow(self):
        """Doc."""

        buff_sz = self._param_dict["buff_sz"]
        cnts_arr1D = self.cont_count_buff

        if len(cnts_arr1D) > buff_sz:
            self.cont_count_buff = cnts_arr1D[-buff_sz:]


class Camera:
    """Doc."""

    # TODO: create driver class and move _driver and error handeling there

    def __init__(self, nick, error_dict):
        """Doc."""

        self.nick = nick
        self.error_dict = error_dict
        self.video_timer = QTimer()
        self.video_timer.setInterval(100)  # set to 100 ms
        self.state = False

    @err_hndlr
    def toggle(self, bool):
        """Doc."""

        if bool:
            try:  # this is due to bad error handeling in instrumental-lib...
                self._driver = drivers.UC480_Camera(reopen_policy="new")
            except Exception:
                raise UC480Error
        elif hasattr(self, "_driver"):
            self.video_timer.stop()  # in case video is ON
            self._driver.close()
        self.state = bool

    @err_hndlr
    def set_auto_exposure(self, bool):
        """Doc."""

        self._driver.set_auto_exposure(bool)

    @err_hndlr
    def shoot(self):
        """Doc."""

        if self.video_timer.isActive():
            self.toggle_video(False)
            img = self._driver.grab_image()
            self.toggle_video(True)

        else:
            img = self._driver.grab_image()

        return img

    @err_hndlr
    def toggle_video(self, bool):
        """Doc."""

        if bool:
            self._driver.start_live_video()
            self.video_timer.start()

        else:
            self._driver.stop_live_video()
            self.video_timer.stop()

        self.video_state = bool

    @err_hndlr
    def latest_frame(self):
        """Doc."""

        frame_ready = self._driver.wait_for_frame(timeout="0 ms")
        if frame_ready:
            return self._driver.latest_frame(copy=False)


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
#        self.state = False

        self.toggle(False)
        self.set_current(1500)
        self.get_SHG_temp()

    def toggle(self, bool):
        """Doc."""

        self._write(f"setLDenable {int(bool)}")

    def get_SHG_temp(self):
        """Doc."""

        self.temp = self._query("SHGtemp")

    def get_current(self):
        """Doc."""

        self.current = self._query("LDcurrent 1")

    def get_power(self):
        """Doc."""

        self.power = self._query("Power 0")

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
