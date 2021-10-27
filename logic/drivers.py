"""Drivers Module."""

import re
import sys
from contextlib import suppress
from types import SimpleNamespace
from typing import List

import ftd2xx
import instrumental.drivers.cameras.uc480 as uc480
import nidaqmx as ni
import numpy as np
import pyvisa as visa
from instrumental import list_instruments
from nidaqmx.stream_readers import (
    CounterReader,  # AnalogMultiChannelReader for AI, if ever
)

import utilities.helper as helper
from utilities.errors import IOError, err_hndlr


class Ftd2xx:
    """Doc."""

    ftd2xx_dict = {
        "Single Channel Synchronous 245 FIFO": 0x40,
        "RTS-CTS": ftd2xx.defines.FLOW_RTS_CTS,
    }

    def __init__(self, param_dict):
        param_dict = helper.translate_dict_values(param_dict, self.ftd2xx_dict)
        [setattr(self, key, val) for key, val in param_dict.items()]

        # auto-find UM232H serial number
        num_devs = ftd2xx.createDeviceInfoList()
        for idx in range(num_devs):
            info_dict = ftd2xx.getDeviceInfoDetail(devnum=idx)
            if info_dict["description"] == b"UM232H":
                self.serial = info_dict["serial"]

    def open_instrument(self):
        """Doc."""

        try:
            self._inst = ftd2xx.ftd2xx.openEx(self.serial)
        #            self._inst = ftd2xx.aio.openEx(self.serial)
        except AttributeError:
            raise IOError(f"{self.log_ref} is not plugged in.")
        self._inst.setBitMode(255, self.bit_mode)  # unsure about 255/0
        self._inst.setTimeouts(self.timeout_ms, self.timeout_ms)
        self._inst.setLatencyTimer(self.ltncy_tmr_val)
        self._inst.setFlowControl(self.flow_ctrl)
        self._inst.setUSBParameters(self.tx_size)

    def close_instrument(self) -> None:
        """Doc."""

        self._inst.close()

    def read(self) -> bytes:
        """Doc."""

        return self._inst.read(self.n_bytes)

    # NOT USED, CURRENTLY (using regular read())
    async def async_read(self) -> bytes:
        """Doc."""

        # todo: fix bug where something here blocks - I think it's some error in the async read function.
        # check this out: https://docs.python.org/3/library/asyncio-dev.html#detect-never-retrieved-exceptions
        return await self._inst.read(self.n_bytes)

    def purge(self) -> None:
        """Doc."""

        self._inst.purge(ftd2xx.defines.PURGE_RX)

    def get_queue_status(self) -> None:
        """Doc."""

        return self._inst.getQueueStatus()


class NIDAQmx:
    """Doc."""

    MIN_OUTPUT_RATE_Hz = int(1e3)
    CONT_READ_BFFR_SZ = int(1e5)
    AO_TIMEOUT = 0.1

    def __init__(self, param_dict, task_types, **kwargs):
        [setattr(self, key, val) for key, val in param_dict.items()]
        self.task_types = task_types
        self.tasks = SimpleNamespace(**{type: [] for type in task_types})

        try:
            len(ni.system.system.System().devices)
        except FileNotFoundError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)

    def start_tasks(self, task_type: str) -> None:
        """Doc."""

        task_list = getattr(self.tasks, task_type)
        [task.start() for task in task_list]

    def are_tasks_done(self, task_type: str) -> bool:
        """Doc."""

        task_list = getattr(self.tasks, task_type)
        task_status = [task.is_task_done() for task in task_list]
        return task_status.count(1) == len(task_status)

    def pause_tasks(self, task_type: str) -> None:
        """Doc."""

        task_list = getattr(self.tasks, task_type)
        [task.stop() for task in task_list]

    def close_tasks(self, task_type: str) -> None:
        """Doc."""

        task_list = getattr(self.tasks, task_type)
        [task.close() for task in task_list]
        setattr(self.tasks, task_type, [])

    def close_all_tasks(self):
        """Doc."""

        for type in self.task_types:
            self.close_tasks(type)

    def wait_for_task(self, task_type: str, task_name: str):
        """Doc."""

        task_list = getattr(self.tasks, task_type)
        [task.wait_until_done(timeout=3) for task in task_list if (task.name == task_name)]

    def create_ai_task(
        self,
        name: str,
        chan_specs: List[dict],
        samp_clk_cnfg: dict = {},
        timing_params: dict = {},
    ):
        """Doc."""

        task = ni.Task(new_task_name=name)
        try:
            for chan_spec in chan_specs:
                task.ai_channels.add_ai_voltage_chan(**chan_spec)
            if samp_clk_cnfg:
                task.timing.cfg_samp_clk_timing(**samp_clk_cnfg)
            for key, val in timing_params.items():
                setattr(task.timing, key, val)

        except ni.errors.DaqError:
            task.close()
            raise IOError(
                f"NI device address ({self.ai_x_addr}) is wrong, or data acquisition board is unplugged"
            )

        else:
            self.tasks.ai.append(task)

    def create_ao_task(
        self,
        name: str,
        chan_specs: List[dict],
        samp_clk_cnfg: dict = {},
        timing_params: dict = {},
    ) -> ni.Task:
        """Doc."""

        task = ni.Task(new_task_name=name)
        for chan_spec in chan_specs:
            task.ao_channels.add_ao_voltage_chan(**chan_spec)
        if samp_clk_cnfg:
            task.timing.cfg_samp_clk_timing(**samp_clk_cnfg)
        for key, val in timing_params.items():
            setattr(task.timing, key, val)
        self.tasks.ao.append(task)
        return task

    def create_ci_task(
        self,
        name: str,
        chan_specs: dict,
        samp_clk_cnfg: dict = {},
        timing_params: dict = {},
        chan_xtra_params: dict = {},
        clk_xtra_params: dict = {},
    ):
        """Doc."""

        task = ni.Task(new_task_name=name)
        chan = task.ci_channels.add_ci_count_edges_chan(**chan_specs)
        for key, val in chan_xtra_params.items():
            setattr(chan, key, val)
        if samp_clk_cnfg:
            task.timing.cfg_samp_clk_timing(**samp_clk_cnfg)
        for key, val in timing_params.items():
            setattr(task.timing, key, val)
        for key, val in clk_xtra_params.items():
            setattr(task.timing, key, val)

        task.sr = CounterReader(task.in_stream)
        task.sr.verify_array_shape = False

        self.tasks.ci.append(task)

    def create_co_task(self, name: str, chan_spec: dict, clk_cnfg: dict) -> None:
        """Doc."""

        task = ni.Task(new_task_name=name)
        task.co_channels.add_co_pulse_chan_ticks(**chan_spec)
        task.timing.cfg_implicit_timing(**clk_cnfg)

        self.tasks.co.append(task)

    def analog_read(self, task_name: str, n_samples):
        """Doc."""

        ai_task = [task for task in self.tasks.ai if (task.name == task_name)][0]
        return np.array(ai_task.read(number_of_samples_per_channel=n_samples))

    def analog_write(self, task_name: str, data: np.ndarray, auto_start=None) -> None:
        """Doc."""

        ao_task = [task for task in self.tasks.ao if (task.name == task_name)][0]
        if auto_start is not None:
            ao_task.write(data, auto_start=auto_start, timeout=self.AO_TIMEOUT)
        else:
            ao_task.write(data, timeout=self.AO_TIMEOUT)

    def counter_stream_read(self) -> int:
        """
         Reads all available samples on board into self.cont_read_buffer
         (1D NumPy array, overwritten each read), and returns the
        number of samples read.
        """

        ci_task = self.tasks.ci[0]  # only ever one task
        return ci_task.sr.read_many_sample_uint32(
            self.cont_read_buffer,
            number_of_samples_per_channel=ni.constants.READ_ALL_AVAILABLE,
        )

    def digital_write(self, bool) -> None:
        """Doc."""

        with ni.Task() as do_task:
            do_task.do_channels.add_do_chan(self.address)
            do_task.write(bool)


class PyVISA:
    """Doc."""

    def __init__(
        self,
        param_dict,
        read_termination="",
        write_termination="",
    ):
        [setattr(self, key, val) for key, val in param_dict.items()]
        self.read_termination = read_termination
        self.write_termination = write_termination
        self._rm = visa.ResourceManager()

    def autofind_address(self) -> None:
        """Doc."""

        # list all resource addresses
        resource_address_tuple = self._rm.list_resources()

        # open and save the opened ones in a list
        inst_list = []
        for resource_name in resource_address_tuple:
            with suppress(visa.errors.VisaIOError):
                # VisaIOError - failed to open instrument, skip
                # try to open instrument
                inst = self._rm.open_resource(
                    resource_name,
                    read_termination=self.read_termination,
                    write_termination=self.write_termination,
                    timeout=50,  # ms
                    open_timeout=50,  # ms
                )
                # managed to open instrument, add to list.
                inst_list.append(inst)

        # 'model_query' the opened devices, and whoever responds is the one
        for idx, inst in enumerate(inst_list):
            try:
                self.model = inst.query(self.model_query)
            except visa.errors.VisaIOError as exc:
                if exc.abbreviation == "VI_ERROR_RSRC_BUSY":
                    raise IOError(f"{self.log_ref} couldn't be opened - a restart might help!")
                else:
                    pass
            else:
                self.address = resource_address_tuple[idx]
                break

        # close all saved resources
        with suppress(visa.errors.VisaIOError):
            # VisaIOError - this happens if device was disconnected during the autofind process...
            [inst.close() for inst in inst_list]

    def open_instrument(self) -> None:
        """Doc."""

        # auto-find serial connection for depletion laser
        if hasattr(self, "model_query"):
            self.autofind_address()

        try:
            # open
            self._rsrc = self._rm.open_resource(
                self.address,
                read_termination=self.read_termination,
                write_termination=self.write_termination,
                timeout=50,  # ms
                open_timeout=50,  # ms
            )
        except AttributeError:
            # failed to auto-find address for depletion laser
            raise IOError(
                f"{self.log_ref} couldn't be opened - it is either turned off, unplugged or the address is used by another process"
            )
        except ValueError:
            raise IOError(f"{self.log_ref} is unplugged or the address ({self.address}) is wrong.")
        except visa.errors.VisaIOError:
            raise IOError(f"{self.log_ref} was manually disconnected while opening the instrument.")

    def close_instrument(self) -> None:
        """Doc."""

        try:
            self._rsrc.close()
        except AttributeError:
            raise IOError(f"{self.log_ref} was never opened")
        except visa.errors.VisaIOError:
            raise IOError(f"{self.log_ref} disconnected during operation.")

    def write(self, cmnd: str) -> None:
        """Sends a command to the VISA instrument."""

        self._rsrc.write(cmnd)

    def query(self, cmnd: str) -> float:
        """Doc."""

        try:
            response = self._rsrc.query(cmnd)
        except visa.errors.VisaIOError:
            raise IOError(f"{self.log_ref} disconnected! Reconnect and restart.")
        else:
            try:
                extracted_float_string = re.findall(r"-?\d+\.?\d*", response)[0]
            except IndexError:
                # rarely happens
                return 0
            return float(extracted_float_string)

    def flush(self) -> None:
        """Doc."""

        mask = visa.constants.BufferOperation(4) | visa.constants.BufferOperation(6)
        try:
            self._rsrc.flush(mask)
        except visa.errors.VisaIOError:
            raise IOError(f"{self.log_ref} disconnected! Reconnect and restart.")


class Instrumental:
    """Doc."""

    def __init__(self, param_dict):
        [setattr(self, key, val) for key, val in param_dict.items()]
        self._inst = None

        if not list_instruments(module="cameras"):
            exc = IOError("No UC480 cameras detected.")
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)

    def open_instrument(self):
        """Doc."""

        try:
            self._inst = uc480.UC480_Camera(serial=self.serial.encode(), reopen_policy="new")
        except Exception as exc:
            # general 'Exception' is due to bad exception handeling in instrumental-lib...
            raise IOError(f"{self.log_ref} disconnected - {exc}")

    def close_instrument(self):
        """Doc."""

        if self._inst is not None:
            self._inst.close()

    def toggle_video_mode(self, should_turn_on: bool) -> None:
        """Doc."""

        try:
            if should_turn_on:
                if self.is_auto_exposure_on:
                    self._inst.start_live_video()
                    self._inst.set_auto_exposure(True)
                else:
                    self._inst.start_live_video(exposure_time=self._inst._get_exposure())
            else:
                self._inst.stop_live_video()
        except uc480.UC480Error:
            raise IOError(f"{self.log_ref} disconnected after initialization.")
        else:
            self.is_in_video_mode = should_turn_on

    def grab_image(self) -> np.ndarray:
        """Doc."""

        try:
            return self._inst.grab_image(copy=False, exposure_time=self._inst._get_exposure())
        except uc480.UC480Error:
            raise IOError(f"{self.log_ref} disconnected after initialization.")

    def get_latest_frame(self) -> np.ndarray:
        """Doc."""

        try:
            return self._inst.latest_frame(copy=False)
        except uc480.UC480Error:
            raise IOError(f"{self.log_ref} disconnected after initialization.")

    def update_parameter_ranges(self):
        """Doc."""

        *self.pixel_clock_range, _ = tuple(
            self._inst._dev.PixelClock(uc480.lib.PIXELCLOCK_CMD_GET_RANGE)
        )
        self.framerate_range = (1, self._inst.pixelclock._magnitude / 1.6)
        *self.exposure_range, _ = tuple(
            self._inst._dev.Exposure(uc480.lib.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE)
        )

    def set_parameter(self, name, value) -> None:
        """Doc."""

        if name not in {"pixel_clock", "framerate", "exposure"}:
            raise ValueError(f"Unknown parameter '{name}'.")

        valid_value = helper.limit(value, *getattr(self, f"{name}_range"))

        try:
            if name == "pixel_clock":
                self._inst._dev.PixelClock(uc480.lib.PIXELCLOCK_CMD_SET, int(valid_value))
                self.update_parameter_ranges()
            elif name == "framerate":
                self._inst._dev.SetFrameRate(valid_value)
                self.update_parameter_ranges()
            elif name == "exposure":
                self._inst._set_exposure(f"{valid_value}ms")
        except uc480.UC480Error:
            exc = IOError(f"{self.log_ref} was not properly closed.\nRestart to fix.")
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)

    def set_auto_exposure(self, should_turn_on: bool) -> None:
        """Doc."""

        try:
            self._inst.set_auto_exposure(should_turn_on)
        except uc480.UC480Error:
            exc = IOError(f"{self.log_ref} was not properly closed.\nRestart to fix.")
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.is_auto_exposure_on = should_turn_on
