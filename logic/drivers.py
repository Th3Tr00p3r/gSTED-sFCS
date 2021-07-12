"""Drivers Module."""

import re
from types import SimpleNamespace
from typing import List

import ftd2xx
import nidaqmx as ni
import numpy as np
import pyvisa as visa
from instrumental.drivers.cameras import uc480
from nidaqmx.stream_readers import AnalogMultiChannelReader, CounterReader  # NOQA

from utilities.errors import IOError
from utilities.helper import translate_dict_values


class Ftd2xx:
    """Doc."""

    ftd2xx_dict = {
        "Single Channel Synchronous 245 FIFO": 0x40,
        "RTS-CTS": ftd2xx.defines.FLOW_RTS_CTS,
    }

    def __init__(self, param_dict):
        param_dict = translate_dict_values(param_dict, self.ftd2xx_dict)
        [setattr(self, key, val) for key, val in param_dict.items()]
        self.error_dict = None

        # auto-find UM232H serial number
        num_devs = ftd2xx.createDeviceInfoList()
        for idx in range(num_devs):
            info_dict = ftd2xx.getDeviceInfoDetail(devnum=idx)
            if info_dict["description"] == b"UM232H":
                self.serial = info_dict["serial"]

    def open(self):
        """Doc."""

        try:
            self._inst = ftd2xx.aio.openEx(self.serial)
        except AttributeError:
            raise IOError(f"{self.log_ref} is not plugged in.")
        self._inst.setBitMode(255, self.bit_mode)  # unsure about 255/0
        self._inst.setTimeouts(self.timeout_ms, self.timeout_ms)
        self._inst.setLatencyTimer(self.ltncy_tmr_val)
        self._inst.setFlowControl(self.flow_ctrl)
        self._inst.setUSBParameters(self.tx_size)

    async def read(self) -> np.ndarray:
        """Doc."""

        # TODO: fix bug where something here blocks - I think it's some error in the async read function.
        # check this out: https://docs.python.org/3/library/asyncio-dev.html#detect-never-retrieved-exceptions
        raw_bytes = await self._inst.read(self.n_bytes)
        read_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)
        return read_bytes

    def purge(self) -> None:
        """Doc."""

        self._inst.purge(ftd2xx.defines.PURGE_RX)

    def close(self) -> None:
        """Doc."""

        self._inst.close()
        self.state = False

    def get_queue_status(self) -> None:
        """Doc."""

        return self._inst.getQueueStatus()


class NIDAQmx:
    """Doc."""

    MIN_OUTPUT_RATE_Hz = 1000
    CONT_READ_BFFR_SZ = 10000
    AO_TIMEOUT = 0.1

    def __init__(self, param_dict, task_types, **kwargs):
        self.error_dict = None
        [setattr(self, key, val) for key, val in param_dict.items()]
        self.tasks = SimpleNamespace()
        self.task_types = task_types
        [setattr(self.tasks, type, []) for type in self.task_types]

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
                f"NI device address ({self.ai_x_addr}) is wrong, or Data acquisition board is unplugged"
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
        #        ai_task = getattr(self.tasks, task_type)[task_name]
        return np.array(ai_task.read(number_of_samples_per_channel=n_samples))

    def analog_write(self, task_name: str, data: np.ndarray, auto_start=None) -> None:
        """Doc."""

        ao_task = [task for task in self.tasks.ao if (task.name == task_name)][0]
        #        ao_task = self.tasks.ao[task_name]
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
        self.error_dict = None
        [setattr(self, key, val) for key, val in param_dict.items()]
        self.read_termination = read_termination
        self.write_termination = write_termination
        self._rm = visa.ResourceManager()

        # auto-find serial connection for depletion laser
        if hasattr(self, "model_query"):
            self.autofind_address()

    def autofind_address(self) -> None:
        """Doc."""

        # list all resource addresses
        resource_address_tuple = self._rm.list_resources()

        # open and save the opened ones in a list
        inst_list = []
        for resource_name in resource_address_tuple:
            try:
                # try to open instrument
                inst = self._rm.open_resource(
                    resource_name,
                    read_termination=self.read_termination,
                    write_termination=self.write_termination,
                    timeout=50,  # ms
                    open_timeout=50,  # ms
                )
            except visa.errors.VisaIOError:
                # failed to open instrument, skip
                pass
            else:
                # managed to open instrument, add to list.
                inst_list.append(inst)

        # 'model_query' the opened devices, and whoever responds is the one
        for idx, inst in enumerate(inst_list):
            try:
                self.model = inst.query(self.model_query)
            except visa.errors.VisaIOError:
                pass
            else:
                self.address = resource_address_tuple[idx]
                break

        # close all saved resources
        [inst.close() for inst in inst_list]

    def open_inst(self) -> None:
        """Doc."""

        try:
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

    def close_inst(self) -> None:
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
        self.error_dict = None
        [setattr(self, key, val) for key, val in param_dict.items()]
        self._inst = None

    def init_cam(self):
        """Doc."""

        try:
            self._inst = uc480.UC480_Camera(reopen_policy="new")
        except Exception:
            # general 'Exception' is due to bad error handeling in instrumental-lib...
            raise uc480.UC480Error(msg="Camera disconnected")

    def close_cam(self):
        """Doc."""

        if self._inst is not None:
            self._inst.close()

    def grab_image(self):
        """Doc."""

        return self._inst.grab_image()

    def toggle_vid(self, state):
        """Doc."""

        if state is True:
            self._inst.start_live_video()
        else:
            self._inst.stop_live_video()

    def get_latest_frame(self):
        """Doc."""

        frame_ready = self._inst.wait_for_frame(timeout="0 ms")
        if frame_ready:
            return self._inst.latest_frame(copy=False)
