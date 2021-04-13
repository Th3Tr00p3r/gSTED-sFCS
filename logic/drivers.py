# -*- coding: utf-8 -*-
"""Drivers Module."""

from array import array
from types import SimpleNamespace
from typing import List, NoReturn

import nidaqmx as ni
import numpy as np
import pyvisa as visa
from instrumental.drivers.cameras.uc480 import UC480_Camera, UC480Error
from nidaqmx.stream_readers import AnalogMultiChannelReader, CounterReader  # NOQA
from pyftdi.ftdi import Ftdi, FtdiError
from pyftdi.usbtools import UsbTools


class FtdiInstrument:
    """Doc."""

    def __init__(self, param_dict):

        [setattr(self, key, val) for key, val in param_dict.items()]
        self.error_dict = None

        self._inst = Ftdi()  # URL - ftdi://ftdi:232h:FT3TG15/1

    def open(self):
        """Doc."""

        UsbTools.flush_cache()

        self._inst.open(self.vend_id, self.prod_id)
        self._inst.set_bitmode(0, getattr(Ftdi.BitMode, self.bit_mode))
        self._inst.set_latency_timer(self.ltncy_tmr_val)
        self._inst.set_flowctrl(self.flow_ctrl)
        self._inst.read_data_set_chunksize(self.n_bytes)

        self.state = True

    def read(self):
        """Doc."""

        read_bytes = self._inst._usb_dev.read(
            self._inst._out_ep,
            self._inst._readbuffer_chunksize,
        )
        self.check_status(read_bytes[:2])

        return read_bytes[2:]

    def purge(self):
        """Doc."""

        self._inst.purge_rx_buffer()

    def close(self):
        """Doc."""

        self._inst.close()
        self.state = False

    def reset_dvc(self):
        """Doc."""

        self._inst.reset(usb_reset=True)

    def check_status(self, status: array) -> NoReturn:
        """Doc."""

        if status[1] & self._inst.ERROR_BITS[1]:
            s = " ".join(self._inst.decode_modem_status(status, True)).title()
            raise FtdiError(f"FTDI error: {status[0]:02x}:{ status[1]:02x} {s}")


class NIDAQmxInstrument:
    """Doc."""

    MIN_OUTPUT_RATE_Hz = 1000
    CONT_READ_BFFR_SZ = 10000
    AO_TIMEOUT = 0.1

    def __init__(self, param_dict, task_types, **kwargs):
        self.error_dict = None
        [setattr(self, key, val) for key, val in param_dict.items()]
        self.tasks = SimpleNamespace()
        self.task_types = task_types
        [setattr(self.tasks, type, {}) for type in self.task_types]

    def start_tasks(self, task_type: str):
        """Doc."""

        [task.start() for task in getattr(self.tasks, task_type).values()]

    def are_tasks_done(self, task_type: str) -> bool:
        """Doc."""

        task_status = [
            task.is_task_done() for task in getattr(self.tasks, task_type).values()
        ]
        return task_status.count(1) == len(task_status)

    def close_tasks(self, task_type: str):
        """Doc."""

        [task.close() for task in getattr(self.tasks, task_type).values()]
        setattr(self.tasks, task_type, {})

    def close_all_tasks(self):
        """Doc."""

        for type in self.task_types:
            self.close_tasks(type)

    def wait_for_task(self, task_type: str, task_name: str):
        """Doc."""

        getattr(self.tasks, task_type)[task_name].wait_until_done(timeout=3)

    def create_ai_task(
        self,
        name: str,
        chan_specs: List[dict],
        samp_clk_cnfg: dict = {},
        timing_params: dict = {},
    ):
        """Doc."""

        task = ni.Task(new_task_name=name)
        for chan_spec in chan_specs:
            task.ai_channels.add_ai_voltage_chan(**chan_spec)
        if samp_clk_cnfg:
            task.timing.cfg_samp_clk_timing(**samp_clk_cnfg)
        for key, val in timing_params.items():
            setattr(task.timing, key, val)

        self.tasks.ai[name] = task

        #        # TODO: stream reading currently not working for some reason - reading only one channel, the other two stay at zero
        #        self.sreader = AnalogMultiChannelReader(self.in_tasks.in_stream)
        #        self.sreader.verify_array_shape = False

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
        self.tasks.ao[name] = task
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

        self.tasks.ci[name] = task

    def create_co_task(self, name: str, chan_spec: dict, clk_cnfg: dict):
        """Doc."""

        task = ni.Task(new_task_name=name)
        task.co_channels.add_co_pulse_chan_ticks(**chan_spec)
        task.timing.cfg_implicit_timing(**clk_cnfg)

        self.tasks.co[name] = task

    def analog_read(self, task_name: str, n_samples, task_type="ai"):
        """Doc."""

        #        # TODO: stream reading currently not working for some reason - reading only one channel, the other two stay at zero
        #        import numpy as np
        #        self.ai_buffer = np.empty(shape=(3, 10000), dtype=np.float)
        #        num_samps_read = self.sreader.read_many_sample(
        #            self.ai_buffer,
        #            number_of_samples_per_channel=ni.constants.READ_ALL_AVAILABLE,
        #        )
        #        return num_samps_read

        ai_task = getattr(self.tasks, task_type)[task_name]
        return ai_task.read(number_of_samples_per_channel=n_samples)

    def analog_write(
        self, task_name: str, data: np.ndarray, auto_start=None
    ) -> NoReturn:
        """Doc."""

        ao_task = self.tasks.ao[task_name]
        if auto_start is not None:
            ao_task.write(data, auto_start=auto_start, timeout=self.AO_TIMEOUT)
        else:
            ao_task.write(data, timeout=self.AO_TIMEOUT)

    def counter_stream_read(self):
        """
         Reads all available samples on board into self.cont_read_buffer
         (1D NumPy array, overwritten each read), and returns the
        number of samples read.
        """

        ci_task = self.tasks.ci["Continuous CI"]
        return ci_task.sr.read_many_sample_uint32(
            self.cont_read_buffer,
            number_of_samples_per_channel=ni.constants.READ_ALL_AVAILABLE,
        )

    def digital_write(self, bool):
        """Doc."""

        with ni.Task() as do_task:
            do_task.do_channels.add_do_chan(self.address)
            do_task.write(bool)


class VisaInstrument:
    """Doc."""

    # TODO: try to free the relevant USB port (COM3?) to avoid getting stuck when trying to run the application when dep is powered ON

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
        self.rm = visa.ResourceManager()

    def _write(self, cmnd: str) -> NoReturn:
        """Sends a command to the VISA instrument."""

        with self.Task(self) as task:
            task.write(cmnd)

    def _query(self, cmnd: str) -> float:
        """Doc."""

        with self.Task(self) as task:
            return float(task.query(cmnd))

    class Task:
        """Doc."""

        def __init__(self, dvc):
            """Doc."""

            self._dvc = dvc

        def __enter__(self):
            """Doc."""

            self._rsrc = self._dvc.rm.open_resource(
                self._dvc.address,
                read_termination=self._dvc.read_termination,
                write_termination=self._dvc.write_termination,
                timeout=1,
                open_timeout=1,
            )
            self._rsrc.query_delay = 0.1
            return self._rsrc

        def __exit__(self, exc_type, exc_value, exc_tb):
            """Doc."""

            self._rsrc.close()


class UC480Instrument:
    """Doc."""

    def __init__(self, param_dict):
        self.error_dict = None
        [setattr(self, key, val) for key, val in param_dict.items()]
        self._inst = None

    def init_cam(self):
        """Doc."""

        try:
            self._inst = UC480_Camera(reopen_policy="new")
        # general exc is due to bad error handeling in instrumental-lib...
        except Exception:
            raise UC480Error(msg="Camera disconnected")

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
