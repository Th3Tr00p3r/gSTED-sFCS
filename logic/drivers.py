# -*- coding: utf-8 -*-
"""Drivers Module."""

import asyncio
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

import utilities.helper as helper
from utilities.errors import dvc_err_hndlr as err_hndlr


class FtdiInstrument:
    """Doc."""

    def __init__(self, param_dict, **kwargs):

        [setattr(self, key, val) for key, val in param_dict.items()]
        self.error_dict = None

        self._inst = Ftdi()  # URL - ftdi://ftdi:232h:FT3TG15/1

    @err_hndlr
    def open(self):
        """Doc."""

        UsbTools.flush_cache()

        self._inst.open(self.vend_id, self.prod_id)
        self._inst.set_bitmode(0, getattr(Ftdi.BitMode, self.bit_mode))
        self._inst.set_latency_timer(self.ltncy_tmr_val)
        self._inst.set_flowctrl(self.flow_ctrl)
        self._inst.read_data_set_chunksize(self.n_bytes)

        self.state = True

    @err_hndlr
    def read(self):
        """Doc."""

        read_bytes = self._inst._usb_dev.read(
            self._inst._out_ep,
            self._inst._readbuffer_chunksize,
        )
        self.check_status(read_bytes[:2])

        return read_bytes[2:]

    @err_hndlr
    def purge(self):
        """Doc."""

        self._inst.purge_rx_buffer()

    @err_hndlr
    def close(self):
        """Doc."""

        self._inst.close()
        self.state = False

    @err_hndlr
    def reset_dvc(self):
        """Doc."""

        self._inst.reset(usb_reset=True)

    @err_hndlr
    def check_status(self, status: array) -> NoReturn:
        """Doc."""

        if status[1] & self._inst.ERROR_BITS[1]:
            s = " ".join(self._inst.decode_modem_status(status, True)).title()
            raise FtdiError(f"FTDI error: {status[0]:02x}:{ status[1]:02x} {s}")


class NIDAQmxInstrument:
    """Doc."""

    MIN_OUTPUT_RATE_Hz = 1000
    # TODO: the following two lines are counter-specific, move them to Counter in devices.py
    CONT_READ_BFFR_SZ = 10000
    cont_read_buffer = np.zeros(shape=(CONT_READ_BFFR_SZ,), dtype=np.uint32)

    def __init__(self, param_dict, **kwargs):
        self.error_dict = None
        # TODO: next line should be in devices.py
        [setattr(self, key, val) for key, val in {**param_dict, **kwargs}.items()]
        self.tasks = SimpleNamespace()
        self.task_types = ["ai", "ao", "ci", "co"]
        [setattr(self.tasks, type, {}) for type in self.task_types]

    @err_hndlr
    def start_tasks(self, task_type: str):
        """Doc."""

        [task.start() for task in getattr(self.tasks, task_type).values()]

    def close_tasks(self, task_type: str):
        """Doc."""

        type_tasks_dict = getattr(self.tasks, task_type)
        [type_tasks_dict[task_name].close() for task_name in type_tasks_dict.keys()]
        setattr(self.tasks, task_type, {})

    @err_hndlr
    def close_all_tasks(self):
        """Doc."""

        for type in self.task_types:
            self.close_tasks(type)

    @err_hndlr
    def wait_for_task(self, task_type: str, task_name: str):
        """Doc."""

        getattr(self.tasks, task_type)[task_name].wait_until_done(timeout=3)

    @err_hndlr
    def create_ai_task(
        self,
        name: str,
        chan_specs: List[dict],
        samp_clk_cnfg: dict,
        clk_params: dict = {},
    ):
        """Doc."""

        task = ni.Task(new_task_name=name)
        for chan_spec in chan_specs:
            # TODO: see if following check can go to devices.py - interrupts possible merge of 'create task' functions
            if helper.count_words(chan_spec["physical_channel"]) == 1:
                chan_spec = {
                    **chan_spec,
                    "terminal_config": ni.constants.TerminalConfiguration.RSE,
                }
            task.ai_channels.add_ai_voltage_chan(**chan_spec)
        task.timing.cfg_samp_clk_timing(**samp_clk_cnfg)
        for key, val in clk_params.items():
            setattr(task.timing, key, val)

        self.tasks.ai[name] = task

        #        # TODO: stream reading currently not working for some reason - reading only one channel, the other two stay at zero
        #        self.sreader = AnalogMultiChannelReader(self.in_tasks.in_stream)
        #        self.sreader.verify_array_shape = False

    @err_hndlr
    def create_ao_task(
        self,
        name: str,
        chan_specs: List[dict],
        samp_clk_cnfg: dict = {},
        clk_params: dict = {},
    ) -> NoReturn:
        """Doc."""

        task = ni.Task(new_task_name=name)
        for chan_spec in chan_specs:
            task.ao_channels.add_ao_voltage_chan(**chan_spec)
        if samp_clk_cnfg:
            task.timing.cfg_samp_clk_timing(**samp_clk_cnfg)
        for key, val in clk_params.items():
            setattr(task.timing, key, val)

        self.tasks.ao[name] = task

    @err_hndlr
    def create_ci_task(
        self,
        name: str,
        chan_spec: dict,
        samp_clk_cnfg: dict,
        chan_xtra_params: dict = {},
        clk_xtra_params: dict = {},
    ):
        """Doc."""

        task = ni.Task(new_task_name=name)
        chan = task.ci_channels.add_ci_count_edges_chan(**chan_spec)
        for key, val in chan_xtra_params.items():
            setattr(chan, key, val)
        task.timing.cfg_samp_clk_timing(**samp_clk_cnfg)
        for key, val in clk_xtra_params.items():
            setattr(task.timing, key, val)

        # TODO: move to seperate function (interrupts merge of 'create task' functions
        task.sr = CounterReader(task.in_stream)
        task.sr.verify_array_shape = False

        self.tasks.ci[name] = task

    @err_hndlr
    def create_co_task(self, name: str, chan_spec: dict, clk_cnfg: dict):
        """Doc."""

        task = ni.Task(new_task_name=name)
        task.co_channels.add_co_pulse_chan_ticks(**chan_spec)
        task.timing.cfg_implicit_timing(**clk_cnfg)

        self.tasks.co[name] = task

    @err_hndlr
    def analog_read(self, task_name: str, n_samples):
        """Doc."""

        #        # TODO: stream reading currently not working for some reason - reading only one channel, the other two stay at zero
        #        import numpy as np
        #        self.ai_buffer = np.empty(shape=(3, 10000), dtype=np.float)
        #        num_samps_read = self.sreader.read_many_sample(
        #            self.ai_buffer,
        #            number_of_samples_per_channel=ni.constants.READ_ALL_AVAILABLE,
        #        )
        #        return num_samps_read

        ai_task = self.tasks.ai[task_name]
        return ai_task.read(number_of_samples_per_channel=n_samples)

    @err_hndlr
    def analog_write(
        self, task_name: str, data: np.ndarray, auto_start=None
    ) -> NoReturn:
        """Doc."""

        ao_task = self.tasks.ao[task_name]
        if auto_start is not None:
            ao_task.write(data, auto_start=auto_start, timeout=self.ao_timeout)
        else:
            ao_task.write(data, timeout=self.ao_timeout)

    @err_hndlr
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

    @err_hndlr
    def digital_write(self, bool):
        """Doc."""

        with ni.Task() as do_task:
            do_task.do_channels.add_do_chan(self.address)
            do_task.write(bool)
        self.state = bool


class VisaInstrument:
    """Doc."""

    def __init__(
        self,
        param_dict,
        read_termination="",
        write_termination="",
        **kwargs,
    ):
        self.error_dict = None
        [setattr(self, key, val) for key, val in param_dict.items()]
        self.read_termination = read_termination
        self.write_termination = write_termination
        self.rm = visa.ResourceManager()

    @err_hndlr
    def _write(self, cmnd: str) -> NoReturn:
        """
        Sends a command to the VISA instrument.
        """

        with self.Task(self) as task:
            task.write(cmnd)

        if cmnd.startswith("setLDenable"):  # change state if toggle is performed
            *_, toggle_val = cmnd
            self.state = bool(int(toggle_val))

    @err_hndlr
    async def _aquery(self, cmnd: str) -> float:
        """Doc."""

        with self.Task(self) as task:
            task.write(cmnd)
            await asyncio.sleep(0.1)
            ans = task.read()
            return float(ans)

    @err_hndlr
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
                timeout=3,
                open_timeout=3,
            )
            self._rsrc.query_delay = 0.1
            return self._rsrc

        def __exit__(self, exc_type, exc_value, exc_tb):
            """Doc."""

            self._rsrc.close()


class UC480Instrument:
    """Doc."""

    def __init__(self, param_dict, **kwargs):
        self.error_dict = None
        [setattr(self, key, val) for key, val in param_dict.items()]
        self._inst = None

    @err_hndlr
    def init_cam(self):
        """Doc."""

        try:  # this is due to bad error handeling in instrumental-lib...
            self._inst = UC480_Camera(reopen_policy="new")
        except Exception:
            raise UC480Error(msg="Camera disconnected")

    @err_hndlr
    def close_cam(self):
        """Doc."""

        if self._inst is not None:
            self._inst.close()

    @err_hndlr
    def grab_image(self):
        """Doc."""

        return self._inst.grab_image()

    @err_hndlr
    def toggle_vid(self, state):
        """Doc."""

        if state is True:
            self._inst.start_live_video()
        else:
            self._inst.stop_live_video()

    @err_hndlr
    def get_latest_frame(self):
        """Doc."""

        frame_ready = self._inst.wait_for_frame(timeout="0 ms")
        if frame_ready:
            return self._inst.latest_frame(copy=False)
