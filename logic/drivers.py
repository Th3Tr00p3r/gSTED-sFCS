# -*- coding: utf-8 -*-
"""Drivers Module."""

import asyncio
from array import array
from typing import NoReturn

import nidaqmx as ni
import pyvisa as visa
from instrumental.drivers.cameras.uc480 import UC480_Camera, UC480Error
from nidaqmx.stream_readers import AnalogMultiChannelReader, CounterReader
from pyftdi.ftdi import Ftdi, FtdiError
from pyftdi.usbtools import UsbTools

from utilities.errors import dvc_err_hndlr as err_hndlr

# TODO: Unite all NI drivers


class FtdiInstrument:
    """Doc."""

    def __init__(self, nick, param_dict):

        self.nick = nick
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

    def __init__(self, nick, param_dict, **kwargs):

        self.nick = nick
        self.error_dict = None
        [setattr(self, key, val) for key, val in {**param_dict, **kwargs}.items()]

    @err_hndlr
    def start_task(self, task):
        """Doc."""

        getattr(self, task).start()
        self.state = True

    @err_hndlr
    def close_task(self, task):
        """Doc."""

        getattr(self, task).close()
        self.state = False

    @err_hndlr
    def create_ai_task(self, type: str):
        """Doc."""

        self.ai_task = ni.Task(new_task_name=f"{type} AI")

        # x-galvo
        self.ai_task.ai_channels.add_ai_voltage_chan(
            physical_channel=self.ai_x_addr,
            name_to_assign_to_channel="aix",
            terminal_config=ni.constants.TerminalConfiguration.RSE,
            min_val=-5.0,
            max_val=5.0,
        )

        # x-galvo
        self.ai_task.ai_channels.add_ai_voltage_chan(
            physical_channel=self.ai_y_addr,
            name_to_assign_to_channel="aiy",
            terminal_config=ni.constants.TerminalConfiguration.RSE,
            min_val=-5.0,
            max_val=5.0,
        )

        # z-piezo
        self.ai_task.ai_channels.add_ai_voltage_chan(
            physical_channel=self.ai_z_addr,
            name_to_assign_to_channel="aiz",
            terminal_config=ni.constants.TerminalConfiguration.RSE,
            min_val=0.0,
            max_val=10.0,
        )

        self.ai_task.timing.cfg_samp_clk_timing(
            # source is unspecified - uses onboard clock (see https://nidaqmx-python.readthedocs.io/en/latest/timing.html#nidaqmx._task_modules.timing.Timing.cfg_samp_clk_timing)
            rate=self.ai_clk_rate,  # TODO - move these to settings -> param_dict
            sample_mode=ni.constants.AcquisitionType.CONTINUOUS,
            samps_per_chan=self.ai_buff_sz,  # TODO - move these to settings -> param_dict
        )

        self.sreader = AnalogMultiChannelReader(self.ai_task.in_stream)
        self.sreader.verify_array_shape = False

    @err_hndlr
    def create_ci_task(self, ai_task, type: str):
        """Doc."""

        self.ci_task = ni.Task(new_task_name=f"{type} CI")

        chan = self.ci_task.ci_channels.add_ci_count_edges_chan(
            counter=self.address,
            edge=ni.constants.Edge.RISING,
            initial_count=0,
            count_direction=ni.constants.CountDirection.COUNT_UP,
        )
        chan.ci_count_edges_term = self.CI_cnt_edges_term

        self.ci_task.timing.cfg_samp_clk_timing(
            rate=ai_task.timing.samp_clk_rate,
            source=ai_task.timing.samp_clk_term,
            sample_mode=ni.constants.AcquisitionType.CONTINUOUS,
            samps_per_chan=len(self.ci_buffer),
        )
        self.ci_task.sr = CounterReader(self.ci_task.in_stream)
        self.ci_task.sr.verify_array_shape = False

    @err_hndlr
    def analog_read(self):
        """Doc."""

        #        # TODO: stream reading currently not working for some reason - reading only one channel, the other two stay at zero
        #        num_samps_read = self.sreader.read_many_sample(
        #            self.ai_buffer,
        #            number_of_samples_per_channel=READ_ALL_AVAILABLE,
        #        )
        #        return num_samps_read

        read_samples = self.ai_task.read(
            number_of_samples_per_channel=ni.constants.READ_ALL_AVAILABLE
        )
        return read_samples

    @err_hndlr
    async def analog_write(
        self, ao_addresses: iter, vals: iter, limits: iter
    ) -> NoReturn:
        """Doc."""

        with ni.Task(new_task_name="ao task") as task:
            for (ao_addr, limits) in zip(ao_addresses, limits):
                task.ao_channels.add_ao_voltage_chan(ao_addr, **limits)
            task.write(vals, timeout=self.ao_timeout)
            # TODO: add task.timing for finite samples - it works without it, but I could try to see if it changes anything
            await asyncio.sleep(self.ao_timeout)

    @err_hndlr
    def counter_read(self):
        """
         Reads all available samples on board into self.ci_buffer
         (1D NumPy array, overwritten each read), and returns the
        number of samples read.
        """

        return self.ci_task.sr.read_many_sample_uint32(
            self.ci_buffer,
            number_of_samples_per_channel=ni.constants.READ_ALL_AVAILABLE,
        )

    @err_hndlr
    def digital_write(self, bool):
        """Doc."""

        with ni.Task() as task:
            task.do_channels.add_do_chan(self.address)
            task.write(bool)
        self.state = bool


class VisaInstrument:
    """Doc."""

    def __init__(
        self,
        nick,
        param_dict,
        read_termination="",
        write_termination="",
    ):

        self.nick = nick
        self.error_dict = None
        [setattr(self, key, val) for key, val in param_dict.items()]
        self.read_termination = read_termination
        self.write_termination = write_termination
        self.rm = visa.ResourceManager()

    @err_hndlr
    def _write(self, cmnd: str) -> bool:
        """
        Sends a command to the VISA instrument.
        Returns True if no errors occured.

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

    def __init__(self, nick, param_dict):

        self.nick = nick
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
