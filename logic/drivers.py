# -*- coding: utf-8 -*-
"""Drivers Module."""

import asyncio

import nidaqmx
import numpy as np
import pyvisa as visa
from instrumental.drivers.cameras.uc480 import UC480_Camera, UC480Error
from nidaqmx.constants import (
    READ_ALL_AVAILABLE,
    AcquisitionType,
    CountDirection,
    Edge,
    TerminalConfiguration,
)
from nidaqmx.stream_readers import AnalogMultiChannelReader, CounterReader
from pyftdi.ftdi import Ftdi
from pyftdi.usbtools import UsbTools

from utilities.errors import dvc_err_hndlr as err_hndlr


class FTDI_Instrument:
    """Doc."""

    def __init__(self, nick, param_dict, error_dict):

        self.nick = nick
        self._param_dict = param_dict
        self.error_dict = error_dict
        self._inst = Ftdi()

    @err_hndlr
    def open(self):
        """Doc."""

        UsbTools.flush_cache()

        #        self._inst.open_bitbang(
        #            vendor=self._param_dict["vend_id"],
        #            product=self._param_dict["prod_id"],
        #            latency=self._param_dict["ltncy_tmr_val"],
        #            baudrate = 256000,
        #            sync=True)
        #        self._inst._usb_read_timeout = self._param_dict["read_timeout"]
        #        self._inst.set_flowctrl(self._param_dict["flow_ctrl"])

        self._inst.open(self._param_dict["vend_id"], self._param_dict["prod_id"])
        self._inst.set_bitmode(0, getattr(Ftdi.BitMode, self._param_dict["bit_mode"]))
        self._inst._usb_read_timeout = self._param_dict["read_timeout"]
        self._inst.set_latency_timer(self._param_dict["ltncy_tmr_val"])
        self._inst.set_flowctrl(self._param_dict["flow_ctrl"])
        self.eff_baud_rate = self._inst.set_baudrate(
            self._param_dict["baud_rate"], constrain=True
        )

        self.state = True

    def read(self):
        """Doc."""

        return self._inst.read_data_bytes(self._param_dict["n_bytes"])

    @err_hndlr
    def is_read_error(self):
        """Doc."""

        #        return bool(self._inst.get_cts() ^ self._inst.get_cd())
        pass

    @err_hndlr
    def purge(self):
        """Doc."""

        self._inst.purge_rx_buffer()

    @err_hndlr
    def close(self):
        """Doc."""

        self._inst.close()
        self.state = False


class DAQmxInstrumentAIO:
    """Doc."""

    def __init__(self, nick, param_dict, error_dict):

        self.nick = nick
        self._param_dict = param_dict
        self.error_dict = error_dict
        self.read_buffer = np.zeros(shape=(3, 5000), dtype=np.double)

        self._init_ai_task()

    @err_hndlr
    def _init_ai_task(self):
        """Doc."""

        self.ai_task = nidaqmx.Task(new_task_name="ai task")

        # x-galvo
        self.ai_task.ai_channels.add_ai_voltage_chan(
            physical_channel=self._param_dict["ai_x_addr"],
            name_to_assign_to_channel="aix",
            terminal_config=TerminalConfiguration.DIFFERENTIAL,
            min_val=-5.0,
            max_val=5.0,
        )

        # x-galvo
        self.ai_task.ai_channels.add_ai_voltage_chan(
            physical_channel=self._param_dict["ai_y_addr"],
            name_to_assign_to_channel="aiy",
            terminal_config=TerminalConfiguration.DIFFERENTIAL,
            min_val=-5.0,
            max_val=5.0,
        )

        # z-piezo
        self.ai_task.ai_channels.add_ai_voltage_chan(
            physical_channel=self._param_dict["ai_z_addr"],
            name_to_assign_to_channel="aiz",
            terminal_config=TerminalConfiguration.RSE,
            min_val=0.0,
            max_val=10.0,
        )

        self.ai_task.timing.cfg_samp_clk_timing(
            # source is unspecified - uses onboard clock (see https://nidaqmx-python.readthedocs.io/en/latest/timing.html#nidaqmx._task_modules.timing.Timing.cfg_samp_clk_timing)
            rate=self.ai_clk_rate,  # TODO - move these to settings -> param_dict
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self.buff_sz,  # TODO - move these to settings -> param_dict
        )

        self.ai_task.sr = AnalogMultiChannelReader(self.ai_task.in_stream)
        self.ai_task.sr.verify_array_shape = False

    @err_hndlr
    def start_ai(self):
        """Doc."""

        self.ai_task.start()
        self.ai_state = True

    @err_hndlr
    def close_ai(self):
        """Doc."""

        self.ai_task.close()
        self.ai_state = False

    @err_hndlr
    def read(self):
        """Doc."""

        # TODO: possibly switch to multiple samples (read buffer) as in labview, to have the option to plot and compare to AO (which also needs to be adapted to save its values in a buffer)

        num_samps_read = self.ai_task.sr.read_many_sample(
            self.read_buffer,
            number_of_samples_per_channel=READ_ALL_AVAILABLE,
        )
        return num_samps_read

    @err_hndlr
    async def write(self, ao_addrs: iter, vals: iter, limits: iter):
        """Doc."""

        with nidaqmx.Task() as task:
            for (ao_addr, limits) in zip(ao_addrs, limits):
                task.ao_channels.add_ao_voltage_chan(ao_addr, **limits)
            task.write(vals, timeout=self.ao_timeout)
            await asyncio.sleep(self.ao_timeout)


class DAQmxInstrumentCI:
    """Doc."""

    def __init__(self, nick, param_dict, error_dict, ai_task):
        """Doc."""

        self.nick = nick
        self._param_dict = param_dict
        self.error_dict = error_dict
        self.ai_task = ai_task
        self.read_buffer = np.zeros(shape=(5000,), dtype=np.uint32)

        self._init_task()

    @err_hndlr
    def _init_task(self):
        """Doc."""

        self._task = nidaqmx.Task("ci task")

        chan = self._task.ci_channels.add_ci_count_edges_chan(
            counter=self._param_dict["photon_cntr"],
            edge=Edge.RISING,
            initial_count=0,
            count_direction=CountDirection.COUNT_UP,
        )
        chan.ci_count_edges_term = self._param_dict["CI_cnt_edges_term"]

        self._task.timing.cfg_samp_clk_timing(
            rate=self.ai_task.timing.samp_clk_rate,
            source=self.ai_task.timing.samp_clk_term,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self._param_dict["buff_sz"],
        )
        self._task.sr = CounterReader(self._task.in_stream)
        self._task.sr.verify_array_shape = False

    @err_hndlr
    def start(self):
        """Doc."""

        self._task.start()
        self.state = True

    @err_hndlr
    def read(self):
        """Doc."""

        num_samps_read = self._task.sr.read_many_sample_uint32(
            self.read_buffer,
            number_of_samples_per_channel=READ_ALL_AVAILABLE,
        )
        return num_samps_read

    @err_hndlr
    def close(self):
        """Doc."""

        self._task.close()
        self.state = False


class DAQmxInstrumentDO:
    """Doc."""

    def __init__(self, nick, address, error_dict):

        self.nick = nick
        self._address = address
        self.error_dict = error_dict

    @err_hndlr
    def write(self, bool):
        """Doc."""

        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(self._address)
            task.write(bool)

        self.state = bool


class VISAInstrument:
    """Doc."""

    def __init__(
        self,
        nick,
        address,
        error_dict,
        read_termination="",
        write_termination="",
    ):

        self.nick = nick
        self.address = address
        self.error_dict = error_dict
        self.read_termination = read_termination
        self.write_termination = write_termination
        self.rm = visa.ResourceManager()

    @err_hndlr
    def _write(self, cmnd: str) -> bool:
        """
        Sends a command to the VISA instrument.
        Returns True if no errors occured.

        """

        with VISAInstrument.Task(self) as task:
            task.write(cmnd)

        if cmnd.startswith("setLDenable"):  # change state if toggle is performed
            *_, toggle_val = cmnd
            self.state = bool(int(toggle_val))

    @err_hndlr
    async def _aquery(self, cmnd) -> float:
        """Doc."""

        with VISAInstrument.Task(self) as task:
            task.write(cmnd)
            await asyncio.sleep(0.1)
            ans = task.read()
            return float(ans)

    @err_hndlr
    def _query(self, cmnd):
        """Doc."""

        with VISAInstrument.Task(self) as task:
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

    def __init__(self, nick, error_dict):
        self.nick = nick
        self.error_dict = error_dict
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
    def toggle_vid(self, bool):
        """Doc."""

        self.vid_state = bool

        if bool:
            self._inst.start_live_video()
        else:
            self._inst.stop_live_video()

    @err_hndlr
    def get_latest_frame(self):
        """Doc."""

        frame_ready = self._inst.wait_for_frame(timeout="0 ms")
        if frame_ready:
            return self._inst.latest_frame(copy=False)
