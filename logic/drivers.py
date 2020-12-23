# -*- coding: utf-8 -*-
"""Drivers Module."""

import asyncio

import nidaqmx
import pyvisa as visa
from instrumental.drivers.cameras.uc480 import UC480_Camera, UC480Error
from pyftdi.ftdi import Ftdi
from pyftdi.usbtools import UsbTools

from utilities.errors import dvc_err_hndlr as err_hndlr


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

        self._init_ai_task()

    @err_hndlr
    def _init_ai_task(self):
        """Doc."""

        self.ai_task = nidaqmx.Task(new_task_name="ai task")

        # x-galvo
        self.ai_task.ai_channels.add_ai_voltage_chan(
            physical_channel=self._param_dict["ai_x_addr"],
            name_to_assign_to_channel="aix",
            terminal_config=nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL,
            min_val=-5.0,
            max_val=5.0,
        )

        # x-galvo
        self.ai_task.ai_channels.add_ai_voltage_chan(
            physical_channel=self._param_dict["ai_y_addr"],
            name_to_assign_to_channel="aiy",
            terminal_config=nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL,
            min_val=-5.0,
            max_val=5.0,
        )

        # z-piezo
        self.ai_task.ai_channels.add_ai_voltage_chan(
            physical_channel=self._param_dict["ai_z_addr"],
            name_to_assign_to_channel="aiz",
            terminal_config=nidaqmx.constants.TerminalConfiguration.RSE,
            min_val=0.0,
            max_val=10.0,
        )

        self.ai_task.timing.cfg_samp_clk_timing(
            # source is unspecified - uses onboard clock (see https://nidaqmx-python.readthedocs.io/en/latest/timing.html#nidaqmx._task_modules.timing.Timing.cfg_samp_clk_timing)
            rate=1000,  # TODO: value taken from PID settings in LabVIEW. turn this into a constant later
            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
            samps_per_chan=1000,  # TODO: value taken from PID settings in LabVIEW. turn this into a constant later
        )

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

        pass

    #        self.ai_task.read()

    @err_hndlr
    def write(self, axis, vltg):
        """Doc."""

        if axis in {"x", "y"}:
            limits = {"min_val": -5.0, "max_val": 5.0}
        else:
            limits = {"min_val": 0.0, "max_val": 10.0}

        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(
                self._param_dict[f"ao_{axis}_addr"], **limits
            )
            task.write(vltg)


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


class DAQmxInstrumentCI:
    """Doc."""

    def __init__(self, nick, param_dict, error_dict, ai_task):
        """Doc."""

        self.nick = nick
        self._param_dict = param_dict
        self.error_dict = error_dict
        self.ai_task = ai_task

        self._init_task()

    @err_hndlr
    def _init_task(self):
        """Doc."""

        self._task = nidaqmx.Task("ci task")

        chan = self._task.ci_channels.add_ci_count_edges_chan(
            counter=self._param_dict["photon_cntr"],
            edge=nidaqmx.constants.Edge.RISING,
            initial_count=0,
            count_direction=nidaqmx.constants.CountDirection.COUNT_UP,
        )
        chan.ci_count_edges_term = self._param_dict["CI_cnt_edges_term"]

        self._task.timing.cfg_samp_clk_timing(
            rate=self.ai_task.timing.samp_clk_rate,
            source=self.ai_task.timing.samp_clk_term,
            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
            samps_per_chan=self._param_dict["buff_sz"],
        )

    @err_hndlr
    def start(self):
        """Doc."""

        self._task.start()
        self.state = True

    @err_hndlr
    def read(self):
        """Doc."""

        return self._task.read(
            number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE,
        )[0]

    @err_hndlr
    def close(self):
        """Doc."""

        self._task.close()
        self.state = False


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
        """Doc."""

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

        def __init__(self, inst):
            """Doc."""

            self._inst = inst

        def __enter__(self):
            """Doc."""

            self._rsrc = self._inst.rm.open_resource(
                self._inst.address,
                read_termination=self._inst.read_termination,
                write_termination=self._inst.write_termination,
                timeout=3,
                open_timeout=3,
            )
            self._rsrc.query_delay = 0.1
            return self._rsrc

        def __exit__(self, exc_type, exc_value, exc_tb):
            """Doc."""

            self._rsrc.close()
