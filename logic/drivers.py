# -*- coding: utf-8 -*-
"""Drivers Module."""

import nidaqmx
import pyvisa as visa
from instrumental.drivers.cameras.uc480 import UC480_Camera  # NOQA
from pyftdi.ftdi import Ftdi

from utilities.errors import driver_error_handler as err_hndlr


class FTDI_Instrument:
    """Doc."""

    def __init__(self, nick, param_dict, error_dict):

        self.nick = nick
        self._param_dict = param_dict
        self.error_dict = error_dict
        self.inst = Ftdi()

    @err_hndlr
    def open(self):
        """Doc."""

        self.inst.open(self._param_dict["vend_id"], self._param_dict["prod_id"])
        self.inst.set_bitmode(0, getattr(Ftdi.BitMode, self._param_dict["bit_mode"]))
        self.inst._usb_read_timeout = self._param_dict["read_timeout"]
        self.inst._usb_write_timeout = self._param_dict["read_timeout"]
        self.inst.set_latency_timer(self._param_dict["ltncy_tmr_val"])
        self.inst.set_flowctrl(self._param_dict["flow_ctrl"])
        self.eff_baud_rate = self.inst.set_baudrate(
            self._param_dict["baud_rate"], constrain=True
        )

        self.state = True

    @err_hndlr
    def read_bytes(self, n_bytes):
        """Doc."""

        return self.inst.read_data_bytes(n_bytes)

    @err_hndlr
    def is_read_error(self):
        """Doc."""

        #        return bool(self.inst.get_cts() ^ self.inst.get_cd())
        pass

    @err_hndlr
    def purge(self):
        """Doc."""

        self.inst.purge_buffers()

    @err_hndlr
    def close(self):
        """Doc."""

        self.inst.close()
        self.state = False


class DAQmxInstrumentDO:
    """Doc."""

    def __init__(self, nick, address, error_dict):

        self.nick = nick
        self._address = address
        self.error_dict = error_dict

        self.toggle(False)

    @err_hndlr
    def _write(self, bool):
        """Doc."""

        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(self._address)
            task.write(bool)

        self.state = bool

    def toggle(self, bool):
        """Doc."""

        self._write(bool)


class DAQmxInstrumentCI:
    """Doc."""

    def __init__(self, nick, param_dict, error_dict):
        """Doc."""

        self.nick = nick
        self._param_dict = param_dict
        self.error_dict = error_dict
        self._task = nidaqmx.Task()
        self._init_chan()

    @err_hndlr
    def _init_chan(self):
        """Doc."""

        chan = self._task.ci_channels.add_ci_count_edges_chan(
            counter=self._param_dict["photon_cntr"],
            edge=nidaqmx.constants.Edge.RISING,
            initial_count=0,
            count_direction=nidaqmx.constants.CountDirection.COUNT_UP,
        )
        chan.ci_count_edges_term = self._param_dict["CI_cnt_edges_term"]

    #        chan.ci_dup_count_prevention = self._params['CI_dup_prvnt']

    @err_hndlr
    def start(self):
        """Doc."""

        self._task.start()
        self.state = True

    @err_hndlr
    def read(self):
        """Doc."""

        counts = self._task.read(number_of_samples_per_channel=-1)[0]

        return counts

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
    def _write(self, cmnd):
        """Doc."""

        with VISAInstrument.Task(self) as task:
            task.write(cmnd)

        if cmnd.startswith("setLDenable"):  # change state if toggled
            self.state = bool(int(cmnd[-1]))

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
                open_timeout=3,
            )
            self._rsrc.query_delay = 0.1
            return self._rsrc

        def __exit__(self, exc_type, exc_value, exc_tb):
            """Doc."""

            self._rsrc.close()
