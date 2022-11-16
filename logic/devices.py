"""Devices Module."""

import asyncio
import logging
import sys
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from string import ascii_letters, digits
from typing import List, Tuple, Union

import nidaqmx.constants as ni_consts
import numpy as np
import PIL
from matplotlib.patches import Ellipse
from nidaqmx.errors import DaqError

from gui.widgets import QtWidgetAccess, QtWidgetCollection
from logic.drivers import Ftd2xx, Instrumental, NIDAQmx, PyVISA
from logic.timeout import TIMEOUT_INTERVAL
from utilities.dialog import ErrorDialog
from utilities.errors import DeviceCheckerMetaClass, DeviceError, IOError, err_hndlr
from utilities.fit_tools import FitError, fit_2d_gaussian_to_image, linear_fit
from utilities.helper import (
    Limits,
    deep_getattr,
    div_ceil,
    generate_numbers_from_string,
    str_to_num,
)


@dataclass
class DeviceAttrs:
    log_ref: str
    param_widgets: QtWidgetCollection
    led_color: str = "green"
    synced_dvc_attrs: List[Tuple[str, str]] = None


class BaseDevice:
    """Doc."""

    log_ref: str

    def __init__(self, attrs, app, **kwargs):

        self.icon_dict = app.icon_dict

        param_dict = attrs.param_widgets.hold_widgets(app.gui).gui_to_dict(app.gui)
        param_dict["log_ref"] = attrs.log_ref
        param_dict["led_icon"] = app.icon_dict[f"led_{attrs.led_color}"]
        param_dict["error_display"] = QtWidgetAccess(
            "deviceErrorDisplay", "QLineEdit", "main", True
        ).hold_widget(app.gui.main)

        if attrs.synced_dvc_attrs:
            for attr_name, deep_attr in attrs.synced_dvc_attrs:
                param_dict[attr_name] = deep_getattr(app, deep_attr)

        self.error_dict = None
        self._app_loop = app.loop
        self._app_gui = app.gui
        super().__init__(param_dict, **kwargs)
        self.led_widget: QtWidgetAccess
        self.switch_widget: QtWidgetAccess

    def change_icons(self, command, **kwargs):
        """Doc."""

        def set_icon_wdgts(
            led_icon, has_switch: bool, switch_icon=None, led_widget_name="led_widget"
        ):
            """Doc."""

            getattr(self, led_widget_name).set(led_icon)
            if has_switch:
                self.switch_widget.set(switch_icon)

        has_switch = hasattr(self, "switch_widget")
        if command == "on":
            set_icon_wdgts(self.led_icon, has_switch, self.icon_dict["switch_on"], **kwargs)
        elif command == "off":
            set_icon_wdgts(
                self.icon_dict["led_off"], has_switch, self.icon_dict["switch_off"], **kwargs
            )
        elif command == "error":
            set_icon_wdgts(self.icon_dict["led_red"], False, **kwargs)

    def close(self):
        """Defualt device close(). Some classes override this."""

        self.toggle(False)

    def toggle_led_and_switch(self, is_being_switched_on: bool, should_change_icons=True):
        """Toggle the device's LED widget ON/OFF"""

        if not self.error_dict and should_change_icons:
            self.change_icons("on" if is_being_switched_on else "off")

    def disable_switch(self) -> None:
        """Disable the GUI switch for the device (used to avoid further errors)"""

        with suppress(AttributeError):  # Device has no switch
            self.switch_widget.obj.setEnabled(False)
            self.switch_widget.obj.setToolTip(
                f"{self.log_ref} switch is disabled due to device error. Press LED for details."
            )

    def enable_switch(self) -> None:
        """Disable the GUI switch for the device (used to avoid further errors)"""

        with suppress(AttributeError):  # Device has no switch
            self.switch_widget.obj.setEnabled(True)
            self.switch_widget.obj.setToolTip(f"Toggle {self.log_ref}.")


class SimpleDODevice(BaseDevice, NIDAQmx, metaclass=DeviceCheckerMetaClass):
    """ON/OFF NIDAQmx-controlled device (excitation laser, depletion shutter, TDC)."""

    off_timer_min = 60

    def __init__(self, *args):
        super().__init__(
            *args,
            task_types=("do",),
        )
        self.turn_on_time = None
        with suppress(DeviceError):
            self.toggle(False)

    def toggle(self, is_being_switched_on):
        """Doc."""

        try:
            self.digital_write(self.address, is_being_switched_on)
        except DaqError:
            exc = IOError(
                f"NI device address ({self.address}) is wrong, or data acquisition board is unplugged"
            )
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.toggle_led_and_switch(is_being_switched_on)
            self.is_on = is_being_switched_on
            self.turn_on_time = time.perf_counter() if is_being_switched_on else None


class FastGatedSPAD(BaseDevice, Ftd2xx, metaclass=DeviceCheckerMetaClass):
    """Doc."""

    attrs = DeviceAttrs(
        log_ref="Fast-Gated SPAD",
        param_widgets=QtWidgetCollection(
            led_widget=("ledSPAD", "QIcon", "main", True),
            gate_led_widget=("ledGate", "QIcon", "main", True),
            set_gate_width_wdgt=("spadGateWidth", "QSpinBox", "main", True),
            eff_gate_width_wdgt=("spadEffGateWidth", "QSpinBox", "main", True),
            mode=("spadMode", "QLineEdit", "main", True),
            temperature_c=("spadTemp", "QDoubleSpinBox", "main", True),
            description=("spadDescription", "QLineEdit", "settings", False),
            baud_rate=("spadBaudRate", "QSpinBox", "settings", False),
            timeout_ms=("spadTimeout", "QSpinBox", "settings", False),
            laser_freq_mhz=("TDClaserFreq", "QSpinBox", "settings", False),
        ),
    )

    update_interval_s = 1
    gate_width_limits = Limits(10000, 98000)

    code_attr_dict = {
        "VPOL": "is_on",
        "FR": "mode",
        "ERR": "error",
        "VT": "input_thresh_mv",
        "SE": "pulse_edge",
        "TS": "trigger_select",
        "GF": "gate_freq_hz",
        "TO": "gate_width_ps",
        "TK": "temperature",
        "HO": "hold_off",
        "CT": "avalanch_thresh",
        "CE": "excess_bias",
        "CC": "current",
    }

    # NOTE: see jupyter notebook ('Getting SPAD settings calibration using linear fits') for details on how these were measured
    lin_fit_consts_dict = {
        "hold_off_ns": (3.0, 18.0),
        "avalanch_thresh_mv": (-2.0, 400.0),
        "excess_bias_v": (-0.02409447, 6.9712588),
        "current_ma": (-0.45454545, 115.45454545),
    }

    def __init__(self, app):
        super().__init__(
            self.attrs,
            app,
        )
        self.is_on = False
        try:
            self.toggle(True)
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.purge(True)
            self._app_loop.create_task(self.toggle_mode("free running"))

            self.gate_ns = None
            self.is_paused = False  # used when ceding control to MPD interface
            self.settings = {}

    def toggle(self, should_turn_on: bool):
        """Doc."""

        if should_turn_on:
            self.open_instrument()
        else:
            self.close_instrument()
        self.toggle_led_and_switch(should_turn_on)

    def pause(self, should_pause: bool) -> None:
        """Doc."""

        try:
            self.is_paused = should_pause
            self.toggle(not should_pause)
        except IOError as exc:
            self.is_paused = not should_pause
            if not exc.__str__() == f"{self.log_ref} MPD interface not closed!":
                err_hndlr(exc, sys._getframe(), locals(), dvc=self)

    async def get_stats(self) -> None:
        """Doc."""

        was_on = self.is_on

        try:
            responses, command = await self.mpd_command([("AQ", None), ("DC", None)])
        except ValueError:
            print(f"{self.log_ref} did not respond to stats query ('{command}')")
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            # get a dictionary of read values instead of codes
            with suppress(TypeError, KeyError):
                stats_dict = {
                    self.code_attr_dict[str_.rstrip(digits + "-")]: str_to_num(
                        str_.lstrip(ascii_letters)
                    )
                    for str_ in responses
                    if str_.startswith(tuple(self.code_attr_dict.keys()))
                }

                # raise error if one is encountered in the responses
                if stats_dict.get("error"):
                    self.change_icons("error")
                    exc_ = DeviceError(f"{self.log_ref} error number {stats_dict['error']}.")
                    err_hndlr(exc_, sys._getframe(), locals(), dvc=self)

                # set attributes based on 'stats_dict'
                self.is_on = bool(stats_dict["is_on"])
                self.settings["mode"] = "free running" if stats_dict["mode"] == 1 else "external"
                self.settings["input_thresh_mv"] = stats_dict["input_thresh_mv"]
                self.settings["pulse_edge"] = "falling" if stats_dict["mode"] == 1 else "rising"
                self.settings["gate_freq_mhz"] = stats_dict["gate_freq_hz"] * 1e-6
                self.settings["gate_width_ns"] = stats_dict["gate_width_ps"] * 1e-3
                self.settings["temperature_c"] = stats_dict["temperature"] / 10 - 273

                # The following properties are derived from fitting set values (in MPD interface) to obtained values from 'DC' command,
                # and using said fits to estimate the actual setting (returned values are module-specific)
                for attr_name, beta in self.lin_fit_consts_dict.items():
                    _, response_name_flipped = attr_name[::-1].split("_", maxsplit=1)
                    response_name = response_name_flipped[::-1]
                    calc_value = round(linear_fit(stats_dict[response_name], *beta))
                    self.settings[attr_name] = calc_value

                # Write to widgets
                self.attrs.param_widgets.obj_to_gui(self._app_gui, self.settings)

        if was_on != self.is_on:
            self.toggle_led_and_switch(self.is_on)

    async def toggle_mode(self, mode: str) -> None:
        """Doc."""

        mode_cmnd_dict = {"free running": "FR", "external": "GM"}
        with suppress(IOError):  # TESTESTEST
            await self.mpd_command(
                [("TS0", Limits(0, 1)), (mode_cmnd_dict[mode], None), ("AD", None)]
            )

    async def set_gate_width(self, gate_width_ns=None):
        """Doc."""

        if gate_width_ns is None:  # Manually set from GUI (needed?)
            gate_width_ps = int(self.set_gate_width_wdgt.get() * 1e3)
        else:
            gate_width_ps = gate_width_ns * 1e3

        response, _ = await self.mpd_command(
            [
                (f"TO{gate_width_ps}", self.gate_width_limits),
                ("AD", None),
            ]
        )

        # handle weird stuff happening with what seems to be mixing of responses (from get_stats)
        if isinstance(response, list):
            response = response[0]

        with suppress(TypeError):  # response is empty?
            effective_gatewidth_ns = int(next(generate_numbers_from_string(response)) * 1e-3)
            self.eff_gate_width_wdgt.set(effective_gatewidth_ns)


class PicoSecondDelayer(BaseDevice, Ftd2xx, metaclass=DeviceCheckerMetaClass):
    """Doc."""

    attrs = DeviceAttrs(
        log_ref="Pico-Second Delayer",
        param_widgets=QtWidgetCollection(
            led_widget=("ledPSD", "QIcon", "main", True),
            switch_widget=("psdSwitch", "QIcon", "main", True),
            set_delay_wdgt=("psdDelay", "QSpinBox", "main", True),
            eff_delay_wdgt=("psdEffDelay", "QSpinBox", "main", True),
            description=("psdDescription", "QLineEdit", "settings", False),
            baud_rate=("psdBaudRate", "QSpinBox", "settings", False),
            timeout_ms=("psdTimeout", "QSpinBox", "settings", False),
            threshold_mV=("psdThreshold_mV", "QSpinBox", "settings", False),
            freq_divider=("psdFreqDiv", "QSpinBox", "settings", False),
            # NOTE: sync_delay_ns was by measuring a detector-gated sample, getting its actual delay by syncing to the laser sample's (below) TDC calibration
            # NOTE: the sync delay might be better calibrated by setting the delay so that there is no change in countrate when switching from free-running mode to gated mode
            sync_delay_ns=("syncDelay", "QDoubleSpinBox", "settings", True),
        ),
    )

    update_interval_s = 1

    psd_delay_limits_ps = Limits(1, 49090)
    pulsewidth_limits_ns = Limits(1, 250)

    def __init__(self, app):
        super().__init__(self.attrs, app)
        try:
            self.open_instrument()
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.purge(True)
            self.settings = dict(sync_delay_ns=self.sync_delay_ns)
            try:
                self._app_loop.create_task(
                    self.mpd_command(
                        [
                            ("EM0", Limits(0, 1)),  # cancel echo mode
                            (f"SH{self.threshold_mV}", Limits(-2000, 2000)),  # set input threshold
                            (f"SV{self.freq_divider}", Limits(1, 999)),  # set frequency divider
                        ]
                    )
                )
            except ValueError as exc:
                exc = IOError(f"{self.log_ref} did not respond to initialization commands [{exc}]")
                err_hndlr(exc, sys._getframe(), locals(), dvc=self)
                self.effective_delay_ns = 0

        self.is_on = False

    async def toggle(self, is_being_switched_on: bool):
        """Doc."""

        try:
            await self.mpd_command(
                (f"EO{int(is_being_switched_on)}", Limits(0, 1))
            )  # enable/disable output
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.toggle_led_and_switch(is_being_switched_on)
            self.is_on = is_being_switched_on

    async def close(self):
        """Doc."""

        await self.toggle(False)
        await self.mpd_command(("EM1", Limits(0, 1)))  # return to echo mode (for MPD software)
        self.close_instrument()

    async def set_lower_gate(self, lower_gate_ns: float):
        """
        Set the delay according to the chosen lower gate in ns.
        This is achieved by a combination of setting the pulse width as a mechanism of coarse delay,
        and on top of that setting the signal delay in picoseconds.
        The value for each is automatically found by calibrating ahead of time the delay needed
        for syncronizing the laser pulse with the fluorescence pulse.
        (See the 'Laser Propagation Time Calibration' Jupyter Notebook)
        """

        try:
            req_total_delay_ns = self.sync_delay_ns.get() + lower_gate_ns

            # ensure pulsewidth step doesn't go past 'req_delay_ns' by subtracting 5 ns (steps are 3 or 4 ns)
            # yet stays close so all relevant gates are still reachable with the PSD delay (up to ~50 ns)
            req_pulsewidth_ns = round((req_total_delay_ns - 5))
            response, _ = await self.mpd_command(
                (f"SP{req_pulsewidth_ns}", self.pulsewidth_limits_ns)
            )
            pulsewidth_ns = int(response)

            req_psd_delay_ps = round((req_total_delay_ns - pulsewidth_ns) * 1e3)
            response, _ = await self.mpd_command(
                (f"SD{req_psd_delay_ps}", self.psd_delay_limits_ps)
            )
            psd_delay_ps = int(response)
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            effective_delay_ns = pulsewidth_ns + psd_delay_ps * 1e-3

            self.settings["pulsewidth_ns"] = pulsewidth_ns
            self.settings["psd_delay_ns"] = psd_delay_ps * 1e-3
            self.settings["effective_delay_ns"] = effective_delay_ns

            # Show delay in GUI
            self.eff_delay_wdgt.set(effective_delay_ns)


class TDC(SimpleDODevice):
    """Controls time-to-digital converter ON/OFF switch"""

    attrs = DeviceAttrs(
        log_ref="TDC",
        param_widgets=QtWidgetCollection(
            led_widget=("ledTdc", "QIcon", "main", True),
            address=("TDCaddress", "QLineEdit", "settings", False),
            data_vrsn=("TDCdataVersion", "QLineEdit", "settings", False),
            laser_freq_mhz=("TDClaserFreq", "QSpinBox", "settings", False),
            fpga_freq_mhz=("TDCFPGAFreq", "QSpinBox", "settings", False),
            tdc_vrsn=("TDCversion", "QSpinBox", "settings", False),
        ),
    )

    def __init__(self, app):
        super().__init__(self.attrs, app)


class UM232H(BaseDevice, Ftd2xx, metaclass=DeviceCheckerMetaClass):
    """
    Represents the FTDI chip used to transfer data from the FPGA
    to the PC.
    """

    attrs = DeviceAttrs(
        log_ref="UM232H",
        param_widgets=QtWidgetCollection(
            led_widget=("ledUm232h", "QIcon", "main", True),
            description=("um232Description", "QLineEdit", "settings", False),
            bit_mode=("um232BitMode", "QLineEdit", "settings", False),
            timeout_ms=("um232Timeout", "QSpinBox", "settings", False),
            ltncy_tmr_val=("um232LatencyTimerVal", "QSpinBox", "settings", False),
            flow_ctrl=("um232FlowControl", "QLineEdit", "settings", False),
            tx_size=("um232TxSize", "QSpinBox", "settings", False),
            n_bytes=("um232NumBytes", "QSpinBox", "settings", False),
        ),
    )

    def __init__(self, app):
        super().__init__(
            self.attrs,
            app,
        )
        try:
            self.open_instrument()
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.init_data()
            self.purge()

    def close(self):
        """Doc."""

        self.close_instrument()

    async def read_TDC(self, async_read=False):
        """Doc."""

        if async_read:
            byte_array = await self.async_read()
        else:
            byte_array = self.read()
            await asyncio.sleep(TIMEOUT_INTERVAL)

        self.data += byte_array
        self.tot_bytes_read += len(byte_array)

    def init_data(self):
        """Doc."""

        self.data = []
        self.tot_bytes_read = 0

    def purge_buffers(self):
        """Doc."""

        self.purge()


class Scanners(BaseDevice, NIDAQmx, metaclass=DeviceCheckerMetaClass):
    """
    Scanners encompasses all analog focal point positioning devices
    (X: x_galvo, Y: y_galvo, Z: z_piezo)
    """

    # TODO: for better modularity and simplicity, create a seperate class for each galvo and the piezo seperately, and have this class control all of them

    attrs = DeviceAttrs(
        log_ref="Scanners",
        param_widgets=QtWidgetCollection(
            led_widget=("ledScn", "QIcon", "main", True),
            ao_x_init_vltg=("xAOV", "QDoubleSpinBox", "main", False),
            ao_y_init_vltg=("yAOV", "QDoubleSpinBox", "main", False),
            ao_z_init_vltg=("zAOV", "QDoubleSpinBox", "main", False),
            x_origin=("xOrigin", "QDoubleSpinBox", "settings", False),
            y_origin=("yOrigin", "QDoubleSpinBox", "settings", False),
            z_origin=("zOrigin", "QDoubleSpinBox", "settings", False),
            x_um2v_const=("xConv", "QDoubleSpinBox", "settings", True),
            y_um2v_const=("yConv", "QDoubleSpinBox", "settings", True),
            z_um2v_const=("zConv", "QDoubleSpinBox", "settings", True),
            ai_x_addr=("AIXaddr", "QLineEdit", "settings", False),
            ai_y_addr=("AIYaddr", "QLineEdit", "settings", False),
            ai_z_addr=("AIZaddr", "QLineEdit", "settings", False),
            ai_laser_mon_addr=("AIlaserMonAddr", "QLineEdit", "settings", False),
            ai_clk_div=("AIclkDiv", "QSpinBox", "settings", False),
            ai_trg_src=("AItrigSrc", "QLineEdit", "settings", False),
            ao_x_addr=("AOXaddr", "QLineEdit", "settings", False),
            ao_y_addr=("AOYaddr", "QLineEdit", "settings", False),
            ao_z_addr=("AOZaddr", "QLineEdit", "settings", False),
            ao_int_x_addr=("AOXintAddr", "QLineEdit", "settings", False),
            ao_int_y_addr=("AOYintAddr", "QLineEdit", "settings", False),
            ao_int_z_addr=("AOZintAddr", "QLineEdit", "settings", False),
            ao_dig_trg_src=("AOdigTrigSrc", "QLineEdit", "settings", False),
            ao_trg_edge=("AOtriggerEdge", "QComboBox", "settings", False),
            ao_wf_type=("AOwfType", "QComboBox", "settings", False),
        ),
    )

    AXIS_INDEX = {"x": 0, "y": 1, "z": 2, "X": 0, "Y": 1, "Z": 2}
    AXES_TO_BOOL_TUPLE_DICT = {
        "X": (True, False, False),
        "Y": (False, True, False),
        "Z": (False, False, True),
        "XY": (True, True, False),
        "XZ": (True, False, True),
        "YZ": (False, True, True),
        "XYZ": (True, True, True),
    }
    X_AO_LIMITS = Limits(-5.0, 5.0, ("min_val", "max_val"))
    Y_AO_LIMITS = Limits(-5.0, 5.0, ("min_val", "max_val"))
    Z_AO_LIMITS = Limits(0.0, 10.0, ("min_val", "max_val"))
    AI_BUFFER_SIZE = int(1e4)

    def __init__(self, app):
        super().__init__(
            self.attrs,
            app,
            task_types=("ai", "ao"),
        )
        RSE = ni_consts.TerminalConfiguration.RSE
        DIFF = ni_consts.TerminalConfiguration.DIFFERENTIAL
        self.ai_chan_specs = [
            {
                "physical_channel": getattr(self, f"ai_{axis}_addr"),
                "name_to_assign_to_channel": f"{axis}-{inst} AI",
                "min_val": -10.0,
                "max_val": 10.0,
                "terminal_config": RSE,
            }
            for axis, inst in zip("xyz", ("galvo", "galvo", "piezo"))
        ]
        self.ao_int_chan_specs = [
            {
                "physical_channel": getattr(self, f"ao_int_{axis}_addr"),
                "name_to_assign_to_channel": f"{axis}-{inst} internal AO",
                **getattr(self, f"{axis.upper()}_AO_LIMITS").as_dict(),
                "terminal_config": trm_cnfg,
            }
            for axis, trm_cnfg, inst in zip("xyz", (DIFF, DIFF, RSE), ("galvo", "galvo", "piezo"))
        ]
        self.ao_chan_specs = [
            {
                "physical_channel": getattr(self, f"ao_{axis}_addr"),
                "name_to_assign_to_channel": f"{axis}-{inst} AO",
                **getattr(self, f"{axis.upper()}_AO_LIMITS").as_dict(),
            }
            for axis, inst in zip("xyz", ("galvo", "galvo", "piezo"))
        ]
        self.origin = tuple(getattr(self, f"{ax}_origin") for ax in "xyz")
        self.ai_buffer: Union[list, deque]

        with suppress(DeviceError):
            self.start_continuous_read_task()

    @property
    def um_v_ratio(self):
        """Get this property from the GUI dynamically"""

        return tuple(getattr(self, f"{ax}_um2v_const").get() for ax in "xyz")

    def close(self):
        """Doc."""

        self.close_all_tasks()

    def start_continuous_read_task(self) -> None:
        """Doc."""

        try:
            self.close_tasks("ai")
            self.create_ai_task(
                name="Continuous AI",
                address=self.ai_x_addr,
                chan_specs=self.ai_chan_specs + self.ao_int_chan_specs,
                samp_clk_cnfg={
                    "rate": self.MIN_OUTPUT_RATE_Hz,
                    "sample_mode": ni_consts.AcquisitionType.CONTINUOUS,
                    "samps_per_chan": self.CONT_READ_BFFR_SZ,
                },
            )
            self.init_ai_buffer()
            self.start_tasks("ai")
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.toggle_led_and_switch(False)

    def start_scan_read_task(self, samp_clk_cnfg, timing_params) -> None:
        """Doc."""

        try:
            self.close_tasks("ai")
            self.create_ai_task(
                name="Continuous AI",
                address=self.ai_x_addr,
                chan_specs=self.ai_chan_specs + self.ao_int_chan_specs,
                samp_clk_cnfg=samp_clk_cnfg,
                timing_params=timing_params,
            )
            self.start_tasks("ai")
        except DaqError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)

    def start_write_task(
        self,
        ao_data: np.ndarray,
        type: str,
        samp_clk_cnfg_xy: dict = {},
        samp_clk_cnfg_z: dict = {},
        start=True,
    ) -> None:
        """Doc."""

        def smooth_start(
            axis: str, ao_chan_specs: list, final_pos: float, step_sz: float = 0.25
        ) -> None:
            """Doc."""
            # NOTE: Ask Oleg why we used 40 steps in LabVIEW (this is why I use a step size of 10/40 V)

            try:
                init_pos = self.ai_buffer[-1][3:][self.AXIS_INDEX[axis]]
            except IndexError:
                init_pos = self.last_int_ao[self.AXIS_INDEX[axis]]

            total_dist = abs(final_pos - init_pos)
            n_steps = div_ceil(total_dist, step_sz)

            if n_steps < 2:  # one small step needs no smoothing
                return

            else:
                ao_data = np.linspace(init_pos, final_pos, n_steps)
                # move
                task_name = "Smooth AO Z"
                try:
                    self.close_tasks("ao")
                    ao_task = self.create_ao_task(
                        name=task_name,
                        chan_specs=ao_chan_specs,
                        samp_clk_cnfg={
                            "rate": self.MIN_OUTPUT_RATE_Hz,  # WHY? see CreateAOTask.vi
                            "samps_per_chan": n_steps,
                            "sample_mode": ni_consts.AcquisitionType.FINITE,
                        },
                    )
                    ao_data = self._limit_ao_data(ao_task, ao_data)
                    self.analog_write(task_name, ao_data, auto_start=False)
                    self.start_tasks("ao")
                    self.wait_for_task("ao", task_name)
                    self.close_tasks("ao")
                except Exception as exc:
                    err_hndlr(exc, sys._getframe(), locals(), dvc=self)

        axes_to_use = self.AXES_TO_BOOL_TUPLE_DICT[type]

        xy_chan_spcs = []
        z_chan_spcs = []
        ao_data_xy = np.empty(shape=(0,))
        ao_data_row_idx = 0
        for ax, is_ax_used, ax_chn_spcs in zip("XYZ", axes_to_use, self.ao_chan_specs):
            if is_ax_used:
                if ax in "XY":
                    xy_chan_spcs.append(ax_chn_spcs)
                    if ao_data_xy.size == 0:
                        # first concatenate the X/Y data to empty array to have 1D array
                        ao_data_xy = np.concatenate((ao_data_xy, ao_data[ao_data_row_idx]))
                    else:
                        # then, if XY scan, stack the Y data below the X data to have 2D array
                        ao_data_xy = np.vstack((ao_data_xy, ao_data[ao_data_row_idx]))
                    ao_data_row_idx += 1
                else:  # "Z"
                    z_chan_spcs.append(ax_chn_spcs)
                    ao_data_z = ao_data[ao_data_row_idx]

        # start smooth
        if z_chan_spcs:
            smooth_start(axis="z", ao_chan_specs=z_chan_spcs, final_pos=ao_data_z[0])

        try:
            self.close_tasks("ao")

            if xy_chan_spcs:
                xy_task_name = "AO XY"
                ao_task = self.create_ao_task(
                    name=xy_task_name,
                    chan_specs=xy_chan_spcs,
                    samp_clk_cnfg=samp_clk_cnfg_xy,
                )
                ao_data_xy = self._limit_ao_data(ao_task, ao_data_xy)
                ao_data_xy = self._diff_vltg_data(ao_data_xy)
                self.analog_write(xy_task_name, ao_data_xy)

            if z_chan_spcs:
                z_task_name = "AO Z"
                ao_task = self.create_ao_task(
                    name=z_task_name,
                    chan_specs=z_chan_spcs,
                    samp_clk_cnfg=samp_clk_cnfg_z,
                )
                ao_data_z = self._limit_ao_data(ao_task, ao_data_z)
                self.analog_write(z_task_name, ao_data_z)
                # TODO: perhaps floating z should be introduced here and not during pattern creation?
                # This is because there is no reason why the AOZ array should be the same length as the AOX/Y array -
                # this unnecessary requirement limits how slow the Z variation can be (should be very slow, and even out of sync with XY)

            if start is True:
                self.start_tasks("ao")

        except DaqError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)

        else:
            self.toggle_led_and_switch(True)

    def init_ai_buffer(self, type: str = "circular", size=None) -> None:
        """Doc."""

        with suppress(AttributeError, IndexError):
            # case ai_buffer not created yet, or just created and not populated yet
            self.last_int_ao = self.ai_buffer[-1][3:]

        if type == "circular":
            if size is None:
                size = self.AI_BUFFER_SIZE
            self.ai_buffer = deque([], maxlen=size)
        elif type == "inf":
            self.ai_buffer = []
        else:
            raise ValueError("type parameter must be either 'standard' or 'inf'.")

    def fill_ai_buffer(
        self, task_name: str = "Continuous AI", n_samples=ni_consts.READ_ALL_AVAILABLE
    ) -> None:
        """Doc."""

        read_samples = self.analog_read(task_name, n_samples)
        read_samples = self._diff_to_rse(read_samples)
        self.ai_buffer.extend(read_samples)

    def _diff_to_rse(self, read_samples: np.ndarray) -> list:
        """Doc."""

        n_chans, n_samps = read_samples.shape
        conv_array = np.empty((n_chans - 2, n_samps), dtype=np.float)
        conv_array[:3, :] = read_samples[:3, :]
        conv_array[3, :] = (read_samples[3, :] - read_samples[4, :]) / 2
        conv_array[4, :] = (read_samples[5, :] - read_samples[6, :]) / 2
        conv_array[5, :] = read_samples[7, :]

        return conv_array.T.tolist()

    def _limit_ao_data(self, ao_task, ao_data: np.ndarray) -> np.ndarray:
        ao_min = ao_task.channels.ao_min
        ao_max = ao_task.channels.ao_max
        return np.clip(ao_data, ao_min, ao_max)

    def _diff_vltg_data(self, ao_data: np.ndarray) -> np.ndarray:
        """
        For each row in 'ao_data', add the negative of that row
        as a row right after it, e.g.:

        [[0.5, 0.7, -0.2], [0.1, 0., 0.]] ->
        [[0.5, 0.7, -0.2], [-0.5, -0.7, 0.2], [0.1, 0., 0.], [-0.1, 0., 0.]]
        """

        if len(ao_data.shape) == 2:
            # 2D array
            diff_ao_data = np.empty(
                shape=(ao_data.shape[0] * 2, ao_data.shape[1]), dtype=np.float64
            )
            n_rows = ao_data.shape[0]
            for row_idx in range(n_rows):
                diff_ao_data[row_idx * 2] = ao_data[row_idx]
                diff_ao_data[row_idx * 2 + 1] = -ao_data[row_idx]
        else:
            # 1D array
            diff_ao_data = np.empty(shape=(2, ao_data.size), dtype=np.float64)
            diff_ao_data[0, :] = ao_data
            diff_ao_data[1, :] = -ao_data
        return diff_ao_data


class PhotonCounter(BaseDevice, NIDAQmx, metaclass=DeviceCheckerMetaClass):
    """
    Represents the DAQ card which counts the green
    fluorescence photons coming from the detector.
    """

    attrs = DeviceAttrs(
        log_ref="Photon Counter",
        param_widgets=QtWidgetCollection(
            led_widget=("ledCounter", "QIcon", "main", True),
            pxl_clk=("counterPixelClockAddress", "QLineEdit", "settings", False),
            pxl_clk_output=("pixelClockCounterIntOutputAddress", "QLineEdit", "settings", False),
            trggr=("counterTriggerAddress", "QLineEdit", "settings", False),
            trggr_armstart_digedge=(
                "counterTriggerArmStartDigEdgeSrc",
                "QLineEdit",
                "settings",
                False,
            ),
            trggr_edge=("counterTriggerEdge", "QComboBox", "settings", False),
            address=("counterAddress", "QLineEdit", "settings", False),
            CI_cnt_edges_term=("counterCIcountEdgesTerm", "QLineEdit", "settings", False),
            CI_dup_prvnt=("counterCIdupCountPrevention", "QCheckBox", "settings", False),
        ),
        synced_dvc_attrs=[("scanners_ai_tasks", "devices.scanners.tasks.ai")],
    )

    CI_BUFFER_SIZE = int(1e4)

    def __init__(self, app):
        super().__init__(
            self.attrs,
            app,
            task_types=("ci",),
        )

        try:
            cont_ai_task = [
                task for task in self.scanners_ai_tasks if (task.name == "Continuous AI")
            ][0]
            self.ai_cont_rate = cont_ai_task.timing.samp_clk_rate
            self.ai_cont_src = cont_ai_task.timing.samp_clk_term
        except IndexError:
            exc = DeviceError(
                f"{self.log_ref} can't be synced because scanners failed to Initialize"
            )
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.cont_read_buffer = np.zeros(shape=(self.CONT_READ_BFFR_SZ,), dtype=np.uint32)
            self.last_avg_time = time.perf_counter()
            self.num_reads_since_avg = 0
            self.ci_chan_specs = {
                "name_to_assign_to_channel": "photon counter",
                "counter": self.address,
                "edge": ni_consts.Edge.RISING,
                "initial_count": 0,
                "count_direction": ni_consts.CountDirection.COUNT_UP,
            }

            with suppress(DeviceError):
                self.start_continuous_read_task()
                self.toggle_led_and_switch(True)

    def close(self):
        """Doc."""

        self.close_all_tasks()
        self.toggle_led_and_switch(False)

    def start_continuous_read_task(self) -> None:
        """Doc."""

        try:
            self.close_tasks("ci")
            self.create_ci_task(
                name="Continuous CI",
                chan_specs=self.ci_chan_specs,
                chan_xtra_params={"ci_count_edges_term": self.CI_cnt_edges_term},
                samp_clk_cnfg={
                    "rate": self.ai_cont_rate,
                    "source": self.ai_cont_src,
                    "sample_mode": ni_consts.AcquisitionType.CONTINUOUS,
                    "samps_per_chan": self.CONT_READ_BFFR_SZ,
                },
            )
            self.init_ci_buffer()
            self.start_tasks("ci")
        except DaqError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)

    def start_scan_read_task(self, samp_clk_cnfg, timing_params) -> None:
        """Doc."""

        try:
            self.close_tasks("ci")
            self.create_ci_task(
                name="Scan CI",
                chan_specs=self.ci_chan_specs,
                chan_xtra_params={
                    "ci_count_edges_term": self.CI_cnt_edges_term,
                    "ci_data_xfer_mech": ni_consts.DataTransferActiveTransferMode.DMA,
                },
                samp_clk_cnfg=samp_clk_cnfg,
                timing_params=timing_params,
            )
            self.start_tasks("ci")
        except DaqError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)

    def fill_ci_buffer(self, n_samples=ni_consts.READ_ALL_AVAILABLE):
        """Doc."""

        num_samps_read = self.counter_stream_read(self.cont_read_buffer)
        self.ci_buffer.extend(self.cont_read_buffer[:num_samps_read])
        self.num_reads_since_avg += num_samps_read

    def average_counts(self, interval_s: float, rate=None) -> None:
        """Doc."""

        if rate is None:
            rate = self.tasks.ci[-1].timing.samp_clk_rate

        n_reads = div_ceil(interval_s, (1 / rate))

        if len(self.ci_buffer) > n_reads:
            self.avg_cnt_rate_khz = (
                (self.ci_buffer[-1] - self.ci_buffer[-(n_reads + 1)]) / interval_s * 1e-3
            )
        else:
            # if buffer is too short for the requested interval, average over whole buffer
            interval_s = len(self.ci_buffer) * (1 / rate)
            with suppress(IndexError):
                # IndexError - buffer is empty, keep last value
                self.avg_cnt_rate_khz = (self.ci_buffer[-1] - self.ci_buffer[0]) / interval_s * 1e-3

    def init_ci_buffer(self, type: str = "circular", size=None) -> None:
        """Doc."""

        self.ci_buffer: Union[list, deque]

        if type == "circular":
            if size is None:
                size = self.CI_BUFFER_SIZE
            self.ci_buffer = deque([], maxlen=size)
        elif type == "inf":
            self.ci_buffer = []
        else:
            raise ValueError("type parameter must be either 'standard' or 'inf'.")


class PixelClock(BaseDevice, NIDAQmx, metaclass=DeviceCheckerMetaClass):
    """
    The pixel clock is fed to the DAQ board from the FPGA.
    Base frequency is 4 MHz. Used for scans, where it is useful to
    have divisible frequencies for both the laser pulses and AI/AO.
    """

    attrs = DeviceAttrs(
        log_ref="Pixel Clock",
        param_widgets=QtWidgetCollection(
            led_widget=("ledPxlClk", "QIcon", "main", True),
            low_ticks=("pixelClockLowTicks", "QSpinBox", "settings", False),
            high_ticks=("pixelClockHighTicks", "QSpinBox", "settings", False),
            cntr_addr=("pixelClockCounterAddress", "QLineEdit", "settings", False),
            tick_src=("pixelClockSrcOfTicks", "QLineEdit", "settings", False),
            out_term=("pixelClockOutput", "QLineEdit", "settings", False),
            out_ext_term=("pixelClockOutputExt", "QLineEdit", "settings", False),
            freq_MHz=("pixelClockFreq", "QSpinBox", "settings", False),
        ),
    )

    def __init__(self, app):
        super().__init__(
            self.attrs,
            app,
            task_types=("co",),
        )

        with suppress(DeviceError):
            self.toggle(False)

    def toggle(self, is_being_switched_on):
        """Doc."""

        try:
            if is_being_switched_on:
                self._start_co_clock_sync()
            else:
                self.close_all_tasks()
        except DaqError:
            exc = IOError(
                f"NI device address ({self.address}) is wrong, or data acquisition board is unplugged"
            )
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.toggle_led_and_switch(is_being_switched_on)
            self.is_on = is_being_switched_on

    def _start_co_clock_sync(self) -> None:
        """Doc."""

        task_name = "Pixel Clock CO"

        self.close_tasks("co")
        self.create_co_task(
            name=task_name,
            chan_spec={
                "name_to_assign_to_channel": "pixel clock",
                "counter": self.cntr_addr,
                "source_terminal": self.tick_src,
                "low_ticks": self.low_ticks,
                "high_ticks": self.high_ticks,
            },
            clk_cnfg={"sample_mode": ni_consts.AcquisitionType.CONTINUOUS},
        )
        self.start_tasks("co")


class ExcitationLaser(SimpleDODevice):
    """Controls excitation laser"""

    attrs = DeviceAttrs(
        log_ref="Excitation Laser",
        led_color="blue",
        param_widgets=QtWidgetCollection(
            led_widget=("ledExc", "QIcon", "main", True),
            switch_widget=("excOnButton", "QIcon", "main", True),
            model=("excMod", "QLineEdit", "settings", False),
            trg_src=("excTriggerSrc", "QComboBox", "settings", False),
            ext_trg_addr=("excTriggerExtAddr", "QLineEdit", "settings", False),
            int_trg_addr=("excTriggerIntAddr", "QLineEdit", "settings", False),
            address=("excAddr", "QLineEdit", "settings", False),
            off_timer_min=("excIdleTimer", "QSpinBox", "settings", False),
        ),
    )

    def __init__(self, app):
        super().__init__(self.attrs, app)


class DepletionLaser(BaseDevice, PyVISA, metaclass=DeviceCheckerMetaClass):
    """Control depletion laser through pyVISA"""

    attrs = DeviceAttrs(
        log_ref="Depletion Laser",
        led_color="orange",
        param_widgets=QtWidgetCollection(
            led_widget=("ledDep", "QIcon", "main", True),
            switch_widget=("depEmissionOn", "QIcon", "main", True),
            hours_of_operation_widget=("depTimeOfOperation", "QSpinBox", "main", True),
            model_query=("depModelQuery", "QLineEdit", "settings", False),
            off_timer_min=("depIdleTimer", "QSpinBox", "settings", False),
        ),
    )

    update_interval_s = 0.5
    MIN_SHG_TEMP_C = 45  # Celsius # was 53 for old laser # TODO: move to settings
    power_limits_mW = Limits(99, 1000)
    current_limits_mA = Limits(1500, 2500)

    def __init__(self, app):

        super().__init__(
            self.attrs,
            app,
            read_termination="\r",
            write_termination="\r",
        )
        self.address = None  # found automatically
        self.is_on = None
        self.is_emission_on = None
        self.turn_on_time = None
        self.time_of_operation_hr = None

        with suppress(DeviceError):
            self.toggle(True, should_change_icons=False)
            if self.is_on:
                self.set_current(1500)

    def toggle(self, is_being_switched_on, **kwargs):
        """Doc."""

        try:
            if is_being_switched_on:
                self.open_instrument(model_query=self.model_query)
                self.laser_toggle(False)
            else:
                if self.is_on is True:
                    self.laser_toggle(False)
                self.close_instrument()
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.toggle_led_and_switch(is_being_switched_on, **kwargs)
            self.is_on = is_being_switched_on

    def laser_toggle(self, is_being_switched_on):
        """Doc."""

        cmnd = f"setLDenable {int(is_being_switched_on)}"
        try:
            self.write(cmnd)
        except Exception as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.is_emission_on = is_being_switched_on
            self.change_icons("on" if is_being_switched_on else "off")
            self.turn_on_time = time.perf_counter() if is_being_switched_on else None

    def get_prop(self, prop):
        """Doc."""

        prop_cmnd_dict = {
            "temp": "SHGtemp",
            "curr": "LDcurrent 1",
            "pow": "power 0",
            "on_time": "GETTIMEOP",
        }
        cmnd = prop_cmnd_dict[prop]

        try:
            response = self.query(cmnd)
            try:
                number, *_ = generate_numbers_from_string(response)
            except ValueError:
                return 0
            else:
                return number
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)

    def set_power(self, value_mW):
        """Doc."""

        # check that value is within range
        if value_mW in self.power_limits_mW:
            # change the mode to power
            cmnd = "powerenable 1"
            self.write(cmnd)
            # then set the power
            cmnd = f"setpower 0 {value_mW}"
            self.write(cmnd)
        else:
            ErrorDialog(custom_txt=f"Power out of range {self.power_limits_mW}").display()

    def set_current(self, value_mA):
        """Doc."""

        # check that value is within range
        if value_mA in self.current_limits_mA:
            # change the mode to current
            cmnd = "powerenable 0"
            self.write(cmnd)
            # then set the current
            cmnd = f"setLDcur 1 {value_mA}"
            self.write(cmnd)
        else:
            ErrorDialog(custom_txt=f"Current out of range {self.current_limits_mA}").display()


class Shutter(SimpleDODevice):
    """Controls depletion laser external shutter"""

    attrs = DeviceAttrs(
        log_ref="Shutter",
        param_widgets=QtWidgetCollection(
            led_widget=("ledShutter", "QIcon", "main", True),
            switch_widget=("depShutterOn", "QIcon", "main", True),
            address=("depShutterAddr", "QLineEdit", "settings", False),
        ),
    )

    def __init__(self, app):
        super().__init__(self.attrs, app)


class StepperStage(BaseDevice, PyVISA, metaclass=DeviceCheckerMetaClass):
    """Control stepper stage through Arduino chip using PyVISA."""

    # TODO: add support for saving all movements done since init
    # and add button to move back to origin. also save to file and rewrite only when moving after app opens again

    attrs = DeviceAttrs(
        log_ref="Stage",
        param_widgets=QtWidgetCollection(
            led_widget=("ledStage", "QIcon", "main", True),
            switch_widget=("stageOn", "QIcon", "main", True),
            address=("arduinoAddr", "QLineEdit", "settings", False),
        ),
    )

    def __init__(self, app):
        super().__init__(self.attrs, app)

        with suppress(DeviceError):
            self.toggle(True)
            self.toggle(False)

    def toggle(self, is_being_switched_on: bool):
        """Doc."""

        try:
            self.open_instrument() if is_being_switched_on else self.close_instrument()
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.toggle_led_and_switch(is_being_switched_on)
            self.is_on = is_being_switched_on

    async def move(self, dir, steps):
        """Doc."""

        cmd_dict = {
            "UP": f"my {-steps}",
            "DOWN": f"my {steps}",
            "LEFT": f"mx {steps}",
            "RIGHT": f"mx {-steps}",
        }
        self.write(cmd_dict[dir])
        await asyncio.sleep(500 * 1e-3)
        self.write("ryx ")  # release


class Camera(BaseDevice, Instrumental, metaclass=DeviceCheckerMetaClass):
    """Doc."""

    DEFAULT_PARAM_DICT = {"pixel_clock": 25, "framerate": 15.0, "exposure": 1.0}
    PIXEL_SIZE_UM = 3.6

    def __init__(self, *args):
        super().__init__(
            *args,
        )
        # TODO: Get these from the widgets during initialization!
        self.is_in_video_mode = False
        self.is_in_grayscale_mode = True
        self.should_get_diameter = False
        self.is_auto_exposure_on = False
        self.last_snapshot = np.zeros((100, 100))

        with suppress(DeviceError):
            self.open()
            self.update_parameter_ranges()

    def open(self) -> None:
        """Doc."""

        try:
            self.open_instrument()
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)

    def close(self) -> None:
        """Doc."""

        self.close_instrument()

    async def get_image(self, should_display=True) -> None:
        """Doc."""

        try:
            if self.is_in_video_mode:
                img = await self.get_latest_frame()  # fast (~0.3 ms)
            else:
                img = self.capture_image()  # slow (~70 ms)
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.last_snapshot = img

            if should_display:
                self._display_last_img()

    def _display_last_img(self) -> None:
        """Doc."""

        if self.last_snapshot is not None:
            img_arr = self.last_snapshot

            if self.is_in_grayscale_mode:
                img_arr = np.asarray(PIL.Image.fromarray(img_arr, mode="RGB").convert("L"))

            # display in GUI
            self.display.obj.display_image(img_arr, cursor=True)

            if self.should_get_diameter:
                self.get_and_display_beam_width(img_arr)

    def toggle_video(self, should_turn_on: bool, keep_off=False, **kwargs) -> bool:
        """Doc."""

        if keep_off:
            should_turn_on = False

        try:
            self.toggle_video_mode(should_turn_on, **kwargs)
            self.toggle_led_and_switch(should_turn_on)
            self.is_in_video_mode = should_turn_on
            return should_turn_on
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
            return not should_turn_on

    def set_parameters(self, param_dict: dict = DEFAULT_PARAM_DICT) -> None:
        """Set pixel_clock, framerate and exposure"""

        [self.set_parameter(name, value) for name, value in param_dict.items()]

    def get_and_display_beam_width(self, img_arr):
        """Doc."""

        # properly convert to grayscale
        if not self.is_in_grayscale_mode:
            gs_img = PIL.Image.fromarray(img_arr, mode="RGB").convert("L")
        else:
            gs_img = PIL.Image.fromarray(img_arr)

        # cropping to square - assuming beam is centered, takes the same amount from both sides of the longer dimension
        width, height = gs_img.size
        crop_delta_x = abs(width - height) / 2 if width > height else 0
        crop_delta_y = abs(width - height) / 2 if width < height else 0
        crop_dims = (crop_delta_x, crop_delta_y, width - crop_delta_x, height - crop_delta_y)
        cropped_gs_img = gs_img.crop(crop_dims)

        # resizing
        RESCALE_FACTOR = 10
        resized_cropped_gs_img = cropped_gs_img.resize(
            (
                round(min(width, height) / RESCALE_FACTOR),
                round(min(width, height) / RESCALE_FACTOR),
            ),
            resample=PIL.Image.NEAREST,
        )

        # converting back to Numpy array
        resized_cropped_gs_img_arr = np.asarray(resized_cropped_gs_img)

        # fitting
        try:
            fp = fit_2d_gaussian_to_image(resized_cropped_gs_img_arr)
        except FitError as exc:
            logging.debug(f"{self.log_ref}: Gaussian fit failed! [{exc}]")
            return
        x0, y0, sigma_x, sigma_y, phi = (
            fp.beta["x0"],
            fp.beta["y0"],
            fp.beta["sigma_x"],
            fp.beta["sigma_y"],
            fp.beta["phi"],
        )

        max_sigma_y, max_sigma_x = resized_cropped_gs_img_arr.shape
        if (
            x0 < 0
            or y0 < 0
            or abs(1 - sigma_x / sigma_y) > 2
            or sigma_x > max_sigma_x
            or sigma_y > max_sigma_y
        ):
            print(
                f"{self.log_ref}: Gaussian fit is irrational! Center on CCD and avoid saturation.\n({fp.beta})"
            )
            return

        # calculating the FWHM
        FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))  # 1/e^2 width is FWHM * 1.699
        one_over_e2_factor = 1.699 * FWHM_FACTOR
        diameter_mm = (
            np.mean([sigma_x, sigma_y])
            * one_over_e2_factor
            * self.PIXEL_SIZE_UM
            * 1e-3
            * RESCALE_FACTOR
        )
        diameter_mm_err = (
            np.std([sigma_x, sigma_y])
            * one_over_e2_factor
            * self.PIXEL_SIZE_UM
            * 1e-3
            * RESCALE_FACTOR
        )

        logging.debug(
            f"{self.log_ref}: 1/e^2 width determined to be {diameter_mm:.2f} +/- {diameter_mm_err:.2f} mm"
        )

        # plotting the FWHM on top of the image
        ellipse = Ellipse(
            xy=(x0 * RESCALE_FACTOR + crop_delta_x, y0 * RESCALE_FACTOR + crop_delta_y),
            width=sigma_y * FWHM_FACTOR * RESCALE_FACTOR,
            height=sigma_x * FWHM_FACTOR * RESCALE_FACTOR,
            angle=phi,
        )
        ellipse.set_facecolor((0, 0, 0, 0))
        ellipse.set_edgecolor("red")

        self.display.obj.add_patch(
            ellipse,
            annotation=f"$1/e^2$: {diameter_mm:.2f}$\\pm${diameter_mm_err:.2f} mm\n$\\chi^2$={fp.chi_sq_norm:.2f}",
        )


class Camera1(Camera):
    """Doc."""

    attrs = DeviceAttrs(
        log_ref="Camera 1",
        param_widgets=QtWidgetCollection(
            led_widget=("ledCam1", "QIcon", "main", True),
            switch_widget=("videoSwitch1", "QIcon", "main", True),
            display=("ImgDisp1", None, "main", True),
            serial=("cam1Serial", "QLineEdit", "settings", False),
        ),
    )

    def __init__(self, app):
        super().__init__(self.attrs, app)


class Camera2(Camera):
    """Doc."""

    attrs = DeviceAttrs(
        log_ref="Camera 2",
        param_widgets=QtWidgetCollection(
            led_widget=("ledCam2", "QIcon", "main", True),
            switch_widget=("videoSwitch2", "QIcon", "main", True),
            display=("ImgDisp2", None, "main", True),
            serial=("cam2Serial", "QLineEdit", "settings", False),
        ),
    )

    def __init__(self, app):
        super().__init__(self.attrs, app)
