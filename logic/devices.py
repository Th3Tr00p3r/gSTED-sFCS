"""Devices Module."""

import asyncio
import sys
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import List

import nidaqmx.constants as ni_consts
import numpy as np
from nidaqmx.errors import DaqError

import utilities.helper as helper
from gui.dialog import Error
from gui.icons import icons
from gui.widgets import QtWidgetCollection
from logic.drivers import Ftd2xx, Instrumental, NIDAQmx, PyVISA
from logic.timeout import GUI_UPDATE_INTERVAL, TIMEOUT_INTERVAL
from utilities.errors import DeviceCheckerMetaClass, DeviceError, IOError, err_hndlr


class BaseDevice:
    """Doc."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_dict = None

    def change_icons(self, command):
        """Doc."""

        def set_icon_wdgts(led_icon, has_switch: bool, switch_icon=None):
            """Doc."""

            self.led_widget.set(led_icon)
            if has_switch:
                self.switch_widget.set(switch_icon)

        if not hasattr(self, "icon_dict"):
            # get icons
            self.icon_dict = icons.get_icon_paths()

        has_switch = hasattr(self, "switch_widget")
        if command == "on":
            set_icon_wdgts(self.led_icon, has_switch, self.icon_dict["switch_on"])
        elif command == "off":
            set_icon_wdgts(self.icon_dict["led_off"], has_switch, self.icon_dict["switch_off"])
        elif command == "error":
            set_icon_wdgts(self.icon_dict["led_red"], False)

    def close(self):
        """Defualt device close(). Some classes override this."""

        self.toggle(False)

    def toggle_led(self, is_being_switched_on, should_change_icons=True):
        """Toggle the devices LED widget ON/OFF"""

        if not self.error_dict and should_change_icons:
            self.change_icons("on" if is_being_switched_on else "off")


class UM232H(BaseDevice, Ftd2xx, metaclass=DeviceCheckerMetaClass):
    """
    Represents the FTDI chip used to transfer data from the FPGA
    to the PC.
    """

    def __init__(self, param_dict):
        super().__init__(
            param_dict=param_dict,
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

        self.data.extend(byte_array)
        self.tot_bytes_read += len(byte_array)

    def init_data(self):
        """Doc."""

        self.data = []
        self.tot_bytes_read = 0

    def purge_buffers(self):
        """Doc."""
        try:
            self.purge()
        except Exception as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)

    def get_status(self):
        """Doc."""

        try:
            return self.get_queue_status()
        except Exception as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)


class Scanners(BaseDevice, NIDAQmx, metaclass=DeviceCheckerMetaClass):
    """
    Scanners encompasses all analog focal point positioning devices
    (X: x_galvo, Y: y_galvo, Z: z_piezo)
    """

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
    ORIGIN = (0.0, 0.0, 5.0)
    X_AO_LIMITS = {"min_val": -5.0, "max_val": 5.0}
    Y_AO_LIMITS = {"min_val": -5.0, "max_val": 5.0}
    Z_AO_LIMITS = {"min_val": 0.0, "max_val": 10.0}
    AI_BUFFER_SIZE = int(1e4)

    def __init__(self, param_dict):
        super().__init__(
            param_dict,
            task_types=("ai", "ao"),
        )
        rse = ni_consts.TerminalConfiguration.RSE
        diff = ni_consts.TerminalConfiguration.DIFFERENTIAL
        self.ai_chan_specs = [
            {
                "physical_channel": getattr(self, f"ai_{axis}_addr"),
                "name_to_assign_to_channel": f"{axis}-{inst} AI",
                "min_val": -10.0,
                "max_val": 10.0,
                "terminal_config": rse,
            }
            for axis, inst in zip("xyz", ("galvo", "galvo", "piezo"))
        ]
        self.ao_int_chan_specs = [
            {
                "physical_channel": getattr(self, f"ao_int_{axis}_addr"),
                "name_to_assign_to_channel": f"{axis}-{inst} internal AO",
                **getattr(self, f"{axis.upper()}_AO_LIMITS"),
                "terminal_config": trm_cnfg,
            }
            for axis, trm_cnfg, inst in zip("xyz", (diff, diff, rse), ("galvo", "galvo", "piezo"))
        ]
        self.ao_chan_specs = [
            {
                "physical_channel": getattr(self, f"ao_{axis}_addr"),
                "name_to_assign_to_channel": f"{axis}-{inst} AO",
                **getattr(self, f"{axis.upper()}_AO_LIMITS"),
            }
            for axis, inst in zip("xyz", ("galvo", "galvo", "piezo"))
        ]
        self.um_v_ratio = tuple(getattr(self, f"{ax}_um2v_const") for ax in "xyz")

        try:
            self.start_continuous_read_task()
        except FileNotFoundError as exc:  # TODO: fix this by rasing an IOError in drivers.py (appears in other places too). can only be discovered on laptop
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)

    def close(self):
        """Doc."""

        self.close_all_tasks()

    def start_continuous_read_task(self) -> None:
        """Doc."""

        try:
            self.close_tasks("ai")
            self.create_ai_task(
                name="Continuous AI",
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
            self.toggle_led(False)

    def start_scan_read_task(self, samp_clk_cnfg, timing_params) -> None:
        """Doc."""

        try:
            self.close_tasks("ai")
            self.create_ai_task(
                name="Continuous AI",
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
            axis: str, ao_chan_specs: dict, final_pos: float, step_sz: float = 0.25
        ) -> None:
            """Doc."""
            # NOTE: Ask Oleg why we used 40 steps in LabVIEW (this is why I use a step size of 10/40 V)

            try:
                init_pos = self.ai_buffer[-1][3:][self.AXIS_INDEX[axis]]
            except IndexError:
                init_pos = self.last_int_ao[self.AXIS_INDEX[axis]]

            total_dist = abs(final_pos - init_pos)
            n_steps = helper.div_ceil(total_dist, step_sz)

            if n_steps < 2:
                # one small step needs no smoothing
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

            if start is True:
                self.start_tasks("ao")

        except DaqError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)

        else:
            self.toggle_led(True)

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
            diff_ao_data = np.empty(shape=(ao_data.shape[0] * 2, ao_data.shape[1]), dtype=np.float)
            n_rows = ao_data.shape[0]
            for row_idx in range(n_rows):
                diff_ao_data[row_idx * 2] = ao_data[row_idx]
                diff_ao_data[row_idx * 2 + 1] = -ao_data[row_idx]
        else:
            # 1D array
            diff_ao_data = np.empty(shape=(2, ao_data.size), dtype=np.float)
            diff_ao_data[0, :] = ao_data
            diff_ao_data[1, :] = -ao_data
        return diff_ao_data


class PhotonDetector(BaseDevice, NIDAQmx, metaclass=DeviceCheckerMetaClass):
    """
    Represents the detector which counts the green
    fluorescence photons coming from the sample.
    """

    CI_BUFFER_SIZE = int(1e4)

    def __init__(self, param_dict, scanners_ai_tasks):
        super().__init__(
            param_dict,
            task_types=("ci",),
        )

        try:
            cont_ai_task = [task for task in scanners_ai_tasks if (task.name == "Continuous AI")][0]
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

            self.start_continuous_read_task()
            self.toggle_led(True)

    def close(self):
        """Doc."""

        self.close_all_tasks()
        self.toggle_led(False)

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

        num_samps_read = self.counter_stream_read()
        self.ci_buffer.extend(self.cont_read_buffer[:num_samps_read])
        self.num_reads_since_avg += num_samps_read

    def average_counts(self, interval_s: float, rate=None) -> None:
        """Doc."""

        if rate is None:
            rate = self.tasks.ci[-1].timing.samp_clk_rate

        n_reads = helper.div_ceil(interval_s, (1 / rate))

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

    def __init__(self, param_dict):
        super().__init__(
            param_dict,
            task_types=("co",),
        )

        self.toggle(False)

    def toggle(self, is_being_switched_on):
        """Doc."""

        if is_being_switched_on:
            self._start_co_clock_sync()
            self.toggle_led(True)
        else:
            self.close_all_tasks()
            self.toggle_led(False)

        self.state = is_being_switched_on

    def _start_co_clock_sync(self) -> None:
        """Doc."""

        task_name = "Pixel Clock CO"

        try:
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
        except Exception as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)


class SimpleDO(BaseDevice, NIDAQmx, metaclass=DeviceCheckerMetaClass):
    """ON/OFF device (excitation laser, depletion shutter, TDC)."""

    def __init__(self, param_dict):
        super().__init__(
            param_dict,
            task_types=("do",),
        )
        self.toggle(False)

    def toggle(self, is_being_switched_on):
        """Doc."""

        try:
            self.digital_write(is_being_switched_on)
        except DaqError:
            exc = IOError(
                f"NI device address ({self.address}) is wrong, or data acquisition board is unplugged"
            )
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        except FileNotFoundError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.toggle_led(is_being_switched_on)
            self.state = is_being_switched_on


class DepletionLaser(BaseDevice, PyVISA, metaclass=DeviceCheckerMetaClass):
    """Control depletion laser through pyVISA"""

    update_interval_s = 0.3
    MIN_SHG_TEMP = 53  # Celsius
    power_limits_mW = dict(low=99, high=1000)
    current_limits_mA = dict(low=1500, high=2500)

    def __init__(self, param_dict):

        super().__init__(
            param_dict,
            read_termination="\r",
            write_termination="\r",
        )

        self.state = None
        self.emission_state = None
        self.toggle(True, should_change_icons=False)

        if self.state:
            self.set_current(1500)

    def toggle(self, is_being_switched_on, **kwargs):
        """Doc."""

        try:
            if is_being_switched_on:
                self.open_instrument()
                self.laser_toggle(False)
            else:
                if self.state is True:
                    self.laser_toggle(False)
                self.close_instrument()
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.toggle_led(is_being_switched_on, **kwargs)
            self.state = is_being_switched_on

    def laser_toggle(self, is_being_switched_on):
        """Doc."""

        cmnd = f"setLDenable {int(is_being_switched_on)}"
        try:
            self.write(cmnd)
        except Exception as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.emission_state = is_being_switched_on
            self.change_icons("on" if is_being_switched_on else "off")

    def get_prop(self, prop):
        """Doc."""

        prop_cmnd_dict = {
            "temp": "SHGtemp",
            "curr": "LDcurrent 1",
            "pow": "power 0",
        }
        cmnd = prop_cmnd_dict[prop]

        try:
            self.flush()  # get fresh response
            response = self.query(cmnd)
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            return response

    def set_power(self, value_mW):
        """Doc."""

        # check that value is within range
        if self.power_limits_mW["low"] <= value_mW <= self.power_limits_mW["high"]:
            try:
                # change the mode to power
                cmnd = "powerenable 1"
                self.write(cmnd)
                # then set the power
                cmnd = f"setpower 0 {value_mW}"
                self.write(cmnd)
            except Exception as exc:
                err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            Error(
                custom_txt=f"Power out of range [{self.power_limits_mW['low']}, {self.power_limits_mW['high']}]"
            ).display()

    def set_current(self, value_mA):
        """Doc."""

        # check that value is within range
        if self.current_limits_mA["low"] <= value_mA <= self.current_limits_mA["high"]:
            try:
                # change the mode to power
                cmnd = "powerenable 0"
                self.write(cmnd)
                # then set the power
                cmnd = f"setLDcur 1 {value_mA}"
                self.write(cmnd)
            except Exception as exc:
                err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            Error(
                custom_txt=f"Current out of range [{self.current_limits_mA['low']}, {self.current_limits_mA['high']}]"
            ).display()


class StepperStage(BaseDevice, PyVISA, metaclass=DeviceCheckerMetaClass):
    """Control stepper stage through Arduino chip using PyVISA."""

    # TODO: add support for saving all movements done since init
    # and add button to move back to origin. also save to file and rewrite only when moving after app opens again

    def __init__(self, param_dict):
        super().__init__(param_dict)

        self.toggle(True)
        self.toggle(False)

    def toggle(self, is_being_switched_on):
        """Doc."""

        try:
            self.open_instrument() if is_being_switched_on else self.close_instrument()
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.toggle_led(is_being_switched_on)
            self.state = is_being_switched_on

    async def move(self, dir, steps):
        """Doc."""

        cmd_dict = {
            "UP": f"my {-steps}",
            "DOWN": f"my {steps}",
            "LEFT": f"mx {steps}",
            "RIGHT": f"mx {-steps}",
        }
        try:
            self.write(cmd_dict[dir])
            await asyncio.sleep(500 * 1e-3)
            self.write("ryx ")  # release
        except Exception as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)


class Camera(BaseDevice, Instrumental, metaclass=DeviceCheckerMetaClass):
    """Doc."""

    video_interval = GUI_UPDATE_INTERVAL

    def __init__(self, param_dict):
        super().__init__(
            param_dict,
        )
        self.is_in_video_mode = False
        self.is_connected = False

    def open(self) -> bool:
        """Doc."""

        try:
            self.open_instrument()
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
            return False
        else:
            self.is_connected = True
            return True

    def close(self) -> None:
        """Doc."""

        self.close_instrument()
        self.is_in_video_mode = False
        self.is_connected = False


@dataclass
class DeviceAttrs:
    class_name: str
    log_ref: str
    param_widgets: QtWidgetCollection
    led_color: str = "green"
    cls_xtra_args: List[str] = None


DEVICE_ATTR_DICT = {
    "exc_laser": DeviceAttrs(
        class_name="SimpleDO",
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
        ),
    ),
    "dep_shutter": DeviceAttrs(
        class_name="SimpleDO",
        log_ref="Shutter",
        param_widgets=QtWidgetCollection(
            led_widget=("ledShutter", "QIcon", "main", True),
            switch_widget=("depShutterOn", "QIcon", "main", True),
            address=("depShutterAddr", "QLineEdit", "settings", False),
        ),
    ),
    "TDC": DeviceAttrs(
        class_name="SimpleDO",
        log_ref="TDC",
        param_widgets=QtWidgetCollection(
            led_widget=("ledTdc", "QIcon", "main", True),
            address=("TDCaddress", "QLineEdit", "settings", False),
            data_vrsn=("TDCdataVersion", "QLineEdit", "settings", False),
            laser_freq_mhz=("TDClaserFreq", "QSpinBox", "settings", False),
            fpga_freq_mhz=("TDCFPGAFreq", "QSpinBox", "settings", False),
            tdc_vrsn=("TDCversion", "QSpinBox", "settings", False),
        ),
    ),
    "dep_laser": DeviceAttrs(
        class_name="DepletionLaser",
        log_ref="Depletion Laser",
        led_color="orange",
        param_widgets=QtWidgetCollection(
            led_widget=("ledDep", "QIcon", "main", True),
            switch_widget=("depEmissionOn", "QIcon", "main", True),
            model_query=("depModelQuery", "QLineEdit", "settings", False),
        ),
    ),
    "stage": DeviceAttrs(
        class_name="StepperStage",
        log_ref="Stage",
        param_widgets=QtWidgetCollection(
            led_widget=("ledStage", "QIcon", "main", True),
            switch_widget=("stageOn", "QIcon", "main", True),
            address=("arduinoAddr", "QLineEdit", "settings", False),
        ),
    ),
    "UM232H": DeviceAttrs(
        class_name="UM232H",
        log_ref="UM232H",
        param_widgets=QtWidgetCollection(
            led_widget=("ledUm232h", "QIcon", "main", True),
            bit_mode=("um232BitMode", "QLineEdit", "settings", False),
            timeout_ms=("um232Timeout", "QSpinBox", "settings", False),
            ltncy_tmr_val=("um232LatencyTimerVal", "QSpinBox", "settings", False),
            flow_ctrl=("um232FlowControl", "QLineEdit", "settings", False),
            tx_size=("um232TxSize", "QSpinBox", "settings", False),
            n_bytes=("um232NumBytes", "QSpinBox", "settings", False),
        ),
    ),
    "camera_1": DeviceAttrs(
        class_name="Camera",
        log_ref="Camera 1",
        param_widgets=QtWidgetCollection(
            led_widget=("ledCam1", "QIcon", "camera", True),
            serial=("cam1Serial", "QLineEdit", "settings", False),
        ),
    ),
    "camera_2": DeviceAttrs(
        class_name="Camera",
        log_ref="Camera 2",
        param_widgets=QtWidgetCollection(
            led_widget=("ledCam2", "QIcon", "camera", True),
            serial=("cam2Serial", "QLineEdit", "settings", False),
        ),
    ),
    "scanners": DeviceAttrs(
        class_name="Scanners",
        log_ref="Scanners",
        param_widgets=QtWidgetCollection(
            led_widget=("ledScn", "QIcon", "main", True),
            ao_x_init_vltg=("xAOV", "QDoubleSpinBox", "main", False),
            ao_y_init_vltg=("yAOV", "QDoubleSpinBox", "main", False),
            ao_z_init_vltg=("zAOV", "QDoubleSpinBox", "main", False),
            x_um2v_const=("xConv", "QDoubleSpinBox", "settings", False),
            y_um2v_const=("yConv", "QDoubleSpinBox", "settings", False),
            z_um2v_const=("zConv", "QDoubleSpinBox", "settings", False),
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
    ),
    "photon_detector": DeviceAttrs(
        class_name="PhotonDetector",
        cls_xtra_args=["devices.scanners.tasks.ai"],
        log_ref="Photon Detector",
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
    ),
    "pixel_clock": DeviceAttrs(
        class_name="PixelClock",
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
    ),
}
