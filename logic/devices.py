"""Devices Module."""
import asyncio
import sys
import time

import numpy as np
from nidaqmx.errors import DaqError
from pyvisa.errors import VisaIOError

import utilities.constants as consts
import utilities.dialog as dialog
from logic.drivers import Ftd2xx, Instrumental, NIDAQmx, PyVISA
from utilities.errors import err_hndlr
from utilities.helper import div_ceil, sync_to_thread

# TODO: refactoring - toggling should be seperate from opening/closing connection with device.
# this would make redundent many flag arguments I have in dvc_toggle(), device_toggle_button_released(), toggle() etc.
# and would make a distinction between toggle vs. "connect" (e.g), which would be clearer, cleaner and more modular.


class BaseDevice:
    """Doc."""

    def change_icons(self, command):
        """Doc."""

        def set_icon_wdgts(led_icon, has_switch: bool, switch_icon=None):
            """Doc."""

            self.led_widget.set(led_icon)
            if has_switch:
                self.switch_widget.set(switch_icon)

        has_switch = hasattr(self, "switch_widget")
        if command == "on":
            set_icon_wdgts(self.led_icon, has_switch, consts.SWITCH_ON_ICON)
        elif command == "off":
            set_icon_wdgts(consts.LED_OFF_ICON, has_switch, consts.SWITCH_OFF_ICON)
        elif command == "error":
            set_icon_wdgts(consts.LED_ERROR_ICON, False)

    def toggle(self, is_being_switched_on, should_change_icons=True):
        """Doc."""

        try:
            self._toggle(is_being_switched_on)
        except Exception as exc:
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)
        else:
            self.state = is_being_switched_on
            if should_change_icons:
                self.change_icons("on" if is_being_switched_on else "off")


class UM232H(BaseDevice, Ftd2xx):
    """
    Represents the FTDI chip used to transfer data from the FPGA
    to the PC.
    """

    def __init__(self, param_dict):
        super().__init__(
            param_dict=param_dict,
        )

        self.init_data()
        self.toggle(True)

    def _toggle(self, is_being_switched_on):
        """Doc."""

        if is_being_switched_on:
            self.open()
            self.purge()
        else:
            self.close()

    async def read_TDC(self):
        """Doc."""

        try:
            byte_array = await self.read()
            self.data = np.concatenate((self.data, byte_array), axis=0)
            self.tot_bytes_read += len(byte_array)
        except Exception as exc:
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)

    def init_data(self):
        """Doc."""

        self.data = np.empty(shape=(0,), dtype=np.uint8)
        self.tot_bytes_read = 0

    def purge_buffers(self):
        """Doc."""
        try:
            self.purge()
        except Exception as exc:
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)

    def get_status(self):
        """Doc."""

        try:
            return self.get_queue_status()
        except Exception as exc:
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)


class Scanners(BaseDevice, NIDAQmx):
    """
    Scanners encompasses all analog focal point positioning devices
    (X: x_galvo, Y: y_galvo, Z: z_piezo)
    """

    origin = (0.0, 0.0, 5.0)
    x_ao_limits = {"min_val": -5.0, "max_val": 5.0}
    y_ao_limits = {"min_val": -5.0, "max_val": 5.0}
    z_ao_limits = {"min_val": 0.0, "max_val": 10.0}

    def __init__(self, param_dict):
        super().__init__(
            param_dict,
            task_types=("ai", "ao"),
        )

        rse = consts.NI.TerminalConfiguration.RSE
        diff = consts.NI.TerminalConfiguration.DIFFERENTIAL

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
                **getattr(self, f"{axis}_ao_limits"),
                "terminal_config": trm_cnfg,
            }
            for axis, trm_cnfg, inst in zip("xyz", (diff, diff, rse), ("galvo", "galvo", "piezo"))
        ]

        self.ao_chan_specs = [
            {
                "physical_channel": getattr(self, f"ao_{axis}_addr"),
                "name_to_assign_to_channel": f"{axis}-{inst} AO",
                **getattr(self, f"{axis}_ao_limits"),
            }
            for axis, inst in zip("xyz", ("galvo", "galvo", "piezo"))
        ]

        self.um_v_ratio = (self.x_um2V_const, self.y_um2V_const, self.z_um2V_const)

        self.toggle(True)

    def _toggle(self, is_being_switched_on):
        """Doc."""

        if is_being_switched_on:
            try:
                self.start_continuous_read_task()
            except DaqError:
                raise
        else:
            self.close_all_tasks()

    def start_continuous_read_task(self) -> None:
        """Doc."""

        try:
            task_name = "Continuous AI"

            self.close_tasks("ai")
            self.create_ai_task(
                name=task_name,
                chan_specs=self.ai_chan_specs + self.ao_int_chan_specs,
                samp_clk_cnfg={
                    "rate": self.MIN_OUTPUT_RATE_Hz,
                    "sample_mode": consts.NI.AcquisitionType.CONTINUOUS,
                    "samps_per_chan": self.CONT_READ_BFFR_SZ,
                },
            )
            self.init_ai_buffer()
            self.start_tasks("ai")
        except DaqError as exc:
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)

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
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)

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
            # TODO: Ask Oleg why we used 40 steps in LabVIEW (this is why I use a step size of 10/40 V)

            try:
                init_pos = self.ai_buffer[3:, -1][consts.AX_IDX[axis]]
            except IndexError:
                init_pos = self.last_int_ao[consts.AX_IDX[axis]]

            total_dist = abs(final_pos - init_pos)
            n_steps = div_ceil(total_dist, step_sz)

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
                            "sample_mode": consts.NI.AcquisitionType.FINITE,
                        },
                    )
                    ao_data = self.limit_ao_data(ao_task, ao_data)
                    self.analog_write(task_name, ao_data, auto_start=False)
                    self.start_tasks("ao")
                    self.wait_for_task("ao", task_name)
                    self.close_tasks("ao")
                except Exception as exc:
                    err_hndlr(exc, locals(), sys._getframe(), dvc=self)

        axes_to_use = consts.AXES_TO_BOOL_TUPLE_DICT[type]

        xy_chan_spcs = []
        z_chan_spcs = []
        ao_data_xy = np.empty(shape=(0,))
        ao_data_row_idx = 0
        for ax, use_ax, ax_chn_spcs in zip("XYZ", axes_to_use, self.ao_chan_specs):
            if use_ax is True:
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
                ao_data_xy = self.limit_ao_data(ao_task, ao_data_xy)
                ao_data_xy = self.diff_vltg_data(ao_data_xy)
                self.analog_write(xy_task_name, ao_data_xy)

            if z_chan_spcs:
                z_task_name = "AO Z"
                ao_task = self.create_ao_task(
                    name=z_task_name,
                    chan_specs=z_chan_spcs,
                    samp_clk_cnfg=samp_clk_cnfg_z,
                )
                ao_data_z = self.limit_ao_data(ao_task, ao_data_z)
                self.analog_write(z_task_name, ao_data_z)

            if start is True:
                self.start_tasks("ao")

        except Exception as exc:
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)

    def init_ai_buffer(self) -> None:
        """Doc."""

        try:
            self.last_int_ao = self.ai_buffer[3:, -1]
        except (AttributeError, IndexError):
            # case ai_buffer not created yet, or just created and not populated yet
            pass
        finally:
            self.ai_buffer = np.empty(shape=(6, 0), dtype=np.float)

    def fill_ai_buffer(
        self, task_name: str = "Continuous AI", n_samples=consts.NI.READ_ALL_AVAILABLE
    ) -> None:
        """Doc."""

        #        # TODO: stream reading currently not working for some reason - reading only one channel, the other two stay at zero
        #        num_samps_read = self.read()
        #        self.ai_buffer = np.concatenate(
        #        (self.ai_buffer, self.read_buffer[:, :num_samps_read]), axis=1
        #        )

        try:
            read_samples = self.analog_read(task_name, n_samples)
        except Exception as exc:
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)
        else:
            read_samples = np.concatenate(
                (read_samples[:3, :], self.diff_to_rse(read_samples[3:, :])), axis=0
            )
            self.ai_buffer = np.concatenate((self.ai_buffer, read_samples), axis=1)

    def dump_ai_buff_overflow(self):
        """Doc."""

        ai_buffer_len = self.ai_buffer.shape[1]
        if ai_buffer_len > self.CONT_READ_BFFR_SZ:
            self.ai_buffer = self.ai_buffer[:, -self.CONT_READ_BFFR_SZ :]

    def diff_to_rse(self, read_samples: np.ndarray) -> np.ndarray:
        """Doc."""

        rse_samples = np.empty(shape=(3, read_samples.shape[1]), dtype=np.float)
        rse_samples[0, :] = (read_samples[0, :] - read_samples[1, :]) / 2
        rse_samples[1, :] = (read_samples[2, :] - read_samples[3, :]) / 2
        rse_samples[2, :] = read_samples[4, :]
        return rse_samples

    def limit_ao_data(self, ao_task, ao_data: np.ndarray) -> np.ndarray:
        ao_min = ao_task.channels.ao_min
        ao_max = ao_task.channels.ao_max
        return np.clip(ao_data, ao_min, ao_max)

    def diff_vltg_data(self, ao_data: np.ndarray) -> np.ndarray:
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
            # TODO: can be coded better
            diff_ao_data = np.empty(shape=(2, ao_data.size), dtype=np.float)
            diff_ao_data[0, :] = ao_data
            diff_ao_data[1, :] = -ao_data
        return diff_ao_data


class Counter(BaseDevice, NIDAQmx):
    """
    Represents the detector which counts the green
    fluorescence photons coming from the sample.
    """

    updt_time = 0.2

    def __init__(self, param_dict, scanners_ai_tasks):
        super().__init__(
            param_dict,
            task_types=("ci", "co"),
        )
        self.cont_read_buffer = np.zeros(shape=(self.CONT_READ_BFFR_SZ,), dtype=np.uint32)

        self.last_avg_time = time.perf_counter()
        self.num_reads_since_avg = 0

        try:
            self.ai_cont_rate = scanners_ai_tasks["Continuous AI"].timing.samp_clk_rate
            self.ai_cont_src = scanners_ai_tasks["Continuous AI"].timing.samp_clk_term
        except KeyError:
            exc = RuntimeError("Counter can't be synced since scanners failed to Initialize")
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)

        self.ci_chan_specs = {
            "name_to_assign_to_channel": "photon counter",
            "counter": self.address,
            "edge": consts.NI.Edge.RISING,
            "initial_count": 0,
            "count_direction": consts.NI.CountDirection.COUNT_UP,
        }

        self.toggle(True)

    def _toggle(self, is_being_switched_on):
        """Doc."""

        if is_being_switched_on:
            self.start_continuous_read_task()
        else:
            self.close_all_tasks()

    def start_continuous_read_task(self) -> None:
        """Doc."""

        task_name = "Continuous CI"

        try:
            self.close_tasks("ci")
            self.create_ci_task(
                name=task_name,
                chan_specs=self.ci_chan_specs,
                chan_xtra_params={"ci_count_edges_term": self.CI_cnt_edges_term},
                samp_clk_cnfg={
                    "rate": self.ai_cont_rate,
                    "source": self.ai_cont_src,
                    "sample_mode": consts.NI.AcquisitionType.CONTINUOUS,
                    "samps_per_chan": self.CONT_READ_BFFR_SZ,
                },
            )
            self.init_ci_buffer()
            self.start_tasks("ci")
        except DaqError as exc:
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)

    def start_scan_read_task(self, samp_clk_cnfg, timing_params) -> None:
        """Doc."""

        try:
            self.close_tasks("ci")
            self.create_ci_task(
                name="Continuous CI",
                chan_specs=self.ci_chan_specs,
                chan_xtra_params={
                    "ci_count_edges_term": self.CI_cnt_edges_term,
                    "ci_data_xfer_mech": consts.NI.DataTransferActiveTransferMode.DMA,
                },
                samp_clk_cnfg=samp_clk_cnfg,
                timing_params=timing_params,
            )
            self.start_tasks("ci")
        except DaqError as exc:
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)

    def fill_ci_buffer(
        self,
        task_name: str = "Continuous CI",
        n_samples=consts.NI.READ_ALL_AVAILABLE,
    ):
        """Doc."""

        try:
            num_samps_read = self.counter_stream_read()
        except Exception as exc:
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)
        else:
            self.ci_buffer = np.concatenate(
                (self.ci_buffer, self.cont_read_buffer[:num_samps_read])
            )
            self.num_reads_since_avg += num_samps_read

    def average_counts(self) -> None:
        """Doc."""

        actual_intrvl = time.perf_counter() - self.last_avg_time
        start_idx = len(self.ci_buffer) - self.num_reads_since_avg

        if start_idx > 0:
            avg_cnt_rate = (
                self.ci_buffer[-1] - self.ci_buffer[-(self.num_reads_since_avg + 1)]
            ) / actual_intrvl
            avg_cnt_rate = avg_cnt_rate / 1000  # Hz -> KHz
        else:
            avg_cnt_rate = 0

        self.num_reads_since_avg = 0
        self.avg_cnt_rate = avg_cnt_rate
        self.last_avg_time = time.perf_counter()

    def init_ci_buffer(self) -> None:
        """Doc."""

        self.ci_buffer = np.empty(shape=(0,))

    def dump_ci_buff_overflow(self):
        """Doc."""

        if len(self.ci_buffer) > self.CONT_READ_BFFR_SZ:
            self.ci_buffer = self.ci_buffer[-self.CONT_READ_BFFR_SZ :]


class PixelClock(BaseDevice, NIDAQmx):
    """
    The pixel clock is fed to the DAQ board from the FPGA.
    Base frequency is 4 MHz. Used for scans, where it is useful to
    have divisible frequencies for both the laser pulses and AI/AO.
    """

    def __init__(self, param_dict):
        super().__init__(
            param_dict,
            task_types=("ci", "co"),
        )

        self.toggle(False)

    def _toggle(self, is_being_switched_on):
        """Doc."""

        if is_being_switched_on:
            self._start_co_clock_sync()
        else:
            self.close_all_tasks()

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
                clk_cnfg={"sample_mode": consts.NI.AcquisitionType.CONTINUOUS},
            )
            self.start_tasks("co")
        except Exception as exc:
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)


class SimpleDO(BaseDevice, NIDAQmx):
    """ON/OFF device (excitation laser, depletion shutter, TDC)."""

    def __init__(self, param_dict):
        super().__init__(
            param_dict,
            task_types=("do"),
        )
        self.toggle(False)

    def _toggle(self, is_being_switched_on):
        """Doc."""

        self.digital_write(is_being_switched_on)


class DepletionLaser(BaseDevice, PyVISA):
    """Control depletion laser through pyVISA"""

    min_SHG_temp = 53  # Celsius

    def __init__(self, param_dict):
        super().__init__(
            param_dict,
            read_termination="\r",
            write_termination="\r",
        )

        self.updt_time = 0.3
        self.state = None
        self.emission_state = False
        self.toggle(True, should_change_icons=False)

        if self.state is False:
            self.set_current(1500)

    def _toggle(self, is_being_switched_on):
        """Doc."""

        if is_being_switched_on:
            self.open_inst()
        else:
            if self.state is True:
                self.laser_toggle(False)
            try:
                self.close_inst()
            except VisaIOError:
                raise RuntimeError(
                    "Depletion laser disconnected during operation. Reconnect and reopen application to fix."
                )
            except AttributeError:
                raise RuntimeError("Depletion laser failed to open, no need to close.")

    def laser_toggle(self, is_being_switched_on):
        """Doc."""

        cmnd = f"setLDenable {int(is_being_switched_on)}"
        try:
            self.write(cmnd)
        except Exception as exc:
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)
        else:
            self.emission_state = is_being_switched_on
            self.change_icons("on" if is_being_switched_on else "off")

    def get_prop(self, prop):
        """Doc."""

        prop_cmnd_dict = {
            "temp": "SHGtemp",
            "curr": "LDcurrent 1",
            "pow": "Power 0",
        }
        cmnd = prop_cmnd_dict[prop]

        try:
            self.flush()  # get fresh response
            response = self.query(cmnd)
        except IndexError:
            return 0
        except Exception as exc:
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)
            return 0
        else:
            return response

    def set_power(self, value_mW):
        """Doc."""

        # check that value is within range
        if 99 <= value_mW <= 1000:
            try:
                # change the mode to power
                cmnd = "Powerenable 1"
                self.write(cmnd)
                # then set the power
                cmnd = f"Setpower 0 {value_mW}"
                self.write(cmnd)
            except Exception as exc:
                err_hndlr(exc, locals(), sys._getframe(), dvc=self)
        else:
            dialog.Error(error_txt="Power out of range").display()

    def set_current(self, value_mA):
        """Doc."""

        # check that value is within range
        if 1500 <= value_mA <= 2500:
            try:
                # change the mode to power
                cmnd = "Powerenable 0"
                self.write(cmnd)
                # then set the power
                cmnd = f"setLDcur 1 {value_mA}"
                self.write(cmnd)
            except Exception as exc:
                err_hndlr(exc, locals(), sys._getframe(), dvc=self)
        else:
            dialog.Error(error_txt="Current out of range").display()


class StepperStage(BaseDevice, PyVISA):
    """Control stepper stage through Arduino chip using PyVISA."""

    def __init__(self, param_dict):
        super().__init__(param_dict)

        self.toggle(True)
        self.toggle(False)

    def _toggle(self, is_being_switched_on):
        """Doc."""

        self.open_inst() if is_being_switched_on else self.close_inst()

    # TODO: try to make this async and include a 'release' command after async-sleeping for some time
    def move(self, dir, steps):
        """Doc."""

        cmd_dict = {
            "UP": f"my {-steps}",
            "DOWN": f"my {steps}",
            "LEFT": f"mx {steps}",
            "RIGHT": f"mx {-steps}",
        }
        try:
            self.write(cmd_dict[dir])
        except Exception as exc:
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)

    def release(self):
        """Doc."""

        try:
            self.write("ryx ")
        except Exception as exc:
            err_hndlr(exc, locals(), sys._getframe(), dvc=self)


class Camera(BaseDevice, Instrumental):
    """Doc."""

    vid_intrvl = 0.3

    def __init__(self, param_dict, loop, gui):
        super().__init__(
            param_dict,
        )

        self._loop = loop
        self._gui = gui
        self.state = False
        self.vid_state = False

    def _toggle(self, is_being_switched_on):
        """Doc."""

        self.init_cam() if is_being_switched_on else self.close_cam()

    async def shoot(self):
        """Doc."""

        if self.vid_state is False:
            img = self.grab_image()
            self._imshow(img)
        else:
            self.toggle_video(False)
            await asyncio.sleep(0.2)
            img = self.grab_image()
            self._imshow(img)
            self.toggle_video(True)

    def toggle_video(self, new_state):
        """Doc."""

        if self.vid_state != new_state:
            self.toggle_vid(new_state)
            self.vid_state = new_state

        if new_state is True:
            self._loop.create_task(sync_to_thread(self._vidshow))

    def _vidshow(self):
        """Doc."""

        while self.vid_state is True:
            img = self.get_latest_frame()
            if img is not None:
                self._imshow(img)
            time.sleep(self.vid_intrvl)

    def _imshow(self, img):
        """Plot image"""

        self._gui.figure.clear()
        ax = self._gui.figure.add_subplot(111)
        ax.imshow(img)
        self._gui.canvas.draw()
