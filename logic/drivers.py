"""Drivers Module."""

import asyncio
import logging
import sys
import time
import queue
import threading
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
import warnings
from contextlib import suppress
from types import SimpleNamespace
from typing import List, Tuple, Union

import ftd2xx
import instrumental.drivers.cameras.uc480 as uc480
import nidaqmx as ni
import numpy as np
import pyvisa as visa
from instrumental import list_instruments
from instrumental.errors import TimeoutError
from nidaqmx.stream_readers import (
    CounterReader,  # AnalogMultiChannelReader for AI, if ever
)

from utilities.errors import IOError, err_hndlr
from utilities.helper import Limits, generate_numbers_from_string, update_attributes

warnings.simplefilter("ignore", UserWarning)  # TESTESTEST - visa related on laptop
warnings.simplefilter("ignore", ResourceWarning)  # TESTESTEST - visa related on laptop


class Ftd2xx:
    """Doc."""

    ftd2xx_dict = {
        "Single Channel Synchronous 245 FIFO": 0x40,
        "RTS-CTS": ftd2xx.defines.FLOW_RTS_CTS,
    }
    n_bytes: int = 200
    log_ref: str

    def __init__(self):

        update_attributes(self, self.ftd2xx_dict)
        super().__init__()

        # auto-find serial number from description
        num_devs = ftd2xx.createDeviceInfoList()
        for idx in range(num_devs):
            info_dict = ftd2xx.getDeviceInfoDetail(devnum=idx)
            if info_dict["description"].decode("utf-8") == self.description:
                self.serial = info_dict["serial"]
                print(f"(FTDI SN: {self.serial.decode('utf-8')} connection opened)")

        self.is_queried = False

    def open_instrument(self):
        """Doc."""
        # TODO: FIX THIS! IT LOOKS TERRIBLE

        try:
            self._inst = ftd2xx.ftd2xx.openEx(self.serial)
        except AttributeError:
            raise IOError(f"{self.log_ref} is not plugged in.")
        except ftd2xx.ftd2xx.DeviceError:
            raise IOError(f"{self.log_ref} MPD interface not closed!")
        with suppress(AttributeError):  # set params if they are defined
            self._inst.setBitMode(255, self.bit_mode)  # TODO: try setting to 0
        with suppress(AttributeError):  # set params if they are defined
            self._inst.setTimeouts(self.timeout_ms, self.timeout_ms)
        with suppress(AttributeError):  # set params if they are defined
            self._inst.setLatencyTimer(self.ltncy_tmr_val)
        with suppress(AttributeError):  # set params if they are defined
            self._inst.setFlowControl(self.flow_ctrl)
        with suppress(AttributeError):  # set params if they are defined
            self._inst.setUSBParameters(self.tx_size)
        with suppress(AttributeError):  # set params if they are defined
            self._inst.setBaudRate(self.baud_rate)

    def close_instrument(self) -> None:
        """Doc."""

        with suppress(AttributeError):
            try:
                self._inst.resetPort()
                self._inst.close()
            except ftd2xx.ftd2xx.DeviceError:
                raise IOError(f"{self.log_ref} is not plugged in.")

    def read(self) -> bytes:
        """Doc."""

        try:
            return self._inst.read(self.n_bytes)
        except ftd2xx.DeviceError:
            raise IOError("disconnected during measurement.")

    def write(self, byte_data: bytes) -> None:
        """Doc."""

        return self._inst.write(byte_data)

    async def mpd_command(
        self, command_list: Union[List[Tuple[str, Limits]], Tuple[str, Limits]]
    ) -> Tuple[Union[List[str], str], str]:
        """Doc."""

        self.is_queried = True

        # purge buffers before writing
        self.purge(should_purge_write=True)

        # convert (command, limt) list to MPD format
        if isinstance(command_list, tuple):  # single command
            command_list = [command_list]
        cmnd_str = self.limit_and_convert_to_mpd_format(command_list)

        # I/O
        self.write(cmnd_str.encode("utf-8"))

        # wait between reading and writing to allow time to generate response
        await asyncio.sleep(0.3)

        try:
            response = self.read().decode("utf-8").split(sep="#")
        except UnicodeDecodeError:
            raise IOError(
                f"Got a partial byte response for commands: {cmnd_str} - unable to decode..."
            )
        finally:
            self.is_queried = False

        if len(response) < len(command_list) - 1:
            raise IOError(
                f"Got {len(response)} responses for {len(command_list)} commands: {cmnd_str}..."
            )
        # single commands
        if len(response) == 1 or (len(response) == 2 and response[-1] == ""):
            single_response = response[0]
            if single_response:
                return single_response, cmnd_str
            else:
                return None, cmnd_str
        # multiple commands
        else:
            return [elem for elem in response if elem], cmnd_str

    def limit_and_convert_to_mpd_format(self, command_list: List[Tuple[str, Limits]]) -> str:
        """helper function"""

        command_chain = []
        for command, limits in command_list:
            if limits:
                value, *_ = generate_numbers_from_string(command)
                command_chain.append(f"{command[:2]}{limits.clamp(value)}")
            else:
                command_chain.append(command)
        return ";".join(command_chain) + "#"

    async def async_read(self) -> bytes:
        """Doc."""
        # NOTE: NOT CURRENTLY USED (using regular read())

        # todo: fix bug where something here blocks - I think it's some error in the async read function.
        # check this out: https://docs.python.org/3/library/asyncio-dev.html#detect-never-retrieved-exceptions
        return await self._inst.read(self.n_bytes)

    def purge(self, should_purge_write=False) -> None:
        """Doc."""

        try:
            self._inst.purge(ftd2xx.defines.PURGE_RX)
            if should_purge_write:
                self._inst.purge(ftd2xx.defines.PURGE_TX)
        except ftd2xx.ftd2xx.DeviceError as exc:
            raise IOError(exc)


class NIDAQmx:
    """Doc."""

    MIN_OUTPUT_RATE_Hz = int(1e3)
    CONT_READ_BFFR_SZ = int(1e5)
    AO_TIMEOUT = 0.1
    log_ref: str

    def __init__(self, task_types):
        super().__init__()
        self.task_types = task_types
        self.tasks = SimpleNamespace(**{type: [] for type in task_types})

        try:
            len(ni.system.system.System().devices)
        except FileNotFoundError:
            exc = IOError("National Instruments drivers not installed!")
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            print("(NIDAQmx connection opened.)")

    def start_tasks(self, task_type: str) -> None:
        """Doc."""

        task_list = getattr(self.tasks, task_type)
        [task.start() for task in task_list]

    def are_tasks_done(self, task_type: str) -> bool:
        """Doc."""

        task_list = getattr(self.tasks, task_type)
        task_status = [task.is_task_done() for task in task_list]
        return task_status.count(1) == len(task_status)

    def stop_tasks(self, task_type: str) -> None:
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
        [task.wait_until_done(timeout=5) for task in task_list if (task.name == task_name)]

    def create_ai_task(
        self,
        name: str,
        address: str,
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
                f"NI device address ({address}) is wrong, or data acquisition board is unplugged"
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
        return np.array(ai_task.read(number_of_samples_per_channel=n_samples))

    def analog_write(self, task_name: str, data: np.ndarray, auto_start=None) -> None:
        """Doc."""

        # TODO: change tasks to dict/namespace instead of list?
        ao_task = [task for task in self.tasks.ao if (task.name == task_name)][0]
        if auto_start is not None:
            ao_task.write(data, auto_start=auto_start, timeout=self.AO_TIMEOUT)
        else:
            ao_task.write(data, timeout=self.AO_TIMEOUT)

    def counter_stream_read(self, cont_read_buffer: np.ndarray) -> int:
        """
         Reads all available samples on board into self.cont_read_buffer
         (1D NumPy array, overwritten each read), and returns the
        number of samples read.
        """

        ci_task, *_ = self.tasks.ci  # only ever one task
        return ci_task.sr.read_many_sample_uint32(
            cont_read_buffer,
            number_of_samples_per_channel=ni.constants.READ_ALL_AVAILABLE,
        )

    def digital_write(self, address, _bool: bool) -> None:
        """Doc."""

        with ni.Task() as do_task:
            do_task.do_channels.add_do_chan(address)
            do_task.write(_bool)


class PyVISA:
    """Doc."""

    log_ref: str

    def __init__(
        self,
        read_termination="",
        write_termination="",
    ):
        super().__init__()
        self.read_termination = read_termination
        self.write_termination = write_termination
        self._rm = visa.ResourceManager()

    def autofind_address(self, model_query: str, **kwargs) -> None:
        """Doc."""

        # list all resource addresses
        resource_address_tuple = self._rm.list_resources()

        # open and save the opened ones in a list
        inst_list = []
        for resource_name in resource_address_tuple:
            with suppress(visa.errors.VisaIOError):
                # VisaIOError - failed to open instrument, skip
                # try to open instrument
                inst = self._rm.open_resource(
                    resource_name,
                    read_termination=self.read_termination,
                    write_termination=self.write_termination,
                    timeout=50,  # ms
                    open_timeout=50,  # ms
                    **kwargs,
                )
                # managed to open instrument, add to list.
                inst_list.append(inst)

        # 'model_query' the opened devices, and whoever responds is the one
        for idx, inst in enumerate(inst_list):
            try:
                self.model = inst.query(model_query)
            except visa.errors.VisaIOError as exc:
                if exc.abbreviation == "VI_ERROR_RSRC_BUSY":
                    raise IOError(f"{self.log_ref} couldn't be opened - a restart might help!")
                else:
                    pass
            else:
                self.address = resource_address_tuple[idx]
                break

        # close all saved resources
        with suppress(visa.errors.VisaIOError):
            # VisaIOError - this happens if device was disconnected during the autofind process...
            [inst.close() for inst in inst_list]

    def open_instrument(self, model_query=None, **kwargs) -> None:
        """Doc."""

        # auto-find serial connection for depletion laser
        if model_query is not None and self.address is None:
            self.autofind_address(model_query, **kwargs)

        try:
            # open
            self._rsrc = self._rm.open_resource(
                self.address,
                read_termination=self.read_termination,
                write_termination=self.write_termination,
                timeout=100,  # ms
                open_timeout=50,  # ms
                **kwargs,
            )
            print(f"(VISA: {self._rsrc.resource_info.alias} connection opened.)")
        except (AttributeError, visa.errors.VisaIOError, ValueError):
            # failed to auto-find address for VISA device
            raise IOError(
                f"{self.log_ref} couldn't be opened - it is either turned off, unplugged or the address is used by another process"
            )

    def close_instrument(self) -> None:
        """Doc."""

        try:
            self._rsrc.close()
        except AttributeError:
            raise IOError(f"{self.log_ref} was never opened")
        except visa.errors.VisaIOError:
            raise IOError(f"{self.log_ref} disconnected during operation.")

    def read(self, n_bytes=None) -> Union[str, bytes]:
        """Doc."""

        try:
            if n_bytes is not None:
                return self._rsrc.read_bytes(n_bytes)
            else:
                return self._rsrc.read()
        except visa.errors.VisaIOError as exc:
            if exc.abbreviation == "VI_ERROR_TMO":
                raise IOError("Resource timed-out during read.")
            else:
                raise exc

    def write(self, cmnd: str) -> None:
        """Sends a command to the VISA instrument."""

        self._rsrc.write(cmnd)

    def query(self, cmnd: str) -> str:
        """Doc."""

        self.flush()  # get fresh response
        try:
            return self._rsrc.query(cmnd)
        except visa.errors.VisaIOError:
            raise IOError(f"{self.log_ref} disconnected! Reconnect and restart.")

    def flush(self) -> None:
        """Doc."""

        mask = visa.constants.BufferOperation(4) | visa.constants.BufferOperation(6)
        try:
            self._rsrc.flush(mask)
        except visa.errors.VisaIOError:
            raise IOError(f"{self.log_ref} disconnected! Reconnect and restart.")


class Spinnaker:
    """Doc."""

    log_ref: str

    ...


CAMERA_CALL_TIMEOUT_S = 5.0  # a camera call that takes longer than this is taken as hung
CAMERA_OPEN_TIMEOUT_S = 20.0
CAMERA_GRAB_TIMEOUT_S = 15.0  # a single (non-video) frame, at up to ~1 s exposure
FRAME_WAIT_MS = 500  # how long one request waits for a new video frame (in the worker)
CAMERA_STALL_S = 5.0  # video on but no new frame for this long: restart the live video


class _CameraWorker:
    """
    One daemon thread that runs submitted calls one at a time. A daemon, unlike the
    threads of concurrent.futures' ThreadPoolExecutor, because a call stuck in the driver
    must not stop the program from quitting (the executor's threads are joined at exit).
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="uc480 camera", daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            future, fn, args, kwargs = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)

    def submit(self, fn, *args, **kwargs) -> Future:
        future = Future()
        self._queue.put((future, fn, args, kwargs))
        return future

    def shutdown(self):
        """Stop after the current call (if it ever returns); queued calls are dropped."""

        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                item[0].cancel()
        self._queue.put(None)


class UC480:
    """
    Thorlabs/IDS uc480 camera, via instrumental-lib.

    Every call into the camera driver runs on a worker thread of this camera's own, one
    call at a time, with a time limit. They used to run on the GUI thread, so a driver
    call that never returned (which happens with these cameras, typically right after a
    button press) froze the whole program, and 'Restart Camera' could not even run.
    Now such a call is abandoned after CAMERA_CALL_TIMEOUT_S, the camera is reported as
    not responding (and which call it was stuck in), and the GUI carries on; 'Restart
    Camera' then starts a fresh worker. Running the calls one at a time also stops a
    parameter change from landing in the middle of a frame read.
    """

    log_ref: str  # NOTE: these are for mypy to be silent. what's needed here is an ABC.
    last_snapshot: np.ndarray

    def __init__(self):

        self._inst = None
        self.is_waiting_for_frame = False  # kept for the timeout loop; frames wait in the worker
        self.frame_count = 0  # new frames delivered since the program started
        self.last_frame_time = None  # time.monotonic() of the last new frame
        self._hung_call = None  # the call the worker is stuck in, if any
        self._worker_generation = 0
        self._worker = _CameraWorker()
        self._last_video_restart = 0.0
        self._is_stall_reported = False

        if not list_instruments(module="cameras"):
            exc = IOError("No UC480 cameras detected.")
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)

        super().__init__()

    # ------------------------------------------------------------------ worker

    def _not_responding_error(self) -> IOError:
        return IOError(
            f"{self.log_ref} is not responding (driver stuck in '{self._hung_call}'). "
            "Replug its USB cable, then press Restart Camera."
        )

    def _mark_hung(self, what: str, timeout: float) -> IOError:
        self._hung_call = what
        logging.error(
            f"{self.log_ref}: driver call '{what}' did not return within {timeout:g} s - "
            "camera marked as not responding"
        )
        return self._not_responding_error()

    def _call(self, what: str, fn, *args, timeout=CAMERA_CALL_TIMEOUT_S, **kwargs):
        """Run a driver call on the camera's worker and wait for it, at most 'timeout' s."""

        if self._hung_call is not None:
            raise self._not_responding_error()
        future = self._worker.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            raise self._mark_hung(what, timeout)

    async def _acall(self, what: str, fn, *args, timeout=CAMERA_CALL_TIMEOUT_S, **kwargs):
        """The same, awaited - the GUI keeps running while the camera works."""

        if self._hung_call is not None:
            raise self._not_responding_error()
        generation = self._worker_generation
        future = self._worker.submit(fn, *args, **kwargs)
        try:
            result = await asyncio.wait_for(asyncio.wrap_future(future), timeout)
        except asyncio.TimeoutError:
            if generation != self._worker_generation:
                raise _StaleCall()
            raise self._mark_hung(what, timeout)
        except Exception:
            if generation != self._worker_generation:
                raise _StaleCall()
            raise
        if generation != self._worker_generation:
            raise _StaleCall()  # the camera was restarted meanwhile - result belongs to the old one
        return result

    def reset_camera_worker(self) -> None:
        """
        Abandon the worker thread (if a call is stuck in it, it stays stuck there, but
        no longer blocks anything) and the old driver handle, and start a fresh worker.
        """

        if self._hung_call is not None:
            logging.warning(
                f"{self.log_ref}: abandoning the call stuck in '{self._hung_call}' and its "
                "driver handle"
            )
            self._inst = None  # closing it would only get stuck too
        self._worker.shutdown()
        self._worker = _CameraWorker()
        self._worker_generation += 1
        self._hung_call = None
        self._is_stall_reported = False

    # ------------------------------------------------------------------ driver calls

    def open_instrument(self):
        """Doc."""

        def _open():
            return uc480.UC480_Camera(serial=self.serial.encode(), reopen_policy="new")

        try:
            self._inst = self._call("open", _open, timeout=CAMERA_OPEN_TIMEOUT_S)
        except IOError:
            raise
        except Exception as exc:
            # general 'Exception' is unavoidable due to bad exception handeling in instrumental-lib...
            raise IOError(f"{self.log_ref} disconnected - {exc}")
        print(f"(UC480 SN: {self._inst._paramset['serial'].decode('utf-8')}) connection opened")

    def close_instrument(self):
        """Doc."""

        if self._inst is not None:
            inst, self._inst = self._inst, None
            self._call("close", inst.close)

    def _start_live_video(self):
        inst = self._inst
        if self.is_auto_exposure_on:
            inst.start_live_video()
            inst.set_auto_exposure(True)
        else:
            inst.start_live_video(exposure_time=inst._get_exposure())

    def toggle_video_mode(self, should_turn_on: bool) -> None:
        """Doc."""

        try:
            if should_turn_on:
                self._call("start live video", self._start_live_video)
                self.last_frame_time = time.monotonic()
                self._is_stall_reported = False
            else:
                self._call("stop live video", self._inst.stop_live_video)
        except uc480.UC480Error:
            raise IOError(f"{self.log_ref} disconnected after initialization.")
        else:
            self.is_in_video_mode = should_turn_on

    async def capture_image(self) -> np.ndarray:
        """Doc."""

        def _grab():
            inst = self._inst
            image = inst.grab_image(copy=False, exposure_time=inst._get_exposure())
            return np.flipud(np.array(image, copy=True))

        try:
            return await self._acall("grab image", _grab, timeout=CAMERA_GRAB_TIMEOUT_S)
        except _StaleCall:
            return self.last_snapshot
        except uc480.UC480Error:
            raise IOError(f"{self.log_ref} disconnected after initialization.")
        except TimeoutError:
            raise IOError(f"{self.log_ref} timed out.")

    def _fetch_new_frame(self):
        """In the worker: wait for a new video frame, and return a copy of it (or None)."""

        inst = self._inst
        if not inst.wait_for_frame(f"{FRAME_WAIT_MS}ms"):
            return None
        # a copy: the driver writes the next frame into the same memory
        return np.flipud(np.array(inst.latest_frame(copy=False), copy=True))

    async def get_latest_frame(self) -> np.ndarray:
        """The newest video frame, or the last one if no new frame came in time."""

        try:
            frame = await self._acall("wait for frame", self._fetch_new_frame)
        except _StaleCall:
            return self.last_snapshot
        except uc480.UC480Error:
            raise IOError(f"{self.log_ref} disconnected after initialization.")

        now = time.monotonic()
        if frame is not None:
            self.frame_count += 1
            self.last_frame_time = now
            if self._is_stall_reported:
                logging.info(f"{self.log_ref}: frames are coming in again")
            self._is_stall_reported = False
            return frame

        # no new frame: if that has gone on for a while, restart the live video once in a
        # while (the driver call itself returned, so the camera is alive - just not sending)
        if self.last_frame_time is not None and now - self.last_frame_time > CAMERA_STALL_S:
            if not self._is_stall_reported:
                logging.warning(
                    f"{self.log_ref}: no new frame for {now - self.last_frame_time:.0f} s "
                    f"({self.frame_count} frames so far) - restarting its live video"
                )
                self._is_stall_reported = True
            if now - self._last_video_restart > 2 * CAMERA_STALL_S:
                self._last_video_restart = now
                await self._restart_live_video()
        return self.last_snapshot

    async def _restart_live_video(self) -> None:
        def _restart():
            with suppress(uc480.UC480Error):
                self._inst.stop_live_video()
            self._start_live_video()

        try:
            await self._acall("restart live video", _restart)
        except _StaleCall:
            pass
        except Exception as exc:
            logging.warning(f"{self.log_ref}: could not restart the live video [{exc}]")
        else:
            self.last_frame_time = time.monotonic()

    def update_parameter_ranges(self):
        """Doc."""

        if self._inst is None:  # camera not connected
            return

        def _read():
            inst = self._inst
            return (
                tuple(inst._dev.PixelClock(uc480.lib.PIXELCLOCK_CMD_GET_RANGE)),
                (1, inst.pixelclock._magnitude / 1.6),
                tuple(inst._dev.Exposure(uc480.lib.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE)),
            )

        try:
            pixel_clock, framerate, exposure = self._call("read parameter ranges", _read)
        except (AttributeError, IOError, uc480.UC480Error) as exc:
            logging.debug(f"{self.log_ref}: parameter ranges not read [{exc}]")
            return
        *self.pixel_clock_range, _ = pixel_clock
        self.framerate_range = framerate
        *self.exposure_range, _ = exposure

    def set_parameter(self, name, value) -> None:
        """Doc."""

        if name not in {"pixel_clock", "framerate", "exposure"}:
            raise ValueError(f"Unknown parameter '{name}'.")

        valid_value = Limits(getattr(self, f"{name}_range")).clamp(value)

        def _set():
            inst = self._inst
            if name == "pixel_clock":
                inst._dev.PixelClock(uc480.lib.PIXELCLOCK_CMD_SET, int(valid_value))
            elif name == "framerate":
                inst._dev.SetFrameRate(valid_value)
            elif name == "exposure":
                inst._set_exposure(f"{valid_value}ms")

        try:
            self._call(f"set {name}", _set)
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        except uc480.UC480Error:
            exc = IOError(f"{self.log_ref} was not properly closed.\nRestart to fix.")
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            if name in ("pixel_clock", "framerate"):
                self.update_parameter_ranges()

    def set_auto_exposure(self, should_turn_on: bool) -> None:
        """Doc."""

        try:
            self._call("set auto exposure", self._inst.set_auto_exposure, should_turn_on)
        except IOError as exc:
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        except uc480.UC480Error:
            exc = IOError(f"{self.log_ref} was not properly closed.\nRestart to fix.")
            err_hndlr(exc, sys._getframe(), locals(), dvc=self)
        else:
            self.is_auto_exposure_on = should_turn_on


class _StaleCall(Exception):
    """A camera call that finished after the camera was restarted - its result is ignored."""
