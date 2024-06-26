"""Timeout module."""

import asyncio
import logging
import sys
import time
from collections import deque
from contextlib import suppress

import utilities.helper as helper
from utilities.errors import DeviceError, err_hndlr

TIMEOUT_INTERVAL = 0.02  # 20 ms
GUI_UPDATE_INTERVAL = 0.2  # 200 ms


class Timeout:
    """Doc."""

    cntr_avg_interval_s = 5

    def __init__(self, app) -> None:
        """Doc."""

        self._app = app
        self.main_gui = self._app.gui.main

        self.log_buffer_deque = deque([""], maxlen=50)

        self.cntr_dvc = self._app.devices.photon_counter
        self.scan_dvc = self._app.devices.scanners

        # start
        self.not_finished = True
        logging.debug("Initiating timeout function.")
        self._app.loop.create_task(self.timeout())

    # MAIN LOOP
    async def timeout(self) -> None:
        """awaits all individual async loops, each with its own repeat time."""

        try:
            await asyncio.gather(
                self._read_ci_and_ai(),
                self._update_lasers(),
                self._update_delayer_temperature(),
                self._update_spad_status(),
                self._update_main_gui(),
            )
        except Exception as exc:
            err_hndlr(exc, sys._getframe(), locals())
        logging.debug("timeout function exited")

    async def _read_ci_and_ai(self) -> None:
        """Doc."""

        while self.not_finished:
            if not self._app.are_ai_ci_paused:
                # CI (Photon Counter)
                if not self.cntr_dvc.error_dict:
                    self.cntr_dvc.fill_ci_buffer()
                # AI (Scanners)
                if not self.scan_dvc.error_dict:
                    self.scan_dvc.fill_ai_buffer()
            await asyncio.sleep(TIMEOUT_INTERVAL)

    async def _update_main_gui(self) -> None:  # noqa: C901
        """Doc."""

        def updat_ai():
            """Doc."""

            app = self._app

            with suppress(IndexError, AttributeError):
                # IndexError - AI buffer has just been initialized
                # AttributeError - Scanners failed to initialize

                x_ai, y_ai, z_ai = app.devices.scanners.ai
                app.gui.main.xAIV.setValue(x_ai)
                app.gui.main.yAIV.setValue(y_ai)
                app.gui.main.zAIV.setValue(z_ai)

                x_ao_int, y_ao_int, z_ao_int = app.devices.scanners.ao_int
                app.gui.main.xAOVint.setValue(x_ao_int)
                app.gui.main.yAOVint.setValue(y_ao_int)
                app.gui.main.zAOVint.setValue(z_ao_int)

                x_um, y_um, z_um = app.devices.scanners.origin_disp_um
                app.gui.main.xAOum.setValue(x_um)
                app.gui.main.yAOum.setValue(y_um)
                app.gui.main.zAOum.setValue(z_um)

        def update_avg_counts(meas) -> None:
            """Doc."""

            if not self.cntr_dvc.error_dict:
                if meas.is_running and meas.scanning:
                    if meas.type == "SFCSSolution":
                        rate = meas.scan_params["ao_sampling_freq_hz"]
                    elif meas.type == "SFCSImage":
                        rate = meas.scan_params["line_freq_hz"] * meas.scan_params["ppl"] / 2
                else:
                    # no scanning measurement running
                    rate = None

                self.cntr_dvc.average_counts(GUI_UPDATE_INTERVAL, rate)
                self._app.gui.main.counts.setValue(self.cntr_dvc.avg_cnt_rate_khz)
                self.cntr_dvc.average_counts(self.cntr_avg_interval_s, rate)
                self._app.gui.main.counts2.setValue(self.cntr_dvc.avg_cnt_rate_khz)

        def update_meas_progress(meas) -> None:
            """Doc."""

            with suppress(AttributeError):
                # AttributeError - depletion  turned on before beginning measurement (5 s wait)
                if meas.type == "SFCSSolution":
                    progress = (
                        meas.time_passed_s / meas.duration_s * meas.prog_bar_wdgt.obj.maximum()
                    )
                    meas.time_left_wdgt.set(round(meas.duration_s - meas.time_passed_s))
                elif meas.type == "SFCSImage":
                    progress = (
                        meas.time_passed_s
                        / meas.est_total_duration_s
                        * meas.prog_bar_wdgt.obj.maximum()
                    )
                meas.prog_bar_wdgt.set(round(progress))

        def update_log_wdgt() -> None:
            """Doc."""

            try:
                last_line = helper.file_last_line(self._app.log_file_path)
            except UnicodeDecodeError:
                # TODO: what's this?
                print("Something wrong with log file...")
            else:
                if last_line is None:
                    logging.info("Log file initialized.")
                elif last_line.find("INFO") != -1:
                    line_time, line_text = last_line[12:23], last_line[38:]
                    last_line = line_time + line_text

                    if last_line != self.log_buffer_deque[0]:
                        self.log_buffer_deque.appendleft(last_line)

                        text = "".join(self.log_buffer_deque)
                        self._app.gui.main.lastAction.setPlainText(text)

        while self.not_finished:

            meas = self._app.meas

            # TODO: add GUI switch to override this condition (for testing)
            if not meas.is_running or meas.scan_type == "static":
                # scanners
                if not self.scan_dvc.error_dict:
                    updat_ai()

                # log file widget
                update_log_wdgt()

            if meas.is_running:
                # MeasurementProcedure progress bar and time left
                update_meas_progress(meas)

            #            else:
            # DeviceError - camera error
            video_cams = [cam for cam in self.main_gui.impl.cameras if cam.is_in_video_mode]
            for video_cam in video_cams:
                with suppress(DeviceError):
                    if not video_cam.is_waiting_for_frame:
                        self._app.loop.create_task(video_cam.get_image())

            # photon_counter count rate
            if not self.cntr_dvc.error_dict:
                update_avg_counts(meas)

            await asyncio.sleep(GUI_UPDATE_INTERVAL)

    async def _update_lasers(self) -> None:
        """Update depletion laser GUI"""

        while self.not_finished:

            dep_dvc = self._app.devices.dep_laser
            exc_dvc = self._app.devices.exc_laser
            dep_shutter_dvc = self._app.devices.dep_shutter
            video_cams = [cam for cam in self.main_gui.impl.cameras if cam.is_in_video_mode]

            if not dep_dvc.error_dict and not self._app.meas.is_running and not video_cams:

                # get time of operation of laser head (once)
                if dep_dvc.time_of_operation_hr is None:
                    await asyncio.sleep(
                        1
                    )  # TODO: ensure that this prevents depletion error on startup
                    dep_dvc.time_of_operation_hr = dep_dvc.get_prop("on_time")
                    dep_dvc.hours_of_operation_widget.set(dep_dvc.time_of_operation_hr)

                # Track SHG temperature at all times
                self.main_gui.depTemp.setValue(temp := dep_dvc.get_prop("temp"))
                if temp < dep_dvc.MIN_SHG_TEMP_C:
                    self.main_gui.depTemp.setStyleSheet("background-color: red; color: white;")
                else:
                    self.main_gui.depTemp.setStyleSheet("background-color: white; color: black;")

                if dep_dvc.is_emission_on:
                    # TODO: can commands to dep be united as in MPD commands?
                    # TODO: these widgets should belong to device and set by it, possibly inside 'get_prop'.
                    self.main_gui.depActualPow.setValue(dep_dvc.get_prop("pow"))
                    self.main_gui.depActualCurr.setValue(dep_dvc.get_prop("curr"))

            # automatic shutdowns when not measuring
            if not dep_dvc.error_dict and dep_dvc.is_emission_on and not self._app.meas.is_running:
                mins_since_turned_on = (time.perf_counter() - dep_dvc.turn_on_time) / 60
                if mins_since_turned_on > dep_dvc.off_timer_min:
                    logging.info(
                        f"Shutting down {dep_dvc.log_ref} automatically (idle for {dep_dvc.off_timer_min} mins)"
                    )
                    dep_dvc.laser_toggle(False)
                    dep_shutter_dvc.toggle(False)

            if not exc_dvc.error_dict and exc_dvc.is_on and not self._app.meas.is_running:
                mins_since_turned_on = (time.perf_counter() - exc_dvc.turn_on_time) / 60
                if mins_since_turned_on > exc_dvc.off_timer_min:
                    logging.info(
                        f"Shutting down {exc_dvc.log_ref} automatically (idle for {exc_dvc.off_timer_min} mins)"
                    )
                    exc_dvc.toggle(False)

            await asyncio.sleep(dep_dvc.update_interval_s)

    async def _update_delayer_temperature(self) -> None:
        """Update delayer temperature"""

        while self.not_finished:

            delayer_dvc = self._app.devices.delayer

            if (
                (not delayer_dvc.error_dict)
                and (not self._app.meas.is_running)
                and (not delayer_dvc.is_queried)
            ):
                if delayer_dvc.is_on:
                    with suppress(ValueError, TypeError):
                        temp, _ = await delayer_dvc.mpd_command(("RT", None))
                        self.main_gui.psdTemp.setValue(
                            float(temp)
                        )  # TODO: move this (widget) within the device

            await asyncio.sleep(delayer_dvc.update_interval_s)

    async def _update_spad_status(self) -> None:
        """Doc."""

        while self.not_finished:

            spad_dvc = self._app.devices.spad
            delayer_dvc = self._app.devices.delayer
            exc_dvc = self._app.devices.exc_laser

            if (
                not spad_dvc.error_dict
                and not spad_dvc.is_paused
                and not self._app.meas.is_running
                and not spad_dvc.is_queried
            ):

                # display status and mode
                await spad_dvc.get_stats()

                # gating
                icon_name = "on" if delayer_dvc.is_on and exc_dvc.is_on else "off"
                spad_dvc.change_icons(icon_name, led_widget_name="gate_led_widget")

            await asyncio.sleep(spad_dvc.update_interval_s)
