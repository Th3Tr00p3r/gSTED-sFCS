"""Timeout module."""

import asyncio
import logging
import sys
from collections import deque
from contextlib import suppress

import utilities.helper as helper
from utilities.errors import err_hndlr

TIMEOUT_INTERVAL = 0.005  # 5 ms
GUI_UPDATE_INTERVAL = 0.2  # 200 ms
LOG_PATH = "./log/log"


class Timeout:
    """Doc."""

    cntr_avg_interval_s = 5

    def __init__(self, app) -> None:
        """Doc."""

        self._app = app
        self.main_gui = self._app.gui.main

        self.log_buffer_deque = deque([""], maxlen=50)

        self.cntr_dvc = self._app.devices.photon_detector
        self.scan_dvc = self._app.devices.scanners
        self.dep_dvc = self._app.devices.dep_laser

        # start
        self.not_finished = True
        logging.debug("Initiating timeout function.")
        self._app.loop.create_task(self.timeout())

    # MAIN
    async def timeout(self) -> None:
        """awaits all individual async loops, each with its own repeat time."""

        try:
            await asyncio.gather(
                self._read_ci_and_ai(),
                self._update_dep(),
                self._update_gui(),
            )
        except Exception as exc:
            err_hndlr(exc, sys._getframe(), locals())
        logging.debug("timeout function exited")

    async def _read_ci_and_ai(self) -> None:
        """Doc."""

        def fill_buffers() -> None:
            """Doc."""

            # CI (Photon Detector)
            if not self.cntr_dvc.error_dict:
                self.cntr_dvc.fill_ci_buffer()

            # AI (Scanners)
            if not self.scan_dvc.error_dict:
                self.scan_dvc.fill_ai_buffer()

        while self.not_finished:
            fill_buffers()
            await asyncio.sleep(TIMEOUT_INTERVAL)

    async def _update_gui(self) -> None:  # noqa: C901
        """Doc."""

        def updt_scn_pos():
            """Doc."""

            app = self._app

            with suppress(IndexError, AttributeError):
                # IndexError - AI buffer has just been initialized
                # AttributeError - Scanners failed to initialize
                (
                    x_ai,
                    y_ai,
                    z_ai,
                    x_ao_int,
                    y_ao_int,
                    z_ao_int,
                ) = app.devices.scanners.ai_buffer[-1]

                (x_um, y_um, z_um) = tuple(
                    (axis_vltg - axis_org) * axis_ratio
                    for axis_vltg, axis_ratio, axis_org in zip(
                        (x_ao_int, y_ao_int, z_ao_int),
                        app.devices.scanners.um_v_ratio,
                        app.devices.scanners.ORIGIN,
                    )
                )

                app.gui.main.xAIV.setValue(x_ai)
                app.gui.main.yAIV.setValue(y_ai)
                app.gui.main.zAIV.setValue(z_ai)

                app.gui.main.xAOVint.setValue(x_ao_int)
                app.gui.main.yAOVint.setValue(y_ao_int)
                app.gui.main.zAOVint.setValue(z_ao_int)

                app.gui.main.xAOum.setValue(x_um)
                app.gui.main.yAOum.setValue(y_um)
                app.gui.main.zAOum.setValue(z_um)

        def update_avg_counts(meas) -> None:
            """Doc."""

            if not self.cntr_dvc.error_dict:
                if meas.is_running and meas.scanning:
                    if meas.type == "SFCSSolution":
                        rate = meas.scan_params.ao_samp_freq_Hz
                    elif meas.type == "SFCSImage":
                        rate = meas.scan_params.line_freq_Hz * meas.scan_params.ppl
                else:
                    # no scanning measurement running
                    rate = None

                self.cntr_dvc.average_counts(GUI_UPDATE_INTERVAL, rate)
                self._app.gui.main.counts.setValue(self.cntr_dvc.avg_cnt_rate_khz)
                self.cntr_dvc.average_counts(self.cntr_avg_interval_s, rate)
                self._app.gui.main.counts2.setValue(self.cntr_dvc.avg_cnt_rate_khz)

        def updt_meas_progress(meas) -> None:
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
                meas.prog_bar_wdgt.set(progress)

        def update_log_wdgt() -> None:
            """Doc."""

            last_line = helper.file_last_line(LOG_PATH)

            if (last_line is not None) and (last_line.find("INFO") != -1):
                line_time, line_text = last_line[12:23], last_line[38:]
                last_line = line_time + line_text

                if last_line != self.log_buffer_deque[0]:
                    self.log_buffer_deque.appendleft(last_line)

                    text = "".join(self.log_buffer_deque)
                    self._app.gui.main.lastAction.setPlainText(text)

        while self.not_finished:

            # scanners
            if not self.scan_dvc.error_dict:
                updt_scn_pos()

            meas = self._app.meas

            # photon_detector count rate
            if not self.cntr_dvc.error_dict:
                update_avg_counts(meas)

            # Measurement progress bar and time left
            if self._app.meas.is_running:
                updt_meas_progress(meas)

            # log file widget
            update_log_wdgt()

            await asyncio.sleep(GUI_UPDATE_INTERVAL)

    async def _update_dep(self) -> None:
        """Update depletion laser GUI"""

        while self.not_finished:

            if not self.dep_dvc.error_dict:
                temp, pow, curr = (
                    self.dep_dvc.get_prop("temp"),
                    self.dep_dvc.get_prop("pow"),
                    self.dep_dvc.get_prop("curr"),
                )

                self.main_gui.depTemp.setValue(temp)
                self.main_gui.depActualPow.setValue(pow)
                self.main_gui.depActualCurr.setValue(curr)

                if temp < self.dep_dvc.MIN_SHG_TEMP:
                    self.main_gui.depTemp.setStyleSheet("background-color: red; color: white;")
                else:
                    self.main_gui.depTemp.setStyleSheet("background-color: white; color: black;")

            await asyncio.sleep(self.dep_dvc.update_interval_s)
