"""Timeout module."""

import asyncio
import logging
import os
import sys
from collections import deque

from utilities.errors import DeviceError, err_hndlr

# import time
# import numpy as np

TIMEOUT = 0.010  # seconds (10 ms)
LOG_PATH = "./log/log"


class Timeout:
    """Doc."""

    def __init__(self, app) -> None:
        """Doc."""

        self._app = app
        self.main_gui = self._app.gui.main

        # initial intervals (some change during run)
        self.updt_intrvl = {
            "gui": 0.2,
            "dep": self._app.devices.dep_laser.UPDATE_TIME,
            "cntr_avg": self._app.devices.photon_detector.UPDATE_TIME,
        }

        maxlen = self._app.gui.main.logNumLinesSlider.value()
        self.log_buffer_deque = deque([""], maxlen=maxlen)

        self.um232_buff_sz = self._app.devices.UM232H.tx_size

        self.cntr_dvc = self._app.devices.photon_detector
        self.dep_dvc = self._app.devices.dep_laser

        # start
        self.not_finished = True
        logging.debug("Initiating timeout function.")
        self._app.loop.create_task(self.timeout())

    # MAIN
    async def timeout(self) -> None:
        """awaits all individual async loops, each with its own repeat time."""

        await asyncio.gather(
            self._updt_CI_and_AI(),
            self._update_dep(),
            self._updt_application_log(),
            self._update_gui(),
            # self._updt_um232h_status(), # TODO: this seems to cause an issue during measurements (noticed in solution scan) - try to see if it does and catch the error
        )

        logging.debug("timeout function exited")

    async def _updt_application_log(self) -> None:
        """Doc."""

        def get_last_line(file_path) -> str:
            """
            Return the last line of a text file.
            (https://stackoverflow.com/questions/46258499/read-the-last-line-of-a-file-in-python)
            """

            try:
                with open(file_path, "rb") as f:
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b"\n":
                        f.seek(-2, os.SEEK_CUR)
                    last_line = f.readline().decode()
            except FileNotFoundError:
                return "Log File Not Found!!!"
            else:
                return last_line

        while self.not_finished:

            new_maxlen = self._app.gui.main.logNumLinesSlider.value()
            if self.log_buffer_deque.maxlen != new_maxlen:
                self.log_buffer_deque = deque(self.log_buffer_deque, maxlen=new_maxlen)

            last_line = get_last_line(LOG_PATH)

            if last_line.find("INFO") != -1:
                line_time = last_line[12:23]
                line_text = last_line[38:]
                last_line = line_time + line_text

                if last_line != self.log_buffer_deque[0]:
                    self.log_buffer_deque.appendleft(last_line)

                    text = "".join(self.log_buffer_deque)
                    self._app.gui.main.lastAction.setPlainText(text)

            await asyncio.sleep(0.3)

    async def _updt_CI_and_AI(self) -> None:
        """Doc."""

        #        N_TOC = 100 # TESTING
        #        tocs = np.zeros(shape=(N_TOC,)) # TESTING
        #        toc_idx = 0 # TESTING
        #        tic = time.perf_counter() # TESTING

        while self.not_finished:

            #            if toc_idx == N_TOC: # TESTING
            #                print(f"average timeout elapsed: {tocs.mean()}. Should be: {TIMEOUT}") # TESTING
            #                toc_idx = 0 # TESTING
            #            tocs[toc_idx] = time.perf_counter() - tic # TESTING
            #            toc_idx += 1 # TESTING

            # CI (Photon Detector)
            if not self._app.devices.photon_detector.error_dict:
                self._app.devices.photon_detector.fill_ci_buffer()

            # AI (Scanners)
            if not self._app.devices.scanners.error_dict:
                self._app.devices.scanners.fill_ai_buffer()

            #            tic = time.perf_counter() # TESTING

            await asyncio.sleep(TIMEOUT)

    async def _update_gui(self) -> None:
        """Doc."""

        def updt_scn_pos(app):
            """Doc."""

            try:
                (
                    x_ai,
                    y_ai,
                    z_ai,
                    x_ao_int,
                    y_ao_int,
                    z_ao_int,
                ) = app.devices.scanners.ai_buffer[-1]

            except IndexError:
                # AI buffer has just been initialized
                pass

            else:
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

        def updt_meas_progbar(meas) -> None:
            """Doc."""

            try:
                if meas.type == "SFCSSolution":
                    if not meas.cal:
                        progress = (
                            (meas.total_time_passed + meas.time_passed)
                            / (meas.total_duration * meas.duration_multiplier)
                            * meas.prog_bar_wdgt.obj.maximum()
                        )
                    else:
                        progress = 0
                elif meas.type == "SFCSImage":
                    progress = (
                        meas.time_passed
                        / meas.est_total_duration
                        * meas.prog_bar_wdgt.obj.maximum()
                    )
                meas.prog_bar_wdgt.set(progress)

            except AttributeError:
                # happens when depletion is turned on before beginning measurement (5 s wait)
                pass

            except Exception as exc:
                err_hndlr(exc, locals(), sys._getframe())

        while self.not_finished:

            # scanners
            updt_scn_pos(self._app)

            # photon_detector count rate
            try:
                self.cntr_dvc.average_counts(self.updt_intrvl["cntr_avg"])
            except DeviceError:
                pass
            else:
                self._app.gui.main.counts.setValue(self.cntr_dvc.avg_cnt_rate)

            # Measurement progress bar
            if self._app.meas.is_running:
                updt_meas_progbar(self._app.meas)

            await asyncio.sleep(self.updt_intrvl["gui"])

    async def _updt_um232h_status(self):
        """Doc."""

        while self.not_finished:

            try:
                rx_bytes = self._app.devices.UM232H.get_status()

            except DeviceError:
                pass

            else:
                fill_perc = rx_bytes / self.um232_buff_sz * 100
                if fill_perc > 90:
                    logging.warning("UM232H buffer might be overfilling")

            finally:
                await asyncio.sleep(self.updt_intrvl["gui"])

    async def _update_dep(self) -> None:
        """Update depletion laser GUI"""

        while self.not_finished:

            try:
                temp, pow, curr = (
                    self.dep_dvc.get_prop("temp"),
                    self.dep_dvc.get_prop("pow"),
                    self.dep_dvc.get_prop("curr"),
                )

            except DeviceError:
                # do nothing if error in depletion laser
                pass

            else:
                self.main_gui.depTemp.setValue(temp)
                self.main_gui.depActualPow.setValue(pow)
                self.main_gui.depActualCurr.setValue(curr)

                if temp < self.dep_dvc.MIN_SHG_TEMP:
                    self.main_gui.depTemp.setStyleSheet("background-color: red; color: white;")
                else:
                    self.main_gui.depTemp.setStyleSheet("background-color: white; color: black;")

            finally:
                await asyncio.sleep(self.updt_intrvl["dep"])
