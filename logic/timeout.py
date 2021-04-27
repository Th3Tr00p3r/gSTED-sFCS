# -*- coding: utf-8 -*-
"""Timeout module."""

import asyncio
import logging
import os
import time
from collections import deque
from typing import NoReturn

import utilities.constants as consts


class Timeout:
    """Doc."""

    def __init__(self, app) -> NoReturn:
        """Doc."""

        self._app = app

        # initial intervals (some change during run)
        self.updt_intrvl = {
            "gui": 0.2,
            "dep": self._app.devices.DEP_LASER.updt_time,
            "cntr_avg": self._app.devices.COUNTER.updt_time,
        }

    # MAIN
    async def _main(self) -> NoReturn:
        """Doc."""

        await asyncio.gather(
            self._updt_CI_and_AI(),
            self._update_avg_counts(),
            self._update_dep(),  # TODO: try (in lab) to have each call in thread (seperately).
            self._updt_current_state(),
            self._update_gui(),
        )
        logging.debug("_main function exited")

    def start(self) -> NoReturn:
        """Doc."""

        self.not_finished = True
        self._app.loop.create_task(self._main())
        logging.debug("Starting main timer.")

    def finish(self) -> NoReturn:
        """Doc."""

        self.not_finished = False

    async def _updt_current_state(self) -> NoReturn:
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
                last_line = "Log File Not Found!!!"
                self._app.gui.main.lastAction.setPlainText(last_line)
            finally:
                return last_line

        now_timestamp = time.strftime("%H:%M:%S", time.localtime())
        first_line = f"[{now_timestamp}] Application Started"

        maxlen = self._app.gui.main.logNumLinesSlider.value()
        buffer_deque = deque([first_line], maxlen=maxlen)

        while self.not_finished:
            new_maxlen = self._app.gui.main.logNumLinesSlider.value()
            if buffer_deque.maxlen != new_maxlen:
                buffer_deque = deque(buffer_deque, maxlen=new_maxlen)

            last_line = get_last_line(consts.LOG_FOLDER_PATH + "log")

            if last_line.find("INFO") != -1:
                line_time = last_line[12:23]
                line_text = last_line[38:]
                last_line = line_time + line_text

                if last_line != buffer_deque[0]:
                    buffer_deque.appendleft(last_line)

                    text = "".join(buffer_deque)
                    self._app.gui.main.lastAction.setPlainText(text)

            await asyncio.sleep(0.3)

    async def _updt_CI_and_AI(self) -> NoReturn:
        """Doc."""

        while self.not_finished:

            # COUNTER
            if self._app.devices.COUNTER.error_dict is None:
                self._app.devices.COUNTER.fill_ci_buffer()
                if self._app.meas.type not in {"SFCSSolution", "SFCSImage"}:
                    self._app.devices.COUNTER.dump_ci_buff_overflow()

            # AI
            if self._app.devices.SCANNERS.error_dict is None:
                self._app.devices.SCANNERS.fill_ai_buffer()
                if self._app.meas.type not in {"SFCSSolution", "SFCSImage"}:
                    self._app.devices.SCANNERS.dump_ai_buff_overflow()

            await asyncio.sleep(consts.TIMEOUT)

    async def _update_gui(self) -> NoReturn:
        """Doc."""

        def updt_scn_pos(app):
            """Doc."""

            if self._app.devices.SCANNERS.error_dict is None:
                try:
                    (
                        x_ai,
                        y_ai,
                        z_ai,
                        x_ao_int,
                        y_ao_int,
                        z_ao_int,
                    ) = self._app.devices.SCANNERS.ai_buffer[:, -1]
                except IndexError:
                    # AI buffer has just been initialized
                    pass
                else:
                    (x_um, y_um, z_um) = tuple(
                        (axis_vltg - axis_org) * axis_ratio
                        for axis_vltg, axis_ratio, axis_org in zip(
                            (x_ao_int, y_ao_int, z_ao_int),
                            self._app.devices.SCANNERS.um_V_ratio,
                            self._app.devices.SCANNERS.origin,
                        )
                    )

                    self._app.gui.main.xAIV.setValue(x_ai)
                    self._app.gui.main.yAIV.setValue(y_ai)
                    self._app.gui.main.zAIV.setValue(z_ai)

                    self._app.gui.main.xAOVint.setValue(x_ao_int)
                    self._app.gui.main.yAOVint.setValue(y_ao_int)
                    self._app.gui.main.zAOVint.setValue(z_ao_int)

                    self._app.gui.main.xAOum.setValue(x_um)
                    self._app.gui.main.yAOum.setValue(y_um)
                    self._app.gui.main.zAOum.setValue(z_um)

        def updt_meas_progbar(meas) -> NoReturn:
            """Doc."""

            if (self._app.devices.UM232H.error_dict is None) and meas.is_running:
                if meas.type == "FCS":
                    progress = (
                        meas.time_passed
                        / (meas.duration * meas.duration_multiplier)
                        * meas.prog_bar_wdgt.obj.maximum()
                    )
                elif meas.type == "SFCSSolution":
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
                        / meas.est_duration
                        * meas.prog_bar_wdgt.obj.maximum()
                    )
                meas.prog_bar_wdgt.set(progress)

        while self.not_finished:

            # SCANNERS
            updt_scn_pos(self._app)

            # Measurement progress bar
            updt_meas_progbar(self._app.meas)

            await asyncio.sleep(self.updt_intrvl["gui"])

    async def _update_avg_counts(self) -> NoReturn:
        """
        Read new counts, and dump buffer overflow.
        if update ready also average and show in GUI.

        """

        cntr_dvc = self._app.devices.COUNTER
        while self.not_finished:
            if cntr_dvc.error_dict is None:
                cntr_dvc.average_counts()
                self._app.gui.main.counts.setValue(cntr_dvc.avg_cnt_rate)

            await asyncio.sleep(self.updt_intrvl["cntr_avg"])

    async def _update_dep(self) -> NoReturn:
        """Update depletion laser GUI"""

        def update_SHG_temp(dep_dvc, main_gui) -> NoReturn:
            """Doc."""

            temp = dep_dvc.get_prop("temp")
            if temp < dep_dvc.min_SHG_temp:
                main_gui.depTemp.setStyleSheet("background-color: red; color: white;")
            else:
                main_gui.depTemp.setStyleSheet("background-color: white; color: black;")
            main_gui.depTemp.setValue(temp)

        def update_power(dep_dvc, main_gui) -> NoReturn:
            """Doc."""

            main_gui.depActualPow.setValue(dep_dvc.get_prop("pow"))

        def update_current(dep_dvc, main_gui) -> NoReturn:
            """Doc."""

            main_gui.depActualCurr.setValue(dep_dvc.get_prop("curr"))

        def update_props(dep_err_dict, dep_dvc, main_gui) -> NoReturn:
            """Doc."""

            if dep_err_dict is None:
                update_SHG_temp(dep_dvc, main_gui)

                # check current/power only if laser is ON
                if dep_dvc.state is True:
                    update_power(dep_dvc, main_gui)
                    update_current(dep_dvc, main_gui)

        while self.not_finished:
            update_props(
                self._app.devices.DEP_LASER.error_dict,
                self._app.devices.DEP_LASER,
                self._app.gui.main,
            )

            await asyncio.sleep(self.updt_intrvl["dep"])
