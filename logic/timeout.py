# -*- coding: utf-8 -*-
"""Timeout module."""

import asyncio
import logging
import os
import time
from collections import deque
from typing import NoReturn

import utilities.constants as consts
from utilities.errors import logic_error_handler as err_hndlr


class Timeout:
    """Doc."""

    def __init__(self, app) -> NoReturn:
        """Doc."""

        self._app = app

        # initial intervals (some change during run)
        self.updt_intrvl = {
            "gui": 0.2,
            "dep": self._app.devices.DEP_LASER.update_time,
            "cntr_avg": self._app.devices.COUNTER.update_time,
        }

    # MAIN
    async def _main(self) -> NoReturn:
        """Doc."""

        await asyncio.gather(
            self._updt_CI_and_AI(),
            self._update_avg_counts(),
            self._app.loop.run_in_executor(None, self._update_dep),
            self._updt_current_state(),
            self._update_gui(),
        )
        logging.debug("_main function exited")

    def start(self) -> NoReturn:
        """Doc."""

        self.running = True
        self.not_finished = True
        self._app.loop.create_task(self._main())
        logging.debug("Starting main timer.")

    def finish(self) -> NoReturn:
        """Doc."""

        self.not_finished = False

    def pause(self) -> NoReturn:
        """Doc."""

        self.running = False
        logging.debug("Stopping main timer.")

    def resume(self) -> NoReturn:
        """Doc."""

        if not self.running:
            self.running = True
            logging.debug("Resuming main timer.")

    async def _updt_current_state(self) -> NoReturn:
        """Doc."""

        @err_hndlr
        def get_last_line(file_path) -> str:
            """
            Return the last line of a text file.
            (https://stackoverflow.com/questions/46258499/read-the-last-line-of-a-file-in-python)
            """

            with open(file_path, "rb") as f:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
                last_line = f.readline().decode()
            return last_line

        now_timestamp = time.strftime("%H:%M:%S", time.localtime())
        first_line = f"[{now_timestamp}] Application Started"
        buffer_deque = deque(
            [first_line], maxlen=5
        )  # TODO: decide where to control the size

        while self.not_finished:
            if self.running:

                last_line = get_last_line(consts.LOG_FOLDER_PATH + "log")

                if last_line.find("INFO") != -1:
                    last_line = (
                        last_line[12:23] + last_line[38:]
                    )  # TODO: explain these indices (see log file) or use regular expression instead

                    if last_line != buffer_deque[0]:
                        buffer_deque.appendleft(last_line)

                        text = "".join(buffer_deque)
                        self._app.gui.main.lastAction.setPlainText(text)

                await asyncio.sleep(0.3)

    async def _updt_CI_and_AI(self) -> NoReturn:
        """Doc."""
        # TODO: (later, if anything slows down) consider running in thread (see _updt_dep in this module)

        while self.not_finished:
            if self.running:
                # COUNTER
                if self._app.devices.COUNTER.error_dict is None:
                    self._app.devices.COUNTER.count()
                    self._app.devices.COUNTER.dump_buff_overflow()

                # AI
                if self._app.devices.SCANNERS.error_dict is None:
                    self._app.devices.SCANNERS.fill_ai_buff()
                    self._app.devices.SCANNERS.dump_buff_overflow()

            await asyncio.sleep(consts.TIMEOUT)

    async def _update_gui(self) -> NoReturn:
        """Doc."""

        def updt_scn_pos(app):
            """Doc."""

            if self._app.devices.SCANNERS.error_dict is None:

                (x_ai, y_ai, z_ai) = tuple(self._app.devices.SCANNERS.ai_buffer[:, -1])

                (x_um, y_um, z_um) = tuple(
                    (axis_vltg - axis_org) * axis_ratio
                    for axis_vltg, axis_ratio, axis_org in zip(
                        (x_ai, y_ai, z_ai),
                        self._app.devices.SCANNERS.um_V_ratio,
                        consts.ORIGIN,
                    )
                )

                self._app.gui.main.xAiV.setValue(x_ai)
                self._app.gui.main.yAiV.setValue(y_ai)
                self._app.gui.main.zAiV.setValue(z_ai)

                self._app.gui.main.xAoUm.setValue(x_um)
                self._app.gui.main.yAoUm.setValue(y_um)
                self._app.gui.main.zAoUm.setValue(z_um)

        def updt_meas_progbar(meas) -> NoReturn:
            """Doc."""

            if (
                (self._app.devices.UM232H.error_dict is None)
                and meas.type in {"FCS", "SFCSSolution"}
                and meas.is_running
            ):
                if meas.type == "FCS":
                    progress = (
                        meas.time_passed
                        / (meas.duration_gui.value() * meas.duration_multiplier)
                        * 100
                    )
                elif meas.type == "SFCSSolution":
                    if not meas.cal:
                        progress = (
                            (meas.total_time_passed + meas.time_passed)
                            / (
                                meas.total_duration_gui.value()
                                * meas.duration_multiplier
                            )
                            * 100
                        )
                    else:
                        progress = 0
                meas.prog_bar.setValue(progress)

        while self.not_finished:
            if self.running:

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

        while self.not_finished:
            if self.running:
                if self._app.devices.COUNTER.error_dict is None:
                    avg_counts = self._app.devices.COUNTER.average_counts()
                    self._app.gui.main.countsSpinner.setValue(avg_counts)

            await asyncio.sleep(self.updt_intrvl["cntr_avg"])

    def _update_dep(self) -> NoReturn:
        """Update depletion laser GUI"""

        def update_SHG_temp(dep_dvc, main_gui) -> NoReturn:
            """Doc."""

            temp = dep_dvc.get_prop("temp")
            main_gui.depTemp.setValue(temp)
            if temp < dep_dvc.min_SHG_temp:
                main_gui.depTemp.setStyleSheet("background-color: red; color: white;")
            else:
                main_gui.depTemp.setStyleSheet("background-color: white; color: black;")

        def update_power(dep_dvc, main_gui) -> NoReturn:
            """Doc."""

            main_gui.depActualPowerSpinner.setValue(dep_dvc.get_prop("pow"))

        def update_current(dep_dvc, main_gui) -> NoReturn:
            """Doc."""

            main_gui.depActualCurrSpinner.setValue(dep_dvc.get_prop("curr"))

        def update_props(dep_err_dict, dep_dvc, main_gui) -> NoReturn:
            """Doc."""

            if dep_err_dict is None:
                update_SHG_temp(dep_dvc, main_gui)

                # check current/power only if laser is ON
                if dep_dvc.state is True:
                    update_power(dep_dvc, main_gui)
                    update_current(dep_dvc, main_gui)

        while self.not_finished:
            if self.running:
                update_props(
                    self._app.devices.DEP_LASER.error_dict,
                    self._app.devices.DEP_LASER,
                    self._app.gui.main,
                )

            time.sleep(self.updt_intrvl["dep"])
