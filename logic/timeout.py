# -*- coding: utf-8 -*-
"""Timeout module."""

import asyncio
import logging

import utilities.constants as const


class Timeout:
    """Doc."""

    def __init__(self, app):
        """Doc."""

        self._app = app

        # initial intervals (some changed during run)
        self.updt_intrvl = {
            "gui": 0.2,
            "dep": self._app.dvc_dict["DEP_LASER"].update_time,
            "cntr_avg": self._app.dvc_dict["COUNTER"].update_time,
        }

    # MAIN
    async def _main(self):
        """Doc."""

        await asyncio.gather(
            self._timeout(),
            self._update_avg_counts(),
            self._update_dep(),
            self._update_gui(),
        )
        logging.debug("_main function exited")

    def start(self):
        """Doc."""

        self.running = True
        self.not_finished = True
        self._app.loop.create_task(self._main())
        logging.debug("Starting main timer.")

    async def finish(self):
        """Doc."""

        self.not_finished = False

    def pause(self):
        """Doc."""

        self.running = False
        logging.debug("Stopping main timer.")

    def resume(self):
        """Doc."""

        if not self.running:
            self.running = True
            logging.debug("Resuming main timer.")

    async def _timeout(self):
        """Doc."""

        while self.not_finished:

            if self.running:
                # COUNTER
                if self._app.error_dict["COUNTER"] is None:
                    self._app.dvc_dict["COUNTER"].count()
                    self._app.dvc_dict["COUNTER"].dump_buff_overflow()

                # AI
                if self._app.error_dict["SCANNERS"] is None:
                    self._app.dvc_dict["SCANNERS"].fill_ai_buff()
                    self._app.dvc_dict["SCANNERS"].dump_buff_overflow()

            await asyncio.sleep(const.TIMEOUT)

    async def _update_gui(self):
        """Doc."""

        def updt_scn_pos(app):
            """Doc."""

            if self._app.error_dict["SCANNERS"] is None:

                (x_ai, y_ai, z_ai) = tuple(
                    self._app.dvc_dict["SCANNERS"].ai_buffer[:, -1]
                )

                (x_um, y_um, z_um) = tuple(
                    (axis_vltg - axis_org) * axis_ratio
                    for axis_vltg, axis_ratio, axis_org in zip(
                        (x_ai, y_ai, z_ai),
                        self._app.dvc_dict["SCANNERS"].um_V_ratio,
                        const.ORIGIN,
                    )
                )

                self._app.win_dict["main"].xAiV.setValue(x_ai)
                self._app.win_dict["main"].yAiV.setValue(y_ai)
                self._app.win_dict["main"].zAiV.setValue(z_ai)

                self._app.win_dict["main"].xAoUm.setValue(x_um)
                self._app.win_dict["main"].yAoUm.setValue(y_um)
                self._app.win_dict["main"].zAoUm.setValue(z_um)

        def updt_fcs_progbar(meas):
            """Doc."""

            if (
                (self._app.error_dict["UM232"] is None)
                and meas.type in {"FCS", "SFCSSolution"}
                and meas.is_running
            ):
                if meas.prog_bar:
                    meas.prog_bar.setValue(
                        meas.time_passed
                        / (meas.duration_spinner.value() * meas.duration_multiplier)
                        * 100
                    )

        while self.not_finished:

            if self.running:

                # SCANNERS
                updt_scn_pos(self._app)

                # FCS progress bar
                updt_fcs_progbar(self._app.meas)

            await asyncio.sleep(self.updt_intrvl["gui"])

    async def _update_avg_counts(self):
        """
        Read new counts, and dump buffer overflow.
        if update ready also average and show in GUI.

        """

        nick = "COUNTER"

        while self.not_finished:

            if self.running:
                if self._app.error_dict[nick] is None:
                    avg_counts = self._app.dvc_dict[nick].average_counts()
                    self._app.win_dict["main"].countsSpinner.setValue(avg_counts)

            await asyncio.sleep(self.updt_intrvl["cntr_avg"])

    async def _update_dep(self):
        """Update depletion laser GUI"""

        nick = "DEP_LASER"

        async def update_SHG_temp(dep_dvc, main_gui):
            """Doc."""

            main_gui.depTemp.setValue(await dep_dvc.get_prop("tmp"))
            if dep_dvc.temp < dep_dvc.min_SHG_temp:
                main_gui.depTemp.setStyleSheet("background-color: red; color: white;")
            else:
                main_gui.depTemp.setStyleSheet("background-color: white; color: black;")

        async def update_power(dep_dvc, main_gui):
            """Doc."""

            main_gui.depActualPowerSpinner.setValue(await dep_dvc.get_prop("pow"))

        async def update_current(dep_dvc, main_gui):
            """Doc."""

            main_gui.depActualCurrSpinner.setValue(await dep_dvc.get_prop("curr"))

        async def update_props(dep_err_dict, dep_dvc, main_gui):
            """Doc."""

            if dep_err_dict is None:
                await update_SHG_temp(dep_dvc, main_gui)

                # check current/power only if laser is ON
                if dep_dvc.state is True:
                    await update_power(dep_dvc, main_gui)
                    await update_current(dep_dvc, main_gui)

        while self.not_finished:

            if self.running:
                await update_props(
                    self._app.error_dict[nick],
                    self._app.dvc_dict[nick],
                    self._app.win_dict["main"],
                )

            await asyncio.sleep(self.updt_intrvl["dep"])
