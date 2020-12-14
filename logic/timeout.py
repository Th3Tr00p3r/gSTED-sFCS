# -*- coding: utf-8 -*-
"""Timeout module."""

import asyncio
import logging

from PyQt5.QtGui import QIcon

import gui.icons.icon_paths as icon
import utilities.constants as const


class Timeout:
    """Doc."""

    def __init__(self, app):
        """Doc."""

        self._app = app

        # init updateables
        self._gui_updt_intrvl = 2
        self._meas_updt_intrvl = 1
        self._dep_updt_intrvl = self._app.dvc_dict["DEP_LASER"].update_time
        self._cnts_updt_intrvl = self._app.dvc_dict["COUNTER"].update_time

    # MAIN
    async def _main(self):
        """Doc."""

        await asyncio.gather(
            self._cntr_read(),  # TODO: move all counter-related into single function?
            self._update_avg_counts(),  # TODO: move all counter-related into single function?
            self._check_cntr_err(),  # TODO: move all counter-related into single function?
            self._update_dep(),
            self._update_gui(),
            self._update_measurement(),
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
        await asyncio.sleep(0.1)

    def pause(self):
        """Doc."""

        self.running = False
        logging.debug("Stopping main timer.")

    def resume(self):
        """Doc."""

        self.running = True
        logging.debug("Resuming main timer.")

    async def _cntr_read(self):
        """Doc."""

        nick = "COUNTER"

        while self.not_finished:

            if self.running:
                if self._app.error_dict[nick] is None:

                    self._app.dvc_dict[nick].count()
                    self._app.dvc_dict[nick].dump_buff_overflow()

            await asyncio.sleep(const.TIMEOUT)

    async def _update_avg_counts(self):
        """
        Read new counts, and dump buffer overflow.
        if update ready also average and show in GUI.

        """

        nick = "COUNTER"

        while self.not_finished:

            if self.running:
                if self._app.error_dict[nick] is None:
                    avg_interval = (
                        self._app.win_dict["main"].countsAvg.value() / 1000
                    )  # to get seconds
                    avg_counts = self._app.dvc_dict[nick].average_counts(avg_interval)
                    self._app.win_dict["main"].countsSpinner.setValue(avg_counts)

            await asyncio.sleep(self._cnts_updt_intrvl)

    async def _check_cntr_err(self):
        """Doc."""

        def are_last_n_ident(lst, n):
            """Check if the last n elements of a list are identical"""

            if len(lst) > n:
                return len(set(lst[-n:])) == 1

        while self.not_finished:

            if self.running:
                if are_last_n_ident(self._app.dvc_dict["COUNTER"].cont_count_buff, 10):
                    self._app.error_dict["COUNTER"] = "Detector is disconnected"

            await asyncio.sleep(
                self._gui_updt_intrvl
            )  # TODO: using same interval as for GUI update - should I?

    async def _update_dep(self):
        """Update depletion laser GUI"""

        nick = "DEP_LASER"

        def update_SHG_temp(dep, main_gui):
            """Doc."""
            # TODO: fix None type when turning off dep laser while app running

            dep.get_SHG_temp()
            main_gui.depTemp.setValue(dep.temp)
            if dep.temp < dep.min_SHG_temp:
                main_gui.depTemp.setStyleSheet("background-color: red; color: white;")
            else:
                main_gui.depTemp.setStyleSheet("background-color: white; color: black;")

        def update_power(dep, main_gui):
            """Doc."""

            dep.get_power()
            main_gui.depActualPowerSpinner.setValue(dep.power)

        def update_current(dep, main_gui):
            """Doc."""

            dep.get_current()
            main_gui.depActualCurrSpinner.setValue(dep.current)

        while self.not_finished:

            if self.running:
                if self._app.error_dict[nick] is None:
                    update_SHG_temp(
                        self._app.dvc_dict[nick], self._app.win_dict["main"]
                    )

                    if (
                        self._app.dvc_dict[nick].state is True
                    ):  # check current/power only if laser is ON
                        update_power(
                            self._app.dvc_dict[nick], self._app.win_dict["main"]
                        )
                        update_current(
                            self._app.dvc_dict[nick], self._app.win_dict["main"]
                        )

            await asyncio.sleep(self._dep_updt_intrvl)

    async def _update_gui(self):
        """Doc."""
        # TODO: update error GUI according to errors in 'error_dict'

        while self.not_finished:

            if self.running:

                for nick in self._app.dvc_dict.keys():

                    if self._app.error_dict[nick] is not None:  # case ERROR
                        gui_led_object = getattr(
                            self._app.win_dict["main"],
                            const.ICON_DICT[nick]["LED"],
                        )
                        red_led = QIcon(icon.LED_RED)
                        gui_led_object.setIcon(red_led)

            await asyncio.sleep(self._gui_updt_intrvl)

    async def _update_measurement(self):
        """Doc."""

        while self.not_finished:

            if self.running:

                meas = self._app.meas

                if self._app.error_dict["UM232"] is None:

                    if meas.type == "FCS":
                        self._app.dvc_dict["UM232"].read_TDC_data()

                        if meas.time_passed < meas.duration_spinner.value():
                            meas.time_passed += 1

                        else:  # timer finished
                            meas.disp_ACF()
                            self._app.dvc_dict["UM232"].init_data()
                            self._app.dvc_dict["UM232"].purge()
                            meas.time_passed = 1

                        if meas.prog_bar:
                            prog = (
                                meas.time_passed / meas.duration_spinner.value() * 100
                            )
                            meas.prog_bar.setValue(prog)

                elif meas.type is not None:  # case UM232 error while measuring
                    meas.stop()
                    self._app.win_dict["main"].startFcsMeasurementButton.setText(
                        "Start \nMeasurement"
                    )

            await asyncio.sleep(self._meas_updt_intrvl)
