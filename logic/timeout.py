# -*- coding: utf-8 -*-
"""Timeout module."""

import asyncio
import collections
import itertools as it

# import concurrent.futures
import logging

from PyQt5.QtGui import QIcon

import gui.icons.icon_paths as icon
import utilities.constants as const
import utilities.errors as errors


class Timeout:
    """Doc."""

    def __init__(self, app):
        """Doc."""

        self._app = app

        self.init_intrvls()

    def init_intrvls(self):
        """Doc."""

        self._gui_updt_intrvl = 2
        self._meas_updt_intrvl = 1
        self._dep_updt_intrvl = self._app.dvc_dict["DEP_LASER"].update_time
        self._cnts_updt_intrvl = self._app.dvc_dict["COUNTER"].update_time
        self._cntr_chck_intrvl = 3

    # MAIN
    async def _main(self):
        """Doc."""

        await asyncio.gather(
            self._cntr_read(),  # TODO: move all counter-related into single function?
            self._update_avg_counts(),
            self._check_cntr_err(),
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

    def pause(self):
        """Doc."""

        self.running = False
        logging.debug("Stopping main timer.")

    def resume(self):
        """Doc."""

        if not self.running:
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
                    avg_counts = self._app.dvc_dict[nick].average_counts()
                    self._app.win_dict["main"].countsSpinner.setValue(avg_counts)

            await asyncio.sleep(self._cnts_updt_intrvl)

    async def _check_cntr_err(self):
        """Doc."""

        def are_last_n_ident(lst, n):
            """Check if the last n elements of a list are identical"""

            def tail(n, iterable):
                """
                Return an iterator over the last n items.
                (https://docs.python.org/3.8/library/itertools.html?highlight=itertools#itertools-recipes)

                tail(3, 'ABCDEFG') --> E F G
                """

                return iter(collections.deque(iterable, maxlen=n))

            def all_equal(iterable):
                """
                Returns True if all the elements are equal to each other
                (https://docs.python.org/3.8/library/itertools.html?highlight=itertools#itertools-recipes)
                """

                g = it.groupby(iterable)
                return next(g, True) and not next(g, False)

            return all_equal(tail(n, lst))

        while self.not_finished:

            await asyncio.sleep(self._cntr_chck_intrvl)

            if self.running:
                try:
                    if are_last_n_ident(
                        self._app.dvc_dict["COUNTER"].cont_count_buff, 10
                    ):
                        raise errors.CounterError("Detector is disconnected")
                except errors.CounterError as exc:
                    self._app.error_dict["COUNTER"] = errors.build_err_dict(exc)
                    logging.error("Detector is disconnected", exc_info=False)
                    self._cntr_chck_intrvl = 999

    async def _update_dep(self):
        """Update depletion laser GUI"""

        nick = "DEP_LASER"

        async def update_SHG_temp(dep_dvc, main_gui):
            """Doc."""

            await dep_dvc.get_SHG_temp()
            main_gui.depTemp.setValue(dep_dvc.temp)
            if dep_dvc.temp < dep_dvc.min_SHG_temp:
                main_gui.depTemp.setStyleSheet("background-color: red; color: white;")
            else:
                main_gui.depTemp.setStyleSheet("background-color: white; color: black;")

        async def update_power(dep_dvc, main_gui):
            """Doc."""

            await dep_dvc.get_power()
            main_gui.depActualPowerSpinner.setValue(dep_dvc.power)

        async def update_current(dep_dvc, main_gui):
            """Doc."""

            await dep_dvc.get_current()
            main_gui.depActualCurrSpinner.setValue(dep_dvc.current)

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
            #                with concurrent.futures.ThreadPoolExecutor() as pool:
            #                    result = await self._app.loop.run_in_executor(
            #                        pool, update_props(
            #                            self._app.error_dict[nick], self._app.dvc_dict[nick], self._app.win_dict["main"]
            #                            )
            #                    )
            #                    print(result)

            await asyncio.sleep(self._dep_updt_intrvl)

    async def _update_gui(self):
        """Doc."""

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

                        else:  # measurement finished
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
