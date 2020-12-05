# -*- coding: utf-8 -*-
"""Timeout module."""

import time

import gui.icons.icon_paths as icon
import utilities.constants as const
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon


class Updatable:
    """Doc."""

    def __init__(self, intrvl):
        self.ready = True
        self.intrvl = intrvl

        self.last = time.perf_counter()

    def is_ready(self):

        if time.perf_counter() > (self.last + self.intrvl):
            self.ready = True

    def just_updated(self):

        self.last = time.perf_counter()
        self.ready = False


class Timeout:
    """Doc."""

    def __init__(self, app):
        """Doc."""

        self._app = app

        self._timer = QTimer()
        self._timer.setInterval(const.TIMEOUT)
        self._timer.timeout.connect(self._main)

        # init updateables
        self._gui_up = Updatable(intrvl=2)
        self._meas_up = Updatable(intrvl=1)
        self._dep_up = Updatable(intrvl=self._app.dvc_dict["DEP_LASER"].update_time)
        self._cntr_up = Updatable(intrvl=self._app.dvc_dict["COUNTER"].update_time)

        self.start()

    def _check_readiness(self):
        """Doc."""

        self._gui_up.is_ready()
        self._meas_up.is_ready()
        self._dep_up.is_ready()
        self._cntr_up.is_ready()

    def stop(self):
        """Doc."""

        self._timer.stop()
        self._app.log.update("stopping main timer.", tag="verbose")

    def start(self):
        """Doc."""

        self._timer.start()
        self._app.log.update("starting main timer.", tag="verbose")

    def _update_dep(self):
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

        if (self._app.error_dict[nick] is None) and (self._dep_up.ready):
            update_SHG_temp(self._app.dvc_dict[nick], self._app.win_dict["main"])

            if (
                self._app.dvc_dict[nick].state is True
            ):  # check current/power only if laser is ON
                update_power(self._app.dvc_dict[nick], self._app.win_dict["main"])
                update_current(self._app.dvc_dict[nick], self._app.win_dict["main"])

            self._dep_up.just_updated()

    def _update_gui(self):
        """Doc."""
        # TODO: update error GUI according to errors in 'error_dict'

        if self._gui_up.ready is True:

            for nick in self._app.dvc_dict.keys():

                if self._app.error_dict[nick] is not None:  # case ERROR
                    gui_led_object = getattr(
                        self._app.win_dict["main"],
                        const.ICON_DICT[nick]["LED"],
                    )
                    red_led = QIcon(icon.LED_RED)
                    gui_led_object.setIcon(red_led)

                elif self._app.dvc_dict[nick].state is True:  # case ON
                    gui_led_object = getattr(
                        self._app.win_dict["main"],
                        const.ICON_DICT[nick]["LED"],
                    )
                    on_icon = QIcon(const.ICON_DICT[nick]["ICON"])
                    gui_led_object.setIcon(on_icon)

                else:  # case OFF
                    gui_led_object = getattr(
                        self._app.win_dict["main"],
                        const.ICON_DICT[nick]["LED"],
                    )
                    off_led = QIcon(icon.LED_OFF)
                    gui_led_object.setIcon(off_led)

            # TODO: decide whether to move the following to a new update function or change this function's name
            # -------------------------------------------------------------------------------------------------------------------------
            def are_last_n_ident(list, n):
                """Doc."""

                if len(list) > n:
                    return len(set(list[-n:])) == 1

            if are_last_n_ident(self._app.dvc_dict["COUNTER"].cont_count_buff, 10):
                self._app.error_dict["COUNTER"] = "Detector is disconnected"
            # -------------------------------------------------------------------------------------------------------------------------

            self._gui_up.just_updated()

    def _update_counter(self):
        """
        Read new counts, and dump buffer overflow.
        if update ready also average and show in GUI.

        """

        nick = "COUNTER"

        if self._app.error_dict[nick] is None:

            self._app.dvc_dict[nick].count()
            self._app.dvc_dict[nick].dump_buff_overflow()

            if self._cntr_up.ready:
                avg_interval = self._app.win_dict["main"].countsAvg.value()
                self._app.win_dict["main"].countsSpinner.setValue(
                    self._app.dvc_dict[nick].average_counts(avg_interval)
                )
                self._cntr_up.just_updated()

    def _update_measurement(self):
        """Doc."""

        meas = self._app.meas

        if self._app.error_dict["UM232"] is None:

            if meas.type == "FCS":
                self._app.dvc_dict["UM232"].read_TDC_data()

                if self._meas_up.ready:

                    if meas.time_passed < meas.duration_spinner.value():
                        meas.time_passed += 1

                    else:  # timer finished
                        meas.disp_ACF()
                        self._app.dvc_dict["UM232"].init_data()
                        self._app.dvc_dict["UM232"].purge()
                        meas.time_passed = 1

                    if meas.prog_bar:
                        prog = meas.time_passed / meas.duration_spinner.value() * 100
                        meas.prog_bar.setValue(prog)

                    self._meas_up.just_updated()

        elif meas.type is not None:  # case UM232 error while measuring
            meas.stop()
            self._app.win_dict["main"].startFcsMeasurementButton.setText(
                "Start \nMeasurement"
            )

    # MAIN
    def _main(self):
        """Doc."""

        self._check_readiness()
        self._update_dep()
        self._update_counter()
        self._update_gui()
        self._update_measurement()
