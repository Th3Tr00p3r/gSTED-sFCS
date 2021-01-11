# -*- coding: utf-8 -*-
"""Measurements Module."""

import logging
import time


class Measurement:
    """Base class for measurements"""

    def __init__(
        self,
        app,
        type=None,
        duration_spinner=None,
        duration_multiplier=1,
        prog_bar=None,
    ):

        self._app = app
        self.type = type
        self.duration_spinner = duration_spinner
        self.duration_multiplier = duration_multiplier
        self.prog_bar = prog_bar
        self.start_time = None
        self.is_running = False

    async def start(self):
        """Doc."""

        self._app.dvc_dict["UM232"].purge()  # TODO: is this needed?
        self._app.win_dict["main"].imp.dvc_toggle("TDC")
        self.is_running = True

        logging.info(f"{self.type} measurement started")

        await self.run()

    def stop(self):
        """Doc."""

        self.is_running = False
        self._app.meas.type = None
        self._app.win_dict["main"].imp.dvc_toggle("TDC")

        if self.prog_bar:
            self.prog_bar.setValue(0)

        logging.info(f"{self.type} measurement stopped")


class SFCSSolutionMeasurement(Measurement):
    """Doc."""

    def __init__(self, app, duration_spinner=None, prog_bar=None):
        super().__init__(
            app=app,
            type="SFCSSolution",
            duration_spinner=duration_spinner,
            duration_multiplier=60,
            prog_bar=prog_bar,
        )

    async def run(self):
        """Doc."""

        def disp_ACF(meas_dvc):
            """Doc."""

            print(
                f"Measurement Finished:\n"
                f"Full Data = {meas_dvc.data}\n"
                f"Total Bytes = {meas_dvc.tot_bytes_read}\n"
            )

        self.start_time = time.perf_counter()
        self.time_passed = 0

        await self._app.loop.run_in_executor(
            None, self._app.dvc_dict["UM232"].stream_read_TDC, self
        )

        disp_ACF(meas_dvc=self._app.dvc_dict["UM232"])
        self._app.dvc_dict["UM232"].init_data()

        if self.is_running:  # if not manually stopped
            self._app.win_dict["main"].imp.toggle_SFCSSolution_meas()


class FCSMeasurement(Measurement):
    """Repeated static FCS measurement, intended fo system calibration"""

    def __init__(self, app, duration_spinner=None, prog_bar=None):
        super().__init__(
            app=app, type="FCS", duration_spinner=duration_spinner, prog_bar=prog_bar
        )

    async def run(self):
        """Doc."""

        def disp_ACF(meas_dvc):
            """Doc."""

            print(
                f"Measurement Finished:\n"
                f"Full Data = {meas_dvc.data}\n"
                f"Total Bytes = {meas_dvc.tot_bytes_read}\n"
            )

        while self.is_running:
            #            await asyncio.to_thread(mock_io, self.duration_spinner) # TODO: Try when upgrade to Python 3.9 is feasible

            self.intrvl_done = False
            self.start_time = time.perf_counter()
            self.time_passed = 0

            await self._app.loop.run_in_executor(
                None, self._app.dvc_dict["UM232"].stream_read_TDC, self
            )

            disp_ACF(meas_dvc=self._app.dvc_dict["UM232"])
            self._app.dvc_dict["UM232"].init_data()
