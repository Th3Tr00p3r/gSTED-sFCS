# -*- coding: utf-8 -*-
"""Measurements Module."""

import logging
import time


class Measurement:
    """Doc."""

    def __init__(self, app, type=None, duration_spinner=None, prog_bar=None):

        self._app = app
        self.type = type
        self.duration_spinner = duration_spinner
        self.prog_bar = prog_bar
        self.start_time = None
        self.intrvl_done = False
        self.is_running = None

    async def start(self):
        """Doc."""

        self._app.dvc_dict["UM232"].purge()
        self._app.win_dict["main"].imp.dvc_toggle("TDC")
        self.is_running = True

        logging.info(f"{self.type} measurement started")

        await self.run()

    async def run(self):
        """Doc."""

        while self.is_running:
            #            await asyncio.to_thread(mock_io, self.duration_spinner) # TODO: Try when upgrade to Python 3.9 is feasible

            self.intrvl_done = False
            self.start_time = time.perf_counter()
            self.time_passed = 0

            await self._app.loop.run_in_executor(
                None, self._app.dvc_dict["UM232"].stream_read_TDC, self
            )

            self.disp_ACF()
            self._app.dvc_dict["UM232"].init_data()

    def stop(self):
        """Doc."""

        meas_name = self.type
        self.type = None

        self._app.win_dict["main"].imp.dvc_toggle("TDC")

        if self.prog_bar:
            self.prog_bar.setValue(0)

        self.is_running = False

        logging.info(f"{meas_name} measurement stopped")

    def disp_ACF(self):
        """Doc."""

        print(
            f"Measurement Finished:\n"
            f"Full Data = {self._app.dvc_dict['UM232'].data}\n"
            f"Total Bytes = {self._app.dvc_dict['UM232'].tot_bytes}\n"
        )
