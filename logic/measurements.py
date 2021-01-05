# -*- coding: utf-8 -*-
"""Measurements Module."""

import concurrent.futures
import logging
import random
import time

import utilities.constants as const


class Measurement:
    """Doc."""

    def __init__(self, app, type=None, duration_spinner=None, prog_bar=None):

        self._app = app
        self.type = type
        self.duration_spinner = duration_spinner
        self.prog_bar = prog_bar
        self.start_time = None
        self.intrvl_done = False

    async def start(self):
        """Doc."""

        # TEST TEST TEST - THREADING -----------------------------------------------
        def mock_callback(buffer):

            buffer += [random.randint(1, 255) for i in range(random.randint(1, 1))]

        def mock_readstream(buffer, meas_time):

            start = time.perf_counter()
            now = start
            while now - start < meas_time:
                time.sleep(0.001)
                mock_callback(buffer)
                now = time.perf_counter()

        def mock_io(duration_spinner):

            time.sleep(self.duration_spinner.value())

        # ----------------------------------------------------------------------------------------

        self._app.dvc_dict["UM232"].purge()
        self._app.win_dict["main"].imp.dvc_toggle("TDC")
        self._app.timeout_loop.updt_intrvl["meas"] = const.TIMEOUT

        # TEST TEST TEST - THREADING -----------------------------------------------
        while self.type == "FCS":
            #            await asyncio.to_thread(mock_io, self.duration_spinner) # TODO: Try when upgrade to Python 3.9 is feasible

            self.intrvl_done = False
            self.start_time = time.perf_counter()
            await self._app.loop.run_in_executor(None, mock_io, self.duration_spinner)
            #            self.executor.shutdown(wait=False)
            print(f"ran blocking IO task for {self.duration_spinner.value()} seconds.")

        #        with concurrent.futures.ThreadPoolExecutor() as executor:
        ##            await self._app.loop.run_in_executor(executor, mock_readstream, self.mock_buffer, 10)
        #            await self._app.loop.run_in_executor(executor, mock_io, 10)
        #            print(self.mock_buffer)
        # ----------------------------------------------------------------------------------------

        logging.info(f"{self.type} measurement started")

    def stop(self):
        """Doc."""

        meas_name = self.type
        self.type = None
        self._app.timeout_loop.updt_intrvl["meas"] = 0.5

        self._app.win_dict["main"].imp.dvc_toggle("TDC")
        self._app.dvc_dict["UM232"].purge()

        if self.prog_bar:
            self.prog_bar.setValue(0)

        logging.info(f"{meas_name} measurement stopped")

    def disp_ACF(self):
        """Doc."""

        print(
            f"Measurement Finished:\n"
            f"Full Data = {self._app.dvc_dict['UM232'].data}\n"
            f"Total Bytes = {self._app.dvc_dict['UM232'].tot_bytes}\n"
        )
