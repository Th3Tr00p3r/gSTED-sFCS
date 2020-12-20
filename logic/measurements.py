# -*- coding: utf-8 -*-
"""Measurements Module."""

import logging


class Measurement:
    """Doc."""

    def __init__(self, app, type=None, duration_spinner=None, prog_bar=None):

        self._app = app
        self.type = type
        self.duration_spinner = duration_spinner
        self.prog_bar = prog_bar
        self.time_passed = 0

    def start(self):
        """Doc."""

        self._app.dvc_dict["UM232"].purge()
        self._app.win_dict["main"].imp.dvc_toggle("TDC")
        logging.info(f"{self.type} measurement started")

    def stop(self):
        """Doc."""

        meas_name = self.type
        self.type = None

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
