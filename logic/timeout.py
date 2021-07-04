"""Timeout module."""

import asyncio
import logging
import os
import sys
from collections import deque

from utilities.errors import DeviceError, err_hndlr

# import time
# import numpy as np

TIMEOUT = 0.010  # seconds (10 ms)
LOG_PATH = "./log/log"


class Timeout:
    """Doc."""

    def __init__(self, app) -> None:
        """Doc."""

        self._app = app
        self.main_gui = self._app.gui.main

        # initial intervals (some change during run)
        self.updt_intrvl = {
            "gui": 0.2,
            "dep": self._app.devices.dep_laser.updt_time,
            "cntr_avg": self._app.devices.photon_detector.updt_time,
        }

        maxlen = self._app.gui.main.logNumLinesSlider.value()
        self.log_buffer_deque = deque([""], maxlen=maxlen)

        self.um232_buff_wdgt = self._app.gui.main.um232buff
        self.um232_buff_sz = self._app.devices.UM232H.tx_size

        self.cntr_dvc = self._app.devices.photon_detector
        self.dep_dvc = self._app.devices.dep_laser

        # start
        self.not_finished = True
        logging.debug("Initiating _timeout function.")
        self._app.loop.create_task(self._timeout())

    # MAIN
    async def _timeout(self) -> None:
        """awaits all individual async loops, each with its own repeat time."""

        # start the loops
        await asyncio.gather(
            update_CI_and_AI(self),
            update_avg_counts(self),
            update_dep(self),
            update_application_log(self),
            update_gui(self),
            # self._update_um232h_status(self), # TODO: this seems to cause an issue during measurements (noticed in solution scan) - try to see if it does and catch the error
        )

        logging.debug("_timeout function exited")


async def update_application_log(master) -> None:
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
            return "Log File Not Found!!!"
        else:
            return last_line

    while master.not_finished:

        new_maxlen = master._app.gui.main.logNumLinesSlider.value()
        if master.log_buffer_deque.maxlen != new_maxlen:
            master.log_buffer_deque = deque(master.log_buffer_deque, maxlen=new_maxlen)

        last_line = get_last_line(LOG_PATH)

        if last_line.find("INFO") != -1:
            line_time = last_line[12:23]
            line_text = last_line[38:]
            last_line = line_time + line_text

            if last_line != master.log_buffer_deque[0]:
                master.log_buffer_deque.appendleft(last_line)

                text = "".join(master.log_buffer_deque)
                master._app.gui.main.lastAction.setPlainText(text)

        await asyncio.sleep(0.3)


async def update_CI_and_AI(master) -> None:
    """Doc."""

    #        N_TOC = 100 # TESTING
    #        tocs = np.zeros(shape=(N_TOC,)) # TESTING
    #        toc_idx = 0 # TESTING
    #        tic = time.perf_counter() # TESTING

    while master.not_finished:

        #            if toc_idx == N_TOC: # TESTING
        #                print(f"average timeout elapsed: {tocs.mean()}. Should be: {TIMEOUT}") # TESTING
        #                toc_idx = 0 # TESTING
        #            tocs[toc_idx] = time.perf_counter() - tic # TESTING
        #            toc_idx += 1 # TESTING

        # photon_detector
        if not master._app.devices.photon_detector.error_dict:
            master._app.devices.photon_detector.fill_ci_buffer()
            if master._app.meas.type not in {"SFCSImage"}:
                master._app.devices.photon_detector.dump_ci_buff_overflow()

        # AI
        if not master._app.devices.scanners.error_dict:
            master._app.devices.scanners.fill_ai_buffer()
            if master._app.meas.type not in {"SFCSSolution", "SFCSImage"}:
                master._app.devices.scanners.dump_ai_buff_overflow()

        #            tic = time.perf_counter() # TESTING

        await asyncio.sleep(TIMEOUT)


async def update_gui(master) -> None:
    """Doc."""

    def update_scn_pos(app):
        """Doc."""

        try:
            (
                x_ai,
                y_ai,
                z_ai,
                x_ao_int,
                y_ao_int,
                z_ao_int,
            ) = app.devices.scanners.ai_buffer[:, -1]

        except IndexError:
            # AI buffer has just been initialized
            pass

        else:
            (x_um, y_um, z_um) = tuple(
                (axis_vltg - axis_org) * axis_ratio
                for axis_vltg, axis_ratio, axis_org in zip(
                    (x_ao_int, y_ao_int, z_ao_int),
                    app.devices.scanners.um_v_ratio,
                    app.devices.scanners.ORIGIN,
                )
            )

            app.gui.main.xAIV.setValue(x_ai)
            app.gui.main.yAIV.setValue(y_ai)
            app.gui.main.zAIV.setValue(z_ai)

            app.gui.main.xAOVint.setValue(x_ao_int)
            app.gui.main.yAOVint.setValue(y_ao_int)
            app.gui.main.zAOVint.setValue(z_ao_int)

            app.gui.main.xAOum.setValue(x_um)
            app.gui.main.yAOum.setValue(y_um)
            app.gui.main.zAOum.setValue(z_um)

    def update_meas_progbar(meas) -> None:
        """Doc."""

        try:
            if meas.type == "SFCSSolution":
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
                    meas.time_passed / meas.est_total_duration * meas.prog_bar_wdgt.obj.maximum()
                )
            meas.prog_bar_wdgt.set(progress)

        except AttributeError:
            # happens when depletion is turned on before beginning measurement (5 s wait)
            pass

        except Exception as exc:
            err_hndlr(exc, locals(), sys._getframe())

    while master.not_finished:

        # scanners
        update_scn_pos(master._app)

        # Measurement progress bar
        if master._app.meas.is_running:
            update_meas_progbar(master._app.meas)

        await asyncio.sleep(master.updt_intrvl["gui"])


async def update_um232h_status(master):
    """Doc."""

    while master.not_finished:

        try:
            rx_bytes = master._app.devices.UM232H.get_status()

        except DeviceError:
            pass

        else:
            fill_perc = rx_bytes / master.um232_buff_sz * 100
            master.um232_buff_wdgt.setValue(fill_perc)
            if fill_perc > 90:
                logging.warning("UM232H buffer might be overfilling")

        finally:
            await asyncio.sleep(master.updt_intrvl["gui"])


async def update_avg_counts(master) -> None:
    """
    Read new counts, and dump buffer overflow.
    if update ready also average and show in GUI.

    """

    while master.not_finished:

        try:
            master.cntr_dvc.average_counts()

        except DeviceError:
            pass

        else:
            master._app.gui.main.counts.setValue(master.cntr_dvc.avg_cnt_rate)

        finally:
            await asyncio.sleep(master.updt_intrvl["cntr_avg"])


async def update_dep(master) -> None:
    """Update depletion laser GUI"""

    while master.not_finished:

        try:
            temp, pow, curr = (
                master.dep_dvc.get_prop("temp"),
                master.dep_dvc.get_prop("pow"),
                master.dep_dvc.get_prop("curr"),
            )

        except DeviceError:
            # do nothing if error in depletion laser
            pass

        else:
            master.main_gui.depTemp.setValue(temp)
            master.main_gui.depActualPow.setValue(pow)
            master.main_gui.depActualCurr.setValue(curr)

            if temp < master.dep_dvc.min_SHG_temp:
                master.main_gui.depTemp.setStyleSheet("background-color: red; color: white;")
            else:
                master.main_gui.depTemp.setStyleSheet("background-color: white; color: black;")

        finally:
            await asyncio.sleep(master.updt_intrvl["dep"])
