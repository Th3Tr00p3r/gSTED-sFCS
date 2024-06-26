""" GUI windows implementations module. """

import asyncio
import logging
import os
import re
import shutil
import sys
import webbrowser
from collections import namedtuple
from contextlib import contextmanager, suppress
from datetime import datetime as dt
from pathlib import Path
from typing import List, Tuple, cast

import ftd2xx
import nidaqmx.system as nisys
import numpy as np
import pyvisa
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem, QWidget

import gui.widgets as wdgts
import logic.measurements as meas
import utilities.dialog as dialog
from data_analysis.correlation_function import (
    ImageSFCSMeasurement,
    SolutionSFCSExperiment,
    SolutionSFCSMeasurement,
)
from logic.scan_patterns import ScanPatternAO
from utilities import file_utilities, fit_tools, helper
from utilities.display import GuiDisplay, auto_brightness_and_contrast
from utilities.errors import DeviceError, err_hndlr


class MainWin:
    """Doc."""

    ####################
    ## General
    ####################
    def __init__(self, main_gui, app):
        """Doc."""

        self._app = app
        self._gui = app.gui
        self.main_gui = main_gui
        self.cameras = []

    def close(self, event):
        """Doc."""

        self._app.exit_app(event)

    def save(self) -> None:
        """Doc."""

        wdgts.write_gui_to_file(
            self.main_gui, wdgts.MAIN_TYPES, self._app.DEFAULT_LOADOUT_FILE_PATH
        )
        logging.debug("Loadout saved.")

    def load(self) -> None:
        """Doc."""

        wdgts.read_file_to_gui(self._app.DEFAULT_LOADOUT_FILE_PATH, self.main_gui)
        logging.debug("Loadout loaded.")

    def device_toggle(
        self, nick, toggle_mthd="toggle", state_attr="is_on", leave_on=False, leave_off=False
    ) -> bool:
        """Returns False in case operation fails"""

        was_toggled = False  # assume didn't work
        dvc = getattr(self._app.devices, nick)  # get device

        try:  # try probing device state
            is_dvc_on = getattr(dvc, state_attr)

        except AttributeError:  # faulty device
            exc = DeviceError(f"{dvc.log_ref} was not properly initialized. Cannot toggle.")
            if nick == "stage":
                err_hndlr(exc, sys._getframe(), locals(), lvl="warning")
            else:
                raise exc

        # have device state, keep going
        else:
            # no need to do anything if already ON/OFF and meant to stay that way
            if (leave_on and is_dvc_on) or (leave_off and not is_dvc_on):
                was_toggled = True

            # otherwise, need to toggle!
            else:
                # switch ON
                if not is_dvc_on:
                    try:
                        if asyncio.iscoroutinefunction(getattr(dvc, toggle_mthd)):
                            self._app.loop.create_task(getattr(dvc, toggle_mthd)(True))
                        else:
                            getattr(dvc, toggle_mthd)(True)
                    except DeviceError as exc:
                        err_hndlr(exc, sys._getframe(), locals(), lvl="warning")
                    else:
                        # if managed to turn ON
                        if is_dvc_on := getattr(dvc, state_attr):
                            logging.debug(f"{dvc.log_ref} toggled ON")
                            if nick == "stage":
                                # TODO: try to move this inside device (add widgets to device)
                                self.main_gui.stageButtonsGroup.setEnabled(True)
                            elif nick == "exc_laser":
                                # TODO: try to move this inside device (add widgets to device)
                                self.main_gui.setPsdDelay.setEnabled(False)
                            was_toggled = True

                # switch OFF
                else:
                    with suppress(DeviceError):
                        if asyncio.iscoroutinefunction(getattr(dvc, toggle_mthd)):
                            self._app.loop.create_task(getattr(dvc, toggle_mthd)(False))
                        if nick == "delayer":
                            with suppress(DeviceError):
                                # TODO: try to move this inside device (add widgets to device)
                                self._app.loop.create_task(
                                    self._app.devices.spad.toggle_mode("free running")
                                )
                        else:
                            getattr(dvc, toggle_mthd)(False)

                    if not (is_dvc_on := getattr(dvc, state_attr)):
                        # if managed to turn OFF
                        logging.debug(f"{dvc.log_ref} toggled OFF")

                        if nick == "stage":
                            # TODO: try to move this inside device (add widgets to device)
                            self.main_gui.stageButtonsGroup.setEnabled(False)

                        if nick == "dep_laser":
                            # TODO: try to move this inside device (add widgets to device)
                            # set curr/pow values to zero when depletion is turned OFF
                            self.main_gui.depActualCurr.setValue(0)
                            self.main_gui.depActualPow.setValue(0)

                        elif nick == "exc_laser":
                            # TODO: try to move this inside device (add widgets to device)
                            self.main_gui.setPsdDelay.setEnabled(True)

                        was_toggled = True

        return was_toggled

    def led_clicked(self, led_obj_name) -> None:
        """Doc."""

        led_name_to_nick_dict = {
            helper.deep_getattr(dvc, "led_widget.obj_name"): nick
            for nick, dvc in self._app.devices.__dict__.items()
        }

        dvc_nick = led_name_to_nick_dict[led_obj_name]
        dvc = getattr(self._app.devices, dvc_nick)
        error_dict = dvc.error_dict
        if error_dict is not None:
            # attempt to reconnect
            with suppress(DeviceError):
                if asyncio.iscoroutinefunction(dvc.close):
                    self._app.loop.create_task(
                        dvc.close()
                    )  # TESTESTEST - was toggle instead of close
                else:
                    dvc.close()
            setattr(
                self._app.devices, dvc_nick, self._app.device_nick_class_dict[dvc_nick](self._app)
            )
            dvc = getattr(self._app.devices, dvc_nick)
            error_dict = dvc.error_dict
            if error_dict is not None:
                # show error dialog if reconnection fails
                dialog.ErrorDialog(
                    **error_dict,
                    custom_title=dvc.log_ref,
                ).display()
            else:
                dvc.change_icons("off")
                dvc.enable_switch()

    def dep_sett_apply(self):
        """Doc."""

        with suppress(DeviceError):
            if self.main_gui.currMode.isChecked():  # current mode
                val = self.main_gui.depCurr.value()
                self._app.devices.dep_laser.set_current(val)
            else:  # power mode
                val = self.main_gui.depPow.value()
                self._app.devices.dep_laser.set_power(val)

    def move_scanners(self, axes_used: str = "XYZ", destination=None) -> None:
        """Doc."""

        scanners_dvc = self._app.devices.scanners

        # if no destination is specified, use the AO from the GUI
        destination = destination or tuple(
            getattr(self.main_gui, f"{ax}AOV").value() for ax in "xyz"
        )

        data = []
        type_str = ""
        for ax, vltg_V in zip("XYZ", destination):
            if ax in axes_used:
                data.append([vltg_V])
                type_str += ax

        try:
            scanners_dvc.start_write_task(data, type_str)
            scanners_dvc.start_continuous_read_task()  # restart cont. reading
        except DeviceError as exc:
            err_hndlr(exc, sys._getframe(), locals())

        logging.debug(f"{self._app.devices.scanners.log_ref} were moved to {str(destination)} V")

    # TODO: implement a 'go_to_last_position' - same as origin, just need to save last location each time (if it's not the origin)
    def go_to_origin(self, which_axes: str = "XYZ") -> None:
        """Doc."""

        scanners_dvc = self._app.devices.scanners

        for axis, is_chosen, org_axis_vltg in zip(
            "xyz",
            scanners_dvc.AXES_TO_BOOL_TUPLE_DICT[which_axes],
            scanners_dvc.origin,
        ):
            if is_chosen:
                getattr(self.main_gui, f"{axis}AOV").setValue(org_axis_vltg)

        self.move_scanners(which_axes)

        logging.debug(f"{self._app.devices.scanners.log_ref} sent to {which_axes} origin")

    def displace_scanner_axis(self, sign: int) -> None:
        """Doc."""

        um_disp = sign * self.main_gui.axisMoveUm.value()

        if um_disp != 0.0:
            scanners_dvc = self._app.devices.scanners
            axis = self.main_gui.posAxis.currentText()
            try:
                current_vltg = scanners_dvc.ao_int[scanners_dvc.AXIS_INDEX[axis]]
            except IndexError:
                # buffer was initialized? # TESTESTEST
                logging.warning("buffer was initialized? # TESTESTEST")
            else:
                um_v_ratio = dict(zip("XYZ", scanners_dvc.um_v_ratio))[axis]
                delta_vltg = um_disp / um_v_ratio

                new_vltg = getattr(scanners_dvc, f"{axis.upper()}_AO_LIMITS").clamp(
                    (current_vltg + delta_vltg)
                )

                getattr(self.main_gui, f"{axis.lower()}AOV").setValue(new_vltg)
                self.move_scanners(axis)

                logging.debug(
                    f"{self._app.devices.scanners.log_ref}({axis}) was displaced {str(um_disp)} um"
                )

    def move_stage(self, dir: str = None, steps: int = None):
        """Doc."""

        if dir and steps:
            move_vec = helper.Vector(
                steps * int(dir == "LEFT") - steps * int(dir == "RIGHT"),
                steps * int(dir == "UP") - steps * int(dir == "DOWN"),
                "steps",
            )
            self._app.loop.create_task(self._app.devices.stage.move(move_vec))
        else:
            self._app.loop.create_task(
                self._app.devices.stage.move(helper.Vector(0, 0, "steps"), relative=False)
            )

    def set_stage_origin(self):
        """Doc."""

        stage_dvc = self._app.devices.stage

        if stage_dvc.curr_pos != (0, 0):
            pressed = dialog.QuestionDialog(
                txt=f"Are you sure you wish to set {stage_dvc.curr_pos} as the new origin?",
                title=f"Calibrate {stage_dvc.log_ref.capitalize()} Origin",
            ).display()
            if pressed is False:
                return
            else:
                stage_dvc.set_origin()

    def stage_move_to_last_pos(self):
        self._app.loop.create_task(self._app.devices.stage.move_to_last_pos())

    def open_spad_interface(self) -> None:
        """Doc."""

        spad_dvc = self._app.devices.spad

        with suppress(AttributeError):
            if spad_dvc.is_paused:
                spad_dvc.pause(False)
            else:
                spad_dvc.pause(True)  # temporarily close to let MPD software control serial port
                try:
                    os.startfile(
                        Path(
                            "C:/Program Files (x86)/MPD/FastGATED SPAD Module/FastGatedSPAD_Module.exe"
                        )
                    )
                except FileNotFoundError:
                    spad_dvc.pause(False)  # unpause if 'FastGatedSPAD_Module.exe' not found

    async def set_spad_gate(self):
        """Doc."""

        if hasattr(self._app, "devices"):
            try:  # DeviceError):
                self.main_gui.setPsdDelay.setEnabled(False)
                # the following to calls need to happen in this order, so cannot use asyncio.gather to run them concurrently
                await self._app.devices.delayer.set_lower_gate()
                await self._app.devices.spad.set_gate()
            except DeviceError:
                pass
            finally:
                self.main_gui.setPsdDelay.setEnabled(True)

    async def set_spad_gatewidth(self):
        """Doc."""

        if hasattr(self._app, "devices"):
            with suppress(DeviceError, ValueError):
                # ValueError:  writing/reading PSD too fast!
                await self._app.devices.spad.set_gate_width()

    def calibrate_pulse_sync_delay(self):
        """Doc."""

        delayer_dvc = self._app.devices.delayer
        current_sync_time_ns = delayer_dvc.sync_delay_ns
        new_sync_time_ns = delayer_dvc.eff_delay_ns

        if new_sync_time_ns != current_sync_time_ns:
            pressed = dialog.QuestionDialog(
                txt=f"Are you sure you wish to calibrate the new value of {new_sync_time_ns} ns as the new 'zero-gating' time? (current value is {current_sync_time_ns} ns)",
                title="Re-Calibrate Delayer Synchronization Time",
            ).display()
            if pressed is False:
                return
            else:
                delayer_dvc.calibrate_sync_time()
                self._gui.settings.impl.save()
        else:
            dialog.NotificationDialog(
                txt=f"Already calibrated to {new_sync_time_ns} ns!",
                title="Re-Calibrate Delayer Synchronization Time",
            ).display()

    def add_meas_to_queue(self, meas_type=None, laser_mode=None, **kwargs):
        """Doc."""

        if meas_type == "SFCSSolution":
            pattern = self.main_gui.solScanType.currentText()
            if pattern == "angular":
                scan_params = wdgts.SOL_ANG_SCAN_COLL.gui_to_dict(self._gui)
            elif pattern == "circle":
                scan_params = wdgts.SOL_CIRC_SCAN_COLL.gui_to_dict(self._gui)
            elif pattern == "static":
                scan_params = {}

            scan_params["pattern"] = pattern
            # get general scan parameters individually from GUI (can create a new widget collection for them and use .gui_to_dict)
            kwargs = wdgts.SOL_MEAS_COLL.gui_to_dict(self._app.gui)
            scan_params["floating_z_amplitude_um"] = kwargs["floating_z_amplitude_um"]
            scan_params["stage_pattern"] = kwargs["stage_pattern"]
            scan_params["stage_dwelltime_s"] = (
                kwargs["stage_dwelltime_min"] * 60
                if scan_params["stage_pattern"] != "None"
                else None
            )

            # add to FIFO queue
            measurement = meas.SolutionMeasurementProcedure(
                app=self._app,
                scan_params=scan_params,
                laser_mode=laser_mode.lower(),
                processing_options=self.get_processing_options_as_dict(),
                **kwargs,
            )
            self._app.meas_queue.append(measurement)
            gated_str = (
                f"gated{self._app.devices.spad.settings['gate_ns'].hard_gate.lower}ns"
                if self._app.devices.spad.settings["mode"] == "external"
                else "FR"
            )
            stage_str = f"{measurement.initial_stage_pos_steps.x:.0f}_{measurement.initial_stage_pos_steps.y:.0f}"
            new_item = QListWidgetItem(
                f"{kwargs['file_template']}_{pattern}_{laser_mode.lower()}_{gated_str}_StepperPos{stage_str} - {kwargs['duration']:.0f} {kwargs['duration_units']}{' (REPEATED)' if kwargs['repeat'] else ''}{' (FINAL)' if kwargs['final'] else ''}"
            )
            if kwargs["repeat"]:
                new_item.setBackground(QColor(255, 192, 203))
            elif kwargs["final"]:
                new_item.setBackground(QColor(255, 0, 0))
            self._gui.main.measQueue.addItem(new_item)
            self._gui.main.measQueue.setCurrentItem(new_item)
            logging.info(f"{meas_type} measurement added to FIFO queue.")

        elif meas_type == "SFCSImage":
            # get parameters from GUI
            kwargs = wdgts.IMG_MEAS_COLL.gui_to_dict(self._gui)
            scan_params = wdgts.IMG_SCAN_COLL.gui_to_dict(self._gui)
            scan_params["pattern"] = "image"

            # add to FIFO queue
            self._app.meas_queue.append(
                meas.ImageMeasurementProcedure(
                    app=self._app,
                    scan_params=scan_params,
                    laser_mode=laser_mode.lower(),
                    **kwargs,
                )
            )

    def move_meas_in_queue(self, direction: str):
        """Doc."""

        if direction == "UP":
            idx = -1
        elif direction == "DOWN":
            idx = 1
        else:
            raise ValueError(f"Direction must be 'UP' or 'DOWN' (got '{direction}')")

        with suppress(IndexError):
            # IndexError: no measurements in queue...
            meas_idx = (queue_widget := self._gui.main.measQueue).currentRow()
            if 0 <= (meas_idx + idx) <= queue_widget.count() - 1:
                list_item = queue_widget.takeItem(meas_idx)
                queue_widget.insertItem(meas_idx + idx, list_item)
                queue_widget.setCurrentRow(meas_idx + idx)
                meas = self._app.meas_queue.pop(meas_idx)
                self._app.meas_queue.insert(meas_idx + idx, meas)

    def remove_meas_from_queue(self):
        """Doc."""

        with suppress(IndexError):
            # IndexError: no measurements in queue...
            meas_idx = self._gui.main.measQueue.currentRow()
            self._gui.main.measQueue.takeItem(meas_idx)
            self._app.meas_queue.pop(meas_idx)

    async def fire_meas_queue(self, should_turn_off_dep=True, **kwargs):
        """Perform all measurements in queue (FIFO)"""

        if self._app.meas_queue:
            self._gui.main.measQueue.setEnabled(False)
            #            self._gui.main.measQueue.setCurrentRow(0)
            while True:
                # check if can start new measurements
                if not self._app.meas.is_running:
                    try:
                        new_meas = self._app.meas_queue.pop(0)
                    except IndexError:
                        # queue empty
                        break
                    else:
                        # increment the selected row
                        self._gui.main.measQueue.setCurrentRow(
                            self._app.gui.main.measQueue.currentRow() + 1
                        )
                        # perform measurement
                        await self.toggle_meas(new_meas)

                # pause between checks
                else:
                    await asyncio.sleep(0.5)

            # empty queue - final measurement
            self._gui.main.measQueue.clear()
            # relieve depletion laser
            if should_turn_off_dep:
                self._app.devices.dep_laser.laser_toggle(False)
            # re-enable queue
            self._gui.main.measQueue.setEnabled(True)
            # Turn off stage
            self._app.devices.stage.toggle(False)
            logging.debug("All measurements completed.")
        else:
            logging.info("No measurements in queue!")

    async def cancel_queue(self):
        """Cancel current measurement as any pending measurements"""

        # cancel queue
        self._app.meas_queue = []
        self._gui.main.measQueue.clear()
        # stop current measurement
        await self.toggle_meas(self._app.meas)
        # re-enable queue
        self._gui.main.measQueue.setEnabled(True)

    async def fire_single_meas(self, **kwargs):
        """Doc."""

        # start new single measurement
        if not self._app.meas.is_running:
            # cancel existing queue
            self._app.meas_queue = []
            self._gui.main.measQueue.clear()

            # add single measurement to queue and fire it
            self.add_meas_to_queue(**kwargs)
            await self.fire_meas_queue(**kwargs)

        # cancel current measurement
        else:
            await self.cancel_queue()

    async def toggle_meas(self, meas):
        """Doc."""

        if meas.type != (current_type := self._app.meas.type):

            # no meas running
            if not self._app.meas.is_running:

                #                # re-calibrate y-galvo before measurement (if needed)
                #                logging.info("Pefroming automatic Y-galvo calibration before measurement.")
                #                await self._app.gui.settings.impl.recalibrate_y_galvo(should_display=False)

                # adjust GUI
                if meas.type == "SFCSSolution":
                    self.main_gui.solScanMaxFileSize.setEnabled(False)
                    self.main_gui.solScanDur.setEnabled(self.main_gui.repeatSolMeas.isChecked())
                    self.main_gui.solScanDurUnits.setEnabled(False)
                    self.main_gui.solScanFileTemplate.setEnabled(False)
                    self.main_gui.beginMeasurements.setEnabled(False)
                    self.main_gui.stopMeasurements.setEnabled(True)
                    self.main_gui.stopMeasurement.setEnabled(True)
                    self.main_gui.removeMeasFromQueue.setEnabled(False)
                    self.main_gui.moveMeasUpQueue.setEnabled(False)
                    self.main_gui.moveMeasDownQueue.setEnabled(False)
                    self.main_gui.startSolQueueExc.setEnabled(False)
                    self.main_gui.startSolQueueSted.setEnabled(False)
                    self.main_gui.startSolQueueDep.setEnabled(False)

                elif meas.type == "SFCSImage":
                    self.main_gui.startImgScanExc.setEnabled(False)
                    self.main_gui.startImgScanDep.setEnabled(False)
                    self.main_gui.startImgScanSted.setEnabled(False)
                    getattr(
                        self.main_gui, f"startImgScan{meas.laser_mode.capitalize()}"
                    ).setEnabled(True)
                    getattr(self.main_gui, f"startImgScan{meas.laser_mode.capitalize()}").setText(
                        "Stop \nScan"
                    )

                # run the measurement
                self._app.meas = meas
                await meas.run()

            else:
                # other meas running
                logging.warning(
                    f"Another type of measurement " f"({current_type}) is currently running."
                )

        # measurement shutdown
        else:  # current_type == meas.type
            # adjust GUI
            if meas.type == "SFCSSolution":
                self.main_gui.solScanMaxFileSize.setEnabled(True)
                self.main_gui.solScanDur.setEnabled(True)
                self.main_gui.solScanDurUnits.setEnabled(True)
                self.main_gui.solScanFileTemplate.setEnabled(True)
                self.main_gui.beginMeasurements.setEnabled(True)
                self.main_gui.stopMeasurements.setEnabled(False)
                self.main_gui.stopMeasurement.setEnabled(False)
                self.main_gui.removeMeasFromQueue.setEnabled(True)
                self.main_gui.moveMeasUpQueue.setEnabled(True)
                self.main_gui.moveMeasDownQueue.setEnabled(True)
                self.main_gui.startSolQueueExc.setEnabled(True)
                self.main_gui.startSolQueueSted.setEnabled(True)
                self.main_gui.startSolQueueDep.setEnabled(True)

            if meas.type == "SFCSImage":
                self.main_gui.startImgScanExc.setEnabled(True)
                self.main_gui.startImgScanDep.setEnabled(True)
                self.main_gui.startImgScanSted.setEnabled(True)
                getattr(self.main_gui, f"startImgScan{meas.laser_mode.capitalize()}").setText(
                    f"{meas.laser_mode.capitalize()} \nScan"
                )

            # manual stop
            if self._app.meas.is_running:
                await self._app.meas.stop()
                self.remove_meas_from_queue()
                self._gui.main.measQueue.setCurrentRow(self._gui.main.measQueue.currentRow() - 1)
                logging.info(f"{meas.type} measurement stopped.")

    def disp_scn_pttrn(self, pattern: str):
        """Doc."""

        if pattern == "image":
            scan_params_coll = wdgts.IMG_SCAN_COLL
            plt_wdgt = self.main_gui.imgScanPattern
        elif pattern == "angular":
            scan_params_coll = wdgts.SOL_ANG_SCAN_COLL
            plt_wdgt = self.main_gui.solScanPattern
        elif pattern == "circle":
            scan_params_coll = wdgts.SOL_CIRC_SCAN_COLL
            plt_wdgt = self.main_gui.solScanPattern
        elif pattern == "static":
            scan_params_coll = None
            plt_wdgt = self.main_gui.solScanPattern

        if scan_params_coll:
            scan_params = scan_params_coll.gui_to_dict(self._gui)

            try:
                um_v_ratio = self._app.devices.scanners.um_v_ratio
            except AttributeError:
                ...
            else:
                with suppress(ZeroDivisionError, ValueError):
                    # ZeroDivisionError - loadout has bad values
                    ao, scan_params = ScanPatternAO(
                        pattern,
                        um_v_ratio,
                        self._app.devices.scanners.ao_int,
                        scan_params,
                    ).calculate_pattern()
                    x_data, y_data = ao[0, :], ao[1, :]
                    plt_wdgt.display_patterns(x_data, y_data)
                    # display the calculated parameters
                    scan_params_coll.obj_to_gui(self._gui, scan_params)

        else:
            # no scan
            plt_wdgt.display_patterns([], [])

    def open_settwin(self):
        """Doc."""

        self._gui.settings.show()
        self._gui.settings.activateWindow()

    def open_optwin(self):
        """Doc."""

        self._gui.options.show()
        self._gui.options.activateWindow()

    def counts_avg_interval_changed(self, interval_s: int) -> None:
        """Doc."""

        with suppress(AttributeError):
            # AttributeError - devices not yet initialized
            self._app.timeout_loop.cntr_avg_interval_s = interval_s

    ####################
    ## Image Tab
    ####################
    def roi_to_scan(self):
        """Doc"""

        plane_idx = self.main_gui.numPlaneShown.value()
        with suppress(AttributeError):
            # IndexError - 'App' object has no attribute 'curr_img_idx'
            image_tdc = ImageSFCSMeasurement()
            image_data = image_tdc.generate_ci_image_stack_data(
                file_dict=self._app.last_image_scans[self._app.curr_img_idx]
            )
            line_ticks_v = image_data.line_ticks_v
            row_ticks_v = image_data.row_ticks_v
            plane_ticks_v = image_data.plane_ticks_v

            coord_1, coord_2 = (
                round(pos_i) for pos_i in self.main_gui.imgScanPlot.axes[0].cursor.pos
            )

            dim_vltg = (line_ticks_v[coord_1], row_ticks_v[coord_2], plane_ticks_v[plane_idx])

            # order/sort the voltages according to dim_order
            vltgs = tuple(dim_vltg for _, dim_vltg in sorted(zip(image_data.dim_order, dim_vltg)))

            [
                getattr(self.main_gui, f"{axis.lower()}AOV").setValue(vltg)
                for axis, vltg in zip("XYZ", vltgs)
            ]

            self.move_scanners(image_data.plane_orientation)

            logging.debug(f"{self._app.devices.scanners.log_ref} moved to ROI ({vltgs})")

    def auto_scale_image(self, percent_factor: int):
        """Doc."""

        with suppress(AttributeError):
            # AttributeError - no last image yet
            image = self._app.curr_img
            try:
                image = auto_brightness_and_contrast(image, percent_factor)
            except (ZeroDivisionError, IndexError) as exc:
                err_hndlr(exc, sys._getframe(), locals(), lvl="warning")

            self.main_gui.imgScanPlot.display_image(
                image, cursor=True, imshow_kwargs=dict(cmap="bone")
            )

    def disp_plane_img(self, img_idx=None, plane_idx=None, auto_cross=False):
        """Doc."""

        method_dict = {
            "Forward scan - actual counts per pixel": "forward",
            "Forward scan normalization - points per pixel": "forward normalization",
            "Forward scan - normalized": "forward normalized",
            "Backwards scan - actual counts per pixel": "backward",
            "Backwards scan normalization - points per pixel": "backward normalization",
            "Backwards scan - normalized": "backward normalized",
            "Both scans - interlaced": "interlaced",
            "Both scans - averaged": "averaged",
        }

        def auto_crosshair_position(image: np.ndarray, thresholding=False) -> Tuple[float, ...]:
            """
            Beam co-alignment aid. Attempts to fit a 2D Gaussian to an image and returns the fitted center.
            If the fit fails, or if the fitted Gaussian is centered outside the image limits, it instead
            returns the image's center of mass.
            """

            try:
                fp = fit_tools.fit_2d_gaussian_to_image(image)
            except fit_tools.FitError:
                # Gaussian fit failed, using COM
                return helper.center_of_mass(image)
            else:
                x0, y0, sigma_x, sigma_y = (
                    fp.beta["x0"],
                    fp.beta["y0"],
                    fp.beta["sigma_x"],
                    fp.beta["sigma_y"],
                )
                height, width = image.shape
                if (
                    (0 < x0 < width)
                    and (0 < y0 < height)
                    and (sigma_x < width)
                    and (sigma_y < height)
                ):
                    # x0 and y0 are within image boundaries. Using the Gaussian fit
                    return (x0, y0)
                else:
                    # using COM
                    return helper.center_of_mass(image)

        if img_idx is None:
            try:
                img_idx = self._app.curr_img_idx
            except AttributeError:
                # .curr_img_idx not yet set
                img_idx = 0

        img_meas_wdgts = wdgts.IMG_MEAS_COLL.gui_to_dict(self._gui)
        disp_mthd = img_meas_wdgts["image_method"]
        with suppress(IndexError):
            # IndexError - No last_image_scans appended yet
            image_tdc = ImageSFCSMeasurement()
            image = image_tdc.preview(
                file_dict=self._app.last_image_scans[img_idx],
                method=method_dict[disp_mthd],
                plane_idx=plane_idx,
                should_plot=False,
            )
            self._app.curr_img_idx = img_idx
            self._app.curr_img = image
            img_meas_wdgts["image_wdgt"].obj.display_image(
                image, cursor=True, imshow_kwargs=dict(cmap="bone")
            )
            if auto_cross and image.any():
                img_meas_wdgts["image_wdgt"].obj.axes[0].cursor.move_to_pos(
                    auto_crosshair_position(image)
                )

    def plane_choice_changed(self, plane_idx):
        """Doc."""

        img_meas_wdgts = wdgts.IMG_MEAS_COLL.gui_to_dict(self._gui)
        img_meas_wdgts["plane_shown"].set(plane_idx)

        with suppress(AttributeError):
            # AttributeError - no scan performed since app init
            self.disp_plane_img(plane_idx=plane_idx)

    def fill_img_scan_preset_gui(self, curr_text: str) -> None:
        """Doc."""

        img_scn_wdgt_fillout_dict = {
            "Locate Plane - YZ Coarse": ["YZ", 15, 15, 0, None, None, None, 0.9, 1],
            "MFC - XY compartment": ["XY", 70, 70, 0, None, None, None, 0.9, 1],
            "GB -  XY Coarse": ["XY", 15, 15, 0, None, None, None, 0.9, 1],
            "GB - XY bead area": ["XY", 5, 5, 0, None, None, None, 0.9, 1],
            "GB - XY single bead": ["XY", 1, 1, 0, None, None, None, 0.9, 1],
            "GB - YZ single bead": ["YZ", 2.5, 2.5, 0, None, None, None, 0.9, 1],
        }

        wdgts.IMG_SCAN_COLL.obj_to_gui(self._gui, img_scn_wdgt_fillout_dict[curr_text])
        logging.debug(f"Image scan preset configuration chosen: '{curr_text}'")

    def cycle_through_image_scans(self, dir: str) -> None:
        """Doc."""

        with suppress(AttributeError):
            n_stored_images = len(self._app.last_image_scans)
            if dir == "next":
                if self._app.curr_img_idx > 0:
                    self.disp_plane_img(img_idx=self._app.curr_img_idx - 1, auto_cross=False)
            elif dir == "prev":
                if self._app.curr_img_idx < n_stored_images - 1:
                    self.disp_plane_img(img_idx=self._app.curr_img_idx + 1, auto_cross=False)

    def save_current_image(self) -> None:
        """Doc."""

        wdgt_coll = wdgts.IMG_MEAS_COLL.gui_to_dict(self._gui)

        with suppress(AttributeError):
            file_dict = self._app.last_image_scans[self._app.curr_img_idx]
            full_data = file_dict["full_data"]
            file_name = f"{wdgt_coll['file_template']}_{full_data['laser_mode']}_{full_data['scan_settings']['plane_orientation']}_{dt.now().strftime('%H%M%S')}"
            today_dir = Path(wdgt_coll["save_path"]) / dt.now().strftime("%d_%m_%Y")
            dir_path = today_dir / "image"
            file_path = dir_path / (re.sub("\\s", "_", file_name) + ".pkl")
            file_utilities.save_object(
                file_dict, file_path, compression_method="gzip", obj_name="image"
            )
            logging.debug(f"Saved measurement file: '{file_path}'.")

    ####################
    ## Solution Tab
    ####################
    def change_meas_duration(self, new_dur: float):
        """Allow duration change during run only for alignment measurements"""

        if self._app.meas.type == "SFCSSolution" and self._app.meas.repeat is True:
            self._app.meas.duration_s = new_dur * self._app.meas.duration_multiplier

    def fill_sol_meas_preset_gui(self, curr_text: str) -> None:
        """Doc."""

        sol_meas_wdgt_fillout_dict = {
            "Standard Static": {
                "scan_type": "static",
                "repeat": True,
                "should_plot": True,
                "should_fit": True,
                "should_accumulate_corrfuncs": False,
                "stage_pattern": "None",
            },
            "Standard Angular": {
                "scan_type": "angular",
                "regular": True,
                "should_plot": False,
                "should_fit": False,
                "should_accumulate_corrfuncs": True,
                "stage_pattern": "Snake",
            },
            "Standard Circular": {
                "scan_type": "circle",
                "regular": True,
                "should_plot": False,
                "should_fit": False,
                "should_accumulate_corrfuncs": True,
                "stage_pattern": "Snake",
            },
        }

        wdgts.SOL_MEAS_COLL.obj_to_gui(self._gui, sol_meas_wdgt_fillout_dict[curr_text])
        logging.debug(f"Solution measurement preset configuration chosen: '{curr_text}'")

    ####################
    ## Camera Dock
    ####################
    def toggle_camera_dock(self, is_toggled_on: bool) -> None:
        """Doc."""

        self.main_gui.cameraDock.setVisible(is_toggled_on)
        if is_toggled_on:
            if not self.cameras:
                slider_const = 100
                self.cameras = (self._app.devices.camera_1, self._app.devices.camera_2)
                for idx, camera in enumerate(self.cameras):
                    self.update_slider_range(cam_num=idx + 1)
                    [
                        getattr(self.main_gui, f"{name}{idx+1}").setValue(int(val) * slider_const)
                        for name, val in camera.DEFAULT_PARAM_DICT.items()
                    ]
            self.main_gui.move(0, 0)
            self.main_gui.setFixedSize(1700, 950)
        else:
            self.main_gui.move(300, 30)
            self.main_gui.setFixedSize(1310, 950)
            [
                self.device_toggle(
                    f"camera_{cam_num}", "toggle_video", "is_in_video_mode", leave_off=True
                )
                for cam_num in (1, 2)
            ]
        self.main_gui.setMaximumSize(int(1e5), int(1e5))

    def update_slider_range(self, cam_num: int) -> None:
        """Doc."""

        slider_const = 100

        camera = self.cameras[cam_num - 1]

        with suppress(AttributeError):
            # AttributeError - camera not properly initialized
            for name in ("pixel_clock", "framerate", "exposure"):
                range = tuple(
                    int(limit * slider_const) for limit in getattr(camera, f"{name}_range")
                )
                getattr(self.main_gui, f"{name}{cam_num}").setRange(*range)

    def set_parameter(self, cam_num: int, param_name: str, value) -> None:
        """Doc."""

        if not self.cameras:
            return

        slider_const = 100

        # convert from slider
        value /= slider_const

        getattr(self.main_gui, f"{param_name}_val{cam_num}").setValue(value)

        camera = self.cameras[cam_num - 1]
        with suppress(DeviceError):
            # DeviceError - camera not properly initialized
            camera.set_parameters({param_name: value})
            getattr(self.main_gui, f"autoExp{cam_num}").setChecked(False)
        self.update_slider_range(cam_num)

    def take_and_show_image(self, cam_num: int):
        """Doc."""

        camera = self.cameras[cam_num - 1]

        with suppress(DeviceError, ValueError):
            # TODO: not sure if 'suppress' is necessary
            # ValueError - new_image is None
            # DeviceError - error in camera
            self._app.loop.create_task(camera.get_image())
            logging.debug(f"Camera {cam_num} photo taken")

    def set_auto_exposure(self, cam_num: int, is_checked: bool):
        """Doc."""

        if not self.cameras:
            return

        camera = self.cameras[cam_num - 1]
        camera.set_auto_exposure(is_checked)
        if not is_checked:
            parameter_names = ("pixel_clock", "framerate", "exposure")
            parameter_values = (
                getattr(self.main_gui, f"{param_name}_val{cam_num}").value()
                for param_name in parameter_names
            )
            param_dict = {name: value for name, value in zip(parameter_names, parameter_values)}
            with suppress(DeviceError):
                camera.set_parameters(param_dict)

    def set_transposed_mode(self, cam_num: int, is_checked: bool):
        """Transpose camera images"""

        if not self.cameras:
            return

        camera = self.cameras[cam_num - 1]
        camera.is_in_transposed_mode = is_checked

    def set_grayscale_mode(self, cam_num: int, is_checked: bool):
        """Doc."""

        if not self.cameras:
            return

        camera = self.cameras[cam_num - 1]
        camera.is_in_grayscale_mode = is_checked

    def save_last_image(self, cam_num: int):
        """Doc."""

        camera = self.cameras[cam_num - 1]

        file_path = Path(self._gui.settings.camDataPath.text()) / Path(
            f"cam{cam_num}_" + dt.now().strftime("%d%m%y_%H%M%S") + ".pkl"
        )
        with suppress(AttributeError):
            file_utilities.save_object(
                camera.last_snapshot,
                file_path,
                compression_method="gzip",
                obj_name=f"cam{cam_num}_image",
            )

    def get_gaussian_diameter(self, cam_num: int, is_checked: bool):
        """Returns the Gaussian FWHM diameter (mm) of a beam image"""

        if self.cameras:
            camera = self.cameras[cam_num - 1]
            camera.should_get_diameter = is_checked

    ####################
    ## Analysis Tab - Raw Data
    ####################
    def switch_data_type(self) -> None:
        """Doc."""

        # TODO: switch thses names e.g. DATA_IMPORT_COLL -> data_import_wdgts
        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL.gui_to_dict(self._gui)

        if DATA_IMPORT_COLL["is_image_type"]:
            DATA_IMPORT_COLL["import_stacked"].set(0)
            DATA_IMPORT_COLL["analysis_stacked"].set(0)
            type_ = "image"
        elif DATA_IMPORT_COLL["is_solution_type"]:
            DATA_IMPORT_COLL["import_stacked"].set(1)
            DATA_IMPORT_COLL["analysis_stacked"].set(1)
            type_ = "solution"

        self.populate_all_data_dates(type_)

    def populate_all_data_dates(self, type_) -> None:
        """Doc."""

        # TODO: switch thses names e.g. DATA_IMPORT_COLL -> data_import_wdgts
        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL.gui_to_dict(self._gui)

        if type_ == "image":
            save_path = wdgts.IMG_MEAS_COLL.gui_to_dict(self._gui)["save_path"]
        elif type_ == "solution":
            save_path = wdgts.SOL_MEAS_COLL.gui_to_dict(self._gui)["save_path"]
        else:
            raise ValueError(
                f"Data type '{type_}'' is not supported; use either 'image' or 'solution'."
            )

        wdgts.DATA_IMPORT_COLL.clear_all_objects()

        with suppress(TypeError, IndexError, FileNotFoundError):
            # no directories found... (dir_years is None or [])
            # FileNotFoundError - data directory deleted while app running!
            dir_years = helper.dir_date_parts(save_path, type_)
            DATA_IMPORT_COLL["data_years"].obj.addItems(dir_years)

    def populate_data_dates_from_year(self, year: str) -> None:
        """Doc."""

        if not year:
            # ignore if combobox was just cleared
            return

        # define widgets
        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL.gui_to_dict(self._gui)

        if DATA_IMPORT_COLL["is_image_type"]:
            meas_sett = wdgts.IMG_MEAS_COLL.gui_to_dict(self._gui)
        elif DATA_IMPORT_COLL["is_solution_type"]:
            meas_sett = wdgts.SOL_MEAS_COLL.gui_to_dict(self._gui)
        save_path = meas_sett["save_path"]
        sub_dir = meas_sett["sub_dir_name"]

        DATA_IMPORT_COLL["data_months"].obj.clear()

        with suppress(TypeError, IndexError):
            # no directories found... (dir_years is None or [])
            dir_months = helper.dir_date_parts(save_path, sub_dir, year=year)
            DATA_IMPORT_COLL["data_months"].obj.addItems(dir_months)

    def populate_data_dates_from_month(self, month: str) -> None:
        """Doc."""

        if not month:
            # ignore if combobox was just cleared
            return

        # define widgets
        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL.gui_to_dict(self._gui)
        year = DATA_IMPORT_COLL["data_years"].get()
        days_combobox = DATA_IMPORT_COLL["data_days"].obj

        if DATA_IMPORT_COLL["is_image_type"]:
            meas_sett = wdgts.IMG_MEAS_COLL.gui_to_dict(self._gui)
        elif DATA_IMPORT_COLL["is_solution_type"]:
            meas_sett = wdgts.SOL_MEAS_COLL.gui_to_dict(self._gui)
        save_path = meas_sett["save_path"]
        sub_dir = meas_sett["sub_dir_name"]

        days_combobox.clear()

        with suppress(TypeError, IndexError):
            # no directories found... (dir_years is None or [])
            dir_days = helper.dir_date_parts(save_path, sub_dir, year=year, month=month)
            days_combobox.addItems(dir_days)

    def pkl_and_mat_templates(self, dir_path: Path) -> List[str]:
        """
        Accepts a directory path and returns a list of file templates
        ending in the form '*.pkl' or '*.mat' where * is any number.
        The templates are sorted by their time stamp of the form "HHMMSS".
        In case the folder contains legacy templates, they are sorted without a key.
        """

        is_solution_type = "solution" in dir_path.parts

        pkl_template_set = {
            (
                re.sub("[0-9]+.pkl", "*.pkl", str(file_path.name))
                if is_solution_type
                else file_path.name
            )
            for file_path in dir_path.iterdir()
            if file_path.suffix == ".pkl"
        }
        mat_template_set = {
            (
                re.sub("[0-9]+.mat", "*.mat", str(file_path.name))
                if is_solution_type
                else file_path.name
            )
            for file_path in dir_path.iterdir()
            if file_path.suffix == ".mat"
        }
        try:
            sorted_templates = sorted(
                pkl_template_set.union(mat_template_set),
                key=lambda tmpl: int(re.split("(\\d{6})_*", tmpl)[1]),
            )
        except IndexError:
            # Dealing with legacy templates which do not contain the appropriate time format
            sorted_templates = sorted(pkl_template_set.union(mat_template_set))
            logging.info(
                "Folder contains irregular file templates! Templates may not be properly sorted."
            )
        finally:
            return sorted_templates

    def populate_data_templates_from_day(self, day: str) -> None:
        """Doc."""

        if not day:
            # ignore if combobox was just cleared
            return

        # define widgets
        templates_combobox = wdgts.DATA_IMPORT_COLL.data_templates.obj

        templates_combobox.clear()

        with suppress(TypeError, IndexError):
            # no directories found... (dir_years is None or [])
            templates = self.pkl_and_mat_templates(self.current_date_type_dir_path())
            templates_combobox.addItems(templates)

    def show_num_files(self, template: str) -> None:
        """Doc."""

        if template:
            n_files_wdgt = wdgts.DATA_IMPORT_COLL.n_files
            dir_path = Path(self.current_date_type_dir_path())
            n_files = sum(1 for file_path in dir_path.glob(template))
            n_files_wdgt.set(f"({n_files} Files)")

    def cycle_through_data_templates(self, dir: str) -> None:
        """Cycle through the daily data templates in order (next or previous)"""

        data_templates_combobox = wdgts.DATA_IMPORT_COLL.data_templates.obj
        curr_idx = data_templates_combobox.currentIndex()
        n_items = data_templates_combobox.count()

        if dir == "next":
            if curr_idx + 1 < n_items:
                data_templates_combobox.setCurrentIndex(curr_idx + 1)
            else:  # cycle to beginning
                data_templates_combobox.setCurrentIndex(0)
        elif dir == "prev":
            if curr_idx - 1 >= 0:
                data_templates_combobox.setCurrentIndex(curr_idx - 1)
            else:  # cycle to end
                data_templates_combobox.setCurrentIndex(n_items - 1)

    def prefill_new_template(self, curr_template):
        """Pre-fill the 'new template' field with the modifiable part of the current template, (workflow)"""

        if curr_template:
            try:
                # get the current prefix - anything before the timestamp
                curr_template_prefix = re.findall("(^.*)(?=_[0-9]{6})", curr_template)[0]
            except IndexError:
                # legacy template which has no timestamp
                try:
                    curr_template_prefix = re.findall("(^.*)(?=_\\*\\.[a-z]{3})", curr_template)[0]
                except IndexError:
                    # legacy template which has no underscores
                    try:
                        curr_template_prefix = re.findall("(^.*)(?=\\*\\.[a-z]{3})", curr_template)[
                            0
                        ]
                    except IndexError:
                        # 'freestyle' template
                        curr_template_prefix = curr_template

            data_import_wdgts = wdgts.DATA_IMPORT_COLL.gui_to_dict(self._app.gui)
            data_import_wdgts["new_template"].set(curr_template_prefix)

    def rename_template(self) -> None:
        """Doc."""

        dir_path = self.current_date_type_dir_path()
        data_import_wdgts = wdgts.DATA_IMPORT_COLL.gui_to_dict(self._app.gui)
        curr_template = data_import_wdgts["data_templates"].get()
        new_template_prefix = data_import_wdgts["new_template"].get()

        # cancel if no new prefix supplied
        if not new_template_prefix:
            return

        try:
            # get the current prefix - anything before the timestamp
            curr_template_prefix = re.findall("(^.*)(?=_[0-9]{6})", curr_template)[0]
        except IndexError:
            # legacy template which has no timestamp
            try:
                curr_template_prefix = re.findall("(^.*)(?=_\\*\\.[a-z]{3})", curr_template)[0]
            except IndexError:
                # legacy template which has no underscores
                try:
                    curr_template_prefix = re.findall("(^.*)(?=\\*\\.[a-z]{3})", curr_template)[0]
                except IndexError:
                    # 'freestyle' template
                    curr_template_prefix = curr_template

        if new_template_prefix == curr_template_prefix:
            logging.warning(
                f"rename_template(): Requested template and current template are identical ('{curr_template}'). Operation canceled."
            )
            return

        curr_file_paths = [str(filepath) for filepath in dir_path.glob(curr_template)]
        curr_byte_data_paths = [
            str(
                Path(file_path_str).with_name(
                    Path(file_path_str).name.replace(".pkl", "_byte_data.npy")
                )
            )
            for file_path_str in curr_file_paths
        ]
        # check if current template doesn't exists (can only happen if deleted manually between discovering the template and running this function)
        if not curr_file_paths:
            logging.warning("Current template is missing! (Probably manually deleted)")
            return

        new_template = re.sub(curr_template_prefix, new_template_prefix, curr_template, count=1)
        # check if new template already exists (unlikely)
        if list(dir_path.glob(new_template)):
            logging.warning(
                f"New template '{new_template}' already exists in '{dir_path}'. Operation canceled."
            )
            return

        pressed = dialog.QuestionDialog(
            txt=f"Change current template from:\n{curr_template}\nto:\n{new_template}\n?",
            title="Edit File Template",
        ).display()
        if pressed is False:
            return

        # generate new filanames
        new_file_paths = [
            re.sub(curr_template_prefix, new_template_prefix, curr_filepath)
            for curr_filepath in curr_file_paths
        ]
        # rename the files
        [
            Path(curr_filepath).rename(new_filepath)
            for curr_filepath, new_filepath in zip(curr_file_paths, new_file_paths)
        ]
        logging.info(
            f"Renamed {len(curr_file_paths)} 'file_dict' files. ({curr_template} -> {new_template})"
        )

        # rename the data files
        try:
            [
                Path(curr_byte_data_path).rename(
                    str(
                        Path(new_filepath_str).with_name(
                            Path(new_filepath_str).name.replace(".pkl", "_byte_data.npy")
                        )
                    )
                )
                for curr_byte_data_path, new_filepath_str in zip(
                    curr_byte_data_paths, new_file_paths
                )
            ]
        except FileNotFoundError:
            print("Byte-data (.npy) files not found. Is this pre-separation data?")
        else:
            logging.info(
                f"Renamed {len(curr_file_paths)} 'byte_data' files. ({curr_template} -> {new_template})"
            )

        # rename the log file, if applicable
        with suppress(FileNotFoundError):
            pattern = re.sub("\\*?", "[0-9]*", curr_template)
            replacement = re.sub("_\\*", "", curr_template)
            current_log_filepath = re.sub(pattern, replacement, curr_file_paths[0])
            current_log_filepath = re.sub("\\.pkl", ".log", current_log_filepath)
            new_log_filepath = re.sub(
                curr_template_prefix, new_template_prefix, current_log_filepath
            )
            Path(current_log_filepath).rename(new_log_filepath)

        # refresh templates
        day = data_import_wdgts["data_days"]
        self.populate_data_templates_from_day(day)

    def current_date_type_dir_path(self) -> Path:
        """Returns path to directory of currently selected date and measurement type"""

        import_wdgts = wdgts.DATA_IMPORT_COLL.gui_to_dict(self._gui)

        day = import_wdgts["data_days"].get()
        year = import_wdgts["data_years"].get()
        month = import_wdgts["data_months"].get()

        if import_wdgts["is_image_type"]:
            meas_settings = wdgts.IMG_MEAS_COLL.gui_to_dict(self._gui)
        elif import_wdgts["is_solution_type"]:
            meas_settings = wdgts.SOL_MEAS_COLL.gui_to_dict(self._gui)
        save_path = Path(meas_settings["save_path"])
        sub_dir = meas_settings["sub_dir_name"]
        return save_path / f"{day.rjust(2, '0')}_{month.rjust(2, '0')}_{year}" / sub_dir

    def open_data_dir(self) -> None:
        """Doc."""

        if (dir_path := self.current_date_type_dir_path()).is_dir():
            webbrowser.open(str(dir_path))
        else:
            # dir was deleted, refresh all dates
            self.switch_data_type()

    def update_dir_log_file(self) -> None:
        """Doc."""

        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL
        text_lines = cast(str, DATA_IMPORT_COLL.log_text.get()).split("\n")
        curr_template = cast(str, DATA_IMPORT_COLL.data_templates.get())
        log_filename = Path(curr_template).stem.replace("_*", ".log")
        with suppress(AttributeError, TypeError, FileNotFoundError):
            # no directories found
            file_path = self.current_date_type_dir_path() / log_filename
            helper.list_to_file(file_path, text_lines)
            self.update_dir_log_wdgt(curr_template)

    def search_database(self) -> None:
        """Doc."""

        read_wdgts = {
            **wdgts.SOL_MEAS_COLL.gui_to_dict(self._gui),
            **wdgts.DATA_IMPORT_COLL.gui_to_dict(self._gui),
        }
        query_str_list = read_wdgts["database_search_query"].split()
        if query_str_list:
            data_root = Path(read_wdgts["save_path"])
            result_text = file_utilities.search_database(data_root, query_str_list)
            wdgts.DATA_IMPORT_COLL.database_search_results.set(result_text)

    def get_daily_alignment(self):
        """Doc."""

        date_dir_path = Path(re.sub("(solution|image)", "", str(self.current_date_type_dir_path())))

        try:
            # TODO: make this work for multiple templates (exc and sted), add both to log file and also the ratios
            template = self.pkl_and_mat_templates(date_dir_path)[0]
        except IndexError:
            # no alignment file found
            g0, tau = (None, None)
        else:
            # TODO: loading properly (err hndling) should be a seperate function

            # Inferring data_dype from template
            data_type = self.infer_data_type_from_template(template)

            meas = SolutionSFCSMeasurement(data_type)
            try:
                meas.read_fpga_data(date_dir_path / template)
                meas.correlate_and_average(
                    cf_name=f"{data_type} alignment", afterpulsing_method="filter"
                )

            #            except AttributeError:
            #                # No directories found
            #                g0, tau = (None, None)

            except (NotImplementedError, RuntimeError, ValueError, FileNotFoundError) as exc:
                err_hndlr(exc, sys._getframe(), locals())

            first_cf = list(meas.cf.values())[0]
            #            cf = meas.cf[data_type]

            try:
                fp = first_cf.fit_correlation_function()
            except fit_tools.FitError as exc:
                err_hndlr(exc, sys._getframe(), locals())
                g0, tau = (None, None)
            else:
                g0, tau = fp.beta["G0"], fp.beta["tau"]

        return g0, tau

    def update_dir_log_wdgt(self, template: str) -> None:
        """Doc."""

        def initialize_dir_log_file(file_path, g0, tau) -> None:
            """Doc."""
            # TODO: change arguments to exc_params, sted_params which would be tuples (g0, tau) and set the free atto sted alignment too

            basic_header = []
            basic_header.append("-" * 40)
            basic_header.append("Measurement Log File")
            basic_header.append("-" * 40)
            basic_header.append("Excitation Power: 20 uW @ BFP")
            basic_header.append("Depletion Power: 200 mW @ BFP (set to 260 mW)")
            if g0 is not None:
                basic_header.append(
                    f"Free Atto FCS @ 13.5 uW: G0 = {g0/1e3:.2f} k/ tau = {tau*1e3:.2f} us"
                )
            else:
                basic_header.append("Free Atto FCS @ 13.5 uW: Not Available")
            basic_header.append("-" * 40)
            basic_header.append("EDIT_HERE")

            helper.list_to_file(file_path, basic_header)

        if not template:
            return

        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL.gui_to_dict(self._gui)
        DATA_IMPORT_COLL["log_text"].set("")  # clear first

        # get the log file path
        if DATA_IMPORT_COLL["is_solution_type"]:
            log_filename = re.sub("_?\\*\\.\\w{3}", ".log", template)
        elif DATA_IMPORT_COLL["is_image_type"]:
            log_filename = re.sub("\\.\\w{3}", ".log", template)
        file_path = self.current_date_type_dir_path() / log_filename

        try:  # load the log file in path
            text_lines = helper.file_to_list(file_path)
        except (
            FileNotFoundError,
            OSError,
            UnicodeDecodeError,
        ):  # initialize a new log file if no existing file
            with suppress(OSError, IndexError):
                # OSError - missing file/folder (deleted during operation)
                # IndexError - alignment file does not exist
                initialize_dir_log_file(file_path, *self.get_daily_alignment())
                text_lines = helper.file_to_list(file_path)
        finally:  # write file to widget
            with suppress(UnboundLocalError):
                # UnboundLocalError
                DATA_IMPORT_COLL["log_text"].set("\n".join(text_lines))

    def save_processed_data(self):
        """Doc."""

        with self.get_measurement_from_template() as meas:
            was_saved = meas.save_processed(should_force=True, is_verbose=True)
            if was_saved:
                logging.info("Saved the processed data.")
            else:
                logging.info("Failed to save.")

    def delete_all_processed_data(self) -> None:
        """Doc."""

        import_wdgts = wdgts.DATA_IMPORT_COLL.gui_to_dict(self._gui)

        if import_wdgts["is_image_type"]:
            meas_settings = wdgts.IMG_MEAS_COLL.gui_to_dict(self._gui)
        elif import_wdgts["is_solution_type"]:
            meas_settings = wdgts.SOL_MEAS_COLL.gui_to_dict(self._gui)
        save_path = Path(meas_settings["save_path"])

        if processed_dir_paths := [
            item / "solution" / "processed"
            for item in save_path.iterdir()
            if (item / "solution" / "processed").is_dir()
        ]:
            pressed = dialog.QuestionDialog(
                txt="Are you sure you wish to delete all processed data?",
                title="Clearing Processed Data",
            ).display()
            if pressed is False:
                return

            for dir_path in processed_dir_paths:
                shutil.rmtree(dir_path, ignore_errors=True)

            self.main_gui.solImportLoadProcessed.setEnabled(False)

    def toggle_save_processed_enabled(self):
        """Doc."""

        with self.get_measurement_from_template() as meas:
            if meas is None:
                self.main_gui.solImportSaveProcessed.setEnabled(False)
            else:
                self.main_gui.solImportSaveProcessed.setEnabled(True)

    def toggle_load_processed_enabled(self, current_template: str):
        """Doc."""

        curr_dir = self.current_date_type_dir_path()
        file_path = curr_dir / "processed" / re.sub("_[*].pkl", "", current_template)
        self.main_gui.solImportLoadProcessed.setEnabled(file_path.exists())

    def convert_files_to_matlab_format(self) -> None:
        """
        Convert files of current template to '.mat',
        after translating their dictionaries to the legacy matlab format.
        """

        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL
        current_template = cast(str, DATA_IMPORT_COLL.data_templates.get())
        current_dir_path = self.current_date_type_dir_path()

        if not current_template or current_template.endswith(".mat"):
            return

        pressed = dialog.QuestionDialog(
            txt=f"Are you sure you wish to convert '{current_template}'?",
            title="Conversion to .mat Format",
        ).display()
        if pressed is False:
            return

        unsorted_paths = list(current_dir_path.glob(current_template))
        file_paths = file_utilities.sort_file_paths_by_file_number(unsorted_paths)
        byte_data_paths = [
            Path(file_path_str).with_name(
                Path(file_path_str).name.replace(".pkl", "_byte_data.npy")
            )
            for file_path_str in file_paths
        ]

        Path.mkdir(current_dir_path / "matlab", parents=True, exist_ok=True)

        print(f"Converting {len(file_paths)} files to '.mat' in legacy MATLAB format...", end=" ")
        for idx, (file_path, byte_data_path) in enumerate(zip(file_paths, byte_data_paths)):
            file_utilities.save_mat(file_path, byte_data_path)
            print(f"({idx+1})", end=" ")

        print("Done.")
        logging.info(f"{current_template} converted to MATLAB format")

    #####################
    ## Analysis Tab - Single Image
    #####################

    def preview_img_scan(self, template: str) -> None:
        """Doc."""

        if not template:
            return

        data_import_wdgts = wdgts.DATA_IMPORT_COLL.gui_to_dict(self._gui)

        if data_import_wdgts["is_image_type"]:
            # import the data
            try:
                image_tdc = ImageSFCSMeasurement()
                image = image_tdc.preview(
                    file_path=self.current_date_type_dir_path() / template,
                    should_plot=False,
                )
            except FileNotFoundError:
                self.switch_data_type()
                return

            else:
                # plot it (below)
                data_import_wdgts["img_preview_disp"].obj.display_image(
                    image, imshow_kwargs=dict(cmap="bone"), scroll_zoom=False
                )

    def import_image_data(self, should_load_processed=False) -> None:
        """Doc."""

        import_wdgts = wdgts.DATA_IMPORT_COLL.gui_to_dict(self._gui)
        current_template = import_wdgts["data_templates"].get()
        curr_dir = self.current_date_type_dir_path()

        if self._app.analysis.loaded_measurements.get(current_template) is not None:
            logging.info(f"Data '{current_template}' already loaded - ignoring.")
            return

        with self._app.pause_ai_ci():

            meas = None

            if should_load_processed or import_wdgts["auto_load_processed"]:
                try:
                    file_path = curr_dir / "processed" / re.sub("_[*].pkl", "", current_template)
                    logging.info(f"Loading processed data '{current_template}' from hard drive...")
                    meas = file_utilities.load_processed_solution_measurement(
                        file_path,
                        current_template,
                    )
                    print("Done.")
                except OSError:
                    print(
                        f"Pre-processed measurement not found at: '{file_path}'. Processing data regularly."
                    )

            if meas is None:  # process data
                options_dict = self.get_processing_options_as_dict()

                # Inferring data_dype from template
                data_type = self.infer_data_type_from_template(current_template)

                # Plotting lifetime images
                try:
                    meas = ImageSFCSMeasurement()
                    img_data = meas.generate_lifetime_image_stack_data(
                        file_path=curr_dir / current_template,
                        **options_dict,
                    )

                    from utilities.display import Plotter

                    with Plotter(
                        super_title=f"{current_template}:\nLifetime Images",
                        subplots=(1, meas.lifetime_image_data.image_stack_forward.shape[2]),
                    ) as axes:
                        try:
                            for plane_idx, ax in enumerate(axes):
                                ax.imshow(
                                    img_data.construct_plane_image("forward normalized", plane_idx)
                                )
                                ax.set_title(f"Scan/Plane #{plane_idx+1}")
                        except TypeError:
                            # 'Axes' object is not iterable
                            axes.imshow(img_data.construct_plane_image("forward normalized"))

                    # Plotting TDC images with resolution estimate
                    img_data = meas.tdc_image_data

                    with Plotter(
                        super_title=f"{current_template}:\nResolution Estimate",
                        subplots=(1, img_data.image_stack_forward.shape[2]),
                    ) as axes:
                        try:
                            try:
                                for plane_idx, ax in enumerate(axes):
                                    meas.estimate_spatial_resolution(
                                        parent_ax=ax, plane_idx=plane_idx
                                    )
                                    ax.set_title(f"Scan/Plane #{plane_idx+1}")
                            except TypeError:
                                # 'Axes' object is not iterable
                                meas.estimate_spatial_resolution(parent_ax=axes)
                        except RuntimeError as exc:
                            # Fit failed?
                            print(f"Spatial resolution estimate failed! [{exc}]")

                except (NotImplementedError, RuntimeError, ValueError, FileNotFoundError) as exc:
                    err_hndlr(exc, sys._getframe(), locals())
                    return

    ##########################
    ## Analysis Tab - Single Measurement
    ##########################

    def import_sol_data(self, should_load_processed=False) -> None:
        """Doc."""

        import_wdgts = wdgts.DATA_IMPORT_COLL.gui_to_dict(self._gui)
        current_template = import_wdgts["data_templates"].get()
        curr_dir = self.current_date_type_dir_path()

        if self._app.analysis.loaded_measurements.get(current_template) is not None:
            logging.info(f"Data '{current_template}' already loaded - ignoring.")
            return

        with self._app.pause_ai_ci():

            meas = None

            if should_load_processed or import_wdgts["auto_load_processed"]:
                try:
                    file_path = curr_dir / "processed" / re.sub("_[*].pkl", "", current_template)
                    logging.info(f"Loading processed data '{current_template}' from hard drive...")
                    meas = file_utilities.load_processed_solution_measurement(
                        file_path,
                        current_template,
                    )
                    print("Done.")
                except OSError:
                    print(
                        f"Pre-processed measurement not found at: '{file_path}'. Processing data regularly."
                    )
            with suppress(AttributeError):  # TODO: delete button should be disabled!
                if import_wdgts["should_re_correlate"]:
                    options_dict = self.get_processing_options_as_dict()
                    # Inferring data_dype from template
                    data_type = self.infer_data_type_from_template(current_template)

                    meas.correlate_data(
                        cf_name=data_type,
                        is_verbose=True,
                        **options_dict,
                    )

            if meas is None:  # process data
                options_dict = self.get_processing_options_as_dict()

                # Inferring data_dype from template
                data_type = self.infer_data_type_from_template(current_template)

                # loading and correlating
                try:
                    #                    with suppress(AttributeError):
                    #                        # AttributeError - No directories found
                    meas = SolutionSFCSMeasurement(data_type)
                    meas.read_fpga_data(
                        curr_dir / current_template,
                        **options_dict,
                    )
                    meas.correlate_data(
                        cf_name=data_type,
                        is_verbose=True,
                        **options_dict,
                    )

                except (NotImplementedError, RuntimeError, ValueError, FileNotFoundError) as exc:
                    err_hndlr(exc, sys._getframe(), locals())
                    return

            # save data and populate combobox
            imported_combobox1 = wdgts.SOL_MEAS_ANALYSIS_COLL.imported_templates
            imported_combobox2 = wdgts.SOL_EXP_ANALYSIS_COLL.imported_templates

            self._app.analysis.loaded_measurements[current_template] = meas

            imported_combobox1.obj.addItem(current_template)
            imported_combobox2.obj.addItem(current_template)

            imported_combobox1.set(current_template)
            imported_combobox2.set(current_template)

            logging.info(f"Data '{current_template}' ready for analysis.")

            self.toggle_save_processed_enabled()  # refresh save option
            self.toggle_load_processed_enabled(current_template)  # refresh load option

    def get_processing_options_as_dict(self) -> dict:
        """Doc."""

        loading_options = wdgts.PROC_OPTIONS_COLL.gui_to_dict(self._gui)

        # update loading_options with file selection # TODO: this should also move to 'PROC_OPTIONS_COLL'
        import_collection = wdgts.DATA_IMPORT_COLL.gui_to_dict(self._gui)
        # file selection
        if import_collection["sol_file_dicrimination"].objectName() == "solImportUse":
            loading_options[
                "file_selection"
            ] = f"{import_collection['sol_file_use_or_dont']} {import_collection['sol_file_selection']}"
        else:
            loading_options["file_selection"] = "Use All"
        # data slicing
        if loading_options["should_data_slice"]:
            start_idx = loading_options["slice_start_idx"]
            stop_idx = (
                loading_options["slice_stop_idx"]
                if loading_options["slice_stop_idx"] > start_idx
                else None
            )
            loading_options["byte_data_slice"] = slice(start_idx, stop_idx, None)

        return loading_options

    @contextmanager
    def get_measurement_from_template(
        self,
        template: str = "",
    ):
        """Doc."""

        if not template:
            template = cast(str, wdgts.SOL_MEAS_ANALYSIS_COLL.imported_templates.get())
        curr_data_type, *_ = re.split(" -", template)
        meas = self._app.analysis.loaded_measurements.get(curr_data_type)
        yield meas

    def infer_data_type_from_template(self, template: str) -> str:
        """Doc."""

        if ((data_type_hint := "_exc_") in template) or ((data_type_hint := "_sted_") in template):
            data_type = "confocal" if (data_type_hint == "_exc_") else "sted"
            return data_type
        else:
            print(
                f"Data type could not be inferred from template ({template}). Consider changing the template."
            )
            return "Unknown"

    def populate_sol_meas_analysis(self, imported_template):
        """Doc."""

        sol_data_analysis_wdgts = wdgts.SOL_MEAS_ANALYSIS_COLL.gui_to_dict(self._app.gui)

        with self.get_measurement_from_template(imported_template) as meas:
            if meas is None:
                # no imported templates (deleted)
                wdgts.SOL_MEAS_ANALYSIS_COLL.clear_all_objects()
                sol_data_analysis_wdgts["scan_img_file_num"].obj.setRange(1, 1)
                sol_data_analysis_wdgts["scan_img_file_num"].set(1)
            else:
                num_files = meas.n_files
                logging.debug("Populating analysis GUI...")

                # populate general measurement properties
                sol_data_analysis_wdgts["n_files"].set(num_files)
                sol_data_analysis_wdgts["scan_duration_min"].set(meas.duration_min)
                sol_data_analysis_wdgts["avg_cnt_rate_khz"].set(meas.avg_cnt_rate_khz)
                sol_data_analysis_wdgts["std_cnt_rate_khz"].set(meas.std_cnt_rate_khz)

                if meas.scan_type == "circle":
                    # populate scan image tab
                    logging.debug("Displaying scan images...")
                    sol_data_analysis_wdgts["scan_img_file_num"].obj.setEnabled(False)
                    sol_data_analysis_wdgts["scan_img_file_num"].set(0)
                    self.display_circular_scan_image(imported_template)
                    self.display_patterns(imported_template)

                    # calculate average and display
                    logging.debug("Averaging and plotting...")
                    self.calculate_and_show_sol_mean_acf(imported_template)

                    # TODO: make this into a function which generates the string from the scan_settings (handle non-Numpy Iterables)
                    scan_settings_text = "\n\n".join(
                        [
                            f"{key}: {np.array_str(val[:5], precision=2) if isinstance(val, np.ndarray) else (f'{val:.2f}' if helper.can_float(val) else val)}"
                            for key, val in meas.scan_settings.items()
                        ]
                    )

                if meas.scan_type == "angular":
                    # populate scan images tab
                    logging.debug("Displaying scan images...")
                    sol_data_analysis_wdgts["scan_img_file_num"].obj.setEnabled(True)
                    sol_data_analysis_wdgts["scan_img_file_num"].obj.setRange(1, num_files)
                    sol_data_analysis_wdgts["scan_img_file_num"].set(1)
                    self.display_angular_scan_image(imported_template)
                    self.display_patterns(imported_template)

                    # calculate average and display
                    logging.debug("Averaging and plotting...")
                    self.calculate_and_show_sol_mean_acf(imported_template)

                    # TODO: make this into a function which generates the string from the scan_settings (handle non-Numpy Iterables)
                    scan_settings_text = "\n\n".join(
                        [
                            f"{key}: {np.array_str(val[:5], precision=2) if isinstance(val, np.ndarray) else (f'{val:.2f}' if helper.can_float(val) else val)}"
                            for key, val in meas.scan_settings.items()
                        ]
                    )

                elif meas.scan_type == "static":
                    logging.debug("Averaging, plotting and fitting...")
                    self.calculate_and_show_sol_mean_acf(imported_template)
                    scan_settings_text = "no scan."
                    wdgts.SOL_MEAS_ANALYSIS_COLL.scan_image_disp.obj.clear()
                    wdgts.SOL_MEAS_ANALYSIS_COLL.pattern_wdgt.obj.clear()

                sol_data_analysis_wdgts["scan_settings"].set(scan_settings_text)

                logging.debug("Done.")

    def display_circular_scan_image(self, imported_template: str = None):
        """Doc."""

        with suppress(IndexError, KeyError, AttributeError):
            # IndexError - data import failed
            # KeyError, AttributeError - data deleted
            with self.get_measurement_from_template(imported_template) as meas:
                img = meas.scan_image

                scan_image_disp = wdgts.SOL_MEAS_ANALYSIS_COLL.scan_image_disp.obj
                scan_image_disp.display_image(img)
                scan_image_disp.entitle_and_label("Pixel Index", "File Index")

    def display_angular_scan_image(self, imported_template: str = None):
        """Doc."""

        # TODO: I still don't know why this happens/is needed, but sometimes I get an input of 2 seemingly out of nowhere
        if isinstance(imported_template, int):
            imported_template = None

        with suppress(IndexError, KeyError, AttributeError):
            # IndexError - data import failed
            # KeyError, AttributeError - data deleted
            with self.get_measurement_from_template(imported_template) as meas:

                sol_data_analysis_wdgts = wdgts.SOL_MEAS_ANALYSIS_COLL.gui_to_dict(self._app.gui)

                file_idx = sol_data_analysis_wdgts["scan_img_file_num"].get() - 1
                should_bw_mask = sol_data_analysis_wdgts["should_bw_mask"]
                should_normalize_rows = sol_data_analysis_wdgts["should_normalize_rows"]

                mask = meas.data[file_idx].general.image_bw_mask
                img = meas.scan_images_dstack[:, :, file_idx].copy()
                sec_roi_list = meas.rois[file_idx]

                if should_normalize_rows:
                    img = helper.normalize_scan_img_rows(img, mask)
                if should_bw_mask:
                    img *= mask

                scan_image_disp = wdgts.SOL_MEAS_ANALYSIS_COLL.scan_image_disp.obj
                scan_image_disp.display_image(img)
                for sec_roi in sec_roi_list:
                    scan_image_disp.plot(sec_roi["col"], sec_roi["row"], color="white", lw=0.6)
                scan_image_disp.entitle_and_label("Pixel Index", "Line Index")

    def display_patterns(self, imported_template: str = None):
        """Doc."""

        with suppress(IndexError, KeyError, AttributeError):
            # IndexError - data import failed
            # KeyError, AttributeError - data deleted
            with self.get_measurement_from_template(imported_template) as meas:
                display = wdgts.SOL_MEAS_ANALYSIS_COLL.pattern_wdgt.obj
                aox, aoy, *_ = meas.scan_settings["ao"].T
                aix, aiy, _, aox_int, aoy_int, _ = meas.scan_settings["ai"].T
                display.display_patterns(
                    [(aox, aoy), (aox_int, aoy_int), (aix, aiy)],
                    labels=["AO", "AO_int", "AI"],
                    scroll_zoom=True,
                )

    def calculate_and_show_sol_mean_acf(self, imported_template: str = None) -> None:
        """Doc."""

        sol_data_analysis_wdgts = wdgts.SOL_MEAS_ANALYSIS_COLL.gui_to_dict(self._app.gui)

        with self.get_measurement_from_template(imported_template) as meas:
            if meas is None:
                return

            if meas.scan_type == "circle":
                data_type = self.infer_data_type_from_template(meas.template)
                try:
                    cf = meas.cf[data_type]
                except KeyError:
                    # TODO: TEST THIS - possibly needed in other scan_types? (seems related to detector-gated measurements)
                    print(
                        f"Infered data_type ({data_type}) is not a key of CorrFunc dictionary (probably a detector-gated measurement). Using first CorrFunc"
                    )
                    cf = list(meas.cf.values())[0]

                cf.average_correlation()

                # setting values and plotting
                if sol_data_analysis_wdgts["plot_spatial"]:
                    x = cf.vt_um
                    x_label = r"disp. ($um^2$)"
                else:
                    x = cf.lag
                    x_label = "lag (ms)"

                sol_data_analysis_wdgts["mean_g0"].set(cf.g0 / 1e3)  # shown in thousands
                sol_data_analysis_wdgts["mean_tau"].set(0)
                sol_data_analysis_wdgts["row_acf_disp"].obj.clear()
                sol_data_analysis_wdgts["row_acf_disp"].obj.plot_acfs(
                    x,
                    cf.avg_cf_cr,
                    cf.g0,
                )
                sol_data_analysis_wdgts["row_acf_disp"].obj.entitle_and_label(x_label, "G0")

                sol_data_analysis_wdgts["countrate_disp"].obj.plot(
                    np.cumsum(cf.split_durations_s),
                    cf.countrate_list,
                    should_clear=True,
                )
                sol_data_analysis_wdgts["countrate_disp"].obj.entitle_and_label(
                    "measurement time (s)", "countrate"
                )

            if meas.scan_type == "angular":
                row_disc_method = sol_data_analysis_wdgts["row_dicrimination"].objectName()
                if row_disc_method == "solAnalysisRemoveDbscan":
                    avg_corr_kwargs = dict(
                        should_use_clustering=True,
                        min_noise_thresh=sol_data_analysis_wdgts["dbscan_noise_thresh"],
                    )
                elif row_disc_method == "solAnalysisRemoveOver":
                    avg_corr_kwargs = dict(rejection=sol_data_analysis_wdgts["remove_over"])
                elif row_disc_method == "solAnalysisRemoveWorst":
                    avg_corr_kwargs = dict(
                        rejection=None, reject_n_worst=sol_data_analysis_wdgts["remove_worst"].get()
                    )
                else:  # use all rows
                    avg_corr_kwargs = dict(rejection=None)

                with suppress(AttributeError, RuntimeError):
                    # AttributeError - no data loaded
                    data_type = self.infer_data_type_from_template(meas.template)
                    try:
                        cf = meas.cf[data_type]
                    except KeyError:
                        cf = list(meas.cf.values())[-1]
                        print(
                            "Warning! data type was errorneously inferred from template - possible bad template naming/gating. Using last CorrFunc object."
                        )
                    cf.average_correlation(**avg_corr_kwargs)

                    if sol_data_analysis_wdgts["plot_spatial"]:
                        x = cf.vt_um
                        x_label = r"squared displacement ($um^2$)"
                    else:
                        x = cf.lag
                        x_label = "lag (ms)"

                    sol_data_analysis_wdgts["row_acf_disp"].obj.plot_acfs(
                        x,
                        cf.avg_cf_cr,
                        cf.g0,
                        cf.cf_cr[cf.j_good, :],
                    )
                    sol_data_analysis_wdgts["row_acf_disp"].obj.entitle_and_label(x_label, "G0")

                    sol_data_analysis_wdgts["countrate_disp"].obj.plot(
                        np.cumsum(cf.split_durations_s),
                        cf.countrate_list,
                        should_clear=True,
                    )
                    sol_data_analysis_wdgts["countrate_disp"].obj.entitle_and_label(
                        "measurement time (s)", "countrate"
                    )

                    sol_data_analysis_wdgts["mean_g0"].set(cf.g0 / 1e3)  # shown in thousands
                    sol_data_analysis_wdgts["mean_tau"].set(0)

                    sol_data_analysis_wdgts["n_good_rows"].set(n_good := len(cf.j_good))
                    sol_data_analysis_wdgts["n_bad_rows"].set(n_bad := len(cf.j_bad))
                    sol_data_analysis_wdgts["remove_worst"].obj.setMaximum(n_good + n_bad - 2)

            elif meas.scan_type == "static":
                data_type = self.infer_data_type_from_template(meas.template)
                cf = list(meas.cf.values())[0]
                cf.average_correlation()
                try:
                    proc_opt_dict = self.get_processing_options_as_dict()
                    fp = cf.fit_correlation_function(
                        should_weight_fits=proc_opt_dict["should_weight_fits"]
                    )
                except fit_tools.FitError as exc:
                    # fit failed, use g0 calculated in 'average_correlation()'
                    err_hndlr(exc, sys._getframe(), locals(), lvl="warning")
                    sol_data_analysis_wdgts["mean_g0"].set(cf.g0 / 1e3)  # shown in thousands
                    sol_data_analysis_wdgts["mean_tau"].set(0)
                    sol_data_analysis_wdgts["row_acf_disp"].obj.clear()
                    sol_data_analysis_wdgts["row_acf_disp"].obj.plot_acfs(
                        cf.lag,
                        cf.avg_cf_cr,
                        cf.g0,
                    )
                else:  # fit succeeded
                    g0, tau = fp.beta["G0"], fp.beta["tau"]
                    fit_func = fp.fit_func
                    sol_data_analysis_wdgts["mean_g0"].set(g0 / 1e3)  # shown in thousands
                    sol_data_analysis_wdgts["mean_tau"].set(tau * 1e3)
                    y_fit = fit_func(cf.lag, *fp.beta.values())
                    sol_data_analysis_wdgts["row_acf_disp"].obj.clear()
                    sol_data_analysis_wdgts["row_acf_disp"].obj.plot_acfs(
                        cf.lag,
                        cf.avg_cf_cr,
                        cf.g0,
                    )
                    sol_data_analysis_wdgts["row_acf_disp"].obj.plot(cf.lag, y_fit, color="red")
                finally:
                    sol_data_analysis_wdgts["countrate_disp"].obj.plot(
                        np.cumsum(cf.split_durations_s),
                        cf.countrate_list,
                        should_clear=True,
                    )

    def assign_template(self, type) -> None:
        """Doc."""

        curr_template = wdgts.SOL_MEAS_ANALYSIS_COLL.imported_templates.get()
        assigned_wdgt = getattr(wdgts.SOL_MEAS_ANALYSIS_COLL, type)
        assigned_wdgt.set(curr_template)

    def remove_imported_template(self) -> None:
        """Doc."""

        imported_templates1 = wdgts.SOL_MEAS_ANALYSIS_COLL.imported_templates
        imported_templates2 = wdgts.SOL_EXP_ANALYSIS_COLL.imported_templates

        template = imported_templates1.get()
        self._app.analysis.loaded_measurements[template] = None

        imported_templates1.obj.removeItem(imported_templates1.obj.currentIndex())
        imported_templates2.obj.removeItem(imported_templates2.obj.currentIndex())

        self.toggle_save_processed_enabled()  # refresh save option

        # TODO: clear image properties!

    #########################
    ## Analysis Tab - Single Experiment
    #########################

    def assign_measurement(self, meas_type: str) -> None:
        """Doc."""

        wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.gui_to_dict(self._gui)
        options_dict = self.get_processing_options_as_dict()

        if wdgt_coll["should_assign_loaded"]:
            template = wdgt_coll["imported_templates"].get()
            method = "loaded"
        elif wdgt_coll["should_assign_raw"]:
            template = wdgt_coll["data_templates"].get()
            method = "raw"

        MeasAssignParams = namedtuple("MeasAssignParams", "template method options")
        self._app.analysis.assigned_to_experiment[meas_type] = MeasAssignParams(
            template, method, options_dict
        )

        wdgt_to_assign_to = wdgt_coll[f"assigned_{meas_type}_template"]
        wdgt_to_assign_to.set(template)

    def load_experiment(self) -> None:
        """Doc."""

        wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.gui_to_dict(self._gui)
        data_import_wdgt_coll = wdgts.DATA_IMPORT_COLL.gui_to_dict(self._gui)

        if not (experiment_name := wdgt_coll["experiment_name"].get()):
            logging.info("Can't load unnamed experiment!")
            return

        kwargs = dict()
        with suppress(KeyError):
            for meas_type, assignment_params in self._app.analysis.assigned_to_experiment.items():

                if assignment_params.method == "loaded":
                    with self.get_measurement_from_template(
                        wdgt_coll[f"assigned_{meas_type}_template"].get(),
                    ) as meas:
                        kwargs[meas_type] = meas

                elif assignment_params.method == "raw":
                    curr_dir = self.current_date_type_dir_path()
                    kwargs[f"{meas_type}_template"] = (
                        curr_dir / wdgt_coll[f"assigned_{meas_type}_template"].get()
                    )
                    kwargs["force_processing"] = not data_import_wdgt_coll["auto_load_processed"]
                    kwargs["should_re_correlate"] = data_import_wdgt_coll["should_re_correlate"]
                # get loading options as kwargs
                kwargs[f"{meas_type}_kwargs"] = assignment_params.options

        # plotting properties
        kwargs["gui_display"] = wdgt_coll["gui_display_loading"].obj
        kwargs["gui_options"] = GuiDisplay.GuiDisplayOptions(show_axis=True)
        kwargs["fontsize"] = 10
        kwargs["should_plot_meas"] = False

        experiment = SolutionSFCSExperiment(experiment_name)
        try:
            experiment.load_experiment(**kwargs)
        except RuntimeError as exc:
            logging.info(str(exc))
        else:
            # save experiment in memory and GUI
            self._app.analysis.loaded_experiments[experiment_name] = experiment
            wdgt_coll["loaded_experiments"].obj.addItem(experiment_name)

            # save measurements seperately in memory and GUI
            for meas_type in ("confocal", "sted"):
                with suppress(AttributeError):
                    # AttributeError - measurement does not exist
                    meas = getattr(experiment, meas_type)
                    self._app.analysis.loaded_measurements[meas.template] = meas
                    wdgts.SOL_MEAS_ANALYSIS_COLL.imported_templates.obj.addItem(meas.template)
                    wdgts.SOL_EXP_ANALYSIS_COLL.imported_templates.obj.addItem(meas.template)

            # reset the dict and clear relevant widgets
            self._app.analysis.assigned_to_experiment = dict()
            wdgt_coll["assigned_confocal_template"].obj.clear()
            wdgt_coll["assigned_sted_template"].obj.clear()
            wdgt_coll["experiment_name"].obj.clear()

            with suppress(IndexError):
                # IndexError - either confocal or STED weren't loaded
                conf_g0 = list(experiment.confocal.cf.values())[0].g0
                sted_g0 = list(experiment.sted.cf.values())[0].g0
                wdgt_coll["g0_ratio"].set(conf_g0 / sted_g0)
            logging.info(f"Experiment '{experiment_name}' loaded successfully.")

            # display existing TDC calibrations in appropriate tab
            kwargs["gui_display"] = wdgt_coll["gui_display_tdc_cal"].obj
            experiment.plot_tdc_calib(**kwargs)

            # estimate resolution
            kwargs["gui_display"] = wdgt_coll["gui_display_resolution"].obj
            experiment.estimate_spatial_resolution(**kwargs)

    def remove_experiment(self) -> None:
        """Doc."""

        loaded_templates_combobox = wdgts.SOL_EXP_ANALYSIS_COLL.loaded_experiments
        # delete from memory
        if experiment_name := loaded_templates_combobox.get():
            self._app.analysis.loaded_experiments[experiment_name] = None
            # delete from GUI
            loaded_templates_combobox.obj.removeItem(loaded_templates_combobox.obj.currentIndex())
            logging.info(f"Experiment '{experiment_name}' removed.")

    def get_experiment(
        self,
    ) -> SolutionSFCSExperiment:
        """Doc."""

        experiment_name = wdgts.SOL_EXP_ANALYSIS_COLL.loaded_experiments.get()
        return self._app.analysis.loaded_experiments.get(experiment_name)

    def calibrate_tdc(self) -> None:
        """Doc."""

        wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.gui_to_dict(self._gui)

        display_kwargs = dict(gui_options=GuiDisplay.GuiDisplayOptions(show_axis=True), fontsize=10)
        experiment = self.get_experiment()
        try:
            display_kwargs["gui_display"] = wdgt_coll["gui_display_tdc_cal"].obj
            experiment.calibrate_tdc(
                calib_time_ns=wdgt_coll["calibration_gating"], is_verbose=True, **display_kwargs
            )
        except RuntimeError:  # confocal not loaded
            logging.info("Confocal measurement was not loaded - cannot calibrate TDC.")
        except AttributeError:
            logging.info("Can't calibrate TDC, no experiment is loaded!")
        else:
            display_kwargs["gui_display"] = wdgt_coll["gui_display_comp_lifetimes"].obj
            experiment.compare_lifetimes(**display_kwargs)
            self.get_lifetime_params()

    def get_lifetime_params(self):
        """Doc."""

        experiment = self.get_experiment()
        if experiment is not None and hasattr(experiment.confocal, "scan_type"):
            wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.gui_to_dict(self._gui)

            if hasattr(experiment.confocal, "scan_type"):
                lt_params = experiment.get_lifetime_parameters()

                # display parameters in GUI
                wdgt_coll["fluoresence_lifetime"].set(lt_params.lifetime_ns)
                wdgt_coll["laser_pulse_delay"].set(lt_params.laser_pulse_delay_ns)
                if hasattr(experiment.sted, "scan_type"):
                    wdgt_coll["sigma_sted"].set(lt_params.sigma_sted)

    def assign_gate(self) -> None:
        """Doc."""

        wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.gui_to_dict(self._gui)

        gate = helper.Gate(
            wdgt_coll["custom_gate_to_assign_lower"],
            wdgt_coll["custom_gate_to_assign_upper"],
            units=None,
        )

        if wdgt_coll["gate_units"] == "lifetimes":
            experiment = self.get_experiment()
            try:
                laser_pulse_delay_ns = experiment.lifetime_params.laser_pulse_delay_ns
                lifetime_ns = experiment.lifetime_params.lifetime_ns
            except AttributeError:
                logging.info(
                    "Cannot use lifetime gate if no STED measurement was loaded prior to TDC calibration! (laser pulse time and fluorophore lifetime are derived there)"
                )
                return
            else:
                gate_ns = gate * lifetime_ns + laser_pulse_delay_ns
        else:
            gate_ns = gate

        gates_combobox = wdgt_coll["assigned_gates"].obj
        existing_gate_list = [
            helper.Gate(gates_combobox.itemText(i), from_string=True)
            for i in range(gates_combobox.count())
        ]

        if gate_ns not in existing_gate_list:  # add only new gates
            wdgt_coll["assigned_gates"].obj.addItem(str(gate_ns))

    def remove_assigned_gate(self) -> None:
        """Doc."""

        wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.gui_to_dict(self._gui)
        wdgt = wdgt_coll["assigned_gates"]
        if gate_to_remove := wdgt.get():
            # delete from GUI
            wdgt.obj.removeItem(wdgt.obj.currentIndex())
            logging.info(f"Gate '{gate_to_remove}' was unassigned.")

    def gate(self) -> None:
        """Doc."""

        wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.gui_to_dict(self._gui)

        gates_combobox = wdgt_coll["assigned_gates"].obj
        gate_list = [
            helper.Gate(gates_combobox.itemText(i), from_string=True)
            for i in range(gates_combobox.count())
        ]

        experiment = self.get_experiment()
        proc_options_dict = self.get_processing_options_as_dict()
        kwargs = dict(
            should_plot=True,
            gui_display=wdgt_coll["gui_display_sted_gating"].obj,
            gui_options=GuiDisplay.GuiDisplayOptions(show_axis=True),
            fontsize=10,
            meas_type=wdgt_coll["gate_meas_type"].lower(),
            **proc_options_dict,
        )
        try:
            experiment.add_gates(gate_list, **kwargs)
        except AttributeError:
            logging.info("Can't gate, no experiment is loaded!")
        else:
            wdgt_coll["available_gates"].obj.addItems([str(gate) for gate in gate_list])

        # estimate resolution again (include new gates
        kwargs["gui_display"] = wdgt_coll["gui_display_resolution"].obj
        experiment.estimate_spatial_resolution(**kwargs)

    def remove_available_gate(self) -> None:
        """Doc."""

        wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.gui_to_dict(self._gui)
        wdgt = wdgt_coll["available_gates"]
        gate_to_remove = wdgt.get()
        # delete from object
        experiment = self.get_experiment()
        try:
            meas_type = wdgt_coll["gate_meas_type"].lower()
            getattr(experiment, meas_type).cf.pop(f"gated {meas_type} {gate_to_remove}")
        except AttributeError:  # no experiment loaded or no STED
            # TODO: this shouldn't happen - the list of gates needs to be cleared when experiment is cleared
            logging.info(
                "Can't remove gate. (this shouldn't happen - the list of gates needs to be cleared when experiment is cleared)"
            )
        except KeyError:  # no assigned gate
            # TODO: clear the available_gates when removing experiment!
            print("THIS SHOULD NOT HAPPEN! (remove_available_gate)")
            pass
        else:
            # re-plot
            kwargs = dict(
                gui_display=wdgt_coll["gui_display_sted_gating"].obj,
                gui_options=GuiDisplay.GuiDisplayOptions(show_axis=True),
                fontsize=10,
            )
            experiment.plot_correlation_functions(**kwargs)
            # delete from GUI
            wdgt.obj.removeItem(wdgt.obj.currentIndex())
            logging.info(f"Gate '{gate_to_remove}' removed from '{experiment.name}' experiment.")

            # estimate resolution again (withouut gate)
            kwargs["gui_display"] = wdgt_coll["gui_display_resolution"].obj
            experiment.estimate_spatial_resolution(**kwargs)

    def calculate_structure_factors(self) -> None:
        """Doc."""

        wdgt_coll = wdgts.SOL_EXP_ANALYSIS_COLL.gui_to_dict(self._gui)
        experiment = self.get_experiment()
        cal_exp = self.get_experiment()  # TESTESTEST - need to set actual calibration experiment
        try:
            experiment.calculate_structure_factors(
                cal_exp,  # TESTESTEST - need to set actual calibration experiment
                #                fr_interp_lims=(wdgt_coll["g_min"], np.inf),
                n_robust=wdgt_coll["n_robust"],
            )
        except AttributeError:
            logging.info("Can't calculate structure factors, no experiment is loaded!")
        else:
            logging.info(f"Calculated all structure factors for '{experiment.name}' experiment.")


class SettWin:
    """Doc."""

    def __init__(self, settings_gui, app):
        """Doc."""

        self._app = app
        self._gui = app.gui
        self.settings_gui = settings_gui
        self.check_on_close = True

    def clean_up(self):
        """Doc."""

        if self.check_on_close is True:
            curr_file_path = self.settings_gui.settingsFileName.text()

            current_state = set(
                wdgts.wdgt_items_to_text_lines(self.settings_gui, wdgts.SETTINGS_TYPES)
            )
            last_loaded_state = set(helper.file_to_list(curr_file_path))

            if not len(current_state) == len(last_loaded_state):
                err_hndlr(
                    RuntimeError(
                        "Something was changed in the GUI. This probably means that the default settings need to be overwritten"
                    ),
                    sys._getframe(),
                    locals(),
                )

            if current_state != last_loaded_state:
                pressed = dialog.QuestionDialog(
                    "Keep changes if made? " "(otherwise, revert to last loaded settings file.)"
                ).display()
                if pressed is False:
                    self.load(self.settings_gui.settingsFileName.text())

        else:
            dialog.NotificationDialog("Using Current settings.").display()

    def save(self) -> None:
        """
        Write all QLineEdit, QspinBox and QdoubleSpinBox
        of settings window to 'file_path'.
        Show 'file_path' in 'settingsFileName' QLineEdit.
        """

        file_path, _ = QFileDialog.getSaveFileName(
            self.settings_gui,
            "Save Settings",
            str(self._app.SETTINGS_DIR_PATH),
        )
        if file_path != "":
            self.settings_gui.frame.findChild(QWidget, "settingsFileName").setText(file_path)
            wdgts.write_gui_to_file(self.settings_gui.frame, wdgts.SETTINGS_TYPES, file_path)
            logging.debug(f"Settings file saved as: '{file_path}'")

    def load(self, file_path=""):
        """
        Read 'file_path' and write to matching widgets of settings window.
        Show 'file_path' in 'settingsFileName' QLineEdit.
        Default settings is chosen according to './settings/default_settings_choice'.
        """

        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self.settings_gui,
                "Load Settings",
                str(self._app.SETTINGS_DIR_PATH),
            )
        if file_path != "":
            self.settings_gui.frame.findChild(QWidget, "settingsFileName").setText(str(file_path))
            wdgts.read_file_to_gui(file_path, self.settings_gui.frame)
            logging.debug(f"Settings file loaded: '{file_path}'")

    def get_all_device_details(self):
        """Doc."""

        dvc_list = []

        # pyvisa
        dvc_list.append("VISA:")
        visa_rm = pyvisa.ResourceManager()
        for resource_name in visa_rm.list_resources():
            with suppress(pyvisa.errors.VisaIOError):
                # VisaIOError - failed to open instrument, skip
                inst = visa_rm.open_resource(resource_name)
                dvc_list.append(f"{inst.resource_info.alias}: {inst.resource_name}")
                inst.close()

        # nidaqmx
        dvc_list.append("\nNI:")
        system = nisys.System.local()
        dvc_list += system.devices.device_names

        # ftd2xx
        dvc_list.append("\nFTDI:")
        # auto-find UM232H serial number
        num_devs = ftd2xx.createDeviceInfoList()
        for idx in range(num_devs):
            info_dict = ftd2xx.getDeviceInfoDetail(devnum=idx)
            dvc_list.append(info_dict["description"].decode("utf-8"))

        self.settings_gui.deviceDetails.setText("\n".join(dvc_list))

    async def recalibrate_y_galvo(self, should_display=True):
        """
        Perform a circular scan using current um/Volt calibration parameters
        and display analog I/O in order to gauge said calibration.
        """

        # perform a short circular scan
        scan_params = dict(
            pattern="circle",
            ao_sampling_freq_hz=10000,
            diameter_um=20,
            speed_um_s=6000,
            n_circles=960,
        )
        self._app.meas = meas.SolutionMeasurementProcedure(
            self._app,
            scan_params,
            scan_type="circle",
            laser_mode="nolaser",
            duration=0.1,
            duration_units="seconds",
            final=False,
            repeat=False,  # TODO: why are both final and repeat needed? aren't they mutually exclusive?
            should_plot_acf=False,
        )
        # run measurement
        await self._app.meas.run(should_save=False)

        # auto-fix the issue if needed
        ai_buffer = self._app.last_meas_data["full_data"]["scan_settings"]["ai"]
        #        self._app.devices.scanners.recalibrate_y_galvo(scan_params, ai_buffer) # equate axes (wrong!)

        # display the AO_ext and AI in the appointed widget
        if should_display:
            display = self.settings_gui.xyCalibDisplay
            aix, aiy, _, aox_int, aoy_int, _ = ai_buffer.T
            display.display_patterns(
                [(aox_int, aoy_int), (aix, aiy)],
                labels=["AO_int", "AI"],
                scroll_zoom=True,
            )


class ProcessingOptionsWindow:
    """Doc."""

    def __init__(self, processing_gui, app):
        """Doc."""

        self._app = app
        self._gui = app.gui
        self.processing_gui = processing_gui

    def save(self) -> None:
        """Save default options"""

        file_path = self._app.DEFAULT_PROCESSING_OPTIONS_FILE_PATH
        wdgts.write_gui_to_file(self.processing_gui.frame, wdgts.SETTINGS_TYPES, file_path)
        logging.debug(f"Processing options file saved as: '{file_path}'")

    def load(self):
        """Load default options"""

        file_path = self._app.DEFAULT_PROCESSING_OPTIONS_FILE_PATH
        wdgts.read_file_to_gui(file_path, self.processing_gui.frame)
        logging.debug(f"Processing options file loaded: '{file_path}'")
