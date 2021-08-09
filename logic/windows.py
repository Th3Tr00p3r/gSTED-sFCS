""" GUI windows implementations module. """

import glob
import logging
import os
import re
import sys
import webbrowser
from collections.abc import Iterable
from contextlib import suppress
from types import SimpleNamespace
from typing import Tuple

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget

import logic.devices as dvcs
import logic.measurements as meas
import utilities.helper as helper
import utilities.widget_collections as wdgt_colls
from data_analysis import file_utilities, fit_tools
from data_analysis.correlation_function import CorrFuncTDC
from data_analysis.photon_data import PhotonData
from logic.scan_patterns import ScanPatternAO
from utilities.dialog import Error, Notification, Question
from utilities.errors import DeviceError, err_hndlr


# TODO: refactoring - this is getting too big, I need to think on how to break it down.
# Perhaps I can divide MainWin into seperate classes for each tab (image, solution, analysis).
# That would require either replacing the gui.main object with 3 different ones (gui.image_tab, gui.solution_tab, gui.analysis_tab)
# or adding 3 seperate namespaces to main (gui.main.image_tab, etc.).
# MainWin would still need to exist for restart/closing/loadouts/other general stuff such as device/measurement toggling, somehow.
# It would also make sense to rename this module 'gui_implementation.py' or something.
class MainWin:
    """Doc."""

    ####################
    ## General
    ####################
    def __init__(self, gui, app):
        """Doc."""

        self._app = app
        self._gui = gui

    def close(self, event):
        """Doc."""

        self._app.exit_app(event)

    def restart(self) -> None:
        """Restart all devices (except camera) and the timeout loop."""

        pressed = Question(txt="Are you sure?", title="Restarting Application").display()
        if pressed == QMessageBox.Yes:
            self._app.loop.create_task(self._app.clean_up_app(restart=True))

    def save(self) -> None:
        """Doc."""

        file_path, _ = QFileDialog.getSaveFileName(
            self._gui,
            "Save Loadout",
            self._app.LOADOUT_DIR_PATH,
        )
        if file_path != "":
            helper.write_gui_to_file(self._gui, file_path)
            logging.debug(f"Loadout saved as: '{file_path}'")

    def load(self, file_path="") -> None:
        """Doc."""

        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self._gui,
                "Load Loadout",
                self._app.LOADOUT_DIR_PATH,
            )
        if file_path != "":
            helper.read_file_to_gui(file_path, self._gui)
            logging.debug(f"Loadout loaded: '{file_path}'")

    def dvc_toggle(
        self, nick, toggle_mthd="toggle", state_attr="state", leave_on=False, leave_off=False
    ) -> bool:
        """Returns False in case operation fails"""

        dvc = getattr(self._app.devices, nick)
        try:
            is_dvc_on = getattr(dvc, state_attr)
        except AttributeError:
            exc = DeviceError(f"{dvc.log_ref} was not properly initialized. Cannot Toggle.")
            if nick == "stage":
                err_hndlr(exc, sys._getframe(), locals(), lvl="warning")
                return False
            else:
                raise exc

        if (leave_on and is_dvc_on) or (leave_off and not is_dvc_on):
            return True

        if not is_dvc_on:
            # switch ON
            try:
                getattr(dvc, toggle_mthd)(True)
            except DeviceError as exc:
                err_hndlr(exc, sys._getframe(), locals(), lvl="warning")
                return False

            if (is_dvc_on := getattr(dvc, state_attr)) :
                # if managed to turn ON
                logging.debug(f"{dvc.log_ref} toggled ON")
                if nick == "stage":
                    self._gui.stageButtonsGroup.setEnabled(True)
                return True

        else:
            # switch OFF
            try:
                getattr(dvc, toggle_mthd)(False)
            except DeviceError:
                return False

            if not (is_dvc_on := getattr(dvc, state_attr)):
                # if managed to turn OFF
                logging.debug(f"{dvc.log_ref} toggled OFF")

                if nick == "stage":
                    self._gui.stageButtonsGroup.setEnabled(False)

                if nick == "dep_laser":
                    # set curr/pow values to zero when depletion is turned OFF
                    self._gui.depActualCurr.setValue(0)
                    self._gui.depActualPow.setValue(0)

                return True

    def led_clicked(self, led_obj_name) -> None:
        """Doc."""

        led_name_to_nick_dict = {
            "ledExc": "exc_laser",
            "ledTdc": "TDC",
            "ledDep": "dep_laser",
            "ledShutter": "dep_shutter",
            "ledStage": "stage",
            "ledUm232h": "UM232H",
            "ledCam": "camera",
            "ledScn": "scanners",
            "ledCounter": "photon_detector",
            "ledPxlClk": "pixel_clock",
        }

        dvc_nick = led_name_to_nick_dict[led_obj_name]
        error_dict = getattr(self._app.devices, dvc_nick).error_dict
        if error_dict is not None:
            Error(**error_dict, custom_title=dvcs.DEVICE_ATTR_DICT[dvc_nick].log_ref).display()

    def dep_sett_apply(self):
        """Doc."""

        with suppress(DeviceError):
            if self._gui.currMode.isChecked():  # current mode
                val = self._gui.depCurr.value()
                self._app.devices.dep_laser.set_current(val)
            else:  # power mode
                val = self._gui.depPow.value()
                self._app.devices.dep_laser.set_power(val)

    def move_scanners(self, axes_used: str = "XYZ", destination=None) -> None:
        """Doc."""

        scanners_dvc = self._app.devices.scanners

        if destination is None:
            # if no destination is specified, use the AO from the GUI
            destination = tuple(getattr(self._gui, f"{ax}AOV").value() for ax in "xyz")

        data = []
        type_str = ""
        for ax, vltg_V in zip("XYZ", destination):
            if ax in axes_used:
                data.append([vltg_V])
                type_str += ax

        try:
            scanners_dvc.start_write_task(data, type_str)
            scanners_dvc.toggle(True)  # restart cont. reading
        except DeviceError as exc:
            err_hndlr(exc, sys._getframe(), locals())

        logging.debug(
            f"{dvcs.DEVICE_ATTR_DICT['scanners'].log_ref} were moved to {str(destination)} V"
        )

    # TODO: implement a 'go_to_last_position' - same as origin, just need to save last location each time (if it's not the origin)
    def go_to_origin(self, which_axes: str) -> None:
        """Doc."""

        scanners_dvc = self._app.devices.scanners

        for axis, is_chosen, org_axis_vltg in zip(
            "xyz",
            scanners_dvc.AXES_TO_BOOL_TUPLE_DICT[which_axes],
            scanners_dvc.ORIGIN,
        ):
            if is_chosen:
                getattr(self._gui, f"{axis}AOV").setValue(org_axis_vltg)

        self.move_scanners(which_axes)

        logging.debug(f"{dvcs.DEVICE_ATTR_DICT['scanners'].log_ref} sent to {which_axes} origin")

    def displace_scanner_axis(self, sign: int) -> None:
        """Doc."""

        def limit(val: float, min: float, max: float) -> float:
            if min <= val <= max:
                return val
            elif val < max:
                return min
            else:
                return max

        um_disp = sign * self._gui.axisMoveUm.value()

        if um_disp != 0.0:
            scanners_dvc = self._app.devices.scanners
            axis = self._gui.axesGroup.checkedButton().text()
            current_vltg = scanners_dvc.ai_buffer[-1][3:][scanners_dvc.AXIS_INDEX[axis]]
            um_V_RATIO = dict(zip("XYZ", scanners_dvc.um_v_ratio))[axis]
            delta_vltg = um_disp / um_V_RATIO

            axis_ao_limits = getattr(scanners_dvc, f"{axis.upper()}_AO_LIMITS")
            new_vltg = limit(
                (current_vltg + delta_vltg),
                axis_ao_limits["min_val"],
                axis_ao_limits["max_val"],
            )

            getattr(self._gui, f"{axis.lower()}AOV").setValue(new_vltg)
            self.move_scanners(axis)

            logging.debug(
                f"{dvcs.DEVICE_ATTR_DICT['scanners'].log_ref}({axis}) was displaced {str(um_disp)} um"
            )

    def move_stage(self, dir: str, steps: int):
        """Doc."""

        self._app.loop.create_task(self._app.devices.stage.move(dir=dir, steps=steps))
        logging.info(
            f"{dvcs.DEVICE_ATTR_DICT['stage'].log_ref} moved {str(steps)} steps {str(dir)}"
        )

    def show_laser_dock(self):
        """Make the laser dock visible (convenience)."""

        if not self._gui.laserDock.isVisible():
            self._gui.laserDock.setVisible(True)
            self._gui.actionLaser_Control.setChecked(True)

    def show_stage_dock(self):
        """Make the laser dock visible (convenience)."""

        if not self._gui.stepperDock.isVisible():
            self._gui.stepperDock.setVisible(True)
            self._gui.actionStepper_Stage_Control.setChecked(True)

    async def toggle_meas(self, meas_type, laser_mode):
        """Doc."""

        if meas_type != (current_type := self._app.meas.type):

            if not self._app.meas.is_running:
                # no meas running
                if meas_type == "SFCSSolution":
                    pattern = self._gui.solScanType.currentText()
                    if pattern == "angular":
                        scan_params = wdgt_colls.sol_ang_scan_wdgts.read_namespace_from_gui(
                            self._app
                        )
                    elif pattern == "circle":
                        scan_params = wdgt_colls.sol_circ_scan_wdgts.read_namespace_from_gui(
                            self._app
                        )
                    elif pattern == "static":
                        scan_params = SimpleNamespace()

                    scan_params.pattern = pattern

                    self._app.meas = meas.SFCSSolutionMeasurement(
                        app=self._app,
                        scan_params=scan_params,
                        laser_mode=laser_mode.lower(),
                        **wdgt_colls.sol_meas_wdgts.read_dict_from_gui(self._app),
                    )

                    self._gui.startSolScanExc.setEnabled(False)
                    self._gui.startSolScanDep.setEnabled(False)
                    self._gui.startSolScanSted.setEnabled(False)
                    helper.deep_getattr(self._gui, f"startSolScan{laser_mode}.setEnabled")(True)
                    helper.deep_getattr(self._gui, f"startSolScan{laser_mode}.setText")(
                        "Stop \nScan"
                    )

                    self._gui.solScanMaxFileSize.setEnabled(False)
                    self._gui.solScanDur.setEnabled(self._gui.repeatSolMeas.isChecked())
                    self._gui.solScanDurUnits.setEnabled(False)
                    self._gui.solScanFileTemplate.setEnabled(False)

                elif meas_type == "SFCSImage":
                    self._app.meas = meas.SFCSImageMeasurement(
                        app=self._app,
                        scan_params=wdgt_colls.img_scan_wdgts.read_namespace_from_gui(self._app),
                        laser_mode=laser_mode.lower(),
                        **wdgt_colls.img_meas_wdgts.read_dict_from_gui(self._app),
                    )

                    self._gui.startImgScanExc.setEnabled(False)
                    self._gui.startImgScanDep.setEnabled(False)
                    self._gui.startImgScanSted.setEnabled(False)
                    helper.deep_getattr(self._gui, f"startImgScan{laser_mode}.setEnabled")(True)
                    helper.deep_getattr(self._gui, f"startImgScan{laser_mode}.setText")(
                        "Stop \nScan"
                    )

                self._app.loop.create_task(self._app.meas.run())
            else:
                # other meas running
                logging.warning(
                    f"Another type of measurement " f"({current_type}) is currently running."
                )

        else:  # current_type == meas_type
            # manual shutdown
            if meas_type == "SFCSSolution":
                self._gui.startSolScanExc.setEnabled(True)
                self._gui.startSolScanDep.setEnabled(True)
                self._gui.startSolScanSted.setEnabled(True)
                helper.deep_getattr(self._gui, f"startSolScan{laser_mode}.setText")(
                    f"{laser_mode} \nScan"
                )
                self._gui.imp.go_to_origin("XY")
                # TODO: add all of the following to a QButtonsGroup and en/disable them together (see gui.py)
                self._gui.solScanMaxFileSize.setEnabled(True)
                self._gui.solScanDur.setEnabled(True)
                self._gui.solScanDurUnits.setEnabled(True)
                self._gui.solScanFileTemplate.setEnabled(True)

            elif meas_type == "SFCSImage":
                self._gui.startImgScanExc.setEnabled(True)
                self._gui.startImgScanDep.setEnabled(True)
                self._gui.startImgScanSted.setEnabled(True)
                helper.deep_getattr(self._gui, f"startImgScan{laser_mode}.setText")(
                    f"{laser_mode} \nScan"
                )

            if self._app.meas.is_running:
                # manual stop
                await self._app.meas.stop()

    def disp_scn_pttrn(self, pattern: str):
        """Doc."""

        if pattern == "image":
            scan_params_coll = wdgt_colls.img_scan_wdgts
            plt_wdgt = self._gui.imgScanPattern
        elif pattern == "angular":
            scan_params_coll = wdgt_colls.sol_ang_scan_wdgts
            plt_wdgt = self._gui.solScanPattern
        elif pattern == "circle":
            scan_params_coll = wdgt_colls.sol_circ_scan_wdgts
            plt_wdgt = self._gui.solScanPattern
        elif pattern == "static":
            scan_params_coll = None
            plt_wdgt = self._gui.solScanPattern

        if scan_params_coll:
            scan_params = scan_params_coll.read_namespace_from_gui(self._app)

            with suppress(AttributeError, ZeroDivisionError):
                # AttributeError - devices not yet initialized
                # ZeroDivisionError - loadout has bad values
                um_v_ratio = self._app.devices.scanners.um_v_ratio
                ao, *_ = ScanPatternAO(pattern, scan_params, um_v_ratio).calculate_pattern()
                x_data, y_data = ao[0, :], ao[1, :]
                plt_wdgt.plot(x_data, y_data, clear=True)

        else:
            # no scan
            plt_wdgt.plot([], [], clear=True)

    def open_settwin(self):
        """Doc."""

        self._app.gui.settings.show()
        self._app.gui.settings.activateWindow()

    async def open_camwin(self):
        # TODO: simply making this func async doesn't help. the blocking function here is 'UC480_Camera(reopen_policy="new")'
        # from 'drivers.py', and I can't yet see a way to make it async (since I don't want to touch the API) I should try threading for this.
        """Doc."""

        self._gui.actionCamera_Control.setEnabled(False)
        self._app.gui.camera.show()
        self._app.gui.camera.activateWindow()
        self._app.gui.camera.imp.init_cam()

    def counts_avg_interval_changed(self, val: int) -> None:
        """Doc."""

        self._app.timeout_loop.updt_intrvl["cntr_avg"] = val / 1000  # convert to seconds

    ####################
    ## Image Tab
    ####################
    def roi_to_scan(self):
        """Doc"""

        plane_idx = self._gui.numPlaneShown.value()
        with suppress(AttributeError):
            line_ticks_v = self._app.last_img_scn.plane_images_data[plane_idx].line_ticks_v
            row_ticks_v = self._app.last_img_scn.plane_images_data[plane_idx].row_ticks_v
            plane_ticks = self._app.last_img_scn.set_pnts_planes

            coord_1, coord_2 = (
                round(self._gui.imgScanPlot.vLine.value()) - 1,
                round(self._gui.imgScanPlot.hLine.value()) - 1,
            )

            dim1_vltg = line_ticks_v[coord_1]
            dim2_vltg = row_ticks_v[coord_2]
            dim3_vltg = plane_ticks[plane_idx]

            plane_type = self._app.last_img_scn.plane_type

            if plane_type == "XY":
                vltgs = (dim1_vltg, dim2_vltg, dim3_vltg)
            elif plane_type == "XZ":
                vltgs = (dim1_vltg, dim3_vltg, dim2_vltg)
            elif plane_type == "YZ":
                vltgs = (dim3_vltg, dim1_vltg, dim2_vltg)

            [
                getattr(self._gui, f"{axis.lower()}AOV").setValue(vltg)
                for axis, vltg in zip("XYZ", vltgs)
            ]

            self.move_scanners(plane_type)

            logging.debug(f"{dvcs.DEVICE_ATTR_DICT['scanners'].log_ref} moved to ROI ({vltgs})")

    def build_image(self, img_data, method, plane_idx):
        """Doc."""

        if method == "Forward scan - actual counts per pixel":
            return img_data.image_forward[:, :, plane_idx]

        elif method == "Forward scan - points per pixel":
            return img_data.norm_forward[:, :, plane_idx]

        elif method == "Forward scan - normalized":
            return img_data.image_forward[:, :, plane_idx] / img_data.norm_forward[:, :, plane_idx]

        elif method == "Backwards scan - actual counts per pixel":
            return img_data.image_backward[:, :, plane_idx]

        elif method == "Backwards scan - points per pixel":
            return img_data.norm_backward[:, :, plane_idx]

        elif method == "Backwards scan - normalized":
            return (
                img_data.image_backward[:, :, plane_idx] / img_data.norm_backward[:, :, plane_idx]
            )

        elif method == "Both scans - interlaced":
            p1_norm = (
                img_data.image_forward[:, :, plane_idx] / img_data.norm_forward[:, :, plane_idx]
            )
            p2_norm = (
                img_data.image_backward[:, :, plane_idx] / img_data.norm_backward[:, :, plane_idx]
            )
            n_lines = p1_norm.shape[0] + p2_norm.shape[0]
            p = np.zeros(p1_norm.shape)
            p[:n_lines:2, :] = p1_norm[:n_lines:2, :]
            p[1:n_lines:2, :] = p2_norm[1:n_lines:2, :]
            return p

        elif method == "Both scans - averaged":
            p1 = img_data.image_forward[:, :, plane_idx]
            p2 = img_data.image_backward[:, :, plane_idx]
            norm1 = img_data.norm_forward[:, :, plane_idx]
            norm2 = img_data.norm_backward[:, :, plane_idx]
            return (p1 + p2) / (norm1 + norm2)

    def disp_plane_img(self, plane_idx, auto_cross=False):
        """Doc."""

        # TODO: replace this with Gaussian fit (2D)
        def auto_crosshair_position(image: np.ndarray) -> Tuple[float, float]:
            """
            Gets the current image, calculates its weighted mean
            location and returns the position to which to move the crosshair.
            (later perhaps using gaussian fit?)
            """

            # Use some good-old heuristic thresholding
            n, bin_edges = np.histogram(image.ravel())
            T = 1
            for i in range(1, len(n)):
                if n[i] <= n.max() * 0.1:
                    if n[i + 1] >= n[i] * 10:
                        continue
                    else:
                        T = i
                        break
                else:
                    continue
            thresh = (bin_edges[T] - bin_edges[T - 1]) / 2
            image[image < thresh] = 0

            # calculate the weighted mean
            x = 1 / image.sum() * np.dot(np.arange(image.shape[1]), image.sum(axis=0))
            y = 1 / image.sum() * np.dot(np.arange(image.shape[0]), image.sum(axis=1))
            return (x, y)

        disp_mthd = self._gui.imgShowMethod.currentText()
        with suppress(AttributeError):
            # No last_img_scn yet
            image_data = self._app.last_img_scn.plane_images_data
            image = self.build_image(image_data, disp_mthd, plane_idx)
            self._gui.imgScanPlot.replace_image(image.T)
            if auto_cross:
                self._gui.imgScanPlot.move_crosshair(auto_crosshair_position(image))

    def plane_choice_changed(self, plane_idx):
        """Doc."""

        with suppress(AttributeError):
            # no scan performed since app init
            self._app.meas.plane_shown.set(plane_idx)
            self.disp_plane_img(plane_idx)

    def fill_img_scan_preset_gui(self, curr_text: str) -> None:
        """Doc."""

        img_scn_wdgt_fillout_dict = {
            "Locate Plane - YZ Coarse": ["YZ", 15, 15, 10, 80, 1000, 20, 0.9, 1],
            "MFC - XY compartment": ["XY", 70, 70, 0, 80, 1000, 20, 0.9, 1],
            "GB -  XY Coarse": ["XY", 15, 15, 0, 80, 1000, 20, 0.9, 1],
            "GB - XY bead area": ["XY", 5, 5, 0, 80, 1000, 20, 0.9, 1],
            "GB - XY single bead": ["XY", 1, 1, 0, 80, 1000, 20, 0.9, 1],
            "GB - YZ single bead": ["YZ", 2.5, 2.5, 0, 80, 1000, 20, 0.9, 1],
        }

        wdgt_colls.img_scan_wdgts.write_to_gui(self._app, img_scn_wdgt_fillout_dict[curr_text])
        logging.debug(f"Image scan preset configuration chosen: '{curr_text}'")

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
            "Standard Alignment": {
                "scan_type": "static",
                "repeat": True,
            },
            "Standard Angular": {
                "scan_type": "angular",
                "repeat": False,
            },
            "Standard Circular": {
                "scan_type": "circle",
                "repeat": False,
            },
        }

        wdgt_colls.sol_meas_wdgts.write_to_gui(self._app, sol_meas_wdgt_fillout_dict[curr_text])
        logging.debug(f"Solution measurement preset configuration chosen: '{curr_text}'")

    ####################
    ## Analysis Tab
    ####################

    def populate_all_data_dates(self) -> None:
        """Doc."""

        # define widgets
        data_import_wdgts = wdgt_colls.data_import_wdgts
        is_image_type = data_import_wdgts.is_image_type.get()
        is_solution_type = data_import_wdgts.is_solution_type.get()
        years_combobox = data_import_wdgts.data_years.obj
        months_combobox = data_import_wdgts.data_months.obj
        days_combobox = data_import_wdgts.data_days.obj
        templates_combobox = data_import_wdgts.data_templates.obj

        if is_image_type:
            save_path = wdgt_colls.img_meas_wdgts.read_namespace_from_gui(self._app).save_path
            data_import_wdgts.import_stacked.set(0)
            sub_dir = "image"
        elif is_solution_type:
            save_path = wdgt_colls.sol_meas_wdgts.read_namespace_from_gui(self._app).save_path
            data_import_wdgts.import_stacked.set(1)
            sub_dir = "solution"

        years_combobox.clear()
        months_combobox.clear()
        days_combobox.clear()
        templates_combobox.clear()

        with suppress(TypeError, IndexError, FileNotFoundError):
            # no directories found... (dir_years is None or [])
            # FileNotFoundError - data directory deleted while app running!
            dir_years = helper.dir_date_parts(save_path, sub_dir)
            years_combobox.addItems(dir_years)

    def populate_data_dates_from_year(self, year: str) -> None:
        """Doc."""

        if not year:
            # ignore if combobox was just cleared
            return

        # define widgets
        data_import_wdgts = wdgt_colls.data_import_wdgts
        is_image_type = data_import_wdgts.is_image_type.get()
        is_solution_type = data_import_wdgts.is_solution_type.get()
        months_combobox = data_import_wdgts.data_months.obj

        if is_image_type:
            meas_sett = wdgt_colls.img_meas_wdgts.read_namespace_from_gui(self._app)
        elif is_solution_type:
            meas_sett = wdgt_colls.sol_meas_wdgts.read_namespace_from_gui(self._app)
        save_path = meas_sett.save_path
        sub_dir = meas_sett.sub_dir_name

        months_combobox.clear()

        with suppress(TypeError, IndexError):
            # no directories found... (dir_years is None or [])
            dir_months = helper.dir_date_parts(save_path, sub_dir, year=year)
            months_combobox.addItems(dir_months)

    def populate_data_dates_from_month(self, month: str) -> None:
        """Doc."""

        if not month:
            # ignore if combobox was just cleared
            return

        # define widgets
        data_import_wdgts = wdgt_colls.data_import_wdgts
        is_image_type = data_import_wdgts.is_image_type.get()
        is_solution_type = data_import_wdgts.is_solution_type.get()
        year = data_import_wdgts.data_years.get()
        days_combobox = data_import_wdgts.data_days.obj

        if is_image_type:
            meas_sett = wdgt_colls.img_meas_wdgts.read_namespace_from_gui(self._app)
        elif is_solution_type:
            meas_sett = wdgt_colls.sol_meas_wdgts.read_namespace_from_gui(self._app)
        save_path = meas_sett.save_path
        sub_dir = meas_sett.sub_dir_name

        days_combobox.clear()

        with suppress(TypeError, IndexError):
            # no directories found... (dir_years is None or [])
            dir_days = helper.dir_date_parts(save_path, sub_dir, year=year, month=month)
            days_combobox.addItems(dir_days)

    def pkl_and_mat_templates(self, dir_path: str) -> Iterable:
        """Doc."""

        is_solution_type = "solution" in dir_path

        pkl_template_set = {
            (re.sub("[0-9]+.pkl", "*.pkl", item) if is_solution_type else item)
            for item in os.listdir(dir_path)
            if item.endswith(".pkl")
        }
        mat_template_set = {
            (re.sub("[0-9]+.mat", "*.mat", item) if is_solution_type else item)
            for item in os.listdir(dir_path)
            if item.endswith(".mat")
        }
        return sorted(pkl_template_set.union(mat_template_set))

    def populate_data_templates_from_day(self, day: str) -> None:
        """Doc."""

        if not day:
            # ignore if combobox was just cleared
            return

        # define widgets
        data_import_wdgts = wdgt_colls.data_import_wdgts
        templates_combobox = data_import_wdgts.data_templates.obj

        templates_combobox.clear()

        with suppress(TypeError, IndexError):
            # no directories found... (dir_years is None or [])
            templates = self.pkl_and_mat_templates(self.current_date_type_dir_path())
            templates_combobox.addItems(templates)

    def cycle_through_data_templates(self, dir: str) -> None:
        """Cycle through the daily data templates in order (next or previous)"""

        data_templates_combobox = wdgt_colls.data_import_wdgts.data_templates.obj
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

    def current_date_type_dir_path(self) -> str:
        """Returns path to directory of currently selected date and measurement type"""

        import_wdgts = wdgt_colls.data_import_wdgts

        is_image_type = import_wdgts.is_image_type.get()
        is_solution_type = import_wdgts.is_solution_type.get()
        day = import_wdgts.data_days.get()
        year = import_wdgts.data_years.get()
        month = import_wdgts.data_months.get()

        if is_image_type:
            meas_settings = wdgt_colls.img_meas_wdgts.read_namespace_from_gui(self._app)
        elif is_solution_type:
            meas_settings = wdgt_colls.sol_meas_wdgts.read_namespace_from_gui(self._app)
        save_path = meas_settings.save_path
        sub_dir = meas_settings.sub_dir_name
        return os.path.join(save_path, f"{day.rjust(2, '0')}_{month.rjust(2, '0')}_{year}", sub_dir)

    def open_data_dir(self) -> None:
        """Doc."""

        if os.path.isdir(dir_path := self.current_date_type_dir_path()):
            webbrowser.open(dir_path)
        else:
            # dir was deleted, refresh all dates
            self.populate_all_data_dates()

    def update_dir_log_file(self) -> None:
        """Doc."""

        data_import_wdgts = wdgt_colls.data_import_wdgts
        text_lines = data_import_wdgts.log_text.get().split("\n")
        curr_template = data_import_wdgts.data_templates.get()
        log_filename = re.sub("_\\*.\\w{3}", ".log", curr_template)
        with suppress(AttributeError, TypeError):
            # no directories found
            file_path = os.path.join(self.current_date_type_dir_path(), log_filename)
            helper.write_list_to_file(file_path, text_lines)
            self.update_dir_log_wdgt(curr_template)

    def get_daily_alignment(self):
        """Doc."""

        date_dir_path = re.sub("(solution|image)", "", self.current_date_type_dir_path())

        try:
            # TODO: make this work for multiple templates (exc and sted), add both to log file and also the ratios
            template = self.pkl_and_mat_templates(date_dir_path)[0]
        except IndexError:
            # no alignment file found
            g0, tau = (None, None)
        else:
            # TODO: loading properly (err hndling) should be a seperate function
            full_data = CorrFuncTDC()
            try:
                full_data.read_fpga_data(
                    os.path.join(date_dir_path, template),
                    no_plot=True,
                )
                full_data.correlate_regular_data()

            except AttributeError:
                # No directories found
                g0, tau = (None, None)

            except (NotImplementedError, RuntimeError, ValueError, FileNotFoundError) as exc:
                err_hndlr(exc, sys._getframe(), locals())

            full_data.average_correlation()
            try:
                full_data.fit_correlation_function()
            except fit_tools.FitError as exc:
                err_hndlr(exc, sys._getframe(), locals())
                g0, tau = (None, None)
            else:
                fit_params = full_data.fit_param["diffusion_3d_fit"]
                g0, tau, _ = fit_params["beta"]

        return g0, tau

    def update_dir_log_wdgt(self, template: str) -> None:
        """Doc."""

        def initialize_dir_log_file(file_path, g0, tau) -> None:
            """Doc."""

            basic_header = []
            basic_header.append("-" * 40)
            basic_header.append("Measurement Log File")
            basic_header.append("-" * 40)
            basic_header.append("Excitation Power: ")
            basic_header.append("Depletion Power: ")
            if g0 is not None:
                basic_header.append(
                    f"Free Atto FCS @ 12 uW: G0 = {g0/1e3:.2f} k/ tau = {tau*1e3:.2f} us"
                )
            else:
                basic_header.append("Free Atto FCS @ 12 uW: Not Available")
            basic_header.append("-" * 40)
            basic_header.append("EDIT_HERE")

            helper.write_list_to_file(file_path, basic_header)

        if not template:
            return

        data_import_wdgts = wdgt_colls.data_import_wdgts
        data_import_wdgts.log_text.set("")  # clear first

        # get the log file path
        if data_import_wdgts.is_solution_type.get():
            log_filename = re.sub("_?\\*\\.\\w{3}", ".log", template)
        elif data_import_wdgts.is_image_type.get():
            log_filename = re.sub("\\.\\w{3}", ".log", template)
        file_path = os.path.join(self.current_date_type_dir_path(), log_filename)

        try:  # load the log file in path
            text_lines = helper.read_file_to_list(file_path)
        except FileNotFoundError:  # initialize a new log file if no existing file
            with suppress(OSError, IndexError):
                # OSError - missing file/folder (deleted during operation)
                # IndexError - alignment file does not exist
                initialize_dir_log_file(file_path, *self.get_daily_alignment())
                text_lines = helper.read_file_to_list(file_path)
        finally:  # write file to widget
            data_import_wdgts.log_text.set("\n".join(text_lines))

    def preview_img_scan(self, template: str) -> None:
        """Doc."""

        if not template:
            return

        wdgts = wdgt_colls.data_import_wdgts

        if wdgts.is_image_type.get():
            # import the data
            file_path = os.path.join(self.current_date_type_dir_path(), template)
            file_dict = file_utilities.load_file_dict(file_path)
            # get the center plane image
            counts = file_dict["ci"]
            ao = file_dict["ao"]
            scan_param = file_dict["scan_param"]
            um_v_ratio = file_dict["xyz_um_to_v"]
            p = PhotonData()
            p.convert_counts_to_images(counts, ao, scan_param, um_v_ratio)
            image = self.build_image(
                p.image_data, "Forward scan - actual counts per pixel", scan_param["n_planes"] // 2
            )
            # plot it (below)
            wdgts.img_preview_disp.obj.display_image(image, axes="off")

        pass

    def import_sol_data(self) -> None:
        """Doc."""

        data_import_wdgts = wdgt_colls.data_import_wdgts
        current_template = data_import_wdgts.data_templates.get()
        is_calibration = data_import_wdgts.is_calibration.get()

        # TODO: create a context manager for pausing/resuming AI/CI tasks
        logging.debug("Importing Data. Pausing 'ai' and 'ci' tasks")
        self._app.devices.scanners.pause_tasks("ai")
        self._app.devices.photon_detector.pause_tasks("ci")

        full_data = CorrFuncTDC()
        fix_shift = wdgt_colls.sol_data_analysis_wdgts.fix_shift.get()
        # TODO: add support for choosing a single file (already in GUI)

        try:
            with suppress(AttributeError):
                # No directories found
                full_data.read_fpga_data(
                    os.path.join(self.current_date_type_dir_path(), current_template),
                    fix_shift=fix_shift,
                    no_plot=True,
                )
                if full_data.type == "angular_scan":
                    full_data.correlate_angular_scan_data()
                elif full_data.type == "static":
                    full_data.correlate_regular_data()

        except (NotImplementedError, RuntimeError, ValueError, FileNotFoundError) as exc:
            err_hndlr(exc, sys._getframe(), locals())

        else:
            # save data and populate combobox
            imported_combobox = wdgt_colls.sol_data_analysis_wdgts.imported_templates
            if is_calibration:
                if "_exc_" in current_template:
                    item = "CAL_EXC - " + current_template
                    self._app.analysis.loaded_data["CAL_EXC"] = full_data
                elif "_sted_" in current_template:
                    item = "CAL_STED - " + current_template
                    self._app.analysis.loaded_data["CAL_STED"] = full_data
                else:
                    item = "CAL_UNKNOWN - " + current_template
                    self._app.analysis.loaded_data["CAL_UNKNOWN"] = full_data
            else:
                if "_exc_" in current_template:
                    item = "SAMP_EXC - " + current_template
                    self._app.analysis.loaded_data["SAMP_EXC"] = full_data
                elif "_sted_" in current_template:
                    item = "SAMP_STED - " + current_template
                    self._app.analysis.loaded_data["SAMP_STED"] = full_data
                else:
                    item = "SAMP_UNKNOWN - " + current_template
                    self._app.analysis.loaded_data["SAMP_UNKNOWN"] = full_data
            imported_combobox.obj.addItem(item)
            imported_combobox.set(item)

            logging.info(f"Data loaded for analysis: {item}")

        with suppress(DeviceError):
            # devices not initialized
            self._app.devices.scanners.init_ai_buffer()
            self._app.devices.scanners.start_tasks("ai")
            self._app.devices.photon_detector.init_ci_buffer()
            self._app.devices.photon_detector.start_tasks("ci")
            logging.debug("Data import finished. Resuming 'ai' and 'ci' tasks")

    def get_current_full_data(self, imported_template: str = None) -> CorrFuncTDC:
        """Doc."""

        if imported_template is None:
            imported_template = wdgt_colls.sol_data_analysis_wdgts.imported_templates.get()
        curr_data_type, *_ = re.split(" -", imported_template)
        return self._app.analysis.loaded_data.get(curr_data_type)

    def populate_sol_meas_analysis(self, imported_template):
        """Doc."""

        wdgts = wdgt_colls.sol_data_analysis_wdgts

        try:
            full_data = self.get_current_full_data(imported_template)
            num_files = len(full_data.data)
        except AttributeError:
            # no imported templates (deleted)
            wdgts.scan_image_disp.obj.clear()
            wdgts.row_acf_disp.obj.clear()
            wdgts.scan_img_file_num.obj.setRange(1, 1)
            wdgts.scan_img_file_num.set(1)
        else:
            print("Populating analysis GUI...", end=" ")

            # populate general measurement properties
            wdgts.n_files.set(num_files)
            wdgts.scan_duration_min.set(full_data.duration_min)

            if full_data.type == "angular_scan":
                # populate scan images tab
                print("Displaying scan images...", end=" ")
                wdgts.scan_img_file_num.obj.setRange(1, num_files)
                wdgts.scan_img_file_num.set(1)
                self.display_scan_image(1, imported_template)

                # calculate average and display
                # TODO: perhaps should average by default, before displaying that way I can put it all in one function?
                print("Averaging and plotting...", end=" ")
                self.calculate_and_show_sol_mean_acf(imported_template)

                scan_settings_text = "\n\n".join(
                    [
                        f"{key}: {(val[:5] if isinstance(val, Iterable) else val)}"
                        for key, val in full_data.angular_scan_settings.items()
                    ]
                )

            elif full_data.type == "static":
                print("Averaging, plotting and fitting...", end=" ")
                self.calculate_and_show_sol_mean_acf(imported_template)
                scan_settings_text = "no scan."

            wdgts.scan_settings.set(scan_settings_text)

            print("Done.")

    def display_scan_image(self, file_num, imported_template: str = None):
        """Doc."""

        with suppress(IndexError, KeyError, AttributeError):
            # IndexError - data import failed
            # KeyError, AttributeError - data deleted
            full_data = self.get_current_full_data(imported_template)
            img = full_data.data[file_num - 1].image
            roi = full_data.data[file_num - 1].roi

            scan_image_disp = wdgt_colls.sol_data_analysis_wdgts.scan_image_disp.obj
            scan_image_disp.display_image(img)
            scan_image_disp.plot(roi["col"], roi["row"], color="white")
            scan_image_disp.entitle_and_label("Pixel Number", "Line Number")

    def calculate_and_show_sol_mean_acf(self, imported_template: str = None) -> None:
        """Doc."""

        wdgts = wdgt_colls.sol_data_analysis_wdgts
        full_data = self.get_current_full_data(imported_template)

        if full_data.type == "angular_scan":
            row_disc_method = wdgts.row_dicrimination.get().objectName()
            if row_disc_method == "solAnalysisRemoveOver":
                avg_corr_args = dict(rejection=wdgts.remove_over.get())
            elif row_disc_method == "solAnalysisRemoveWorst":
                avg_corr_args = dict(rejection=None, reject_n_worst=wdgts.remove_worst.get())
            else:  # use all rows
                avg_corr_args = dict(rejection=None)

            with suppress(AttributeError):
                # no data loaded
                full_data.average_correlation(**avg_corr_args)

                wdgts.row_acf_disp.obj.plot_acfs(
                    full_data.lag,
                    full_data.average_cf_cr,
                    full_data.g0,
                    full_data.cf_cr[full_data.j_good, :],
                )
                wdgts.row_acf_disp.obj.entitle_and_label("lag (ms)", "?")

                wdgts.mean_g0.set(full_data.g0 / 1e3)  # shown in thousands
                wdgts.mean_tau.set(0)

                wdgts.n_good_rows.set(n_good := len(full_data.j_good))
                wdgts.n_bad_rows.set(n_bad := len(full_data.j_bad))
                wdgts.remove_worst.obj.setMaximum(n_good + n_bad - 2)

        if full_data.type == "static":
            full_data.average_correlation()
            try:
                full_data.fit_correlation_function()
            except fit_tools.FitError as exc:
                # fit failed, use g0 calculated in 'average_correlation()'
                err_hndlr(exc, sys._getframe(), locals(), lvl="warning")
                wdgts.mean_g0.set(full_data.g0 / 1e3)  # shown in thousands
                wdgts.mean_tau.set(0)
            else:  # fit succeeded
                fit_params = full_data.fit_param["diffusion_3d_fit"]
                g0, tau, _ = fit_params["beta"]
                fit_func = getattr(fit_tools, fit_params["fit_func"])
                wdgts.mean_g0.set(g0 / 1e3)  # shown in thousands
                wdgts.mean_tau.set(tau * 1e3)
                y_fit = fit_func(fit_params["x"], *fit_params["beta"])
                wdgts.row_acf_disp.obj.clear()
                wdgts.row_acf_disp.obj.plot_acfs(
                    full_data.lag,
                    full_data.average_cf_cr,
                    full_data.g0,
                )
                wdgts.row_acf_disp.obj.plot(fit_params["x"], y_fit, color="red")

    def remove_imported_template(self):
        """Doc."""

        imported_templates = wdgt_colls.sol_data_analysis_wdgts.imported_templates
        template = imported_templates.get()
        data_type, *_ = re.split(" -", template)
        self._app.analysis.loaded_data[data_type] = None
        imported_templates.obj.removeItem(imported_templates.obj.currentIndex())

    def convert_files_to_matlab_format(self) -> None:
        """
        Convert files of current template to '.mat',
        after translating their dictionaries to the legacy matlab format.
        """

        data_import_wdgts = wdgt_colls.data_import_wdgts
        current_template = data_import_wdgts.data_templates.get()
        current_dir_path = self.current_date_type_dir_path()

        file_template_path = os.path.join(current_dir_path, current_template)
        file_paths = helper.sort_file_paths_by_file_number(glob.glob(file_template_path))

        print(f"Converting {len(file_paths)} files to '.mat' in legacy MATLAB format...", end=" ")

        for idx, file_path in enumerate(file_paths):
            file_dict = file_utilities.load_file_dict(file_path)
            mat_file_path = re.sub(".pkl", ".mat", file_path)
            if "solution" in mat_file_path:
                mat_file_path = re.sub("solution", r"solution\\matlab", mat_file_path)
            if "image" in mat_file_path:
                mat_file_path = re.sub("image", r"image\\matlab", mat_file_path)
            os.makedirs(os.path.join(current_dir_path, "matlab"), exist_ok=True)
            file_utilities.save_mat(file_dict, mat_file_path)
            print(f"({idx+1})", end=" ")

        print("Done.")
        logging.info(f"{current_template} converted to MATLAB format")


class SettWin:
    """Doc."""

    def __init__(self, gui, app):
        """Doc."""

        self._app = app
        self._gui = gui
        self.check_on_close = True

    def clean_up(self):
        """Doc."""

        if self.check_on_close is True:
            curr_file_path = self._gui.settingsFileName.text()

            current_state = set(helper.wdgt_items_to_text_lines(self._gui))
            last_loaded_state = set(helper.read_file_to_list(curr_file_path))

            if not len(current_state) == len(last_loaded_state):
                err_hndlr(
                    RuntimeError(
                        "Something was changed in the GUI. This probably means that the default settings need to be overwritten"
                    ),
                    sys._getframe(),
                    locals(),
                )

            if current_state != last_loaded_state:
                pressed = Question(
                    "Keep changes if made? " "(otherwise, revert to last loaded settings file.)"
                ).display()
                if pressed == QMessageBox.No:
                    self.load(self._gui.settingsFileName.text())

        else:
            Notification("Using Current settings.").display()

    def save(self):
        """
        Write all QLineEdit, QspinBox and QdoubleSpinBox
        of settings window to 'file_path'.
        Show 'file_path' in 'settingsFileName' QLineEdit.
        """

        file_path, _ = QFileDialog.getSaveFileName(
            self._gui,
            "Save Settings",
            self._app.SETTINGS_DIR_PATH,
        )
        if file_path != "":
            self._gui.frame.findChild(QWidget, "settingsFileName").setText(file_path)
            helper.write_gui_to_file(self._gui.frame, file_path)
            logging.debug(f"Settings file saved as: '{file_path}'")

    def load(self, file_path=""):
        """
        Read 'file_path' and write to matching widgets of settings window.
        Show 'file_path' in 'settingsFileName' QLineEdit.
        Default settings is chosen according to './settings/default_settings_choice'.
        """

        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self._gui,
                "Load Settings",
                self._app.SETTINGS_DIR_PATH,
            )
        if file_path != "":
            self._gui.frame.findChild(QWidget, "settingsFileName").setText(file_path)
            helper.read_file_to_gui(file_path, self._gui.frame)
            logging.debug(f"Settings file loaded: '{file_path}'")

    def confirm(self):
        """Doc."""

        self.check_on_close = False


class CamWin:
    """Doc."""

    # TODO:  Try to have the whole camera window operate in a thread

    def __init__(self, gui, app):
        """Doc."""

        self._app = app
        self._gui = gui
        self._cam = None

    def init_cam(self):
        """Doc."""

        self._cam = self._app.devices.camera
        self._app.gui.main.imp.dvc_toggle("camera")
        #        self._cam.video_timer.timeout.connect(self._video_timeout)
        logging.debug("Camera connection opened")

    def clean_up(self):
        """clean up before closing window"""

        if self._cam is not None:
            self.toggle_video()
            self._app.gui.main.imp.dvc_toggle("camera")
            self._app.gui.main.actionCamera_Control.setEnabled(True)
            self._cam = None
            logging.debug("Camera connection closed")

    def toggle_video(self):
        """Doc."""

        try:
            if self._cam.vid_state is False:  # turn On
                self._gui.videoButton.setStyleSheet(
                    "background-color: " "rgb(225, 245, 225); " "color: black;"
                )
                self._gui.videoButton.setText("Video ON")

                self._cam.toggle_video(True)

                logging.debug("Camera video mode ON")

            elif self._cam is not None:  # turn Off
                self._gui.videoButton.setStyleSheet(
                    "background-color: " "rgb(225, 225, 225); " "color: black;"
                )
                self._gui.videoButton.setText("Start Video")

                self._cam.toggle_video(False)

                logging.debug("Camera video mode OFF")

        except DeviceError as exc:
            err_hndlr(exc, sys._getframe(), locals())

    def shoot(self):
        """Doc."""

        self._app.loop.create_task(self._cam.shoot())
        logging.info("Camera photo taken")
