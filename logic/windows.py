""" GUI windows implementations module. """

import asyncio
import glob
import logging
import os
import re
import sys
import webbrowser
from collections.abc import Iterable
from contextlib import suppress
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget

import logic.devices as dvcs
import logic.measurements as meas
from data_analysis.correlation_function import CorrFuncTDC
from data_analysis.image import ImageScanData
from gui import widgets as wdgts
from logic.scan_patterns import ScanPatternAO
from utilities import display, file_utilities, fit_tools, helper
from utilities.dialog import ErrorDialog, NotificationDialog, QuestionDialog
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

        pressed = QuestionDialog(txt="Are you sure?", title="Restarting Application").display()
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
            wdgts.write_gui_to_file(self._gui, wdgts.MAIN_TYPES, file_path)
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
            wdgts.read_file_to_gui(file_path, self._gui)
            logging.debug(f"Loadout loaded: '{file_path}'")

    def dvc_toggle(
        self, nick, toggle_mthd="toggle", state_attr="state", leave_on=False, leave_off=False
    ) -> bool:
        """Returns False in case operation fails"""

        dvc = getattr(self._app.devices, nick)
        try:
            is_dvc_on = getattr(dvc, state_attr)
        except AttributeError:
            exc = DeviceError(f"{dvc.log_ref} was not properly initialized. Cannot toggle.")
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
            ErrorDialog(
                **error_dict, custom_title=dvcs.DEVICE_ATTR_DICT[dvc_nick].log_ref
            ).display()

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
                        scan_params = wdgts.SOL_ANG_SCAN_COLL.read_gui(self._app)
                    elif pattern == "circle":
                        scan_params = wdgts.SOL_CIRC_SCAN_COLL.read_gui(self._app)
                    elif pattern == "static":
                        scan_params = SimpleNamespace()

                    scan_params.pattern = pattern

                    self._app.meas = meas.SFCSSolutionMeasurement(
                        app=self._app,
                        scan_params=scan_params,
                        laser_mode=laser_mode.lower(),
                        **wdgts.SOL_MEAS_COLL.read_gui(self._app, "dict"),
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
                        scan_params=wdgts.IMG_SCAN_COLL.read_gui(self._app),
                        laser_mode=laser_mode.lower(),
                        **wdgts.IMG_MEAS_COLL.read_gui(self._app, "dict"),
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
                self._gui.impl.go_to_origin("XY")
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
            scan_params_coll = wdgts.IMG_SCAN_COLL
            plt_wdgt = self._gui.imgScanPattern
        elif pattern == "angular":
            scan_params_coll = wdgts.SOL_ANG_SCAN_COLL
            plt_wdgt = self._gui.solScanPattern
        elif pattern == "circle":
            scan_params_coll = wdgts.SOL_CIRC_SCAN_COLL
            plt_wdgt = self._gui.solScanPattern
        elif pattern == "static":
            scan_params_coll = None
            plt_wdgt = self._gui.solScanPattern

        if scan_params_coll:
            scan_params = scan_params_coll.read_gui(self._app)

            with suppress(AttributeError, ZeroDivisionError, ValueError):
                # AttributeError - devices not yet initialized
                # ZeroDivisionError - loadout has bad values
                um_v_ratio = self._app.devices.scanners.um_v_ratio
                ao, *_ = ScanPatternAO(pattern, scan_params, um_v_ratio).calculate_pattern()
                x_data, y_data = ao[0, :], ao[1, :]
                plt_wdgt.display_pattern(x_data, y_data)

        else:
            # no scan
            plt_wdgt.display_pattern([], [])

    def open_settwin(self):
        """Doc."""

        self._app.gui.settings.show()
        self._app.gui.settings.activateWindow()

    async def open_camwin(self):
        """Doc."""

        self._gui.actionCamera_Control.setEnabled(False)
        self._app.gui.camera.show()
        self._app.gui.camera.activateWindow()
        self._app.gui.camera.impl.init_cam()

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
            line_ticks_v = self._app.last_img_scn.plane_images_data.line_ticks_v
            row_ticks_v = self._app.last_img_scn.plane_images_data.row_ticks_v
            plane_ticks = self._app.last_img_scn.set_pnts_planes

            coord_1, coord_2 = (round(pos_i) for pos_i in self._gui.imgScanPlot.ax.cursor.pos)

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

    def auto_scale_image(self, percent_factor: int):
        """Doc."""

        with suppress(AttributeError):
            # AttributeError - no last image yet
            image = self._app.last_img_scn.last_img
            try:
                image = display.auto_brightness_and_contrast(image, percent_factor)
            except (ZeroDivisionError, IndexError) as exc:
                err_hndlr(exc, sys._getframe(), locals(), lvl="warning")

            self._gui.imgScanPlot.display_image(image, cursor=True, cmap="bone")

    def disp_plane_img(self, plane_idx, auto_cross=False):
        """Doc."""

        method_dict = {
            "Forward scan - actual counts per pixel": "forward",
            "Forward scan - points per pixel": "forward normalization",
            "Forward scan - normalized": "forward normalized",
            "Backwards scan - actual counts per pixel": "backward",
            "Backwards scan - points per pixel": "backward normalization",
            "Backwards scan - normalized": "backward normalized",
            "Both scans - interlaced": "interlaced",
            "Both scans - averaged": "averaged",
        }

        def auto_crosshair_position(image: np.ndarray) -> Tuple[float, float]:
            """
            Beam co-alignment aid. Attempts to fit a 2D Gaussian to an image and returns the fitted center.
            If the fit fails, or if the fitted Gaussian is centered outside the image limits, it instead
            returns the center of mass.
            """

            #            # Use some good-old heuristic thresholding
            #            n, bin_edges = np.histogram(image.ravel())
            #            T = 1
            #            for i in range(1, len(n)):
            #                if n[i] <= n.max() * 0.1:
            #                    if n[i + 1] >= n[i] * 10:
            #                        continue
            #                    else:
            #                        T = i
            #                        break
            #                else:
            #                    continue
            #            thresh = (bin_edges[T] - bin_edges[T - 1]) / 2
            #            image[image < thresh] = 0

            try:
                fit_param = fit_tools.fit_2d_gaussian(image)

            except fit_tools.FitError:
                #                print("Gaussian fit failed, using COM") # TESTESTEST
                return helper.center_of_mass(image)

            else:
                _, x0, y0, sigma_x, sigma_y, *_ = fit_param["beta"]
                height, width = image.shape
                if (
                    (0 < x0 < width)
                    and (0 < y0 < height)
                    and (sigma_x < width)
                    and (sigma_y < height)
                ):
                    #                    print(f"x0 ({x0:.1f}) and y0 ({y0:.1f}) are within image boundaries. Good!") # TESTESTEST
                    return (x0, y0)
                else:
                    #                    print(f"x0 ({x0:.1f}) and y0 ({y0:.1f}) are OUTSIDE image boundaries. using COM") # TESTESTEST
                    return helper.center_of_mass(image)

        img_meas_wdgts = wdgts.IMG_MEAS_COLL.read_gui(self._app)
        disp_mthd = img_meas_wdgts.image_method
        with suppress(AttributeError):
            # No last_img_scn yet
            image_data = self._app.last_img_scn.plane_images_data
            image = image_data.build_image(method_dict[disp_mthd], plane_idx)
            self._app.last_img_scn.last_img = image.T
            img_meas_wdgts.image_wdgt.obj.display_image(image.T, cursor=True, cmap="bone")
            if auto_cross:
                img_meas_wdgts.image_wdgt.obj.ax.cursor.move_to_pos(auto_crosshair_position(image))

    def plane_choice_changed(self, plane_idx):
        """Doc."""

        with suppress(AttributeError):
            # AttributeError - no scan performed since app init
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

        wdgts.IMG_SCAN_COLL.write_to_gui(self._app, img_scn_wdgt_fillout_dict[curr_text])
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
                "regular": True,
            },
            "Standard Circular": {
                "scan_type": "circle",
                "regular": True,
            },
        }

        wdgts.SOL_MEAS_COLL.write_to_gui(self._app, sol_meas_wdgt_fillout_dict[curr_text])
        logging.debug(f"Solution measurement preset configuration chosen: '{curr_text}'")

    ####################
    ## Analysis Tab
    ####################

    def populate_all_data_dates(self) -> None:
        """Doc."""

        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL.read_gui(self._app)

        if DATA_IMPORT_COLL.is_image_type:
            save_path = wdgts.IMG_MEAS_COLL.read_gui(self._app).save_path
            DATA_IMPORT_COLL.import_stacked.set(0)
            sub_dir = "image"
        elif DATA_IMPORT_COLL.is_solution_type:
            save_path = wdgts.SOL_MEAS_COLL.read_gui(self._app).save_path
            DATA_IMPORT_COLL.import_stacked.set(1)
            sub_dir = "solution"

        wdgts.DATA_IMPORT_COLL.clear_all_objects()

        with suppress(TypeError, IndexError, FileNotFoundError):
            # no directories found... (dir_years is None or [])
            # FileNotFoundError - data directory deleted while app running!
            dir_years = helper.dir_date_parts(save_path, sub_dir)
            DATA_IMPORT_COLL.data_years.obj.addItems(dir_years)

    def populate_data_dates_from_year(self, year: str) -> None:
        """Doc."""

        if not year:
            # ignore if combobox was just cleared
            return

        # define widgets
        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL.read_gui(self._app)

        if DATA_IMPORT_COLL.is_image_type:
            meas_sett = wdgts.IMG_MEAS_COLL.read_gui(self._app)
        elif DATA_IMPORT_COLL.is_solution_type:
            meas_sett = wdgts.SOL_MEAS_COLL.read_gui(self._app)
        save_path = meas_sett.save_path
        sub_dir = meas_sett.sub_dir_name

        DATA_IMPORT_COLL.data_months.obj.clear()

        with suppress(TypeError, IndexError):
            # no directories found... (dir_years is None or [])
            dir_months = helper.dir_date_parts(save_path, sub_dir, year=year)
            DATA_IMPORT_COLL.data_months.obj.addItems(dir_months)

    def populate_data_dates_from_month(self, month: str) -> None:
        """Doc."""

        if not month:
            # ignore if combobox was just cleared
            return

        # define widgets
        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL.read_gui(self._app)
        year = DATA_IMPORT_COLL.data_years.get()
        days_combobox = DATA_IMPORT_COLL.data_days.obj

        if DATA_IMPORT_COLL.is_image_type:
            meas_sett = wdgts.IMG_MEAS_COLL.read_gui(self._app)
        elif DATA_IMPORT_COLL.is_solution_type:
            meas_sett = wdgts.SOL_MEAS_COLL.read_gui(self._app)
        save_path = meas_sett.save_path
        sub_dir = meas_sett.sub_dir_name

        days_combobox.clear()

        with suppress(TypeError, IndexError):
            # no directories found... (dir_years is None or [])
            dir_days = helper.dir_date_parts(save_path, sub_dir, year=year, month=month)
            days_combobox.addItems(dir_days)

    def pkl_and_mat_templates(self, dir_path: str) -> List[str]:
        """
        Accepts a directory path and returns a list of file templates
        ending in the form '*.pkl' or '*.mat' where * is any number.
        The templates are sorted by their time stamp of the form "HHMMSS".
        In case the folder contains legacy templates, they are sorted without a key.
        """

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
        try:
            sorted_templates = sorted(
                pkl_template_set.union(mat_template_set),
                key=lambda tmpl: int(re.split("(\\d{6})_*", tmpl)[1]),
            )
        except IndexError:
            # Dealing with legacy templates which do not contain the appropriate time format
            sorted_templates = sorted(pkl_template_set.union(mat_template_set))
            logging.warning(
                "Folder contains legacy file templates! Templates may not be properly sorted."
            )
        finally:
            return sorted_templates

    def populate_data_templates_from_day(self, day: str) -> None:
        """Doc."""

        if not day:
            # ignore if combobox was just cleared
            return

        # define widgets
        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL
        templates_combobox = DATA_IMPORT_COLL.data_templates.obj

        templates_combobox.clear()

        with suppress(TypeError, IndexError):
            # no directories found... (dir_years is None or [])
            templates = self.pkl_and_mat_templates(self.current_date_type_dir_path())
            templates_combobox.addItems(templates)

    def show_num_files(self, template) -> None:
        """Doc."""

        n_files_wdgt = wdgts.DATA_IMPORT_COLL.n_files
        dir_path = self.current_date_type_dir_path()
        n_files = len(glob.glob(os.path.join(dir_path, template)))
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

    def rename_template(self) -> None:
        """Doc."""

        dir_path = self.current_date_type_dir_path()
        data_import_wdgts = wdgts.DATA_IMPORT_COLL.read_gui(self._app)
        curr_template = data_import_wdgts.data_templates.get()
        new_template_prefix = data_import_wdgts.new_template

        if not new_template_prefix:
            return

        try:
            curr_template_prefix = re.findall("(^.*)(?=_[0-9]{6})", curr_template)[0]
        except IndexError:
            # template does not contain timestamp
            curr_template_prefix = re.findall("(^.*)(?=_\\*\\.[a-z]{3})", curr_template)[0]

        if new_template_prefix == curr_template_prefix:
            logging.warning(
                f"rename_template(): Requested template and current template are identical ('{curr_template}'). Operation canceled."
            )
            return

        new_template = re.sub(curr_template_prefix, new_template_prefix, curr_template, count=1)

        # check if new template already exists (unlikely)
        if glob.glob(os.path.join(dir_path, new_template)):
            logging.warning(
                f"New template '{new_template}' already exists in '{dir_path}'. Operation canceled."
            )
            return
        # check if current template doesn't exists (can only happen if deleted manually between discovering the template and running this function)
        if not (curr_filenames := glob.glob(os.path.join(dir_path, curr_template))):
            logging.warning("Current template is missing! (Probably manually deleted)")
            return

        pressed = QuestionDialog(
            txt=f"Change current template from:\n{curr_template}\nto:\n{new_template}\n?",
            title="Edit File Template",
        ).display()
        if pressed == QMessageBox.No:
            return

        # generate new filanames
        new_filenames = [
            re.sub(curr_template[:-5], new_template[:-5], curr_filename)
            for curr_filename in curr_filenames
        ]
        # rename the files
        [
            os.rename(curr_filename, new_filename)
            for curr_filename, new_filename in zip(curr_filenames, new_filenames)
        ]
        # rename the log file, if applicable
        with suppress(FileNotFoundError):
            os.rename(
                os.path.join(dir_path, curr_template[:-6] + ".log"),
                os.path.join(dir_path, new_template[:-6] + ".log"),
            )

        # refresh templates
        day = data_import_wdgts.data_days
        self.populate_data_templates_from_day(day)

    def current_date_type_dir_path(self) -> str:
        """Returns path to directory of currently selected date and measurement type"""

        import_wdgts = wdgts.DATA_IMPORT_COLL.read_gui(self._app)

        day = import_wdgts.data_days.get()
        year = import_wdgts.data_years.get()
        month = import_wdgts.data_months.get()

        if import_wdgts.is_image_type:
            meas_settings = wdgts.IMG_MEAS_COLL.read_gui(self._app)
        elif import_wdgts.is_solution_type:
            meas_settings = wdgts.SOL_MEAS_COLL.read_gui(self._app)
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

        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL
        text_lines = DATA_IMPORT_COLL.log_text.get().split("\n")
        curr_template = DATA_IMPORT_COLL.data_templates.get()
        log_filename = re.sub("_\\*.\\w{3}", ".log", curr_template)
        with suppress(AttributeError, TypeError, FileNotFoundError):
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
                full_data.correlate_and_average()

            except AttributeError:
                # No directories found
                g0, tau = (None, None)

            except (NotImplementedError, RuntimeError, ValueError, FileNotFoundError) as exc:
                err_hndlr(exc, sys._getframe(), locals())

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

        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL.read_gui(self._app)
        DATA_IMPORT_COLL.log_text.set("")  # clear first

        # get the log file path
        if DATA_IMPORT_COLL.is_solution_type:
            log_filename = re.sub("_?\\*\\.\\w{3}", ".log", template)
        elif DATA_IMPORT_COLL.is_image_type:
            log_filename = re.sub("\\.\\w{3}", ".log", template)
        file_path = os.path.join(self.current_date_type_dir_path(), log_filename)

        try:  # load the log file in path
            text_lines = helper.read_file_to_list(file_path)
        except (FileNotFoundError, OSError):  # initialize a new log file if no existing file
            with suppress(OSError, IndexError):
                # OSError - missing file/folder (deleted during operation)
                # IndexError - alignment file does not exist
                initialize_dir_log_file(file_path, *self.get_daily_alignment())
                text_lines = helper.read_file_to_list(file_path)
        finally:  # write file to widget
            DATA_IMPORT_COLL.log_text.set("\n".join(text_lines))

    def preview_img_scan(self, template: str) -> None:
        """Doc."""

        if not template:
            return

        data_import_wdgts = wdgts.DATA_IMPORT_COLL.read_gui(self._app)

        if data_import_wdgts.is_image_type:
            # import the data
            file_path = os.path.join(self.current_date_type_dir_path(), template)
            file_dict = file_utilities.load_file_dict(file_path)
            # get the center plane image
            counts = file_dict["ci"]
            ao = file_dict["ao"]
            scan_param = file_dict["scan_param"]
            um_v_ratio = file_dict["xyz_um_to_v"]
            image_data = ImageScanData(counts, ao, scan_param, um_v_ratio)
            image = image_data.build_image("forward", scan_param["n_planes"] // 2)
            # plot it (below)
            data_import_wdgts.img_preview_disp.obj.display_image(image.T, axis=False, cmap="bone")

        pass

    def import_sol_data(self) -> None:
        """Doc."""

        import_wdgts = wdgts.DATA_IMPORT_COLL.read_gui(self._app)
        current_template = import_wdgts.data_templates.get()
        sol_analysis_wdgts = wdgts.SOL_ANALYSIS_COLL.read_gui(self._app)
        curr_dir = self.current_date_type_dir_path()

        if import_wdgts.sol_use_processed:
            file_path = os.path.join(curr_dir, "processed", re.sub("_[*]", "", current_template))

        with self._app.pause_ai_ci():

            if import_wdgts.sol_use_processed and os.path.isfile(file_path):
                print(
                    f"Loading pre-processed data '{current_template}' from hard drive...", end=" "
                )
                full_data = file_utilities.load_pkl(file_path)
                # Load runtimes as int64 if they are not already of that type
                for p in full_data.data:
                    p.runtime = p.runtime.astype(np.int64, copy=False)
                print("Done.")

            else:  # process data
                full_data = CorrFuncTDC()

                # file selection
                if import_wdgts.sol_file_dicrimination.objectName() == "solImportUse":
                    sol_file_selection = (
                        f"{import_wdgts.sol_file_use_or_dont} {import_wdgts.sol_file_selection}"
                    )
                else:
                    sol_file_selection = ""

                try:
                    with suppress(AttributeError):
                        # No directories found
                        full_data.read_fpga_data(
                            os.path.join(curr_dir, current_template),
                            file_selection=sol_file_selection,
                            should_fix_shift=sol_analysis_wdgts.fix_shift,
                            no_plot=not sol_analysis_wdgts.external_plotting,
                        )
                        full_data.correlate_data(verbose=True)

                    if import_wdgts.sol_save_processed:
                        print("Saving the processed data...", end=" ")
                        file_utilities.save_processed_solution_meas(full_data, curr_dir)
                        print("Done.")

                except (NotImplementedError, RuntimeError, ValueError, FileNotFoundError) as exc:
                    err_hndlr(exc, sys._getframe(), locals())
                    return

            # save data and populate combobox
            imported_combobox = wdgts.SOL_ANALYSIS_COLL.imported_templates
            self._app.analysis.loaded_data[current_template] = full_data
            imported_combobox.obj.addItem(current_template)
            imported_combobox.set(current_template)

            logging.info(f"Data loaded for analysis: {current_template}")

    def get_current_full_data(self, imported_template: str = None) -> CorrFuncTDC:
        """Doc."""

        if imported_template is None:
            imported_template = wdgts.SOL_ANALYSIS_COLL.imported_templates.get()
        curr_data_type, *_ = re.split(" -", imported_template)
        return self._app.analysis.loaded_data.get(curr_data_type)

    def populate_sol_meas_analysis(self, imported_template):
        """Doc."""

        sol_data_analysis_wdgts = wdgts.SOL_ANALYSIS_COLL.read_gui(self._app)

        try:
            full_data = self.get_current_full_data(imported_template)
            num_files = len(full_data.data)
        except AttributeError:
            # no imported templates (deleted)
            wdgts.SOL_ANALYSIS_COLL.clear_all_objects()
            sol_data_analysis_wdgts.scan_img_file_num.obj.setRange(1, 1)
            sol_data_analysis_wdgts.scan_img_file_num.set(1)
        else:
            print("Populating analysis GUI...", end=" ")

            # populate general measurement properties
            sol_data_analysis_wdgts.n_files.set(num_files)
            sol_data_analysis_wdgts.scan_duration_min.set(full_data.duration_min)
            sol_data_analysis_wdgts.avg_cnt_rate_khz.set(full_data.avg_cnt_rate_khz)

            if full_data.type == "angular_scan":
                # populate scan images tab
                print("Displaying scan images...", end=" ")
                sol_data_analysis_wdgts.scan_img_file_num.obj.setRange(1, num_files)
                sol_data_analysis_wdgts.scan_img_file_num.set(1)
                self.display_scan_image(1, imported_template)

                # calculate average and display
                print("Averaging and plotting...", end=" ")
                self.calculate_and_show_sol_mean_acf(imported_template)

                scan_settings_text = "\n\n".join(
                    [
                        f"{key}: {(', '.join([f'{ele:.2f}' for ele in val[:5]]) if isinstance(val, Iterable) else f'{val:.2f}')}"
                        for key, val in full_data.angular_scan_settings.items()
                    ]
                )

            elif full_data.type == "static":
                print("Averaging, plotting and fitting...", end=" ")
                self.calculate_and_show_sol_mean_acf(imported_template)
                scan_settings_text = "no scan."

            sol_data_analysis_wdgts.scan_settings.set(scan_settings_text)

            print("Done.")

    def display_scan_image(self, file_num, imported_template: str = None):
        """Doc."""

        with suppress(IndexError, KeyError, AttributeError):
            # IndexError - data import failed
            # KeyError, AttributeError - data deleted
            full_data = self.get_current_full_data(imported_template)
            img = full_data.data[file_num - 1].image
            roi = full_data.data[file_num - 1].roi

            scan_image_disp = wdgts.SOL_ANALYSIS_COLL.scan_image_disp.obj
            scan_image_disp.display_image(img)
            scan_image_disp.plot(roi["col"], roi["row"], color="white")
            scan_image_disp.entitle_and_label("Pixel Number", "Line Number")

    def calculate_and_show_sol_mean_acf(self, imported_template: str = None) -> None:
        """Doc."""

        sol_data_analysis_wdgts = wdgts.SOL_ANALYSIS_COLL.read_gui(self._app)
        full_data = self.get_current_full_data(imported_template)

        if full_data is None:
            return

        if full_data.type == "angular_scan":
            row_disc_method = sol_data_analysis_wdgts.row_dicrimination.objectName()
            if row_disc_method == "solAnalysisRemoveOver":
                avg_corr_args = dict(rejection=sol_data_analysis_wdgts.remove_over)
            elif row_disc_method == "solAnalysisRemoveWorst":
                avg_corr_args = dict(
                    rejection=None, reject_n_worst=sol_data_analysis_wdgts.remove_worst
                )
            else:  # use all rows
                avg_corr_args = dict(rejection=None)

            with suppress(AttributeError):
                # no data loaded
                full_data.average_correlation(**avg_corr_args)

                if sol_data_analysis_wdgts.plot_spatial:
                    x = (full_data.vt_um, "disp")
                    x_label = r"squared displacement ($um^2$)"
                else:
                    x = (full_data.lag, "lag")
                    x_label = "lag (ms)"

                sol_data_analysis_wdgts.row_acf_disp.obj.plot_acfs(
                    x,
                    full_data.average_cf_cr,
                    full_data.g0,
                    full_data.cf_cr[full_data.j_good, :],
                )
                sol_data_analysis_wdgts.row_acf_disp.obj.entitle_and_label(x_label, "G0")

                sol_data_analysis_wdgts.mean_g0.set(full_data.g0 / 1e3)  # shown in thousands
                sol_data_analysis_wdgts.mean_tau.set(0)

                sol_data_analysis_wdgts.n_good_rows.set(n_good := len(full_data.j_good))
                sol_data_analysis_wdgts.n_bad_rows.set(n_bad := len(full_data.j_bad))
                sol_data_analysis_wdgts.remove_worst.obj.setMaximum(n_good + n_bad - 2)

        if full_data.type == "static":
            full_data.average_correlation()
            try:
                full_data.fit_correlation_function()
            except fit_tools.FitError as exc:
                # fit failed, use g0 calculated in 'average_correlation()'
                err_hndlr(exc, sys._getframe(), locals(), lvl="warning")
                sol_data_analysis_wdgts.mean_g0.set(full_data.g0 / 1e3)  # shown in thousands
                sol_data_analysis_wdgts.mean_tau.set(0)
            else:  # fit succeeded
                fit_params = full_data.fit_param["diffusion_3d_fit"]
                g0, tau, _ = fit_params["beta"]
                fit_func = getattr(fit_tools, fit_params["func_name"])
                sol_data_analysis_wdgts.mean_g0.set(g0 / 1e3)  # shown in thousands
                sol_data_analysis_wdgts.mean_tau.set(tau * 1e3)
                y_fit = fit_func(fit_params["x"], *fit_params["beta"])
                sol_data_analysis_wdgts.row_acf_disp.obj.clear()
                sol_data_analysis_wdgts.row_acf_disp.obj.plot_acfs(
                    (full_data.lag, "lag"),
                    full_data.average_cf_cr,
                    full_data.g0,
                )
                sol_data_analysis_wdgts.row_acf_disp.obj.plot(fit_params["x"], y_fit, color="red")

    def calibrate_tdc(self):
        """Doc."""
        # TODO: this is still a work in progress - only use for testing!

        imported_template = wdgts.SOL_ANALYSIS_COLL.imported_templates.get()
        full_data = self.get_current_full_data(imported_template)
        full_data.calibrate_tdc()
        full_data.fit_lifetime_hist()

    def remove_imported_template(self):
        """Doc."""

        imported_templates = wdgts.SOL_ANALYSIS_COLL.imported_templates
        template = imported_templates.get()
        data_type, *_ = re.split(" -", template)
        self._app.analysis.loaded_data[data_type] = None
        imported_templates.obj.removeItem(imported_templates.obj.currentIndex())
        # TODO: clear image properties!

    def convert_files_to_matlab_format(self) -> None:
        """
        Convert files of current template to '.mat',
        after translating their dictionaries to the legacy matlab format.
        """

        DATA_IMPORT_COLL = wdgts.DATA_IMPORT_COLL
        current_template = DATA_IMPORT_COLL.data_templates.get()
        current_dir_path = self.current_date_type_dir_path()

        pressed = QuestionDialog(
            txt=f"Are you sure you wish to convert '{current_template}'?",
            title="Conversion to .mat Format",
        ).display()
        if pressed == QMessageBox.No:
            return

        file_template_path = os.path.join(current_dir_path, current_template)
        file_paths = file_utilities.sort_file_paths_by_file_number(glob.glob(file_template_path))

        print(f"Converting {len(file_paths)} files to '.mat' in legacy MATLAB format...", end=" ")

        for idx, file_path in enumerate(file_paths):
            file_dict = file_utilities.load_file_dict(file_path)
            mat_file_path = re.sub(".pkl", ".mat", file_path)
            if "solution" in mat_file_path:
                mat_file_path = re.sub("solution", r"solution\\matlab", mat_file_path)
            elif "image" in mat_file_path:
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

            current_state = set(wdgts.wdgt_items_to_text_lines(self._gui, wdgts.SETTINGS_TYPES))
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
                pressed = QuestionDialog(
                    "Keep changes if made? " "(otherwise, revert to last loaded settings file.)"
                ).display()
                if pressed == QMessageBox.No:
                    self.load(self._gui.settingsFileName.text())

        else:
            NotificationDialog("Using Current settings.").display()

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
            wdgts.write_gui_to_file(self._gui.frame, wdgts.SETTINGS_TYPES, file_path)
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
            wdgts.read_file_to_gui(file_path, self._gui.frame)
            logging.debug(f"Settings file loaded: '{file_path}'")

    def confirm(self):
        """Doc."""

        self.check_on_close = False


class CamWin:
    """Doc."""

    def __init__(self, gui, app):
        """Doc."""

        self._app = app
        self._gui = gui
        self._cam = None
        self.is_video_on = False

    def init_cam(self):
        """Doc."""

        self._cam = self._app.devices.camera
        self._app.gui.main.impl.dvc_toggle("camera")
        logging.debug("Camera connection opened")

    async def clean_up(self):
        """clean up before closing window"""

        if self._cam is not None:
            await self.toggle_video(keep_off=True)
            self._app.gui.main.impl.dvc_toggle("camera")
            self._app.gui.main.actionCamera_Control.setEnabled(True)
            self._cam = None
            logging.debug("Camera connection closed")

    async def toggle_video(self, keep_off=False):
        """Doc."""

        if not self.is_video_on and not keep_off:  # Turning video ON
            logging.info("Camera video mode is ON")
            self._gui.videoButton.setStyleSheet(
                "background-color: " "rgb(225, 245, 225); " "color: black;"
            )
            self._gui.videoButton.setText("Video ON")

            self.is_video_on = True
            while self.is_video_on:
                self.shoot(verbose=False)
                await asyncio.sleep(0.3)

        else:  # Turning video Off
            logging.info("Camera video mode is OFF")
            self._gui.videoButton.setStyleSheet(
                "background-color: " "rgb(225, 225, 225); " "color: black;"
            )
            self._gui.videoButton.setText("Start Video")

            self.is_video_on = False

    def shoot(self, verbose=True):
        """Doc."""

        self._gui.ImgDisp.display_image(self._cam.grab_image(), cursor=True)
        if verbose:
            logging.info("Camera photo taken")
