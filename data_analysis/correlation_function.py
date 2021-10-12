"""Data organization and manipulation."""

import os
import re
from collections import deque
from contextlib import contextmanager, suppress

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, stats
from skimage import filters as skifilt
from skimage import morphology

from data_analysis.photon_data import TDCPhotonData
from data_analysis.software_correlator import CorrelatorType, SoftwareCorrelator
from utilities import display, file_utilities, fit_tools, helper


class CorrFunc:
    """Doc."""

    def __init__(self):
        self.lag = []
        self.fit_param = dict()

    def average_correlation(
        self,
        rejection=2,
        reject_n_worst=None,
        norm_range=(1e-3, 2e-3),
        delete_list=[],
        should_plot=False,
        **kwargs,
    ):
        """Doc."""

        def calc_weighted_avg(cf_cr, weights):
            """Doc."""

            tot_weights = weights.sum(0)
            avg_cf_cr = (cf_cr * weights).sum(0) / tot_weights
            error_cf_cr = np.sqrt((weights ** 2 * (cf_cr - avg_cf_cr) ** 2).sum(0)) / tot_weights

            return avg_cf_cr, error_cf_cr

        self.rejection = rejection
        self.norm_range = norm_range
        self.delete_list = delete_list
        self.average_all_cf_cr = (self.cf_cr * self.weights).sum(0) / self.weights.sum(0)
        self.median_all_cf_cr = np.median(self.cf_cr, 0)
        JJ = (self.lag > norm_range[1]) & (self.lag < 100)  # work in the relevant part
        self.score = (
            (1 / np.var(self.cf_cr[:, JJ], 0))
            * (self.cf_cr[:, JJ] - self.median_all_cf_cr[JJ]) ** 2
            / len(JJ)
        ).sum(axis=1)

        total_n_rows, _ = self.cf_cr.shape

        if reject_n_worst not in {None, 0}:
            delete_list = np.argsort(self.score)[-reject_n_worst:]
        elif rejection is not None:
            delete_list = np.where(self.score >= self.rejection)[0]
            if len(delete_list) == total_n_rows:
                raise RuntimeError(
                    "All rows are in 'delete_list'! Increase the rejection limit. Ignoring."
                )

        # if 'reject_n_worst' and 'rejection' are both None, use supplied delete list.
        # if no delete list is supplied, use all rows.
        self.j_bad = delete_list
        self.j_good = [row for row in range(total_n_rows) if row not in delete_list]

        self.avg_cf_cr, self.error_cf_cr = calc_weighted_avg(
            self.cf_cr[self.j_good, :], self.weights[self.j_good, :]
        )

        j_t = (self.lag > self.norm_range[0]) & (self.lag < self.norm_range[1])
        self.g0 = (self.avg_cf_cr[j_t] / self.error_cf_cr[j_t] ** 2).sum() / (
            1 / self.error_cf_cr[j_t] ** 2
        ).sum()
        self.normalized = self.avg_cf_cr / self.g0
        self.error_normalized = self.error_cf_cr / self.g0

        if should_plot:
            self.plot_correlation_function(**kwargs)

    def plot_correlation_function(
        self, x_field="lag", y_field="avg_cf_cr", x_scale="log", y_scale="linear", **kwargs
    ):

        x = getattr(self, x_field)
        y = getattr(self, y_field)
        if x_scale == "log":  # remove zero point data
            x, y = x[1:], y[1:]

        with display.show_external_axes() as ax:
            ax.set_xlabel(x_field)
            ax.set_ylabel(y_field)
            ax.set_xscale(x_scale)
            ax.set_yscale(y_scale)
            ax.plot(x, y, "o", **kwargs)
            ax.legend(loc="best")

    def fit_correlation_function(
        self,
        x_field="lag",
        y_field="avg_cf_cr",
        y_error_field="error_cf_cr",
        fit_name="diffusion_3d_fit",
        fit_param_estimate=None,
        fit_range=(1e-3, 100),
        x_scale="log",
        y_scale="linear",
        should_plot=False,
    ):

        if not fit_param_estimate:
            fit_param_estimate = [self.g0, 0.035, 30.0]

        x = getattr(self, x_field)
        y = getattr(self, y_field)
        error_y = getattr(self, y_error_field)
        if x_scale == "log":
            x = x[1:]  # remove zero point data
            y = y[1:]
            error_y = error_y[1:]

        fit_param = fit_tools.curve_fit_lims(
            fit_name,
            fit_param_estimate,
            x,
            y,
            error_y,
            x_limits=fit_range,
            should_plot=should_plot,
            x_scale=x_scale,
            y_scale=y_scale,
        )

        self.fit_param[fit_param["func_name"]] = fit_param

    def accumulate(self, sc):
        """Doc."""

        try:
            self.corrfunc_list.append(sc.corrfunc)
            self.weights_list.append(sc.weights)
            self.cf_cr_list.append(
                sc.countrate * sc.corrfunc - self.after_pulse[: sc.corrfunc.size]
            )
            self.countrate_list.append(sc.countrate)
        except AttributeError:
            raise RuntimeError(
                "Use 'accumulate()' in the context of the 'accumulator()' context manager!"
            )

    @contextmanager
    def accumulator(self):
        """Doc."""

        self.corrfunc_list = []
        self.weights_list = []
        self.cf_cr_list = []
        self.countrate_list = []

        try:
            yield

        finally:
            n_corrfuncs = len(self.corrfunc_list)
            lag_len = len(self.lag)
            self.corrfunc = np.empty(shape=(n_corrfuncs, lag_len), dtype=np.float64)
            self.weights = np.empty(shape=(n_corrfuncs, lag_len), dtype=np.float64)
            self.cf_cr = np.empty(shape=(n_corrfuncs, lag_len), dtype=np.float64)
            for idx in range(n_corrfuncs):
                pad_len = lag_len - len(self.corrfunc_list[idx])
                self.corrfunc[idx] = np.pad(self.corrfunc_list[idx], (0, pad_len))
                self.weights[idx] = np.pad(self.weights_list[idx], (0, pad_len))
                self.cf_cr[idx] = np.pad(self.cf_cr_list[idx], (0, pad_len))

            self.countrate_list = self.countrate_list

            delattr(self, "corrfunc_list")
            delattr(self, "weights_list")
            delattr(self, "cf_cr_list")
            delattr(self, "countrate_list")


class CorrFuncTDC(TDCPhotonData):
    """Doc."""

    NAN_PLACEBO = -100
    DUMP_PATH = "C:/temp_sfcs_data/"

    def __init__(self):
        self.data = []  # list to hold the data of each file
        self.cf = dict()
        self.is_data_dumped = False

    def read_fpga_data(
        self,
        file_path_template: str,
        file_selection: str = "",
        roi_selection: str = "auto",
        should_fix_shift: bool = False,
        should_plot: bool = True,
    ):
        """Doc."""

        print("\nLoading FPGA data from hard drive:", end=" ")
        file_paths = file_utilities.prepare_file_paths(file_path_template, file_selection)
        n_files = len(file_paths)
        _, self.template = os.path.split(file_path_template)
        self.name_on_disk = re.sub("_[*]", "", self.template)

        for idx, file_path in enumerate(file_paths):
            print(f"Loading file No. {idx+1}/{n_files}: '{file_path}'...", end=" ")
            try:
                file_dict = file_utilities.load_file_dict(file_path)
            except OSError:
                print("File has not been downloaded fully from cloud! Skipping.\n")
                continue
            print("Done.")

            full_data = file_dict["full_data"]

            if idx == 0:
                self.after_pulse_param = file_dict["system_info"]["after_pulse_param"]
                self.laser_freq_hz = int(full_data["laser_freq_mhz"] * 1e6)
                self.fpga_freq_hz = int(full_data["fpga_freq_mhz"] * 1e6)

            # Circular sFCS
            if full_data.get("circle_speed_um_s"):
                self.type = "circular_scan"
                self.v_um_ms = full_data["circle_speed_um_s"] * 1e-3  # to um/ms
                raise NotImplementedError("Circular scan analysis not yet implemented...")

            # Angular sFCS
            elif full_data.get("angular_scan_settings"):
                if idx == 0:
                    self.type = "angular_scan"
                    self.angular_scan_settings = full_data["angular_scan_settings"]
                    self.LINE_END_ADDER = 1000
                if (
                    p := self.process_angular_scan_data(
                        full_data, idx, should_fix_shift, roi_selection, should_plot
                    )
                ) is None:
                    continue

            # FCS
            else:
                self.type = "static"
                if (p := self.process_static_data(full_data, idx, verbose=True)) is None:
                    continue

            p.file_path = file_path
            self.data.append(p)

            print(f"Finished processing file No. {idx+1}\n")

        if not len(self.data):
            raise RuntimeError(
                f"Loading FPGA data catastrophically failed (all {n_files}/{n_files} files skipped)."
            )

        # calculate average count rate
        self.avg_cnt_rate_khz = sum([p.avg_cnt_rate_khz for p in self.data]) / len(self.data)

        if full_data.get("duration_s") is not None:
            self.duration_min = full_data["duration_s"] / 60
        else:
            # calculate duration if not supplied
            self.duration_min = (
                np.mean([np.diff(p.runtime).sum() for p in self.data]) / self.laser_freq_hz / 60
            )
            print(f"Calculating duration (not supplied): {self.duration_min:.1f} min\n")

        print(f"Finished loading FPGA data ({len(self.data)}/{n_files} files used).\n")

    def process_static_data(self, full_data, idx, **kwargs):
        """Doc."""

        print("Converting raw data to photons...", end=" ")
        p = self.convert_fpga_data_to_photons(
            full_data["data"],
            version=full_data["version"],
            locate_outliers=True,
            **kwargs,
        )
        print("Done.")

        p.file_num = idx + 1
        p.avg_cnt_rate_khz = full_data["avg_cnt_rate_khz"]

        return p

    def process_angular_scan_data(
        self, full_data, idx, should_fix_shift, roi_selection, should_plot
    ):
        """Doc."""

        print("Converting raw data to photons...", end=" ")
        p = self.convert_fpga_data_to_photons(
            full_data["data"], version=full_data["version"], verbose=True
        )
        print("Done.")

        p.file_num = idx + 1
        p.avg_cnt_rate_khz = full_data["avg_cnt_rate_khz"]

        angular_scan_settings = full_data["angular_scan_settings"]
        linear_part = angular_scan_settings["linear_part"].round().astype(np.uint16)
        self.v_um_ms = angular_scan_settings["actual_speed_um_s"] * 1e-3
        sample_freq_hz = int(angular_scan_settings["sample_freq_hz"])
        ppl_tot = int(angular_scan_settings["points_per_line_total"])
        n_lines = int(angular_scan_settings["n_lines"])

        print("Converting angular scan to image...", end=" ")

        runtime = p.runtime
        cnt, n_pix_tot, n_pix, line_num = convert_angular_scan_to_image(
            runtime, self.laser_freq_hz, sample_freq_hz, ppl_tot, n_lines
        )

        if should_fix_shift:
            pix_shift = fix_data_shift(cnt)
            runtime = p.runtime + pix_shift * round(self.laser_freq_hz / sample_freq_hz)
            cnt, n_pix_tot, n_pix, line_num = convert_angular_scan_to_image(
                runtime, self.laser_freq_hz, sample_freq_hz, ppl_tot, n_lines
            )
            print(f"Shifted by {pix_shift} pixels. Done.")
        else:
            print("Done.")

        # invert every second line
        cnt[1::2, :] = np.flip(cnt[1::2, :], 1)

        print("ROI selection: ", end=" ")

        if roi_selection == "auto":
            print("automatic. Thresholding and smoothing...", end=" ")
            try:
                bw = threshold_and_smooth(cnt)
            except ValueError:
                print("Thresholding failed, skipping file.")
                return None
        else:
            raise NotImplementedError(f"roi_selection={roi_selection} is not implemented.")

        # cut edges
        bw_temp = np.full(bw.shape, False, dtype=bool)
        bw_temp[:, linear_part] = bw[:, linear_part]
        bw = bw_temp

        # discard short and fill long rows
        m2 = np.sum(bw, axis=1)
        bw[m2 < 0.5 * m2.max(), :] = False

        while self.LINE_END_ADDER < bw.shape[0]:
            self.LINE_END_ADDER *= 10

        print("Building ROI...", end=" ")

        line_starts = []
        line_stops = []
        line_start_lables = []
        line_stop_labels = []
        roi = {"row": deque([]), "col": deque([])}

        bw_rows, _ = bw.shape
        for row_idx in range(bw_rows):
            nonzero_row_idxs = bw[row_idx, :].nonzero()[0]
            if nonzero_row_idxs.size != 0:  # if bw row has nonzero elements
                # set mask to True between non-zero edges of row
                left_edge, right_edge = nonzero_row_idxs[0], nonzero_row_idxs[-1]
                bw[row_idx, left_edge:right_edge] = True
                # add row to ROI
                roi["row"].appendleft(row_idx)
                roi["col"].appendleft(left_edge)
                roi["row"].append(row_idx)
                roi["col"].append(right_edge)

                line_starts_new_idx = np.ravel_multi_index((row_idx, left_edge), bw.shape)
                line_starts_new = list(range(line_starts_new_idx, n_pix_tot[-1], bw.size))
                line_stops_new_idx = np.ravel_multi_index((row_idx, right_edge), bw.shape)
                line_stops_new = list(range(line_stops_new_idx, n_pix_tot[-1], bw.size))

                line_start_lables.extend([-row_idx for elem in range(len(line_starts_new))])
                line_stop_labels.extend(
                    [(-row_idx - self.LINE_END_ADDER) for elem in range(len(line_stops_new))]
                )
                line_starts.extend(line_starts_new)
                line_stops.extend(line_stops_new)

        try:
            # repeat first point to close the polygon
            roi["row"].append(roi["row"][0])
            roi["col"].append(roi["col"][0])
        except IndexError:
            print("ROI is empty (need to figure out the cause). Skipping file.", end=" ")
            return None

        # convert lists/deques to numpy arrays
        roi = {key: np.array(val, dtype=np.uint16) for key, val in roi.items()}
        line_start_lables = np.array(line_start_lables, dtype=np.int16)
        line_stop_labels = np.array(line_stop_labels, dtype=np.int16)
        line_starts = np.array(line_starts, dtype=np.int64)
        line_stops = np.array(line_stops, dtype=np.int64)

        print("Done.")

        runtime_line_starts = line_starts * round(self.laser_freq_hz / sample_freq_hz)
        runtime_line_stops = line_stops * round(self.laser_freq_hz / sample_freq_hz)

        runtime = np.hstack((runtime_line_starts, runtime_line_stops, runtime))
        sorted_idxs = np.argsort(runtime)
        p.runtime = runtime[sorted_idxs]
        p.time_stamps = np.diff(p.runtime).astype(np.int32)
        p.line_num = np.hstack(
            (
                line_start_lables,
                line_stop_labels,
                line_num * bw[line_num, n_pix].flatten(),
            )
        )[sorted_idxs]
        p.coarse = np.hstack(
            (
                np.full(runtime_line_starts.size, self.NAN_PLACEBO, dtype=np.int16),
                np.full(runtime_line_stops.size, self.NAN_PLACEBO, dtype=np.int16),
                p.coarse,
            )
        )[sorted_idxs]
        p.coarse2 = np.hstack(
            (
                np.full(runtime_line_starts.size, self.NAN_PLACEBO, dtype=np.int16),
                np.full(runtime_line_stops.size, self.NAN_PLACEBO, dtype=np.int16),
                p.coarse2,
            )
        )[sorted_idxs]
        p.fine = np.hstack(
            (
                np.full(runtime_line_starts.size, self.NAN_PLACEBO, dtype=np.int16),
                np.full(runtime_line_stops.size, self.NAN_PLACEBO, dtype=np.int16),
                p.fine,
            )
        )[sorted_idxs]

        p.image = cnt
        p.roi = roi

        # plotting of scan image and ROI
        if should_plot:
            with display.show_external_axes(should_force_aspect=True) as ax:
                ax.set_title(f"file No. {p.file_num} of {self.template}")
                ax.set_xlabel("Pixel Number")
                ax.set_ylabel("Line Number")
                ax.imshow(cnt)
                ax.plot(roi["col"], roi["row"], color="white")

        # reverse rows again
        bw[1::2, :] = np.flip(bw[1::2, :], 1)
        p.bw_mask = bw

        # get image line correlation to subtract trends
        img = p.image * p.bw_mask

        p.image_line_corr = line_correlations(img, p.bw_mask, roi, sample_freq_hz)

        return p

    def correlate_and_average(self, **kwargs):
        """Doc."""

        cf = self.correlate_data(**kwargs)
        cf.average_correlation(**kwargs)

    @file_utilities.rotate_data_to_disk
    def correlate_data(self, **kwargs):
        """Doc."""

        if hasattr(self, "angular_scan_settings"):
            cf = self.correlate_angular_scan_data(**kwargs)
        else:
            cf = self.correlate_static_data(**kwargs)

        if "cf_name" in kwargs:
            cf_name = kwargs["cf_name"]
        else:
            cf_name = f"gate {cf.gate_ns}"
        self.cf[cf_name] = cf

        return cf

    def correlate_static_data(
        self,
        gate_ns=(0, np.inf),
        run_duration=None,
        min_time_frac=0.5,
        n_runs_requested=60,
        verbose=False,
        **kwargs,
    ):
        """Correlates Data for static FCS"""

        if verbose:
            print(f"Correlating {self.template}:", end=" ")

        if run_duration is None:  # auto determination of run duration
            if len(self.cf) > 0:  # read run_time from the last calculated correlation function
                run_duration = next(reversed(self.cf.values())).run_duration
            else:  # auto determine
                total_duration_estimate = 0
                for p in self.data:
                    mu = np.median(p.time_stamps) / np.log(2)
                    total_duration_estimate = (
                        total_duration_estimate + mu * len(p.runtime) / self.laser_freq_hz
                    )
                run_duration = total_duration_estimate / n_runs_requested

        self.min_duration_frac = min_time_frac
        duration = []

        SC = SoftwareCorrelator()

        CF = CorrFunc()
        CF.run_duration = run_duration
        CF.skipped_duration = 0
        CF.gate_ns = gate_ns

        with CF.accumulator():
            for p in self.data:

                if verbose:
                    print(f"({p.file_num})", end=" ")

                # Ignore short segments (default is below half the run_duration)
                for se_idx, (se_start, se_end) in enumerate(p.all_section_edges):
                    segment_time = (p.runtime[se_end] - p.runtime[se_start]) / self.laser_freq_hz
                    if segment_time < min_time_frac * run_duration:
                        if verbose:
                            print(
                                f"Duration of segment No. {se_idx} of file No. {p.file_num} ({segment_time:.2f}s) is too short. Skipping segment...",
                                end=" ",
                            )
                        CF.skipped_duration += segment_time
                        continue

                    runtime = p.runtime[se_start : se_end + 1]

                    # Gating
                    if hasattr(p, "delay_time"):
                        delay_time = p.delay_time[se_start : se_end + 1]
                        j_gate = np.logical_and(
                            delay_time >= CF.gate_ns[0], delay_time <= CF.gate_ns[1]
                        )
                        runtime = runtime[j_gate]
                        delay_time = delay_time[j_gate]  # TODO: why is this not used anywhere?
                    elif gate_ns != (0, np.inf):
                        raise RuntimeError("For gating, TDC must first be calibrated!")

                    # split into segments of approx time of run_duration
                    n_splits = helper.div_ceil(segment_time, run_duration)
                    splits = np.linspace(0, runtime.size, n_splits + 1, dtype=np.int32)
                    time_stamps = np.diff(runtime).astype(np.int32)

                    for k in range(n_splits):

                        ts_split = time_stamps[splits[k] : splits[k + 1]]
                        duration.append(ts_split.sum() / self.laser_freq_hz)
                        SC.soft_cross_correlate(
                            ts_split,
                            CorrelatorType.PH_DELAY_CORRELATOR,
                            timebase_ms=1000 / self.laser_freq_hz,
                        )  # time base of 20MHz to ms

                        gate_to_laser_pulses = np.min(
                            [1, (gate_ns[1] - gate_ns[0]) * self.laser_freq_hz / 1e9]
                        )
                        if len(CF.lag) < len(SC.lag):
                            CF.lag = SC.lag
                            if self.after_pulse_param[0] == "multi_exponent_fit":
                                # work with any number of exponents
                                beta = self.after_pulse_param[1]
                                CF.after_pulse = gate_to_laser_pulses * np.dot(
                                    beta[::2], np.exp(-np.outer(beta[1::2], CF.lag))
                                )

                        # Append new correlation functions
                        CF.accumulate(SC)

        CF.total_duration = sum(duration)

        if verbose:
            if CF.skipped_duration:
                skipped_ratio = CF.skipped_duration / CF.total_duration
                print(f"Skipped/total duration: {skipped_ratio:.1%}", end=" ")
            print("- Done.")

        return CF

    def correlate_angular_scan_data(
        self, gate_ns=(0, np.inf), min_time_frac=0.5, subtract_bg_corr=True, **kwargs
    ):
        """Correlates data for angular scans"""

        print(f"Correlating angular scan data '{self.template}':", end=" ")

        self.min_duration_frac = min_time_frac
        duration = []

        SC = SoftwareCorrelator()

        CF = CorrFunc()
        CF.lag = []
        CF.skipped_duration = 0
        CF.gate_ns = gate_ns

        with CF.accumulator():

            for p in self.data:
                print(f"({p.file_num})", end=" ")
                line_num = p.line_num
                min_line, max_line = line_num[line_num > 0].min(), line_num.max()
                runtm = p.runtime
                if hasattr(p, "delay_time"):
                    delaytm = p.delay_time
                    Jgate = np.logical_or(
                        np.logical_and(delaytm >= CF.gate_ns[0], delaytm <= CF.gate_ns[1]),
                        delaytm == np.nan,
                    )
                    runtm = runtm[Jgate]
                    delaytm = delaytm[Jgate]
                    line_num = line_num[Jgate]
                elif gate_ns != (0, np.inf):
                    raise RuntimeError("For gating you need to run TDCcalibration first")

                time_stamps = np.diff(runtm).astype(np.int32)
                for line_idx, j in enumerate(range(min_line, max_line + 1)):
                    valid = (line_num == j).astype(np.int8)
                    valid[line_num == -j] = -1
                    valid[line_num == -j - self.LINE_END_ADDER] = -2
                    # both photons separated by time-stamp should belong to the line
                    valid = valid[1:]

                    # remove photons from wrong lines
                    timest = time_stamps[valid != 0]
                    valid = valid[valid != 0]

                    if not valid.size:
                        print(f"No valid photons in line {j}. Skipping.")
                        continue

                    # check that we start with the line beginning and not its end
                    if valid[0] != -1:
                        # remove photons till the first found beginning
                        j_start = np.where(valid == -1)[0]
                        if len(j_start) > 0:
                            timest = timest[j_start[0] :]
                            valid = valid[j_start[0] :]

                        # check that we stop with the line ending and not its beginning
                    if valid[-1] != -2:
                        # remove photons till the last found ending
                        j_start = np.where(valid == -1)[0]
                        if len(j_start) > 0:
                            timest = timest[j_start[0] :]
                            valid = valid[j_start[0] :]

                    # the first photon in line measures the time from line start and the line end (-2) finishes the duration of the line
                    dur = timest[(valid == 1) | (valid == -2)].sum() / self.laser_freq_hz
                    duration.append(dur)
                    ts_split = np.vstack((timest, valid))

                    SC.soft_cross_correlate(
                        ts_split,
                        CorrelatorType.PH_DELAY_CORRELATOR_LINES,
                        # time base of 20MHz to ms
                        timebase_ms=1000 / self.laser_freq_hz,
                    )

                    # remove background correlation
                    if subtract_bg_corr:
                        bg_corr = np.interp(
                            SC.lag,
                            p.image_line_corr[line_idx]["lag"],
                            p.image_line_corr[line_idx]["corrfunc"],
                            right=0,
                        )
                    else:
                        bg_corr = 0
                    SC.corrfunc -= bg_corr

                    if len(CF.lag) < len(SC.lag):
                        CF.lag = SC.lag
                        CF.after_pulse = self.calculate_afterpulse(gate_ns, CF.lag)

                    # append new corrfunc, weights, cf_cr and countrate to inner lists
                    CF.accumulate(SC)

        CF.vt_um = self.v_um_ms * CF.lag
        CF.total_duration = sum(duration)

        if CF.skipped_duration:
            skipped_ratio = CF.skipped_duration / CF.total_duration
            print(f"Skipped/total duration: {skipped_ratio:.1%}", end=" ")
        print("- Done.")

        return CF

    def plot_correlation_functions(
        self, x_field="lag", y_field="normalized", x_scale="log", y_scale="linear", **kwargs
    ):

        fig, _ = display.get_fig_with_axes()
        for cf_name, cf in self.cf.items():
            cf.plot_correlation_function(
                x_field, y_field, x_scale, y_scale, label=cf_name, fig_handle=fig
            )

    def calculate_afterpulse(self, gate_ns, lag):
        gate_to_laser_pulses = np.min([1, (gate_ns[1] - gate_ns[0]) * self.laser_freq_hz / 1e9])
        if self.after_pulse_param[0] == "multi_exponent_fit":
            # work with any number of exponents
            beta = self.after_pulse_param[1]
            after_pulse = gate_to_laser_pulses * np.dot(
                beta[::2], np.exp(-np.outer(beta[1::2], lag))
            )
        elif self.after_pulse_param[0] == "exponent_of_polynom_of_log":  # for old Matlab files
            beta = self.after_pulse_param[1]
            if lag[0] == 0:
                lag[0] = np.nan
            after_pulse = gate_to_laser_pulses * np.exp(np.polyval(beta, np.log(lag)))

        return after_pulse

    def dump_or_load_data(self, should_load: bool):
        """Doc."""

        with suppress(AttributeError):
            # AttributeError - name_on_disk is not defined (happens when doing alignment, for example)
            if should_load:  # loading data
                if self.is_data_dumped:
                    print(
                        f"Loading dumped '{self.name_on_disk}' from '{self.DUMP_PATH}'...", end=" "
                    )
                    with suppress(FileNotFoundError):
                        self.data = file_utilities.load_pkl(
                            os.path.join(self.DUMP_PATH + self.name_on_disk)
                        )
                        self.is_data_dumped = False
                    print("Done.")
            else:  # dumping data
                print(f"Dumping '{self.name_on_disk}' to '{self.DUMP_PATH}'...", end=" ")
                is_saved = file_utilities.save_object_to_disk(
                    self.data, self.DUMP_PATH, self.name_on_disk
                )
                if is_saved:
                    self.data = []
                    self.is_data_dumped = True
                    print("Done.")
                else:
                    print("Data size under 100 Mb, not dumping.")


#        file_utilities.deep_size_estimate(self)  # TESTESTEST - show 'self' sizes after loading or dumping 'self.data'


class SFCSExperiment:
    """Doc."""

    def __init__(self, name=None, **kwargs):
        self.confocal = CorrFuncTDC()
        self.sted = CorrFuncTDC()
        self.name = name

    def load_experiment(
        self,
        scanName: str = "confocal",  # or sted
        file_path_template: str = "",
        file_selection: str = "",
        should_plot=True,
        **kwargs,
    ):

        scan = getattr(self, scanName)
        file_paths = file_path_template + file_selection
        if "should_fix_shift" not in kwargs:
            if file_paths[-4:] == ".mat":  # old Matlab data need a fix shift by default
                kwargs["should_fix_shift"] = True
            else:
                kwargs["should_fix_shift"] = False

        if "cf_name" not in kwargs:
            if scanName == "confocal":
                kwargs["cf_name"] = "Confocal"
            else:  # sted
                kwargs["cf_name"] = "CW STED"
        cf_name = kwargs["cf_name"]

        scan.read_fpga_data(file_paths, should_plot=should_plot, **kwargs)

        if "x_field" not in kwargs:
            if scan.type == "static":
                kwargs["x_field"] = "lag"
            else:  # angular or circular scan
                kwargs["x_field"] = "vt_um"

        scan.correlate_and_average(should_plot=False, **kwargs)

        # TODO: adapt to use display.py
        if should_plot:
            plt.figure()
            scan.cf[cf_name].plot_correlation_function(
                y_field="average_all_cf_cr", label="average_all_cf_cr", **kwargs
            )
            scan.cf[cf_name].plot_correlation_function(
                y_field="avg_cf_cr", label="avg_cf_cr", **kwargs
            )
            plt.legend(loc="best")
            plt.show()


def convert_angular_scan_to_image(runtime, laser_freq_hz, sample_freq_hz, ppl_tot, n_lines):
    """utility function for opening Angular Scans"""

    n_pix_tot = runtime * sample_freq_hz // laser_freq_hz
    # to which pixel photon belongs
    n_pix = np.mod(n_pix_tot, ppl_tot)
    line_num_tot = n_pix_tot // ppl_tot
    # one more line is for return to starting positon
    line_num = np.mod(line_num_tot, n_lines + 1)

    cnt = np.empty((n_lines + 1, ppl_tot), dtype=np.uint16)
    bins = np.arange(-0.5, ppl_tot)
    for j in range(n_lines + 1):
        cnt[j, :], _ = np.histogram(n_pix[line_num == j], bins=bins)

    return cnt, n_pix_tot, n_pix, line_num


def fix_data_shift(cnt) -> int:
    """Doc."""

    def get_best_pix_shift(img: np.ndarray, min_shift, max_shift) -> int:
        """Doc."""

        score = np.empty(shape=(max_shift - min_shift), dtype=np.uint32)
        pix_shifts = np.arange(min_shift, max_shift)
        for idx, pix_shift in enumerate(range(min_shift, max_shift)):
            rolled_img = np.roll(img, pix_shift).astype(np.uint32)
            score[idx] = ((rolled_img[:-1:2, :] - np.fliplr(rolled_img[1::2, :])) ** 2).sum()
        return pix_shifts[score.argmin()]

    print("Fixing line shift...", end=" ")

    height, width = cnt.shape

    min_pix_shift = -round(width / 2)
    max_pix_shift = min_pix_shift + width + 1
    pix_shift = get_best_pix_shift(cnt, min_pix_shift, max_pix_shift)

    # Test if not stuck in local minimum (outer_half_sum > inner_half_sum)
    # OR if the 'return row' (the empty one) is not at the bottom for some reason
    # TODO: ask Oleg how the latter can happen
    rolled_cnt = np.roll(cnt, pix_shift)
    inner_half_sum = rolled_cnt[:, int(width * 0.25) : int(width * 0.75)].sum()
    outer_half_sum = rolled_cnt.sum() - inner_half_sum
    return_row_idx = rolled_cnt.sum(axis=1).argmin()

    if (outer_half_sum > inner_half_sum) or return_row_idx != height - 1:
        if return_row_idx != height - 1:
            print("Data is heavily shifted, check it out!", end=" ")
        min_pix_shift = -round(cnt.size / 2)
        max_pix_shift = min_pix_shift + cnt.size + 1
        pix_shift = get_best_pix_shift(cnt, min_pix_shift, max_pix_shift)

    return pix_shift


def threshold_and_smooth(img, otsu_classes=4, n_bins=256, disk_radius=2) -> np.ndarray:
    """Doc."""

    thresh = skifilt.threshold_multiotsu(
        skifilt.median(img).astype(np.float32), otsu_classes, nbins=n_bins
    )  # minor filtering of outliers
    cnt_dig = np.digitize(img, bins=thresh)
    plateau_lvl = np.median(img[cnt_dig == (otsu_classes - 1)])
    std_plateau = stats.median_absolute_deviation(img[cnt_dig == (otsu_classes - 1)])
    dev_cnt = img - plateau_lvl
    bw = dev_cnt > -std_plateau
    bw = ndimage.binary_fill_holes(bw)
    disk_open = morphology.selem.disk(radius=disk_radius)
    bw = morphology.opening(bw, selem=disk_open)
    return bw


def line_correlations(image, bw_mask, roi, sampling_freq) -> list:
    """Returns a list of auto-correlations of the lines of an image"""

    image_line_corr = []
    for j in range(roi["row"].min(), roi["row"].max() + 1):
        line = image[j][bw_mask[j] > 0]
        try:
            c, lags = auto_corr(
                line.astype(np.float64)
            )  # TODO: try making line int16 instead of uint16, so that this astype won't be needed (check corr afterwards!)
        except ValueError:
            print(f"Auto correlation of line #{j} has failed. Skipping.", end=" ")
        else:
            c = c / line.mean() ** 2 - 1
            c[0] -= 1 / line.mean()  # subtracting shot noise, small stuff really
            image_line_corr.append(
                {
                    "lag": lags * 1e3 / sampling_freq,  # in ms
                    "corrfunc": c,
                }
            )
    return image_line_corr


def auto_corr(a):
    """Does correlation similar to Matlab xcorr, cuts positive lags, normalizes properly"""

    c = np.correlate(a, a, mode="full")
    c = c[c.size // 2 :]
    c = c / np.arange(c.size, 0, -1)
    lags = np.arange(c.size, dtype=np.uint16)

    return c, lags
