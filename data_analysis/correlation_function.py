"""Data organization and manipulation."""

import os
from collections import deque

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy import ndimage, stats
from skimage import filters as skifilt
from skimage import morphology

from data_analysis.photon_data import PhotonData
from data_analysis.software_correlator import CorrelatorType, SoftwareCorrelator
from utilities import display, file_utilities, fit_tools, helper


class CorrFuncData:
    """Doc."""

    def average_correlation(
        self,
        rejection=2,
        reject_n_worst=None,
        norm_range=(1e-3, 2e-3),
        delete_list=[],
        no_plot=True,
    ):
        """Doc."""

        def calc_weighted_avg(cf_cr, weights):
            """Doc."""

            tot_weights = weights.sum(0)

            average_cf_cr = (cf_cr * weights).sum(0) / tot_weights

            error_cf_cr = (
                np.sqrt((weights ** 2 * (cf_cr - average_cf_cr) ** 2).sum(0)) / tot_weights
            )

            return average_cf_cr, error_cf_cr

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

        # if 'reject_n_worst' and 'rejection' are both None, use supplied delete list.
        # if no delete list is supplied, use all rows.
        self.j_bad = delete_list
        self.j_good = [row for row in range(total_n_rows) if row not in delete_list]

        self.average_cf_cr, self.error_cf_cr = calc_weighted_avg(
            self.cf_cr[self.j_good, :], self.weights[self.j_good, :]
        )

        Jt = (self.lag > self.norm_range[0]) & (self.lag < self.norm_range[1])
        self.g0 = (self.average_cf_cr[Jt] / self.error_cf_cr[Jt] ** 2).sum() / (
            1 / self.error_cf_cr[Jt] ** 2
        ).sum()
        self.normalized = self.average_cf_cr / self.g0
        self.error_normalized = self.error_cf_cr / self.g0

        if not no_plot:
            self.plot_correlation_function()

    def plot_correlation_function(
        self, x_field="lag", y_field="average_cf_cr", x_scale="log", y_scale="linear"
    ):

        x = getattr(self, x_field)
        y = getattr(self, y_field)
        if x_scale == "log":  # remove zero point data
            x, y = x[1:], y[1:]

        with display.show_external_ax() as ax:
            ax.set_xlabel(x_field)
            ax.set_ylabel(y_field)
            ax.set_xscale(x_scale)
            ax.set_yscale(y_scale)
            ax.plot(x, y, "o")

    def fit_correlation_function(
        self,
        x_field="lag",
        y_field="average_cf_cr",
        y_error_field="error_cf_cr",
        fit_func="diffusion_3d_fit",
        fit_param_estimate=None,
        fit_range=(1e-3, 100),
        no_plot=True,
        x_scale="log",
        y_scale="linear",
    ):

        if not fit_param_estimate:
            fit_param_estimate = (self.g0, 0.035, 30.0)

        x = getattr(self, x_field)
        y = getattr(self, y_field)
        error_y = getattr(self, y_error_field)
        if x_scale == "log":
            x = x[1:]  # remove zero point data
            y = y[1:]
            error_y = error_y[1:]

        FP = fit_tools.curve_fit_lims(
            fit_func,
            fit_param_estimate,
            x,
            y,
            error_y,
            x_limits=fit_range,
            no_plot=no_plot,
            x_scale=x_scale,
            y_scale=y_scale,
        )

        if not hasattr(self, "fit_param"):
            self.fit_param = dict()

        self.fit_param[FP["func_name"]] = FP


class CorrFuncTDC(CorrFuncData):
    """Doc."""

    def __init__(self):
        self.data = []  # list to hold the data of each file
        self.nan_placebo = -100

    def read_fpga_data(
        self,
        file_path_template: str,
        file_selection: str = "",
        should_fix_shift: bool = False,
        roi_selection: str = "auto",
        no_plot: bool = False,
    ):
        """Doc."""

        print("\nLoading FPGA data from hard drive:", end=" ")
        file_paths = file_utilities.prepare_file_paths(file_path_template, file_selection)
        n_files = len(file_paths)
        _, self.template = os.path.split(file_path_template)

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
                self.laser_freq_hz = full_data["laser_freq_mhz"] * 1e6
                self.fpga_freq_hz = full_data["fpga_freq_mhz"] * 1e6

            # Circular sFCS
            if full_data.get("circle_speed_um_s"):
                self.type = "circular_scan"
                self.v_um_ms = full_data["circle_speed_um_s"] / 1000  # to um/ms
                raise NotImplementedError("Circular scan analysis not yet implemented...")

            # Angular sFCS
            elif full_data.get("angular_scan_settings"):
                if idx == 0:
                    self.type = "angular_scan"
                    self.angular_scan_settings = full_data["angular_scan_settings"]
                    self.line_end_adder = 1000
                if (
                    p := self.process_angular_scan_data(
                        full_data, idx, should_fix_shift, roi_selection, no_plot
                    )
                ) is None:
                    continue

            # FCS
            else:
                self.type = "static"
                if (p := self.process_static_data(full_data, idx)) is None:
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

    def process_angular_scan_data(
        self, full_data, idx, should_fix_shift, roi_selection, no_plot
    ) -> PhotonData:
        """Doc."""

        print("Converting raw data to photons...", end=" ")
        p = PhotonData()
        p.convert_fpga_data_to_photons(
            full_data["data"], version=full_data["version"], verbose=True
        )
        print("Done.")

        p.file_num = idx + 1
        p.avg_cnt_rate_khz = full_data["avg_cnt_rate_khz"]

        angular_scan_settings = full_data["angular_scan_settings"]
        linear_part = np.array(np.round(angular_scan_settings["linear_part"]), dtype=np.int32)
        self.v_um_ms = angular_scan_settings["actual_speed_um_s"] / 1000
        sample_freq_hz = angular_scan_settings["sample_freq_hz"]
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

        while self.line_end_adder < bw.shape[0]:
            self.line_end_adder *= 10

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
                    [(-row_idx - self.line_end_adder) for elem in range(len(line_stops_new))]
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
        roi = {key: np.array(val) for key, val in roi.items()}
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
        p.line_num = np.hstack(
            (
                line_start_lables,
                line_stop_labels,
                line_num * bw[line_num, n_pix].flatten(),
            )
        )[sorted_idxs]
        # TODO: Ask Oleg - will nan's be needed in TDC analysis?
        p.coarse = np.hstack(
            (
                np.full(runtime_line_starts.size, self.nan_placebo, dtype=np.int16),
                np.full(runtime_line_stops.size, self.nan_placebo, dtype=np.int16),
                p.coarse,
            )
        )[sorted_idxs]
        p.coarse2 = np.hstack(
            (
                np.full(runtime_line_starts.size, self.nan_placebo, dtype=np.int16),
                np.full(runtime_line_stops.size, self.nan_placebo, dtype=np.int16),
                p.coarse2,
            )
        )[sorted_idxs]
        p.fine = np.hstack(
            (
                np.full(runtime_line_starts.size, self.nan_placebo, dtype=np.int16),
                np.full(runtime_line_stops.size, self.nan_placebo, dtype=np.int16),
                p.fine,
            )
        )[sorted_idxs]

        p.image = cnt
        p.roi = roi

        # plotting of scan image and ROI
        if not no_plot:
            with display.show_external_ax(should_force_aspect=True) as ax:
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

    def process_static_data(self, full_data, idx):
        """Doc."""

        print("Converting raw data to photons...", end=" ")
        p = PhotonData()
        p.convert_fpga_data_to_photons(
            full_data["data"], version=full_data["version"], verbose=True
        )
        print("Done.")

        p.file_num = idx + 1
        p.avg_cnt_rate_khz = full_data["avg_cnt_rate_khz"]

        return p

    def correlate_and_average(self, verbose=False, **kwargs):
        self.correlate_data(verbose=verbose, **kwargs)
        self.average_correlation(**kwargs)

    def correlate_data(self, verbose=False, **kwargs):
        if hasattr(self, "angular_scan_settings"):
            self.correlate_angular_scan_data(**kwargs)
        else:
            self.correlate_regular_data(verbose=verbose, **kwargs)

    def correlate_regular_data(
        self,
        run_duration=-1,
        min_time_frac=0.5,
        max_outlier_prob=1e-5,
        n_runs_requested=60,
        verbose=False,
    ):
        """Correlates Data for static (regular) FCS"""

        cf = SoftwareCorrelator()

        if run_duration < 0:  # auto determination of run duration
            total_duration_estimate = 0
            for p in self.data:
                time_stamps = np.diff(p.runtime)
                mu = np.median(time_stamps) / np.log(2)
                total_duration_estimate = (
                    total_duration_estimate + mu * len(p.runtime) / self.laser_freq_hz
                )

            run_duration = total_duration_estimate / n_runs_requested

        self.min_duration_frac = min_time_frac
        duration = []
        self.lag = []
        self.corrfunc = []
        self.weights = []
        self.cf_cr = []
        self.countrate = []

        self.total_duration_skipped = 0

        if verbose:
            print(f"Correlating {self.template}:", end=" ")

        for p in self.data:

            if verbose:
                print(f"({p.file_num})", end=" ")
            # find additional outliers
            time_stamps = np.diff(p.runtime).astype(np.int32)
            # for exponential distribution MEDIAN and MAD are the same, but for
            # biexponential MAD seems more sensitive
            mu = max(
                np.median(time_stamps), np.abs(time_stamps - time_stamps.mean()).mean()
            ) / np.log(2)
            max_time_stamp = stats.expon.ppf(1 - max_outlier_prob / len(time_stamps), scale=mu)
            sec_edges = (time_stamps > max_time_stamp).nonzero()[0]
            no_outliers = len(sec_edges)
            if no_outliers > 0:
                if verbose:
                    print(f"{no_outliers} of all outliers")

            sec_edges = np.append(np.insert(sec_edges, 0, 0), len(time_stamps))
            p.all_section_edges = np.array([sec_edges[:-1], sec_edges[1:]]).T

            for se_idx, (se_start, se_end) in enumerate(p.all_section_edges):
                # split into segments of approx time of run_duration
                segment_time = (p.runtime[se_end] - p.runtime[se_start]) / self.laser_freq_hz
                if segment_time < min_time_frac * run_duration:
                    if verbose:
                        print(
                            f"Duration of segment No. {se_idx} of file {p.fname} is {segment_time}s: too short. Skipping segment..."
                        )
                    self.total_duration_skipped += segment_time
                    continue

                n_splits = helper.div_ceil(segment_time, run_duration)
                splits = np.linspace(0, (se_end - se_start), n_splits + 1, dtype=np.int32)
                ts = time_stamps[se_start:se_end]

                for k in range(n_splits):

                    ts_split = ts[splits[k] : splits[k + 1]]
                    duration.append(ts_split.sum() / self.laser_freq_hz)
                    cf.soft_cross_correlate(
                        ts_split,
                        CorrelatorType.PH_DELAY_CORRELATOR,
                        timebase_ms=1000 / self.laser_freq_hz,
                    )  # time base of 20MHz to ms
                    if len(self.lag) < len(cf.lag):
                        self.lag = cf.lag
                        if self.after_pulse_param[0] == "multi_exponent_fit":
                            # work with any number of exponents
                            beta = self.after_pulse_param[1]
                            self.after_pulse = np.dot(
                                beta[::2], np.exp(-np.outer(beta[1::2], self.lag))
                            )

                    self.corrfunc.append(cf.corrfunc)
                    self.weights.append(cf.weights)
                    self.countrate.append(cf.countrate)
                    self.cf_cr.append(
                        cf.countrate * cf.corrfunc - self.after_pulse[: cf.corrfunc.size]
                    )

        # zero pad
        lag_len = len(self.lag)
        for idx in range(n_splits):
            pad_len = lag_len - len(self.corrfunc[idx])
            self.corrfunc[idx] = np.pad(self.corrfunc[idx], (0, pad_len), "constant")
            self.weights[idx] = np.pad(self.weights[idx], (0, pad_len), "constant")
            self.cf_cr[idx] = np.pad(self.cf_cr[idx], (0, pad_len), "constant")

        self.corrfunc = np.array(self.corrfunc)
        self.weights = np.array(self.weights)
        self.cf_cr = np.array(self.cf_cr)

        self.total_duration = sum(duration)
        if verbose:
            if self.total_duration_skipped:
                print(
                    f"- Done.\n{self.total_duration_skipped:.2f} s skipped out of {self.total_duration:.2f} s."
                )
            else:
                print("- Done.")

    def correlate_angular_scan_data(self, min_time_frac=0.5, subtract_bg_corr=True):
        """Doc."""

        cf = SoftwareCorrelator()

        self.min_time_frac = min_time_frac
        duration = []
        self.lag = []
        self.corrfunc = []
        self.weights = []
        self.cf_cr = []
        self.countrate = []
        self.total_duration_skipped = 0

        print(f"Correlating angular scan data '{self.template}':", end=" ")

        for p in self.data:
            print(f"({p.file_num})", end=" ")

            time_stamps = np.diff(p.runtime).astype(np.int32)
            line_num = p.line_num
            min_line, max_line = line_num[line_num > 0].min(), line_num.max()
            for line_idx, j in enumerate(range(min_line, max_line + 1)):
                valid = (line_num == j).astype(np.int8)
                valid[line_num == -j] = -1
                valid[line_num == -j - self.line_end_adder] = -2
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
                    Jstrt = np.where(valid == -1)[0][0]
                    timest = timest[Jstrt:]
                    valid = valid[Jstrt:]

                    # check that we stop with the line ending and not its beginning
                if valid[-1] != -2:
                    # remove photons till the last found ending
                    Jend = np.where(valid == -2)[0][-1]
                    timest = timest[:Jend]
                    valid = valid[:Jend]

                # the first photon in line measures the time from line start and the line end (-2) finishes the duration of the line
                dur = timest[(valid == 1) | (valid == -2)].sum() / self.laser_freq_hz
                duration.append(dur)
                ts_split = np.vstack((timest, valid))
                cf.soft_cross_correlate(
                    ts_split,
                    CorrelatorType.PH_DELAY_CORRELATOR_LINES,
                    # time base of 20MHz to ms
                    timebase_ms=1000 / self.laser_freq_hz,
                )

                if subtract_bg_corr:
                    bg_corr = np.interp(
                        cf.lag,
                        p.image_line_corr[line_idx]["lag"],
                        p.image_line_corr[line_idx]["corrfunc"],
                        right=0,
                    )
                else:
                    bg_corr = 0

                cf.corrfunc = cf.corrfunc - bg_corr

                if len(self.lag) < len(cf.lag):
                    self.lag = cf.lag
                    if self.after_pulse_param[0] == "multi_exponent_fit":
                        # work with any number of exponents
                        # y = beta(1)*exp(-beta(2)*t) + beta(3)*exp(-beta(4)*t) + beta(5)*exp(-beta(6)*t);
                        beta = self.after_pulse_param[1]
                        self.after_pulse = np.dot(
                            beta[::2], np.exp(-np.outer(beta[1::2], self.lag))
                        )

                self.corrfunc.append(cf.corrfunc)
                self.weights.append(cf.weights)
                self.countrate.append(cf.countrate)
                self.cf_cr.append(cf.countrate * cf.corrfunc - self.after_pulse[: cf.corrfunc.size])

        print("- Done.")

        len_lag = len(self.lag)
        for idx in range(len(self.corrfunc)):  # zero pad
            pad_len = len_lag - len(self.corrfunc[idx])
            self.corrfunc[idx] = np.pad(self.corrfunc[idx], (0, pad_len), "constant")
            self.weights[idx] = np.pad(self.weights[idx], (0, pad_len), "constant")
            self.cf_cr[idx] = np.pad(self.cf_cr[idx], (0, pad_len), "constant")

        self.corrfunc = np.array(self.corrfunc)
        self.weights = np.array(self.weights)
        self.cf_cr = np.array(self.cf_cr)
        self.vt_um = self.v_um_ms * self.lag
        self.total_duration = sum(duration)

        if self.total_duration_skipped:
            print(
                f"{self.total_duration_skipped:.2f} s skipped out of {self.total_duration:.2f} s. Done."
            )

    def calibrate_tdc(  # NOQA C901
        self,
        tdc_chain_length=128,
        pick_valid_bins_method="auto",
        pick_calib_bins_method="auto",
        calib_time_s=40e-9,
        n_zeros_for_fine_bounds=10,
        fine_shift=0,
        time_bins_for_hist_ns=0.1,
        valid_coarse_bins=np.arange(19),
        exmpl_photon_data=None,
        sync_coarse_time_to=None,
        calibration_coarse_bins=np.arange(3, 12),
    ):

        self.tdc_calib = dict()

        # keep runtime elements of each file for array size allocation
        n_elem = [0]
        for p in self.data:
            n_elem.append(p.runtime.size)

        n_elem = np.cumsum(n_elem)
        # unite coarse and fine times from all files
        coarse = np.empty(shape=(n_elem[-1],), dtype=np.int16)
        fine = np.empty(shape=(n_elem[-1],), dtype=np.int16)
        for i, p in enumerate(self.data):
            coarse[n_elem[i] : n_elem[i + 1]] = p.coarse
            fine[n_elem[i] : n_elem[i + 1]] = p.fine

        if self.type == "angular_scan":
            phtns = fine > self.nan_placebo  # remove line starts/ends
            coarse = coarse[phtns]
            fine = fine[phtns]

        h_all = np.bincount(coarse.astype("int"))
        x_all = np.arange(coarse.max() + 1)

        if pick_valid_bins_method == "auto":
            h_all = h_all[coarse.min() :]
            x_all = np.arange(coarse.min(), coarse.max() + 1)
            j = np.asarray(h_all > np.median(h_all) / 100).nonzero()[0]
            x = x_all[j]
            h = h_all[j]
        elif pick_valid_bins_method == "forced":
            x = valid_coarse_bins
            h = h_all[x]
        elif pick_valid_bins_method == "by example":
            x = exmpl_photon_data.coarse["bins"]
            h = h_all[x]
        elif pick_valid_bins_method == "interactive":
            raise NotImplementedError("'interactive' valid bins selection is not yet implemented.")
        else:
            raise ValueError(f"Unknown method '{pick_valid_bins_method}' for picking valid bins!")

        self.coarse = dict(bins=x, h=h)

        # rearranging the bins
        if sync_coarse_time_to is None:
            max_j = np.argmax(h)
        elif isinstance(sync_coarse_time_to, int):
            max_j = sync_coarse_time_to
        elif isinstance(sync_coarse_time_to, dict) and hasattr(sync_coarse_time_to, "tdc_calib"):
            max_j = sync_coarse_time_to.tdc_calib["max_j"]
        else:
            raise ValueError(
                "Syncing coarse time is possible to either a number or to an object that has the attribute 'tdc_calib'!"
            )

        jj = np.arange(len(h))
        j_shift = np.roll(jj, -max_j + 2)

        if pick_calib_bins_method == "auto":
            # pick data at more than 20ns delay from maximum
            j = np.where(j >= (calib_time_s * self.fpga_freq_hz + 2))[0]
            j_calib = j_shift[j]
            x_calib = x[j_calib]
        elif pick_calib_bins_method == "forced":
            x_calib = calibration_coarse_bins
        elif (
            pick_calib_bins_method == "by example"
            or pick_calib_bins_method == "External calibration"
        ):
            x_calib = exmpl_photon_data.tdc_calib["coarse_bins"]
        elif pick_valid_bins_method == "interactive":
            raise NotImplementedError(
                "'interactive' calibration bins selection is not yet implemented."
            )
        else:
            raise ValueError(
                f"Unknown method '{pick_calib_bins_method}' for picking calibration bins!"
            )

        if pick_calib_bins_method == "External calibration":
            self.tdc_calib = exmpl_photon_data.tdc_calib
            max_j = exmpl_photon_data.tdc_calib["max_j"]

            if "l_quarter_tdc" in self.tdc_calib:
                l_quarter_tdc = self.tdc_calib["l_quarter_tdc"]
                r_quarter_tdc = self.tdc_calib["r_quarter_tdc"]
        else:
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].semilogy(
                x_all, h_all, "-o", x, h, "-o", x[np.isin(x, x_calib)], h[np.isin(x, x_calib)], "-o"
            )
            plt.legend(["all hist", "valid bins", "calibration bins"], loc="lower right")
            plt.show()

            self.tdc_calib["coarse_bins"] = x_calib

            fine_calib = fine[np.isin(coarse, x_calib)]

            self.tdc_calib["fine_bins"] = np.arange(tdc_chain_length)
            # x_tdc_calib_nonzero, h_tdc_calib_nonzero = np.unique(fine_calib, return_counts=True) #histogram check also np.bincount
            # h_tdc_calib = np.zeros(x_tdc_calib.shape, dtype = h_tdc_calib_nonzero.dtype)
            # h_tdc_calib[x_tdc_calib_nonzero] = h_tdc_calib_nonzero
            h_tdc_calib = np.bincount(fine_calib, minlength=tdc_chain_length)

            # find effective range of TDC by, say, finding 10 zeros in a row
            cumusum_h_tdc_calib = np.cumsum(h_tdc_calib)
            diff_cumusum_h_tdc_calib = (
                cumusum_h_tdc_calib[(n_zeros_for_fine_bounds - 1) :]
                - cumusum_h_tdc_calib[: (-n_zeros_for_fine_bounds + 1)]
            )
            zeros_tdc = np.where(diff_cumusum_h_tdc_calib == 0)[0]
            # find middle of TDC
            mid_tdc = np.mean(fine_calib)

            # left TDC edge: zeros stretch closest to mid_tdc
            left_tdc = np.max(zeros_tdc[zeros_tdc < mid_tdc]) + n_zeros_for_fine_bounds - 1
            right_tdc = np.min(zeros_tdc[zeros_tdc > mid_tdc]) + 1

            l_quarter_tdc = round(left_tdc + (right_tdc - left_tdc) / 4)
            r_quarter_tdc = round(right_tdc - (right_tdc - left_tdc) / 4)

            self.tdc_calib["l_quarter_tdc"] = l_quarter_tdc
            self.tdc_calib["r_quarter_tdc"] = r_quarter_tdc

            # zero those out of TDC: I think h_tdc_calib[left_tdc] = 0, so does not actually need to be set to 0
            h_tdc_calib[:left_tdc] = 0
            h_tdc_calib[right_tdc:] = 0

            # h_tdc_calib = circshift(h_tdc_calib, [0 fine_shift]); seems no longer used. Test and remove fine_shift from parameter list
            t_calib = (
                (1 - np.cumsum(h_tdc_calib) / np.sum(h_tdc_calib)) / self.fpga_freq_hz * 1e9
            )  # invert and to ns

            self.tdc_calib["h"] = h_tdc_calib
            self.tdc_calib["t_calib"] = t_calib

            coarse_len = self.coarse["bins"].size

            t_weight = np.tile(self.tdc_calib["h"] / np.mean(self.tdc_calib["h"]), coarse_len)
            # t_weight = np.flip(t_weight)
            coarse_times = (
                np.tile(np.arange(coarse_len), [t_calib.size, 1]) / self.fpga_freq_hz * 1e9
            )
            delay_times = np.tile(t_calib, coarse_len) + coarse_times.flatten("F")
            Js = np.argsort(
                delay_times
            )  # initially delay times are piece wise inverted. After "flip" in line 323 there should be no need in sorting
            self.tdc_calib["delay_times"] = delay_times[Js]

            self.tdc_calib["t_weight"] = t_weight[Js]

            self.tdc_calib["max_j"] = max_j  # added on 14.01.17 for processing by example

            axs[0, 1].plot(self.tdc_calib["t_calib"], "-o")
            plt.legend(["TDC calibration"], loc="upper left")
            plt.show()

        # assign time delays to all photons
        self.tdc_calib["total_laser_pulses"] = 0
        lastCoarseBin = self.coarse["bins"][-1]
        max_j_m1 = max_j - 1
        if max_j_m1 == -1:
            max_j_m1 = lastCoarseBin

        delay_time = np.ndarray((0,), dtype=np.float64)
        # for i in self['Jgood']:
        for p in self.data:
            p.delay_time = np.ndarray(p.coarse.shape, dtype=np.float64)
            crs = np.minimum(p.coarse, lastCoarseBin) - self.coarse["bins"][max_j_m1]
            crs[crs < 0] = crs[crs < 0] + lastCoarseBin - self.coarse["bins"][0] + 1

            delta_coarse = p.coarse2 - p.coarse
            delta_coarse[delta_coarse == -3] = 1  # 2bit limitation

            # in the TDC midrange use "coarse" counter
            in_mid_tdc = (p.fine >= l_quarter_tdc) & (p.fine <= r_quarter_tdc)
            delta_coarse[in_mid_tdc] = 0

            # on the right of TDC use "coarse2" counter (no change in delta)
            # on the left of TDC use "coarse2" counter decremented by 1
            on_left_tdc = p.fine < l_quarter_tdc
            delta_coarse[on_left_tdc] = delta_coarse[on_left_tdc] - 1

            phtns = p.fine > self.nan_placebo  # self.nan_placebo are starts/ends of lines
            p.delay_time[phtns] = (
                self.tdc_calib["t_calib"][p.fine[phtns]]
                + (crs[phtns] + delta_coarse[phtns]).astype(np.float16) / self.fpga_freq_hz * 1e9
            )
            self.tdc_calib["total_laser_pulses"] += p.runtime[-1].astype(np.int64)
            p.delay_time[~phtns] = np.nan  # line ends/starts

            delay_time = np.append(delay_time, p.delay_time[phtns])

        bin_edges = np.arange(
            -time_bins_for_hist_ns / 2,
            np.max(delay_time) + time_bins_for_hist_ns,
            time_bins_for_hist_ns,
        )

        self.tdc_calib["t_hist"] = (bin_edges[:-1] + bin_edges[1:]) / 2
        k = np.digitize(self.tdc_calib["delay_times"], bin_edges)  # starts from 1

        self.tdc_calib["hist_weight"] = np.ndarray(self.tdc_calib["t_hist"].shape, dtype=np.float64)
        for i in range(len(self.tdc_calib["t_hist"])):
            j = k == (i + 1)
            self.tdc_calib["hist_weight"][i] = np.sum(self.tdc_calib["t_weight"][j])

        self.tdc_calib["all_hist"] = np.histogram(delay_time, bins=bin_edges)[0]
        self.tdc_calib["all_hist_norm"] = np.ndarray(
            self.tdc_calib["all_hist"].shape, dtype=np.float64
        )
        self.tdc_calib["error_all_hist_norm"] = np.ndarray(
            self.tdc_calib["all_hist"].shape, dtype=np.float64
        )
        nonzero = self.tdc_calib["hist_weight"] > 0
        self.tdc_calib["all_hist_norm"][nonzero] = (
            self.tdc_calib["all_hist"][nonzero]
            / self.tdc_calib["hist_weight"][nonzero]
            / self.tdc_calib["total_laser_pulses"]
        )
        self.tdc_calib["error_all_hist_norm"][nonzero] = (
            np.sqrt(self.tdc_calib["all_hist"][nonzero])
            / self.tdc_calib["hist_weight"][nonzero]
            / self.tdc_calib["total_laser_pulses"]
        )

        self.tdc_calib["all_hist_norm"][~nonzero] = np.nan
        self.tdc_calib["error_all_hist_norm"][~nonzero] = np.nan

        axs[1, 0].semilogy(self.tdc_calib["t_hist"], self.tdc_calib["all_hist_norm"], "-o")
        plt.legend(["Photon lifetime histogram"], loc="upper right")
        plt.show()

    def compare_lifetimes(
        self,
        normalization_type="Per Time",
        legend_label="",
        **kwargs  # dictionary, where keys are to be used as legends and values are objects that are
        # supposed to have their own compare_lifetimes method. But there can be other key/value
        # pairs related e.g. to plot fonts.
    ):

        # Possible normalization types: 'NO', 'Per Time'
        if normalization_type == "NO":
            H = self.tdc_calib["all_hist"] / self.tdc_calib["t_weight"]
        elif normalization_type == "Per Time":
            H = self.tdc_calib["all_hist_norm"]
        elif normalization_type == "By Sum":
            H = self.tdc_calib["all_hist_norm"] / np.sum(
                self.tdc_calib["all_hist_norm"][np.isfinite(self.tdc_calib["all_hist_norm"])]
            )
        else:
            raise Exception("Unknown normalization type")

        if "fontsize" not in kwargs:
            axisLabelFontSize = 18
        else:
            axisLabelFontSize = kwargs["fontsize"]
        #    kwargs.pop('fontsize')

        # plt.subplots(1, 1)
        plt.semilogy(self.tdc_calib["t_hist"], H, "-o", label=legend_label)
        print(legend_label)

        for key, value in kwargs.items():
            if hasattr(
                value, "compare_lifetimes"
            ):  # assume other objects that have TDCcalib structures
                value.compare_lifetimes(normalization_type, legend_label=key)

        # set(gca, 'FontSize', 16);
        plt.xlabel("life time (ns)", fontsize=axisLabelFontSize)
        plt.ylabel("freq", fontsize=axisLabelFontSize)
        plt.legend(loc="best")
        plt.show()

    def fit_lifetime_hist(
        self,
        fit_func="exponent_with_background_fit",
        x_field="t_hist",
        y_field="all_hist_norm",
        y_error_field="error_all_hist_norm",
        fit_param_estimate=np.array([0.1, 4, 0.001]),
        fit_range=(3.5, 30),
        x_scale="linear",
        y_scale="log",
        constant_param=[],
        no_plot=False,
        MaxIter=3,
        **kwargs,
    ):

        x = self.tdc_calib[x_field]
        y = self.tdc_calib[y_field]
        error_y = self.tdc_calib[y_error_field]

        isfiniteY = np.isfinite(y)
        x = x[isfiniteY]
        y = y[isfiniteY]
        error_y = error_y[isfiniteY]

        FP = fit_tools.curve_fit_lims(
            fit_func,
            fit_param_estimate,
            x,
            y,
            error_y,
            x_limits=fit_range,
            no_plot=no_plot,
            x_scale=x_scale,
            y_scale=y_scale,
        )

        if "fit_param" in self.tdc_calib:
            self.tdc_calib["fit_param"][FP["func_name"]] = FP
        else:
            self.tdc_calib["fit_param"] = dict()
            self.tdc_calib["fit_param"][FP["func_name"]] = FP


@nb.njit(cache=True)
def convert_angular_scan_to_image(runtime, laser_freq_hz, sample_freq_hz, ppl_tot, n_lines):
    """utility function for opening Angular Scans"""

    n_pix_tot = np.floor(runtime * sample_freq_hz / laser_freq_hz).astype(np.int64)
    # to which pixel photon belongs
    n_pix = np.mod(n_pix_tot, ppl_tot)
    line_num_tot = np.floor(n_pix_tot / ppl_tot)
    # one more line is for return to starting positon
    line_num = np.mod(line_num_tot, n_lines + 1).astype(np.int16)  # TESTESTEST was int32

    cnt = np.empty((n_lines + 1, ppl_tot), dtype=np.int32)
    bins = np.arange(-0.5, ppl_tot)
    for j in range(n_lines + 1):
        cnt_line, _ = np.histogram(n_pix[line_num == j], bins=bins)
        cnt[j, :] = cnt_line

    return cnt, n_pix_tot, n_pix, line_num


def fix_data_shift(cnt) -> int:
    """Doc."""

    def get_best_pix_shift(img: np.ndarray, min_shift, max_shift) -> int:
        """Doc."""

        score = np.empty(shape=(max_shift - min_shift), dtype=np.float64)
        pix_shifts = np.arange(min_shift, max_shift)
        for idx, pix_shift in enumerate(range(min_shift, max_shift)):
            rolled_img = np.roll(img, pix_shift).astype(np.float64)
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
    rolled_cnt = np.roll(cnt, pix_shift).astype(np.double)
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
        skifilt.median(img).astype(np.float), otsu_classes, nbins=n_bins
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
        prof = image[j]
        prof = prof[bw_mask[j] > 0]
        try:
            c, lags = auto_corr(prof)
        except ValueError:
            print(f"Auto correlation of line #{j} has failed. Skipping.", end=" ")
        else:
            c = c / prof.mean() ** 2 - 1
            c[0] -= 1 / prof.mean()  # subtracting shot noise, small stuff really
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
    lags = np.arange(c.size)

    return c, lags
