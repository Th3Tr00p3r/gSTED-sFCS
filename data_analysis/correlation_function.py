"""Data organization and manipulation."""

import glob
import logging
import os
from collections import deque

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy import ndimage, stats
from skimage import filters as skifilt
from skimage import morphology

from data_analysis import fit_tools
from data_analysis.file_loading_utilities import load_file_dict
from data_analysis.photon_data import PhotonData
from data_analysis.software_correlator import CorrelatorType, SoftwareCorrelator
from utilities.helper import div_ceil, force_aspect, sort_file_paths_by_file_number


class CorrFuncData:
    """Doc."""

    def average_correlation(
        self,
        rejection=2,
        reject_n_worst=None,
        norm_range=(1e-3, 2e-3),
        delete_list=[],
        no_plot=True,
        use_numba=True,
    ):
        """Doc."""

        def calc_weighted_avg(cf_cr, weights):
            """Note: enable 'use_numba' for speed-up"""

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

        if use_numba:
            func = nb.njit(calc_weighted_avg, cache=True)
        else:
            func = calc_weighted_avg
        self.average_cf_cr, self.error_cf_cr = func(
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
        if x_scale == "log":
            x = x[1:]  # remove zero point data
            y = y[1:]
        plt.plot(x, y, "o")  # skip 0 lag time
        plt.xlabel(x_field)
        plt.ylabel(y_field)
        plt.gca().set_x_scale(x_scale)
        plt.gca().set_y_scale(y_scale)
        plt.show()

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

        # self.plot_correlation_function()
        if not hasattr(self, "fit_param"):
            self.fit_param = dict()

        self.fit_param[FP["fit_func"]] = FP


class CorrFuncTDC(CorrFuncData):
    """Doc."""

    def __init__(self):
        # list to hold the data of each file
        self.data = []
        # dictionary for TDC calibration
        self.tdc_calib = dict()

    def read_fpga_data(
        self,
        file_template_path,
        fix_shift=False,
        roi_selection="auto",
        no_plot=False,
    ):
        """Doc."""

        print("\nLoading FPGA data from hard drive:")

        # TODO: test what happens if 'file_template_path' is wrong or None
        file_paths = sort_file_paths_by_file_number(glob.glob(file_template_path))
        _, self.template = os.path.split(file_template_path)

        n_files = len(file_paths)

        for idx, file_path in enumerate(file_paths):
            print(f"Loading file No. {idx+1}/{n_files}: '{file_path}'...", end=" ")

            # load file
            file_dict = load_file_dict(file_path)

            print("Done.")

            if idx == 0:
                self.after_pulse_param = file_dict["system_info"]["after_pulse_param"]

            full_data = file_dict["full_data"]

            print("Converting raw data to photons...", end=" ")
            p = PhotonData()
            p.convert_fpga_data_to_photons(
                np.array(full_data["data"]).astype("B"), version=full_data["version"], verbose=True
            )
            print("Done.")

            p.file_num = idx + 1

            self.laser_freq_hz = full_data["laser_freq_mhz"] * 1e6
            self.fpga_freq_hz = full_data["fpga_freq_mhz"] * 1e6

            if full_data.get("circle_speed_um_s"):
                # circular scan
                self.type = "circular_scan"
                self.v_um_ms = full_data["circle_speed_um_s"] / 1000  # to um/ms
                raise NotImplementedError("Circular scan analysis not yet implemented...")

            elif full_data.get("angular_scan_settings"):
                # angular scan
                self.type = "angular_scan"
                if idx == 0:
                    # not assigned yet - this way assignment happens once
                    angular_scan_settings = full_data["angular_scan_settings"]
                    linear_part = np.array(angular_scan_settings["linear_part"], dtype=np.int32)
                    self.v_um_ms = angular_scan_settings["actual_speed_um_s"] / 1000
                    sample_freq_hz = angular_scan_settings["sample_freq_hz"]
                    ppl_tot = angular_scan_settings["points_per_line_total"]
                    n_lines = angular_scan_settings["n_lines"]
                    self.angular_scan_settings = angular_scan_settings
                    self.line_end_adder = 1000

                print("Converting angular scan to image...", end=" ")

                runtime = p.runtime
                cnt, n_pix_tot, n_pix, line_num = convert_angular_scan_to_image(
                    runtime, self.laser_freq_hz, sample_freq_hz, ppl_tot, n_lines
                )

                if fix_shift:
                    pix_shift = fix_data_shift(cnt)
                    runtime = p.runtime + pix_shift * round(self.laser_freq_hz / sample_freq_hz)
                    cnt, n_pix_tot, n_pix, line_num = convert_angular_scan_to_image(
                        runtime, self.laser_freq_hz, sample_freq_hz, ppl_tot, n_lines
                    )
                    print(f"Fixed line shift: {pix_shift} pixels. Done.")
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
                        logging.warning("Thresholding failed, skipping file.")
                        continue
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
                            [
                                (-row_idx - self.line_end_adder)
                                for elem in range(len(line_stops_new))
                            ]
                        )
                        line_starts.extend(line_starts_new)
                        line_stops.extend(line_stops_new)

                try:
                    # repeat first point to close the polygon
                    roi["row"].append(roi["row"][0])
                    roi["col"].append(roi["col"][0])
                except IndexError:
                    print("ROI is empty (need to figure out the cause). Skipping file.")
                    continue

                # convert lists/deques to numpy arrays
                roi = {key: np.array(val) for key, val in roi.items()}
                line_start_lables = np.array(line_start_lables)
                line_stop_labels = np.array(line_stop_labels)
                line_starts = np.array(line_starts).astype(np.int64)
                line_stops = np.array(line_stops).astype(np.int64)

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
                p.coarse = np.hstack(
                    (
                        np.full(runtime_line_starts.shape, np.nan),
                        np.full(runtime_line_stops.shape, np.nan),
                        p.coarse,
                    )
                )[sorted_idxs]
                p.coarse2 = np.hstack(
                    (
                        np.full(runtime_line_starts.shape, np.nan),
                        np.full(runtime_line_stops.shape, np.nan),
                        p.coarse2,
                    )
                )[sorted_idxs]
                p.fine = np.hstack(
                    (
                        np.full(runtime_line_starts.shape, np.nan),
                        np.full(runtime_line_stops.shape, np.nan),
                        p.fine,
                    )
                )[sorted_idxs]

                p.image = cnt
                p.roi = roi

                # plotting of scan image and ROI
                if not no_plot:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.set_title(f"file No. {p.file_num} of {self.template}")
                    ax.set_xlabel("Pixel Number")
                    ax.set_ylabel("Line Number")
                    ax.imshow(cnt)
                    ax.plot(roi["col"], roi["row"], color="white")  # plot the ROI
                    force_aspect(ax, aspect=1)
                    fig.show()

                # reverse rows again
                bw[1::2, :] = np.flip(bw[1::2, :], 1)
                p.bw_mask = bw

                # get image line correlation to subtract trends
                img = p.image * p.bw_mask
                # lineNos = find(sum(p.bw_mask, 2));

                p.image_line_corr = line_correlations(img, p.bw_mask, roi, sample_freq_hz)

            else:
                # static FCS - nothing else needs to be done
                self.type = "static"
                pass

            p.file_path = file_path
            self.data.append(p)

            print(f"Finished processing file No. {idx+1}\n")

        if full_data.get("duration_s") is not None:
            self.duration_min = full_data["duration_s"] / 60
        else:
            # calculate duration if not supplied
            self.duration_min = (
                np.mean([np.diff(p.runtime).sum() for p in self.data]) / self.laser_freq_hz / 60
            )
            print(f"Calculating duration (not supplied): {self.duration_min:.1f} min\n")

        print(f"Finished loading FPGA data ({len(self.data)}/{n_files} files used).\n")

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

        self.requested_duration = run_duration
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
            time_stamps = np.diff(p.runtime)
            # for exponential distribution MEDIAN and MAD are the same, but for
            # biexponential MAD seems more sensitive
            mu = np.maximum(
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

            for se_idx, (se_start, se_end) in enumerate(
                p.all_section_edges
            ):  # TODO: try to split to se_start, se_end??

                # split into segments of approx time of run_duration
                segment_time = (p.runtime[se_end] - p.runtime[se_start]) / self.laser_freq_hz
                if segment_time < min_time_frac * run_duration:
                    if verbose:
                        print(
                            f"Duration of segment No. {se_idx} of file {p.fname} is {segment_time}s: too short. Skipping segment..."
                        )
                    self.total_duration_skipped += segment_time
                    continue

                n_splits = div_ceil(segment_time, run_duration)
                splits = np.linspace(0, (se_end - se_start), n_splits + 1, dtype=np.int32)
                ts = time_stamps[se_start:se_end]

                for k in range(n_splits):

                    ts_split = ts[splits[k] : splits[k + 1]].astype(np.int32)
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
                            # y = beta(1)*exp(-beta(2)*t) + beta(3)*exp(-beta(4)*t) + beta(5)*exp(-beta(6)*t);
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
        if verbose and self.total_duration_skipped:
            print(
                f"Done.\n{self.total_duration_skipped:.2f} s skipped out of {self.total_duration:.2f} s. Done."
            )

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

        print(f"Correlating angular scan data ({self.template}):", end=" ")

        for p in self.data:
            print(f"({p.file_num})", end=" ")

            time_stamps = np.diff(p.runtime)
            line_num = p.line_num
            min_line, max_line = line_num[line_num > 0].min(), line_num.max()
            for line_idx, j in enumerate(range(min_line, max_line + 1)):
                valid = (line_num == j).astype(np.int32)
                valid[line_num == -j] = -1
                valid[line_num == -j - self.line_end_adder] = -2
                # both photons separated by time-stamp should belong to the line
                valid = valid[1:]

                # remove photons from wrong lines
                timest = time_stamps[valid != 0].astype(np.int32)
                valid = valid[valid != 0]
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


@nb.njit(cache=True)
def convert_angular_scan_to_image(runtime, laser_freq_hz, sample_freq_hz, ppl_tot, n_lines):
    """utility function for opening Angular Scans"""

    n_pix_tot = np.floor(runtime * sample_freq_hz / laser_freq_hz).astype(np.int64)
    # to which pixel photon belongs
    n_pix = np.mod(n_pix_tot, ppl_tot)
    line_num_tot = np.floor(n_pix_tot / ppl_tot)
    # one more line is for return to starting positon
    line_num = np.mod(line_num_tot, n_lines + 1).astype(np.int32)
    cnt = np.empty((n_lines + 1, ppl_tot), dtype=np.int32)

    bins = np.arange(-0.5, ppl_tot)
    for j in range(n_lines + 1):
        belong_to_line = line_num == j
        cnt_line, _ = np.histogram(n_pix[belong_to_line], bins=bins)
        cnt[j, :] = cnt_line

    return cnt, n_pix_tot, n_pix, line_num


def fix_data_shift(cnt) -> int:
    """Doc."""

    _, width = cnt.shape

    score = []
    pix_shifts = []
    min_pix_shift = -round(width / 2)
    max_pix_shift = min_pix_shift + width + 1
    for pix_shift in range(min_pix_shift, max_pix_shift):
        cnt2 = np.roll(cnt, pix_shift).astype(np.double)
        diff_cnt2 = (cnt2[:-1:2, :] - np.flip(cnt2[1::2, :], 1)) ** 2
        score.append(diff_cnt2.sum())
        pix_shifts.append(pix_shift)

    # verify not using local minimum by checking if there's a shift
    # yielding a significantly brighter image center for the 10 highest scoring shifts
    center_sum = 0
    for idx in np.argsort(score)[:10]:
        new_pix_shift = pix_shifts[idx]
        rolled_cnt = np.roll(cnt, new_pix_shift)
        new_center_sum = rolled_cnt[:, int(width * 0.25) : int(width * 0.75)].sum()
        if new_center_sum > center_sum * 1.5:
            center_sum = new_center_sum
            pix_shift = pix_shifts[idx]

    return pix_shift


# NOTE: 'disk_radius' was 3
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
            c, lags = x_corr(prof, prof)
        except ValueError:
            logging.warning(f"Auto correlation of line #{j} has failed. Skipping.")
        else:
            c = c / prof.mean() ** 2 - 1
            c[0] -= 1 / prof.mean()  # subtracting shot noise, small stuff really
            image_line_corr.append(
                {
                    "lag": lags * 1000 / sampling_freq,  # in ms
                    "corrfunc": c,
                }
            )  # c/mean(prof).^2-1;
    return image_line_corr


def x_corr(a, b):
    """Does correlation similar to Matlab xcorr, cuts positive lags, normalizes properly"""

    if a.size != b.size:
        raise ValueError("For unequal lengths of a, b the meaning of lags is not clear!")
    c = np.correlate(a, b, mode="full")
    c = c[c.size // 2 :]
    c = c / np.arange(c.size, 0, -1)
    lags = np.arange(c.size)

    return c, lags
