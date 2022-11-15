"""Data Processing."""

from collections import deque
from contextlib import suppress
from copy import copy
from dataclasses import dataclass
from itertools import count as infinite_range
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import scipy
import skimage

from utilities.display import Plotter
from utilities.fit_tools import FIT_NAME_DICT, FitError, FitParams, curve_fit_lims
from utilities.helper import Gate, Limits, div_ceil, nan_helper, xcorr


class CircularScanDataMixin:
    """Doc."""

    def _sum_scan_circles(self, pulse_runtime, laser_freq_hz, ao_sampling_freq_hz, circle_freq_hz):
        """Doc."""

        # calculate the number of samples obtained at each photon arrival, since beginning of file
        sample_runtime = pulse_runtime * ao_sampling_freq_hz // laser_freq_hz
        # calculate to which pixel each photon belongs (possibly many samples per pixel)
        samples_per_circle = int(ao_sampling_freq_hz / circle_freq_hz)
        pixel_num = sample_runtime % samples_per_circle

        # build the 'line image'
        bins = np.arange(-0.5, samples_per_circle)
        img, _ = np.histogram(pixel_num, bins=bins)

        return img, samples_per_circle


class AngularScanDataMixin:
    """Doc."""

    NAN_PLACEBO = -100  # marks starts/ends of lines
    LINE_END_ADDER = 1000

    def _convert_angular_scan_to_image(
        self, pulse_runtime, laser_freq_hz, ao_sampling_freq_hz, samples_per_line, n_lines
    ):
        """Converts angular scan pulse_runtime into an image."""

        # calculate the number of samples obtained at each photon arrival, since beginning of file
        sample_runtime = pulse_runtime * ao_sampling_freq_hz // laser_freq_hz
        # calculate to which pixel each photon belongs (possibly many samples per pixel)
        pixel_num = sample_runtime % samples_per_line
        # calculate to which line each photon belongs (global, not considering going back to the first line)
        line_num_tot = sample_runtime // samples_per_line
        # calculate to which line each photon belongs (extra line is for returning to starting position)
        line_num = (line_num_tot % (n_lines + 1)).astype(np.int16)

        # build the image
        img = np.empty((n_lines + 1, samples_per_line), dtype=np.uint16)
        bins = np.arange(-0.5, samples_per_line)
        for j in range(n_lines + 1):
            img[j, :], _ = np.histogram(pixel_num[line_num == j], bins=bins)

        return img, sample_runtime, pixel_num, line_num

    def _get_data_shift(self, cnt: np.ndarray) -> int:
        """Doc."""

        def get_best_pix_shift(img: np.ndarray, min_shift, max_shift) -> int:
            """Doc."""

            score = np.empty(shape=(max_shift - min_shift), dtype=np.uint32)
            pix_shifts = np.arange(min_shift, max_shift)
            for idx, pix_shift in enumerate(range(min_shift, max_shift)):
                rolled_img = np.roll(img, pix_shift).astype(np.int32)
                score[idx] = (abs(rolled_img[:-1:2, :] - np.fliplr(rolled_img[1::2, :]))).sum()
            return pix_shifts[score.argmin()]

        height, width = cnt.shape

        # replacing outliers with median value
        med = np.median(cnt)
        cnt[cnt > med * 1.5] = med

        # limit initial attempt to the width of the image
        min_pix_shift = -round(width / 2)
        max_pix_shift = min_pix_shift + width + 1
        pix_shift = get_best_pix_shift(cnt, min_pix_shift, max_pix_shift)

        # Test if not stuck in local minimum. Either 'outer_half_sum > inner_half_sum'
        # Or if the 'return row' (the empty one) is not at the bottom after shift
        rolled_cnt = np.roll(cnt, pix_shift)
        inner_half_sum = rolled_cnt[:, int(width * 0.25) : int(width * 0.75)].sum()
        outer_half_sum = rolled_cnt.sum() - inner_half_sum
        return_row_idx = rolled_cnt.sum(axis=1).argmin()

        # in case initial attempt fails, limit shift to the flattened size of the image
        if (outer_half_sum > inner_half_sum) or return_row_idx != height - 1:
            if return_row_idx != height - 1:
                print("Data is heavily shifted, check it out!", end=" ")
            min_pix_shift = -round(cnt.size / 2)
            max_pix_shift = min_pix_shift + cnt.size + 1
            pix_shift = get_best_pix_shift(cnt, min_pix_shift, max_pix_shift)

        return pix_shift

    def _threshold_and_smooth(self, img, otsu_classes=4, n_bins=256, disk_radius=2) -> np.ndarray:
        """Doc."""

        thresh = skimage.filters.threshold_multiotsu(
            skimage.filters.median(img).astype(np.float32), otsu_classes, nbins=n_bins
        )  # minor filtering of outliers
        cnt_dig = np.digitize(img, bins=thresh)
        plateau_lvl = np.median(img[cnt_dig == (otsu_classes - 1)])
        std_plateau = scipy.stats.median_absolute_deviation(img[cnt_dig == (otsu_classes - 1)])
        dev_cnt = img - plateau_lvl
        bw = dev_cnt > -std_plateau
        bw = scipy.ndimage.binary_fill_holes(bw)
        disk_open = skimage.morphology.disk(radius=disk_radius)
        bw = skimage.morphology.opening(bw, footprint=disk_open)
        return bw

    def _bg_line_correlations(
        self,
        image1: np.ndarray,
        bw_mask: np.ndarray,
        line_limits: Limits,
        sampling_freq_Hz,
        image2: np.ndarray = None,
    ) -> list:
        """Returns a list of auto-correlations of the lines of an image."""

        is_doing_xcorr = image2 is not None

        line_corr_list = []
        for j in line_limits.as_range():
            line1 = image1[j][bw_mask[j] > 0].astype(np.float64)
            if is_doing_xcorr:
                line2 = image2[j][bw_mask[j] > 0].astype(np.float64)
            try:
                if not is_doing_xcorr:
                    c, lags = xcorr(line1, line1)
                else:
                    c, lags = xcorr(line1, line2)
            except ValueError:
                print(f"Correlation of line No.{j} has failed. Skipping.", end=" ")
            else:
                if not is_doing_xcorr:  # Autocorrelation
                    c = c / line1.mean() ** 2 - 1
                    c[0] -= 1 / line1.mean()  # subtracting shot noise, small stuff really
                else:  # Cross-Correlation
                    c = c / (line1.mean() * line2.mean()) - 1
                    c[0] -= 1 / np.sqrt(
                        line1.mean() * line2.mean()
                    )  # subtracting shot noise, small stuff really

                line_corr_list.append(
                    {
                        "lag": lags * 1e3 / sampling_freq_Hz,  # in ms
                        "corrfunc": c,
                    }
                )
        return line_corr_list


class TDCPhotonData:
    """Holds a single file's worth of processed, TDC-based, temporal photon data"""

    # TODO: separate file data from photon data?
    # TODO: strive to make this a dataclass

    # general
    version: int
    section_runtime_edges: list
    coarse: np.ndarray
    coarse2: np.ndarray
    fine: np.ndarray
    pulse_runtime: np.ndarray
    size_estimate_mb: float
    duration_s: float
    skipped_duration: float
    delay_time: np.ndarray

    avg_cnt_rate_khz: float
    image: np.ndarray
    bg_line_corr: List[Dict[str, Any]]

    # continuous scan
    all_section_edges: np.ndarray

    # circular scan
    ao_sampling_freq_hz: float
    circle_freq_hz: float

    # line scan
    line_num: np.ndarray
    line_limits: Limits
    samples_per_line: int
    n_lines: int
    roi: Dict[str, deque]
    bw_mask: np.ndarray
    NAN_PLACEBO: int
    LINE_END_ADDER: int

    def __init__(self, **kwargs):
        for attr, val in kwargs.items():
            setattr(self, attr, val)

    def get_xcorr_splits_dict(self, scan_type: str, *args, **kwargs):
        """Doc."""

        if scan_type in {"static", "circle"}:
            return self._get_continuous_xcorr_splits_dict(scan_type, *args, **kwargs)
        elif scan_type == "angular":
            return self._get_line_xcorr_splits_dict(scan_type, *args, **kwargs)

    def _get_line_xcorr_splits_dict(
        self,
        scan_type: str,
        xcorr_types: List[str],
        *args,
        gate1_ns=Gate(),
        gate2_ns=Gate(),
        **kwargs,
    ):
        """Splits are all photons belonging to each scan line."""

        # NaNs mark line starts/ends (used to create valid = -1/-2 needed in C code)
        nan_idxs = np.isnan(self.delay_time)
        if "A" in "".join(xcorr_types):
            gate1_idxs = gate1_ns.valid_indices(self.delay_time)
            valid_idxs1 = gate1_idxs | nan_idxs
            dt1 = self.delay_time[valid_idxs1]
            pulse_runtime1 = self.pulse_runtime[valid_idxs1]
            ts1 = np.hstack(([0], np.diff(pulse_runtime1)))
            dt_ts1 = np.vstack((dt1, ts1))
            line_num1 = self.line_num[valid_idxs1]
        if "B" in "".join(xcorr_types):
            gate2_idxs = gate2_ns.valid_indices(self.delay_time)
            valid_idxs2 = gate2_idxs | nan_idxs
            dt2 = self.delay_time[valid_idxs2]
            pulse_runtime2 = self.pulse_runtime[valid_idxs2]
            ts2 = np.hstack(([0], np.diff(pulse_runtime2)))
            dt_ts2 = np.vstack((dt2, ts2))
            line_num2 = self.line_num[valid_idxs2]
        if "AB" in xcorr_types or "BA" in xcorr_types:
            # NOTE: # gate2 is first in line to match how software correlator C code written
            dt_ts12 = np.vstack((self.delay_time, self.pulse_runtime, valid_idxs2, valid_idxs1))[
                :, valid_idxs1 | valid_idxs2
            ]
            dt_ts12[0] = np.hstack(([0], np.diff(dt_ts12[0])))
            line_num12 = self.line_num[valid_idxs1 | valid_idxs2]

        dt_ts_splits_dict: Dict[str, List[np.ndarray]] = {xx: [] for xx in xcorr_types}
        for j in self.line_limits.as_range():
            for xx in xcorr_types:
                if xx == "AA":
                    dt_ts_splits_dict[xx].append(
                        self._prepare_correlator_input(dt_ts1, j, scan_type, line_num=line_num1)
                    )
                if xx == "BB":
                    dt_ts_splits_dict[xx].append(
                        self._prepare_correlator_input(dt_ts2, j, scan_type, line_num=line_num2)
                    )
                if xx == "AB":
                    splitsAB = self._prepare_correlator_input(
                        dt_ts12, j, scan_type, line_num=line_num12
                    )
                    dt_ts_splits_dict[xx].append(splitsAB)
                if xx == "BA":
                    if "AB" in xcorr_types:
                        dt_ts_splits_dict[xx].append(splitsAB[[0, 1, 3, 2, 4], :])
                    else:
                        dt_ts21 = dt_ts12[[0, 1, 3, 2, 4], :]
                        dt_ts_splits_dict[xx].append(
                            self._prepare_correlator_input(
                                dt_ts21, j, scan_type, line_num=line_num12
                            )
                        )

        return dt_ts_splits_dict

    def _get_continuous_xcorr_splits_dict(
        self,
        scan_type: str,
        xcorr_types: List[str],
        laser_freq_hz: int,
        gate1_ns=Gate(),
        gate2_ns=Gate(),
        n_splits_requested=10,
        **kwargs,
    ):
        """Continuous scan/static measurement - splits are arbitrarily cut along the measurement"""

        # TODO: split duration (in bytes! not time) should be decided upon according to how well the correlator performs with said split size. Currently it is arbitrarily decided by 'n_splits_requested' which causes inconsistent processing times for each split
        split_duration = self.duration_s / n_splits_requested
        for se_idx, (se_start, se_end) in enumerate(self.all_section_edges):
            # split into sections of approx time of run_duration
            section_time = (
                self.pulse_runtime[se_end] - self.pulse_runtime[se_start]
            ) / laser_freq_hz

            section_pulse_runtime = self.pulse_runtime[se_start : se_end + 1]
            section_delay_time = self.delay_time[se_start : se_end + 1]

            # split the data into parts A/B according to gates
            # TODO: in order to make this more general and use it with both auto and cross correlations, the function should optionally accept 2 seperate data,
            # i.e. the split according to gates should be performed externally and beforehand.
            if "A" in "".join(xcorr_types):
                gate1_idxs = gate1_ns.valid_indices(section_delay_time)
                section_prt1 = section_pulse_runtime[gate1_idxs]
                section_ts1 = np.hstack(([0], np.diff(section_prt1)))
                section_dt1 = section_delay_time[gate1_idxs]
                section_dt_ts1 = np.vstack((section_dt1, section_ts1))
            if "B" in "".join(xcorr_types):
                gate2_idxs = gate2_ns.valid_indices(section_delay_time)
                section_prt2 = section_pulse_runtime[gate2_idxs]
                section_ts2 = np.hstack(([0], np.diff(section_prt2)))
                section_dt2 = section_delay_time[gate2_idxs]
                section_dt_ts2 = np.vstack((section_dt2, section_ts2))
            if "AB" in xcorr_types or "BA" in xcorr_types:
                section_ts12 = np.hstack(([0], np.diff(section_pulse_runtime)))
                section_dt_ts12 = np.vstack(
                    (section_delay_time, section_ts12, gate2_idxs, gate1_idxs)
                )[:, gate1_idxs | gate2_idxs]

            dt_ts_splits_dict: Dict[str, List[np.ndarray]] = {xx: [] for xx in xcorr_types}
            for j in range(n_splits := div_ceil(section_time, split_duration)):
                for xx in xcorr_types:
                    if xx == "AA":
                        dt_ts_splits_dict[xx].append(
                            self._prepare_correlator_input(
                                section_dt_ts1, j, scan_type, n_splits=n_splits
                            )
                        )
                    if xx == "BB":
                        dt_ts_splits_dict[xx].append(
                            self._prepare_correlator_input(
                                section_dt_ts2, j, scan_type, n_splits=n_splits
                            )
                        )
                    if xx == "AB":
                        splitsAB = self._prepare_correlator_input(
                            section_dt_ts12,
                            j,
                            scan_type,
                            n_splits=n_splits,
                        )
                        dt_ts_splits_dict[xx].append(splitsAB)
                    if xx == "BA":
                        if "AB" in xcorr_types:
                            dt_ts_splits_dict[xx].append(splitsAB[[0, 1, 3, 2], :])
                        else:
                            section_dt_ts21 = section_dt_ts12[[0, 1, 3, 2], :]
                            dt_ts_splits_dict[xx].append(
                                self._prepare_correlator_input(
                                    section_dt_ts21,
                                    j,
                                    scan_type,
                                    n_splits=n_splits,
                                )
                            )

        return dt_ts_splits_dict

    def _prepare_correlator_input(
        self, dt_ts_in, idx, scan_type, line_num=None, n_splits=None
    ) -> np.ndarray:
        """Doc."""

        if scan_type in {"static", "circle"}:
            splits = np.linspace(0, dt_ts_in.shape[1], n_splits + 1, dtype=np.int32)
            return dt_ts_in[:, splits[idx] : splits[idx + 1]]

        elif scan_type == "angular":
            valid = (line_num == idx).astype(np.int8)
            valid[line_num == -idx] = -1
            valid[line_num == -idx - self.LINE_END_ADDER] = -2

            #  remove photons from wrong lines
            dt_ts_out = dt_ts_in[:, valid != 0]
            valid = valid[valid != 0]

            if not valid.any():
                return np.vstack(([], []))

            # the first photon in line measures the time from line start and the line end (-2) finishes the duration of the line
            # check that we start with the line beginning and not its end
            if valid[0] != -1:
                # remove photons till the first found beginning
                j_start = np.where(valid == -1)[0]

                if len(j_start) > 0:
                    dt_ts_out = dt_ts_out[:, j_start[0] :]
                    valid = valid[j_start[0] :]

            # check that we stop with the line ending and not its beginning
            if valid[-1] != -2:
                # remove photons after the last found ending
                j_end = np.where(valid == -2)[0]

                if len(j_end) > 0:
                    *_, j_end_last = j_end
                    dt_ts_out = dt_ts_out[:, : j_end_last + 1]
                    valid = valid[: j_end_last + 1]

            return np.vstack((dt_ts_out, valid))


@dataclass
class AfterulsingFilter:
    """Doc."""

    t_hist: np.ndarray
    all_hist_norm: np.ndarray
    baseline: float
    I_j: np.ndarray
    norm_factor: float
    M: np.ndarray
    filter: np.ndarray

    def plot(self, parent_ax=None, **plot_kwargs):
        """Doc."""

        n = len(self.I_j)

        with Plotter(
            parent_ax=parent_ax,
            subplots=(1, 2),
            **plot_kwargs,
        ) as axes:
            axes[0].set_title("Filter Ingredients")
            axes[0].set_yscale("log")
            axes[0].plot(
                self.t_hist[:n], self.all_hist_norm / self.norm_factor, label="norm. raw histogram"
            )
            axes[0].plot(self.t_hist[:n], self.I_j / self.norm_factor, label="norm. I_j (fit)")
            axes[0].plot(
                self.t_hist[:n],
                self.baseline / self.norm_factor * np.ones(self.t_hist[:n].shape),
                label="norm. baseline",
            )
            axes[0].plot(
                self.t_hist[:n], self.M.T[0], label="M_j1 (ideal fluorescence decay curve)"
            )
            axes[0].plot(
                self.t_hist[:n], self.M.T[1], label="M_j2 (ideal afterpulsing 'decay' curve)"
            )
            axes[0].legend()
            axes[0].set_ylim(self.baseline / self.norm_factor / 10, None)

            axes[1].set_title("Filter")
            axes[1].plot(self.t_hist, self.filter.T)
            axes[1].plot(self.t_hist, self.filter.sum(axis=0))
            axes[1].legend(["F_1j (signal)", "F_2j (afterpulsing)", "F.sum(axis=0)"])


@dataclass
class TDCCalibration:
    """Doc."""

    x_all: np.ndarray
    h_all: np.ndarray
    coarse_bins: np.ndarray
    h: np.ndarray
    max_j: int
    coarse_calib_bins: Any
    fine_bins: Any
    l_quarter_tdc: Any
    r_quarter_tdc: Any
    t_calib: Any
    hist_weight: np.ndarray
    delay_times: Any
    total_laser_pulses: int
    h_tdc_calib: Any
    t_hist: Any
    all_hist: Any
    all_hist_norm: Any
    error_all_hist_norm: Any
    t_weight: Any

    def plot(self, parent_axes=None, **plot_kwargs) -> None:
        """Doc."""

        try:
            x_all = self.x_all
            h_all = self.h_all
            coase_bins = self.coarse_bins
            h = self.h
            coarse_calib_bins = self.coarse_calib_bins
            t_calib = self.t_calib
            t_hist = self.t_hist
            all_hist_norm = self.all_hist_norm
        except AttributeError:
            return

        with Plotter(
            parent_ax=parent_axes,
            subplots=(2, 2),
            **plot_kwargs,
        ) as axes:
            axes[0, 0].semilogy(x_all, h_all, "-o", label="All Hist")
            axes[0, 0].semilogy(coase_bins, h, "o", label="Valid Bins")
            axes[0, 0].semilogy(
                coase_bins[np.isin(coase_bins, coarse_calib_bins)],
                h[np.isin(coase_bins, coarse_calib_bins)],
                "o",
                markersize=4,
                label="Calibration Bins",
            )
            axes[0, 0].legend()

            axes[0, 1].plot(t_calib, "-o", label="TDC Calibration")
            axes[0, 1].legend()

            axes[1, 0].semilogy(t_hist, all_hist_norm, "-o", label="Photon Lifetime Histogram")
            axes[1, 0].legend()

    def fit_lifetime_hist(
        self,
        x_field="t_hist",
        y_field="all_hist_norm",
        y_error_field="error_all_hist_norm",
        fit_param_estimate=[0.1, 4, 0.001],
        fit_range=(3.5, 30),
        x_scale="linear",
        y_scale="log",
        should_plot=False,
    ) -> FitParams:
        """Doc."""

        is_finite_y = np.isfinite(getattr(self, y_field))

        return curve_fit_lims(
            FIT_NAME_DICT["exponent_with_background_fit"],
            fit_param_estimate,
            xs=getattr(self, x_field)[is_finite_y],
            ys=getattr(self, y_field)[is_finite_y],
            ys_errors=getattr(self, y_error_field)[is_finite_y],
            x_limits=Limits(fit_range),
            should_plot=should_plot,
            plot_kwargs=dict(x_scale=x_scale, y_scale=y_scale),
        )

    def calculate_afterpulsing_filter(
        self,
        gate_ns,
        should_plot=False,
        **kwargs,
    ) -> np.ndarray:
        """Doc."""

        # make copies so that originals are preserved
        all_hist_norm = copy(self.all_hist_norm)
        t_hist = copy(self.t_hist)

        # crop-out the gated part of the histogram
        if gate_ns:
            in_gate_idxs = gate_ns.valid_indices(t_hist)
            t_hist = t_hist[in_gate_idxs]
            all_hist_norm = all_hist_norm[in_gate_idxs]
            lower_idx = round(gate_ns.lower * 10)
            F_pad_before = np.full((2, lower_idx), np.nan)
            try:
                upper_idx = round(gate_ns.upper * 10)
                F_pad_after = np.full((2, len(self.t_hist) - upper_idx - 1), np.nan)
            except (OverflowError, ValueError):
                # OverflowError: gate_ns.upper == np.inf
                # ValueError: len(self.t_hist) == upper_idx
                F_pad_after = np.array([[], []])
        else:
            lower_idx = 0

        # interpolate over NaNs
        nans, x = nan_helper(all_hist_norm)  # get nans and a way to interpolate over them later
        #        print("nans: ", nans.sum()) # TESTESTEST - why always 23?
        all_hist_norm[nans] = np.interp(x(nans), x(~nans), all_hist_norm[~nans])

        # Use exponential fit to get the underlying exponentioal decay and background (Smoothing is what's really essential here)
        # TODO: for STED measurement I will need a different fit - decay is no longer purely exponential
        fit_lower_x_limit = t_hist[np.argmax(all_hist_norm)] + 2
        try:
            fp = curve_fit_lims(
                (fit_func := FIT_NAME_DICT["exponent_with_background_fit"]),
                [1e-2, 4, 1e-4],
                xs=t_hist,
                ys=all_hist_norm,
                x_limits=Limits(fit_lower_x_limit, np.inf),
                plot_kwargs=dict(y_scale="log"),
            )
        except FitError as exc:
            raise ValueError(f"Gate {gate_ns} is too narrow! [{exc}]")

        baseline = fp.beta["bg"]
        fitted_all_hist_norm = fit_func(t_hist.astype(np.float64), *fp.beta.values())

        # normalization factor
        norm_factor = (fitted_all_hist_norm - baseline).sum()

        # define matrices and calculate F
        M_j1 = (fitted_all_hist_norm - baseline) / norm_factor  # p1
        M_j2 = 1 / len(t_hist) * np.ones(t_hist.shape)  # p2
        M = np.vstack((M_j1, M_j2)).T

        I_j = fitted_all_hist_norm  # / norm_factor
        I = np.diag(I_j)  # NOQA E741
        inv_I = np.linalg.pinv(I)

        F = np.linalg.pinv(M.T @ inv_I @ M) @ M.T @ inv_I

        # Return the filter to original dimensions by adding zeros in the detector-gated zone
        if gate_ns:
            F = np.hstack((F_pad_before, F, F_pad_after))

        ap_filter = AfterulsingFilter(self.t_hist, all_hist_norm, baseline, I_j, norm_factor, M, F)

        if should_plot:
            ap_filter.plot()

        return ap_filter


class TDCPhotonDataProcessor(AngularScanDataMixin, CircularScanDataMixin):
    """For processing raw bytes data"""

    GROUP_LEN: int = 7
    MAX_VAL: int = 256 ** 3

    def __init__(self, laser_freq_hz: int, fpga_freq_hz: int, detector_gate_ns: Gate):
        self.laser_freq_hz = laser_freq_hz
        self.fpga_freq_hz = fpga_freq_hz
        self.detector_gate_ns = detector_gate_ns

    def process_data(self, full_data, **proc_options) -> TDCPhotonData:
        """Doc."""

        if (version := full_data["version"]) < 2:
            raise ValueError(f"Data version ({version}) must be greater than 2 to be handled.")

        # sFCS
        if scan_settings := full_data.get("scan_settings"):
            if (scan_type := scan_settings["pattern"]) == "circle":  # Circular sFCS
                p = self._process_circular_scan_data_file(full_data, **proc_options)
            elif scan_type == "angular":  # Angular sFCS
                p = self._process_angular_scan_data_file(full_data, **proc_options)

        # FCS
        else:
            scan_type = "static"
            p = self._convert_fpga_data_to_photons(
                full_data["byte_data"],
                version,
                is_scan_continuous=True,
                **proc_options,
            )

        with suppress(AttributeError):
            # add general properties
            p.version = version
            p.avg_cnt_rate_khz = full_data["avg_cnt_rate_khz"]

            # TODO: fix it so NAN_PLACEBO is only given to angular scans
            p.NAN_PLACEBO = self.NAN_PLACEBO

        return p

    def _convert_fpga_data_to_photons(
        self,
        byte_data,
        version,
        is_scan_continuous=False,
        should_use_all_sections=True,
        len_factor=0.01,
        is_verbose=False,
        byte_data_slice=None,
        **proc_options,
    ) -> TDCPhotonData:
        """Doc."""

        if is_verbose:
            print("Converting raw data to photons...", end=" ")

        # option to use only certain parts of data (for testing)
        if byte_data_slice is not None:
            byte_data = byte_data[byte_data_slice]

        section_edges, tot_single_errors = self._find_all_section_edges(byte_data)
        section_lengths = [edge_stop - edge_start for (edge_start, edge_stop) in section_edges]

        if should_use_all_sections:
            photon_idxs_list: List[int] = []
            section_runtime_edges = []
            for start_idx, end_idx in section_edges:
                if end_idx - start_idx > sum(section_lengths) * len_factor:
                    section_runtime_start = len(photon_idxs_list)
                    section_photon_indxs = list(range(start_idx, end_idx, self.GROUP_LEN))
                    section_runtime_end = section_runtime_start + len(section_photon_indxs)
                    photon_idxs_list += section_photon_indxs
                    section_runtime_edges.append((section_runtime_start, section_runtime_end))

            photon_idxs = np.array(photon_idxs_list)

        else:  # using largest section only
            max_sec_start_idx, max_sec_end_idx = section_edges[np.argmax(section_lengths)]
            photon_idxs = np.arange(max_sec_start_idx, max_sec_end_idx, self.GROUP_LEN)
            section_runtime_edges = [(0, len(photon_idxs))]

        if is_verbose:
            if len(section_edges) > 1:
                print(
                    f"Found {len(section_edges)} sections of lengths: {', '.join(map(str, section_lengths))}.",
                    end=" ",
                )
                if should_use_all_sections:
                    print(
                        f"Using all valid (> {len_factor:.0%}) sections ({len(section_runtime_edges)}/{len(section_edges)}).",
                        end=" ",
                    )
                else:  # Use largest section only
                    print(f"Using largest (section num.{np.argmax(section_lengths)+1}).", end=" ")
            else:
                print(f"Found a single section of length: {section_lengths[0]}.", end=" ")
            if tot_single_errors > 0:
                print(f"Encountered {tot_single_errors} ignoreable single errors.", end=" ")

        # calculate the global pulse_runtime (the number of laser pulses at each photon arrival since the beginning of the file)
        # each index in pulse_runtime represents a photon arrival into the TDC
        pulse_runtime = (
            byte_data[photon_idxs + 1] * 256 ** 2
            + byte_data[photon_idxs + 2] * 256
            + byte_data[photon_idxs + 3]
        ).astype(np.int64)

        time_stamps = np.diff(pulse_runtime)

        # find simple "inversions": the data with a missing byte
        # decrease in pulse_runtime on data j+1, yet the next pulse_runtime data (j+2) is higher than j.
        inv_idxs = np.where((time_stamps[:-1] < 0) & ((time_stamps[:-1] + time_stamps[1:]) > 0))[0]
        if (inv_idxs.size) != 0:
            if is_verbose:
                print(
                    f"Found {inv_idxs.size} instances of missing byte data, ad hoc fixing...",
                    end=" ",
                )
            temp = (time_stamps[inv_idxs] + time_stamps[inv_idxs + 1]) / 2
            time_stamps[inv_idxs] = np.floor(temp).astype(np.int64)
            time_stamps[inv_idxs + 1] = np.ceil(temp).astype(np.int64)
            pulse_runtime[inv_idxs + 1] = pulse_runtime[inv_idxs + 2] - time_stamps[inv_idxs + 1]

        # repairing drops in pulse_runtime (happens when number of laser pulses passes 'MAX_VAL')
        neg_time_stamp_idxs = np.where(time_stamps < 0)[0]
        for i in neg_time_stamp_idxs + 1:
            pulse_runtime[i:] += self.MAX_VAL

        # handling coarse and fine times (for gating)
        coarse = byte_data[photon_idxs + 4].astype(np.int16)
        fine = byte_data[photon_idxs + 5].astype(np.int16)

        # some fix due to an issue in FPGA
        if version >= 3:
            coarse_mod64 = np.mod(coarse, 64)
            coarse2 = coarse_mod64 - np.mod(coarse_mod64, 4) + (coarse // 64)
            coarse = coarse_mod64
        else:
            coarse = coarse
            coarse2 = None

        # Duration calculation
        time_stamps = np.diff(pulse_runtime).astype(np.int32)
        duration_s = (time_stamps / self.laser_freq_hz).sum()

        if is_scan_continuous:  # relevant for static/circular data
            all_section_edges, skipped_duration = self._section_continuous_data(
                pulse_runtime, time_stamps, is_verbose=is_verbose, **proc_options
            )
        else:  # angular scan
            all_section_edges = None
            skipped_duration = 0

        return TDCPhotonData(
            version=version,
            section_runtime_edges=section_runtime_edges,
            coarse=coarse,
            coarse2=coarse2,
            fine=fine,
            pulse_runtime=pulse_runtime,
            all_section_edges=all_section_edges,
            size_estimate_mb=max(section_lengths) / 1e6,
            duration_s=duration_s,
            skipped_duration=skipped_duration,
            delay_time=np.full(pulse_runtime.shape, self.detector_gate_ns.lower, dtype=np.float16),
        )

    def _section_continuous_data(
        self,
        pulse_runtime,
        time_stamps,
        max_outlier_prob=1e-5,
        n_splits_requested=10,
        min_time_frac=0.5,
        is_verbose=False,
        **kwargs,
    ):
        """Find outliers and create sections seperated by them. Short sections are discarded"""

        mu = np.median(time_stamps) / np.log(2)
        duration_estimate_s = mu * len(pulse_runtime) / self.laser_freq_hz

        # find additional outliers (improbably large time_stamps) and break into
        # additional sections if they exist.
        # for exponential distribution MEDIAN and MAD are the same, but for
        # biexponential MAD seems more sensitive
        mu = max(np.median(time_stamps), np.abs(time_stamps - time_stamps.mean()).mean()) / np.log(
            2
        )
        max_time_stamp = scipy.stats.expon.ppf(1 - max_outlier_prob / len(time_stamps), scale=mu)
        sec_edges = (time_stamps > max_time_stamp).nonzero()[0].tolist()
        if (n_outliers := len(sec_edges)) > 0:
            print(f"found {n_outliers} outliers.", end=" ")
        sec_edges = [0] + sec_edges + [len(time_stamps)]
        all_section_edges = np.array([sec_edges[:-1], sec_edges[1:]]).T

        # Filter short sections
        # TODO: duration limitation is unclear
        split_duration = duration_estimate_s / n_splits_requested
        skipped_duration = 0
        all_section_edges_valid = []
        # Ignore short sections (default is below half the run_duration)
        for se_idx, (se_start, se_end) in enumerate(all_section_edges):
            section_time = (pulse_runtime[se_end] - pulse_runtime[se_start]) / self.laser_freq_hz
            if section_time < min_time_frac * split_duration:
                if is_verbose:
                    print(
                        f"Skipping section {se_idx} - too short ({section_time * 1e3:.2f} ms).",
                        end=" ",
                    )
                skipped_duration += section_time
            else:  # use section
                all_section_edges_valid.append((se_start, se_end))

        return np.array(all_section_edges_valid), skipped_duration

    def _find_all_section_edges(
        self,
        byte_data: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], int]:
        """Doc."""

        section_edges = []
        data_end = False
        last_edge_stop = 0
        total_single_errors = 0
        while not data_end:
            remaining_byte_data = byte_data[last_edge_stop:]
            new_edge_start, new_edge_stop, data_end, n_single_errors = self._find_section_edges(
                remaining_byte_data,
            )
            new_edge_start += last_edge_stop
            new_edge_stop += last_edge_stop
            section_edges.append((new_edge_start, new_edge_stop))
            last_edge_stop = new_edge_stop
            total_single_errors += n_single_errors

        return section_edges, total_single_errors

    def _find_section_edges(self, byte_data: np.ndarray):  # NOQA C901
        """Doc."""

        # find index of first complete photon (where 248 and 254 bytes are spaced exatly (GROUP_LEN -1) bytes apart)
        if (edge_start := self._first_full_photon_idx(byte_data)) is None:
            raise RuntimeError("No byte data found! Check detector and FPGA.")

        # slice byte_data where assumed to be 248 and 254 (photon brackets)
        data_presumed_248 = byte_data[edge_start :: self.GROUP_LEN]
        data_presumed_254 = byte_data[(edge_start + self.GROUP_LEN - 1) :: self.GROUP_LEN]

        # find indices where this assumption breaks
        missed_248_idxs = np.where(data_presumed_248 != 248)[0]
        tot_248s_missed = missed_248_idxs.size
        missed_254_idxs = np.where(data_presumed_254 != 254)[0]
        tot_254s_missed = missed_254_idxs.size

        data_end = False
        n_single_errors = 0
        count = 0
        for count, missed_248_idx in enumerate(missed_248_idxs):

            # hold byte_data idx of current missing photon starting bracket
            data_idx_of_missed_248 = edge_start + missed_248_idx * self.GROUP_LEN

            # condition for ignoring single photon error (just test the most significant bytes of the pulse_runtime are close)
            curr_runtime_msb = int(byte_data[data_idx_of_missed_248 + 1])
            prev_runtime_msb = int(byte_data[data_idx_of_missed_248 - (self.GROUP_LEN - 1)])
            ignore_single_error_cond = abs(curr_runtime_msb - prev_runtime_msb) < 3

            # check that this is not a case of a singular mistake in 248 byte: happens very rarely but does happen
            if tot_254s_missed < count + 1:
                # no missing ending brackets, at least one missing starting bracket
                if missed_248_idx == (len(data_presumed_248) - 1):
                    # problem in the last photon in the file
                    # Found single error in last photon of file, ignoring and finishing...
                    edge_stop = data_idx_of_missed_248
                    break

                elif (tot_248s_missed == count + 1) or (
                    np.diff(missed_248_idxs[count : (count + 2)]) > 1
                ):
                    if ignore_single_error_cond:
                        # f"Found single photon error (byte_data[{data_idx_of_missed_248}]), ignoring and continuing..."
                        n_single_errors += 1
                    else:
                        raise RuntimeError("Check byte data for strange section edges!")

                else:
                    raise RuntimeError(
                        "Bizarre problem in byte data: 248 byte out of registry while 254 is in registry!"
                    )

            else:  # (tot_254s_missed >= count + 1)
                # if (missed_248_idxs[count] == missed_254_idxs[count]), # likely a real section
                if np.isin(missed_248_idx, missed_254_idxs):
                    # Found a section, continuing...
                    edge_stop = data_idx_of_missed_248
                    if byte_data[edge_stop - 1] != 254:
                        edge_stop = edge_stop - self.GROUP_LEN
                    break

                elif missed_248_idxs[count] == (
                    missed_254_idxs[count] + 1
                ):  # np.isin(missed_248_idx, (missed_254_idxs[count]+1)): # likely a real section ? why np.isin?
                    # Found a section, continuing...
                    edge_stop = data_idx_of_missed_248
                    if byte_data[edge_stop - 1] != 254:
                        edge_stop = edge_stop - self.GROUP_LEN
                    break

                elif missed_248_idx < missed_254_idxs[count]:  # likely a singular error on 248 byte
                    if ignore_single_error_cond:
                        # f"Found single photon error (byte_data[{data_idx_of_missed_248}]), ignoring and continuing..."
                        n_single_errors += 1
                        continue
                    else:
                        raise RuntimeError("Check byte data for strange section edges!")

                else:  # likely a signular mistake on 254 byte
                    if ignore_single_error_cond:
                        # f"Found single photon error (byte_data[{data_idx_of_missed_248}]), ignoring and continuing..."
                        n_single_errors += 1
                        continue
                    else:
                        raise RuntimeError("Check byte data for strange section edges!")

        # did reach the end of the loop without breaking?
        were_no_breaks = not tot_248s_missed or (count == (missed_248_idxs.size - 1))
        # case there were no problems
        if were_no_breaks:
            edge_stop = edge_start + (data_presumed_254.size - 1) * self.GROUP_LEN
            data_end = True

        return edge_start, edge_stop, data_end, n_single_errors

    def _first_full_photon_idx(self, byte_data: np.ndarray) -> int:
        """
        Return the starting index of the first intact photon - a sequence of 'GROUP_LEN'
        bytes starting with 248 and ending with 254. If no intact photons are found, returns 'None'
        """

        for idx in infinite_range():
            try:
                if (byte_data[idx] == 248) and (byte_data[idx + (self.GROUP_LEN - 1)] == 254):
                    return idx
            except IndexError:
                break
        return None

    def _process_circular_scan_data_file(self, full_data, **proc_options) -> TDCPhotonData:
        """
        Processes a single circular sFCS data file ('full_data').
        Returns the processed results as a 'TDCPhotonData' object.
        '"""
        # TODO: can this method be moved to the appropriate Mixin class?

        p = self._convert_fpga_data_to_photons(
            full_data["byte_data"],
            full_data["version"],
            is_scan_continuous=True,
            **proc_options,
        )

        scan_settings = full_data["scan_settings"]
        p.ao_sampling_freq_hz = int(scan_settings["ao_sampling_freq_hz"])
        p.circle_freq_hz = scan_settings["circle_freq_hz"]

        print("Converting circular scan to image...", end=" ")
        pulse_runtime = p.pulse_runtime
        cnt, _ = self._sum_scan_circles(  # TODO: test circualr scan - made a change here
            pulse_runtime,
            self.laser_freq_hz,
            p.ao_sampling_freq_hz,
            p.circle_freq_hz,
        )

        p.image = cnt

        # get background correlation
        cnt = cnt.astype(np.float64)
        c, lags = xcorr(cnt, cnt)
        c = c / cnt.mean() ** 2 - 1
        c[0] -= 1 / cnt.mean()  # subtracting shot noise, small stuff really
        p.bg_line_corr = [
            {
                "lag": lags * 1e3 / p.ao_sampling_freq_hz,  # in ms
                "corrfunc": c,
            }
        ]

        return p

    def _process_angular_scan_data_file(
        self,
        full_data,
        should_fix_shift=True,
        roi_selection="auto",
        **kwargs,
    ) -> TDCPhotonData:
        """
        Processes a single angular sFCS data file ('full_data').
        Returns the processed results as a 'TDCPhotonData' object.
        '"""
        # TODO: can this method be moved to the appropriate Mixin class?

        p = self._convert_fpga_data_to_photons(
            full_data["byte_data"], full_data["version"], is_verbose=True
        )

        scan_settings = full_data["scan_settings"]
        linear_part = scan_settings["linear_part"].round().astype(np.uint16)
        p.ao_sampling_freq_hz = int(scan_settings["ao_sampling_freq_hz"])
        p.samples_per_line = int(scan_settings["samples_per_line"])
        p.n_lines = int(scan_settings["n_lines"])

        print("Converting angular scan to image...", end=" ")

        pulse_runtime = np.empty(p.pulse_runtime.shape, dtype=np.int64)
        cnt = np.zeros((p.n_lines + 1, p.samples_per_line), dtype=np.uint16)
        sample_runtime = np.empty(pulse_runtime.shape, dtype=np.int64)
        pixel_num = np.empty(pulse_runtime.shape, dtype=np.int64)
        line_num = np.empty(pulse_runtime.shape, dtype=np.int16)
        for sec_idx, (start_idx, end_idx) in enumerate(p.section_runtime_edges):
            sec_pulse_runtime = p.pulse_runtime[start_idx:end_idx]
            (
                sec_cnt,
                sec_sample_runtime,
                sec_pixel_num,
                sec_line_num,
            ) = self._convert_angular_scan_to_image(
                sec_pulse_runtime,
                self.laser_freq_hz,
                p.ao_sampling_freq_hz,
                p.samples_per_line,
                p.n_lines,
            )

            if should_fix_shift:
                print(f"Fixing line shift of section {sec_idx+1}...", end=" ")
                pix_shift = self._get_data_shift(sec_cnt.copy())
                sec_pulse_runtime = sec_pulse_runtime + pix_shift * round(
                    self.laser_freq_hz / p.ao_sampling_freq_hz
                )
                (
                    sec_cnt,
                    sec_sample_runtime,
                    sec_pixel_num,
                    sec_line_num,
                ) = self._convert_angular_scan_to_image(
                    sec_pulse_runtime,
                    self.laser_freq_hz,
                    p.ao_sampling_freq_hz,
                    p.samples_per_line,
                    p.n_lines,
                )
                print(f"({pix_shift} pixels).", end=" ")

            pulse_runtime[start_idx:end_idx] = sec_pulse_runtime
            cnt += sec_cnt
            sample_runtime[start_idx:end_idx] = sec_sample_runtime
            pixel_num[start_idx:end_idx] = sec_pixel_num
            line_num[start_idx:end_idx] = sec_line_num

        # invert every second line
        cnt[1::2, :] = np.flip(cnt[1::2, :], 1)

        print("ROI selection: ", end=" ")

        if roi_selection == "auto":
            print("automatic. Thresholding and smoothing...", end=" ")
            try:
                bw = self._threshold_and_smooth(cnt.copy())
            except ValueError:
                print("Thresholding failed, skipping file.\n")
                return None
        elif roi_selection == "all":
            bw = np.full(cnt.shape, True)
        else:
            raise ValueError(
                f"roi_selection='{roi_selection}' is not supported. Only 'auto' is, at the moment."
            )

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
        roi: Dict[str, deque] = {"row": deque([]), "col": deque([])}

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
                line_starts_new = list(range(line_starts_new_idx, sample_runtime[-1], bw.size))
                line_stops_new_idx = np.ravel_multi_index((row_idx, right_edge), bw.shape)
                line_stops_new = list(range(line_stops_new_idx, sample_runtime[-1], bw.size))

                line_start_lables += [-row_idx for elem in range(len(line_starts_new))]
                line_stop_labels += [
                    (-row_idx - self.LINE_END_ADDER) for elem in range(len(line_stops_new))
                ]
                line_starts += line_starts_new
                line_stops += line_stops_new

        try:
            # repeat first point to close the polygon
            roi["row"].append(roi["row"][0])
            roi["col"].append(roi["col"][0])
        except IndexError:
            print("ROI is empty (need to figure out the cause). Skipping file.\n")
            return None

        # convert lists/deques to numpy arrays
        roi = {key: np.array(val, dtype=np.uint16) for key, val in roi.items()}
        line_start_lables = np.array(line_start_lables, dtype=np.int16)
        line_stop_labels = np.array(line_stop_labels, dtype=np.int16)
        line_starts = np.array(line_starts, dtype=np.int64)
        line_stops = np.array(line_stops, dtype=np.int64)

        line_starts_runtime: np.ndarray = line_starts * round(
            self.laser_freq_hz / p.ao_sampling_freq_hz
        )
        line_stops_runtime: np.ndarray = line_stops * round(
            self.laser_freq_hz / p.ao_sampling_freq_hz
        )

        pulse_runtime = np.hstack((line_starts_runtime, line_stops_runtime, pulse_runtime))
        sorted_idxs = np.argsort(pulse_runtime)
        p.pulse_runtime = pulse_runtime[sorted_idxs]
        p.line_num = np.hstack(
            (
                line_start_lables,
                line_stop_labels,
                line_num * bw[line_num, pixel_num].flatten(),
            )
        )[sorted_idxs]
        line_starts_nans = np.full(line_starts_runtime.size, self.NAN_PLACEBO, dtype=np.int16)
        line_stops_nans = np.full(line_stops_runtime.size, self.NAN_PLACEBO, dtype=np.int16)
        p.coarse = np.hstack((line_starts_nans, line_stops_nans, p.coarse))[sorted_idxs]
        p.coarse2 = np.hstack((line_starts_nans, line_stops_nans, p.coarse2))[sorted_idxs]
        p.fine = np.hstack((line_starts_nans, line_stops_nans, p.fine))[sorted_idxs]

        # initialize delay times with lower detector gate (nans at line edges) - filled-in during TDC calibration
        p.delay_time = np.full(p.pulse_runtime.shape, self.detector_gate_ns.lower, dtype=np.float16)
        line_edge_idxs = p.fine == self.NAN_PLACEBO
        p.delay_time[line_edge_idxs] = np.nan

        p.image = cnt
        p.roi = roi

        # reverse rows again
        bw[1::2, :] = np.flip(bw[1::2, :], 1)
        p.bw_mask = bw

        # get background correlation (gated background is bad)
        p.line_limits = Limits(line_num[line_num > 0].min(), line_num.max())
        p.bg_line_corr = self._bg_line_correlations(
            p.image, p.bw_mask, p.line_limits, p.ao_sampling_freq_hz
        )

        p.LINE_END_ADDER = self.LINE_END_ADDER

        return p

    def calibrate_tdc(
        self,
        data: List[TDCPhotonData],
        scan_type,
        tdc_chain_length=128,
        pick_valid_bins_according_to=None,
        sync_coarse_time_to=None,
        pick_calib_bins_according_to=None,
        external_calib=None,
        calib_range_ns: Union[Limits, tuple] = Limits(40, 80),
        n_zeros_for_fine_bounds=10,
        time_bins_for_hist_ns=0.1,
        parent_axes=None,
        **kwargs,
    ) -> TDCCalibration:
        """Doc."""
        # TODO: consider what will be needed for detector gated STED measurements (which are synced to confocal)

        coarse, fine = self._unite_coarse_fine_data(data, scan_type)

        h_all = np.bincount(coarse).astype(np.uint32)
        x_all = np.arange(coarse.max() + 1, dtype=np.uint8)

        if pick_valid_bins_according_to is None:
            h_all = h_all[coarse.min() :]
            x_all = np.arange(coarse.min(), coarse.max() + 1, dtype=np.uint8)
            coarse_bins = x_all
            h = h_all
        elif isinstance(pick_valid_bins_according_to, np.ndarray):
            coarse_bins = sync_coarse_time_to
            h = h_all[coarse_bins]
        elif isinstance(pick_valid_bins_according_to, TDCCalibration):
            coarse_bins = sync_coarse_time_to.coarse_bins
            h = h_all[coarse_bins]
        else:
            raise TypeError(
                f"Unknown type '{type(pick_valid_bins_according_to)}' for picking valid coarse_bins!"
            )

        # rearranging the coarse_bins
        if sync_coarse_time_to is None:
            if (lower_detector_gate_ns := self.detector_gate_ns.lower) > 0:  # detector gating
                max_j = np.argmax(h) - round((lower_detector_gate_ns * 1e-9) * self.fpga_freq_hz)
            else:
                max_j = np.argmax(h)
        elif isinstance(sync_coarse_time_to, int):
            max_j = sync_coarse_time_to
        elif isinstance(sync_coarse_time_to, TDCCalibration):
            max_j = sync_coarse_time_to.max_j
        else:
            raise TypeError(
                f"sync_coarse_time_to must be either a number or a 'TDCCalibration' object! Type '{type(sync_coarse_time_to).name}' was supplied."
            )

        j_shift = np.roll(np.arange(len(h)), -max_j + 2)

        if pick_calib_bins_according_to is None:
            # pick data at more than 'calib_time_ns' delay from peak maximum
            calib_range_bins = (
                ((Limits(calib_range_ns) & self.detector_gate_ns) + self.detector_gate_ns.lower)
                * 1e-9
                * self.fpga_freq_hz
            )
            j = calib_range_bins.valid_indices(coarse_bins)
            if not j.any():
                raise ValueError(f"Gate width is too narrow for calib_time_ns={calib_range_ns}!")
            j_calib = j_shift[j]
            coarse_calib_bins = coarse_bins[j_calib]
        elif isinstance(pick_calib_bins_according_to, (list, np.ndarray)):
            coarse_calib_bins = pick_calib_bins_according_to
        elif isinstance(pick_calib_bins_according_to, TDCCalibration):
            coarse_calib_bins = pick_calib_bins_according_to.coarse_calib_bins
        else:
            raise TypeError(
                f"Unknown type '{type(pick_calib_bins_according_to)}' for picking calibration coarse_bins!"
            )

        # Don't use 'empty' coarse_bins for calibration
        valid_cal_bins = np.nonzero(h_all > (np.median(h_all) / 100))[0]
        coarse_calib_bins = np.array(
            [bin_idx for bin_idx in coarse_calib_bins if bin_idx in valid_cal_bins]
        )

        if isinstance(external_calib, TDCCalibration):
            max_j = external_calib.max_j
            coarse_calib_bins = external_calib.coarse_calib_bins
            fine_bins = external_calib.fine_bins
            t_calib = external_calib.t_calib
            h_tdc_calib = external_calib.h_tdc_calib
            t_weight = external_calib.t_weight
            l_quarter_tdc = external_calib.l_quarter_tdc
            r_quarter_tdc = external_calib.r_quarter_tdc
            h = external_calib.h
            delay_times = external_calib.delay_times

            with suppress(AttributeError):
                # TODO: this will fail - why is it here?
                raise AttributeError("FIGURE THIS OUT!")
        #                l_quarter_tdc = self.tdc_calib.l_quarter_tdc
        #                r_quarter_tdc = self.tdc_calib.r_quarter_tdc

        else:

            fine_calib = fine[np.isin(coarse, coarse_calib_bins)]
            h_tdc_calib = np.bincount(fine_calib, minlength=tdc_chain_length).astype(np.uint32)

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
            left_tdc = max(zeros_tdc[zeros_tdc < mid_tdc]) + n_zeros_for_fine_bounds - 1
            right_tdc = min(zeros_tdc[zeros_tdc > mid_tdc]) + 1

            l_quarter_tdc = round(left_tdc + (right_tdc - left_tdc) / 4)
            r_quarter_tdc = round(right_tdc - (right_tdc - left_tdc) / 4)

            # zero those out of TDC (happened at least once, for old detector data from 10/05/2018)
            if sum(h_tdc_calib[:left_tdc]) or sum(h_tdc_calib[right_tdc:]):
                h_tdc_calib[:left_tdc] = 0
                h_tdc_calib[right_tdc:] = 0

            t_calib = (
                (1 - np.cumsum(h_tdc_calib) / np.sum(h_tdc_calib)) / self.fpga_freq_hz * 1e9
            )  # invert and to ns

            coarse_len = coarse_bins.size

            t_weight = np.tile(h_tdc_calib / np.mean(h_tdc_calib), coarse_len)
            coarse_times = (
                np.tile(np.arange(coarse_len), [t_calib.size, 1]) / self.fpga_freq_hz * 1e9
            )
            delay_times = np.tile(t_calib, coarse_len) + coarse_times.flatten("F")
            j_sorted = np.argsort(delay_times)
            delay_times = delay_times[j_sorted]

            t_weight = t_weight[j_sorted]

        # assign time delays to all photons
        total_laser_pulses = 0
        first_coarse_bin, *_, last_coarse_bin = coarse_bins
        delay_time_list = []
        for p in data:
            p.delay_time = np.full(p.coarse.shape, np.nan, dtype=np.float64)
            crs = np.minimum(p.coarse, last_coarse_bin) - coarse_bins[max_j - 1]
            crs[crs < 0] = crs[crs < 0] + last_coarse_bin - first_coarse_bin + 1

            delta_coarse = p.coarse2 - p.coarse
            delta_coarse[delta_coarse == -3] = 1  # 2bit limitation

            # in the TDC midrange use "coarse" counter
            in_mid_tdc = (p.fine >= l_quarter_tdc) & (p.fine <= r_quarter_tdc)
            delta_coarse[in_mid_tdc] = 0

            # on the right of TDC use "coarse2" counter (no change in delta)
            # on the left of TDC use "coarse2" counter decremented by 1
            on_left_tdc = p.fine < l_quarter_tdc
            delta_coarse[on_left_tdc] = delta_coarse[on_left_tdc] - 1

            photon_idxs = p.fine != p.NAN_PLACEBO  # self.NAN_PLACEBO are starts/ends of lines
            p.delay_time[photon_idxs] = (
                t_calib[p.fine[photon_idxs]]
                + (crs[photon_idxs] + delta_coarse[photon_idxs]) / self.fpga_freq_hz * 1e9
            )
            total_laser_pulses += p.pulse_runtime[-1]

            delay_time_list.append(p.delay_time[photon_idxs])

        delay_time = np.hstack(delay_time_list)

        fine_bins = np.arange(
            -time_bins_for_hist_ns / 2,
            np.max(delay_time) + time_bins_for_hist_ns,
            time_bins_for_hist_ns,
            dtype=np.float16,
        )

        t_hist = (fine_bins[:-1] + fine_bins[1:]) / 2
        k = np.digitize(delay_times, fine_bins)

        hist_weight = np.empty(t_hist.shape, dtype=np.float64)
        for i in range(len(hist_weight)):
            j = k == (i + 1)
            hist_weight[i] = np.sum(t_weight[j])

        all_hist = np.histogram(delay_time, bins=fine_bins)[0].astype(np.uint32)

        all_hist_norm = np.full(all_hist.shape, np.nan, dtype=np.float64)
        error_all_hist_norm = np.full(all_hist.shape, np.nan, dtype=np.float64)
        nonzero = hist_weight > 0
        all_hist_norm[nonzero] = (
            all_hist[nonzero] / hist_weight[nonzero] * np.mean(hist_weight) / total_laser_pulses
        )
        error_all_hist_norm[nonzero] = (
            np.sqrt(all_hist[nonzero])
            / hist_weight[nonzero]
            * np.mean(hist_weight)
            / total_laser_pulses
        )

        return TDCCalibration(
            x_all=x_all,
            h_all=h_all,
            coarse_bins=coarse_bins,
            h=h,
            max_j=max_j,
            coarse_calib_bins=coarse_calib_bins,
            fine_bins=fine_bins,
            l_quarter_tdc=l_quarter_tdc,
            r_quarter_tdc=r_quarter_tdc,
            h_tdc_calib=h_tdc_calib,
            t_calib=t_calib,
            t_hist=t_hist,
            hist_weight=hist_weight,
            all_hist=all_hist,
            all_hist_norm=all_hist_norm,
            error_all_hist_norm=error_all_hist_norm,
            delay_times=delay_times,
            t_weight=t_weight,
            total_laser_pulses=total_laser_pulses,
        )

    def _unite_coarse_fine_data(self, data, scan_type: str):
        """Doc."""

        # keep pulse_runtime elements of each file for array size allocation
        n_elem = np.cumsum([0] + [p.pulse_runtime.size for p in data])

        # unite coarse and fine times from all files
        coarse = np.empty(shape=(n_elem[-1],), dtype=np.int16)
        fine = np.empty(shape=(n_elem[-1],), dtype=np.int16)
        for i, p in enumerate(data):
            coarse[n_elem[i] : n_elem[i + 1]] = p.coarse
            fine[n_elem[i] : n_elem[i + 1]] = p.fine

        # remove line starts/ends from angular scan data
        if scan_type == "angular":
            photon_idxs = fine > p.NAN_PLACEBO
            coarse = coarse[photon_idxs]
            fine = fine[photon_idxs]

        return coarse, fine


@dataclass
class CountsImageStackData:
    """
    Holds a stack of images along with some relevant scan data,
    built from CI (counter input) NIDAQmx data.
    """

    image_stack_forward: np.ndarray
    norm_stack_forward: np.ndarray
    image_stack_backward: np.ndarray
    norm_stack_backward: np.ndarray
    line_ticks_v: np.ndarray
    row_ticks_v: np.ndarray
    plane_ticks_v: np.ndarray
    n_planes: int
    plane_orientation: str
    dim_order: Tuple[int, int, int]

    def construct_image(self, method: str, plane_idx: int = None) -> np.ndarray:
        """Doc."""

        if plane_idx is None:
            plane_idx = self.n_planes // 2

        if method == "forward":
            img = self.image_stack_forward[:, :, plane_idx]
        elif method == "forward normalization":
            img = self.norm_stack_forward[:, :, plane_idx]
        elif method == "forward normalized":
            img = (
                self.image_stack_forward[:, :, plane_idx] / self.norm_stack_forward[:, :, plane_idx]
            )
        elif method == "backward":
            img = self.image_stack_backward[:, :, plane_idx]
        elif method == "backward normalization":
            img = self.norm_stack_backward[:, :, plane_idx]
        elif method == "backward normalized":
            img = (
                self.image_stack_backward[:, :, plane_idx]
                / self.norm_stack_backward[:, :, plane_idx]
            )
        elif method == "interlaced":
            p1_norm = (
                self.image_stack_forward[:, :, plane_idx] / self.norm_stack_forward[:, :, plane_idx]
            )
            p2_norm = (
                self.image_stack_backward[:, :, plane_idx]
                / self.norm_stack_backward[:, :, plane_idx]
            )
            n_lines = p1_norm.shape[0] + p2_norm.shape[0]
            img = np.zeros(p1_norm.shape)
            img[:n_lines:2, :] = p1_norm[:n_lines:2, :]
            img[1:n_lines:2, :] = p2_norm[1:n_lines:2, :]
        elif method == "averaged":
            p1 = self.image_stack_forward[:, :, plane_idx]
            p2 = self.image_stack_backward[:, :, plane_idx]
            norm1 = self.norm_stack_forward[:, :, plane_idx]
            norm2 = self.norm_stack_backward[:, :, plane_idx]
            img = (p1 + p2) / (norm1 + norm2)

        return img


class CountsImageMixin:
    """Doc."""

    def create_image_stack_data(self, file_dict: dict) -> CountsImageStackData:
        """Doc."""

        scan_settings = file_dict["scan_settings"]
        um_v_ratio = file_dict["system_info"]["xyz_um_to_v"]
        ao = file_dict["ao"]
        counts = file_dict["ci"]

        n_planes = scan_settings["n_planes"]
        n_lines = scan_settings["n_lines"]
        pxl_size_um = scan_settings["dim2_um"] / n_lines
        pxls_per_line = div_ceil(scan_settings["dim1_um"], pxl_size_um)
        dim_order = scan_settings["dim_order"]
        ppl = scan_settings["ppl"]
        ppp = n_lines * ppl
        turn_idx = ppl // 2

        first_dim = dim_order[0]
        dim1_center = scan_settings["initial_ao"][first_dim]
        um_per_v = um_v_ratio[first_dim]

        line_len_v = scan_settings["dim1_um"] / um_per_v
        dim1_min = dim1_center - line_len_v / 2

        pxl_size_v = pxl_size_um / um_per_v
        pxls_per_line = div_ceil(scan_settings["dim1_um"], pxl_size_um)

        # prepare to remove counts from outside limits
        dim1_ao_single = ao[0][:ppl]
        eff_idxs = ((dim1_ao_single - dim1_min) // pxl_size_v + 1).astype(np.int16)
        eff_idxs_forward = eff_idxs[:turn_idx]
        eff_idxs_backward = eff_idxs[-1 : (turn_idx - 1) : -1]

        # create counts stack shaped (n_lines, ppl, n_planes) - e.g. 80 x 1000 x 1
        j0 = ppp * np.arange(n_planes)[:, np.newaxis]
        J = np.tile(np.arange(ppp), (n_planes, 1)) + j0
        counts_stack = np.diff(np.concatenate((j0, counts[J]), axis=1))
        counts_stack = counts_stack.T.reshape(n_lines, ppl, n_planes)
        counts_stack_forward = counts_stack[:, :turn_idx, :]
        counts_stack_backward = counts_stack[:, -1 : (turn_idx - 1) : -1, :]

        # calculate the images and normalization separately for the forward/backward parts of the scan
        image_stack_forward, norm_stack_forward = self._calculate_plane_image_stack(
            counts_stack_forward, eff_idxs_forward, pxls_per_line
        )
        image_stack_backward, norm_stack_backward = self._calculate_plane_image_stack(
            counts_stack_backward, eff_idxs_backward, pxls_per_line
        )

        return CountsImageStackData(
            image_stack_forward=image_stack_forward,
            norm_stack_forward=norm_stack_forward,
            image_stack_backward=image_stack_backward,
            norm_stack_backward=norm_stack_backward,
            line_ticks_v=dim1_min + np.arange(pxls_per_line) * pxl_size_v,
            row_ticks_v=scan_settings["set_pnts_lines_odd"],
            plane_ticks_v=scan_settings.get("set_pnts_planes"),  # doesn't exist in older versions
            n_planes=n_planes,
            plane_orientation=scan_settings["plane_orientation"],
            dim_order=dim_order,
        )

    def _calculate_plane_image_stack(self, counts_stack, eff_idxs, pxls_per_line):
        """Doc."""

        n_lines, _, n_planes = counts_stack.shape
        image_stack = np.empty((n_lines, pxls_per_line, n_planes), dtype=np.int32)
        norm_stack = np.empty((n_lines, pxls_per_line, n_planes), dtype=np.int32)

        for i in range(pxls_per_line):
            image_stack[:, i, :] = counts_stack[:, eff_idxs == i, :].sum(axis=1)
            norm_stack[:, i, :] = counts_stack[:, eff_idxs == i, :].shape[1]

        return image_stack, norm_stack
