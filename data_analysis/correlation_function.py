"""Data organization and manipulation."""

import logging
import multiprocessing as mp
import re
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import scipy
import skimage
from sklearn import linear_model

from data_analysis.photon_data import (
    CountsImageMixin,
    TDCPhotonData,
    TDCPhotonDataMixin,
)
from data_analysis.software_correlator import CorrelatorType, SoftwareCorrelator
from utilities import file_utilities
from utilities.display import Plotter
from utilities.fit_tools import FitParams, curve_fit_lims, multi_exponent_fit
from utilities.helper import Limits, div_ceil, timer

laser_pulse_tdc_calib = file_utilities.load_object("./data_analysis./laser_pulse_tdc_calib.pkl")


class AngularScanMixin:
    """Doc."""

    NAN_PLACEBO = -100

    def _convert_angular_scan_to_image(
        self, pulse_runtime, laser_freq_hz, ao_sampling_freq_hz, samples_per_line, n_lines
    ):
        """utility function for opening Angular Scans"""

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

    def _get_best_pix_shift(self, img: np.ndarray, min_shift, max_shift) -> int:
        """Doc."""

        score = np.empty(shape=(max_shift - min_shift), dtype=np.uint32)
        pix_shifts = np.arange(min_shift, max_shift)
        for idx, pix_shift in enumerate(range(min_shift, max_shift)):
            rolled_img = np.roll(img, pix_shift).astype(np.uint32)
            score[idx] = ((rolled_img[:-1:2, :] - np.fliplr(rolled_img[1::2, :])) ** 2).sum()
        return pix_shifts[score.argmin()]

    def _fix_data_shift(self, cnt) -> int:
        """Doc."""

        height, width = cnt.shape

        # replacing outliers with median value
        med = np.median(cnt)
        cnt[cnt > med * 1.5] = med

        # limit initial attempt to the width of the image
        min_pix_shift = -round(width / 2)
        max_pix_shift = min_pix_shift + width + 1
        pix_shift = self._get_best_pix_shift(cnt, min_pix_shift, max_pix_shift)

        # Test if not stuck in local minimum (outer_half_sum > inner_half_sum)
        # OR if the 'return row' (the empty one) is not at the bottom for some reason
        # TODO: ask Oleg how the latter can happen
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
            pix_shift = self._get_best_pix_shift(cnt, min_pix_shift, max_pix_shift)

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
        disk_open = skimage.morphology.selem.disk(radius=disk_radius)
        bw = skimage.morphology.opening(bw, selem=disk_open)
        return bw

    def _bg_line_correlations(
        self,
        image1: np.ndarray,
        bw_mask: np.ndarray,
        line_limits: Limits,
        sampling_freq_Hz,
        image2: np.ndarray = None,
    ) -> list:
        """Returns a list of auto-correlations of the lines of an image"""

        def xcorr(a, b):
            """Does correlation similar to Matlab xcorr, cuts positive lags, normalizes properly"""

            c = np.correlate(a, b, mode="full")
            c = c[c.size // 2 :]
            c = c / np.arange(c.size, 0, -1)
            lags = np.arange(c.size, dtype=np.uint16)

            return c, lags

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


@dataclass
class StructureFactor:
    """Holds structure factor data"""

    # parameters
    n_interp_pnts: int
    r_max: float
    vt_min_sq: float
    g_min: float

    r: np.ndarray
    fr: np.ndarray
    fr_linear_interp: np.ndarray

    q: np.ndarray
    sq: np.ndarray
    sq_linear_interp: np.ndarray
    sq_error: np.ndarray


class CorrFunc:
    """Doc."""

    run_duration: float
    afterpulse: np.ndarray
    vt_um: np.ndarray
    cf_cr: np.ndarray
    g0: float
    EPSILON = 1e-5  # TODO: why is this needed (in low-data measurements)

    def __init__(self, name, gate_ns, laser_freq_hz):
        self.name = name
        self.gate_ns = Limits(gate_ns)
        self.laser_freq_hz = laser_freq_hz

        self.skipped_duration = 0
        self.fit_params: Dict[str, FitParams] = dict()
        self.total_duration = 0
        self.afterpulse = None

    def correlate_measurement(
        self,
        correlator_type: int,
        time_stamp_split_list: List[np.ndarray],
        *args,
        #        should_parallel_process=False,
        is_verbose: bool = False,
        **kwargs,
    ):
        """Doc."""

        if is_verbose:
            print(f"Correlating {len(time_stamp_split_list)} splits -", end=" ")

        SC = SoftwareCorrelator()

        #        # TODO: parallel correlation isn't operational. SoftwareCorrelator objects contain ctypes containing pointers, which cannot be pickled for seperate processes.
        #        # See: https://stackoverflow.com/questions/18976937/multiprocessing-and-ctypes-with-pointers
        #        if should_parallel_process and len(time_stamp_split_list) > 1:  # parallel-correlate
        #            N_CORES = mp.cpu_count() // 2 - 1  # division by 2 due to hyperthreading in intel CPUs
        #            func = partial(
        #                SC.correlate, c_type=correlator_type, timebase_ms=1000 / laser_freq_hz
        #            )
        #            print(f"Parallel processing using {N_CORES} CPUs/processes.")
        #            with mp.get_context("spawn").Pool(N_CORES) as pool:
        #                correlator_output_list = list(pool.imap(func, time_stamp_split_list))
        #
        #        else:  # serial-correlate
        output = SC.correlate_list(
            time_stamp_split_list,
            correlator_type,
            timebase_ms=1000 / self.laser_freq_hz,
            is_verbose=is_verbose,
        )
        self.lag = max(output.lag_list, key=len)

        if is_verbose:
            print(". Processing...", end=" ")

        self._process_correlator_list_output(output, *args, **kwargs)

    def _process_correlator_list_output(
        self,
        corr_output,
        after_pulse_params,
        bg_corr_list,
        should_subtract_afterpulse: bool = True,
        external_afterpulse: np.ndarray = None,
        **kwargs,
    ):
        """Doc."""

        # remove_background_correlation
        self._remove_background_correlation(corr_output, bg_corr_list)

        # calculate afterpulsing
        if external_afterpulse is not None:
            self.afterpulse = external_afterpulse
        elif should_subtract_afterpulse:
            self.afterpulse = calculate_afterpulse(
                after_pulse_params,
                self.lag,
                self.laser_freq_hz,
                self.gate_ns,
            )

        self.countrate_list = corr_output.countrate_list
        self._generate_cf_cr(corr_output)

    def _remove_background_correlation(self, corr_output, bg_corr_list) -> None:
        """Doc."""

        for idx, bg_corr_dict in enumerate(bg_corr_list):
            bg_corr = np.interp(
                corr_output.lag_list[idx],
                bg_corr_dict["lag"],
                bg_corr_dict["corrfunc"],
                right=0,
            )
            corr_output.corrfunc_list[idx] -= bg_corr

    def _generate_cf_cr(self, corr_output) -> None:
        """Zero-pad and generate cf_cr"""

        lag_len = len(self.lag)
        n_corrfuncs = len(corr_output.corrfunc_list)
        shape = (n_corrfuncs, lag_len)
        self.corrfunc = np.empty(shape=shape, dtype=np.float64)
        self.weights = np.empty(shape=shape, dtype=np.float64)
        self.cf_cr = np.empty(shape=shape, dtype=np.float64)
        for idx in range(n_corrfuncs):
            pad_len = lag_len - len(corr_output.corrfunc_list[idx])
            self.corrfunc[idx] = np.pad(corr_output.corrfunc_list[idx], (0, pad_len))
            self.weights[idx] = np.pad(corr_output.weights_list[idx], (0, pad_len))
            try:
                self.cf_cr[idx] = corr_output.countrate_list[idx] * self.corrfunc[idx]
            except ValueError:  # Cross-correlation - countrate is a 2-tuple
                self.cf_cr[idx] = np.mean(corr_output.countrate_list[idx]) * self.corrfunc[idx]
            if self.afterpulse is not None:
                self.cf_cr[idx] -= self.afterpulse[:lag_len]  # ext. afterpulse might be too long

    def average_correlation(
        self,
        rejection=2,
        reject_n_worst=None,
        norm_range=(1e-3, 2e-3),
        delete_list=[],
        should_plot=False,
        plot_kwargs={},
        **kwargs,
    ):
        """Doc."""

        self.rejection = rejection
        self.norm_range = Limits(norm_range)
        self.delete_list = delete_list
        self.average_all_cf_cr = (self.cf_cr * self.weights).sum(0) / self.weights.sum(0)
        self.median_all_cf_cr = np.median(self.cf_cr, axis=0)
        jj = Limits(self.norm_range.upper, 100).valid_indices(self.lag)  # work in the relevant part

        try:
            self.score = (
                (1 / np.var(self.cf_cr[:, jj], 0))
                * (self.cf_cr[:, jj] - self.median_all_cf_cr[jj]) ** 2
                / len(jj)
            ).sum(axis=1)
        except RuntimeWarning:  # division by zero
            # TODO: why does this happen?
            self.score = (
                (1 / (np.var(self.cf_cr[:, jj], 0) + self.EPSILON))
                * (self.cf_cr[:, jj] - self.median_all_cf_cr[jj]) ** 2
                / len(jj)
            ).sum(axis=1)
            print(
                f"Division by zero avoided by adding EPSILON={self.EPSILON}. Why does this happen (zero in variance)?"
            )

        total_n_rows, _ = self.cf_cr.shape

        if reject_n_worst not in {None, 0}:
            delete_list = np.argsort(self.score)[-reject_n_worst:]
        elif rejection is not None:
            delete_list = np.where(self.score >= self.rejection)[0]
            if len(delete_list) == total_n_rows:
                raise RuntimeError(
                    "All rows are in 'delete_list'! Increase the rejection limit. Ignoring."
                )

        # if 'reject_n_worst' and 'rejection' are both None, use supplied delete list. If no delete list is supplied, use all rows.
        self.j_bad = delete_list
        self.j_good = [row for row in range(total_n_rows) if row not in delete_list]

        self.avg_cf_cr, self.error_cf_cr = self._calculate_weighted_avg(
            self.cf_cr[self.j_good, :], self.weights[self.j_good, :]
        )

        j_t = self.norm_range.valid_indices(self.lag)

        try:
            self.g0 = (self.avg_cf_cr[j_t] / self.error_cf_cr[j_t] ** 2).sum() / (
                1 / self.error_cf_cr[j_t] ** 2
            ).sum()
        except RuntimeWarning:  # division by zero
            # TODO: why does this happen?
            self.g0 = (self.avg_cf_cr[j_t] / (self.error_cf_cr[j_t] + self.EPSILON) ** 2).sum() / (
                1 / (self.error_cf_cr[j_t] + self.EPSILON) ** 2
            ).sum()
            print(
                f"Division by zero avoided by adding EPSILON={self.EPSILON}. Why does this happen (zero in variance)?"
            )

        self.normalized = self.avg_cf_cr / self.g0
        self.error_normalized = self.error_cf_cr / self.g0

        if should_plot:
            self.plot_correlation_function(plot_kwargs=plot_kwargs)

    def _calculate_weighted_avg(self, cf_cr, weights):
        """Doc."""

        tot_weights = weights.sum(0)
        try:
            avg_cf_cr = (cf_cr * weights).sum(0) / tot_weights
        except RuntimeWarning:  # division by zero - caused by zero total weights element/s
            # TODO: why does this happen?
            tot_weights += self.EPSILON
            avg_cf_cr = (cf_cr * weights).sum(0) / tot_weights
            print(
                f"Division by zero avoided by adding epsilon={self.EPSILON}. Why does this happen (zero total weight)?"
            )
        finally:
            error_cf_cr = np.sqrt((weights ** 2 * (cf_cr - avg_cf_cr) ** 2).sum(0)) / tot_weights

        return avg_cf_cr, error_cf_cr

    def plot_correlation_function(
        self,
        x_field="lag",
        y_field="avg_cf_cr",
        x_scale="log",
        y_scale="linear",
        **kwargs,
    ):

        x = getattr(self, x_field)
        y = getattr(self, y_field)
        if x_scale == "log":  # remove zero point data
            x, y = x[1:], y[1:]

        with Plotter(
            x_scale=x_scale,
            y_scale=y_scale,
            should_autoscale=True,
            **kwargs,
        ) as ax:
            ax.set_xlabel(x_field)
            ax.set_ylabel(y_field)
            ax.plot(x, y, "-", **kwargs.get("plot_kwargs", {}))

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
        **kwargs,
    ):

        if fit_param_estimate is None:
            fit_param_estimate = (self.g0, 0.035, 30.0)

        x = getattr(self, x_field)
        y = getattr(self, y_field)
        error_y = getattr(self, y_error_field)
        if x_scale == "log":
            x = x[1:]  # remove zero point data
            y = y[1:]
            error_y = error_y[1:]

        self.fit_params[fit_name] = curve_fit_lims(
            fit_name,
            fit_param_estimate,
            x,
            y,
            error_y,
            x_limits=Limits(fit_range),
            should_plot=should_plot,
            x_scale=x_scale,
            y_scale=y_scale,
            **kwargs,
        )

    def calculate_structure_factor(
        self,
        n_interp_pnts=512,
        r_max=10,
        r_min=0.05,
        g_min=1e-2,
        n_robust=2,
        **kwargs,
    ) -> None:
        """Doc."""

        vt_min_sq = r_min ** 2

        c0 = scipy.special.jn_zeros(0, n_interp_pnts)  # Bessel function zeros
        r = (r_max / c0[n_interp_pnts - 1]) * c0  # Radius vector

        # choose interpolation range (extrapolate the noisy parts)
        j_vt_um_sq_over_min = np.nonzero(self.vt_um ** 2 > vt_min_sq)[0]
        j_vt_um_under_r_max = np.nonzero(self.vt_um < r_max)[0]
        j_normalized_over_gmin = np.nonzero(self.normalized > g_min)[0][:-1]
        #        j_intrp = sorted(set(j_vt_um_sq_over_min) & set(j_normalized_over_gmin))
        j_intrp = sorted(
            set(j_vt_um_sq_over_min) & set(j_vt_um_under_r_max) & set(j_normalized_over_gmin)
        )

        sample_vt_um = self.vt_um[j_intrp]
        sample_normalized = self.normalized[j_intrp]

        #  Gaussian interpolation
        if n_robust:
            log_fr = self.robust_interpolation(
                r ** 2, sample_vt_um ** 2, np.log(sample_normalized), n_robust
            )
        else:
            interpolator = scipy.interpolate.interp1d(
                sample_vt_um ** 2,
                np.log(sample_normalized),
                fill_value="extrapolate",
                assume_sorted=True,
            )
            log_fr = interpolator(r ** 2)
        fr = np.exp(log_fr)

        #  linear interpolation
        if n_robust:
            fr_linear_interp = self.robust_interpolation(
                r, sample_vt_um, sample_normalized, n_robust
            )
        else:
            interpolator = scipy.interpolate.interp1d(
                sample_vt_um,
                sample_normalized,
                fill_value="extrapolate",
                assume_sorted=True,
            )
            fr_linear_interp = interpolator(r)

        # plot interpolations for checks
        with Plotter(
            super_title=f"{self.name.capitalize()}: Interpolation Testing",
            xlim=(0, self.vt_um[max(j_intrp) + 5] ** 2),
            ylim=(1e-1, 1.3),
        ) as ax:
            ax.semilogy(self.vt_um ** 2, self.normalized, "x")
            ax.semilogy(sample_vt_um ** 2, sample_normalized, "o")
            ax.semilogy(r ** 2, fr)
            ax.semilogy(r ** 2, fr_linear_interp)
            ax.legend(
                [
                    "Normalized",
                    "Interpolation Sample",
                    "Gaussian Intep/Extrap",
                    "Linear Intep/Extrap",
                ]
            )
            ax.set_xlabel("Displacement $(\\mu m^2)$")
            ax.set_ylabel("ACF (Normalized)")

        #  zero-pad
        fr_linear_interp[r > self.vt_um[j_normalized_over_gmin[-1]]] = 0

        # Fourier Transform
        q, fq = self.hankel_transform(r, fr)

        #  linear interpolated Fourier Transform
        _, fq_linear_interp = self.hankel_transform(r, fr_linear_interp)

        # TODO: show this to Oleg - can't figure out what's wrong
        #        #  Estimating error of transform
        #        print("Estimating error of transform...") # TESTESTEST
        #        fq_allfunc = np.empty(shape=(len(self.j_good), q.size))
        #        for idx, j in enumerate(self.j_good):
        #            sample_cf_cr = self.cf_cr[j, j_intrp]
        #            sample_cf_cr[sample_cf_cr <= 0] = 1e-3  # TODO: is that the best way to solve this?
        #            if n_robust and False: # TESTESTEST - CANELLED FOR NOW
        #                log_fr = self.robust_interpolation(
        #                    r ** 2, sample_vt_um ** 2, np.log(sample_cf_cr), n_robust
        #                )
        #            else:
        #                log_fr = np.interp(r ** 2, sample_vt_um ** 2, np.log(sample_cf_cr))
        #            fr = np.exp(log_fr)
        #            fr[np.nonzero(fr < 0)[0]] = 0
        #
        #            _, fq_allfunc[idx] = self.hankel_transform(r, fr)
        #
        #        fq_error = np.std(fq_allfunc, axis=0, ddof=1) / np.sqrt(len(self.j_good)) / self.g0
        fq_error = None

        self.structure_factor = StructureFactor(
            n_interp_pnts,
            r_max,
            vt_min_sq,
            g_min,
            r,
            fr,
            fr_linear_interp,
            q,
            fq,
            fq_linear_interp,
            fq_error,
        )

    def robust_interpolation(
        self,
        x,
        xi,
        yi,
        n_pnts,
    ):
        """Doc."""

        # translated from: [h, bin] = histc(x, np.array([-np.inf, xi, inf]))
        xi_inf = np.concatenate(([-np.inf], xi, [np.inf]))
        (h, _), bin = np.histogram(x, xi_inf), np.digitize(x, xi_inf)
        ch = np.cumsum(h)
        start = max(bin[0] - 1, n_pnts)
        finish = min(bin[x.size - 1] - 1, xi.size - n_pnts - 1)

        y = np.empty(shape=x.shape)
        ransac = linear_model.RANSACRegressor()
        for i in range(start, finish):
            #        for i in range(start, finish + 1):
            #            ji = range(i - n_pnts, i + n_pnts + 1) # TESTESTEST
            ji = slice(i - n_pnts, i + n_pnts + 1)

            # Robustly fit linear model with RANSAC algorithm
            ransac.fit(xi[ji][:, np.newaxis], yi[ji])
            p0, p1 = ransac.estimator_.intercept_, ransac.estimator_.coef_[0]

            if i == start:
                j = range(ch[i + 1] + 1)
            elif i == finish - 1:
                j = range((ch[i] + 1), x.size)
            else:
                j = range((ch[i] + 1), ch[i + 1] + 1)

            y[j] = p0 + p1 * x[j]

        return y

    def hankel_transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        should_inverse: bool = False,
        n_robust: int = 2,
        should_do_gaussian_interpolation: bool = False,
        dr=None,
    ):
        """Doc."""

        n = x.size

        # prepare the Hankel transformation matrix C
        c0 = scipy.special.jn_zeros(0, n)
        bessel_j0 = scipy.special.j0
        bessel_j1 = scipy.special.j1

        j_n, j_m = np.meshgrid(c0, c0)

        C = (
            (2 / c0[n - 1])
            * bessel_j0(j_n * j_m / c0[n - 1])
            / (abs(bessel_j1(j_n)) * abs(bessel_j1(j_m)))
        )

        if not should_inverse:

            r_max = max(x)
            q_max = c0[n - 1] / (2 * np.pi * r_max)  # Maximum frequency

            r = c0.T * r_max / c0[n - 1]  # Radius vector
            q = c0.T / (2 * np.pi * r_max)  # Frequency vector

            m1 = (abs(bessel_j1(c0)) / r_max).T  # m1 prepares input vector for transformation
            m2 = m1 * r_max / q_max  # m2 prepares output vector for display

            # end preparations for Hankel transform

            if n_robust:  # use robust interpolation
                if should_do_gaussian_interpolation:  # Gaussian
                    y_interp = np.exp(
                        self.robust_interpolation(r ** 2, x ** 2, np.log(y), n_robust)
                    )
                    y_interp[
                        r > x[-n_robust]
                    ] = 0  # zero pad last points that do not have full interpolation
                else:  # linear
                    y_interp = self.robust_interpolation(r, x, y, n_robust)

                y_interp = y_interp.ravel()

            else:
                if should_do_gaussian_interpolation:
                    interpolator = scipy.interpolate.interp1d(
                        x ** 2,
                        np.log(y),
                        fill_value="extrapolate",
                        assume_sorted=True,
                    )
                    y_interp = np.exp(interpolator(r ** 2))
                else:  # linear
                    interpolator = scipy.interpolate.interp1d(
                        x,
                        y,
                        fill_value="extrapolate",
                        assume_sorted=True,
                    )
                    y_interp = interpolator(r)

            return 2 * np.pi * q, C @ (y_interp / m1) * m2

        else:  # inverse transform

            if dr is not None:
                q_max = 1 / (2 * dr)
            else:
                q_max = max(x) / (2 * np.pi)

            r_max = c0[n - 1] / (2 * np.pi * q_max)  # Maximum radius

            r = c0.T * r_max / c0[n - 1]  # Radius vector
            q = c0.T / (2 * np.pi * r_max)  # Frequency vector
            q = 2 * np.pi * q

            m1 = (abs(bessel_j1(c0)) / r_max).T  # m1 prepares input vector for transformation
            m2 = m1 * r_max / q_max  # m2 prepares output vector for display
            # end preparations for Hankel transform

            interpolator = scipy.interpolate.interp1d(
                x,
                y,
                fill_value="extrapolate",
                assume_sorted=True,
            )

            return r, C @ (interpolator(q) / m2) * m1


class SolutionSFCSMeasurement(TDCPhotonDataMixin, AngularScanMixin):
    """Doc."""

    NAN_PLACEBO: int
    DUMP_PATH = Path("C:/temp_sfcs_data/")

    def __init__(self, name=""):
        self.name = name
        self.data = []  # list to hold the data of each file
        self.cf = dict()
        self.xcf = dict()
        self.is_data_dumped = False
        self.scan_type: str
        self.duration_min: float = None

    @file_utilities.rotate_data_to_disk(does_modify_data=True)
    def read_fpga_data(
        self,
        file_path_template: Union[str, Path],
        file_selection: str = "Use All",
        should_plot=False,
        **kwargs,
    ) -> None:
        """Processes a complete FCS measurement (multiple files)."""

        if not file_selection:
            file_selection = "Use All"
        file_paths = file_utilities.prepare_file_paths(Path(file_path_template), file_selection)
        self.n_paths = len(file_paths)
        self.file_path_template = file_path_template
        *_, self.template = Path(file_path_template).parts
        self.name_on_disk = re.sub("\\*", "", re.sub("_[*]", "", self.template))

        print("\nLoading FPGA data from disk -")
        print(f"Template path: '{file_path_template}'")
        print(f"Files: {self.n_paths}, Selection: '{file_selection}'\n")

        # data processing
        self.data = self.process_all_data(file_paths, **kwargs)

        # count the files and ensure there's at least one file
        self.n_files = len(self.data)
        if not self.n_files:
            raise RuntimeError(
                f"Loading FPGA data catastrophically failed ({self.n_paths}/{self.n_paths} files skipped)."
            )

        # aggregate images and ROIs for sFCS
        if self.scan_type == "circle":
            self.scan_image = np.vstack(tuple(p.image for p in self.data))

        #            # TODO: do same as for angular (gated background)
        #            bg_corr_array = np.empty((len(self.data), self.scan_settings["samples_per_circle"]))
        #            for idx, p in enumerate(self.data):
        #                bg_corr_array[idx] = p.bg_line_corr_list["corrfunc"]
        #            avg_bg_corr = bg_corr_array.mean(axis=0)
        #            self.bg_corr_list = [dict(lag=p.bg_line_corr_list["lag"], corrfunc=avg_bg_corr)]
        #            # /TODO: do same as for angular (gated background)

        if self.scan_type == "angular":
            # aggregate images and ROIs for angular sFCS
            self.scan_images_dstack = np.dstack(tuple(p.image for p in self.data))
            self.roi_list = [p.roi for p in self.data]

        # calculate average count rate
        self.avg_cnt_rate_khz = np.mean([p.avg_cnt_rate_khz for p in self.data])
        try:
            self.std_cnt_rate_khz = np.std([p.avg_cnt_rate_khz for p in self.data], ddof=1)
        except RuntimeWarning:  # single file
            self.std_cnt_rate_khz = 0.0

        calc_duration_mins = (
            sum([np.diff(p.pulse_runtime).sum() for p in self.data]) / self.laser_freq_hz / 60
        )
        if self.duration_min is not None:
            if abs(calc_duration_mins - self.duration_min) > self.duration_min * 0.05:
                print(
                    f"Attention! calculated duration ({calc_duration_mins} mins) is significantly different than the set duration ({self.duration_min} min). Using calculated.\n"
                )
        else:
            print(f"Calculating duration (not supplied): {calc_duration_mins:.1f} mins\n")
        self.requested_duration_min = self.duration_min
        self.duration_min = calc_duration_mins

        # done with loading
        print(f"Finished loading FPGA data ({self.n_files}/{self.n_paths} files used).\n")

        # plotting of scan image and ROI
        if should_plot:
            print("Displaying scan images...", end=" ")
            if self.scan_type == "angular":
                with Plotter(
                    subplots=(1, self.n_files), fontsize=8, should_force_aspect=True
                ) as axes:
                    if not hasattr(
                        axes, "size"
                    ):  # if axes is not an ndarray (only happens if reading just one file)
                        axes = np.array([axes])
                    for file_idx, (ax, image, roi) in enumerate(
                        zip(axes, np.moveaxis(self.scan_images_dstack, -1, 0), self.roi_list)
                    ):
                        ax.set_title(f"file #{file_idx+1} of\n'{self.name}' measurement")
                        ax.set_xlabel("Pixel Number")
                        ax.set_ylabel("Line Number")
                        ax.imshow(image, interpolation="none")
                        ax.plot(roi["col"], roi["row"], color="white")
            elif self.scan_type == "circle":
                # TODO: FILL ME IN (plotting in jupyter notebook, same as above angular scan stuff)
                pass
            print("Done.\n")

    @timer(int(1e4))
    def process_all_data(
        self,
        file_paths: List[Path],
        should_parallel_process: bool = False,
        run_duration=None,
        n_runs_requested=60,
        **kwargs,
    ) -> List[TDCPhotonData]:
        """Doc."""

        self.get_general_properties(file_paths[0], **kwargs)

        # parellel processing
        if should_parallel_process and len(file_paths) > 20:
            N_CORES = mp.cpu_count() // 2 - 1  # /2 due to hyperthreading, -1 to leave one free
            func = partial(self.process_data_file, is_verbose=True, **kwargs)
            print(f"Parallel processing using {N_CORES} CPUs/processes.")
            with mp.get_context("spawn").Pool(N_CORES) as pool:
                data = list(pool.map(func, file_paths))

        # serial processing (default)
        else:
            data = []
            for file_path in file_paths:
                # Processing data
                p = self.process_data_file(file_path, is_verbose=True, **kwargs)
                # Appending data to self
                if p is not None:
                    data.append(p)

        if run_duration is None:  # auto determination of run duration
            total_duration_estimate = 0
            for p in data:
                time_stamps = np.diff(p.pulse_runtime).astype(np.int32)
                mu = np.median(time_stamps) / np.log(2)
                total_duration_estimate = (
                    total_duration_estimate + mu * len(p.pulse_runtime) / self.laser_freq_hz
                )
            self.run_duration = total_duration_estimate / n_runs_requested
        else:  # use supplied value
            self.run_duration = run_duration

        return data

    def get_general_properties(
        self, file_path: Path = None, file_dict: dict = None, **kwargs
    ) -> None:
        """Get general measurement properties from the first data file"""

        if file_path is not None:  # Loading file from disk
            file_dict = file_utilities.load_file_dict(file_path, **kwargs)

        full_data = file_dict["full_data"]

        self.afterpulse_params = file_dict["system_info"]["afterpulse_params"]
        self.detector_settings = full_data.get("detector_settings")
        self.delayer_settings = full_data.get("delayer_settings")
        self.laser_freq_hz = int(full_data["laser_freq_mhz"] * 1e6)
        self.fpga_freq_hz = int(full_data["fpga_freq_mhz"] * 1e6)
        with suppress(KeyError):
            self.duration_min = full_data["duration_s"] / 60

        # Detector Gate (calibrated - actual gate can be inferred from TDC calibration and compared)
        if self.delayer_settings is not None and getattr(self.delayer_settings, "is_gated", False):
            lower_detector_gate_ns = (
                self.delayer_settings.effective_delay_ns - self.delayer_settings.sync_delay_ns
            )
        else:
            lower_detector_gate_ns = 0
        if self.detector_settings is not None:
            gate_width_ns = self.detector_settings.gate_width_ns
            self.detector_gate_ns = Limits(
                lower_detector_gate_ns, lower_detector_gate_ns + gate_width_ns
            )
        else:
            self.detector_gate_ns = Limits(0, np.inf)

        # sFCS
        if scan_settings := full_data.get("scan_settings"):
            self.scan_type = scan_settings["pattern"]
            self.scan_settings = scan_settings
            self.v_um_ms = self.scan_settings["speed_um_s"] * 1e-3
            if self.scan_type == "circle":  # Circular sFCS
                self.ao_sampling_freq_hz = self.scan_settings.get("ao_sampling_freq_hz", int(1e4))
                self.diameter_um = self.scan_settings.get("diameter_um", 50)
            elif self.scan_type == "angular":  # Angular sFCS
                self.LINE_END_ADDER = 1000

        # FCS
        else:
            self.scan_type = "static"

    def process_data_file(
        self, file_path: Path = None, file_dict: dict = None, **kwargs
    ) -> TDCPhotonData:
        """Doc."""

        if file_dict is not None:
            self.get_general_properties(file_dict=file_dict, **kwargs)

        if file_path is not None:  # Loading file from disk
            *_, template = file_path.parts
            file_idx = int(re.split("_(\\d+)\\.", template)[1])
            print(
                f"Loading and processing file No. {file_idx} ({self.n_paths} files): '{template}'...",
                end=" ",
            )
            try:
                file_dict = file_utilities.load_file_dict(file_path)
            except FileNotFoundError:
                print(f"File '{file_path}' not found. Ignoring.")
        else:  # using supplied file_dict
            file_idx = 1

        full_data = file_dict["full_data"]

        # sFCS
        if scan_settings := full_data.get("scan_settings"):
            if scan_settings["pattern"] == "circle":  # Circular sFCS
                p = self._process_circular_scan_data_file(full_data, file_idx, **kwargs)
            elif scan_settings["pattern"] == "angular":  # Angular sFCS
                p = self._process_angular_scan_data_file(full_data, file_idx, **kwargs)

        # FCS
        else:
            self.scan_type = "static"
            p = self._process_static_data_file(full_data, file_idx, **kwargs)

        if file_path is not None and p is not None:
            p.file_path = file_path
            print("Done.\n")

        return p

    def _process_static_data_file(self, full_data, file_idx, **kwargs) -> TDCPhotonData:
        """
        Processes a single static FCS data file ('full_data').
        Returns the processed results as a 'TDCPhotonData' object.
        '"""

        p = self.convert_fpga_data_to_photons(
            full_data["byte_data"],
            version=full_data["version"],
            locate_outliers=True,
            **kwargs,
        )

        p.file_num = file_idx
        p.avg_cnt_rate_khz = full_data["avg_cnt_rate_khz"]

        return p

    def _process_circular_scan_data_file(self, full_data, file_idx, **kwargs) -> TDCPhotonData:
        """
        Processes a single circular sFCS data file ('full_data').
        Returns the processed results as a 'TDCPhotonData' object.
        '"""

        p = self.convert_fpga_data_to_photons(
            full_data["byte_data"],
            version=full_data["version"],
            locate_outliers=True,
            **kwargs,
        )

        p.file_num = file_idx
        p.avg_cnt_rate_khz = full_data["avg_cnt_rate_khz"]

        scan_settings = full_data["scan_settings"]
        p.ao_sampling_freq_hz = int(scan_settings["ao_sampling_freq_hz"])
        p.circle_freq_hz = scan_settings["circle_freq_hz"]

        print("Converting circular scan to image...", end=" ")
        pulse_runtime = p.pulse_runtime
        cnt, self.scan_settings["samples_per_circle"] = _sum_scan_circles(
            pulse_runtime,
            self.laser_freq_hz,
            p.ao_sampling_freq_hz,
            p.circle_freq_hz,
        )

        p.image = cnt

        return p

    def _process_angular_scan_data_file(
        self,
        full_data,
        file_idx,
        should_fix_shift=True,
        roi_selection="auto",
        **kwargs,
    ) -> TDCPhotonData:
        """
        Processes a single angular sFCS data file ('full_data').
        Returns the processed results as a 'TDCPhotonData' object.
        '"""

        p = self.convert_fpga_data_to_photons(
            full_data["byte_data"], version=full_data["version"], is_verbose=True
        )

        p.file_num = file_idx
        p.avg_cnt_rate_khz = full_data["avg_cnt_rate_khz"]

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
                pix_shift = self._fix_data_shift(sec_cnt.copy())
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
        p.coarse = np.hstack(
            (
                np.full(line_starts_runtime.size, self.NAN_PLACEBO, dtype=np.int16),
                np.full(line_stops_runtime.size, self.NAN_PLACEBO, dtype=np.int16),
                p.coarse,
            )
        )[sorted_idxs]
        p.coarse2 = np.hstack(
            (
                np.full(line_starts_runtime.size, self.NAN_PLACEBO, dtype=np.int16),
                np.full(line_stops_runtime.size, self.NAN_PLACEBO, dtype=np.int16),
                p.coarse2,
            )
        )[sorted_idxs]
        p.fine = np.hstack(
            (
                np.full(line_starts_runtime.size, self.NAN_PLACEBO, dtype=np.int16),
                np.full(line_stops_runtime.size, self.NAN_PLACEBO, dtype=np.int16),
                p.fine,
            )
        )[sorted_idxs]

        p.image = cnt
        p.roi = roi

        # reverse rows again
        bw[1::2, :] = np.flip(bw[1::2, :], 1)
        p.bw_mask = bw

        # for getting image line correlation to subtract trends later
        p.line_limits = Limits(line_num[line_num > 0].min(), line_num.max())

        return p

    def correlate_and_average(self, **kwargs):
        """High level function for correlating and averaging any data."""

        CF = self.correlate_data(**kwargs)
        CF.average_correlation(**kwargs)

    @file_utilities.rotate_data_to_disk()
    def correlate_data(
        self,
        cf_name=None,
        gate_ns=Limits(0, np.inf),
        afterpulse_params=None,
        external_afterpulse=None,
        should_subtract_bg_corr=True,
        is_verbose=False,
        min_time_frac=0.5,
        **kwargs,
    ) -> CorrFunc:
        """
        High level function for correlating any type of data (e.g. static, angular scan, circular scan...)
        Returns a 'CorrFunc' object.
        Data attribute is possibly rotated from/to disk.
        """

        self.min_duration_frac = min_time_frac

        # Unite TDC gate and detector gate
        tdc_gate_ns = Limits(gate_ns)
        if tdc_gate_ns != Limits(0, np.inf):
            effective_lower_gate_ns = max(tdc_gate_ns.lower - self.detector_gate_ns.lower, 0)
            effective_upper_gate_ns = min(tdc_gate_ns.upper, self.detector_gate_ns.upper)
            gate_ns = Limits(effective_lower_gate_ns, effective_upper_gate_ns)

        # correlate data
        if is_verbose:
            print(
                f"Correlating {self.scan_type} data ({self.name} [{gate_ns} ns gating]):", end=" "
            )

        (
            ts_split_list,
            total_duration,
            skipped_duration,
            bg_line_corr_list,
        ) = self._prepare_corr_split_list(is_verbose=is_verbose, gate_ns=gate_ns)

        if self.scan_type in {"static", "circle"}:
            corr_type = CorrelatorType.PH_DELAY_CORRELATOR
        elif self.scan_type == "angular":
            corr_type = CorrelatorType.PH_DELAY_CORRELATOR_LINES

        CF = CorrFunc(cf_name, gate_ns, self.laser_freq_hz)
        CF.correlate_measurement(
            corr_type,
            ts_split_list,
            afterpulse_params if afterpulse_params is not None else self.afterpulse_params,
            bg_line_corr_list if should_subtract_bg_corr else [],
            is_verbose=is_verbose,
            external_afterpulse=external_afterpulse,
            **kwargs,
        )

        try:
            skipped_ratio = skipped_duration / total_duration
        except RuntimeWarning:
            print("Whole measurement was skipped! Something's wrong...", end=" ")
        else:
            if is_verbose:
                if skipped_duration:
                    print(f"Skipped/total duration: {skipped_ratio:.1%}", end=" ")
                print("- Done.")

        CF.total_duration = total_duration
        try:  # temporal to spatial conversion, if scanning
            CF.vt_um = self.v_um_ms * CF.lag
        except AttributeError:
            CF.vt_um = CF.lag  # static

        # name the Corrfunc object
        if cf_name is not None:
            self.cf[cf_name] = CF
        else:
            self.cf[f"gSTED {CF.gate_ns}"] = CF

        return CF

    def _prepare_corr_split_list(self, is_verbose=True, **kwargs):
        """Doc."""

        if is_verbose:
            print("Preparing files for software correlator...", end=" ")

        ts_split_list = []
        total_duration = 0
        total_skipped_duration = 0
        bg_line_corr_list = []

        for p in self.data:
            (
                file_ts_split_list,
                file_duration,
                file_skipped_duration,
                file_bg_line_corr_list,
            ) = self._prepare_file_corr_split_list(p, is_verbose=is_verbose, **kwargs)

            ts_split_list += file_ts_split_list
            total_duration += file_duration

            total_skipped_duration += file_skipped_duration

            bg_line_corr_list += file_bg_line_corr_list

        if is_verbose:
            print("Done.")

        # TODO: units
        return ts_split_list, total_duration, total_skipped_duration, bg_line_corr_list

    def _prepare_file_corr_split_list(
        self,
        p: TDCPhotonData,
        gate_ns=Limits(0, np.inf),
        min_time_frac=0.5,
        is_verbose=False,
        **kwargs,
    ):
        """Doc."""

        file_ts_split_list = []
        file_duration = 0.0
        if self.scan_type in {"static", "circle"}:
            file_skipped_duration = 0
            file_bg_line_corr_list = []
            # Ignore short segments (default is below half the run_duration)
            for se_idx, (se_start, se_end) in enumerate(p.all_section_edges):
                segment_time = (
                    p.pulse_runtime[se_end] - p.pulse_runtime[se_start]
                ) / self.laser_freq_hz
                if segment_time < min_time_frac * self.run_duration:
                    if is_verbose:
                        print(
                            f"Skipping segment {se_idx} of file {p.file_num} - too short ({segment_time:.2f}s).",
                            end=" ",
                        )
                    file_skipped_duration += segment_time
                    continue

                pulse_runtime = p.pulse_runtime[se_start : se_end + 1]

                # Gating
                if hasattr(p, "delay_time"):
                    delay_time = p.delay_time[se_start : se_end + 1]
                    j_gate = gate_ns.valid_indices(delay_time)
                    pulse_runtime = pulse_runtime[j_gate]

                elif gate_ns != (0, np.inf):  # TODO
                    raise RuntimeError(
                        f"A gate '{gate_ns}' was specified for uncalibrated TDC data."
                    )

                # split into segments of approx time of run_duration
                n_splits = div_ceil(segment_time, self.run_duration)
                time_stamps = np.hstack(([0], np.diff(pulse_runtime).astype(np.int32)))

                for j in range(n_splits):
                    file_ts_split_list.append(
                        self.prepare_timestamps(time_stamps, j, n_splits=n_splits)
                    )
                    file_duration += self.get_split_duration(file_ts_split_list[-1])

        elif self.scan_type == "angular":
            line_num = p.line_num
            if hasattr(p, "delay_time"):  # if measurement is TDC-calibrated
                # NaNs mark line starts/ends
                j_gate = gate_ns.valid_indices(p.delay_time) | np.isnan(p.delay_time)
                pulse_runtime = p.pulse_runtime[j_gate]
                line_num = p.line_num[j_gate]

            elif gate_ns != (0, np.inf):  # TODO
                raise RuntimeError(f"A gate '{gate_ns}' was specified for uncalibrated TDC data.")
            else:
                pulse_runtime = p.pulse_runtime

            # get (gated) background correlation
            possibly_gated_image, *_ = self._convert_angular_scan_to_image(
                pulse_runtime,
                self.laser_freq_hz,
                p.ao_sampling_freq_hz,
                p.samples_per_line,
                p.n_lines,
            )
            file_bg_line_corr_list = self._bg_line_correlations(
                possibly_gated_image, p.bw_mask, p.line_limits, p.ao_sampling_freq_hz
            )

            time_stamps = np.hstack(([0], np.diff(pulse_runtime).astype(np.int32)))
            for j in p.line_limits.as_range():
                file_ts_split_list.append(
                    self.prepare_timestamps(time_stamps, j, line_num=line_num)
                )
                file_duration += self.get_split_duration(file_ts_split_list[-1])
                # TODO: what's the difference between segment_time and file_duration?
                file_skipped_duration = 0

        return file_ts_split_list, file_duration, file_skipped_duration, file_bg_line_corr_list

    @file_utilities.rotate_data_to_disk()
    def cross_correlate_data(
        self,
        cf_name=None,
        gate1_ns=Limits(0, np.inf),
        gate2_ns=Limits(0, np.inf),
        afterpulse_params=None,
        should_subtract_bg_corr=True,
        is_verbose=True,
        min_time_frac=0.5,
        **kwargs,
    ) -> List[CorrFunc]:
        """
        High level function for correlating any type of data (e.g. static, angular scan, circular scan...)
        Returns a 'CorrFunc' object.
        Data attribute is possibly rotated from/to disk.
        """

        self.min_duration_frac = min_time_frac

        # Unite TDC gates and detector gates
        for i in (1, 2):
            gate_ns = locals()[f"gate{i}_ns"]
            tdc_gate_ns = Limits(gate_ns)
            if tdc_gate_ns != Limits(0, np.inf):
                effective_lower_gate_ns = max(tdc_gate_ns.lower - self.detector_gate_ns.lower, 0)
                effective_upper_gate_ns = min(tdc_gate_ns.upper, self.detector_gate_ns.upper)
                locals()[f"gate{i}_ns"] = Limits(effective_lower_gate_ns, effective_upper_gate_ns)

        # correlate data
        if is_verbose:
            print(
                f"Cross-Correlating {self.scan_type} data ({self.name} [{gate1_ns} ns vs. {gate2_ns} ns]):",
                end=" ",
            )

        (
            ts_split_lists_tuple,
            total_duration,
            skipped_duration,
            bg_line_corr_lists_tuple,
        ) = self._prepare_xcorr_split_list(
            is_verbose=is_verbose, gate1_ns=gate1_ns, gate2_ns=gate2_ns
        )
        ts_split_list_AB, ts_split_list_A, ts_split_list_B = ts_split_lists_tuple
        (
            bg_line_corr_list_AB,
            bg_line_corr_list_BA,
            bg_line_corr_list_A,
            bg_line_corr_list_B,
        ) = bg_line_corr_lists_tuple

        if self.scan_type in {"static", "circle"}:
            ts_split_list_BA = [ts_split_AB[[0, 2, 1], :] for ts_split_AB in ts_split_list_AB]
            corr_type = CorrelatorType.PH_DELAY_CORRELATOR
            xcorr_type = CorrelatorType.PH_DELAY_CROSS_CORRELATOR
        elif self.scan_type == "angular":
            ts_split_list_BA = [ts_split_AB[[0, 2, 1, 3], :] for ts_split_AB in ts_split_list_AB]
            corr_type = CorrelatorType.PH_DELAY_CORRELATOR_LINES
            xcorr_type = CorrelatorType.PH_DELAY_CROSS_CORRELATOR_LINES

        CF_names = ("AA", "BB", "AB", "BA")
        CF_list = [CorrFunc(f"{cf_name}_{xx}", gate1_ns, self.laser_freq_hz) for xx in CF_names]
        corr_type_list = [corr_type] * 2 + [xcorr_type] * 2
        ts_split_lists = [
            ts_split_list_A,
            ts_split_list_B,
            ts_split_list_AB,
            ts_split_list_BA,
        ]
        bg_line_corr_lists = [
            bg_line_corr_list_A,
            bg_line_corr_list_B,
            bg_line_corr_list_AB,
            bg_line_corr_list_BA,
        ]

        for CF, correlator_type, ts_split_list, bg_line_corr_list in zip(
            CF_list, corr_type_list, ts_split_lists, bg_line_corr_lists
        ):
            CF.correlate_measurement(
                correlator_type,
                ts_split_list,
                afterpulse_params if afterpulse_params is not None else self.afterpulse_params,
                bg_line_corr_list if should_subtract_bg_corr else [],
                is_verbose=is_verbose,
                **kwargs,
            )
            CF.total_duration = total_duration
            try:  # temporal to spatial conversion, if scanning
                CF.vt_um = self.v_um_ms * CF.lag
            except AttributeError:
                CF.vt_um = CF.lag  # static

        try:
            skipped_ratio = skipped_duration / total_duration
        except RuntimeWarning:
            print("Whole measurement was skipped! Something's wrong...", end=" ")
        else:
            if is_verbose:
                if skipped_duration:
                    print(f"Skipped/total duration: {skipped_ratio:.1%}", end=" ")
                print("- Done.")

        # name the Corrfunc object
        if cf_name is not None:
            for xx, CF in zip(CF_names, CF_list):
                self.xcf[f"{cf_name}_{xx} ({gate1_ns} vs. {gate2_ns} ns)"] = CF
        else:
            self.xcf[f"{xx} ({gate1_ns} vs. {gate2_ns} ns)"] = CF

        return CF_list

    def _prepare_xcorr_split_list(self, is_verbose=True, **kwargs):
        """Doc."""

        if is_verbose:
            print("Preparing files for software correlator...", end=" ")

        ts_split_list = []
        ts1_split_list = []
        ts2_split_list = []
        total_duration = 0.0
        total_skipped_duration = 0.0
        bg_line_corr12_list = []
        bg_line_corr21_list = []
        bg_line_corr1_list = []
        bg_line_corr2_list = []

        for p in self.data:
            (
                (file_ts_split_list, file_ts1_split_list, file_ts2_split_list),
                file_duration,
                file_skipped_duration,
                (
                    file_bg_line_corr12_list,
                    file_bg_line_corr21_list,
                    file_bg_line_corr1_list,
                    file_bg_line_corr2_list,
                ),
            ) = self._prepare_file_xcorr_split_list(p, is_verbose=is_verbose, **kwargs)

            ts_split_list += file_ts_split_list
            ts1_split_list += file_ts1_split_list
            ts2_split_list += file_ts2_split_list

            total_duration += file_duration
            total_skipped_duration += file_skipped_duration

            bg_line_corr12_list += file_bg_line_corr12_list
            bg_line_corr21_list += file_bg_line_corr21_list
            bg_line_corr1_list += file_bg_line_corr1_list
            bg_line_corr2_list += file_bg_line_corr2_list

        if is_verbose:
            print("Done.")

        # TODO units??
        return (
            (ts_split_list, ts1_split_list, ts2_split_list),
            total_duration,
            total_skipped_duration,
            (bg_line_corr12_list, bg_line_corr21_list, bg_line_corr1_list, bg_line_corr2_list),
        )

    def _prepare_file_xcorr_split_list(
        self,
        p: TDCPhotonData,
        gate1_ns=Limits(0, np.inf),
        gate2_ns=Limits(0, np.inf),
        min_time_frac=0.5,
        is_verbose=False,
        **kwargs,
    ):
        """Doc."""

        file_ts_split_list = []
        file_ts1_split_list = []
        file_ts2_split_list = []
        file_duration = 0.0
        file_skipped_duration = 0.0
        file_bg_line_corr12_list = []
        file_bg_line_corr21_list = []
        file_bg_line_corr1_list = []
        file_bg_line_corr2_list = []

        if self.scan_type in {"static", "circle"}:
            for se_idx, (se_start, se_end) in enumerate(p.all_section_edges):
                # split into segments of approx time of run_duration
                segment_time = (
                    p.pulse_runtime[se_end] - p.pulse_runtime[se_start]
                ) / self.laser_freq_hz
                if segment_time < min_time_frac * self.run_duration:
                    print(
                        f"Skipping segment {se_idx} of file {p.file_num} - too short ({segment_time:.2f}s).",
                        end=" ",
                    )
                    file_skipped_duration = file_skipped_duration + segment_time
                    continue

                pulse_runtime = p.pulse_runtime[se_start : se_end + 1]
                delay_time = p.delay_time[se_start : se_end + 1]
                j_gate1 = gate1_ns.valid_indices(delay_time)
                j_gate2 = gate2_ns.valid_indices(delay_time)
                pulse_runtime1 = pulse_runtime[j_gate1]
                pulse_runtime2 = pulse_runtime[j_gate2]
                pulse_runtime = np.vstack((pulse_runtime, j_gate1, j_gate2))
                j_gate = j_gate1 | j_gate2
                pulse_runtime = pulse_runtime[:, j_gate]
                ts = pulse_runtime.astype(np.int32)
                ts[0] = np.hstack(([0], np.diff(pulse_runtime[0]).astype(np.int32)))
                ts = ts.astype(np.int32)
                ts1 = np.hstack(([0], np.diff(pulse_runtime1).astype(np.int32)))
                ts2 = np.hstack(([0], np.diff(pulse_runtime2).astype(np.int32)))

                n_splits = div_ceil(segment_time, self.run_duration)
                for j in range(n_splits):
                    file_ts_split_list.append(self.prepare_timestamps(ts, j, n_splits=n_splits))
                    file_ts1_split_list.append(self.prepare_timestamps(ts1, j, n_splits=n_splits))
                    file_ts2_split_list.append(self.prepare_timestamps(ts2, j, n_splits=n_splits))

                    file_duration += self.get_split_duration(file_ts_split_list[-1])

        elif self.scan_type == "angular":
            line_num = np.around(p.line_num)
            j_gate1 = gate1_ns.valid_indices(p.delay_time) | np.isnan(
                p.delay_time
            )  # NaNs mark line starts/ends
            j_gate2 = gate2_ns.valid_indices(p.delay_time) | np.isnan(
                p.delay_time
            )  # NaNs mark line starts/ends
            pulse_runtime1 = p.pulse_runtime[j_gate1]
            pulse_runtime2 = p.pulse_runtime[j_gate2]
            pulse_runtime = np.vstack((p.pulse_runtime, j_gate1, j_gate2))
            j_gate = j_gate1 | j_gate2
            pulse_runtime = pulse_runtime[:, j_gate]
            ts = pulse_runtime.astype(np.int32)
            ts[0] = np.hstack(([0], np.diff(pulse_runtime[0]).astype(np.int32)))
            ts1 = np.hstack(([0], np.diff(pulse_runtime1).astype(np.int32)))
            ts2 = np.hstack(([0], np.diff(pulse_runtime2).astype(np.int32)))
            line_num1 = line_num[j_gate1]
            line_num2 = line_num[j_gate2]
            line_num = line_num[j_gate]

            # get (gated) background correlation
            image1, *_ = self._convert_angular_scan_to_image(
                pulse_runtime1,
                self.laser_freq_hz,
                p.ao_sampling_freq_hz,
                p.samples_per_line,
                p.n_lines,
            )
            image2, *_ = self._convert_angular_scan_to_image(
                pulse_runtime2,
                self.laser_freq_hz,
                p.ao_sampling_freq_hz,
                p.samples_per_line,
                p.n_lines,
            )

            file_bg_line_corr12_list = self._bg_line_correlations(
                image1, p.bw_mask, p.line_limits, p.ao_sampling_freq_hz, image2=image2
            )
            file_bg_line_corr21_list = self._bg_line_correlations(
                image2, p.bw_mask, p.line_limits, p.ao_sampling_freq_hz, image2=image1
            )
            file_bg_line_corr1_list = self._bg_line_correlations(
                image1, p.bw_mask, p.line_limits, p.ao_sampling_freq_hz
            )
            file_bg_line_corr2_list = self._bg_line_correlations(
                image2, p.bw_mask, p.line_limits, p.ao_sampling_freq_hz
            )

            for j in p.line_limits.as_range():
                file_ts_split_list.append(self.prepare_timestamps(ts, j, line_num=line_num))
                file_ts1_split_list.append(self.prepare_timestamps(ts1, j, line_num=line_num1))
                file_ts2_split_list.append(self.prepare_timestamps(ts2, j, line_num=line_num2))

                file_duration += self.get_split_duration(file_ts_split_list[-1])

                # TODO: what's the difference between segment_time and file_duration?
                file_skipped_duration = 0

        return (
            (file_ts_split_list, file_ts1_split_list, file_ts2_split_list),
            file_duration,
            file_skipped_duration,
            (
                file_bg_line_corr12_list,
                file_bg_line_corr21_list,
                file_bg_line_corr1_list,
                file_bg_line_corr2_list,
            ),
        )

    def prepare_timestamps(self, ts_in, idx, line_num=None, n_splits=None):
        """A utility function for linear scan data"""

        if self.scan_type in {"static", "circle"}:
            splits = np.linspace(0, ts_in.size, n_splits + 1, dtype=np.int32)
            try:
                return ts_in[:, splits[idx] : splits[idx + 1]]
            except IndexError:  # 1D
                return ts_in[splits[idx] : splits[idx + 1]]

        elif self.scan_type == "angular":
            valid = (line_num == idx).astype(np.int8)
            valid[line_num == -idx] = -1
            valid[line_num == -idx - self.LINE_END_ADDER] = -2

            #  remove photons from wrong lines
            try:
                ts_out = ts_in[:, valid != 0]
            except IndexError:  # 1D
                ts_out = ts_in[valid != 0]
            valid = valid[valid != 0]

            # the first photon in line measures the time from line start and the line end (-2) finishes the duration of the line
            # check that we start with the line beginning and not its end
            if valid[0] != -1:
                # remove photons till the first found beginning
                j_start = np.where(valid == -1)[0]
                if len(j_start) > 0:
                    ts_out = ts_out[j_start[0] :]
                    valid = valid[j_start[0] :]

            # check that we stop with the line ending and not its beginning
            if valid[-1] != -2:
                # remove photons after the last found ending
                j_end = np.where(valid == -2)[0]

                if len(j_end) > 0:
                    *_, j_end_last = j_end
                    try:
                        ts_out = ts_out[:, : j_end_last + 1]
                    except IndexError:  # 1D
                        ts_out = ts_out[: j_end_last + 1]
                    valid = valid[: j_end_last + 1]

            return np.vstack((ts_out, valid))

    def get_split_duration(self, ts_split: np.ndarray) -> float:
        """Doc."""

        if self.scan_type in {"static", "circle"}:
            split_duration = ts_split.sum() / self.laser_freq_hz
        elif self.scan_type == "angular":
            ts, *_, valid = ts_split
            split_duration = ts[(valid == 1) | (valid == -2)].sum() / self.laser_freq_hz
        return split_duration

    def plot_correlation_functions(
        self,
        x_field="lag",
        y_field="normalized",
        x_scale="log",
        y_scale="linear",
        ylim=(-0.20, 1.4),
        plot_kwargs={},
        **kwargs,
    ) -> List[str]:
        """Doc."""

        with Plotter(super_title=f"'{self.name.capitalize()}' - ACFs", **kwargs) as ax:
            legend_labels = []
            kwargs["parent_ax"] = ax
            for cf_name, cf in {**self.cf, **self.xcf}.items():
                cf.plot_correlation_function(
                    x_field=x_field,
                    y_field=y_field,
                    x_scale=x_scale,
                    y_scale=y_scale,
                    ylim=ylim,
                    plot_kwargs=plot_kwargs,
                    **kwargs,
                )
                legend_labels.append(cf_name)
            ax.legend(legend_labels)

        return legend_labels

    def calculate_structure_factors(self, plot_kwargs={}, **kwargs) -> None:
        """Doc."""

        # calculation
        for cf in self.cf.values():
            #            if not hasattr(cf, "structure_factor"):
            cf.calculate_structure_factor(**kwargs)

        #  plotting
        with Plotter(
            #            xlim=Limits(q[1], np.pi / min(w_xy)),
            super_title=f"{self.name.capitalize()}: Structure Factor ($S(q)$)",
            **kwargs,
        ) as ax:
            legend_labels = []
            for name, cf in self.cf.items():
                s = cf.structure_factor
                ax.set_title("Gaussian vs. linear\nInterpolation")
                ax.loglog(s.q, np.vstack((s.sq, s.sq_linear_interp)).T, **plot_kwargs)
                legend_labels += [
                    f"{name}: Gaussian Interpolation",
                    f"{name}: Linear Interpolation",
                ]
            ax.set_xlabel("$q$ $(\\mu m^{-1})$")
            ax.set_ylabel("$S(q)$")
            ax.legend(legend_labels)

    def dump_or_load_data(self, should_load: bool, method_name=None) -> None:
        """
        Load or save the 'data' attribute.
        (relieve RAM - important during multiple-experiment analysis)
        """

        with suppress(AttributeError):
            # AttributeError - name_on_disk is not defined (happens when doing alignment, for example)
            if should_load:  # loading data
                if self.is_data_dumped:
                    logging.debug(
                        f"{method_name}: Loading dumped data '{self.name_on_disk}' from '{self.DUMP_PATH}'."
                    )
                    with suppress(FileNotFoundError):
                        self.data = file_utilities.load_object(self.DUMP_PATH / self.name_on_disk)
                        self.is_data_dumped = False
            else:  # saving data
                is_saved = file_utilities.save_object(
                    self.data,
                    self.DUMP_PATH / self.name_on_disk,
                    obj_name="dumped data array",
                    element_size_estimate_mb=self.data[0].size_estimate_mb,
                )
                if is_saved:
                    self.data = []
                    self.is_data_dumped = True
                    logging.debug(
                        f"{method_name}: Dumped data '{self.name_on_disk}' to '{self.DUMP_PATH}'."
                    )


class ImageSFCSMeasurement(TDCPhotonDataMixin, CountsImageMixin):
    """Doc."""

    def __init__(self):
        pass

    def read_image_data(self, file_path, **kwargs) -> None:
        """Doc."""

        file_dict = file_utilities.load_file_dict(file_path)
        self.process_data_file(file_dict, **kwargs)

    def process_data_file(self, file_dict: dict, **kwargs) -> None:
        """Doc."""

        # store relevant attributes (add more as needed)
        self.laser_mode = file_dict.get("laser_mode")
        self.scan_params = file_dict.get("scan_params")

        # Get ungated image (excitation or sted)
        self.image_data = self.create_image_stack_data(file_dict)

        # gating stuff (TDC) - not yet implemented
        self.data = None


class SFCSExperiment(TDCPhotonDataMixin):
    """Doc."""

    UPPERֹ_ֹGATE_NS = 20

    def __init__(self, name):
        self.name = name
        self.confocal: SolutionSFCSMeasurement
        self.sted: SolutionSFCSMeasurement

    def load_experiment(
        self,
        confocal_template: Union[str, Path] = None,
        sted_template: Union[str, Path] = None,
        confocal=None,
        sted=None,
        should_plot=False,
        confocal_kwargs={},
        sted_kwargs={},
        **kwargs,
    ):
        """Doc."""

        if (
            (confocal_template is None)
            and (sted_template is None)
            and (confocal is None)
            and (sted is None)
        ):  # check if at least one measurement is available for laoding
            raise RuntimeError("Can't load experiment with no measurements!")

        # load measurements
        for meas_type in ("confocal", "sted"):
            measurement = locals()[meas_type]
            meas_template = locals()[f"{meas_type}_template"]
            meas_kwargs = locals()[f"{meas_type}_kwargs"]
            if measurement is None:
                if meas_template is not None:  # process measurement from template
                    self.load_measurement(
                        meas_type=meas_type,
                        file_path_template=meas_template,
                        should_plot=should_plot,
                        **meas_kwargs,
                        **kwargs,
                    )
                else:  # Use empty measuremnt by default
                    setattr(self, meas_type, SolutionSFCSMeasurement(name=meas_type))
            else:  # use supllied measurement
                setattr(self, meas_type, measurement)
                getattr(self, meas_type).name = meas_type  # remame supplied measurement

        if should_plot:
            super_title = f"Experiment '{self.name.capitalize()}' - All ACFs"
            with Plotter(subplots=(1, 2), super_title=super_title, **kwargs) as axes:
                self.plot_correlation_functions(
                    parent_ax=axes[0],
                    y_field="avg_cf_cr",
                    x_scale="log",
                    xlim=None,  # autoscale x axis
                )

                self.plot_correlation_functions(
                    parent_ax=axes[1],
                )

    def load_measurement(
        self,
        meas_type: str,
        file_path_template: Union[str, Path],
        should_plot: bool = False,
        plot_kwargs: dict = {},
        should_use_preprocessed=False,
        **kwargs,
    ):

        if "cf_name" not in kwargs:
            if meas_type == "confocal":
                kwargs["cf_name"] = "confocal"
            else:  # sted
                kwargs["cf_name"] = "sted"
        cf_name = kwargs["cf_name"]

        if kwargs.get(f"{meas_type}_file_selection"):
            kwargs["file_selection"] = kwargs[f"{meas_type}_file_selection"]

        measurement = SolutionSFCSMeasurement(name=meas_type)

        if should_use_preprocessed:
            try:
                file_path_template = Path(file_path_template)  # str-> Path if needed
                dir_path, file_template = file_path_template.parent, file_path_template.name
                # load pre-processed
                file_path = dir_path / "processed" / re.sub("_[*]", "", file_template)
                measurement = file_utilities.load_processed_solution_measurement(file_path)
                measurement.name = meas_type
                print(f"Loaded pre-processed {meas_type} measurement: '{file_path}'")
            except OSError:
                print(
                    f"Pre-processed {meas_type} measurement not found at: '{file_path}'. Processing data regularly."
                )

        if not measurement.cf:  # Process data
            measurement.read_fpga_data(
                file_path_template,
                should_plot=should_plot,
                **kwargs,
            )
            measurement.correlate_and_average(is_verbose=True, **kwargs)

        if should_plot:

            if (x_field := kwargs.get("x_field")) is None:
                if measurement.scan_type == "static":
                    x_field = "lag"
                else:  # angular or circular scan
                    x_field = "vt_um"

            super_title = f"'{self.name.capitalize()}' Experiment\n'{measurement.name.capitalize()}' Measurement - ACFs"
            with Plotter(super_title=super_title, ylim=(-100, None)) as ax:
                measurement.cf[cf_name].plot_correlation_function(
                    parent_ax=ax,
                    y_field="average_all_cf_cr",
                    x_field=x_field,
                    plot_kwargs=plot_kwargs,
                )
                measurement.cf[cf_name].plot_correlation_function(
                    parent_ax=ax, y_field="avg_cf_cr", x_field=x_field, plot_kwargs=plot_kwargs
                )
                ax.legend(["average_all_cf_cr", "avg_cf_cr"])

        setattr(self, meas_type, measurement)

    def save_processed_measurements(self):
        """Doc."""

        print("Saving processed measurements to disk...", end=" ")
        file_utilities.save_processed_solution_meas(
            self.confocal, self.confocal.file_path_template.parent
        )
        file_utilities.save_processed_solution_meas(self.sted, self.sted.file_path_template.parent)
        print("Done.")

    def calibrate_tdc(self, should_plot=True, **kwargs):
        """Doc."""

        if hasattr(self.confocal, "scan_type"):  # has confocal measurement (minimum requirement)

            self.confocal.calibrate_tdc(**kwargs)
            if hasattr(self.sted, "scan_type"):  # if both measurements quack as if loaded
                self.sted.calibrate_tdc(sync_coarse_time_to=self.confocal.tdc_calib, **kwargs)

            if should_plot:
                super_title = f"'{self.name.capitalize()}' Experiment\nTDC Calibration"
                if hasattr(self.sted, "scan_type"):  # if both measurements quack as if loaded
                    with Plotter(subplots=(2, 4), super_title=super_title, **kwargs) as axes:
                        self.confocal.plot_tdc_calibration(parent_axes=axes[:, :2])
                        self.sted.plot_tdc_calibration(parent_axes=axes[:, 2:])
                else:  # STED measurement not loaded
                    self.confocal.plot_tdc_calibration(super_title=super_title, **kwargs)

        else:
            raise RuntimeError("Can't calibrate TDC with no confocal measurements loaded!")

    def compare_lifetimes(
        self,
        normalization_type="Per Time",
        **kwargs,
    ):
        """Doc."""

        if hasattr(self.confocal, "tdc_calib"):  # if TDC calibration performed
            super_title = f"'{self.name.capitalize()}' Experiment\nLifetime Comparison"
            with Plotter(super_title=super_title, **kwargs) as ax:
                self.confocal.compare_lifetimes(
                    "confocal",
                    compare_to=dict(STED=self.sted),
                    normalization_type=normalization_type,
                    parent_ax=ax,
                )

    def add_gate(self, gate_ns: Tuple[float, float], should_plot=True, **kwargs):
        """Doc."""
        # TODO: this should be a method of SolutionSFCSMeasurement

        try:
            self.sted.correlate_and_average(
                cf_name=f"gSTED {gate_ns}", gate_ns=gate_ns, is_verbose=True, **kwargs
            )
            if should_plot:
                self.plot_correlation_functions(**kwargs)
        except AttributeError:
            # STED measurement not loaded
            raise RuntimeError(
                "Cannot add a gate if there's no STED measurement loaded to the experiment!"
            )

    def add_gates(self, gate_list: List[Tuple[float, float]], should_plot=True, **kwargs):
        """A convecience method for adding multiple gates."""

        for gate_ns in gate_list:
            self.add_gate(gate_ns, should_plot=False, **kwargs)
        if should_plot:
            self.plot_correlation_functions(**kwargs)

    def plot_correlation_functions(self, **kwargs):
        """Doc."""

        if hasattr(self.confocal, "scan_type"):
            ref_meas = self.confocal
        else:
            ref_meas = self.sted

        if ref_meas.scan_type == "static":
            kwargs["x_field"] = "lag"

        if kwargs.get("ylim") is None:
            if kwargs.get("y_field") in {"average_all_cf_cr", "avg_cf_cr"}:
                kwargs["ylim"] = Limits(-1e3, ref_meas.cf[ref_meas.name].g0 * 1.5)

        with Plotter(
            super_title=f"'{self.name.capitalize()}' Experiment - All ACFs",
            **kwargs,
        ) as ax:
            legend_label_lists = [
                getattr(self, meas_type).plot_correlation_functions(
                    parent_ax=ax,
                    x_field=kwargs.get("x_field", "vt_um"),
                    y_field=kwargs.get("y_field", "normalized"),
                    x_scale=kwargs.get("x_scale", "linear"),
                    y_scale=kwargs.get("y_scale", "linear"),
                    xlim=kwargs.get("xlim", (0, 1)),
                    ylim=kwargs.get("ylim", (-0.20, 1.4)),
                    plot_kwargs=kwargs.get("plot_kwargs", {}),
                )
                for meas_type in ("confocal", "sted")
            ]
            confocal_legend_labels, sted_legend_labels = legend_label_lists
            ax.legend(confocal_legend_labels + sted_legend_labels)

    def calculate_structure_factors(self, **kwargs) -> None:
        """Doc."""

        print(
            f"Calculating all structure factors for '{self.name.capitalize()}' experiment...",
            end=" ",
        )

        # calculate all structure factors
        for meas_type in ("confocal", "sted"):
            getattr(self, meas_type).calculate_structure_factors(**kwargs)

        # plot them
        with Plotter(
            subplots=(1, 2),
            super_title=f"Experiment '{self.name.capitalize()}':\nStructure Factors",
        ) as axes:
            axes[0].set_title("Gaussian Interpolation")
            axes[1].set_title("Linear Interpolation")
            legend_labels = []

            for meas_type in ("confocal", "sted"):
                meas = getattr(self, meas_type)
                for cf_name, cf in meas.cf.items():
                    s = cf.structure_factor
                    axes[0].loglog(s.q, s.sq, **kwargs.get("plot_kwargs", {}))
                    axes[1].loglog(s.q, s.sq_linear_interp, **kwargs.get("plot_kwargs", {}))
                    legend_labels.append(cf_name)

            for ax in axes:
                ax.set_xlabel("$q$ $(\\mu m^{-1})$")
                ax.set_ylabel("$S(q)$")
                ax.legend(legend_labels)

    def fit_structure_factors(self, model: str):
        """Doc."""
        # TODO:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.dawsn.html

        pass


def calculate_afterpulse(
    after_pulse_params, lag: np.ndarray, laser_freq_hz: float, gate_ns=Limits(0, np.inf)
) -> np.ndarray:
    """Doc."""

    gate_pulse_period_ratio = min([1.0, gate_ns.interval() * laser_freq_hz / 1e9])
    fit_name, beta = after_pulse_params
    if fit_name == "multi_exponent_fit":
        # work with any number of exponents
        afterpulse = gate_pulse_period_ratio * multi_exponent_fit(lag, *beta)
    #        afterpulse = gate_pulse_period_ratio * np.dot(beta[::2], np.exp(-np.outer(beta[1::2], lag)))
    elif fit_name == "exponent_of_polynom_of_log":  # for old MATLAB files
        if lag[0] == 0:
            lag[0] = np.nan
        afterpulse = gate_pulse_period_ratio * np.exp(np.polyval(beta, np.log(lag)))

    return afterpulse


def _sum_scan_circles(pulse_runtime, laser_freq_hz, ao_sampling_freq_hz, circle_freq_hz):
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
