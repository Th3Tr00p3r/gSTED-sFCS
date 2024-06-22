"""Data organization and manipulation."""

import multiprocessing as mp
import re
from contextlib import suppress
from dataclasses import InitVar, dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Sequence, Tuple, Union, cast

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.special import j0, j1, jn_zeros
from sklearn import linear_model

from data_analysis.data_processing import (
    AfterpulsingFilter,
    ImageStackData,
    TDCCalibration,
    TDCPhotonDataProcessor,
    TDCPhotonFileData,
    TDCPhotonMeasurementData,
)
from data_analysis.software_correlator import CorrelatorType, SoftwareCorrelator
from data_analysis.workers import N_CPU_CORES, data_processing_worker, io_worker
from utilities.display import Plotter, default_colors, plot_acfs
from utilities.file_utilities import (
    DUMP_PATH,
    default_system_info,
    load_file_dict,
    load_processed_solution_measurement,
    prepare_file_paths,
    save_object,
)
from utilities.fit_tools import (
    FitParams,
    curve_fit_lims,
    diffusion_3d_fit,
    fit_2d_gaussian_to_image,
    multi_exponent_fit,
    ratio_of_lifetime_histograms_fit,
    zero_centered_zero_bg_normalized_gaussian_1d_fit,
)
from utilities.helper import (
    EPS,
    Gate,
    InterpExtrap1D,
    Limits,
    dbscan_noise_thresholding,
    extrapolate_over_noise,
    nan_helper,
    unify_length,
)


@dataclass
class HankelTransform:
    """Holds results of Hankel transform"""

    IE: InterpExtrap1D
    q: np.ndarray
    fq: np.ndarray

    def __post_init__(self):
        # normalize to 1
        self.fq /= self.fq.max()

    def plot(self, label_prefix="", **kwargs):
        """Display the transform's results"""

        with Plotter(
            subplots=(1, 2),
            y_scale="log",
            **kwargs,
        ) as axes:
            kwargs.pop("parent_ax", None)  # TODO: should this be included in Plotter init?
            self.IE.plot(parent_ax=axes[0], label_prefix=label_prefix, **kwargs)
            axes[0].set_xscale("function", functions=(lambda x: x**2, lambda x: x ** (1 / 2)))
            axes[0].set_title("Interp./Extrap. Testing")
            axes[0].set_xlabel("vt_um")
            axes[0].set_ylabel(f"{self.IE.interp_type.capitalize()} Interp./Extrap.")
            axes[0].legend()

            axes[1].plot(self.q, self.fq, label=f"{label_prefix}")
            axes[1].set_xscale("log")
            axes[1].set_ylim(1e-4, 2)
            axes[1].set_title("Hankel Transforms")
            axes[1].set_xlabel("$q\\ \\left(\\frac{1}{\\mu m}\\right)$")
            axes[1].set_ylabel("$F(q)$")
            axes[1].legend(loc="lower left")


@dataclass
class StructureFactor:
    """Holds structure factor data"""

    HT: InitVar[HankelTransform]
    cal_HT: InitVar[HankelTransform]

    def __post_init__(self, HT, cal_HT):
        self.q = HT.q
        self.sq = HT.fq / cal_HT.fq

    def plot(self, label_prefix="", **kwargs):
        """
        Plot a single structure factor. Built to be used for hierarchical plotting
        from a measurement or an experiment.
        """

        with Plotter(
            super_title="Structure Factors",
            y_scale="log",
            **kwargs,
        ) as ax:
            ax.plot(self.q, self.sq / self.sq[0], label=label_prefix)
            ax.set_xscale("log")
            ax.set_ylim(1e-4, 2)
            ax.set_xlabel("$q\\ \\left(\\mu m^{-1}\\right)$")
            ax.set_ylabel("$S(q)$")
            ax.legend()


@dataclass
class LifeTimeParams:
    """Doc."""

    lifetime_ns: float
    sigma_sted: Union[float, Tuple[float, float]]
    laser_pulse_delay_ns: float

    def __repr__(self):
        return f"LifeTimeParams(lifetime_ns={self.lifetime_ns:.2f}, sigma_sted={self.sigma_sted:.2f}, laser_pulse_delay_ns={self.laser_pulse_delay_ns:.2f})"

    def __equiv__(self, other):
        return (
            (self.lifetime_ns == other.lifetime_ns)
            and (self.sigma_sted == other.sigma_sted)
            and (self.laser_pulse_delay_ns == other.laser_pulse_delay_ns)
        )


class CorrFunc:
    """Doc."""

    # Initialize the software correlator once (for all instances)
    SC = SoftwareCorrelator()

    afterpulse: np.ndarray
    vt_um: np.ndarray
    cf_cr: np.ndarray
    g0: float
    fit_params: FitParams

    def __init__(self, name: str, correlator_type: int, laser_freq_hz, gate_ns: Gate, **kwargs):
        self.name = name
        self.correlator_type = correlator_type
        self.laser_freq_hz = laser_freq_hz
        self.gate_ns = gate_ns
        self.afterpulsing_filter = kwargs.get("afterpulsing_filter")
        self.duration_min = kwargs.get("duration_min")
        self.structure_factors: Dict[str, StructureFactor] = {}

    def __add__(self, other):
        """Averages (weighted) all attributes of two CorrFunc objects and returns a new CorrFunc instance"""

        # ensure similarity
        if self.correlator_type != other.correlator_type:
            raise ValueError(
                f"Combined CorrFunc objects must have the same 'correlator_type'! ({self.correlator_type}, {other.correlator_type})"
            )
        if self.laser_freq_hz != other.laser_freq_hz:
            raise ValueError(
                f"Combined CorrFunc objects must have the same 'laser_freq_hz'! ({self.laser_freq_hz}, {other.laser_freq_hz})"
            )
        if not (bool(self.afterpulsing_filter) == bool(other.afterpulsing_filter)):
            raise ValueError(
                f"Combined CorrFunc objects must both have/not have an 'afterpulsing_filter'! ({bool(self.afterpulsing_filter)}, {bool(other.afterpulsing_filter)})"
            )

        # instantiate a new CorrFunc object to hold the mean values
        new_CF = CorrFunc(
            f"{self.name}",
            self.correlator_type,
            self.laser_freq_hz,
            Gate(),
            afterpulsing_filter=self.afterpulsing_filter,
        )

        # before averaging, get the maximum lag length of self and other - will need to unify (zero pad) to max length before stacking for averaging
        new_CF.lag = max(self.lag, other.lag, key=len)
        max_length = len(new_CF.lag)
        min_n_rows = min(self.corrfunc.shape[0], other.corrfunc.shape[0])
        req_shape = (max_length, min_n_rows)  # for 2D arrays

        # set the attributes
        # TODO: test me with alignment measurement, then regular
        new_CF.corrfunc = np.average(
            np.dstack(
                (unify_length(self.corrfunc, req_shape), unify_length(other.corrfunc, req_shape))
            ),
            axis=-1,
            weights=[self.duration_min, other.duration_min],
        )
        new_CF.weights = np.average(
            np.dstack(
                (unify_length(self.weights, req_shape), unify_length(other.weights, req_shape))
            ),
            axis=-1,
            weights=[self.duration_min, other.duration_min],
        )
        new_CF.cf_cr = np.average(
            np.dstack((unify_length(self.cf_cr, req_shape), unify_length(other.cf_cr, req_shape))),
            axis=-1,
            weights=[self.duration_min, other.duration_min],
        )
        new_CF.average_all_cf_cr = np.average(
            np.vstack(
                (
                    unify_length(self.average_all_cf_cr, (max_length,)),
                    unify_length(other.average_all_cf_cr, (max_length,)),
                )
            ),
            axis=0,
            weights=[self.duration_min, other.duration_min],
        )
        new_CF.avg_cf_cr = np.average(
            np.vstack(
                (
                    unify_length(self.avg_cf_cr, (max_length,)),
                    unify_length(other.avg_cf_cr, (max_length,)),
                )
            ),
            axis=0,
            weights=[self.duration_min, other.duration_min],
        )
        new_CF.error_cf_cr = np.average(
            np.vstack(
                (
                    unify_length(self.error_cf_cr, (max_length,)),
                    unify_length(other.error_cf_cr, (max_length,)),
                )
            ),
            axis=0,
            weights=[self.duration_min, other.duration_min],
        )
        new_CF.avg_corrfunc = np.average(
            np.vstack(
                (
                    unify_length(self.avg_corrfunc, (max_length,)),
                    unify_length(other.avg_corrfunc, (max_length,)),
                )
            ),
            axis=0,
            weights=[self.duration_min, other.duration_min],
        )
        new_CF.error_corrfunc = np.average(
            np.vstack(
                (
                    unify_length(self.error_corrfunc, (max_length,)),
                    unify_length(other.error_corrfunc, (max_length,)),
                )
            ),
            axis=0,
            weights=[self.duration_min, other.duration_min],
        )
        new_CF.normalized = np.average(
            np.vstack(
                (
                    unify_length(self.normalized, (max_length,)),
                    unify_length(other.normalized, (max_length,)),
                )
            ),
            axis=0,
            weights=[self.duration_min, other.duration_min],
        )
        new_CF.error_normalized = np.average(
            np.vstack(
                (
                    unify_length(self.error_normalized, (max_length,)),
                    unify_length(other.error_normalized, (max_length,)),
                )
            ),
            axis=0,
            weights=[self.duration_min, other.duration_min],
        )
        new_CF.g0 = (self.g0 + other.g0) / 2

        # accumulate the duration (used as weights for the next addition
        new_CF.duration_min = self.duration_min + other.duration_min

        return new_CF

    def correlate_measurement(
        self,
        split_gen: Generator[np.ndarray, None, None],
        *args,
        **kwargs,
    ) -> None:
        """Doc."""

        output, self.split_durations_s, valid_idxs = self.SC.correlate_list(
            split_gen,
            self.laser_freq_hz,
            self.correlator_type,
            timebase_ms=1000 / self.laser_freq_hz,
            **kwargs,
        )
        self.lag = max(output.lag_list, key=len)

        if kwargs.get("is_verbose"):
            print(". Processing correlator output...", end=" ")

        self._process_correlator_list_output(output, valid_idxs, *args, **kwargs)

    def _process_correlator_list_output(
        self,
        corr_output,
        valid_idxs,
        afterpulse_params,
        bg_corr_list,
        temporal_bg_min_lag_ms=2e-3,
        should_subtract_afterpulsing: bool = False,
        external_afterpulsing: np.ndarray = None,
        **kwargs,
    ) -> None:
        """Doc."""

        # subtract background correlations
        if bg_corr_list:
            for idx, valid_idx in enumerate(valid_idxs):
                bg_corr_dict = bg_corr_list[valid_idx]
                bg_corr = np.interp(
                    corr_output.lag_list[idx],
                    bg_corr_dict["lag"],
                    bg_corr_dict["corrfunc"],
                    right=0,
                )
                corr_output.corrfunc_list[idx] -= bg_corr

        # subtract afterpulsing
        if should_subtract_afterpulsing:
            if external_afterpulsing is not None:
                self.subtracted_afterpulsing = external_afterpulsing
            else:
                self.subtracted_afterpulsing = calculate_calibrated_afterpulse(
                    self.lag, afterpulse_params, self.gate_ns, self.laser_freq_hz
                )

        # zero-pad and generate cf_cr
        lag_len = len(self.lag)
        shape = (len(valid_idxs), lag_len)
        self.corrfunc = np.empty(shape=shape, dtype=np.float64)
        self.weights = np.empty(shape=shape, dtype=np.float64)
        self.cf_cr = np.empty(shape=shape, dtype=np.float64)
        for idx, valid_idx in enumerate(valid_idxs):
            pad_len = lag_len - len(corr_output.corrfunc_list[valid_idx])
            self.corrfunc[idx] = np.pad(corr_output.corrfunc_list[valid_idx], (0, pad_len))
            self.weights[idx] = np.pad(corr_output.weights_list[valid_idx], (0, pad_len))
            try:
                self.cf_cr[idx] = corr_output.countrate_list[valid_idx] * self.corrfunc[idx]
            except ValueError:  # Cross-correlation - countrate is a 2-tuple
                self.cf_cr[idx] = corr_output.countrate_list[valid_idx].a * self.corrfunc[idx]
            with suppress(AttributeError):  # no .subtracted_afterpulsing attribute
                # ext. afterpulse might be shorter/longer
                self.cf_cr[idx] -= unify_length(self.subtracted_afterpulsing, (lag_len,))

        # countrates
        try:  # xcorr
            self.countrate_a = np.nanmean(
                [countrate_pair.a for countrate_pair in corr_output.countrate_list]
            )
            self.countrate_b = np.nanmean(
                [countrate_pair.b for countrate_pair in corr_output.countrate_list]
            )
        except AttributeError:  # autocorr
            self.countrate = np.nanmean([countrate for countrate in corr_output.countrate_list])

        # keep all countrates
        self.countrate_list = corr_output.countrate_list

        # TODO: - interpolate over NaNs - THIS SHOULD NOT HAPPEN! Check out why it sometimes does.
        if np.isnan(self.cf_cr).any():
            nan_per_row = np.isnan(self.cf_cr).sum(axis=1)
            print(f"Warning: {(nan_per_row>0).sum()}/{self.cf_cr.shape[0]} ACFs contain NaNs!")
            print(
                f"The bad rows contain {', '.join([str(nans) for nans in nan_per_row if nans])} NaNs."
            )

            # interpolate over NaNs
            nans, x = nan_helper(self.cf_cr)  # get nans and a way to interpolate over them later
            self.cf_cr[nans] = np.interp(x(nans), x(~nans), self.cf_cr[~nans])

    def average_correlation(
        self,
        should_use_clustering=False,
        rejection=2,
        min_noise_thresh=0.25,
        reject_n_worst=None,
        norm_range=(1e-3, 2e-3),
        **kwargs,
    ) -> None:
        """Doc."""

        self.rejection = rejection
        self.norm_range = Limits(norm_range)
        self.average_all_cf_cr = (self.cf_cr * self.weights).sum(0) / self.weights.sum(0)
        self.median_all_cf_cr = np.median(self.cf_cr, axis=0)
        jj = Limits(self.norm_range.upper, 1).valid_indices(
            self.lag
        )  # work in the relevant part (up to 1 ms)

        try:
            self.score = (
                (1.0 / np.var(self.cf_cr[:, jj], 0))
                * (self.cf_cr[:, jj] - self.median_all_cf_cr[jj]) ** 2
                / len(jj)
            ).sum(axis=1)
        except RuntimeWarning:  # division by zero
            # TODO: why does this happen?
            self.score = (
                (1 / (np.var(self.cf_cr[:, jj], 0) + EPS))
                * (self.cf_cr[:, jj] - self.median_all_cf_cr[jj]) ** 2
                / len(jj)
            ).sum(axis=1)
            print(
                f"Division by zero avoided by adding EPSILON={EPS:.2e}. Why does this happen (zero in variance)?"
            )

        total_n_rows, _ = self.cf_cr.shape

        if should_use_clustering:
            delete_idxs = dbscan_noise_thresholding(
                self.cf_cr, label=self.name, min_noise_thresh=min_noise_thresh, **kwargs
            )
            delete_list = np.nonzero(delete_idxs)[0]
        elif reject_n_worst:
            delete_list = np.argsort(self.score)[-reject_n_worst:]
        elif rejection is not None:
            delete_list = np.where(self.score >= self.rejection)[0]
            if len(delete_list) == total_n_rows:
                raise RuntimeError(
                    "All rows are in 'delete_list'! Increase the rejection limit. Ignoring."
                )
        else:
            delete_list = []

        # if 'reject_n_worst' and 'rejection' are both None, use supplied delete list. If no delete list is supplied, use all rows.
        self.j_bad = delete_list
        self.j_good = [row for row in range(total_n_rows) if row not in delete_list]

        self.avg_cf_cr, self.error_cf_cr = self._calculate_weighted_avg(
            self.cf_cr[self.j_good, :], self.weights[self.j_good, :]
        )

        self.avg_corrfunc, self.error_corrfunc = self._calculate_weighted_avg(
            self.corrfunc[self.j_good, :], self.weights[self.j_good, :]
        )

        j_t = self.norm_range.valid_indices(self.lag)

        try:
            self.g0 = (self.avg_cf_cr[j_t] / self.error_cf_cr[j_t] ** 2).sum() / (
                1.0 / self.error_cf_cr[j_t] ** 2
            ).sum()
        except RuntimeWarning:  # division by zero
            self.g0 = (self.avg_cf_cr[j_t] / (self.error_cf_cr[j_t] + EPS) ** 2).sum() / (
                1 / (self.error_cf_cr[j_t] + EPS) ** 2
            ).sum()
            print(
                f"Division by zero avoided by adding EPSILON={EPS:.2e}. Why does this happen (zero in variance)?"
            )

        self.normalized = self.avg_cf_cr / self.g0
        self.error_normalized = self.error_cf_cr / self.g0

    def _calculate_weighted_avg(
        self, cf_cr: np.ndarray, weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates weighted average and standard error of a 2D array (rows are correlation functions)."""

        tot_weights = weights.sum(0)
        try:
            avg_cf_cr = (cf_cr * weights).sum(0) / tot_weights
        except RuntimeWarning:  # division by zero - caused by zero total weights element/s
            # TODO: why does this happen?
            tot_weights += EPS
            avg_cf_cr = (cf_cr * weights).sum(0) / tot_weights
            print(
                f"Division by zero avoided by adding epsilon={EPS:.2e}. Why does this happen (zero total weight)?"
            )
        finally:
            error_cf_cr = np.sqrt((weights**2 * (cf_cr - avg_cf_cr) ** 2).sum(0)) / tot_weights

        return avg_cf_cr, error_cf_cr

    def remove_background(self, name_prefix: str = "", bg_range: Limits = None) -> Limits:
        """
        Manually select the background range and remove it from all averages.
        Returns the background range indices for passing to further measurements in same experiment.
        Re-average to return to original state.
        """

        # manually select the background range
        plot_acfs(
            self.vt_um,
            avg_cf_cr=self.avg_cf_cr,
            # plot kwargs
            super_title=f"{name_prefix + (', ' if name_prefix else '')}{self.name}: Use the mouse to place 2 markers\nmarking the ACF background range:",
            selection_limits=None if bg_range is not None else (bg_range := Limits()),
            should_close_after_selection=True,
            xlabel="vt_um",
            ylabel="avg_cf_cr",
        )

        # get the indices from the range
        bg_idxs = bg_range.valid_indices(self.vt_um)

        avg_cf_cr_bg = self.avg_cf_cr[bg_idxs].mean()

        # remove the mean background using the range indices from each average quantity
        self.average_all_cf_cr -= self.average_all_cf_cr[bg_idxs].mean()
        self.median_all_cf_cr -= self.median_all_cf_cr[bg_idxs].mean()
        self.g0 -= avg_cf_cr_bg
        self.avg_cf_cr -= avg_cf_cr_bg
        self.error_cf_cr -= self.error_cf_cr[bg_idxs].mean()
        self.avg_corrfunc -= self.avg_corrfunc[bg_idxs].mean()
        self.error_corrfunc -= self.error_corrfunc[bg_idxs].mean()

        self.normalized = self.avg_cf_cr / self.g0
        self.error_normalized = self.error_cf_cr / self.g0

        print(
            f"{name_prefix + (', ' if name_prefix else '')}{self.name}: Found and removed a background constant of {avg_cf_cr_bg:.2f} (from 'avg_cf_cr')."
        )
        return bg_range

    def plot_correlation_function(
        self,
        x_field="lag",
        y_field="avg_cf_cr",
        x_scale="log",
        y_scale="linear",
        **kwargs,
    ) -> None:

        x = getattr(self, x_field)
        y = getattr(self, y_field)
        if x_scale == "log":  # remove zero point data
            x, y = x[1:], y[1:]

        label = kwargs.get("label") or self.name

        with Plotter(
            x_scale=x_scale,
            y_scale=y_scale,
            should_autoscale=True,
            **kwargs,
        ) as ax:
            ax.set_xlabel(x_field.capitalize())
            ax.set_ylabel(y_field.capitalize())
            ax.plot(x, y, "-", label=label, **kwargs.get("plot_kwargs", {}))

    def fit_correlation_function(
        self,
        fit_func=diffusion_3d_fit,
        x_field=None,
        y_field=None,
        y_error_field=None,
        fit_param_estimate=None,
        fit_range=None,
        x_scale=None,
        y_scale=None,
        bounds=(np.NINF, np.inf),
        max_nfev=int(1e4),
        plot_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> FitParams:

        if fit_param_estimate is None:
            if fit_func == diffusion_3d_fit:
                fit_param_estimate = (self.g0, 0.035, 30.0)
                bounds = (  # a, tau, w_sq
                    [0, 0, 0],
                    [1e7, 10, np.inf],
                )
                x_field = "lag"
                x_scale = "log"
                y_field = "avg_cf_cr"
                y_scale = "linear"
                y_error_field = "error_cf_cr"
                fit_range = fit_range or (1e-3, 10)

            elif fit_func == zero_centered_zero_bg_normalized_gaussian_1d_fit:
                fit_param_estimate = (0.1,)
                bounds = (  # sigma
                    [0],
                    [1],
                )
                x_field = "vt_um"
                x_scale = "linear"
                y_field = "normalized"
                y_scale = "linear"
                y_error_field = "error_normalized"
                fit_range = fit_range or (1e-2, 100)

        elif fit_range is None:
            fit_range = (np.NINF, np.inf)

        x = getattr(self, x_field)
        y = getattr(self, y_field)
        error_y = getattr(self, y_error_field)
        if x_scale == "log":
            x = x[1:]  # remove zero point data
            y = y[1:]
            error_y = error_y[1:]

        self.fit_params = curve_fit_lims(
            fit_func,
            fit_param_estimate,
            x,
            y,
            error_y,
            x_limits=Limits(fit_range),
            curve_fit_kwargs=dict(max_nfev=max_nfev, bounds=bounds),
            plot_kwargs={**plot_kwargs, "x_scale": x_scale, "y_scale": y_scale},
            **kwargs,
        )

        return self.fit_params

    def calculate_hankel_transform(
        self,
        interp_types,
        **kwargs,
    ) -> Dict[str, HankelTransform]:
        """Doc."""

        def hankel_transform(
            r: np.ndarray,
            fr: np.ndarray,
            interp_type: str,
            max_r=10,
            r_interp_lims: Tuple[float, float] = (0.05, 20),
            fr_interp_lims: Tuple[float, float] = (1e-8, np.inf),
            n: int = 2048,  # number of interpolation points
            n_robust: int = 7,  # number of robust interpolation points (either side)
            **kwargs,
        ) -> HankelTransform:
            """Doc."""

            # prepare the Hankel transformation matrix C
            c0 = jn_zeros(0, n)  # Bessel function zeros

            # Prepare interpolated radial vector
            r_interp = c0.T * max_r / c0[n - 1]

            j_n, j_m = np.meshgrid(c0, c0)

            C = (2 / c0[n - 1]) * j0(j_n * j_m / c0[n - 1]) / (abs(j1(j_n)) * abs(j1(j_m)))

            v_max = c0[n - 1] / (2 * np.pi * max_r)  # Maximum frequency
            v = c0.T / (2 * np.pi * max_r)  # Frequency vector
            m1 = (abs(j1(c0)) / max_r).T  # m1 prepares input vector for transformation
            m2 = m1 * max_r / v_max  # m2 prepares output vector for display
            # end  of preparations for Hankel transform

            # interpolation/extrapolation
            IE = extrapolate_over_noise(
                x=r,
                y=fr,
                x_interp=r_interp,
                x_lims=Limits(r_interp_lims),
                y_lims=Limits(fr_interp_lims),
                n_robust=n_robust,
                interp_type=interp_type,
                name=f"{kwargs['parent_name']},\n{self.name}"
                if kwargs.get("parent_name")
                else self.name,
                **kwargs,
            )

            # returning the transform (includes interpolation for testing)
            return HankelTransform(IE, 2 * np.pi * v, C @ (IE.y_interp / m1) * m2)

        if kwargs.get("is_verbose"):
            print(f"Calculating '{self.name}' Hankel transform...", end=" ")

        # perform Hankel transforms
        self.hankel_transforms = {
            interp_type: hankel_transform(
                self.vt_um,
                self.normalized,
                interp_type,
                **kwargs,
            )
            for interp_type in interp_types
        }

        # TODO: show this to Oleg - can't figure out what's wrong
        #        #  Estimating error of transform
        #        print("Estimating error of transform...") # TESTESTEST
        #        fq_allfunc = np.empty(shape=(len(self.j_good), q.size))
        #        for idx, j in enumerate(self.j_good):
        #            sample_cf_cr = self.cf_cr[j, j_intrp]
        #            sample_cf_cr[sample_cf_cr <= 0] = 1e-3  # TODO: is that the best way to solve this?
        #            if n_robust and False: # TESTESTEST - CANELLED FOR NOW
        #                log_fr = robust_interpolation(
        #                    r ** 2, sample_vt_um ** 2, np.log(sample_cf_cr), n_robust
        #                )
        #            else:
        #                log_fr = np.interp(r ** 2, sample_vt_um ** 2, np.log(sample_cf_cr))
        #            fr = np.exp(log_fr)
        #            fr[np.nonzero(fr < 0)[0]] = 0
        #
        #            _, fq_allfunc[idx] = hankel_transform(r, fr)
        #
        #        fq_error = np.std(fq_allfunc, axis=0, ddof=1) / np.sqrt(len(self.j_good)) / self.g0

        if kwargs.get("is_verbose"):
            print("Done.")

        return self.hankel_transforms

    def plot_hankel_transforms(self, **kwargs):
        """Doc."""

        # get interpolations types
        interp_types = list(self.hankel_transforms.keys())
        n_interps = len(interp_types)

        with Plotter(subplots=(n_interps, 2), **kwargs) as axes:
            kwargs.pop("parent_ax", None)  # TODO: should this be included in Plotter init?
            for HT, ax_row in zip(
                self.hankel_transforms.values(), axes if n_interps > 1 else [axes]
            ):
                HT.plot(parent_ax=ax_row, label_prefix=f"{self.name}: ", **kwargs)

    def calculate_structure_factor(
        self,
        cal_cf,
        interp_types: List[str],
        parent_ax=None,
        should_force=False,
        should_plot=False,
        is_verbose=True,
        **kwargs,
    ):
        """
        Given a calibration CorrFunc object, i.e. one performed on a below-resolution sample
        (e.g. a 300 bp DNA sample labeled with the same fluorophore),
        this method divides this measurement's (self) Hankel transforms by the corresponding ones
        in the calibration measurement (all calculated if needed) and returns the sought structure factors.
        """

        if is_verbose:
            print(f"Calculating '{self.name}' structure factor...", end=" ")

        # calculate Hankel transforms (with selected interpolation type)
        if parent_names := kwargs.pop("parent_names", None):
            parent_name, cal_parent_name = parent_names

        # calculate Hankel transforms if forced or at least one of them is missing
        if should_force or sum(
            [interp_type in self.hankel_transforms for interp_type in interp_types]
        ) == len(interp_types):
            self.calculate_hankel_transform(
                interp_types, is_verbose=False, parent_name=parent_name, **kwargs
            )
        if should_force or sum(
            [interp_type in cal_cf.hankel_transforms for interp_type in interp_types]
        ) == len(interp_types):
            cal_cf.calculate_hankel_transform(
                interp_types, is_verbose=False, parent_name=cal_parent_name, **kwargs
            )

        # get the structure factors by dividing the Hankel transforms
        self.structure_factors = {}
        for interp_type in interp_types:
            HT = self.hankel_transforms[interp_type]
            cal_HT = cal_cf.hankel_transforms[interp_type]
            self.structure_factors[interp_type] = StructureFactor(HT, cal_HT)

        if is_verbose:
            print("Done.")

    def plot_structure_factor(self, **kwargs):
        """Doc."""

        # get interpolations types
        interp_types = list(self.structure_factors.keys())
        n_interps = len(interp_types)

        with Plotter(subplots=(n_interps, 1), **kwargs) as axes:
            kwargs.pop("parent_ax", None)  # TODO: should this be included in Plotter init?
            for (interp_type, structure_factor), ax in zip(
                self.structure_factors.items(), (axes if n_interps > 1 else [axes])
            ):
                structure_factor.plot(parent_ax=ax, label_prefix=f"{self.name}: ", **kwargs)


class SolutionSFCSMeasurement:
    """Doc."""

    tdc_calib: TDCCalibration
    laser_freq_hz: int
    fpga_freq_hz: int
    scan_settings: dict[str, Any]
    detector_settings: dict[str, Any]

    def __init__(self, type):
        self.type = type
        self.cf: dict = dict()
        self.xcf: dict = dict()
        self.scan_type: str
        self.duration_min: float = None
        self.is_loaded = False
        self.was_processed_data_loaded = False
        self.data = TDCPhotonMeasurementData()
        self._was_corr_input_built: bool = False
        self.afterpulsing_filter: AfterpulsingFilter = None
        self.lifetime_params: LifeTimeParams = None

    def __repr__(self):
        return f"SolutionSFCSMeasurement({self.type}, {self.n_files} files, '{self.template}')"

    @property
    def avg_cnt_rate_khz(self):
        """Calculate the mean effective (in sample) countrate using the first CorrFunc object in .cf"""

        in_sample_countrate_list = list(self.cf.values())[0].countrate_list
        return np.nanmean(in_sample_countrate_list) * 1e-3

    @property
    def std_cnt_rate_khz(self):
        """Calculate the standard deviation from mean effective (in sample) countrate using the first CorrFunc object in .cf"""

        in_sample_countrate_list = list(self.cf.values())[0].countrate_list
        return np.nanstd(in_sample_countrate_list, ddof=1) * 1e-3

    def read_fpga_data(
        self,
        file_path_template: Union[str, Path],
        file_selection: str = "Use All",
        **proc_options,
    ) -> None:
        """Processes a complete FCS measurement (multiple files)."""

        file_paths = prepare_file_paths(Path(file_path_template), file_selection, **proc_options)
        self.n_paths = len(file_paths)
        self.file_path_template = file_path_template
        *_, self.template = Path(file_path_template).parts

        self.dump_path = DUMP_PATH / re.sub("\\*", "", re.sub("_[*].pkl", "", self.template))

        print("\nLoading FPGA data from disk -")
        print(f"Template path: '{file_path_template}'")
        print(f"Files: {self.n_paths}, Selection: '{file_selection}'\n")

        # actual data processing
        self._process_all_data(file_paths, **proc_options)

        # count the files and ensure there's at least one file
        self.n_files = len(self.data)
        if not self.n_files:
            raise RuntimeError(
                f"Loading FPGA data catastrophically failed ({self.n_paths}/{self.n_paths} files skipped)."
            )

        # aggregate images and ROIs for sFCS
        if self.scan_type == "circle":
            self.scan_image = np.vstack(tuple(p.general.image for p in self.data))
            samples_per_circle = int(
                self.scan_settings["ao_sampling_freq_hz"] / self.scan_settings["circle_freq_hz"]
            )
            bg_corr_array = np.empty((self.n_files, samples_per_circle))
            for idx, p in enumerate(self.data):
                bg_corr_array[idx] = p.general.bg_line_corr[0]["corrfunc"]
            avg_bg_corr = bg_corr_array.mean(axis=0)
            self.bg_line_corr_list = [
                dict(lag=p.general.bg_line_corr[0]["lag"], corrfunc=avg_bg_corr)
            ]

        if self.scan_type == "angular":
            # aggregate images and ROIs for angular sFCS
            self.scan_images_dstack = np.dstack(tuple(p.general.image for p in self.data))
            self.rois = [p.general.sec_roi_list for p in self.data]
            # aggregate line background corrfuncs - each line in each section of each file has one background corrfunc
            self.bg_line_corr_list = [
                bg_line_corr
                for bg_file_corr in [p.general.bg_line_corr for p in self.data]
                for bg_line_corr in bg_file_corr
            ]

        else:  # static
            self.bg_line_corr_list = []

        # caculate durations
        calc_duration_mins = sum(p.general.duration_s for p in self.data) / 60
        if self.duration_min is not None:
            if abs(calc_duration_mins - self.duration_min) > self.duration_min * 0.05:
                print(
                    f"Attention! calculated duration ({calc_duration_mins:.1f} mins) is significantly different than the set duration ({self.duration_min} min). Using calculated duration.\n"
                )
        else:
            print(f"Calculating duration (not supplied): {calc_duration_mins:.1f} mins\n")
        self.requested_duration_min = self.duration_min
        self.duration_min = calc_duration_mins

        # done with loading
        self.is_loaded = True
        print(f"Finished loading FPGA data ({self.n_files}/{self.n_paths} files used).\n")

        # plotting of scan image and ROI
        if proc_options.get("should_plot"):
            self.display_scan_images(3)

    def _process_all_data(
        self,
        file_paths: List[Path],
        should_parallel_process: bool = False,
        **proc_options,
    ):
        """Doc."""

        self._get_general_properties(file_paths[0], **proc_options)

        # initialize data processor
        self.data_processor = TDCPhotonDataProcessor(
            self.dump_path, self.laser_freq_hz, self.fpga_freq_hz, self.detector_settings["gate_ns"]
        )

        # parellel processing
        # estimate byte_data size (of each file) - total photons times 7 (bytes per photon) in Mega-bytes
        # TODO: estimation should be for a single file (must know the original number of files!) and condition should depend on actual number of file
        total_byte_data_size_estimate_mb = (
            (self.est_avg_cnt_rate_khz * 1e3 * self.duration_min * 60) * 7 / 1e6
        )
        print(
            f"total_byte_data_size_estimate_mb (all files): {total_byte_data_size_estimate_mb:.2f}"
        )  # TESTESTEST
        if (
            should_parallel_process
            and ((n_files := len(file_paths)) >= 5)
            #            and (total_byte_data_size_estimate_mb > 500)
        ):
            # -2 is one CPU left to OS and one for I/O
            N_RAM_PROCESSES = N_CPU_CORES - 2

            print(
                f"Multi-processing {n_files} files using {N_RAM_PROCESSES} processes (+1 process for I/O)... ",
                end="",
            )

            # initialize 2 queues and a list managed by a multiprocessing.Manager, to share between processes
            with mp.Manager() as manager:
                io_queue = manager.Queue()
                data_processing_queue = manager.Queue()
                processed_list = manager.list()

                # initialize a list to keep track of processes
                process_list = []

                # initialize IO worker (one process)
                io_process = mp.Process(
                    target=io_worker,
                    name="IO",
                    args=(
                        io_queue,
                        data_processing_queue,
                        processed_list,
                        self.data_processor,
                        n_files,
                        N_RAM_PROCESSES,
                    ),
                )
                io_process.start()
                process_list.append(io_process)

                # cancel auto-dumping upon processing, to leave the dumping to IO process
                proc_options["should_dump"] = False
                # do not print anything inside processes (slows things down) - if you want to know what's going on, put the text in the results queue
                proc_options["is_verbose"] = False
                # initialize the data processing workers (N_RAM_PROCESSES processes)
                for worker_idx in range(N_RAM_PROCESSES):
                    data_processing_process = mp.Process(
                        target=data_processing_worker,
                        args=(worker_idx, data_processing_queue, io_queue),
                        kwargs=proc_options,
                    )
                    data_processing_process.start()
                    process_list.append(data_processing_process)

                # fill IO queue with loading tasks for the IO worker
                file_dict_load_tasks = [(load_file_dict, file_path) for file_path in file_paths]
                for task in file_dict_load_tasks:
                    io_queue.put(task)

                # join then close each process to block untill all are finished and immediately release resources
                for process in process_list:
                    process.join()
                for process in process_list:
                    process.close()

                # keep the list after manager is closed (should be a more straightforward way to do this)
                for p in processed_list:
                    self.data.append(p)

            # sort the data by file number
            self.data.sort()

            print("\nMultiprocessing complete!")

        # serial processing (default)
        else:
            proc_options["is_verbose"] = True
            for file_path in file_paths:
                # get file number
                file_num = int(re.findall(r"\d+", str(file_path))[-1])
                # Processing data
                p = self.process_data_file(file_path, file_num, **proc_options)
                print("Done.\n")
                # Appending data to self
                if p is not None:
                    self.data.append(p)

        # auto determination of run duration
        self.run_duration = sum(p.general.duration_s for p in self.data)

    def _get_general_properties(
        self,
        file_path: Path = None,
        file_dict: dict = None,
        should_ignore_hard_gate: bool = False,
        **kwargs,
    ) -> None:
        """Get general measurement properties from the first data file"""

        if file_path is not None:  # Loading file from disk
            file_dict = load_file_dict(file_path)

        full_data = file_dict["full_data"]

        # get countrate estimate (for multiprocessing threshod). calculated more precisely in the end of processing.ArithmeticError
        self.est_avg_cnt_rate_khz = full_data.get("avg_cnt_rate_khz")

        self.afterpulse_params = file_dict["system_info"]["afterpulse_params"]
        self.detector_settings = full_data.get("detector_settings")
        self.delayer_settings = full_data.get("delayer_settings")
        self.laser_freq_hz = int(full_data["laser_freq_mhz"] * 1e6)
        self.pulse_period_ns = 1 / self.laser_freq_hz * 1e9
        self.fpga_freq_hz = int(full_data["fpga_freq_mhz"] * 1e6)
        self.duration_min = (
            full_data.get("duration_s", 0) / 60
        )  # TODO: (was the 'None' used? is the new default 0 OK?)

        # TODO: missing gate - move this to legacy handeling
        if self.detector_settings.get("gate_ns") is not None and (
            not self.detector_settings["gate_ns"] and self.detector_settings["mode"] == "external"
        ):
            fixed_gate = Gate(
                hard_gate=(
                    98 - self.detector_settings["gate_width_ns"],
                    self.detector_settings["gate_width_ns"],
                )
            )
            if self.detector_settings.get("gate_ns") != fixed_gate:
                print(
                    "This should not happen (missing detector gate) - move this to legacy handeling!"
                )
                self.detector_settings["gate_ns"] = fixed_gate
        elif self.detector_settings.get("gate_ns") is None or should_ignore_hard_gate:
            self.detector_settings["gate_ns"] = Gate()
        elif self.detector_settings[
            "gate_ns"
        ]:  # hard gate has TDC gate? remove it and leave only hard gate
            try:
                self.detector_settings["gate_ns"] = Gate(
                    hard_gate=self.detector_settings["gate_ns"].hard_gate
                )
            except AttributeError:
                # legacy Limits as Gate
                self.detector_settings["gate_ns"] = Gate(
                    hard_gate=self.detector_settings["gate_ns"]
                )

        # sFCS
        if scan_settings := full_data.get("scan_settings"):
            self.scan_type = scan_settings["pattern"]
            self.scan_settings = scan_settings
            self.v_um_ms = self.scan_settings["speed_um_s"] * 1e-3
            self.ao_sampling_freq_hz = self.scan_settings.get("ao_sampling_freq_hz", int(1e4))
            if self.scan_type == "circle":  # Circular sFCS
                self.diameter_um = self.scan_settings.get("diameter_um", 50)

        # FCS
        else:
            self.scan_type = "static"

    def process_data_file(
        self, file_path: Path = None, file_num: int = None, file_dict: dict = None, **proc_options
    ) -> TDCPhotonFileData:
        """Doc."""

        if not file_path and not file_dict:
            raise ValueError("Must supply either a valid path or a ")

        # if using existing file_dict - usually during alignment measurements
        if file_dict is not None:
            self._get_general_properties(file_dict=file_dict, **proc_options)
            file_num = 1
            self.dump_path = DUMP_PATH / "temp_meas"

        # File Data Loading
        if file_path is not None:  # Loading file from disk
            *_, template = file_path.parts
            print(
                f"Loading and processing file No. {file_num} ({self.n_paths} files): '{template}'...",
                end=" ",
            )
            try:
                file_dict = load_file_dict(file_path)
                proc_options["byte_data"] = file_dict["full_data"].get(
                    "byte_data"
                )  # compatibility with pre-conversion data
                proc_options["byte_data_path"] = file_path.with_name(
                    file_path.name.replace(".pkl", "_byte_data.npy")
                )
            except FileNotFoundError:
                print(f"File '{file_path}' not found. Ignoring.")

        # File Processing
        # initialize data processor if needed (only during alignment)
        if not hasattr(self, "data_processor"):
            self.data_processor = TDCPhotonDataProcessor(
                self.dump_path,
                self.laser_freq_hz,
                self.fpga_freq_hz,
                self.detector_settings["gate_ns"],
            )

        # TODO: all relevant properties from 'file_dict' should have been imported to the 'SolutionSFCSMeasurement' object at this point.
        # It makes no sense to send the 'file_dict' as an argument - send what's relevant.
        return self.data_processor.process_data(file_num, file_dict["full_data"], **proc_options)

    def calibrate_tdc(self, force_processing=True, **kwargs) -> None:
        """Doc."""

        if not force_processing and hasattr(self, "tdc_calib"):
            print(f"\n{self.type}: TDC calibration exists, skipping.")
            if kwargs.get("should_plot"):
                self.tdc_calib.plot()
            return

        if kwargs.get("is_verbose"):
            print(f"\n{self.type}: Calibrating TDC...", end=" ")

        # perform actual TDC calibration
        self.tdc_calib = self.data_processor.calibrate_tdc(self.data, self.scan_type, **kwargs)

        if kwargs.get("should_plot"):
            self.tdc_calib.plot()

        if kwargs.get("is_verbose"):
            print("Done.")

    def correlate_and_average(self, **kwargs) -> CorrFunc:
        """High level function for correlating and averaging any data."""

        CF = self.correlate_data(**kwargs)
        CF.average_correlation(**kwargs)
        return CF

    def generate_combined_inputs(
        self,
        input_gen,
        is_filtered=False,
        get_afterpulsing=False,
        is_verbose=False,
    ):
        """Using the memory-mapped 'input_gen', build and prepare splits for the correlator, as well as optional corresponsding filter splits"""

        for split_idx, split in enumerate(input_gen):
            # skipping empty splits
            if not split.size:
                if is_verbose:
                    print(f" Empty split encountered! Skipping split {split_idx}...", end="")
                yield None, None
                continue

            # Generate final_corr_section_input
            final_split = np.squeeze(split[1:].astype(np.int32))

            # Generate split filter correlator input
            split_filter = (
                self.afterpulsing_filter.get_split_filter_input(split[0], get_afterpulsing)
                if is_filtered
                else None
            )

            yield final_split, split_filter

    def correlate_data(  # NOQA C901
        self,
        cf_name="unnamed",
        tdc_gate_ns=Gate(),
        afterpulsing_method=None,
        external_afterpulse_params=None,
        external_afterpulsing=None,
        get_afterpulsing=False,
        subtract_spatial_bg_corr=True,
        **corr_options,
    ) -> CorrFunc:
        """
        High level function for correlating any type of data (e.g. static, angular scan, circular scan...)
        Returns a 'CorrFunc' object.
        Data attribute is memory-mapped from disk.
        """

        if corr_options.get("is_verbose"):
            print(
                f"{self.type} - Preparing split data ({self.n_files} files) for software correlator...",
                end=" ",
            )

        # maintain afterpulsing method for consistency with future gating
        if not hasattr(self, "afterpulsing_method"):
            self.afterpulsing_method = (
                "filter" if afterpulsing_method is None else afterpulsing_method
            )
        afterpulsing_method = (
            self.afterpulsing_method if afterpulsing_method is None else afterpulsing_method
        )

        # Unite TDC gate and detector gate
        # TODO: detector gate represents the actual effective gate (since pulse travel time is already synchronized before measuring), This means that
        # I should add the fit-deduced pulse travel time (affected by TDC +/- 2.5 ns) to the lower gate to compare with TDC gate???
        laser_pulse_period_ns = 1e9 / self.laser_freq_hz
        if tdc_gate_ns.upper >= laser_pulse_period_ns:
            tdc_gate_ns.upper = np.inf
        gate_ns = Gate(tdc_gate_ns, hard_gate=self.detector_settings["gate_ns"].hard_gate)

        #  add gate to cf_name
        if gate_ns or gate_ns.hard_gate:
            # TODO: cf_name should not contain any description other than gate and "afterpulsing" yes/no (description is in self.type)
            cf_name = f"gated {cf_name} {gate_ns}{f' ({gate_ns.hard_gate} hard)' if gate_ns.hard_gate else ''}"

        # ensure proper valid afterpulsing method
        if afterpulsing_method not in {"subtract calibrated", "filter", "none"}:
            raise ValueError(f"Invalid afterpulsing_method chosen: {afterpulsing_method}.")

        # pre-calibrate TDC (prior to creating splits) in case using afterpulsing filtering (FLCS)
        elif (is_filtered := afterpulsing_method == "filter") or gate_ns:
            if not hasattr(self, "tdc_calib"):  # calibrate TDC (if not already calibrated)
                if corr_options.get("is_verbose"):
                    print("(Calibrating TDC first...)", end=" ")
                self.calibrate_tdc(**corr_options)

        # Calculate afterpulsing filter if doesn't alreay exist (optional)
        if is_filtered:
            # case first time correlating the measurement:
            if not self.afterpulsing_filter:
                if corr_options.get("is_verbose"):
                    print("Preparing Afterpulsing filter... ", end="")
                self.afterpulsing_filter = self.tdc_calib.calculate_afterpulsing_filter(
                    gate_ns.hard_gate, self.type, **corr_options
                )
                if corr_options.get("is_verbose"):
                    print("Done.")
            # TDC gates can use original (hard-gated or not) filter
            else:
                if corr_options.get("is_verbose"):
                    print(f"Using existing {self.type} afterpulsing filter.")

        # choose correct correlator type
        if self.scan_type in {"static", "circle"}:
            if is_filtered:
                correlator_option = CorrelatorType.PH_DELAY_LIFETIME_CORRELATOR
            else:
                correlator_option = CorrelatorType.PH_DELAY_CORRELATOR
        elif self.scan_type == "angular":
            if is_filtered:
                correlator_option = CorrelatorType.PH_DELAY_LIFETIME_CORRELATOR_LINES
            else:
                correlator_option = CorrelatorType.PH_DELAY_CORRELATOR_LINES

        if corr_options.get("is_verbose"):
            try:
                print(
                    f"Correlating {self.scan_type} {cf_name} data ({sum([p.general.n_corr_splits for p in self.data])} splits):",
                    end=" ",
                )
            except TypeError:
                print("Implement this - find number of splits for continuous/static scans...")

        # Correlate data
        CF = CorrFunc(
            cf_name,
            correlator_option,
            self.laser_freq_hz,
            gate_ns,
            afterpulsing_filter=self.afterpulsing_filter if is_filtered else None,
            duration_min=self.duration_min,
        )
        CF.correlate_measurement(
            # TODO: perhaps both 'generate_combined_inputs' and 'data.generate_splits' can be united in data_processing.py
            self.generate_combined_inputs(
                self.data.generate_splits(gate_ns, **corr_options),
                is_filtered,
                get_afterpulsing,
                corr_options.get("is_verbose", False),
            ),
            external_afterpulse_params
            if external_afterpulse_params is not None
            else self.afterpulse_params,
            getattr(self, "bg_line_corr_list", []) if subtract_spatial_bg_corr else [],
            external_afterpulsing=external_afterpulsing,
            should_subtract_afterpulsing=afterpulsing_method == "subtract calibrated",
            **corr_options,
        )

        try:  # temporal to spatial conversion, if scanning
            CF.vt_um = self.v_um_ms * CF.lag
        except AttributeError:
            # static
            CF.vt_um = CF.lag

        if corr_options.get("is_verbose"):
            print("- Done.")

        # name the Corrfunc object
        # TODO: this should be eventually a list, not a dict (only first element and all together are ever interesting)
        self.cf[cf_name] = CF

        return CF

    def remove_backgrounds(self):
        """Remove mean ACF backgrounds from all CorrFunc objects"""

        for cf in self.cf.values():
            cf.remove_background()

    def display_scan_images(self, n_images, **kwargs) -> None:
        """Doc."""

        def spread_elements(input_list, n):
            length = len(input_list)
            if n >= length:
                return input_list[:]  # Return a copy of the entire input list if n >= length

            # Calculate the step size
            step = length / float(n)

            # Initialize the output list
            output_list = []

            # Iterate through indices and add elements to output_list
            for i in range(n):
                index = int(round(i * step))  # Round to the nearest integer index
                output_list.append(input_list[index])

            return output_list

        # get a fraction of the images along the whole measurement
        file_idxs = spread_elements(np.arange(self.n_files), n_images)
        file_numbers = np.array([p.file_num for p in self.data])
        images = np.moveaxis(self.scan_images_dstack, -1, 0)

        try:
            if self.scan_type == "angular":
                with Plotter(
                    subplots=(1, len(file_idxs)),
                    fontsize=8,
                    should_force_aspect=True,
                    **kwargs,
                ) as axes:
                    if not hasattr(
                        axes, "size"
                    ):  # if axes is not an ndarray (only happens if reading just one file)
                        axes = np.array([axes])
                    for file_idx, ax in zip(file_idxs, axes):
                        file_num = file_numbers[file_idx]
                        sec_roi_list = self.rois[file_idx]
                        image = images[file_idx]
                        ax.set_title(f"file #{file_num} of\n'{self.type}' measurement")
                        ax.set_xlabel("Pixel Index")
                        ax.set_ylabel("Line Index")
                        ax.imshow(image, interpolation="none")
                        for sec_roi in sec_roi_list:
                            ax.plot(sec_roi["col"], sec_roi["row"], color="white", lw=0.6)

            elif self.scan_type == "circle":
                with Plotter(fontsize=8, should_force_aspect=True, **kwargs) as ax:
                    ax.imshow(self.scan_image, interpolation="none")

            else:
                raise ValueError(f"Can't display scan images for '{self.scan_type}' scans.")

        except AttributeError:
            # Measurement not loaded!
            raise RuntimeError(f"'{self.type}' not loaded!")

    def plot_correlation_functions(
        self,
        x_field="lag",
        y_field="normalized",
        x_scale="log",
        y_scale="linear",
        xlim=(1e-4, 1),
        ylim=(-0.20, 1.4),
        plot_kwargs={},
        **kwargs,
    ):
        """Doc."""

        with Plotter(super_title=f"'{self.type.capitalize()}' - ACFs", **kwargs) as ax:
            kwargs["parent_ax"] = ax
            for cf_name, cf in {**self.cf, **self.xcf}.items():
                cf.plot_correlation_function(
                    x_field=x_field,
                    y_field=y_field,
                    x_scale=x_scale,
                    y_scale=y_scale,
                    xlim=xlim,
                    ylim=ylim,
                    plot_kwargs=plot_kwargs,
                    **kwargs,
                )
            ax.legend()

    def estimate_spatial_resolution(self, colors=None, fit_range=None, **kwargs) -> Iterator[str]:
        """
        Perform Gaussian fits over 'normalized' vs. 'vt_um' fields of all correlation functions in the measurement
        in order to estimate the resolution improvement. This is relevant only for calibration experiments (i.e. 300 bp samples).
        Returns the legend labels for use with higher-level Plotter instance in SolutionSFCSExperiment.
        """
        # TODO: this should be a higher-level function which delegates the fitting and plotting to the individual CorrFunc objects, and only plots hierarchically.
        #            This would allow better control, e.g. in case only certain gates are needed for a plot.
        #            Take a look at "plot_correlation_functions" or "plot_structure_factors", for examples on how to implement.

        HWHM_FACTOR = np.sqrt(2 * np.log(2))

        for CF in self.cf.values():
            CF.fit_correlation_function(
                fit_func=zero_centered_zero_bg_normalized_gaussian_1d_fit,
                fit_range=fit_range,
                **kwargs,
            )

        with Plotter(
            super_title=f"Resolution Estimation\nGaussian fitting (HWHM) for '{self.type}' ACF(s)",
            x_scale="quadratic",
            y_scale="log",
            xlim=(0.01, 1),
            ylim=(5e-3, 1),
            **kwargs,
        ) as ax:
            # TODO: line below - this issue (line colors in hierarchical plotting) may be general and should be solved in Plotter class (?)
            colors = colors if colors is not None else cycle(default_colors)
            for CF, color in zip(self.cf.values(), colors):
                FP = CF.fit_params
                kwargs.pop("parent_ax", None)  # TODO: consider including this in Plotter __exit__
                hwhm = list(FP.beta.values())[0] * 1e3 * HWHM_FACTOR
                hwhm_error = list(FP.beta_error.values())[0] * 1e3 * HWHM_FACTOR
                fit_label = f"{CF.name}: ${hwhm:.0f}\\pm{hwhm_error:.0f}~nm$ ($\\chi^2={FP.chi_sq_norm:.1e}$)"
                FP.plot(parent_ax=ax, color=color, fit_label=fit_label, **kwargs)
            ax.set_xlabel("vt_um")
            ax.set_ylabel("normalized ACF")

            ax.legend()

        return colors

    def compare_lifetimes(
        self,
        legend_label: str,
        compare_to: dict = None,
        normalization_type="Per Time",
        **kwargs,
    ):
        """
        Plots a comparison of lifetime histograms. 'kwargs' is a dictionary, where keys are to be used as legend labels
        and values are 'TDCPhotonDataMixin'-inheriting objects which are supposed to have their own TDC calibrations.
        """

        # add self (first) to compared TDC calibrations
        compared = {**{legend_label: self}, **compare_to}

        h = []
        for label, measurement in compared.items():
            with suppress(AttributeError):
                # AttributeError - assume other tdc_objects that have tdc_calib structures
                x = measurement.tdc_calib.t_hist
                if normalization_type == "NO":
                    y = measurement.tdc_calib.all_hist / measurement.tdc_calib.t_weight
                elif normalization_type == "Per Time":
                    y = measurement.tdc_calib.all_hist_norm
                elif normalization_type == "By Sum":
                    y = measurement.tdc_calib.all_hist_norm / np.sum(
                        measurement.tdc_calib.all_hist_norm[
                            np.isfinite(measurement.tdc_calib.all_hist_norm)
                        ]
                    )
                else:
                    raise ValueError(f"Unknown normalization type '{normalization_type}'.")
                h.append((x, y, label))

        with Plotter(super_title="Life Time Comparison", **kwargs) as ax:
            for tuple_ in h:
                x, y, label = tuple_
                ax.semilogy(x, y, "-o", label=label)
            ax.set_xlabel("Life Time (ns)")
            ax.set_ylabel("Frequency")
            ax.legend()

    def plot_hankel_transforms(self, **kwargs):
        """Doc."""

        # get interpolations types
        interp_types = list(list(self.cf.values())[0].hankel_transforms.keys())
        n_interps = len(interp_types)

        with Plotter(
            subplots=(n_interps, 2),
            super_title=f"{self.type.capitalize()}: Hankel Transforms",
            **kwargs,
        ) as axes:
            kwargs.pop("parent_ax", None)  # TODO: should this be included in Plotter init?
            for cf in self.cf.values():
                cf.plot_hankel_transforms(parent_ax=axes, **kwargs)

    def calculate_structure_factors(
        self,
        cal_meas,
        interp_types=["gaussian"],
        parent_ax=None,
        is_verbose=True,
        should_force=False,
        **kwargs,
    ):
        """
        Given a calibration SolutionSFCSMeasurement, i.e. one performed on a below-resolution sample
        (e.g. a 300 bp DNA sample labeled with the same fluorophore),
        this method divides this measurement's (self) Hankel transforms by the corresponding ones
        in the calibration measurement (all calculated if needed) and returns the sought structure factors.
        """

        if is_verbose:
            print(f"Calculating all structure factors for '{self.type}' measurement... ", end="")

        # calculate without plotting
        kwargs["parent_names"] = (
            f"{kwargs['parent_names'][0]},\n{self.type}"
            if kwargs.get("parent_names")
            else self.type,
            f"{kwargs['parent_names'][1]},\n{self.type}"
            if kwargs.get("parent_names")
            else self.type,
        )
        for CF, cal_CF in zip(self.cf.values(), cal_meas.cf.values()):
            if not CF.structure_factors or should_force:
                try:
                    CF.calculate_structure_factor(
                        cal_CF, interp_types, is_verbose=is_verbose, **kwargs
                    )
                except RuntimeError:
                    # RuntimeError - exponent overflow
                    # avoid losing everything because of a single bad limit choice/noisy gate
                    continue
            elif is_verbose:
                print("Using existing... Done.")

    def plot_structure_factors(self, **kwargs):
        """Doc."""

        # get interpolations types
        interp_types = list(list(self.cf.values())[0].structure_factors.keys())
        n_interps = len(interp_types)

        with Plotter(
            subplots=(n_interps, 1),
            super_title=f"{self.type.capitalize()}: Structure Factors",
            **kwargs,
        ) as axes:
            for CF in self.cf.values():
                kwargs.pop("parent_ax", None)
                CF.plot_structure_factor(parent_ax=axes, **kwargs)
            for ax, interp_type in zip(axes if n_interps > 1 else [axes], interp_types):
                ax.set_title(f"{interp_type.capitalize()} Interp./Extrap.")

    def calculate_filtered_afterpulsing(
        self, tdc_gate_ns: Union[Gate, Tuple[float, float]] = Gate(), is_verbose=True, **kwargs
    ):
        """Get the afterpulsing by filtering the raw data."""
        # TODO: this might fail if called prior to either TDC calibration or afterpulsing filter calculation

        self.correlate_and_average(
            cf_name="afterpulsing",
            afterpulsing_method="filter",
            get_afterpulsing=True,
            tdc_gate_ns=Gate(tdc_gate_ns),
            **kwargs,
        )

    def save_processed(
        self, should_save_data=True, should_force=False, is_verbose=False, **kwargs
    ) -> bool:
        """
        Save a processed measurement, including the '.data' attribute.
        The template may then be loaded much more quickly.
        """

        was_saved = False

        # save the measurement
        dir_path = (
            cast(Path, self.file_path_template).parent
            / "processed"
            / cast(Path, self.file_path_template).stem.replace("_*", "")
        )
        meas_file_path = dir_path / "SolutionSFCSMeasurement.blosc"
        if not meas_file_path.is_file() or should_force:
            # don't save correlator inputs (re-built when loaded)
            self.filter_input_list = None

            # save the measurement object
            if is_verbose:
                print("Saving SolutionSFCSMeasurement object... ", end="")
            save_object(
                self,
                meas_file_path,
                compression_method="blosc",
                obj_name="processed measurement",
                should_track_progress=is_verbose,
            )

            if is_verbose:
                print("Done.")

            was_saved = True

        # save the raw data separately
        if should_save_data and not self.was_processed_data_loaded:
            data_dir_path = dir_path / "data"

            # compress and save each data file in the temp folder in 'data_dir_path' (optional)
            for p in self.data:
                if p.raw.compressed_file_path is None or (
                    not p.raw.compressed_file_path.exists() or should_force
                ):
                    p.raw.save_compressed(data_dir_path)

            was_saved = True

        return was_saved


class SolutionSFCSExperiment:
    """Doc."""

    def __init__(self, name):
        self.name = name
        self.confocal: SolutionSFCSMeasurement
        self.sted: SolutionSFCSMeasurement

    def __repr__(self):
        return f"""SolutionSFCSExperiment({self.name}, confocal={f"'{self.confocal.template}'" if self.confocal.is_loaded else None}, sted='{f"'{self.sted.template}'" if self.sted.is_loaded else None})
        """

    @property
    def cf_dict(self):
        """unite all CorrFunc objects from both 'confocal' and 'sted' measurements in a single dictionary."""

        return {
            cf_label: cf for meas in [self.confocal, self.sted] for cf_label, cf in meas.cf.items()
        }

    @property
    def lifetime_params(self):
        """Returns the LifeTimeParams property of the confocal/STED measurements if both exist and are the same."""

        if (
            (getattr(self.confocal, "lifetime_params", None) is not None)
            and (getattr(self.sted, "lifetime_params", None) is not None)
            and (self.confocal.lifetime_params == self.sted.lifetime_params)
        ):
            return self.confocal.lifetime_params

    def load_experiment(
        self,
        confocal_template: Union[str, Path] = None,
        sted_template: Union[str, Path] = None,
        confocal=None,
        sted=None,
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
                if meas_template is not None:
                    measurement = self.load_measurement(
                        meas_type=meas_type,
                        file_path_template=meas_template,
                        **{**kwargs, **meas_kwargs},
                    )
                else:  # Use empty measuremnt by default
                    setattr(self, meas_type, SolutionSFCSMeasurement(meas_type))
            else:  # use supllied measurement
                setattr(self, meas_type, measurement)
                getattr(self, meas_type).name = meas_type  # remame supplied measurement

        # always plot
        self.plot_standard(should_add_exp_name=False, **kwargs)

    def load_measurement(
        self,
        meas_type: str,
        file_path_template: Union[str, Path],
        plot_kwargs: dict = {},
        force_processing=True,
        should_re_correlate=False,
        afterpulsing_method="filter",
        should_save=False,
        **kwargs,
    ) -> SolutionSFCSMeasurement:
        """Doc."""

        if "cf_name" not in kwargs:
            if meas_type == "confocal":
                kwargs["cf_name"] = "confocal"
            else:  # sted
                kwargs["cf_name"] = "sted"

        if kwargs.get(f"{meas_type}_file_selection"):
            kwargs["file_selection"] = kwargs[f"{meas_type}_file_selection"]

        # create measurement and set as attribute
        measurement = SolutionSFCSMeasurement(meas_type)
        setattr(self, meas_type, measurement)

        if not force_processing:  # Use pre-processed
            try:
                file_path_template = Path(file_path_template)
                dir_path, file_template = file_path_template.parent, file_path_template.name
                # load pre-processed
                dir_path = dir_path / "processed" / re.sub("_[*].(pkl|mat)", "", file_template)
                measurement = load_processed_solution_measurement(
                    dir_path,
                    file_template,
                    #                    should_load_data=should_re_correlate,
                )
                measurement.type = meas_type
                measurement.was_processed_data_loaded = True
                setattr(self, meas_type, measurement)
                print(f"Loaded pre-processed {meas_type} measurement from: '{dir_path}'")
            except OSError:
                # since not forcing processing, avoid dumping (though will still process to get p.general...) existing raw data
                # TODO: try to avoid processing of existing files altogether - this may require saving the general part as well...
                kwargs["should_avoid_dumping"] = True
                print(
                    f"Pre-processed {meas_type} measurement not found at: '{dir_path}'. Processing (non-existing) data files regularly."
                )

        if not measurement.cf:  # Process data
            measurement.read_fpga_data(
                file_path_template,
                **kwargs,
            )
            # Calibrate TDC (sync with confocal) before correlating if using afterpulsing filtering
            if afterpulsing_method == "filter" and meas_type == "sted" and self.confocal.is_loaded:
                print(f"{self.name}: Calibrating TDC first (syncing STED to confocal)...", end=" ")
                self.calibrate_tdc(**kwargs)
                print("Done.")

        if not measurement.cf or should_re_correlate:  # Correlate and average data
            measurement.cf = {}
            is_verbose = kwargs.pop(
                "is_verbose", False
            )  # avoid duplicate in following call # TODO: fix this
            cf = measurement.correlate_and_average(
                is_verbose=True, afterpulsing_method=afterpulsing_method, **kwargs
            )
            kwargs["is_verbose"] = is_verbose  # avoid duplicate in above call # TODO: fix this

            if should_save:
                print(f"Saving {measurement.type} measurement to disk...", end=" ")
                if measurement.save_processed(**kwargs):
                    print("Done.")
                else:
                    print("Measurement already exists! (set 'should_force = True' to override.)")

        else:  # get existing first corrfunc
            cf = list(measurement.cf.values())[0]

        if kwargs.get("should_plot_meas"):

            x_field = (
                kwargs.get("x_field") or "lag" if measurement.scan_type == "static" else "vt_um"
            )

            with Plotter(
                super_title=f"'{self.name.capitalize()}' Experiment\n'{measurement.type.capitalize()}' Measurement - ACFs",
                ylim=(-100, cf.g0 * 1.5),
            ) as ax:
                cf.plot_correlation_function(
                    parent_ax=ax,
                    y_field="average_all_cf_cr",
                    x_field=x_field,
                    label=f"{cf.name}: average_all_cf_cr",
                    plot_kwargs=plot_kwargs,
                )
                cf.plot_correlation_function(
                    parent_ax=ax,
                    y_field="avg_cf_cr",
                    x_field=x_field,
                    label=f"{cf.name}: avg_cf_cr",
                    plot_kwargs=plot_kwargs,
                )
                ax.legend()

        return measurement

    def remove_backgrounds(self):
        """Remove mean ACF backgrounds from all CorrFunc objects"""

        bg_range = None
        for cf in self.cf_dict.values():
            bg_range = cf.remove_background(name_prefix=self.name, bg_range=bg_range)

    def re_average_all(self, **kwargs):
        """Doc."""

        for cf in self.cf_dict.values():
            cf.average_correlation(**kwargs)

    def save_processed_measurements(self, meas_types=["confocal", "sted"], **kwargs):
        """Doc."""

        print(f"Experiment '{self.name}': Saving processed measurements to disk...")
        if self.confocal.is_loaded and "confocal" in meas_types:
            if self.confocal.save_processed(**kwargs):
                print("Confocal saved.")
            else:
                print(
                    "Not saving - processed measurement already exists (set 'should_force = True' to override.)",
                    end=" ",
                )
        if self.sted.is_loaded and "sted" in meas_types:
            if self.sted.save_processed(**kwargs):
                print("STED saved.")
            else:
                print(
                    "Not saving - processed measurement already exists (set 'should_force = True' to override.)",
                    end=" ",
                )
        print("Done.\n")

    def calibrate_tdc(self, should_plot=True, **kwargs):
        """Doc."""

        # calibrate excitation measurement first, and sync STED to it if STED exists
        if self.confocal.is_loaded:
            # only calibrate TDC for confocal if hasn't before (should exist if afterpulsing-filtered)
            if not hasattr(self.confocal, "tdc_calib"):  # TODO: TEST ME!
                self.confocal.calibrate_tdc(**kwargs)
            if self.sted.is_loaded:  # if both measurements quack as if loaded
                self.sted.calibrate_tdc(sync_coarse_time_to=self.confocal.tdc_calib, **kwargs)
        # calibrate STED only
        else:
            self.sted.calibrate_tdc(**kwargs)

        # optional plotting
        if should_plot:
            self.plot_tdc_calib(**kwargs)

    def plot_tdc_calib(self, **kwargs):
        """Plot TDCCalibrations of the confocal and/or STED measurements."""

        super_title = f"'{self.name.capitalize()}' Experiment\nTDC Calibration"
        # Both measurements loaded
        if hasattr(self.confocal, "scan_type") and hasattr(
            self.sted, "scan_type"
        ):  # if both measurements quack as if loaded
            with Plotter(subplots=(2, 4), super_title=super_title, **kwargs) as axes:
                self.confocal.tdc_calib.plot(parent_ax=axes[:, :2])
                self.sted.tdc_calib.plot(parent_ax=axes[:, 2:])

        # Confocal only
        elif self.confocal.is_loaded:  # STED measurement not loaded
            self.confocal.tdc_calib.plot(super_title=super_title, **kwargs)

        # STED only
        else:
            self.sted.tdc_calib.plot(super_title=super_title, **kwargs)

    def get_lifetime_parameters(
        self,
        sted_field="symmetric",
        drop_idxs=[],
        fit_range=None,
        param_estimates=None,
        bg_range=Limits(40, 60),
        should_plot=True,
        **kwargs,
    ) -> LifeTimeParams:
        """Doc."""
        # TODO: the default bg_range may fail for hard-gated measurements! Check it out and add manual selection or some automatic heuristic solution.

        conf = self.confocal
        sted = self.sted

        conf_hist = conf.tdc_calib.all_hist_norm
        conf_t = conf.tdc_calib.t_hist
        conf_t = conf_t[np.isfinite(conf_hist)]
        conf_hist = conf_hist[np.isfinite(conf_hist)]

        if sted.is_loaded:
            sted_hist = sted.tdc_calib.all_hist_norm
            sted_t = sted.tdc_calib.t_hist
            sted_t = sted_t[np.isfinite(sted_hist)]
            sted_hist = sted_hist[np.isfinite(sted_hist)]

        h_max, j_max = conf_hist.max(), conf_hist.argmax()
        t_max = conf_t[j_max]

        beta0 = (h_max, 4, h_max * 1e-3)

        fit_range = fit_range or Limits(t_max + 0.1, 40)
        param_estimates = param_estimates or beta0

        conf_params = conf.tdc_calib.fit_lifetime_hist(
            fit_range=fit_range, fit_param_estimate=beta0
        )
        lifetime_ns = conf_params.beta["tau"]
        if should_plot:
            conf_params.plot(super_title="Lifetime Fit", y_scale="log")

        if sted.is_loaded:
            # remove background
            conf_bg = np.mean(conf_hist[Limits(bg_range).valid_indices(conf_t)])
            conf_hist = conf_hist - conf_bg
            sted_bg = np.mean(sted_hist[Limits(bg_range).valid_indices(sted_t)])
            sted_hist = sted_hist - sted_bg

            j = conf_t < 20
            t = conf_t[j]
            hist_ratio = conf_hist[j] / np.interp(t, sted_t, sted_hist, right=0)
            if drop_idxs:
                j = np.setdiff1d(np.arange(1, len(t)), drop_idxs)
                t = t[j]
                hist_ratio = hist_ratio[j]

            title = "Robust Linear Fit of the Linear Part of the Histogram Ratio"
            with Plotter(figsize=(11.25, 7.5), super_title=title, **kwargs) as ax:

                # Using inner Plotter for manual selection
                title = "Use the mouse to place 2 markers\nlimiting the linear range:"
                with Plotter(
                    parent_ax=ax,
                    super_title=title,
                    selection_limits=(linear_range := Limits()),
                    **kwargs,
                ) as ax:
                    ax.plot(t, hist_ratio, label="hist_ratio")
                    ax.legend()

                j_selected = linear_range.valid_indices(t)

                if sted_field == "symmetric":
                    # Robustly fit linear model with RANSAC algorithm
                    ransac = linear_model.RANSACRegressor()
                    ransac.fit(t[j_selected][:, np.newaxis], hist_ratio[j_selected])
                    p0, p1 = ransac.estimator_.intercept_, ransac.estimator_.coef_[0]

                    ax.plot(t[j_selected], hist_ratio[j_selected], "oy", label="hist_ratio")
                    ax.plot(
                        t[j_selected],
                        np.polyval([p1, p0], t[j_selected]),
                        "r",
                        label=["linear range", "robust fit"],
                    )
                    ax.legend()

                    sigma_sted = p1 * lifetime_ns
                    try:
                        laser_pulse_delay_ns = (1 - p0) / p1
                    except RuntimeWarning:
                        laser_pulse_delay_ns = None

                elif sted_field == "paraboloid":
                    fit_params = curve_fit_lims(
                        ratio_of_lifetime_histograms_fit,
                        param_estimates=(2, 1, 1),
                        xs=t[j_selected],
                        ys=hist_ratio[j_selected],
                        ys_errors=np.ones(j_selected.sum()),
                        should_plot=True,
                    )

                    sigma_sted = (
                        fit_params.beta["sigma_x"] * lifetime_ns,
                        fit_params.beta["sigma_y"] * lifetime_ns,
                    )
                    laser_pulse_delay_ns = fit_params.beta["t0"]

        # no STED! only (confocal) lifetime is available
        else:
            sigma_sted = 0
            laser_pulse_delay_ns = float(t_max)

        conf.lifetime_params = LifeTimeParams(lifetime_ns, sigma_sted, laser_pulse_delay_ns)
        sted.lifetime_params = conf.lifetime_params
        return conf.lifetime_params

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

    def add_gate(
        self,
        tdc_gate_ns: Tuple[float, float] | Gate,
        meas_type: str,
        should_re_correlate=False,
        is_verbose=True,
        **kwargs,
    ) -> None:
        """
        A high-level method for correlating a measurement (usually STED) while imposing a TDC (post measurement) gate.
        """

        kwargs["is_verbose"] = is_verbose

        if not should_re_correlate and getattr(self, meas_type).cf.get(
            f"gated {meas_type} {tdc_gate_ns}"
        ):
            print(f"{meas_type}: gate {tdc_gate_ns} already exists. Skipping...")
            return

        if meas_type == "confocal":
            self.confocal.correlate_and_average(
                tdc_gate_ns=tdc_gate_ns, cf_name=meas_type, **kwargs
            )
        elif self.sted.is_loaded:
            self.sted.correlate_and_average(tdc_gate_ns=tdc_gate_ns, cf_name=meas_type, **kwargs)
        else:
            # STED measurement not loaded
            print("Cannot add STED gate if there's no STED measurement loaded to the experiment!")
            return

        if kwargs.get("should_plot"):
            self.plot_standard(**kwargs)

    def add_gates(
        self,
        gate_list: Sequence[Tuple[float, float] | Gate],
        meas_type="sted",
        should_plot=False,
        **kwargs,
    ):
        """
        A convecience method for adding multiple gates.
        """

        print(f"Adding multiple '{meas_type}' gates {gate_list} for experiment '{self.name}'...")
        for tdc_gate_ns in gate_list:
            self.add_gate(tdc_gate_ns, meas_type, **kwargs)

        if should_plot:
            self.plot_standard(**kwargs)

    def remove_gates(
        self,
        gate_list: Sequence[Tuple[float, float] | Gate] = None,
        meas_type="sted",
        should_plot=False,
        **kwargs,
    ):
        """
        A convecience method for removing multiple gates.
        """

        meas = getattr(self, meas_type)
        # remove all gates
        if gate_list is None:
            print(
                f"Removing ALL '{meas_type}' TDC gates {gate_list} for experiment '{self.name}'... ",
                end="",
            )
            meas.cf = {cf_name: cf for cf_name, cf in meas.cf.items() if not cf.gate_ns}

        else:
            print(
                f"Removing multiple '{meas_type}' gates {gate_list} for experiment '{self.name}'... ",
                end="",
            )
            for tdc_gate_ns in gate_list:
                for cf_name in meas.cf.keys():
                    if str(tdc_gate_ns) in cf_name:
                        meas.cf.pop(cf_name)
                        print(f"Removed {tdc_gate_ns}... ", end="")

        print("Done.")

        if should_plot:
            self.plot_standard(**kwargs)

    def plot_standard(self, should_add_exp_name=True, **kwargs):
        """Doc."""

        super_title = f"Experiment '{self.name}' - All ACFs"
        with Plotter(subplots=(1, 2), super_title=super_title, **kwargs) as axes:
            self.plot_correlation_functions(
                parent_ax=axes[0],
                y_field="avg_cf_cr",
                x_field="lag",
                should_add_exp_name=should_add_exp_name,
            )

            self.plot_correlation_functions(
                parent_ax=axes[1],
                should_add_exp_name=should_add_exp_name,
            )

    def plot_correlation_functions(  # NOQA C901
        self,
        xlim=None,
        ylim=None,
        x_field=None,
        y_field=None,
        x_scale=None,
        y_scale=None,
        should_add_exp_name=True,
        confocal_only=False,
        sted_only=False,
        **kwargs,
    ) -> List[Line2D]:
        """Doc."""

        if self.confocal.is_loaded and not sted_only:
            ref_meas = self.confocal
        else:
            ref_meas = self.sted

        # auto x_field/x_scale determination
        if x_field is None:
            if ref_meas.scan_type == "static":
                x_field = "lag"
                x_scale = "log" if not x_scale else x_scale
            else:
                x_field = "vt_um"
                x_scale = "linear" if not x_scale else x_scale
        elif x_field == "vt_um" and not x_scale:
            x_scale = "linear"
        elif x_field == "lag" and not x_scale:
            x_scale = "log"

        # auto xlim determination
        if xlim is None:
            if x_field == "vt_um":
                xlim = Limits(1e-3, 10) if x_scale == "log" else Limits(0, 2)
            else:
                xlim = Limits(1e-4, 1)

        # auto y_field/y_scale determination
        y_field = y_field or "normalized"
        y_scale = y_scale or ("log" if x_field == "vt_um" else "linear")

        # auto ylim determination
        if ylim is None:
            if y_field == "normalized":
                ylim = Limits(1e-3, 1) if y_scale == "log" else Limits(0, 1)
            elif y_field in {"average_all_cf_cr", "avg_cf_cr"}:
                # TODO: perhaps cf attribute should be a list and not a dict? all I'm really ever interested in is either showing the first or all together (names are in each CF anyway)
                first_cf = list(ref_meas.cf.values())[0]
                ylim = Limits(-1e3, first_cf.g0 * 1.2)

        with Plotter(
            super_title=f"'{self.name}' Experiment - All ACFs",
            **kwargs,
        ) as ax:

            if (parent_ax := kwargs.get("parent_ax")) is not None:
                existing_lines = parent_ax.get_lines()
                kwargs.pop("parent_ax")

            if confocal_only:
                meas_types = ["confocal"]
            elif sted_only:
                meas_types = ["sted"]
            else:
                meas_types = ["confocal", "sted"]

            for meas_type in meas_types:
                getattr(self, meas_type).plot_correlation_functions(
                    parent_ax=ax,
                    x_field=x_field,
                    xlim=xlim,
                    x_scale=x_scale,
                    y_field=y_field,
                    ylim=ylim,
                    y_scale=y_scale,
                    **kwargs,
                )

            # keep newly added lines
            new_lines = [line for line in ax.get_lines() if line not in existing_lines]

            # add experiment name to labels if plotted hierarchically (multiple experiments)
            # TODO: this could be perhaps a feature of Plotter? i.e., an addition to all labels can be passed at Plotter init?
            if parent_ax is not None:
                for line in new_lines:
                    label = line.get_label()
                    if "_" not in label:
                        line.set_label(f"{self.name}: {label}" if should_add_exp_name else label)

            ax.legend()

        return new_lines

    def estimate_spatial_resolution(self, colors=None, sted_only=False, **kwargs) -> Iterator[str]:
        """
        High-level method for performing Gaussian fits over 'normalized' vs. 'vt_um' fields of all correlation functions
        (confocal, sted and any gates) in order to estimate the resolution improvement.
        This is relevant only for calibration experiments (i.e. 300 bp samples).
        """

        with Plotter(
            super_title="Resolution Comparison by Gaussian Fitting",
            x_scale="quadratic",
            y_scale="log",
            xlim=(0.01, 1),
            ylim=(5e-3, 1),
            **kwargs,
        ) as ax:

            kwargs.pop("gui_display", None)  # TODO: should this be included in Plotter init?
            if (parent_ax := kwargs.pop("parent_ax", None)) is not None:
                # TODO: this could be perhaps a feature of Plotter? i.e., an addition to all labels can be passed at Plotter init?
                existing_lines = parent_ax.get_lines()

            remaining_colors = colors if colors is not None else cycle(default_colors)
            if not sted_only:
                remaining_colors = self.confocal.estimate_spatial_resolution(
                    parent_ax=ax, colors=remaining_colors, **kwargs
                )
            remaining_colors = self.sted.estimate_spatial_resolution(
                parent_ax=ax,
                colors=remaining_colors,
                **kwargs,
            )

            # add experiment name to labels if plotted hierarchically (multiple experiments)
            # TODO: this could be perhaps a feature of Plotter? i.e., an addition to all labels can be passed at Plotter init?
            if parent_ax is not None:
                for line in ax.get_lines():
                    if line not in existing_lines:
                        label = line.get_label()
                        if "_" not in label:
                            line.set_label(f"{self.name}: {label}")

            ax.legend()

        return remaining_colors

    def plot_afterpulsing_filters(self, **kwargs) -> None:
        """Plot afterpulsing filters each measurement"""
        # TODO: this can be improved, (plot both in single figure - plot method of AfterpulsingFilter doesn't match this)

        with suppress(AttributeError):
            for cf in self.cf_dict.values():
                cf.afterpulsing_filter.plot(
                    super_title=f"Afterpulsing Filter\n{cf.name}",
                    **kwargs,
                )

    def calculate_structure_factors(
        self, cal_exp, interp_types=["gaussian"], should_plot=True, is_verbose=True, **kwargs
    ):
        """
        Given a calibration SolutionSFCSExperiment, i.e. one performed
        on a below-resolution sample (e.g. a 300 bp DNA sample labeled with the same fluorophore),
        this method divides this experiment's (self) Hankel transforms by the corresponding ones
        in the calibration experiment (all calculated if needed) and returns the sought structure factors.
        """

        if is_verbose:
            print(f"Calculating all structure factors for '{self.name}' experiment... ", end="")

        # calculated without plotting
        kwargs["parent_names"] = (self.name, cal_exp.name)
        for meas in [getattr(self, meas_type) for meas_type in ("confocal", "sted")]:
            if meas.is_loaded:
                cal_meas = getattr(cal_exp, meas.type)
                getattr(self, meas.type).calculate_structure_factors(
                    cal_meas, interp_types, is_verbose=is_verbose, **kwargs
                )

        # keep reference to calibration experiment
        self.cal_exp = cal_exp

    def plot_structure_factors(self, plot_ht=False, **kwargs):
        """Doc."""

        # get interpolations types
        interp_types = list(list(self.confocal.cf.values())[0].structure_factors.keys())
        n_interps = len(interp_types)

        # optionally plot all transforms of all corrfuncs of all measurements in a single figure (for self and calibration)
        if plot_ht:
            for exp in (self, self.cal_exp):
                with Plotter(
                    subplots=(n_interps, 2),
                    super_title=f"Experiment '{exp.name}': Hankel Transforms",
                    **kwargs,
                ) as axes:
                    for meas in [getattr(exp, meas_type) for meas_type in ("confocal", "sted")]:
                        if meas.is_loaded:
                            meas.plot_hankel_transforms(parent_ax=axes, **kwargs)

        # plot the structure factors in another figure
        with Plotter(
            subplots=(n_interps, 1),
            super_title=f"Experiment '{self.name}': Structure factors",
            **kwargs,
        ) as axes:
            kwargs.pop("parent_ax", None)  # TODO: should this be included in Plotter init?
            for meas in [getattr(self, meas_type) for meas_type in ("confocal", "sted")]:
                if meas.is_loaded:
                    meas.plot_structure_factors(
                        parent_ax=axes,
                        **kwargs,
                    )
            for ax, interp_type in zip(axes if len(interp_types) > 1 else [axes], interp_types):
                ax.set_title(f"{interp_type.capitalize()} Interp./Extrap.")

    def fit_structure_factors(self, model: str):
        """Doc."""
        # NOTE: Check out
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.dawsn.html

        raise NotImplementedError


class ImageSFCSMeasurement:
    """Doc."""

    # TODO: Later, create ImageSFCSExperiment which will have .confocal and .sted (synchronize delay times and everything...)

    laser_freq_hz: int
    fpga_freq_hz: int
    detector_settings: Dict
    tdc_calib: TDCCalibration
    type: str

    def __init__(self):
        self.file_path = None
        self._file_dict = None
        self.data = TDCPhotonMeasurementData()
        self.tdc_image_data = None
        self.lifetime_image_data = None
        self.ci_image_data = None

    def __repr__(self):
        return f"ImageSFCSMeasurement({self.type}, {len(self.data)} files, '{self.file_path}')"

    def read_fpga_data(
        self,
        file_path: Path = None,
        file_dict: Dict = None,
        len_factor=0.005,
        is_verbose=True,
        **proc_options,
    ) -> None:
        """Doc."""

        # load file if needed
        if file_path is not None:
            if is_verbose:
                print("\nLoading image FPGA data from disk -")
                print(f"File path: '{self.file_path}'")
            self.file_path = file_path
            self._file_dict = load_file_dict(file_path)

        # determine type
        self.type = "sted" if self._file_dict["full_data"]["laser_mode"] == "sted" else "confocal"

        # get general properties from file_dict
        self._get_general_properties()

        # initialize data processor
        self.dump_path = DUMP_PATH / self.file_path.stem
        self.data_processor = TDCPhotonDataProcessor(
            self.dump_path, self.laser_freq_hz, self.fpga_freq_hz, self.detector_settings["gate_ns"]
        )
        # actual plane data processing
        if is_verbose:
            print(
                f"Loading and processing total data ({self.scan_settings['n_planes']} planes): '{self.file_path.stem}'...",
                end=" ",
            )
        # Processing data
        p = self.data_processor.process_data(
            0,
            self._file_dict["full_data"],
            len_factor=len_factor,
            is_verbose=is_verbose,
            **proc_options,
        )
        if is_verbose:
            print("Done.\n")
        # Appending data to self
        if p is not None:
            self.data.append(p)
        else:
            raise RuntimeError("Loading FPGA data catastrophically failed.")

        # calculate average count rate
        with suppress(TypeError):
            # TypeError: p.general.avg_cnt_rate_khz is None for legacy measurements.
            self.avg_cnt_rate_khz = np.mean([p.general.avg_cnt_rate_khz for p in self.data])
            try:
                self.std_cnt_rate_khz = np.std(
                    [p.general.avg_cnt_rate_khz for p in self.data], ddof=1
                )
            except RuntimeWarning:  # single file
                self.std_cnt_rate_khz = 0.0

        # done with loading
        if is_verbose:
            print("Finished loading FPGA data.\n")

    def _get_general_properties(
        self,
        should_ignore_hard_gate: bool = False,
        **kwargs,
    ) -> None:
        """Get general measurement properties."""

        full_data = self._file_dict["full_data"]

        self.laser_mode = full_data.get("laser_mode")
        self.scan_settings = full_data.get("scan_settings")

        # get countrate estimate (for multiprocessing threshod). calculated more precisely in the end of processing.
        self.avg_cnt_rate_khz = full_data.get("avg_cnt_rate_khz")

        self.afterpulse_params = self._file_dict["system_info"]["afterpulse_params"]
        self.detector_settings = full_data.get("detector_settings")
        self.delayer_settings = full_data.get("delayer_settings")
        self.laser_freq_hz = int(full_data["laser_freq_mhz"] * 1e6)
        self.pulse_period_ns = 1 / self.laser_freq_hz * 1e9
        self.fpga_freq_hz = int(full_data["fpga_freq_mhz"] * 1e6)
        self.ao_sampling_freq_hz = self.scan_settings["line_freq_hz"] * self.scan_settings["ppl"]

        # TODO: missing gate - move this to legacy handeling
        if self.detector_settings.get("gate_ns") is not None and (
            not self.detector_settings["gate_ns"] and self.detector_settings["mode"] == "external"
        ):
            print(
                "This should not happen (missing detector gate) - move this to legacy file handeling!"
            )
            self.detector_settings["gate_ns"] = Gate(
                hard_gate=(
                    98 - self.detector_settings["gate_width_ns"],
                    self.detector_settings["gate_width_ns"],
                )
            )
        elif should_ignore_hard_gate:
            self.detector_settings["gate_ns"] = None

        # sFCS
        self.scan_type = self.scan_settings["pattern"]

    def calibrate_tdc(self, force_processing=True, **kwargs) -> None:
        """Doc."""

        if not force_processing and hasattr(self, "tdc_calib"):
            print("\nTDC calibration exists, skipping.")
            if kwargs.get("should_plot"):
                self.tdc_calib.plot()
            return

        if kwargs.get("is_verbose"):
            print("\nCalibrating TDC...", end=" ")

        # perform actual TDC calibration
        self.tdc_calib = self.data_processor.calibrate_tdc(self.data, self.scan_type, **kwargs)

        if kwargs.get("should_plot"):
            self.tdc_calib.plot()

        if kwargs.get("is_verbose"):
            print("Done.")

    def generate_tdc_image_stack_data(
        self,
        file_path: Path = None,
        file_dict: Dict = None,
        gate_ns: Gate = Gate(),
        is_multiscan=False,
        **kwargs,
    ) -> ImageStackData:
        """Doc."""

        if not self.data:
            if file_path is not None:
                self.read_fpga_data(file_path=file_path, **kwargs)
            elif file_dict is not None:
                self.read_fpga_data(file_dict=file_dict, **kwargs)
            else:
                raise ValueError(
                    "Must supply either a file path or a file dictionary if no data is loaded!"
                )

        # definitions
        n_lines = self.scan_settings["n_lines"]
        n_planes = self.scan_settings["n_planes"]
        ppl = self.scan_settings["ppl"]
        ppp = ppl * n_lines

        # only one file for images. using .data as list for consistency with SolutionSFCSMeasurement
        data = self.data[0]

        # gate
        if gate_ns:
            # calibrate TDC
            self.calibrate_tdc()
            delay_time = data.raw._delay_time  # using just the first plane
            in_gate_idxs = gate_ns.valid_indices(delay_time)
        # ungated
        else:
            in_gate_idxs = slice(None)

        # define sample runtime
        pulse_runtime = data.raw._pulse_runtime
        sample_runtime = pulse_runtime * self.ao_sampling_freq_hz // self.laser_freq_hz

        # get pixel_num
        pixel_num = sample_runtime % ppl
        pixel_num = pixel_num[in_gate_idxs]  # gating

        # get line_num
        line_num_tot = sample_runtime // ppl
        line_num = (line_num_tot % n_lines).astype(np.int16)
        line_num = line_num[in_gate_idxs]  # gating

        # build the (optionally gated) images
        # planes are unique (slices)
        if not is_multiscan:
            # get plane_num
            plane_num_tot = sample_runtime // ppp
            plane_num = (plane_num_tot % n_planes).astype(np.int8)
            plane_num = plane_num[in_gate_idxs]  # gating
            gated_counts_stack = np.empty((n_lines, ppl, n_planes), dtype=np.uint16)
            for plane_idx in range(n_planes):
                bins = np.arange(-0.5, ppl)
                for line_idx in range(n_lines):
                    gated_counts_stack[line_idx, :, plane_idx], _ = np.histogram(
                        pixel_num[(line_num == line_idx) & (plane_num == plane_idx)], bins=bins
                    )
        # multiscan - plans are "identical" sequential scans - group photons by planes
        else:
            gated_counts_stack = np.empty((n_lines, ppl, 1), dtype=np.uint16)
            bins = np.arange(-0.5, ppl)
            for line_idx in range(n_lines):
                gated_counts_stack[line_idx, :, 0], _ = np.histogram(
                    pixel_num[line_num == line_idx], bins=bins
                )

        # get effective indices, pixels-per-line and line_ticks_v
        # TODO: what do "effective" indices mean?
        eff_idxs, pxls_per_line, line_ticks_v = self._get_effective_idices()

        self.tdc_image_data = ImageStackData(
            image_stack=gated_counts_stack,
            effective_idxs=eff_idxs,
            pxls_per_line=pxls_per_line,
            line_ticks_v=line_ticks_v,
            row_ticks_v=self.scan_settings["set_pnts_lines_odd"],
            plane_ticks_v=self.scan_settings.get("set_pnts_planes"),
            n_planes=n_planes,
            plane_orientation=self.scan_settings["plane_orientation"],
            dim_order=self.scan_settings["dim_order"],
            gate_ns=gate_ns,
        )
        return self.tdc_image_data

    def generate_lifetime_image_stack_data(
        self,
        file_path: Path = None,
        file_dict: Dict = None,
        gate_ns: Gate = Gate(),
        min_n_photons: int = None,
        img_median_factor=1.5,
        is_multiscan=False,
        auto_gating=True,
        auto_gate_width_ns=15,
        **kwargs,
    ) -> ImageStackData:
        """Doc."""

        if not self.data:
            if file_path is not None:
                self.read_fpga_data(file_path=file_path, **kwargs)
            elif file_dict is not None:
                self.read_fpga_data(file_dict=file_dict, **kwargs)
            else:
                raise ValueError(
                    "Must supply either a file path or a file dictionary if no data is loaded!"
                )

        # auto-gating STED
        if (self.type == "sted") and not gate_ns and auto_gating:
            print("Determining gate automatically, according to histogram peak... ", end="")
            # calibrate TDC (if not already calibrated)
            if not hasattr(self, "tdc_calib"):
                self.calibrate_tdc()
            peak_time_ns = self.tdc_calib.t_hist[np.nanargmax(self.tdc_calib.all_hist_norm)]
            gate_ns = Gate(peak_time_ns, auto_gate_width_ns + peak_time_ns)
            print(f"Done: {gate_ns}.")

        # auto-determination of 'min_n_photons' if not given
        if min_n_photons is None:
            print(
                f"Auto-determining 'min_n_photons' according to {img_median_factor:.1f} times median number of photons-per-pixel... ",
                end="",
            )
            # get TDC counts data (use existing or make new)
            if self.tdc_image_data is None:
                counts_stack = self.generate_tdc_image_stack_data(
                    gate_ns=gate_ns, is_multiscan=is_multiscan, **kwargs
                )
            else:
                counts_stack = self.tdc_image_data

            # use the median of the photon-containing pixels as the minimum
            valid_counts = counts_stack.image_stack_forward[counts_stack.image_stack_forward > 0]
            # assume equal contribution for each plane (median of entire stack)
            if (min_n_photons := round(np.median(valid_counts) * img_median_factor)) < 5:
                min_n_photons = 5
                print(f"Warning: min_n_photons < {min_n_photons}. Using 5...")
            else:
                print(f"Using min_n_photons={min_n_photons}.")

        # definitions
        n_lines = self.scan_settings["n_lines"]
        n_planes = self.scan_settings["n_planes"]
        ppl = self.scan_settings["ppl"]
        ppp = ppl * n_lines

        # only one file for images. using .data as list for consistency with SolutionSFCSMeasurement
        data = self.data[0]

        pulse_runtime = data.raw.pulse_runtime
        sample_runtime = pulse_runtime * self.ao_sampling_freq_hz // self.laser_freq_hz
        if not hasattr(self, "tdc_calib"):
            # calibrate TDC (if not already calibrated)
            self.calibrate_tdc()
        delay_time = data.raw.delay_time

        # gate
        if gate_ns:
            in_gate_idxs = gate_ns.valid_indices(delay_time)
            delay_time = delay_time[in_gate_idxs]
        # ungated
        else:
            in_gate_idxs = slice(None)

        # get pixel_num
        pixel_num = sample_runtime % ppl
        pixel_num = pixel_num[in_gate_idxs]  # gating

        # get line_num
        line_num_tot = sample_runtime // ppl
        line_num = (line_num_tot % n_lines).astype(np.int16)
        line_num = line_num[in_gate_idxs]  # gating

        # build the (optionally gated) images
        print("Building image stack (line-by-line)... ", end="")
        bins = np.arange(-0.5, ppl)
        # planes are unique (slices)
        if not is_multiscan:
            # get plane_num
            plane_num_tot = sample_runtime // ppp
            plane_num = (plane_num_tot % n_planes).astype(np.int8)
            plane_num = plane_num[in_gate_idxs]  # gating
            gated_lt_stack = np.zeros((n_lines, ppl, n_planes), dtype=np.float64)
            for plane_idx in range(n_planes):
                for line_idx in range(n_lines):
                    plane_line_idxs = (line_num == line_idx) & (plane_num == plane_idx)
                    bin_idxs_l = np.digitize(pixel_num[plane_line_idxs], bins)
                    for pxl_idx in range(ppl):
                        # check that there are enough photons in pixel
                        if np.nonzero(bin_idxs_l == pxl_idx)[0].size >= min_n_photons:
                            gated_lt_stack[line_idx, pxl_idx, plane_idx] = np.median(
                                delay_time[plane_line_idxs][bin_idxs_l == pxl_idx]
                            )
                    print(".", end="")
                print(f"({plane_idx})")

        # multiscan - plans are "identical" sequential scans - group photons by planes
        else:
            gated_lt_stack = np.zeros((n_lines, ppl, 1), dtype=np.float64)
            for line_idx in range(n_lines):
                line_idxs = line_num == line_idx
                bin_idxs_l = np.digitize(pixel_num[line_idxs], bins)
                for pxl_idx in range(ppl):
                    # check that there are enough photons in pixel
                    if np.nonzero(bin_idxs_l == pxl_idx)[0].size >= min_n_photons:
                        gated_lt_stack[line_idx, pxl_idx, 0] = np.median(
                            delay_time[line_idxs][bin_idxs_l == pxl_idx]
                        )
                print(".", end="")
        print(" Done.")

        # get effective indices, pixels-per-line and line_ticks_v
        # TODO: what do "effective" indices mean?
        eff_idxs, pxls_per_line, line_ticks_v = self._get_effective_idices()

        self.lifetime_image_data = ImageStackData(
            image_stack=gated_lt_stack,
            effective_idxs=eff_idxs,
            pxls_per_line=pxls_per_line,
            line_ticks_v=line_ticks_v,
            row_ticks_v=self.scan_settings["set_pnts_lines_odd"],
            plane_ticks_v=self.scan_settings.get("set_pnts_planes"),
            n_planes=n_planes,
            plane_orientation=self.scan_settings["plane_orientation"],
            dim_order=self.scan_settings["dim_order"],
            gate_ns=gate_ns,
            is_lifetime_img=True,
        )
        return self.lifetime_image_data

    def generate_ci_image_stack_data(
        self, file_path: Path = None, file_dict: Dict = None, is_multiscan=False, **kwargs
    ) -> ImageStackData:
        """Doc."""

        if file_path is not None:
            self.file_path = file_path
            self._file_dict = load_file_dict(file_path)
        elif file_dict is not None:
            self._file_dict = file_dict
        # already loaded once
        elif not hasattr(self, "file_path"):
            raise ValueError("Must supply either a file path or a file dictionary!")

        # Get counts (ungated) image (excitation or sted)
        counts = self._file_dict["full_data"]["ci"]
        self.scan_settings = self._file_dict["full_data"]["scan_settings"]

        n_planes = self.scan_settings["n_planes"]
        n_lines = self.scan_settings["n_lines"]
        dim_order = self.scan_settings["dim_order"]
        ppl = self.scan_settings["ppl"]
        ppp = n_lines * ppl

        # prepare to remove counts from outside limits
        eff_idxs, pxls_per_line, line_ticks_v = self._get_effective_idices()

        # create counts stack shaped (n_lines, ppl, n_planes) - e.g. 80 x 1000 x 1
        j0 = ppp * np.arange(n_planes)[:, np.newaxis]
        J = np.tile(np.arange(ppp), (n_planes, 1)) + j0
        counts_stack = np.diff(np.concatenate((j0, counts[J]), axis=1))
        counts_stack = counts_stack.T.reshape(n_lines, ppl, n_planes)

        # sum the planes if multiscan
        if is_multiscan and counts_stack.shape[2] > 1:
            counts_stack = np.atleast_3d(counts_stack.sum(axis=2))

        self.ci_image_data = ImageStackData(
            image_stack=counts_stack,
            effective_idxs=eff_idxs,
            pxls_per_line=pxls_per_line,
            line_ticks_v=line_ticks_v,
            row_ticks_v=self.scan_settings["set_pnts_lines_odd"],
            plane_ticks_v=self.scan_settings.get("set_pnts_planes"),
            n_planes=n_planes,
            plane_orientation=self.scan_settings["plane_orientation"],
            dim_order=dim_order,
        )
        return self.ci_image_data

    def _get_effective_idices(self):
        """Doc."""

        # existing meas params definitions
        ao = self.scan_settings["ao"].T
        um_v_ratio = self._file_dict["system_info"]["xyz_um_to_v"]
        dim_order = self.scan_settings["dim_order"]
        ppl = self.scan_settings["ppl"]
        n_lines = self.scan_settings["n_lines"]

        # derived params
        pxl_size_um = self.scan_settings["dim2_um"] / n_lines
        first_dim = dim_order[0]
        dim1_center = self.scan_settings["initial_ao"][first_dim]
        um_per_v = um_v_ratio[first_dim]
        line_len_v = self.scan_settings["dim1_um"] / um_per_v
        dim1_min = dim1_center - line_len_v / 2
        pxl_size_v = pxl_size_um / um_per_v
        dim1_ao_single = ao[0][:ppl]

        # calculate and return needed values
        eff_idxs = ((dim1_ao_single - dim1_min) // pxl_size_v + 1).astype(np.int16)
        pxls_per_line = int(np.ceil(self.scan_settings["dim1_um"] / pxl_size_um))
        line_ticks_v = dim1_min + np.arange(pxls_per_line) * pxl_size_v
        return eff_idxs, pxls_per_line, line_ticks_v

    def preview(self, method="forward normalized", should_plot=True, **kwargs) -> np.ndarray:
        """Generate and show the CI image"""

        self.generate_ci_image_stack_data(**kwargs)
        img = self.ci_image_data.construct_plane_image(method, **kwargs)
        if should_plot:
            with Plotter(**kwargs) as ax:
                ax.imshow(img)
        return img

    def estimate_spatial_resolution(self, should_plot=True, **kwargs) -> FitParams:
        """Fit a 2D Gaussian to image in order to estimate the resolution (e.g. for fluorescent beads)"""

        img = self.tdc_image_data.construct_plane_image("forward normalized", **kwargs)
        fp = fit_2d_gaussian_to_image(img)

        x0, y0, sigma_x, sigma_y, phi = (
            fp.beta["x0"],
            fp.beta["y0"],
            fp.beta["sigma_x"],
            fp.beta["sigma_y"],
            fp.beta["phi"],
        )
        _, _, sigma_x_err, sigma_y_err, _ = (
            fp.beta_error["x0"],
            fp.beta_error["y0"],
            fp.beta_error["sigma_x"],
            fp.beta_error["sigma_y"],
            fp.beta_error["phi"],
        )

        max_sigma_y, max_sigma_x = img.shape
        if (
            x0 < 0
            or y0 < 0
            or abs(1 - sigma_x / sigma_y) > 2
            or sigma_x > max_sigma_x
            or sigma_y > max_sigma_y
        ):
            print(f"Gaussian fit is irrational!\n({fp.beta})")

        # calculating the FWHM
        pxl_size_um = self.scan_settings["dim1_um"] / self.tdc_image_data.effective_binned_size
        FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))  # 1/e^2 width is FWHM * 1.699
        one_over_e2_factor = 1.699 * FWHM_FACTOR
        diameter_x_nm = sigma_x * one_over_e2_factor * pxl_size_um * 1e3
        diameter_x_nm_err = sigma_x_err * one_over_e2_factor * pxl_size_um * 1e3
        diameter_y_nm = sigma_y * one_over_e2_factor * pxl_size_um * 1e3
        diameter_y_nm_err = sigma_y_err * one_over_e2_factor * pxl_size_um * 1e3

        # print
        dim1_char, dim2_char = self.scan_settings["plane_orientation"]
        print(
            f"1/e^2 diameter determined to be:\n{dim1_char}: {diameter_x_nm:.0f} +/- {diameter_x_nm_err:.0f} nm\n{dim2_char}: {diameter_y_nm:.0f} +/- {diameter_y_nm_err:.0f} nm"
        )

        if should_plot:
            ellipse = Ellipse(
                xy=(x0, y0),
                width=sigma_x * one_over_e2_factor,
                height=sigma_y * one_over_e2_factor,
                angle=phi,
            )
            ellipse.set_facecolor((0, 0, 0, 0))
            ellipse.set_edgecolor("red")
            annotation = f"$1/e^2$: \n{dim1_char}: {diameter_x_nm:.0f} +/- {diameter_x_nm_err:.0f} nm\n{dim2_char}: {diameter_y_nm:.0f} +/- {diameter_y_nm_err:.0f} nm\n$\\chi^2$={fp.chi_sq_norm:.2e}"
            with Plotter(**kwargs) as ax:
                ax.imshow(img)
                ax.add_artist(ellipse)
                ax.annotate(
                    annotation,
                    ellipse._center,
                    color="w",
                    weight="bold",
                    fontsize=11,
                    ha="center",
                    va="center",
                )

        return fp


def calculate_calibrated_afterpulse(
    lag: np.ndarray,
    afterpulse_params: Tuple[str, np.ndarray] = default_system_info["afterpulse_params"],
    gate_ns: Union[Tuple[float, float], Gate] = (0, np.inf),
    laser_freq_hz: float = 1e7,
) -> np.ndarray:
    """Doc."""

    gate_pulse_period_ratio = (
        Gate(gate_ns).interval(upper_max=1e9 / laser_freq_hz) / 1e9 * laser_freq_hz
    )
    fit_name, beta = afterpulse_params
    if fit_name == "multi_exponent_fit":
        # work with any number of exponents
        afterpulse = gate_pulse_period_ratio * multi_exponent_fit(lag, *beta)
    elif fit_name == "exponent_of_polynom_of_log":  # for old MATLAB files
        if lag[0] == 0:
            lag[0] = np.nan
        afterpulse = gate_pulse_period_ratio * np.exp(np.polyval(beta, np.log(lag)))

    return afterpulse
