"""Data organization and manipulation."""

import logging
import multiprocessing as mp
import re
from contextlib import suppress
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Tuple, Union, cast

import numpy as np
from scipy.special import j0, j1, jn_zeros
from sklearn import linear_model

from data_analysis.data_processing import (
    CountsImageMixin,
    TDCCalibration,
    TDCPhotonDataProcessor,
    TDCPhotonFileData,
    TDCPhotonMeasurementData,
)
from data_analysis.software_correlator import CorrelatorType, SoftwareCorrelator
from data_analysis.workers import N_CPU_CORES, data_processing_worker, io_worker
from utilities.display import Plotter, default_colors
from utilities.file_utilities import (
    DUMP_PATH,
    default_system_info,
    load_file_dict,
    load_processed_solution_measurement,
    prepare_file_paths,
    save_object,
)
from utilities.fit_tools import (
    FIT_NAME_DICT,
    FitParams,
    curve_fit_lims,
    multi_exponent_fit,
)
from utilities.helper import (
    EPS,
    Gate,
    InterpExtrap1D,
    Limits,
    extrapolate_over_noise,
    timer,
    unify_length,
)


@dataclass
class StructureFactor:
    """Holds structure factor data"""

    q: np.ndarray
    sq: np.ndarray
    sq_error: np.ndarray = None

    def plot(self, interp_type: str, label_prefix="", **kwargs):
        """
        Plot a single structure factor. Meant to be used for hierarchical plotting
        directly from a measurement of indirectly from an experiment.
        """

        with Plotter(
            super_title="Structure Factors",
            y_scale="log",
            **kwargs,
        ) as ax:
            ax.plot(self.q, self.sq / self.sq[0], label=label_prefix)
            ax.set_xscale("log")
            ax.set_ylim(1e-4, 2)
            ax.set_xlabel("$q\\ \\left(\\frac{1}{\\mu m}\\right)$")
            ax.set_ylabel("$S(q)$")
            ax.legend()


@dataclass
class HankelTransform:
    """Holds results of Hankel transform"""

    IE: InterpExtrap1D
    q: np.ndarray
    fq: np.ndarray

    def plot(self, label_prefix="", **kwargs):
        """Display the transform's results"""

        with Plotter(
            subplots=(1, 2),
            y_scale="log",
            **kwargs,
        ) as axes:
            kwargs.pop("parent_ax", None)  # TODO: should this be included in Plotter init?
            self.IE.plot(parent_ax=axes[0], label_prefix=label_prefix, **kwargs)
            axes[0].set_xscale("function", functions=(lambda x: x ** 2, lambda x: x ** (1 / 2)))
            axes[0].set_title("Interp./Extrap. Testing")
            axes[0].set_xlabel("vt_um")
            axes[0].set_ylabel(f"{self.IE.interp_type.capitalize()} Interp./Extrap.")
            axes[0].legend()

            axes[1].plot(self.q, self.fq / self.fq[0], label=f"{label_prefix}Hankel transform")
            axes[1].set_xscale("log")
            axes[1].set_ylim(1e-4, 2)
            axes[1].set_title("Hankel Transforms")
            axes[1].set_xlabel("$q\\ \\left(\\frac{1}{\\mu m}\\right)$")
            axes[1].set_ylabel("$F(q)$")
            axes[1].legend(loc="lower left")


@dataclass
class LifeTimeParams:
    """Doc."""

    lifetime_ns: float
    sigma_sted: Union[float, Tuple[float, float]]
    laser_pulse_delay_ns: float


class CorrFunc:
    """Doc."""

    # Initialize the software correlator once (for all instances)
    SC = SoftwareCorrelator()

    afterpulse: np.ndarray
    vt_um: np.ndarray
    cf_cr: np.ndarray
    g0: float

    def __init__(self, name: str, correlator_type: int, laser_freq_hz, **kwargs):
        self.name = name
        self.correlator_type = correlator_type
        self.laser_freq_hz = laser_freq_hz
        self.fit_params: Dict[str, FitParams] = dict()
        self.afterpulsing_filter = kwargs.get("afterpulsing_filter")
        self.duration_min = kwargs.get("duration_min")

    def __add__(self, other):
        """Averages all attributes of two CorrFunc objects and returns a new CorrFunc instance"""

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
            self.afterpulsing_filter,
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
        time_stamp_split_list: List[np.ndarray],
        *args,
        is_verbose: bool = False,
        **kwargs,
    ) -> None:
        """Doc."""

        if kwargs.get("is_verbose"):
            print(f"Correlating {len(time_stamp_split_list)} splits -", end=" ")

        output = self.SC.correlate_list(
            time_stamp_split_list,
            self.correlator_type,
            timebase_ms=1000 / self.laser_freq_hz,
            **kwargs,
        )
        self.lag = max(output.lag_list, key=len)

        if kwargs.get("is_verbose"):
            print(". Processing correlator output...", end=" ")

        self._process_correlator_list_output(output, *args, **kwargs)

    def _process_correlator_list_output(
        self,
        corr_output,
        afterpulse_params,
        bg_corr_list,
        gate_ns=Gate(),
        should_subtract_afterpulsing: bool = False,
        external_afterpulsing: np.ndarray = None,
        **kwargs,
    ) -> None:
        """Doc."""

        # subtract background correlation
        for idx, bg_corr_dict in enumerate(bg_corr_list):
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
                    self.lag, afterpulse_params, gate_ns, self.laser_freq_hz
                )

        # zero-pad and generate cf_cr
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
                self.cf_cr[idx] = corr_output.countrate_list[idx].a * self.corrfunc[idx]
            with suppress(AttributeError):  # no .subtracted_afterpulsing attribute
                # ext. afterpulse might be shorter/longer
                self.cf_cr[idx] -= unify_length(self.subtracted_afterpulsing, (lag_len,))

        self.countrate_list = corr_output.countrate_list
        try:  # xcorr
            self.countrate_a = np.mean([countrate_pair.a for countrate_pair in self.countrate_list])
            self.countrate_b = np.mean([countrate_pair.b for countrate_pair in self.countrate_list])
        except AttributeError:  # autocorr
            self.countrate = np.mean([countrate for countrate in self.countrate_list])

    def average_correlation(
        self,
        rejection=2,
        reject_n_worst=None,
        norm_range=(1e-3, 2e-3),
        delete_list=[],
        plot_kwargs={},
        **kwargs,
    ) -> None:
        """Doc."""

        self.rejection = rejection
        self.norm_range = Limits(norm_range)
        self.delete_list = delete_list
        self.average_all_cf_cr = (self.cf_cr * self.weights).sum(0) / self.weights.sum(0)
        self.median_all_cf_cr = np.median(self.cf_cr, axis=0)
        jj = Limits(self.norm_range.upper, np.inf).valid_indices(
            self.lag
        )  # work in the relevant part

        try:
            self.score = (
                (1 / np.var(self.cf_cr[:, jj], 0))
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
                f"Division by zero avoided by adding EPSILON={EPS}. Why does this happen (zero in variance)?"
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

        self.avg_corrfunc, self.error_corrfunc = self._calculate_weighted_avg(
            self.corrfunc[self.j_good, :], self.weights[self.j_good, :]
        )

        j_t = self.norm_range.valid_indices(self.lag)

        try:
            self.g0 = (self.avg_cf_cr[j_t] / self.error_cf_cr[j_t] ** 2).sum() / (
                1 / self.error_cf_cr[j_t] ** 2
            ).sum()
        except RuntimeWarning:  # division by zero
            self.g0 = (self.avg_cf_cr[j_t] / (self.error_cf_cr[j_t] + EPS) ** 2).sum() / (
                1 / (self.error_cf_cr[j_t] + EPS) ** 2
            ).sum()
            print(
                f"Division by zero avoided by adding EPSILON={EPS}. Why does this happen (zero in variance)?"
            )

        self.normalized = self.avg_cf_cr / self.g0
        self.error_normalized = self.error_cf_cr / self.g0

        if kwargs.get("should_plot"):
            self.plot_correlation_function(plot_kwargs=plot_kwargs)

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
                f"Division by zero avoided by adding epsilon={EPS}. Why does this happen (zero total weight)?"
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
    ) -> None:

        x = getattr(self, x_field)
        y = getattr(self, y_field)
        if x_scale == "log":  # remove zero point data
            x, y = x[1:], y[1:]

        if kwargs.get("label") is None:
            label = self.name

        with Plotter(
            x_scale=x_scale,
            y_scale=y_scale,
            should_autoscale=True,
            **kwargs,
        ) as ax:
            ax.set_xlabel(x_field)
            ax.set_ylabel(y_field)
            ax.plot(x, y, "-", label=label, **kwargs.get("plot_kwargs", {}))

    def fit_correlation_function(
        self,
        fit_name="diffusion_3d_fit",
        x_field=None,
        y_field=None,
        y_error_field=None,
        fit_param_estimate=None,
        fit_range=(np.NINF, np.inf),
        x_scale=None,
        y_scale=None,
        bounds=(np.NINF, np.inf),
        max_nfev=int(1e4),
        **kwargs,
    ) -> None:

        if fit_param_estimate is None:
            if fit_name == "diffusion_3d_fit":
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
                if fit_range == (np.NINF, np.inf):
                    fit_range = (1e-3, 10)

            elif fit_name == "zero_centered_zero_bg_normalized_gaussian_1d_fit":
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
                if fit_range == (np.NINF, np.inf):
                    fit_range = (1e-2, 100)

        x = getattr(self, x_field)
        y = getattr(self, y_field)
        error_y = getattr(self, y_error_field)
        if x_scale == "log":
            x = x[1:]  # remove zero point data
            y = y[1:]
            error_y = error_y[1:]

        FP = curve_fit_lims(
            FIT_NAME_DICT[fit_name],
            fit_param_estimate,
            x,
            y,
            error_y,
            x_limits=Limits(fit_range),
            bounds=bounds,
            max_nfev=max_nfev,
            plot_kwargs=dict(x_scale=x_scale, y_scale=y_scale),
            **kwargs,
        )
        self.fit_params = FP

        return FP

    def calculate_hankel_transform(
        self,
        interp_types,
        **kwargs,
    ) -> None:
        """Doc."""

        def hankel_transform(
            r: np.ndarray,
            fr: np.ndarray,
            interp_type: str,
            max_r=10,
            r_interp_lims: Tuple[float, float] = (0.05, 5),
            fr_interp_lims: Tuple[float, float] = (1e-8, np.inf),  # (3e-2, np.inf),
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

        # plot the transforms with interpolations for testing
        if kwargs.get("should_plot"):
            with Plotter(subplots=((n_rows := len(interp_types)), 2), **kwargs) as axes:
                kwargs.pop("parent_ax", None)  # TODO: should this be included in Plotter init?
                for transform, ax_row in zip(
                    self.hankel_transforms.values(), axes if n_rows > 1 else [axes]
                ):
                    transform.plot(parent_ax=ax_row, label_prefix=f"{self.name}: ", **kwargs)

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

    def calculate_structure_factor(
        self,
        cal_cf,
        interp_types: List[str],
        parent_ax=None,
        should_plot=False,
        is_verbose=False,
        **kwargs,
    ):
        """
        Given a calibration CorrFunc object, i.e. one performed on a below-resolution sample
        (e.g. a 300 bp DNA sample labeled with the same fluorophore),
        this method divides this measurement's (self) Hankel transforms by the corresponding ones
        in the calibration measurement (all calculated if needed) and returns the sought structure factors.
        """

        if is_verbose:
            print(f"Calculating '{self.name}' Hankel transform...", end=" ")

        # calculate Hankel transforms (with selected interpolation type) if needed, without plotting
        for CF in (self, cal_cf):
            # only if needed (usually performed upper in the hierarchy, i.e. in SolutionSFCSMeasurement or SolutionSFCSExperiment)
            if interp_types != list(CF.hankel_transforms.keys()):
                CF.calculate_hankel_transform(interp_types, is_verbose=False, **kwargs)

        # get the structure factors by dividing the Hankel transforms
        self.structure_factors = {}
        for interp_type in interp_types:
            HT = self.hankel_transforms[interp_type]
            cal_HT = cal_cf.hankel_transforms[interp_type]
            self.structure_factors[interp_type] = StructureFactor(HT.q, HT.fq / cal_HT.fq)

        # plot interpolations/transforms for testing, and finally the structure factors
        if should_plot:
            with Plotter(
                subplots=((n_rows := len(interp_types)), 1), parent_ax=parent_ax, **kwargs
            ) as axes:
                kwargs.pop("parent_ax", None)  # TODO: should this be included in Plotter init?
                for (interp_type, structure_factor), ax in zip(
                    self.structure_factors.items(), (axes if n_rows > 1 else [axes])
                ):
                    structure_factor.plot(
                        interp_type, parent_ax=ax, label_prefix=f"{self.name}: ", **kwargs
                    )

        if is_verbose:
            print("Done.")


class SolutionSFCSMeasurement:
    """Doc."""

    tdc_calib: TDCCalibration
    laser_freq_hz: int
    fpga_freq_hz: int
    scan_settings: dict[str, Any]

    def __init__(self, type):
        self.type = type
        self.cf: dict = dict()
        self.xcf: dict = dict()
        self.scan_type: str
        self.duration_min: float = None
        self.is_loaded = False
        self.data = TDCPhotonMeasurementData()

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
            self.roi_list = [p.general.roi for p in self.data]
            self.bg_line_corr_list = [
                bg_line_corr
                for bg_file_corr in [p.general.bg_line_corr for p in self.data]
                for bg_line_corr in bg_file_corr
            ]

        else:  # static
            self.bg_line_corr_list = []

        # calculate average count rate
        self.avg_cnt_rate_khz = np.mean([p.general.avg_cnt_rate_khz for p in self.data])
        try:
            self.std_cnt_rate_khz = np.std([p.general.avg_cnt_rate_khz for p in self.data], ddof=1)
        except RuntimeWarning:  # single file
            self.std_cnt_rate_khz = 0.0

        calc_duration_mins = sum([p.general.duration_s for p in self.data]) / 60
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
                        ax.set_title(f"file #{file_idx+1} of\n'{self.type}' measurement")
                        ax.set_xlabel("Pixel Number")
                        ax.set_ylabel("Line Number")
                        ax.imshow(image, interpolation="none")
                        ax.plot(roi["col"], roi["row"], color="white")
            elif self.scan_type == "circle":
                # TODO: FILL ME IN (plotting in jupyter notebook, same as above angular scan stuff)
                pass
            print("Done.\n")

    @timer()
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
            (self.avg_cnt_rate_khz * 1e3 * self.duration_min * 60) * 7 / 1e6
        )
        print(
            "total_byte_data_size_estimate_mb (all files): ", total_byte_data_size_estimate_mb
        )  # TESTESTEST
        if (
            should_parallel_process
            and ((n_files := len(file_paths)) >= 5)
            and (total_byte_data_size_estimate_mb > 2000)
        ):
            # -2 is one CPU left to OS and one for I/O
            N_RAM_PROCESSES = N_CPU_CORES - 2

            print(
                f"Multi-processing {n_files} files using {N_RAM_PROCESSES} processes (+1 process for I/O)... ",
                end="",
            )

            # initialize 3 queues managed by a Manager
            manager = mp.Manager()
            io_queue = manager.Queue()
            data_processing_queue = manager.Queue()
            processed_queue = manager.Queue()

            # initialize a list to keep track of processes
            process_list = []

            # initialize IO worker (one process)
            io_process = mp.Process(
                target=io_worker,
                name="IO",
                args=(
                    io_queue,
                    data_processing_queue,
                    processed_queue,
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
            for _ in range(N_RAM_PROCESSES):
                data_processing_process = mp.Process(
                    target=data_processing_worker,
                    args=(data_processing_queue, io_queue),
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
                process.close()

            # finally, collect the file data objects
            for _ in range(n_files):
                self.data.append(processed_queue.get())

            print("\nMultiprocessing complete!")

        # serial processing (default)
        else:
            for idx, file_path in enumerate(file_paths):
                # Processing data
                p = self.process_data_file(idx, file_path, is_verbose=True, **proc_options)
                print("Done.\n")
                # Appending data to self
                if p is not None:
                    self.data.append(p)

        # auto determination of run duration
        self.run_duration = sum([p.general.duration_s for p in self.data])

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
        self.avg_cnt_rate_khz = full_data.get("avg_cnt_rate_khz")

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
            print("This should not happen (missing detector gate) - move this to legacy handeling!")
            self.detector_settings["gate_ns"] = Gate(
                98 - self.detector_settings["gate_width_ns"],
                self.detector_settings["gate_width_ns"],
                is_hard=True,
            )
        elif self.detector_settings.get("gate_ns") is None or should_ignore_hard_gate:
            self.detector_settings["gate_ns"] = Gate()

        # sFCS
        if scan_settings := full_data.get("scan_settings"):
            self.scan_type = scan_settings["pattern"]
            self.scan_settings = scan_settings
            self.v_um_ms = self.scan_settings["speed_um_s"] * 1e-3
            if self.scan_type == "circle":  # Circular sFCS
                self.ao_sampling_freq_hz = self.scan_settings.get("ao_sampling_freq_hz", int(1e4))
                self.diameter_um = self.scan_settings.get("diameter_um", 50)

        # FCS
        else:
            self.scan_type = "static"

    def process_data_file(
        self, idx=0, file_path: Path = None, file_dict: dict = None, **proc_options
    ) -> TDCPhotonFileData:
        """Doc."""

        if not file_path and not file_dict:
            raise ValueError("Must supply either a valid path or a ")

        # if using existing file_dict - usually during alignment measurements
        if file_dict is not None:
            self._get_general_properties(file_dict=file_dict, **proc_options)
            file_idx = 1
            self.dump_path = DUMP_PATH / "temp_meas"

        # File Data Loading
        if file_path is not None:  # Loading file from disk
            *_, template = file_path.parts
            try:
                file_idx = int(re.split("_(\\d+)\\.", template)[1])
            except IndexError:  # legacy template style
                file_idx = int(re.split("(\\d+)\\.", template)[1])
            print(
                f"Loading and processing file No. {file_idx} ({self.n_paths} files): '{template}'...",
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

        return self.data_processor.process_data(idx, file_dict["full_data"], **proc_options)

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

    def correlate_data(  # NOQA C901
        self,
        cf_name="unnamed",
        tdc_gate_ns=Gate(),
        afterpulsing_method=None,
        external_afterpulse_params=None,
        external_afterpulsing=None,
        get_afterpulsing=False,
        should_subtract_bg_corr=True,
        is_verbose=False,
        **corr_options,
    ) -> CorrFunc:
        """
        High level function for correlating any type of data (e.g. static, angular scan, circular scan...)
        Returns a 'CorrFunc' object.
        Data attribute is memory-mapped from disk.
        """

        if is_verbose:
            print(
                f"{self.type} - Preparing split data ({self.n_files} files) for software correlator...",
                end=" ",
            )

        # keep afterpulsing method for consistency with future gating
        if not hasattr(self, "afterpulsing_method"):
            self.afterpulsing_method = (
                "filter" if afterpulsing_method is None else afterpulsing_method
            )
        afterpulsing_method = (
            self.afterpulsing_method if afterpulsing_method is None else afterpulsing_method
        )

        # Unite TDC gate and detector gate
        gate_ns = Gate(tdc_gate_ns) & self.detector_settings["gate_ns"]

        #  add gate to cf_name
        if gate_ns:
            # TODO: cf_name should not contain any description other than gate and "afterpulsing" yes/no (description is in self.type)
            cf_name = f"gated {cf_name} {gate_ns}"

        # the following conditions require TDC calibration prior to creating splits
        if afterpulsing_method not in {"subtract calibrated", "filter", "none"}:
            raise ValueError(f"Invalid afterpulsing_method chosen: {afterpulsing_method}.")
        elif (is_filtered := afterpulsing_method == "filter") or gate_ns:
            if not hasattr(self, "tdc_calib"):  # calibrate TDC (if not already calibrated)
                if is_verbose:
                    print("(Calibrating TDC first...)", end=" ")
                self.calibrate_tdc(**corr_options)

        # Calculate afterpulsing filter if doesn't alreay exist (optional)
        afterpulsing_filter = None
        if is_filtered:
            if is_verbose:
                print("Preparing Afterpulsing filter... ", end="")
            afterpulsing_filter = self.tdc_calib.calculate_afterpulsing_filter(
                gate_ns, self.type, **corr_options
            )
            if is_verbose:
                print("Done.")

        # build correlator input - create list of split data (and optionally filters) for correlator. TDC-gating is performed here
        if is_verbose:
            print("Building correlator input splits: ", end="")
        # -
        corr_input_list, filter_input_list = zip(
            *self.data.prepare_xcorr_input(
                ["AA"],
                gate1_ns=gate_ns,
                afterpulsing_filter=afterpulsing_filter if is_filtered else None,
                get_afterpulsing=get_afterpulsing,
            )["AA"]
        )
        if is_verbose:
            print("Done.")

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

        if is_verbose:
            print(f"Correlating {self.scan_type} data ({cf_name}):", end=" ")

        # Correlate data
        CF = CorrFunc(
            cf_name,
            correlator_option,
            self.laser_freq_hz,
            afterpulsing_filter=afterpulsing_filter if is_filtered else None,
            duration_min=self.duration_min,
        )
        CF.correlate_measurement(
            corr_input_list,
            external_afterpulse_params
            if external_afterpulse_params is not None
            else self.afterpulse_params,
            getattr(self, "bg_line_corr_list", []) if should_subtract_bg_corr else [],
            external_afterpulsing=external_afterpulsing,
            gate_ns=gate_ns,
            list_of_filter_arrays=filter_input_list if is_filtered else None,
            should_subtract_afterpulsing=afterpulsing_method == "subtract calibrated",
            **corr_options,
        )

        try:  # temporal to spatial conversion, if scanning
            CF.vt_um = self.v_um_ms * CF.lag
        except AttributeError:
            CF.vt_um = CF.lag  # static

        if is_verbose:
            print("- Done.")

        # name the Corrfunc object
        # TODO: this should be eventually a list, not a dict (only first element and all together are ever interesting)
        self.cf[cf_name] = CF

        return CF

    def cross_correlate_data(
        self,
        xcorr_types=["AB", "BA"],
        cf_name=None,
        gate1_ns=Gate(),
        gate2_ns=Gate(),
        afterpulse_params=None,
        should_subtract_bg_corr=True,
        is_verbose=False,
        should_add_to_xcf_dict=True,
        **kwargs,
    ) -> Dict[str, CorrFunc]:
        """Doc."""
        # TODO: currently does not support 'Enderlein-filtering' - needs new correlator types built and code here fixed according to 'correlate_data'

        if is_verbose:
            print(
                f"{self.type} - Preparing split data ({self.n_files} files) for software correlator...",
                end=" ",
            )

        # Unite TDC gates and detector gates
        gates = []
        for i in (1, 2):
            gate_ns = locals()[f"gate{i}_ns"]
            tdc_gate_ns = Gate(gate_ns)
            if tdc_gate_ns or self.detector_settings["gate_ns"]:
                if not hasattr(self, "tdc_calib"):  # calibrate TDC (if not already calibrated)
                    self.calibrate_tdc(is_verbose=is_verbose)
                effective_lower_gate_ns = max(
                    tdc_gate_ns.lower, self.detector_settings["gate_ns"].lower
                )
                effective_upper_gate_ns = min(
                    tdc_gate_ns.upper, self.detector_settings["gate_ns"].upper
                )
                gates.append(Limits(effective_lower_gate_ns, effective_upper_gate_ns))
        gate1_ns, gate2_ns = gates

        # create list of split data for correlator
        dt_ts_split_dict = self.data.prepare_xcorr_input(
            xcorr_types,
            gate1_ns=gate1_ns,
            gate2_ns=gate2_ns,
        )

        if is_verbose:
            print("Done.")

        # disregard the first line of each split (delay_time, used for gating in autocorrelation) and convert to int32 for correlator
        corr_input_dict: Dict[str, List] = {xx: [] for xx in xcorr_types}
        for xx in xcorr_types:
            for split in dt_ts_split_dict[xx]:
                corr_input_dict[xx].append(np.squeeze(split[1:].astype(np.int32)))

        # correlate data
        if is_verbose:
            print(
                f"Correlating ({', '.join(xcorr_types)}) {self.scan_type} data ({cf_name} [{gate1_ns} vs. {gate2_ns}]):",
                end=" ",
            )

        if self.scan_type in {"static", "circle"}:
            correlator_option = SimpleNamespace(
                auto=CorrelatorType.PH_DELAY_CORRELATOR,
                cross=CorrelatorType.PH_DELAY_CROSS_CORRELATOR,
            )
        elif self.scan_type == "angular":
            correlator_option = SimpleNamespace(
                auto=CorrelatorType.PH_DELAY_CORRELATOR_LINES,
                cross=CorrelatorType.PH_DELAY_CROSS_CORRELATOR_LINES,
            )

        # create a dictionary with instantiated CorrFunc objects according to the needed 'xcorr_types'
        CF_dict = {
            xx: CorrFunc(
                f"{cf_name}_{xx} ({gate1_ns} vs. {gate2_ns} ns)"
                if cf_name is not None
                else f"{xx} ({gate1_ns} vs. {gate2_ns} ns)",
                correlator_option.auto if xx in {"AA", "BB"} else correlator_option.cross,
                self.laser_freq_hz,
            )
            for xx in xcorr_types
        }

        # cross/auto-correlate
        for xx in xcorr_types:
            CF = CF_dict[xx]
            CF.correlate_measurement(
                corr_input_dict[xx],
                afterpulse_params if afterpulse_params is not None else self.afterpulse_params,
                getattr(self, "bg_line_corr_list", []) if should_subtract_bg_corr else [],
                **kwargs,
            )

            try:  # temporal to spatial conversion, if scanning
                CF.vt_um = self.v_um_ms * CF.lag
            except AttributeError:
                CF.vt_um = CF.lag  # static

            if should_add_to_xcf_dict:
                # name the Corrfunc object
                # TODO: this should be eventually a list, not a dict (only first element and all together are ever interesting)
                self.xcf[CF.name] = CF

        if is_verbose:
            print("- Done.")

        return CF_dict

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

    def estimate_spatial_resolution(self, colors=None, **kwargs) -> Iterator[str]:
        """
        Perform Gaussian fits over 'normalized' vs. 'vt_um' fields of all correlation functions in the measurement
        in order to estimate the resolution improvement. This is relevant only for calibration experiments (i.e. 300 bp samples).
        Returns the legend labels for suse with higher-level Plotter instance in SolutionSFCSExperiment.
        """

        HWHM_FACTOR = np.sqrt(2 * np.log(2))

        for CF in self.cf.values():
            CF.fit_correlation_function(
                fit_name="zero_centered_zero_bg_normalized_gaussian_1d_fit",
            )

        with Plotter(
            super_title=f"Resolution Estimation\nGaussian fitting (HWHM) for '{self.type}' ACF(s)",
            xlim=(1e-2, 1),
            ylim=(5e-3, 1),
            **kwargs,
        ) as ax:
            # TODO: line below - this issue (line colors in hierarchical plotting) may be general and should be solved in Plotter class (?)
            colors = colors if colors is not None else cycle(default_colors)
            for CF, color in zip(self.cf.values(), colors):
                FP = CF.fit_params
                with suppress(KeyError):
                    kwargs.pop("parent_ax")
                hwhm = list(FP.beta.values())[0] * 1e3 * HWHM_FACTOR
                hwhm_error = list(FP.beta_error.values())[0] * 1e3 * HWHM_FACTOR
                fit_label = f"{CF.name}: ${hwhm:.0f}\\pm{hwhm_error:.0f}~nm$ ($\\chi^2={FP.chi_sq_norm:.0f}$)"
                FP.plot(parent_ax=ax, color=color, fit_label=fit_label, **kwargs)

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

    def calculate_hankel_transforms(
        self,
        interp_types=["gaussian"],
        is_verbose=False,
        **kwargs,
    ) -> None:
        """Doc."""

        if is_verbose:
            print(f"Calculating all Hankel transforms for '{self.type}' measurement...", end=" ")

        # calculate without plotting
        if not kwargs.get("should_plot"):
            for cf in self.cf.values():
                cf.calculate_hankel_transform(interp_types, is_verbose=False, **kwargs)

        # calculate and plot
        else:
            with Plotter(
                subplots=(len(interp_types), 2),
                super_title=f"{self.type.capitalize()}: Hankel Transforms",
                **kwargs,
            ) as axes:
                kwargs.pop("parent_ax", None)  # TODO: should this be included in Plotter init?
                for cf in self.cf.values():
                    cf.calculate_hankel_transform(interp_types, parent_ax=axes, **kwargs)

        if is_verbose:
            print("Done.")

    def calculate_structure_factors(
        self, cal_meas, interp_types=["gaussian"], parent_ax=None, is_verbose=False, **kwargs
    ):
        """
        Given a calibration SolutionSFCSMeasurement, i.e. one performed on a below-resolution sample
        (e.g. a 300 bp DNA sample labeled with the same fluorophore),
        this method divides this measurement's (self) Hankel transforms by the corresponding ones
        in the calibration measurement (all calculated if needed) and returns the sought structure factors.
        """

        if is_verbose:
            print(f"Calculating all structure factors for '{self.type}' measurement...", end=" ")

        # calculate without plotting
        if not kwargs.get("should_plot"):
            for CF, cal_CF in zip(self.cf.values(), cal_meas.cf.values()):
                CF.calculate_structure_factor(cal_CF, interp_types, is_verbose=False, **kwargs)

        # calculate and plot
        else:
            with Plotter(
                subplots=(len(interp_types), 1),
                super_title=f"{self.type.capitalize()}: Structure Factors",
                parent_ax=parent_ax,
                **kwargs,
            ) as axes:
                for CF, cal_CF in zip(self.cf.values(), cal_meas.cf.values()):
                    CF.calculate_structure_factor(
                        cal_CF, interp_types, parent_ax=axes, is_verbose=False, **kwargs
                    )
                for ax, interp_type in zip(axes if len(interp_types) > 1 else [axes], interp_types):
                    ax.set_title(f"{interp_type.capitalize()} Interp./Extrap.")

        if is_verbose:
            print("Done.")

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

    def save_processed(self, should_save_data=True, should_force=False, **kwargs) -> bool:
        """
        Save a processed measurement, including the '.data' attribute.
        The template may then be loaded much more quickly.
        """

        # save the measurement
        dir_path = (
            cast(Path, self.file_path_template).parent
            / "processed"
            / re.sub("_[*].pkl", "", self.template)
        )
        meas_file_path = dir_path / "SolutionSFCSMeasurement.blosc"
        if not meas_file_path.is_file() or should_force:
            # save the measurement object
            if kwargs.get("is_verbose"):
                print("Saving SolutionSFCSMeasurement object... ", end="")
            save_object(
                self, meas_file_path, compression_method="blosc", obj_name="processed measurement"
            )
            if kwargs.get("is_verbose"):
                print("Done.")

            # save the raw data separately
            if should_save_data:
                data_dir_path = dir_path / "data"

                # compress and save each data file in the temp folder in 'data_dir_path' (optional)
                for p in self.data:
                    if p.raw.compressed_file_path is None or (
                        not p.raw.compressed_file_path.exists() or should_force
                    ):
                        p.raw.save_compressed(data_dir_path)

            return True

        else:
            return False


class SolutionSFCSExperiment:
    """Doc."""

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
                    self.load_measurement(
                        meas_type=meas_type,
                        file_path_template=meas_template,
                        **meas_kwargs,
                        **kwargs,
                    )
                else:  # Use empty measuremnt by default
                    setattr(self, meas_type, SolutionSFCSMeasurement(meas_type))
            else:  # use supllied measurement
                setattr(self, meas_type, measurement)
                getattr(self, meas_type).name = meas_type  # remame supplied measurement

        if kwargs.get("should_plot"):
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
    ):
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
                dir_path = dir_path / "processed" / re.sub("_[*].pkl", "", file_template)
                measurement = load_processed_solution_measurement(
                    dir_path,
                    file_template,
                    should_load_data=should_re_correlate,
                )
                measurement.type = meas_type
                setattr(self, meas_type, measurement)
                print(f"Loaded pre-processed {meas_type} measurement from: '{dir_path}'")
            except OSError:
                print(
                    f"Pre-processed {meas_type} measurement not found at: '{dir_path}'. Processing data regularly."
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
            cf = measurement.correlate_and_average(
                is_verbose=True, afterpulsing_method=afterpulsing_method, **kwargs
            )

            if should_save:
                print(f"Saving {measurement.type} measurement to disk...", end=" ")
                if measurement.save_processed(**kwargs):
                    print("Done.")
                else:
                    print("Measurement already exists! (set 'should_force = True' to override.)")

        else:  # get existing first corrfunc
            cf = list(measurement.cf.values())[0]

        if kwargs.get("should_plot_meas"):

            if (x_field := kwargs.get("x_field")) is None:
                if measurement.scan_type == "static":
                    x_field = "lag"
                else:  # angular or circular scan
                    x_field = "vt_um"

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

    def renormalize_all(self, norm_range: Tuple[float, float], **kwargs):
        """Doc."""

        for meas_type in ("confocal", "sted"):
            for cf in getattr(self, meas_type).cf.values():
                cf.average_correlation(norm_range=norm_range, **kwargs)

    def save_processed_measurements(self, meas_types=["confocal", "sted"], **kwargs):
        """Doc."""

        print("Saving processed measurements to disk...", end=" ")
        if self.confocal.is_loaded and "confocal" in meas_types:
            if self.confocal.save_processed(**kwargs):
                print("Confocal saved...", end=" ")
            else:
                print(
                    "Not saving - processed measurement already exists (set 'should_force = True' to override.)",
                    end=" ",
                )
        if self.sted.is_loaded and "sted" in meas_types:
            if self.sted.save_processed(**kwargs):
                print("STED saved...", end=" ")
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
            self.confocal.calibrate_tdc(**kwargs)
            if self.sted.is_loaded:  # if both measurements quack as if loaded
                self.sted.calibrate_tdc(sync_coarse_time_to=self.confocal.tdc_calib, **kwargs)

        # calibrate STED only
        else:
            self.sted.calibrate_tdc(**kwargs)

        if should_plot:
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
        **kwargs,
    ) -> LifeTimeParams:
        """Doc."""

        conf = self.confocal
        sted = self.sted

        conf_hist = conf.tdc_calib.all_hist_norm
        conf_t = conf.tdc_calib.t_hist
        conf_t = conf_t[np.isfinite(conf_hist)]
        conf_hist = conf_hist[np.isfinite(conf_hist)]

        sted_hist = sted.tdc_calib.all_hist_norm
        sted_t = sted.tdc_calib.t_hist
        sted_t = sted_t[np.isfinite(sted_hist)]
        sted_hist = sted_hist[np.isfinite(sted_hist)]

        h_max, j_max = conf_hist.max(), conf_hist.argmax()
        t_max = conf_t[j_max]

        beta0 = (h_max, 4, h_max * 1e-3)

        if fit_range is None:
            fit_range = Limits(t_max, 40)

        if param_estimates is None:
            param_estimates = beta0

        conf_params = conf.tdc_calib.fit_lifetime_hist(
            fit_range=fit_range, fit_param_estimate=beta0
        )

        # remove background
        sted_bg = np.mean(sted_hist[Limits(bg_range).valid_indices(sted_t)])
        conf_bg = np.mean(conf_hist[Limits(bg_range).valid_indices(conf_t)])
        sted_hist = sted_hist - sted_bg
        conf_hist = conf_hist - conf_bg

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
            linear_range = Limits()
            with Plotter(
                parent_ax=ax, super_title=title, selection_limits=linear_range, **kwargs
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

                lifetime_ns = conf_params.beta["tau"]
                sigma_sted = p1 * lifetime_ns
                try:
                    laser_pulse_delay_ns = (1 - p0) / p1
                except RuntimeWarning:
                    laser_pulse_delay_ns = None

            elif sted_field == "paraboloid":
                fit_params = curve_fit_lims(
                    FIT_NAME_DICT["ratio_of_lifetime_histograms_fit"],
                    param_estimates=(2, 1, 1),
                    xs=t[j_selected],
                    ys=hist_ratio[j_selected],
                    ys_errors=np.ones(j_selected.sum()),
                    should_plot=True,
                )

                lifetime_ns = conf_params.beta["tau"]
                sigma_sted = (
                    fit_params.beta["sigma_x"] * lifetime_ns,
                    fit_params.beta["sigma_y"] * lifetime_ns,
                )
                laser_pulse_delay_ns = fit_params.beta["t0"]

        self.lifetime_params = LifeTimeParams(lifetime_ns, sigma_sted, laser_pulse_delay_ns)
        return self.lifetime_params

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
        tdc_gate_ns: Tuple[float, float],
        meas_type: str,
        should_re_correlate=False,
        **kwargs,
    ) -> None:
        """Doc."""

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
            logging.info(
                "Cannot add STED gate if there's no STED measurement loaded to the experiment!"
            )
            return

        if kwargs.get("should_plot"):
            self.plot_standard(**kwargs)

    def add_gates(self, gate_list: List[Tuple[float, float]], meas_type="sted", **kwargs):
        """
        A convecience method for adding multiple gates.
        """

        print(f"Adding multiple '{meas_type}' gates {gate_list} for experiment '{self.name}'...")
        for tdc_gate_ns in gate_list:
            self.add_gate(tdc_gate_ns, meas_type, **kwargs)

        if kwargs.get("should_plot"):
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

    def plot_correlation_functions(
        self,
        xlim=(5e-5, 1),
        ylim=None,
        x_field=None,
        y_field=None,
        x_scale=None,
        y_scale=None,
        should_add_exp_name=True,
        **kwargs,
    ):
        """Doc."""

        if self.confocal.is_loaded:
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

        # auto y_field/y_scale determination
        if y_field is None:
            y_field = "normalized"
            if y_scale is None:
                if x_field == "vt_um":
                    y_scale = "log"
                else:
                    y_scale = "linear"

        # auto ylim determination
        if ylim is None:
            if y_field == "normalized":
                if y_scale == "log":
                    ylim = (1e-3, 1)
                else:
                    ylim = (0, 1)
            elif y_field in {"average_all_cf_cr", "avg_cf_cr"}:
                # TODO: perhaps cf attricute should be a list and not a dict? all I'm really ever interested in is either showing the first or all together (names are in each CF anyway)
                first_cf = list(ref_meas.cf.values())[0]
                ylim = Limits(-1e3, first_cf.g0 * 1.2)

        with Plotter(
            super_title=f"'{self.name}' Experiment - All ACFs",
            **kwargs,
        ) as ax:

            if (parent_ax := kwargs.get("parent_ax")) is not None:
                existing_lines = parent_ax.get_lines()
                kwargs.pop("parent_ax")

            for meas_type in ("confocal", "sted"):
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

            # add experiment name to labels if plotted hierarchically (multiple experiments)
            # TODO: this could be perhaps a feature of Plotter? i.e., an addition to all labels can be passed at Plotter init?
            if parent_ax is not None:
                for line in ax.get_lines():
                    if line not in existing_lines:
                        label = line.get_label()
                        if "_" not in label:
                            line.set_label(
                                f"{self.name}: {label}" if should_add_exp_name else label
                            )

            ax.legend()

    def estimate_spatial_resolution(self, colors=None, **kwargs) -> Iterator[str]:
        """
        High-level method for performing Gaussian fits over 'normalized' vs. 'vt_um' fields of all correlation functions
        (confocal, sted and any gates) in order to estimate the resolution improvement.
        This is relevant only for calibration experiments (i.e. 300 bp samples).
        """

        with Plotter(xlim=(1e-2, 1), ylim=(0, 1), **kwargs) as ax:

            if (parent_ax := kwargs.pop("parent_ax", None)) is not None:
                # TODO: this could be perhaps a feature of Plotter? i.e., an addition to all labels can be passed at Plotter init?
                existing_lines = parent_ax.get_lines()

            colors = colors if colors is not None else cycle(default_colors)
            remaining_colors = self.confocal.estimate_spatial_resolution(
                parent_ax=ax, colors=colors, **kwargs
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

        for meas_type in ("confocal", "sted"):
            with suppress(AttributeError):
                for cf in getattr(self, meas_type).cf.values():
                    cf.afterpulsing_filter.plot(
                        super_title=f"Afterpulsing Filter\n{meas_type.capitalize()}: {cf.name}",
                        **kwargs,
                    )

    def calculate_hankel_transforms(
        self, interp_types=["gaussian"], is_verbose=False, **kwargs
    ) -> None:
        """Doc."""

        if is_verbose:
            print(f"Calculating all Hankel transforms for '{self.name}' experiment...", end=" ")

        # calculated without plotting
        if not kwargs.get("should_plot"):
            for meas_type in ("confocal", "sted"):
                getattr(self, meas_type).calculate_hankel_transforms(
                    interp_types, is_verbose=False, **kwargs
                )

        # plot all transforms of all corrfuncs of all measurements in a single figure
        else:
            with Plotter(
                subplots=(len(interp_types), 2),
                super_title=f"Experiment '{self.name}': Hankel Transforms",
                **kwargs,
            ) as axes:
                for meas_type in ("confocal", "sted"):
                    getattr(self, meas_type).calculate_hankel_transforms(
                        interp_types, parent_ax=axes, is_verbose=False, **kwargs
                    )

        if is_verbose:
            print("Done.")

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
            print(f"Calculating all structure factors for '{self.name}' experiment...", end=" ")

        # calculated without plotting
        if not should_plot:
            for meas_type in ("confocal", "sted"):
                cal_meas = getattr(cal_exp, meas_type)
                getattr(self, meas_type).calculate_structure_factors(
                    cal_meas, interp_types, should_plot=False, is_verbose=False, **kwargs
                )

        # calculate and plot
        else:
            # Hankel transforms are performed first in this (plotting) case, so they can be shown
            cal_exp.calculate_hankel_transforms(
                interp_types, should_plot=True, is_verbose=False, **kwargs
            )
            self.calculate_hankel_transforms(
                interp_types, should_plot=True, is_verbose=False, **kwargs
            )
            # now getting the Sq and plotting it as well (should be fast since transforms exist in CorrFuncs already)
            with Plotter(
                subplots=(len(interp_types), 1),
                super_title=f"Experiment '{self.name}': Structure factors",
                **kwargs,
            ) as axes:
                kwargs.pop("parent_ax", None)
                for meas_type in ("confocal", "sted"):
                    cal_meas = getattr(cal_exp, meas_type)
                    getattr(self, meas_type).calculate_structure_factors(
                        cal_meas,
                        interp_types,
                        should_plot=True,
                        is_verbose=False,
                        parent_ax=axes,
                        **kwargs,
                    )
                for ax, interp_type in zip(axes if len(interp_types) > 1 else [axes], interp_types):
                    ax.set_title(f"{interp_type.capitalize()} Interp./Extrap.")

        if is_verbose:
            print("Done.")

    def fit_structure_factors(self, model: str):
        """Doc."""
        # TODO:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.dawsn.html

        raise NotImplementedError


class ImageSFCSMeasurement(CountsImageMixin):
    """Doc."""

    def __init__(self):
        pass

    def read_image_data(self, file_path, **kwargs) -> None:
        """Doc."""

        file_dict = load_file_dict(file_path)
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


def calculate_calibrated_afterpulse(
    lag: np.ndarray,
    afterpulse_params: tuple = default_system_info["afterpulse_params"],
    gate_ns: Tuple[float, float] = (0, np.inf),
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
