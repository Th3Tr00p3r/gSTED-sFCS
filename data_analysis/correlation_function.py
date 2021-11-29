"""Data organization and manipulation."""

import logging
import re
from collections import deque, namedtuple
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
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
from utilities.fit_tools import FitParams, curve_fit_lims
from utilities.helper import Limits, div_ceil


@dataclass
class StructureFactor:
    """Holds structure factor data"""

    w_xy: np.ndarray
    w_sq: np.ndarray
    n_interp_pnts: int
    r_max: float
    vt_min_sq: float
    g_min: float

    r: np.ndarray
    fr: np.ndarray
    fr_linear_interp: np.ndarray

    q: np.ndarray
    qz: np.ndarray
    fq: np.ndarray
    fq_linear_interp: np.ndarray
    fq_cor: np.ndarray
    fq_linear_interp_cor: np.ndarray
    fq_error: np.ndarray

    sq: np.ndarray
    sq_linear_interp: np.ndarray
    sq_cor: np.ndarray
    sq_linear_interp_cor: np.ndarray
    sq_error: np.ndarray


class CorrFunc:
    """Doc."""

    fit_params: FitParams
    run_duration: float
    total_duration: float
    afterpulse: np.ndarray
    vt_um: np.ndarray
    cf_cr: np.ndarray
    g0: float

    def __init__(self, gate_ns):
        self.gate_ns = Limits(gate_ns)
        self.lag = []
        self.countrate_list = []
        self.skipped_duration = 0

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
        self.score = (
            (1 / np.var(self.cf_cr[:, jj], 0))
            * (self.cf_cr[:, jj] - self.median_all_cf_cr[jj]) ** 2
            / len(jj)
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

        self.avg_cf_cr, self.error_cf_cr = _calculate_weighted_avg(
            self.cf_cr[self.j_good, :], self.weights[self.j_good, :]
        )

        j_t = self.norm_range.valid_indices(self.lag)
        self.g0 = (self.avg_cf_cr[j_t] / self.error_cf_cr[j_t] ** 2).sum() / (
            1 / self.error_cf_cr[j_t] ** 2
        ).sum()
        self.normalized = self.avg_cf_cr / self.g0
        self.error_normalized = self.error_cf_cr / self.g0

        if should_plot:
            self.plot_correlation_function(plot_kwargs=plot_kwargs)

    def plot_correlation_function(
        self,
        parent_ax=None,
        x_field="lag",
        y_field="avg_cf_cr",
        x_scale="log",
        y_scale="linear",
        xlim=None,
        ylim=None,
        plot_kwargs={},
        **kwargs,
    ):

        x = getattr(self, x_field)
        y = getattr(self, y_field)
        if x_scale == "log":  # remove zero point data
            x, y = x[1:], y[1:]

        with Plotter(
            parent_ax=parent_ax,
            xlim=xlim,
            ylim=ylim,
            xscale=x_scale,
            y_scale=y_scale,
            should_autoscale=True,
        ) as ax:
            ax.set_xlabel(x_field)
            ax.set_ylabel(y_field)
            ax.plot(x, y, "-", **plot_kwargs)

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

        if fit_param_estimate is None:
            fit_param_estimate = [self.g0, 0.035, 30.0]

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
        )

    def calculate_structure_factor(
        self,
        w_xy=np.array([0]),  # used in Gaussian field approx.
        w_sq=1,  # used in Gaussian field approx.
        n_interp_pnts=512,
        r_max=10,
        r_min=0.05,
        g_min=1e-2,
        baseline_range_um=None,
        interp_pnts=2,
        plot_kwargs={},
        **kwargs,
    ) -> StructureFactor:
        """Doc."""

        vt_min_sq = r_min ** 2

        c0 = scipy.special.jn_zeros(0, n_interp_pnts)  # Bessel function zeros
        r = (r_max / c0[n_interp_pnts - 1]) * c0  # Radius vector
        #        print("r:",  r) # TESTESTEST

        if baseline_range_um is not None:
            jj = Limits(baseline_range_um).valid_indxs(self.vt_um)
            baseline = np.mean(self.normalized[jj])
            self.normalized = (self.normalized - baseline) / (1 - baseline)

        j3 = np.nonzero(self.vt_um ** 2 > vt_min_sq)[0]
        temp_vt_um = self.vt_um[j3]
        temp_normalized = self.normalized[j3]

        k = (
            min(np.nonzero(temp_normalized < g_min)[0]) - 1
        )  # point before the first crossing of g_min
        if k.size == 0:
            k = temp_normalized.size

        j2 = np.arange(k)

        #        print("r ** 2:", r ** 2) # TESTESTEST
        #        print("temp_vt_um[j2]:", temp_vt_um[j2]) # TESTESTEST
        #        print("temp_vt_um[j2] ** 2:", temp_vt_um[j2] ** 2) # TESTESTEST
        #        print("np.log(temp_normalized[j2]):", np.log(temp_normalized[j2])) # TESTESTEST

        #  Gaussian interpolation
        if interp_pnts is not None:
            print(
                "robust_interpolation:",
                self.robust_interpolation(
                    r ** 2, temp_vt_um[j2] ** 2, np.log(temp_normalized[j2]), interp_pnts
                ),
            )  # TESTESTEST
            print(
                "max robust_interpolation:",
                self.robust_interpolation(
                    r ** 2, temp_vt_um[j2] ** 2, np.log(temp_normalized[j2]), interp_pnts
                ).max(),
            )  # TESTESTEST
            print(
                "robust_interpolation type:",
                (
                    self.robust_interpolation(
                        r ** 2, temp_vt_um[j2] ** 2, np.log(temp_normalized[j2]), interp_pnts
                    )
                ).dtype,
            )  # TESTESTEST
            with np.errstate(divide="ignore", invalid="ignore"):
                fr = np.exp(
                    self.robust_interpolation(
                        r ** 2, temp_vt_um[j2] ** 2, np.log(temp_normalized[j2]), interp_pnts
                    )
                )
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_temp_normalized_j2 = np.log(temp_normalized[j2])
                log_temp_normalized_j2[np.isnan(log_temp_normalized_j2)] = 0
                print("log_temp_normalized_j2:", log_temp_normalized_j2)  # TESTESTEST
                print("temp_vt_um[j2] ** 2:", temp_vt_um[j2] ** 2)  # TESTESTEST
                print("r ** 2:", r ** 2)  # TESTESTEST
                interpolator = scipy.interpolate.interp1d(
                    temp_vt_um[j2] ** 2, log_temp_normalized_j2, fill_value="extrapolate"
                )
            print("interpolator(r ** 2):", interpolator(r ** 2))  # TESTESTEST
            print("max interpolator(r ** 2):", interpolator(r ** 2).max())  # TESTESTEST
            fr = np.exp(interpolator(r ** 2))

        fr = np.real(fr)  # TODO: is this needed?

        #  linear interpolation
        if interp_pnts is not None:
            fr_linear_interp = self.robust_interpolation(
                r, temp_vt_um[j2], temp_normalized[j2], interp_pnts
            )
        else:
            interpolator = scipy.interpolate.interp1d(
                temp_vt_um[j2], temp_normalized[j2], fill_value="extrapolate"
            )
            fr_linear_interp = np.exp(interpolator(r))

        #  zero-pad
        fr_linear_interp[r > temp_vt_um[k]] = 0

        # Fourier Transform
        q, fq, *_ = self.hankel_transform(r, fr)

        #  linear interpolated Fourier Transform
        _, fq_linear_interp, *_ = self.hankel_transform(r, fr_linear_interp)

        sq = np.exp(q ** 2 * w_xy ** 2 / 4) * np.tile(fq, (1, len(w_xy)))
        sq_linear_interp = np.exp(q ** 2 * w_xy ** 2 / 4) * np.tile(
            fq_linear_interp, (1, len(w_xy))
        )

        # One step  correction  for finite aspect ratio
        w_xy_sq = w_xy ** 2
        dqz = np.median(np.diff(q)) / np.sqrt(w_sq)
        qz = np.arange(0, max(q) / np.sqrt(w_sq), dqz)
        q_mesh, qz_mesh = np.meshgrid(q, qz)
        q_vec = np.sqrt(q_mesh ** 2 + qz_mesh ** 2)

        fq_cor = np.empty((fq.shape[0], len(w_xy_sq)))
        sq_cor = np.empty((fq.shape[0], len(w_xy_sq)))
        fq_linear_interp_cor = np.empty((fq_linear_interp.shape[0], len(w_xy_sq)))
        sq_linear_interp_cor = np.empty((fq_linear_interp.shape[0], len(w_xy_sq)))
        for idx in range(len(w_xy_sq)):
            norm_fz = sum(np.exp(-w_xy_sq[idx] * qz ** 2 * w_sq / 4))
            kern_fz = np.exp(-w_xy_sq[idx] * qz_mesh ** 2 * (w_sq - 1) / 4) / norm_fz

            # zero-pad fq above first zero
            #            print("q_vec shape:", q_vec.shape) # TESTESTEST
            #            print("q shape:", q.shape) # TESTESTEST
            #            print("fq shape:", fq.shape) # TESTESTEST
            d_gq_2d_to_3d = np.interp(q_vec.ravel(), q, fq)
            d_gq_2d_to_3d = np.reshape(d_gq_2d_to_3d, q_vec.shape, order="F")

            d_gq = fq - (d_gq_2d_to_3d * kern_fz).sum().T
            fq_cor[:, idx] = fq + d_gq
            sq_cor[:, idx] = np.exp(q ** 2 * w_xy_sq[idx] / 4) * fq_cor[:, idx]

            d_gq_2d_to_3d = np.interp(q_vec.ravel(), q, fq_linear_interp)
            d_gq_2d_to_3d = np.reshape(d_gq_2d_to_3d, q_vec.shape, order="F")
            d_gq = fq_linear_interp - (d_gq_2d_to_3d * kern_fz).sum().T
            fq_linear_interp_cor[:, idx] = fq_linear_interp + d_gq

            sq_linear_interp_cor[:, idx] = (
                np.exp(q ** 2 * w_xy_sq[idx] / 4) * fq_linear_interp_cor[:, idx]
            )

        #  Estimating error of Fourier transform
        fq_allfunc = np.zeros(shape=(q.size, len(self.j_good)))
        print("Estimating structure factor errors...", end=" ")

        for idx, j in enumerate(self.j_good):
            if interp_pnts is not None:
                print("cf_cr.shape:", self.cf_cr.shape)  # TESTESTEST
                print("self.cf_cr[j, :]:", self.cf_cr[j, :])  # TESTESTEST
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_cf_cr_j = np.log(self.cf_cr[j, :])
                    print("log_cf_cr_j:", log_cf_cr_j)  # TESTESTEST
                    rbst_interp = self.robust_interpolation(
                        r ** 2, self.vt_um[j2] ** 2, log_cf_cr_j, interp_pnts
                    )
                    print("rbst_interp:", rbst_interp)  # TESTESTEST
                    fr = np.exp(rbst_interp)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    fr = np.exp(
                        np.interp(r ** 2, self.vt_um ** 2, np.log(self.cf_cr[j, :]), right=0)
                    )

            fr = np.real(fr)
            fr[np.nonzero(fr < 0)[0]] = 0
            _, fq_allfunc[:, idx], *_ = self.hankel_transform(r, fr)

        fq_error = np.std(fq_allfunc, axis=1, ddof=1) / np.sqrt(len(self.j_good)) / self.g0
        sq_error = np.exp(q ** 2 * w_xy ** 2 / 4) * np.tile(fq_error, (1, len(w_xy)))

        s = StructureFactor(
            w_xy,
            w_sq,
            n_interp_pnts,
            r_max,
            vt_min_sq,
            g_min,
            r,
            fr,
            fr_linear_interp,
            q,
            qz,
            fq,
            fq_linear_interp,
            fq_cor,
            fq_linear_interp_cor,
            fq_error,
            sq,
            sq_linear_interp,
            sq_cor,
            sq_linear_interp_cor,
            sq_error,
        )

        #  plotting
        with Plotter(
            subplot=(1, 3),
            xlim=Limits(q[1], np.pi / min(w_xy)),
            xscale="log",
            yscale="log",
            super_title="Structure Factor: $S(q)$",
        ) as axes:

            axes[0].set_title("Gaussian vs. linear\ninterpolation")
            axes[0].plot(q, [sq, sq_linear_interp], **plot_kwargs)
            legend_labels_1, legend_labels_2 = [], []
            for elem in w_xy:
                legend_labels_1.append(f"Gaussian w_xy = {elem}")
                legend_labels_2.append(f"Linear w_xy = {elem}")
            axes[0].legend(legend_labels_1 + legend_labels_2)

            axes[1].set_title("Corrected for 3D vs. uncorrected\nGaussian interpolation")
            axes[1].plot(q, [sq, sq_cor])
            legend_labels_1, legend_labels_2 = [], []
            for elem in w_xy:
                legend_labels_1.append(f"Uncorrected w_xy = {elem}")
                legend_labels_2.append(f"Corrected w_xy = {elem}")
            axes[1].legend(legend_labels_1 + legend_labels_2)

            axes[2].set_title("Corrected for 3D vs. uncorrected\nLinear interpolation")
            axes[2].plot(q, [sq_linear_interp, sq_linear_interp_cor])
            legend_labels_1, legend_labels_2 = [], []
            for elem in w_xy:
                legend_labels_1.append(f"Uncorrected w_xy = {elem}")
                legend_labels_2.append(f"Corrected w_xy = {elem}")
            axes[2].legend(legend_labels_1 + legend_labels_2)

        return s

    def robust_interpolation(
        self,
        x,
        xi,
        yi,
        n_interp_pnts,
    ):
        """Doc."""

        y = np.empty(shape=x.shape)

        xi = xi.ravel().T
        change_my_name = [-np.inf] + xi.tolist() + [np.inf]
        (h, _), bin = np.histogram(x, np.array(change_my_name)), np.digitize(
            x, np.array(change_my_name)
        )  # translated from: [h, bin] = histc(x, np.array([-np.inf, xi, inf]))
        ch = np.cumsum(h)
        st = max(bin[0] - 1, n_interp_pnts)
        fin = min(bin[x.size - 1] - 1, xi.size - n_interp_pnts)

        D = SimpleNamespace(x=[], slope=[])
        i = st
        for i in range(st, min(bin[x.size - 1] - 1, xi.size - n_interp_pnts)):
            ji = np.arange((i - n_interp_pnts + 1), (i + n_interp_pnts))
            # Robustly fit linear model with RANSAC algorithm
            ransac = linear_model.RANSACRegressor()
            ransac.fit(xi[ji][:, np.newaxis], yi[ji])
            p0, p1 = ransac.estimator_.intercept_, ransac.estimator_.coef_[0]

            fin = min(bin[x.size - 1] - 1, xi.size - n_interp_pnts)

            if i == st:
                j = np.arange(ch[i + 1])
            elif i == fin:
                j = np.arange((ch[i] + 1), x.size)
            else:
                j = np.arange((ch[i] + 1), ch[i + 1])

            y[j] = p0 + p1 * x[j]  # TODO: y shape? need to allocate first
            D.x.append(xi[i])
            D.slope.append(p1)

        return y  # , D

    def hankel_transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        should_inverse: bool = False,
        should_do_robust_interpolation: bool = False,
        should_do_gaussian_interpolation: bool = False,
        n_interp_pnts: int = 2,
        dr=None,
    ):
        """Doc."""

        x = x.ravel()
        y = y.ravel()
        n = len(x)

        c0 = scipy.special.jn_zeros(0, n)
        bessel_j0 = scipy.special.j0
        bessel_j1 = scipy.special.j1

        if not should_inverse:

            HankelTransformation = namedtuple(
                "HankelTransformation",
                "trans_q trans_fq interp_r interp_fr",
            )

            r_max = max(x)
            q_max = c0[n - 1] / (2 * np.pi * r_max)  # Maximum frequency

            r = c0.T * r_max / c0[n - 1]  # Radius vector
            q = c0.T / (2 * np.pi * r_max)  # Frequency vector

            j_n, j_m = np.meshgrid(c0, c0)

            C = (
                (2 / c0[n - 1])
                * bessel_j0(j_n * j_m / c0[n - 1])
                / (abs(bessel_j1(j_n)) * abs(bessel_j1(j_m)))
            )
            # C is the transformation matrix

            m1 = (abs(bessel_j1(c0)) / r_max).T  # m1 prepares input vector for transformation
            m2 = m1 * r_max / q_max  # m2 prepares output vector for display

            # end preparations for Hankel transform

            if should_do_robust_interpolation:
                if should_do_gaussian_interpolation:
                    y_interp = np.exp(
                        self.robust_interpolation(r ** 2, x ** 2, np.log(y), n_interp_pnts)
                    )
                    y_interp[
                        r > x[-n_interp_pnts]
                    ] = 0  # zero pad last points that do not have full interpolation
                else:  # linear
                    y_interp = self.robust_interpolation(r, x, y, n_interp_pnts)

                y_interp = y_interp.ravel()

            else:
                if should_do_gaussian_interpolation:
                    interpolator = scipy.interpolate.interp1d(
                        x ** 2, np.log(y), fill_value="extrapolate"
                    )
                    y_interp = np.exp(interpolator(r ** 2))
                else:  # linear
                    interpolator = scipy.interpolate.interp1d(x, y, fill_value="extrapolate")
                    y_interp = interpolator(r)

            print("C shape", C.shape)  # TESTESTEST
            print("y_interp shape", y_interp.shape)  # TESTESTEST
            print("m1 shape", m1.shape)  # TESTESTEST
            print("m2 shape", m2.shape)  # TESTESTEST

            return HankelTransformation(
                trans_q=2 * np.pi * q,
                trans_fq=C @ (y_interp / m1) * m2,
                interp_r=r,
                interp_fr=y_interp,
            )

        else:  # inverse transform

            InverseHankelTransformation = namedtuple(
                "InverseHankelTransformation",
                "trans_r trans_fr",
            )

            if dr is not None:
                q_max = 1 / (2 * dr)
            else:
                q_max = max(x) / (2 * np.pi)

            r_max = c0[n - 1] / (2 * np.pi * q_max)  # Maximum radius

            r = c0.T * r_max / c0[n - 1]  # Radius vector
            q = c0.T / (2 * np.pi * r_max)  # Frequency vector
            q = 2 * np.pi * q

            j_n, j_m = np.meshgrid(c0, c0)

            C = (
                (2 / c0[n - 1])
                * bessel_j0(j_n * j_m / c0[n - 1])
                / (abs(bessel_j1(j_n)) * abs(bessel_j1(j_m)))
            )
            # C is the transformation matrix

            m1 = (abs(bessel_j1(c0)) / r_max).T  # m1 prepares input vector for transformation
            m2 = m1 * r_max / q_max  # m2 prepares output vector for display
            # end preparations for Hankel transform

            interpolator = scipy.interpolate.interp1d(x, y, fill_value="extrapolate")

            return InverseHankelTransformation(
                trans_r=r,
                trans_fr=C @ (interpolator(q) / m2) * m1,
            )


@dataclass
class CorrFuncAccumulator:
    """
    A convecience class for accumulating the outputs of SoftwareCorrelator
    for a CorrFunc during correlation of measurement data.
    """

    cf: CorrFunc
    corrfunc_list: list[np.ndarray] = field(default_factory=list)
    weights_list: list[np.ndarray] = field(default_factory=list)
    cf_cr_list: list[np.ndarray] = field(default_factory=list)
    n_corrfuncs: int = 0

    def __enter__(self):
        """Initiate a temporary structure for accumulating SoftwareCorrelator outputs."""

        return self

    def __exit__(self, *exc):
        """Create padded 2D ndarrays from the accumulated lists and delete the accumulator."""

        lag_len = len(self.cf.lag)
        self.cf.corrfunc, self.cf.weights, self.cf.cf_cr = self.join_and_pad(lag_len)

    def accumulate(self, SC: SoftwareCorrelator):
        """Doc."""

        self.corrfunc_list.append(SC.corrfunc)
        self.weights_list.append(SC.weights)
        self.cf_cr_list.append(SC.cf_cr)
        self.n_corrfuncs += 1

    def join_and_pad(self, lag_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Doc."""

        shape = (self.n_corrfuncs, lag_len)
        corrfunc = np.empty(shape=shape, dtype=np.float64)
        weights = np.empty(shape=shape, dtype=np.float64)
        cf_cr = np.empty(shape=shape, dtype=np.float64)
        for idx in range(self.n_corrfuncs):
            pad_len = lag_len - len(self.corrfunc_list[idx])
            corrfunc[idx] = np.pad(self.corrfunc_list[idx], (0, pad_len))
            weights[idx] = np.pad(self.weights_list[idx], (0, pad_len))
            cf_cr[idx] = np.pad(self.cf_cr_list[idx], (0, pad_len))

        return corrfunc, weights, cf_cr


class SolutionSFCSMeasurement(TDCPhotonDataMixin):
    """Doc."""

    NAN_PLACEBO = -100  # TODO: belongs in 'AngularScanMixin'
    DUMP_PATH = Path("C:/temp_sfcs_data/")
    SIZE_LIMITS_MB = Limits(10, 1e4)

    def __init__(self, name=""):
        self.name = name
        self.data = []  # list to hold the data of each file
        self.cf = dict()
        self.is_data_dumped = False
        self.scan_type: str
        self.duration_min: float = None

    @file_utilities.rotate_data_to_disk
    def read_fpga_data(
        self,
        file_path_template: Union[str, Path],
        file_selection: str = None,
        should_plot=False,
        **kwargs,
    ) -> None:
        """Processes a complete FCS measurement (multiple files)."""

        print("\nLoading FPGA data from hard drive:", end=" ")

        file_paths = file_utilities.prepare_file_paths(Path(file_path_template), file_selection)
        n_files = len(file_paths)
        *_, self.template = Path(file_path_template).parts
        self.name_on_disk = re.sub("\\*", "", re.sub("_[*]", "", self.template))

        for idx, file_path in enumerate(file_paths):
            # Loading file from disk
            print(f"Loading file No. {idx+1}/{n_files}: '{file_path}'...", end=" ")
            try:
                file_dict = file_utilities.load_file_dict(file_path)
            except OSError:
                print(f"File '{idx+1}' has not been fully downloaded from cloud. Skipping...\n")
                continue
            else:
                print("Done.")

            # Processing data
            p = self.process_data(file_dict, idx, verbose=True, **kwargs)

            # Appending data to self
            if p is not None:
                p.file_path = file_path
                self.data.append(p)
                print(f"Finished processing file No. {idx+1}\n")

        # count the files and ensure there's at least one file
        self.n_files = len(self.data)
        if not self.n_files:
            raise RuntimeError(
                f"Loading FPGA data catastrophically failed ({n_files}/{n_files} files skipped)."
            )

        if self.scan_type == "angular_scan":
            # aggregate images and ROIs for angular sFCS
            self.scan_images_dstack = np.dstack(tuple(p.image for p in self.data))
            self.roi_list = [p.roi for p in self.data]

        # calculate average count rate
        self.avg_cnt_rate_khz = sum([p.avg_cnt_rate_khz for p in self.data]) / len(self.data)

        if self.duration_min is None:
            # calculate duration if not supplied
            self.duration_min = (
                np.mean([np.diff(p.runtime).sum() for p in self.data]) / self.laser_freq_hz / 60
            )
            print(f"Calculating duration (not supplied): {self.duration_min:.1f} min\n")

        print(f"Finished loading FPGA data ({len(self.data)}/{n_files} files used).\n")

        # plotting of scan image and ROI
        if should_plot:
            print("Displaying scan images...", end=" ")
            with Plotter(subplots=(1, n_files), fontsize=8, should_force_aspect=True) as axes:
                if not hasattr(
                    axes, "size"
                ):  # if axes is not an ndarray (only happens if reding just one file)
                    axes = np.array([axes])
                for idx, (ax, image, roi) in enumerate(
                    zip(axes, np.moveaxis(self.scan_images_dstack, -1, 0), self.roi_list)
                ):
                    ax.set_title(f"file #{idx+1} of\n'{self.name}' measurement")
                    ax.set_xlabel("Pixel Number")
                    ax.set_ylabel("Line Number")
                    ax.imshow(image)
                    ax.plot(roi["col"], roi["row"], color="white")
            print("Done.\n")

    def process_data(self, file_dict: dict, idx: int = 0, **kwargs) -> TDCPhotonData:
        """Doc."""

        full_data = file_dict["full_data"]

        if idx == 0:
            self.after_pulse_param = file_dict["system_info"]["after_pulse_param"]
            self.laser_freq_hz = int(full_data["laser_freq_mhz"] * 1e6)
            self.fpga_freq_hz = int(full_data["fpga_freq_mhz"] * 1e6)
            with suppress(KeyError):
                self.duration_min = full_data["duration_s"] / 60

        # Circular sFCS
        if full_data.get("circle_speed_um_s"):
            self.scan_type = "circular_scan"
            self.v_um_ms = full_data["circle_speed_um_s"] * 1e-3  # to um/ms
            raise NotImplementedError("Circular scan analysis not yet implemented...")

        # Angular sFCS
        elif full_data.get("angular_scan_settings"):
            if idx == 0:
                self.scan_type = "angular_scan"
                self.angular_scan_settings = full_data["angular_scan_settings"]
                self.LINE_END_ADDER = 1000
            return self.process_angular_scan_data(full_data, idx, **kwargs)

        # FCS
        else:
            self.scan_type = "static"
            return self.process_static_data(full_data, idx, **kwargs)

    def process_static_data(self, full_data, idx, **kwargs) -> TDCPhotonData:
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

        p.file_num = idx + 1
        p.avg_cnt_rate_khz = full_data["avg_cnt_rate_khz"]

        return p

    def process_angular_scan_data(
        self,
        full_data,
        idx,
        should_fix_shift=True,
        roi_selection="auto",
        **kwargs,
    ) -> TDCPhotonData:
        """
        Processes a single angular sFCS data file ('full_data').
        Returns the processed results as a 'TDCPhotonData' object.
        '"""

        p = self.convert_fpga_data_to_photons(
            full_data["byte_data"], version=full_data["version"], verbose=True
        )

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
        cnt, n_pix_tot, n_pix, line_num = _convert_angular_scan_to_image(
            runtime, self.laser_freq_hz, sample_freq_hz, ppl_tot, n_lines
        )

        if should_fix_shift:
            pix_shift = _fix_data_shift(cnt)
            runtime = p.runtime + pix_shift * round(self.laser_freq_hz / sample_freq_hz)
            cnt, n_pix_tot, n_pix, line_num = _convert_angular_scan_to_image(
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
                bw = _threshold_and_smooth(cnt)
            except ValueError:
                print("Thresholding failed, skipping file.")
                return None
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

        runtime_line_starts: np.ndarray = line_starts * round(self.laser_freq_hz / sample_freq_hz)
        runtime_line_stops: np.ndarray = line_stops * round(self.laser_freq_hz / sample_freq_hz)

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

        # reverse rows again
        bw[1::2, :] = np.flip(bw[1::2, :], 1)
        p.bw_mask = bw

        # get image line correlation to subtract trends
        img = p.image * p.bw_mask

        p.image_line_corr = _line_correlations(img, p.bw_mask, roi, sample_freq_hz)

        return p

    def correlate_and_average(self, **kwargs):
        """High level function for correlating and averaging any data."""

        CF = self.correlate_data(**kwargs)
        CF.average_correlation(**kwargs)

    @file_utilities.rotate_data_to_disk
    def correlate_data(self, cf_name=None, **kwargs):
        """
        High level function for correlating any type of data (e.g. static, angular scan...)
        Returns a 'CorrFunc' object.
        Data attribute is possibly rotated from/to disk.
        """

        if self.scan_type == "angular_scan":
            CF = self.correlate_angular_scan_data(**kwargs)
        elif self.scan_type == "circular_scan":
            CF = self.correlate_circular_scan_data(**kwargs)
        elif self.scan_type == "static":
            CF = self.correlate_static_data(**kwargs)
        else:
            raise NotImplementedError(
                f"Correlating data of type '{self.scan_type}' is not implemented."
            )

        if cf_name is not None:
            self.cf[cf_name] = CF
        else:
            self.cf[f"gSTED {CF.gate_ns}"] = CF

        return CF

    def correlate_static_data(
        self,
        gate_ns=(0, np.inf),
        run_duration=None,
        min_time_frac=0.5,
        n_runs_requested=60,
        subtract_afterpulse=True,
        verbose=False,
        **kwargs,
    ) -> CorrFunc:
        """Correlates data for static FCS. Returns a CorrFunc object"""

        if verbose:
            print(
                f"{self.name} [{gate_ns} gating] - Correlating static data '{self.template}':",
                end=" ",
            )

        if run_duration is None:  # auto determination of run duration
            if len(self.cf) > 0:  # read run_time from the last calculated correlation function
                run_duration = next(reversed(self.cf.values())).run_duration
            else:  # auto determine
                total_duration_estimate = 0
                for p in self.data:
                    time_stamps = np.diff(p.runtime).astype(np.int32)
                    mu = np.median(time_stamps) / np.log(2)
                    total_duration_estimate = (
                        total_duration_estimate + mu * len(p.runtime) / self.laser_freq_hz
                    )
                run_duration = total_duration_estimate / n_runs_requested

        self.min_duration_frac = min_time_frac
        duration = []

        SC = SoftwareCorrelator()
        CF = CorrFunc(gate_ns)
        CF.run_duration = run_duration

        with CorrFuncAccumulator(CF) as Accumulator:
            for p in self.data:

                if verbose:
                    print(f"({p.file_num})", end=" ")

                # Ignore short segments (default is below half the run_duration)
                for se_idx, (se_start, se_end) in enumerate(p.all_section_edges):
                    segment_time = (p.runtime[se_end] - p.runtime[se_start]) / self.laser_freq_hz
                    if segment_time < min_time_frac * run_duration:
                        if verbose:
                            print(
                                f"Skipping segment {se_idx} of file {p.file_num} - too short ({segment_time:.2f}s).",
                                end=" ",
                            )
                        CF.skipped_duration += segment_time
                        continue

                    runtime = p.runtime[se_start : se_end + 1]

                    # Gating
                    if hasattr(p, "delay_time"):
                        delay_time = p.delay_time[se_start : se_end + 1]
                        j_gate = CF.gate_ns.valid_indices(delay_time)
                        runtime = runtime[j_gate]
                    #                        delay_time = delay_time[j_gate]  # TODO: why is this not used anywhere?
                    elif CF.gate_ns != (0, np.inf):
                        raise RuntimeError("For gating, TDC must first be calibrated!")

                    # split into segments of approx time of run_duration
                    n_splits = div_ceil(segment_time, run_duration)
                    splits = np.linspace(0, runtime.size, n_splits + 1, dtype=np.int32)
                    time_stamps = np.diff(runtime).astype(np.int32)

                    for k in range(n_splits):

                        ts_split = time_stamps[splits[k] : splits[k + 1]]
                        duration.append(ts_split.sum() / self.laser_freq_hz)
                        SC.correlate(
                            ts_split,
                            CorrelatorType.PH_DELAY_CORRELATOR,
                            timebase_ms=1000 / self.laser_freq_hz,
                        )  # time base of 20MHz to ms

                        if len(CF.lag) < len(SC.lag):
                            CF.lag = SC.lag
                            CF.afterpulse = self.calculate_afterpulse(CF.gate_ns, CF.lag)

                        # subtract afterpulse
                        if subtract_afterpulse:
                            SC.cf_cr = (
                                SC.countrate * SC.corrfunc - CF.afterpulse[: SC.corrfunc.size]
                            )
                        else:
                            SC.cf_cr = SC.countrate * SC.corrfunc

                        # Append new correlation functions
                        Accumulator.accumulate(SC)

        CF.total_duration = sum(duration)

        if verbose:
            if CF.skipped_duration:
                try:
                    skipped_ratio = CF.skipped_duration / CF.total_duration
                except RuntimeWarning:
                    # CF.total_duration = 0
                    print("Whole measurement was skipped! Something's wrong...", end=" ")
                else:
                    print(f"Skipped/total duration: {skipped_ratio:.1#}", end=" ")
            print("- Done.")

        return CF

    def correlate_angular_scan_data(
        self,
        gate_ns=(0, np.inf),
        min_time_frac=0.5,
        subtract_bg_corr=True,
        subtract_afterpulse=True,
        **kwargs,
    ) -> CorrFunc:
        """Correlates data for angular scans. Returns a CorrFunc object"""

        print(
            f"{self.name} [{gate_ns} gating] - Correlating angular scan data '{self.template}':",
            end=" ",
        )

        self.min_duration_frac = min_time_frac  # TODO: not used?
        duration = []

        SC = SoftwareCorrelator()
        CF = CorrFunc(gate_ns)

        with CorrFuncAccumulator(CF) as Accumulator:
            for p in self.data:
                print(f"({p.file_num})", end=" ")
                line_num = p.line_num  # TODO: change this variable's name - photon_line_num?
                min_line, max_line = line_num[line_num > 0].min(), line_num.max()
                if hasattr(p, "delay_time"):  # if measurement quacks as gated
                    j_gate = CF.gate_ns.valid_indices(p.delay_time) | np.isnan(p.delay_time)
                    runtime = p.runtime[j_gate]
                    #                    delay_time = delay_time[j_gate] # TODO: not used?
                    line_num = p.line_num[j_gate]

                elif CF.gate_ns != (0, np.inf):
                    raise RuntimeError(
                        f"A gate '{CF.gate_ns}' was specified for uncalibrated TDC data."
                    )
                else:
                    runtime = p.runtime

                time_stamps = np.diff(runtime).astype(np.int32)
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

                    # TODO: why do these happen?
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

                    SC.correlate(
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
                        CF.afterpulse = self.calculate_afterpulse(CF.gate_ns, CF.lag)

                    # subtract afterpulse
                    if subtract_afterpulse:
                        SC.cf_cr = SC.countrate * SC.corrfunc - CF.afterpulse[: SC.corrfunc.size]
                    else:
                        SC.cf_cr = SC.countrate * SC.corrfunc

                    # Append new correlation functions
                    Accumulator.accumulate(SC)

        CF.vt_um = self.v_um_ms * CF.lag
        CF.total_duration = sum(duration)

        print("- Done.")

        return CF

    def correlate_circular_scan_data(
        self, gate_ns=(0, np.inf), min_time_frac=0.5, subtract_bg_corr=True, **kwargs
    ) -> CorrFunc:
        """Correlates data for circular scans. Returns a CorrFunc object"""

        raise NotImplementedError("Correlation of circular scan data not yet implemented.")

    def plot_correlation_functions(
        self,
        x_field="lag",
        y_field="normalized",
        x_scale="log",
        y_scale="linear",
        parent_figure=None,
        parent_ax=None,
        ylim=(-0.20, 1.4),
        plot_kwargs={},
        **kwargs,
    ) -> List[str]:
        """Doc."""

        with Plotter(
            parent_figure=parent_figure,
            parent_ax=parent_ax,
            super_title=f"'{self.name}' - ACFs",
        ) as ax:
            legend_labels = []
            for cf_name, cf in self.cf.items():
                cf.plot_correlation_function(
                    parent_ax=ax,
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

    def get_field_parameters(
        self,
        fit_range=(0.01, 0.2),
        dynamic_range=None,
        should_plot: bool = False,
        param_estimates=(1, 0.1),
        max_iter: int = 3,
    ):
        """Doc."""

        fit_range = Limits(fit_range)

        # Do first Diff3D fit
        #        subplot(1, 1, 1)

        # now to field fit
        PlotRange = Limits(0, 2 * fit_range.upper)
        self.field.fit_range = fit_range
        j = (self.fast.vt_um > PlotRange(1)) & (self.fast.vt_um < PlotRange(2))
        self.field.r = self.fast.vt_um(j)
        self.field.G = np.real(
            (self.fast.Normalized[j] / self.slow.Normalized(j)) ** (1.0 / self.slow.Normalized[j])
        )
        self.field.errorG = self.fast.errorNormalized(j)

        #        Jfit = (self.field.r > fit_range(1))&(self.field.r < fit_range(2))
        #        fitfun = @(beta, x) beta(1)*exp(-x.^2/beta(2)^2)
        #        FitParam = nlinfitWeight2(self.field.r(Jfit), self.field.G(Jfit), fitfun, param_estimates, self.field.errorG(Jfit), [])
        #        self.field.wXY = FitParam.beta(2)
        #        self.field.FitParam = FitParam
        if dynamic_range is not None:  # do dynamic range iterations
            for iteration in range(max_iter):
                fit_range.upper = self.field.wXY * np.sqrt(np.log(dynamic_range.upper))
                #                Jfit = (self.field.r > fit_range(1))&(self.field.r < fit_range(2))
                #                FitParam = nlinfitWeight2(self.field.r(Jfit), self.field.G(Jfit), fitfun, FitParam.beta, self.field.errorG(Jfit), [])
                #                self.field.wXY = FitParam.beta(2)
                self.field.fit_range = fit_range

    #                self.field.FitParam = FitParam

    #        if not should_plot:
    #            subplot(1, 1, 1)
    #            semilogy(self.field.r.^2, self.field.G, 'o', self.field.r(Jfit).^2, self.field.G(Jfit), '*',...
    #                self.field.r(Jfit).^2, FitParam.beta(1)*exp(-self.field.r(Jfit).^2/self.field.wXY^2))

    def calculate_structure_factors(self) -> None:
        """Doc."""

    #        for cf in self.cf.values():
    #            if not hasattr(cf, )

    def calculate_afterpulse(self, gate_ns: Limits, lag: np.ndarray) -> np.ndarray:
        """Doc."""

        gate_to_laser_pulses = min([1.0, gate_ns.interval() * self.laser_freq_hz / 1e9])
        if self.after_pulse_param[0] == "multi_exponent_fit":
            # work with any number of exponents
            beta = self.after_pulse_param[1]
            afterpulse = gate_to_laser_pulses * np.dot(
                beta[::2], np.exp(-np.outer(beta[1::2], lag))
            )
        elif self.after_pulse_param[0] == "exponent_of_polynom_of_log":  # for old MATLAB files
            beta = self.after_pulse_param[1]
            if lag[0] == 0:
                lag[0] = np.nan
            afterpulse = gate_to_laser_pulses * np.exp(np.polyval(beta, np.log(lag)))

        return afterpulse

    def dump_or_load_data(self, should_load: bool) -> None:
        """
        Load or save the 'data' attribute.
        (relieve RAM - important during multiple-experiment analysis)
        """

        with suppress(AttributeError):
            # AttributeError - name_on_disk is not defined (happens when doing alignment, for example)
            if should_load:  # loading data
                if self.is_data_dumped:
                    logging.debug(
                        f"Loading dumped data '{self.name_on_disk}' from '{self.DUMP_PATH}'."
                    )
                    with suppress(FileNotFoundError):
                        self.data = file_utilities.load_file(self.DUMP_PATH / self.name_on_disk)
                        self.is_data_dumped = False
            else:  # dumping data
                is_saved = file_utilities.save_object_to_disk(
                    self.data,
                    self.DUMP_PATH / self.name_on_disk,
                    size_limits_mb=self.SIZE_LIMITS_MB,
                    compression_method=None,
                )
                if is_saved:
                    self.data = []
                    self.is_data_dumped = True
                    logging.debug(f"Dumped data '{self.name_on_disk}' to '{self.DUMP_PATH}'.")
                else:
                    logging.debug("Data was too small or too large to be saved.")


class ImageSFCSMeasurement(TDCPhotonDataMixin, CountsImageMixin):
    """Doc."""

    def __init__(self):
        pass

    def read_image_data(self, file_path, **kwargs) -> None:
        """Doc."""

        file_dict = file_utilities.load_file_dict(file_path)
        self.process_data(file_dict, **kwargs)

    def process_data(self, file_dict: dict, **kwargs) -> None:
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

        super_title = f"Experiment '{self.name}' - All ACFs"
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
        **kwargs,
    ):

        measurement = SolutionSFCSMeasurement(name=meas_type)

        if "cf_name" not in kwargs:
            if meas_type == "confocal":
                kwargs["cf_name"] = "Confocal"
            else:  # sted
                kwargs["cf_name"] = "CW STED"
        cf_name = kwargs["cf_name"]

        if kwargs.get(f"{meas_type}_file_selection"):
            kwargs["file_selection"] = kwargs[f"{meas_type}_file_selection"]

        measurement.read_fpga_data(
            file_path_template,
            should_plot=should_plot,
            **kwargs,
        )

        if (x_field := kwargs.get("x_field")) is None:
            if measurement.scan_type == "static":
                x_field = "lag"
            else:  # angular or circular scan
                x_field = "vt_um"

        measurement.correlate_and_average(**kwargs)

        if should_plot:
            super_title = f"'{self.name}' Experiment\n'{measurement.name}' Measurement - ACFs"
            with Plotter(super_title=super_title) as ax:
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

    def calibrate_tdc(self, **kwargs):
        """Doc."""

        if hasattr(self.confocal, "scan_type"):  # if confocal maesurement quacks as if loaded
            super_title = f"'{self.name}' Experiment\nTDC Calibration"
            with Plotter(subplots=(2, 4), super_title=super_title, **kwargs) as axes:
                self.confocal.calibrate_tdc(should_plot=True, parent_axes=axes[:, :2], **kwargs)
                kwargs["sync_coarse_time_to"] = self.confocal  # sync sted to confocal
                if hasattr(self.sted, "scan_type"):  # if sted maesurement quacks as if loaded
                    self.sted.calibrate_tdc(should_plot=True, parent_axes=axes[:, 2:], **kwargs)
        else:
            raise RuntimeError(
                "Cannot calibrate TDC if confocal measurement is not loaded to the experiment!"
            )

    def compare_lifetimes(
        self,
        normalization_type="Per Time",
        **kwargs,
    ):
        """Doc."""

        if hasattr(self.confocal, "tdc_calib"):  # if TDC calibration performed
            super_title = f"'{self.name}' Experiment\nLifetime Comparison"
            with Plotter(super_title=super_title, **kwargs) as ax:
                self.confocal.compare_lifetimes(
                    "confocal",
                    compare_to=dict(STED=self.sted),
                    normalization_type=normalization_type,
                    parent_ax=ax,
                )

    def add_gate(self, gate_ns: Tuple[float, float], should_plot=True, **kwargs):
        """Doc."""

        if hasattr(self.sted, "scan_type"):  # if sted maesurement quacks as if loaded
            self.sted.correlate_and_average(gate_ns=gate_ns, **kwargs)
            if should_plot:
                self.plot_correlation_functions(**kwargs)
        else:
            raise RuntimeError(
                "Cannot add a gate if there's no STED measurement loaded to the experiment!"
            )

    def add_gates(self, gate_list: List[Tuple[float, float]], **kwargs):
        """A convecience method for adding multiple gates."""

        if hasattr(self.sted, "scan_type"):  # if sted maesurement quacks as if loaded
            for gate_ns in gate_list:
                self.add_gate(gate_ns, should_plot=False, **kwargs)
            self.plot_correlation_functions(**kwargs)
        else:
            raise RuntimeError(
                "Cannot add a gate if there's no STED measurement loaded to the experiment!"
            )

    def plot_correlation_functions(
        self,
        x_field="vt_um",
        y_field="normalized",
        x_scale="linear",
        y_scale="linear",
        parent_figure=None,
        parent_ax=None,
        xlim=(0, 1),
        ylim=(-0.20, 1.4),
        plot_kwargs={},
        **kwargs,
    ):
        if self.confocal.scan_type == "static":
            x_field = "lag"

        if y_field in {"average_all_cf_cr", "avg_cf_cr"}:
            ylim = None  # will autoscale y

        super_title = f"'{self.name}' Experiment - All ACFs"
        with Plotter(
            parent_figure=parent_figure,
            parent_ax=parent_ax,
            super_title=super_title,
            **kwargs,
        ) as ax:
            confocal_legend_labels = self.confocal.plot_correlation_functions(
                parent_ax=ax,
                x_field=x_field,
                y_field=y_field,
                x_scale=x_scale,
                y_scale=y_scale,
                xlim=xlim,
                ylim=ylim,
                plot_kwargs=plot_kwargs,
            )
            sted_legend_labels = self.sted.plot_correlation_functions(
                parent_ax=ax,
                x_field=x_field,
                y_field=y_field,
                x_scale=x_scale,
                y_scale=y_scale,
                xlim=xlim,
                ylim=ylim,
                plot_kwargs=plot_kwargs,
            )
            ax.legend(confocal_legend_labels + sted_legend_labels)


# TODO - create an AngularScanMixin class and throw the below functions there
def _convert_angular_scan_to_image(runtime, laser_freq_hz, sample_freq_hz, ppl_tot, n_lines):
    """utility function for opening Angular Scans"""

    n_pix_tot = runtime * sample_freq_hz // laser_freq_hz
    # to which pixel photon belongs
    n_pix = np.mod(n_pix_tot, ppl_tot)
    line_num_tot = np.floor_divide(n_pix_tot, ppl_tot)
    # one more line is for return to starting positon
    line_num = np.mod(line_num_tot, n_lines + 1).astype(np.int16)

    img = np.empty((n_lines + 1, ppl_tot), dtype=np.uint16)
    bins = np.arange(-0.5, ppl_tot)
    for j in range(n_lines + 1):
        img[j, :], _ = np.histogram(n_pix[line_num == j], bins=bins)

    return img, n_pix_tot, n_pix, line_num


def _get_best_pix_shift(img: np.ndarray, min_shift, max_shift) -> int:
    """Doc."""

    score = np.empty(shape=(max_shift - min_shift), dtype=np.uint32)
    pix_shifts = np.arange(min_shift, max_shift)
    for idx, pix_shift in enumerate(range(min_shift, max_shift)):
        rolled_img = np.roll(img, pix_shift).astype(np.uint32)
        score[idx] = ((rolled_img[:-1:2, :] - np.fliplr(rolled_img[1::2, :])) ** 2).sum()
    return pix_shifts[score.argmin()]


def _fix_data_shift(cnt) -> int:
    """Doc."""

    print("Fixing line shift...", end=" ")

    height, width = cnt.shape

    min_pix_shift = -round(width / 2)
    max_pix_shift = min_pix_shift + width + 1
    pix_shift = _get_best_pix_shift(cnt, min_pix_shift, max_pix_shift)

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
        pix_shift = _get_best_pix_shift(cnt, min_pix_shift, max_pix_shift)

    return pix_shift


def _threshold_and_smooth(img, otsu_classes=4, n_bins=256, disk_radius=2) -> np.ndarray:
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


def _line_correlations(image, bw_mask, roi, sampling_freq) -> list:
    """Returns a list of auto-correlations of the lines of an image"""

    image_line_corr = []
    for j in range(roi["row"].min(), roi["row"].max() + 1):
        line = image[j][bw_mask[j] > 0]
        try:
            c, lags = _auto_corr(line.astype(np.float64))
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


def _auto_corr(a):
    """Does correlation similar to Matlab xcorr, cuts positive lags, normalizes properly"""

    c = np.correlate(a, a, mode="full")
    c = c[c.size // 2 :]
    c = c / np.arange(c.size, 0, -1)
    lags = np.arange(c.size, dtype=np.uint16)

    return c, lags


def _calculate_weighted_avg(cf_cr, weights):
    """Doc."""

    tot_weights = weights.sum(0)
    # TODO: error handeling for the row below (zero division) - can/should it be detected beforehand?
    avg_cf_cr = (cf_cr * weights).sum(0) / tot_weights
    error_cf_cr = np.sqrt((weights ** 2 * (cf_cr - avg_cf_cr) ** 2).sum(0)) / tot_weights

    return avg_cf_cr, error_cf_cr
