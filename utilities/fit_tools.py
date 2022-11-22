"""Fit Tools"""

import sys
import warnings
from contextlib import suppress
from dataclasses import dataclass, field
from string import ascii_lowercase
from typing import Callable, Dict

import numpy as np
import scipy.optimize as opt

from utilities.display import Plotter
from utilities.errors import err_hndlr
from utilities.helper import Limits

warnings.simplefilter("error", opt.OptimizeWarning)
warnings.simplefilter("error", RuntimeWarning)


FIT_NAME_DICT = globals()


class FitError(Exception):
    ...


@dataclass
class FitParams:
    """Doc."""

    fit_func: Callable
    beta: Dict[str, float]
    beta_error: Dict[str, float]
    beta_estimate: Dict[str, float]
    xs: np.ndarray
    ys: np.ndarray
    ys_errors: np.ndarray
    valid_idxs: np.ndarray
    chi_sq_norm: float
    x_limits: Limits = field(default_factory=Limits)
    y_limits: Limits = field(default_factory=Limits)

    def __post_init__(self):
        self.x = self.xs[self.valid_idxs]
        self.y = self.ys[self.valid_idxs]
        self.sigma = self.ys_errors[self.valid_idxs]

    def plot(self, color=None, fit_label="Fit", **kwargs):
        """Doc."""

        with Plotter(
            super_title=f"Curve Fit ({self.fit_func.__name__})",
            xlim=(min(self.x), max(self.x)),
            ylim=(min(self.y), max(self.y)),
            **kwargs,
        ) as ax:
            ax.plot(
                self.xs,
                self.ys,
                ".",
                label="_Data",
                zorder=1,
                markersize=2,
                color=color if color is not None else "k",
            )
            if not (self.sigma == 1).all():
                ax.errorbar(
                    self.xs,
                    self.ys,
                    self.ys_errors,
                    fmt="none",
                    label="_Error",
                    elinewidth=0.5,
                    zorder=2,
                    color=color if color is not None else "k",
                )
            ax.plot(
                self.x,
                self.fit_func(self.x, *self.beta.values()),
                "--",
                label=fit_label,
                zorder=3,
                color=color if color is not None else "r",
            )
            ax.legend(["Data", "Error", "Fit"])


def curve_fit_lims(
    fit_func,
    param_estimates,
    xs,
    ys,
    ys_errors=None,
    x_limits=Limits(),
    y_limits=Limits(),
    should_plot=False,
    plot_kwargs={},
    **kwargs,
) -> FitParams:
    """Doc."""

    if ys_errors is None:
        ys_errors = np.ones(ys.shape)

    in_lims = x_limits.valid_indices(xs) & y_limits.valid_indices(ys)
    is_finite_err = (ys_errors > 0) & np.isfinite(ys_errors)
    valid_idxs = in_lims & is_finite_err

    FP = _fit_and_get_param_dict(
        fit_func,
        xs,
        ys,
        param_estimates,
        ys_errors=ys_errors,
        valid_idxs=valid_idxs,
        absolute_sigma=True,
        **kwargs,
    )

    if should_plot:
        FP.plot(**plot_kwargs)

    return FP


def fit_2d_gaussian_to_image(data: np.ndarray) -> FitParams:
    """Doc."""

    height, width = data.shape
    x1 = np.arange(height)
    x2 = np.arange(width)
    x1, x2 = np.meshgrid(x1, x2)
    y = data.ravel()

    # estimate initial parameters from data
    # (amplitude, x0, y0, sigma_x, sigma_y, theta, offset)
    p0 = (y.max() - y.min(), height / 2, width / 2, height / 2, width / 2, 0, y.min())

    return _fit_and_get_param_dict(gaussian_2d_fit, (x1, x2), y, p0)


def fit_lifetime_histogram(xs, ys, meas_type: str, **kwargs):
    """Doc."""
    # TODO: Merge - have 'fit_lifetime_hist' method of 'TDCCalibration' use this.

    if meas_type == "confocal":
        fp = curve_fit_lims(
            FIT_NAME_DICT["exponent_with_background_fit"],
            [ys.max(), 3.5, ys.min() * 10],  # TODO: choose better starting values?
            bounds=([0] * 3, [np.inf, 10, np.inf]),
            xs=xs,
            ys=ys,
            x_limits=Limits(xs[np.argmax(ys)], np.inf),
            plot_kwargs=dict(y_scale="log"),
            **kwargs,
        )
    elif meas_type == "sted":
        fp = curve_fit_lims(
            FIT_NAME_DICT["sted_hist_fit"],
            # A, tau, sigma0, sigma, bg
            [ys.max(), 1, 1e-5, 1, ys.min() * 10],  # TODO: choose better starting values?
            bounds=([0] * 5, [np.inf, 10, 1, np.inf, np.inf]),
            xs=xs,
            ys=ys,
            x_limits=Limits(xs[np.argmax(ys)], np.inf),
            plot_kwargs=dict(y_scale="log"),
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid measurement type: {meas_type}")

    return fp


def _fit_and_get_param_dict(
    fit_func, xs, ys, p0, ys_errors=1, valid_idxs=slice(None), **kwargs
) -> FitParams:
    """Doc."""

    if "linear" in fit_func.__name__:
        with suppress(KeyError):
            kwargs.pop("max_nfev")

    x = xs[valid_idxs]
    y = ys[valid_idxs]
    sigma = ys_errors[valid_idxs]

    try:
        popt, pcov = opt.curve_fit(fit_func, x, y, p0=p0, **kwargs)
    except (RuntimeWarning, RuntimeError, opt.OptimizeWarning, ValueError) as exc:
        raise FitError(err_hndlr(exc, sys._getframe(), None, lvl="debug"))

    param_names = fit_func.__code__.co_varnames[: fit_func.__code__.co_argcount][1:]
    if param_names == ():
        param_names = (letter for _, letter in zip(p0, ascii_lowercase))
    chi_sq_arr = np.square((fit_func(x, *popt) - y) / sigma)
    try:
        chi_sq_norm = chi_sq_arr.sum() / x.size
    except AttributeError:  # x is a tuple (e.g. for 2D Gaussian)
        chi_sq_norm = chi_sq_arr.sum() / x[0].size

    beta = {name: val for name, val in zip(param_names, popt)}
    beta_estimate = {name: val for name, val in zip(param_names, p0)}
    try:
        beta_error = {name: error for name, error in zip(param_names, np.sqrt(np.diag(pcov)))}
    except RuntimeError as exc:
        raise FitError(err_hndlr(exc, sys._getframe(), None, lvl="debug"))

    return FitParams(
        fit_func,
        beta,
        beta_error,
        beta_estimate,
        xs,
        ys,
        ys_errors,
        valid_idxs,
        chi_sq_norm,
        kwargs.get("x_limits", Limits()),
        kwargs.get("y_limits", Limits()),
    )


def gaussian_1d_fit(t, A, mu, sigma, bg):
    return A * np.exp(-1 / 2 * ((t - mu) / sigma) ** 2) + bg


def zero_centered_zero_bg_normalized_gaussian_1d_fit(t, sigma):
    return gaussian_1d_fit(t, 1, 0, sigma, 0)


def gaussian_2d_fit(xy_tuple, amplitude, x0, y0, sigma_x, sigma_y, phi, offset):
    """
    Adapted from:
    https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
    """

    x, y = xy_tuple
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(phi) ** 2) / (2 * sigma_x ** 2) + (np.sin(phi) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * phi)) / (4 * sigma_x ** 2) + (np.sin(2 * phi)) / (4 * sigma_y ** 2)
    c = (np.sin(phi) ** 2) / (2 * sigma_x ** 2) + (np.cos(phi) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(
        -(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2))
    )
    return g.ravel()


def linear_fit(t, m, n):
    return m * t + n


def power_fit(t, a, n):
    return a * t ** n


def polynomial_fit(t, *beta):
    """
    Work with polynomial of any degree, i.e.:
    y = beta[0] + beta[1]*t + beta[2]*t**2 + ... + beta[n]*t**n
    """

    amplitude_row_vec = np.array(beta)[:, np.newaxis].T
    power_column_vec = np.array([t ** n for n in range(len(beta))])
    return (amplitude_row_vec @ power_column_vec).squeeze()


def diffusion_3d_fit(t, G0, tau, w_sq):
    return G0 / (1 + t / tau) / np.sqrt(1 + t / tau / w_sq)


def exponent_with_background_fit(t, A, tau, bg):
    return A * np.exp(-t / tau) + bg


def sted_hist_fit(t, A, tau, sigma0, sigma, bg):
    return A / tau * np.exp(-(sigma0 + 1) * t / tau) / (1 + sigma * t / tau) + bg


def multi_exponent_fit(t, *beta):
    """
    Work with any number of exponents, e.g.:
    y = beta[0]*exp(-beta[1]*t) + beta[2]*exp(-beta[3]*t) + beta[4]*exp(-beta[5]*t)
    """

    amplitude_row_vec = np.array(beta[::2])[:, np.newaxis].T
    decay_column_vec = np.array(beta[1::2])[:, np.newaxis]
    return (amplitude_row_vec @ np.exp(-decay_column_vec * t)).squeeze()


def ratio_of_lifetime_histograms_fit(t, sigma_x, sigma_y, t0):
    if (t >= t0).all():
        return np.sqrt(1 + sigma_x * (t - t0)) * np.sqrt(1 + sigma_y * (t - t0))
    else:
        return 1
