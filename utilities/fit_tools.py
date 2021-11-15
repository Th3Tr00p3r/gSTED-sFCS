"""Fit Tools"""

import sys
import warnings

import numpy as np
import scipy.optimize as opt

from utilities.display import Plotter
from utilities.errors import err_hndlr
from utilities.helper import LimitRange

warnings.simplefilter("error", opt.OptimizeWarning)
warnings.simplefilter("error", RuntimeWarning)


class FitError(Exception):
    pass


def curve_fit_lims(
    fit_name,
    param_estimates,
    xs,
    ys,
    ys_errors,
    x_limits=LimitRange(np.NINF, np.Inf),
    y_limits=LimitRange(np.NINF, np.Inf),
    should_plot=True,
    x_scale="log",
    y_scale="linear",
    plot_kwargs={},
    **kwargs,
) -> dict:
    """Doc."""

    in_lims = x_limits.valid_indices(xs) & y_limits.valid_indices(ys)
    is_finite_err = (ys_errors > 0) & np.isfinite(ys_errors)
    x = xs[in_lims & is_finite_err]
    y = ys[in_lims & is_finite_err]
    y_err = ys_errors[in_lims & is_finite_err]
    fit_func = globals()[fit_name]

    fit_param = _fit_and_get_param_dict(
        fit_func, x, y, param_estimates, sigma=y_err, absolute_sigma=True
    )
    fit_param["x_limits"] = x_limits
    fit_param["y_limits"] = y_limits
    chi_sq_arr = np.square((fit_func(x, *fit_param["beta"]) - y) / y_err)
    fit_param["chi_sq_norm"] = chi_sq_arr.sum() / x.size

    if should_plot:
        with Plotter() as ax:
            ax.set_xscale(x_scale)
            ax.set_yscale(y_scale)
            ax.plot(
                xs[in_lims], fit_func(xs[in_lims], *fit_param["beta"]), zorder=10, **plot_kwargs
            )
            ax.errorbar(xs, ys, ys_errors, fmt=".")

    return fit_param


def fit_2d_gaussian_to_image(data: np.ndarray) -> dict:
    """Doc."""

    height, width = data.shape
    x1 = np.arange(height)
    x2 = np.arange(width)
    x1, x2 = np.meshgrid(x1, x2)
    y = data.ravel()

    # estimate initial parameters from data
    # (amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
    p0 = (y.max() - y.min(), height / 2, width / 2, height / 2, width / 2, 0, y.min())

    return _fit_and_get_param_dict(gaussian_2d_fit, (x1, x2), y, p0)


def _fit_and_get_param_dict(fit_func, x, y, p0, **kwargs) -> dict:
    """Doc."""

    try:
        popt, pcov = opt.curve_fit(fit_func, x, y, p0=p0, **kwargs)
    except (RuntimeWarning, RuntimeError, opt.OptimizeWarning) as exc:
        raise FitError(err_hndlr(exc, sys._getframe(), None, lvl="debug"))

    fit_param = dict()
    fit_param["func_name"] = fit_func.__name__
    fit_param["beta"] = popt
    try:
        fit_param["beta_error"] = np.sqrt(np.diag(pcov))
    except Exception as exc:
        raise FitError(err_hndlr(exc, sys._getframe(), None, lvl="debug"))
    fit_param["x"] = x
    fit_param["y"] = y
    return fit_param


def gaussian_2d_fit(xy_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Adapted from:
    https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
    """

    x, y = xy_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g.ravel()


def linear_fit(t, a, b):
    return a * t + b


def power_fit(t, a, n):
    return a * t ** n


def diffusion_3d_fit(t, A, tau, w_sq):
    return A / (1 + t / tau) / np.sqrt(1 + t / tau / w_sq)


def exponent_with_background_fit(t, A, tau, bg):
    return A * np.exp(-t / tau) + bg


def ratio_of_lifetime_histograms(t, sig_x, sig_y, t0):
    if t >= t0:
        return np.sqrt(1 + sig_x * (t - t0)) * np.sqrt(1 + sig_y * (t - t0))
    else:
        return 1


def weighted_average(data_vectors_list, data_errors_list):
    """Doc."""

    data_vectors = np.array(data_vectors_list)
    data_errors = np.array(data_errors_list)
    weights = data_errors ** (-2)
    wa = np.sum(data_vectors * weights, 0) / np.sum(weights, 0)
    we = np.sum(weights, 0) ** (-2)
    return wa, we
