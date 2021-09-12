"""Fit Tools"""

import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from utilities.errors import err_hndlr

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
    x_limits=(np.NINF, np.Inf),
    y_limits=(np.NINF, np.Inf),
    no_plot=False,
    x_scale="log",
    y_scale="linear",
) -> dict:
    """Doc."""

    x_min, x_max = x_limits
    y_min, y_max = y_limits

    in_lims = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
    is_finite_err = (ys_errors > 0) & np.isfinite(ys_errors)
    x = xs[in_lims & is_finite_err]
    y = ys[in_lims & is_finite_err]
    y_err = ys_errors[in_lims & is_finite_err]
    fit_func = globals()[fit_name]

    fit_param = fit_and_get_param_dict(
        fit_func, x, y, param_estimates, sigma=y_err, absolute_sigma=True
    )
    fit_param["x_limits"] = x_limits
    fit_param["y_limits"] = y_limits
    chi_sq_arr = np.square((fit_func(x, *fit_param["beta"]) - y) / y_err)
    fit_param["chi_sq_norm"] = chi_sq_arr.sum() / x.size

    if not no_plot:
        plt.errorbar(xs, ys, ys_errors, fmt=".")
        plt.plot(xs[in_lims], fit_func(xs[in_lims], *fit_param["beta"]), zorder=10)
        plt.xscale(x_scale)
        plt.yscale(y_scale)
        plt.gcf().canvas.draw()
        plt.autoscale()
        plt.show()

    return fit_param


def fit_2d_gaussian(data: np.ndarray) -> dict:
    """Doc."""

    height, width = data.shape
    x1 = np.arange(height)
    x2 = np.arange(width)
    x1, x2 = np.meshgrid(x1, x2)
    y = data.ravel()

    # estimate initial parameters from data
    # (amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
    p0 = (y.max() - y.min(), height / 2, width / 2, height / 2, width / 2, 0, y.min())

    return fit_and_get_param_dict(gaussian_2d_fit, (x1, x2), y, p0)


def fit_and_get_param_dict(fit_func, x, y, p0, **kwargs) -> dict:
    """Doc."""

    try:
        popt, pcov = opt.curve_fit(fit_func, x, y, p0=p0, **kwargs)
    except (RuntimeWarning, RuntimeError, opt.OptimizeWarning) as exc:
        raise FitError(err_hndlr(exc, sys._getframe(), None, lvl="debug"))

    fit_param = dict()
    fit_param["func_name"] = fit_func.__name__
    fit_param["beta"] = popt
    fit_param["beta_error"] = np.sqrt(np.diag(pcov))
    fit_param["x"] = x
    fit_param["y"] = y
    return fit_param


# Fit functions
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


def weighted_average(data_vectors_list, data_errors_list):
    """Doc."""

    data_vectors = np.array(data_vectors_list)
    data_errors = np.array(data_errors_list)
    weights = data_errors ** (-2)
    wa = np.sum(data_vectors * weights, 0) / np.sum(weights, 0)
    we = np.sum(weights, 0) ** (-2)
    return wa, we