"""Fit Tools"""

import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from utilities.errors import err_hndlr

warnings.simplefilter("error", OptimizeWarning)
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
):
    """Doc."""

    x_min, x_max = x_limits
    y_min, y_max = y_limits

    in_lims = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
    is_finite_err = (ys_errors > 0) & np.isfinite(ys_errors)
    x = xs[in_lims & is_finite_err]
    y = ys[in_lims & is_finite_err]
    y_err = ys_errors[in_lims & is_finite_err]
    fit_func = globals()[fit_name]

    try:
        bta, cov_mat = curve_fit(
            fit_func, x, y, p0=param_estimates, sigma=y_err, absolute_sigma=True
        )
        fit_param = dict()
        fit_param["beta"] = bta
        fit_param["errorBeta"] = np.sqrt(np.diag(cov_mat))

    except (RuntimeWarning, RuntimeError, OptimizeWarning) as exc:
        raise FitError(err_hndlr(exc, sys._getframe(), None, lvl="debug"))

    else:
        chi_sq_arr = np.square((fit_func(x, *bta) - y) / y_err)
        fit_param["chi_sq_norm"] = chi_sq_arr.sum() / x.size
        fit_param["x"] = x
        fit_param["y"] = y
        fit_param["x_limits"] = x_limits
        fit_param["y_limits"] = y_limits
        fit_param["fit_func"] = fit_name

        if not no_plot:
            plt.errorbar(xs, ys, ys_errors, fmt=".")
            plt.plot(xs[in_lims], fit_func(xs[in_lims], *bta), zorder=10)
            plt.xscale(x_scale)
            plt.yscale(y_scale)
            plt.gcf().canvas.draw()
            plt.autoscale()
            plt.show()

    return fit_param


# Fit functions
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
