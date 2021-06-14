#!/usr/bin/env python3
"""
Created on Thu Dec 20 17:20:00 2018

@author: Oleg Krichevsky okrichev@bgu.ac.il
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

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
    x_lim=[],
    y_lim=[],
    no_plot=False,
    x_scale="log",
    y_scale="linear",
):
    """Doc."""

    if len(y_lim) == 0:
        y_lim = np.array([np.NINF, np.Inf])
    elif len(y_lim) == 1:
        y_lim = np.array([y_lim, np.Inf])

    if len(x_lim) == 0:
        x_lim = np.array([np.NINF, np.Inf])
    elif len(x_lim) == 1:
        x_lim = np.array([x_lim, np.Inf])

    in_lims = (xs >= x_lim[0]) & (xs <= x_lim[1]) & (ys >= y_lim[0]) & (ys <= y_lim[1])
    is_finite_err = (ys_errors > 0) & np.isfinite(ys_errors)
    # print(in_lims)
    x = xs[in_lims & is_finite_err]
    #   print(x)
    y = ys[in_lims & is_finite_err]
    y_err = ys_errors[in_lims & is_finite_err]
    #    print(y_err)
    fit_func = globals()[fit_name]
    #    print(fit_name)
    try:
        bta, cov_mat = curve_fit(
            fit_func, x, y, p0=param_estimates, sigma=y_err, absolute_sigma=True
        )
        fit_param = dict()
        fit_param["beta"] = bta
        fit_param["errorBeta"] = np.sqrt(np.diag(cov_mat))
    except (RuntimeWarning, RuntimeError, OptimizeWarning):
        raise FitError("curve_fit() failed.")
    chi_sq_arr = np.square((fit_func(x, *bta) - y) / y_err)
    fit_param["chi_sq_norm"] = chi_sq_arr.sum() / x.size
    fit_param["x"] = x
    fit_param["y"] = y
    fit_param["x_lim"] = x_lim
    fit_param["y_lim"] = y_lim
    fit_param["fit_func"] = fit_name
    # print(fit_param)

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
def LinearFit(t, a, b):
    return a * t + b


def PowerFit(t, a, n):
    return a * t ** n


def Diffusion3Dfit(t, A, tau, w_sq):
    return A / (1 + t / tau) / np.sqrt(1 + t / tau / w_sq)


def WeightedAverage(data_vectors_list, data_errors_list):
    """Doc."""

    data_vectors = np.array(data_vectors_list)
    data_errors = np.array(data_errors_list)
    weights = data_errors ** (-2)
    wa = np.sum(data_vectors * weights, 0) / np.sum(weights, 0)
    we = np.sum(weights, 0) ** (-2)
    return wa, we
