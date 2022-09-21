"""Fit Tools"""

import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import scipy.optimize as opt

from utilities.display import Plotter
from utilities.errors import err_hndlr
from utilities.helper import Limits

warnings.simplefilter("error", opt.OptimizeWarning)
warnings.simplefilter("error", RuntimeWarning)


FIT_NAME_DICT = globals()


class FitError(Exception):
    pass


@dataclass
class FitParams:
    """Doc."""

    func_name: str = None
    beta: Dict[str, float] = None
    beta_error: Dict[str, float] = None
    x: np.ndarray = None
    y: np.ndarray = None
    sigma: np.ndarray = None
    x_limits: Limits = field(default_factory=Limits)
    y_limits: Limits = field(default_factory=Limits)
    chi_sq_norm: float = None


def curve_fit_lims(
    fit_func,
    param_estimates,
    xs,
    ys,
    ys_errors=None,
    x_limits=Limits(),
    y_limits=Limits(),
    should_plot=True,
    plot_kwargs=dict(x_scale="log"),
    **kwargs,
) -> FitParams:
    """Doc."""

    should_plot_errorbars = True

    if ys_errors is None:
        should_plot_errorbars = False
        ys_errors = np.ones(ys.shape)

    in_lims = x_limits.valid_indices(xs) & y_limits.valid_indices(ys)
    is_finite_err = (ys_errors > 0) & np.isfinite(ys_errors)
    x = xs[in_lims & is_finite_err]
    y = ys[in_lims & is_finite_err]
    y_err = ys_errors[in_lims & is_finite_err]

    fit_params = _fit_and_get_param_dict(
        fit_func, x, y, param_estimates, sigma=y_err, absolute_sigma=True, **kwargs
    )

    if should_plot:
        with Plotter(super_title=f"Curve Fit ({fit_func.__name__})", **plot_kwargs) as ax:
            ax.plot(xs[in_lims], ys[in_lims], ".k")
            ax.plot(xs[in_lims], fit_func(xs[in_lims], *fit_params.beta.values()), "--r")
            if should_plot_errorbars:
                ax.errorbar(xs, ys, ys_errors, fmt=".")

    return fit_params


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


def _fit_and_get_param_dict(fit_func, x, y, p0, sigma=1, **kwargs) -> FitParams:
    """Doc."""

    try:
        popt, pcov = opt.curve_fit(fit_func, x, y, p0=p0, **kwargs)
    except (RuntimeWarning, RuntimeError, opt.OptimizeWarning, ValueError) as exc:
        raise FitError(err_hndlr(exc, sys._getframe(), None, lvl="debug"))

    func_name = fit_func.__name__
    param_names = fit_func.__code__.co_varnames[: fit_func.__code__.co_argcount][1:]
    chi_sq_arr = np.square((fit_func(x, *popt) - y) / sigma)
    try:
        chi_sq_norm = chi_sq_arr.sum() / x.size
    except AttributeError:  # x is a tuple (e.g. for 2D Gaussian)
        chi_sq_norm = chi_sq_arr.sum() / x[0].size

    beta = {name: val for name, val in zip(param_names, popt)}
    try:
        beta_error = {name: error for name, error in zip(param_names, np.sqrt(np.diag(pcov)))}
    except Exception as exc:
        raise FitError(err_hndlr(exc, sys._getframe(), None, lvl="debug"))

    return FitParams(
        func_name,
        beta,
        beta_error,
        x,
        y,
        sigma,
        kwargs.get("x_limits", Limits()),
        kwargs.get("y_limits", Limits()),
        chi_sq_norm,
    )


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


def linear_fit(t, a, b):
    return a * t + b


def power_fit(t, a, n):
    return a * t ** n


def diffusion_3d_fit(t, G0, tau, w_sq):
    return G0 / (1 + t / tau) / np.sqrt(1 + t / tau / w_sq)


def exponent_with_background_fit(t, A, tau, bg):
    return A * np.exp(-t / tau) + bg


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
