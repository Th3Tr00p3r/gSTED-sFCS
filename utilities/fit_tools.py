"""Fit Tools"""

import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.optimize as opt

from utilities.display import Plotter
from utilities.errors import err_hndlr
from utilities.helper import Limits

warnings.simplefilter("error", opt.OptimizeWarning)
warnings.simplefilter("error", RuntimeWarning)


class FitError(Exception):
    pass


@dataclass
class FitParams:
    """Doc."""

    func_name: str = None
    beta: tuple = None
    beta_error: tuple = None
    x: np.ndarray = None
    y: np.ndarray = None
    sigma: np.ndarray = None
    x_limits: Limits = field(default_factory=Limits)
    y_limits: Limits = field(default_factory=Limits)
    chi_sq_norm: float = None


def curve_fit_lims(
    fit_name,
    param_estimates,
    xs,
    ys,
    ys_errors=None,
    x_limits=Limits(),
    y_limits=Limits(),
    should_plot=True,
    x_scale="log",
    y_scale="linear",
    plot_kwargs={},
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
    fit_func = globals()[fit_name]

    fit_params = _fit_and_get_param_dict(
        fit_func, x, y, param_estimates, sigma=y_err, absolute_sigma=True, **kwargs
    )

    if should_plot:
        with Plotter(
            super_title=f"Curve Fit ({fit_name})", x_scale=x_scale, y_scale=y_scale, **plot_kwargs
        ) as ax:
            ax.plot(xs[in_lims], ys[in_lims], ".k")
            ax.plot(xs[in_lims], fit_func(xs[in_lims], *fit_params.beta), "--r")
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
    # (amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
    p0 = (y.max() - y.min(), height / 2, width / 2, height / 2, width / 2, 0, y.min())

    return _fit_and_get_param_dict(gaussian_2d_fit, (x1, x2), y, p0)


def _fit_and_get_param_dict(fit_func, x, y, p0, sigma=1, **kwargs) -> FitParams:
    """Doc."""

    try:
        popt, pcov = opt.curve_fit(fit_func, x, y, p0=p0, **kwargs)
    except (RuntimeWarning, RuntimeError, opt.OptimizeWarning) as exc:
        raise FitError(err_hndlr(exc, sys._getframe(), None, lvl="debug"))

    func_name = fit_func.__name__
    beta = popt
    try:
        beta_error = np.sqrt(np.diag(pcov))
    except Exception as exc:
        raise FitError(err_hndlr(exc, sys._getframe(), None, lvl="debug"))
    chi_sq_arr = np.square((fit_func(x, *beta) - y) / sigma)
    try:
        chi_sq_norm = chi_sq_arr.sum() / x.size
    except AttributeError:  # x is a tuple (e.g. for 2D Gaussian)
        chi_sq_norm = chi_sq_arr.sum() / x[0].size

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


def diffusion_3d_fit(t, a, tau, w_sq):
    return a / (1 + t / tau) / np.sqrt(1 + t / tau / w_sq)


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


def decaying_cosine_fit(t, a, omega, phi, tau, c):
    return a * np.cos(omega * t + phi) * np.exp(-t / tau) + c


def ratio_of_lifetime_histograms(t, sig_x, sig_y, t0):
    if (t >= t0).all():
        return np.sqrt(1 + sig_x * (t - t0)) * np.sqrt(1 + sig_y * (t - t0))
    else:
        return 1
