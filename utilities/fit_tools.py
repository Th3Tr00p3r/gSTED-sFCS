"""Fit Tools"""

import sys
import warnings
from dataclasses import dataclass, field
from string import ascii_lowercase
from typing import Any, Callable, Dict

import numpy as np
import scipy as sp

from utilities.display import Plotter
from utilities.errors import err_hndlr
from utilities.helper import Limits, get_func_attr

warnings.simplefilter("error", sp.optimize.OptimizeWarning)
warnings.simplefilter("error", RuntimeWarning)


class FitError(Exception): ...  # noqa: E701


@dataclass
class FitParams:
    """Doc."""

    fit_func: Callable
    beta: Dict[str, float]
    beta_error: Dict[str, float]
    beta0: Dict[str, float]
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
        try:
            self.sigma = self.ys_errors[self.valid_idxs]
        except TypeError:
            self.sigma = 1
        try:
            self.fitted_y = self.fit_func(self.x.astype(np.float64), *self.beta.values())
        except AttributeError:
            # self.x is a tuple - this class really only makes sense for 1D data fitting!
            pass

    def interpolate_y(self, x):
        """Doc."""

        return np.interp(x, self.x, self.fitted_y)

    def plot(self, color=None, fit_label="Fit", errorbars=True, **kwargs):
        """Doc."""

        with Plotter(
            super_title=kwargs.pop("super_title", f"Curve Fit ({self.fit_func.__name__})"),
            xlim=kwargs.pop("xlim", (min(self.x), max(self.x))),
            ylim=kwargs.pop("ylim", (min(self.y), max(self.y))),
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
            if not (self.sigma == 1).all() and errorbars:
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
                self.fitted_y,
                "--",
                label=fit_label,
                zorder=3,
                color=color if color is not None else "r",
            )
            ax.legend()

    def print_fitted_params(self):
        """Doc."""
        print(f"Fitted parameters for {get_func_attr(self.fit_func, '__name__')}:")
        print(f"Chi-squared / N = {self.chi_sq_norm:.3e}")
        for name, val in self.beta.items():
            print(f"{name} = {val:.3e} +/- {self.beta_error[name]:.3e}")


def curve_fit_lims(
    fit_func: Callable,
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

    fit_params = _fit_and_get_param_dict(
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
        fit_params.plot(**plot_kwargs)

    return fit_params


def _fit_and_get_param_dict(
    fit_func: Callable,
    xs,
    ys,
    p0,
    ys_errors=None,
    valid_idxs=slice(None),
    should_weight_fits: bool = False,
    curve_fit_kwargs: Dict[str, Any] = {},
    **kwargs,
) -> FitParams:
    """Doc."""

    if "linear" in get_func_attr(fit_func, "__name__"):
        curve_fit_kwargs.pop("max_nfev", None)

    x = xs[valid_idxs]
    y = ys[valid_idxs]
    try:
        # TODO: using weighted fitting ruins my resolution!
        #  why is that? what is the legitimate method?
        sigma = ys_errors[valid_idxs] if should_weight_fits else np.ones(y.shape)
        ys_errors[~valid_idxs] = 0
    except TypeError:
        sigma = np.ones(y.shape)

    try:
        popt, pcov = sp.optimize.curve_fit(fit_func, x, y, p0=p0, sigma=sigma, **curve_fit_kwargs)
    except (RuntimeWarning, RuntimeError, sp.optimize.OptimizeWarning, ValueError) as exc:
        raise FitError(err_hndlr(exc, sys._getframe(), None, lvl="debug"))

    fit_func_code = get_func_attr(fit_func, "__code__")
    param_names = fit_func_code.co_varnames[: fit_func_code.co_argcount][1:]
    if param_names == ():
        param_names = tuple(letter for _, letter in zip(p0, ascii_lowercase))
    chi_sq_arr = np.square((fit_func(x, *popt) - y) / sigma)
    try:
        chi_sq_norm = chi_sq_arr.sum() / x.size
    except AttributeError:  # x is a tuple (e.g. for 2D Gaussian)
        chi_sq_norm = chi_sq_arr.sum() / x[0].size

    beta = {name: val for name, val in zip(param_names, popt)}
    beta0 = {name: val for name, val in zip(param_names, p0)}
    try:
        beta_error = {name: error for name, error in zip(param_names, np.sqrt(np.diag(pcov)))}
    except RuntimeError as exc:
        raise FitError(err_hndlr(exc, sys._getframe(), None, lvl="debug"))

    return FitParams(
        fit_func,
        beta,
        beta_error,
        beta0,
        xs,
        ys,
        ys_errors,
        valid_idxs,
        chi_sq_norm,
        kwargs.get("x_limits", Limits()),
        kwargs.get("y_limits", Limits()),
    )


def find_dark_core(data: np.ndarray, bg: float = None) -> Dict[str, Any]:
    """
    Locate the dark core of a vortex (donut) beam, and the beam's own centroid.

    A phase singularity forces the field to zero on its own axis at every propagation
    distance, so a camera placed anywhere after the vortex plate shows a dark core whose
    transverse position IS the singularity's position (1:1, no magnification). The offset
    between that core and the beam centroid is therefore the plate's decentring.

    Returns a dict with 'core' (x, y), 'centroid' (x, y), 'offset' (dx, dy) in pixels,
    'offset_px' (its magnitude), 'beam_radius' (px) and 'core_depth' (fraction of peak).
    Raises FitError if no beam is found.
    """

    data = np.asarray(data, dtype=float)

    # background: median of a 10% border frame
    if bg is None:
        h, w = data.shape
        k = max(int(0.1 * min(h, w)), 2)
        bg = float(
            np.median(
                np.concatenate(
                    [data[:k].ravel(), data[-k:].ravel(), data[:, :k].ravel(), data[:, -k:].ravel()]
                )
            )
        )
    a = np.clip(data - bg, 0, None)
    if a.max() <= 0:
        raise FitError("No beam found in image (all background).")

    # beam centroid - the small dark core barely biases it
    yy, xx = np.indices(a.shape)
    weights = np.where(a > 0.05 * a.max(), a, 0.0)
    total = weights.sum()
    if total <= 0:
        raise FitError("No beam found in image.")
    y_beam = (yy * weights).sum() / total
    x_beam = (xx * weights).sum() / total

    # beam radius from the second moment (1/e^2 radius for a Gaussian)
    r_sq = (((yy - y_beam) ** 2 + (xx - x_beam) ** 2) * weights).sum() / total
    beam_radius = float(np.sqrt(2 * r_sq))

    # dark core: minimum of the smoothed image, searched inside the beam only
    smoothed = sp.ndimage.uniform_filter(a, max(int(beam_radius / 15), 1))
    r = np.hypot(yy - y_beam, xx - x_beam)
    masked = np.where(r < 0.5 * beam_radius, smoothed, smoothed.max())
    iy, ix = np.unravel_index(np.argmin(masked), masked.shape)

    # refine: centroid of the inverted image in a small box about that minimum
    box = max(int(0.08 * beam_radius), 3)
    sl = (slice(max(iy - box, 0), iy + box + 1), slice(max(ix - box, 0), ix + box + 1))
    patch_ = smoothed[sl].max() - smoothed[sl]
    if patch_.sum() <= 0:
        raise FitError("Could not localize a dark core.")
    y_core = float((yy[sl] * patch_).sum() / patch_.sum())
    x_core = float((xx[sl] * patch_).sum() / patch_.sum())

    return dict(
        core=(x_core, y_core),
        centroid=(x_beam, y_beam),
        offset=(x_core - x_beam, y_core - y_beam),
        offset_px=float(np.hypot(x_core - x_beam, y_core - y_beam)),
        beam_radius=beam_radius,
        core_depth=float(smoothed[int(round(y_core)), int(round(x_core))] / a.max()),
    )


def fit_2d_gaussian_to_image(data: np.ndarray) -> FitParams:
    """
    Fit an elliptical 2D Gaussian to an image.

    The initial guess is taken from the image's own moments rather than from the image
    size: a guess of sigma = half the image is far from any real beam, and from there the
    optimizer often walks off to a huge, almost flat Gaussian - which is what makes the
    fitted centre (and the ellipse drawn from it) jump between frames. Moments cost one
    pass over the image and land within a few percent of the answer for a beam that is
    roughly centred, so the fit then only has to polish it.
    """

    height, width = data.shape
    x1 = np.arange(height)
    x2 = np.arange(width)
    x1, x2 = np.meshgrid(x1, x2)
    y = data.ravel()

    # --- initial guess from the image moments ---
    # background: median of a 10% border frame (robust to the beam itself)
    k = max(int(0.1 * min(height, width)), 2)
    offset_0 = float(
        np.median(
            np.concatenate(
                [data[:k].ravel(), data[-k:].ravel(), data[:, :k].ravel(), data[:, -k:].ravel()]
            )
        )
    )
    pos = np.clip(data.astype(float) - offset_0, 0, None)
    amplitude_0 = float(pos.max())
    # weight only the bright part, so that camera background does not pull the centroid
    weights = np.where(pos > 0.5 * amplitude_0, pos, 0.0)
    if weights.sum() <= 0:  # featureless frame - fall back to the image centre
        weights = np.ones_like(pos)
    rows, cols = np.indices(data.shape)
    total = weights.sum()
    y0_0 = float((rows * weights).sum() / total)  # row index
    x0_0 = float((cols * weights).sum() / total)  # column index
    # NOTE: in this parametrization x0 is the column coordinate and y0 the row coordinate
    sigma_x_0 = float(np.sqrt(max((((cols - x0_0) ** 2) * weights).sum() / total, 1.0)))
    sigma_y_0 = float(np.sqrt(max((((rows - y0_0) ** 2) * weights).sum() / total, 1.0)))

    # (amplitude, x0, y0, sigma_x, sigma_y, phi, offset)
    p0 = (amplitude_0, x0_0, y0_0, sigma_x_0, sigma_y_0, 0.0, offset_0)

    # keep the optimizer inside the image: a sigma larger than the frame, or a centre
    # outside it, is never a physical beam - and is exactly what used to be returned
    lower = (0.0, 0.0, 0.0, 0.5, 0.5, -np.pi, -np.inf)
    upper = (np.inf, float(width), float(height), float(width), float(height), np.pi, np.inf)

    return _fit_and_get_param_dict(
        gaussian_2d_fit, (x1, x2), y, p0, curve_fit_kwargs=dict(bounds=(lower, upper))
    )


def fit_lifetime_histogram(xs, ys, meas_type: str, **kwargs):
    """Doc."""
    # TODO: Merge - have 'fit_lifetime_hist' method of 'TDCCalibration' use this.

    if meas_type == "confocal":
        fp = curve_fit_lims(
            exponent_with_background_fit,
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
            sted_hist_fit,
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
    a = (np.cos(phi) ** 2) / (2 * sigma_x**2) + (np.sin(phi) ** 2) / (2 * sigma_y**2)
    b = -(np.sin(2 * phi)) / (4 * sigma_x**2) + (np.sin(2 * phi)) / (4 * sigma_y**2)
    c = (np.sin(phi) ** 2) / (2 * sigma_x**2) + (np.cos(phi) ** 2) / (2 * sigma_y**2)
    g = offset + amplitude * np.exp(
        -(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2))
    )
    return g.ravel()


def linear_fit(t, m, n):
    return m * t + n


def zero_intersect_linear_fit(t, m):
    return m * t


def power_fit(t, a, n):
    return a * t**n


def polynomial_fit(t, *beta):
    """
    Work with polynomial of any degree, i.e.:
    y = beta[0] + beta[1]*t + beta[2]*t**2 + ... + beta[n]*t**n
    """

    amplitude_row_vec = np.array(beta)[:, np.newaxis].T
    power_column_vec = np.array([t**n for n in range(len(beta))])
    return (amplitude_row_vec @ power_column_vec).squeeze()


def diffusion_3d_fit(t, G0, tau, w_sq):
    return G0 / (1 + t / tau) / np.sqrt(1 + t / tau / w_sq)


def stretched_diffusion_3d_fit(t, A, tau, n):
    return A / (1 + (t / tau) ** n)


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


####################################################
## Depletion with a GAUSSIAN beam (vortex plate out)
####################################################
# The donut case is 'sted_hist_fit' above: near a zero the depletion profile is
# parabolic, I ~ A r^2, and averaging it over the detection volume gives the algebraic
# factor 1/(1 + sigma t/tau).
#
# With the plate out the depletion maximum sits ON the detection volume, so the profile
# is Gaussian there, and the average is an incomplete gamma function instead. Writing
# s = t/tau, xi = I0/I_sat = k_-1 tau I0 and a = (w_dep/w_det)^2:
#
#     H(t) = A/tau e^-s Gamma(a) P(a, xi s) / (xi s)^a + bg          [P = gammainc]
#
# which tends to a pure exponential as xi -> 0, and to e^-s (xi s)^-a Gamma(a) at long
# times - the UNDEPLETED lifetime times a power law, because the late photons come from
# the wings where there is no depletion. Do NOT use an inverted parabola for this: it
# goes negative beyond w_dep/sqrt(2), which is gain rather than depletion, and the
# integral diverges.


def gaussian_depletion_hist_fit(t, A, tau, xi, bg, a=1.0):
    """
    Fluorescence decay under a co-aligned Gaussian depletion beam.
    'a' = (w_dep/w_det)^2 is a geometry constant, normally fixed, so it comes last.
    """

    s = np.asarray(t, dtype=float) / tau
    x = np.atleast_1d(xi * s)
    shape = np.full(x.shape, 1 / a)  # the xi*s -> 0 limit
    big = x > 1e-8
    if big.any():
        xb = x[big]
        shape[big] = sp.special.gamma(a) * sp.special.gammainc(a, xb) / xb**a
    shape = shape.reshape(np.shape(s)) if np.shape(s) else shape[0]
    return A / tau * np.exp(-s) * shape + bg


def depletion_sweep_fit(delta, xi_eff, sigma, delta_0, bg):
    """
    The offset sweep: tau0/tau_init - 1 against the relative displacement of the two
    foci. Exactly a Gaussian (only the n = 1 term of the offset series survives in the
    initial decay rate), with

        xi_eff = xi a/(a+1)              -> xi = xi_eff (a+1)/a
        sigma  = sqrt(w_dep^2+w_det^2)/2 -> combined width = 2 sigma

    'delta' and 'delta_0' in the same units (nm); 'bg' catches a constant offset.
    """

    return xi_eff * np.exp(-((np.asarray(delta, dtype=float) - delta_0) ** 2) / (2 * sigma**2)) + bg


def fit_gaussian_depletion_hist(xs, ys, a, tau_ns=None, fit_range=None, **kwargs):
    """
    Fit a lifetime histogram taken with the Gaussian depletion beam on.

    xs - time (ns), ys - counts, 'a' = (w_dep/w_det)^2.
    'tau_ns' fixes the unquenched lifetime (measure it first with depletion OFF, e.g.
    meas.tdc_calib.fit_lifetime_hist().beta["tau"]) - strongly recommended, since tau and
    xi are correlated. 'fit_range' is an (min, max) tuple in ns, as TDCCalibration's own
    fit uses (its default there is (3.5, 30)); without it the fit starts at the peak.
    Pass ys_errors=... and should_weight_fits=True to use error_all_hist_norm.
    """

    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    peak_idx = int(np.argmax(ys))
    if tau_ns is None:

        def gaussian_depletion_hist_free_tau_fit(t, A, tau, xi, bg):
            return gaussian_depletion_hist_fit(t, A, tau, xi, bg, a)

        fit_func = gaussian_depletion_hist_free_tau_fit
        p0 = [ys.max(), 3.5, 1.0, max(ys.min(), 1e-9)]
        bounds = ([0, 0, 0, 0], [np.inf, 20, 1e4, np.inf])
    else:

        def gaussian_depletion_hist_fixed_tau_fit(t, A, xi, bg):
            return gaussian_depletion_hist_fit(t, A, tau_ns, xi, bg, a)

        fit_func = gaussian_depletion_hist_fixed_tau_fit
        p0 = [ys.max(), 1.0, max(ys.min(), 1e-9)]
        bounds = ([0, 0, 0], [np.inf, 1e4, np.inf])

    # NOTE: 'bounds' must go through 'curve_fit_kwargs' - a bare 'bounds=' argument to
    # curve_fit_lims() lands in its **kwargs and never reaches scipy's curve_fit
    return curve_fit_lims(
        fit_func,
        p0,
        xs=xs,
        ys=ys,
        curve_fit_kwargs=dict(bounds=bounds),
        x_limits=Limits(fit_range) if fit_range is not None else Limits(xs[peak_idx], np.inf),
        plot_kwargs=dict(y_scale="log"),
        **kwargs,
    )


def fit_depletion_sweep(offsets_nm, rate_ratios, a, **kwargs) -> Dict[str, Any]:
    """
    Fit the offset sweep.

    offsets_nm   - relative displacement of the two foci (e.g. camera-2 pixels x 20.6 nm)
    rate_ratios  - tau0/tau_init - 1 at each offset (NOT the lifetimes themselves)
    a            - (w_dep/w_det)^2

    Returns the FitParams plus the derived numbers: xi at best overlap, the combined
    width of the two foci, and the best-overlap offset.
    """

    offsets_nm = np.asarray(offsets_nm, dtype=float)
    rate_ratios = np.asarray(rate_ratios, dtype=float)
    fp = curve_fit_lims(
        depletion_sweep_fit,
        [rate_ratios.max(), (offsets_nm.max() - offsets_nm.min()) / 4 or 150.0,
         float(offsets_nm[np.argmax(rate_ratios)]), 0.0],
        xs=offsets_nm,
        ys=rate_ratios,
        curve_fit_kwargs=dict(
            bounds=(
                [0, 1, offsets_nm.min() - 500, -np.inf],
                [np.inf, 1e4, offsets_nm.max() + 500, np.inf],
            )
        ),
        **kwargs,
    )
    xi_eff, sigma = fp.beta["xi_eff"], fp.beta["sigma"]
    return dict(
        fit_params=fp,
        xi=xi_eff * (a + 1) / a,
        xi_error=fp.beta_error["xi_eff"] * (a + 1) / a,
        combined_width_nm=2 * sigma,
        combined_width_error_nm=2 * fp.beta_error["sigma"],
        best_offset_nm=fp.beta["delta_0"],
    )


def initial_decay_time(xs, ys, max_t_ns=None, tau0_ns=None, bg=0.0, order=2) -> Dict[str, Any]:
    """
    Extract the INITIAL decay time (ns) of a depleted histogram - the observable the
    offset-sweep law is written in - without the bias a straight-line slope carries.

    The sweep width is power-free only for the rate at t -> 0: there no molecule has been
    depleted yet, so the rate is linear in the local depletion intensity. A straight line
    through log(H) over a finite window averages over the (strongly non-exponential) early
    decay and so underestimates the dip at zero offset more than in the wings, which pulls
    the fitted sweep width LOW - by ~3% at xi = 2 and ~9% at xi = 15 for a 0.5*tau0 window.
    A quadratic over the same window and its slope AT the peak removes most of that:
    164.4 nm truth -> 164.3 (xi = 0.5), 163.7 (xi = 2), 161 (xi = 5), 153 (xi = 15).
    So run the sweep at a moderate xi (~1-3) and keep the high powers for the power sweep.

    xs - time (ns), ys - counts. The fit starts at the histogram peak and runs for
    'max_t_ns' (default 0.5*tau0_ns, or 2 ns). 'bg' is subtracted first. 'order' 1 gives
    the plain log-slope, for comparison.
    """

    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    peak_idx = int(np.argmax(ys))
    if max_t_ns is None:
        max_t_ns = 0.5 * tau0_ns if tau0_ns else 2.0
    t0 = xs[peak_idx]
    in_window = (xs >= t0) & (xs <= t0 + max_t_ns) & (ys > bg)
    if in_window.sum() < order + 2:
        raise ValueError(f"only {in_window.sum()} usable channels in the fit window")

    t = xs[in_window] - t0
    coeffs = np.polyfit(t, np.log(ys[in_window] - bg), order)
    rate = -coeffs[-2]  # d log(H) / dt at t = 0
    if rate <= 0:
        raise ValueError("non-decaying histogram in the fit window")

    out = dict(
        tau_init_ns=1 / rate,
        rate_per_ns=rate,
        n_channels=int(in_window.sum()),
        window_ns=(float(t0), float(t0 + max_t_ns)),
        coeffs=coeffs,
    )
    if tau0_ns:  # the sweep observable itself
        out["rate_ratio"] = tau0_ns / out["tau_init_ns"] - 1
    return out


def xi_from_initial_rate(tau0_ns, tau_init_ns, a) -> float:
    """xi = I0/I_sat from the unquenched and initial (depleted) decay times."""

    return (tau0_ns / tau_init_ns - 1) * (a + 1) / a


def saturation_intensity(xi, power_w, w_dep_um) -> float:
    """
    I_sat in W/cm^2 from a measured xi, the power at the sample and the depletion focus
    size (1/e^2 radius). The peak intensity of a Gaussian focus is I0 = 2P/(pi w^2).
    """

    peak_intensity = 2 * power_w / (np.pi * (w_dep_um * 1e-4) ** 2)
    return peak_intensity / xi


def donut_predictions(i_sat, power_w, w_dep_um, lambda_dep_nm=592, na=1.2) -> Dict[str, Any]:
    """
    Carry a Gaussian-beam calibration over to the donut.

    An ideal vortex focus of the same total power and width has its crest at 0.368 of the
    Gaussian peak (max of u e^-u), and the resolution follows the usual saturation law.
    'sigma_over_xi_ideal' is the power-free figure of merit: measure sigma with the donut
    and xi with the Gaussian at the SAME power, and their ratio should equal this.
    """

    peak_intensity = 2 * power_w / (np.pi * (w_dep_um * 1e-4) ** 2)
    crest_intensity = np.exp(-1) * peak_intensity  # max of u e^-u, at r = 0.59 w_dep
    confocal_nm = lambda_dep_nm / (2 * na)
    saturation_factor = crest_intensity / i_sat
    return dict(
        gaussian_peak_w_cm2=peak_intensity,
        donut_crest_w_cm2=crest_intensity,
        crest_over_i_sat=saturation_factor,
        confocal_resolution_nm=confocal_nm,
        resolution_nm=confocal_nm / np.sqrt(1 + saturation_factor),
        # power needed (W) for a target resolution, same optics
        power_for_resolution_w=lambda target_nm: power_w
        * ((confocal_nm / target_nm) ** 2 - 1)
        / saturation_factor,
        # the power-free check: measure sigma (donut) and xi (Gaussian) at the same power
        sigma_over_xi_ideal=lambda w_det_um: (w_det_um / w_dep_um) ** 2,
    )
