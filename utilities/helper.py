"""Miscellaneous helper functions/classes"""

import asyncio
import functools
import math
import os
import sys
import time
from contextlib import suppress
from copy import copy
from types import SimpleNamespace
from typing import Any, Callable, List, Tuple, TypeVar

import numpy as np
import scipy
from sklearn import linear_model

# from utilities.display import display.Plotter
import utilities.display as display

# import time # TESTING
# tic = time.perf_counter() # TESTING
# print(f"part 1 timing: {(time.perf_counter() - tic)*1e3:0.4f} ms") # TESTING

EPS = sys.float_info.epsilon
Number = TypeVar("Number", int, float)


class Limits:
    """Doc."""

    def __init__(
        self,
        limits=(np.NINF, np.inf),
        upper=np.inf,
        dict_labels: Tuple[str, str] = None,
        from_string=False,
    ):

        self.dict_labels = dict_labels

        if from_string:
            source_str = limits
            self.lower, self.upper = generate_numbers_from_string(source_str)
        else:
            try:
                self.lower, self.upper = limits
            except ValueError:  # limits is not 2-iterable
                raise TypeError(
                    "Arguments must either be a single value (lower limit) or a 2-iterable"
                )
            except TypeError:  # limits is not iterable
                self.lower, self.upper = limits, upper
                if limits is None:
                    return None
            else:
                if None in limits:
                    return None

    def __call__(self, *args, **kwargs):
        self.__init__(*args, **kwargs)

    def __repr__(self):
        return f"Limits(lower={self.lower}, upper={self.upper})"

    def __str__(self):
        lower_frmt = ".2f"
        with suppress(OverflowError):
            if int(self.lower) == float(self.lower):
                lower_frmt = "d"
                self.lower = int(self.lower)  # ensure for stuff like 1e3 (round floats)

        upper_frmt = ".2f"
        with suppress(OverflowError):
            if int(self.upper) == float(self.upper):
                upper_frmt = "d"
                self.upper = int(self.upper)  # ensure for stuff like 1e3 (round floats)

        return f"({self.lower:{lower_frmt}}, {self.upper:{upper_frmt}})"

    def __iter__(self):
        yield from (self.lower, self.upper)

    def __getitem__(self, idx):
        return tuple(self)[idx]

    def __len__(self):
        return 2

    def __and__(self, other):
        self = self if self is not None else Limits()
        other = other if other is not None else Limits()
        lower = max(self.lower, other.lower)
        upper = min(self.upper, other.upper)
        return Limits(lower, upper)

    def __eq__(self, other):
        try:
            return tuple(self) == other
        except TypeError:
            raise TypeError("Can only compare Limits to other instances or tuples")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return other < self.lower
        if isinstance(other, Limits):
            return other.upper < self.lower

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return other > self.upper
        if isinstance(other, Limits):
            return other.lower > self.upper

    def __contains__(self, other):
        """
        Checks if 'other' is in 'Limits'.
        If:
        other is a tuple/Limits: checks if full range is contained and returns bool
        other is number: checks if number is contained in range and returns bool
        """
        try:
            if len(other) == 2:
                return (self[0] <= other[0]) and (self[1] >= other[1])
        except TypeError:  # other is not 2-iterable
            try:
                return self.lower <= other <= self.upper
            except TypeError:  # other is not a number
                if other is None:
                    return False
                else:
                    raise TypeError(
                        "Can only compare Limits to other instances or 2-iterable objects."
                    )

    def __add__(self, other: Number):
        if isinstance(other, (int, float)):
            self_copy = copy(self)
            self_copy.lower += other
            self_copy.upper += other
            return self_copy

    def __mul__(self, other: Number):
        if isinstance(other, (int, float)):
            self_copy = copy(self)
            self_copy.lower *= other
            self_copy.upper *= other
            return self_copy

    __rmul__ = __mul__

    #    def __bool__(self):
    #        return not ((self.lower == 0 or self.lower == np.NINF) and (self.upper == np.inf))

    def valid_indices(self, arr: np.ndarray, as_bool=True):
        """
        Returns indices of array elements within the limits.
        Optionally, can return a boolean array of same shape instead.
        """
        if isinstance(arr, np.ndarray):
            if as_bool:
                return (arr >= self.lower) & (arr <= self.upper)
            else:
                return np.nonzero((arr >= self.lower) & (arr <= self.upper))[0]
        else:
            raise TypeError("Argument 'arr' must be a Numpy ndarray!")

    def as_dict(self):
        if self.dict_labels is not None:
            return {key: val for key, val in zip(self.dict_labels, (self.lower, self.upper))}
        else:
            return {key: val for key, val in zip(("lower", "upper"), (self.lower, self.upper))}

    def interval(self, upper_max=np.inf):
        return abs(min(self.upper, upper_max) - self.lower)

    def center(self):
        """Get the center of the range"""

        return sum(self) * 0.5

    def clamp(self, obj):
        """Force limit range on object"""

        if isinstance(obj, (int, float)):
            return max(min(self.upper, obj), self.lower)
        elif hasattr(obj, "lower") and hasattr(obj, "upper"):
            return Limits(self.clamp(obj.lower), self.clamp(obj.upper))
        else:
            raise TypeError("Clamped object must be either a Limits instance or a number!")

    def as_range(self) -> range:
        """Get a Python 'range' (generator)"""

        return range(self.lower, self.upper)


class Gate(Limits):
    """Convenience class for defining time-gates in ns using the Limits class"""

    def __init__(self, *args, is_hard: bool = False, hard_gate=None, units: str = "ns", **kwargs):
        if not args:
            args = (0, np.inf)
        super().__init__(*args, **kwargs)  # initialize self as Limits
        self.hard_gate = self if is_hard else hard_gate
        self.units = units

        if self.lower < 0:
            raise ValueError(f"Gating limits {self} must be between 0 and positive infinity.")

        if is_hard and self.upper == np.inf:
            raise ValueError("Hardware gating must have a finite upper limit.")

        if self.lower > self.upper:
            raise ValueError(
                f"Lower limit ({self.lower}) must be lower than upper limit ({self.upper})"
            )

    def __bool__(self):
        return not ((self.lower == 0 or self.lower == np.NINF) and (self.upper == np.inf))

    def __repr__(self):
        return super().__repr__().replace("Limits", "Gate")

    def __str__(self):
        return f"{super().__str__()}{f' {self.units}' if self.units else ''}"

    def __and__(self, other):
        self = self if self is not None else Gate()
        other = other if other is not None else Gate()
        lower = max(self.lower, other.lower)
        upper = min(self.upper, other.upper)
        hard_gate = None
        if self.hard_gate:
            hard_gate = self.hard_gate
        elif other.hard_gate:
            hard_gate = other.hard_gate

        return Gate(lower, upper, hard_gate=hard_gate)


def nan_helper(y):
    """
    Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> y = np.array([1.0 , 2.0, np.nan, 4.0])
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    Adapted from:
    https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def largest_n(arr: np.ndarray, n: int):
    """
    Adapted from:
    https://stackoverflow.com/questions/10337533/a-fast-way-to-find-the-largest-n-elements-in-an-numpy-array
    """

    sz = arr.size
    perc = (np.arange(sz - n, sz) + 1.0) / sz * 100
    return np.percentile(arr, perc, interpolation="nearest")


def timer(threshold_ms: float = 0.0) -> Callable:
    """
    Meant to be used as a decorator (@helper.timer(threshold))
    for quickly setting up function timing for testing.
    Works for both regular and asynchronous functions.
    NOTE - asynchronous function timing may include stuff that happens
        while function 'awaits' other coroutines.
    """

    def outer_wrapper(func) -> Callable:
        """Doc."""

        if asyncio.iscoroutinefunction(func):
            # timing async funcitons
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                tic = time.perf_counter()
                value = await func(*args, **kwargs)
                toc = time.perf_counter()
                elapsed_time_ms = (toc - tic) * 1e3
                if elapsed_time_ms > threshold_ms:
                    in_s = elapsed_time_ms > 1000
                    print(
                        f"***TIMER*** Function '{func.__name__}()' took {elapsed_time_ms * (1e-3 if in_s else 1):.2f} {'s' if in_s else 'ms'} (threshold: {threshold_ms * (1e-3 if in_s else 1):.0f} {'s' if in_s else 'ms'}).\n"
                    )
                return value

        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                tic = time.perf_counter()
                value = func(*args, **kwargs)
                toc = time.perf_counter()
                elapsed_time_ms = (toc - tic) * 1e3
                if elapsed_time_ms > threshold_ms:
                    in_s = elapsed_time_ms > 1000
                    print(
                        f"***TIMER*** Function '{func.__name__}()' took {elapsed_time_ms * (1e-3 if in_s else 1):.2f} {'s' if in_s else 'ms'} (threshold: {threshold_ms * (1e-3 if in_s else 1):.0f} {'s' if in_s else 'ms'}).\n"
                    )
                return value

        return wrapper

    return outer_wrapper


# TODO: this needs testing - not working as expected! copy it to Jupyter and test there using
# saved images
def my_threshold(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Applies custom thresholding (idomic) to an image.
    Returns the thresholded image and the threshold.
    """

    n, bin_edges = np.histogram(img.ravel())
    thresh_idx = 1
    for i in range(1, len(n)):
        if n[i] <= n.max() * 0.1:
            if n[i + 1] >= n[i] * 10:
                continue
            else:
                thresh_idx = i
                break
        else:
            continue
    thresh = (bin_edges[thresh_idx] - bin_edges[thresh_idx - 1]) / 2
    img[img < thresh] = 0

    return img, thresh


def moving_average(a: np.ndarray, n: int = 3) -> np.ndarray:
    """
    Adapted from:
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    """

    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def return_outlier_indices(data, m=2.0):
    """
    Adapted from:
    https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.0
    return s >= m


def robust_interpolation(
    x,  # x-values to interpolate onto
    xi,  # real x vals
    yi,  # real y vals
    n_pnts=3,  # number of points to include in each intepolation
) -> np.ndarray:
    """Doc."""

    # translated from: [h, bin] = histc(x, np.array([-np.inf, xi, inf]))
    bins = np.hstack(([-np.inf], xi, [np.inf]))
    (x_bin_counts, _), x_bin_idxs = np.histogram(x, bins), np.digitize(x, bins)
    ch = np.cumsum(x_bin_counts, dtype=np.uint16)
    start = max(x_bin_idxs[0] - 1, n_pnts)
    finish = min(x_bin_idxs[x.size - 1] - 1, xi.size - n_pnts - 1)

    y = np.empty(shape=x.shape)
    ransac = linear_model.RANSACRegressor()
    for i in range(start, finish):

        # Robustly fit linear model with RANSAC algorithm for (1 + 2*n_pnts) points
        ji = slice(i - n_pnts, i + n_pnts + 1)
        ransac.fit(xi[ji][:, np.newaxis], yi[ji])
        p0, p1 = ransac.estimator_.intercept_, ransac.estimator_.coef_[0]

        if i == start:
            j = slice(ch[i + 1] + 1)
        elif i == finish - 1:
            j = slice((ch[i] + 1), x.size)
        else:
            j = slice((ch[i] + 1), ch[i + 1] + 1)

        y[j] = p0 + p1 * x[j]

    return y


def extrapolate_over_noise(
    x,
    y,
    x_interp=None,
    x_lims: Limits = Limits(np.NINF, np.inf),
    y_lims: Limits = Limits(np.NINF, np.inf),
    n_bins=2 ** 17,
    n_robust=0,
    interp_type="linear",  # gaussian
    extrap_x_lims=Limits(np.NINF, np.inf),
    should_plot=False,
) -> SimpleNamespace:
    """Doc."""

    # unify length of y to x (assumes y decays to zero)
    y = unify_length(y, len(x))

    if x_interp is None:
        extrap_x_lims = extrap_x_lims.clamp(Limits(min(x), max(x)))
        initial_x_interp = np.linspace(*extrap_x_lims, n_bins)
    else:
        initial_x_interp = x_interp

    # choose interpolation range (extrapola.te the noisy parts)
    interp_idxs = sorted(
        set(x_lims.valid_indices(x, as_bool=False))
        & set(y_lims.valid_indices(y, as_bool=False)[:-1])
    )

    x_samples = x[interp_idxs]
    y_samples = y[interp_idxs]

    if interp_type == "gaussian":
        x_interp = initial_x_interp ** 2
        x_samples = x_samples ** 2
        y_samples = np.log(y_samples)
    elif interp_type == "linear":
        x_interp = initial_x_interp

    if n_robust:
        y_interp = robust_interpolation(x_interp, x_samples, y_samples, n_robust)
    else:
        interpolator = scipy.interpolate.interp1d(
            x_samples,
            y_samples,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        y_interp = interpolator(x_interp)

    if interp_type == "gaussian":
        y_interp = np.exp(y_interp)
    elif interp_type == "linear":
        #  zero-pad the linear interpolation
        y_interp[x_interp > x[y_lims.valid_indices(y, as_bool=False)[-1]]] = 0

    if should_plot:
        with display.Plotter(
            subplots=(1, 2),
            super_title=f"{interp_type.capitalize()} Interpolation Testing",
            xlabel="$x$",
            ylabel="$y$",
            ylim=(-abs(np.median(y)) * 1.3, np.median(largest_n(y, 5)) * 1.3),
        ) as axes:
            axes[0].plot(x, y, "o", label="before")
            axes[0].plot(initial_x_interp, y_interp, ".", markersize="4", label="after")
            axes[0].legend()

            axes[1].plot(x, y, "o", label="before")
            axes[1].plot(initial_x_interp, y_interp, ".", markersize="4", label="after")
            axes[1].legend()
            axes[1].set_xscale("log")

    return SimpleNamespace(
        x_interp=initial_x_interp,
        y_interp=y_interp,
        x_samples=x_samples,
        y_samples=y_samples,
        interp_idxs=interp_idxs,
    )


def fourier_transform_1d(
    x: np.ndarray,
    y: np.ndarray,
    should_inverse: bool = False,
    is_input_symmetric=False,
    n_bins=2 ** 17,
    bin_size=None,
    should_normalize: bool = False,
    should_plot=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Doc."""

    # Use only positive half to ensure uniform treatment
    if is_input_symmetric:
        x = x[x.size // 2 :]
        y = y[y.size // 2 :]

    # save the zero value for later
    y0 = y[0]

    # interpolate to ensure equal spacing and set bins_size or number of bins
    if bin_size:  # in case the bin size is set
        r = np.arange(min(x), max(x), bin_size)

    else:  # in case the number of bins is set instead
        r = np.linspace(min(x), max(x), n_bins // 2)
        bin_size = r[1] - r[0]

    # Linear interpolation
    interpolator = scipy.interpolate.interp1d(
        x,
        y,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    fr_interp = interpolator(r)

    n_bins = r.size * 2 - 1
    q = 2 * np.pi * np.arange(n_bins) / (bin_size * n_bins)

    # normalization
    if should_normalize:
        fr_interp *= bin_size
        y0 = 1

    # Return the center (0) point
    fr_interp[0] = y0

    # Symmetrize and transform
    r = np.hstack((-np.flip(r[1:]), r))

    if should_inverse:  # Inverse FT
        fixed_for_fft_fr_interp = np.hstack((fr_interp, np.flip(fr_interp[1:])))
        fr_interp = np.hstack((np.flip(fr_interp[1:]), fr_interp))
        fq = scipy.fft.ifft(fixed_for_fft_fr_interp)

    else:  # FT
        fr_interp = np.hstack((np.flip(fr_interp[1:]), fr_interp))
        fq = scipy.fft.fft(fr_interp)
        fq *= np.exp(-1j * q * (min(r)))

    # reorder q and fq (switch first and second halves. shift q)
    k1 = int(n_bins / 2 + 0.9)  # trick to deal with even and odd values of 'n'
    k2 = math.ceil(n_bins / 2)  # + 1)
    q = np.hstack((q[k2:] - 2 * np.pi / bin_size, q[:k1]))
    fq = np.hstack((fq[k2:], fq[:k1]))

    if should_plot:
        with display.Plotter(
            subplots=(2, 2),
            super_title=f"{'Inverse ' if should_inverse else ''}Fourier Transform",
        ) as axes:
            axes[0][0].plot(
                x, np.real(y) * (bin_size if should_normalize else 1), "o", label="before"
            )
            axes[0][0].plot(r, np.real(fr_interp), "x", label="after interpolation")
            axes[0][0].legend()
            axes[0][0].set_xlim(r[r.size // 2 - 10], r[r.size // 2 + 10]),
            axes[0][0].set_title("Interpolation")
            axes[0][0].set_xlabel("$r$")
            axes[0][0].set_ylabel("$f(r)$")

            axes[0][1].plot(
                x,
                np.real(y) * (bin_size if should_normalize else 1),
                "o",
                label="before interpolation",
            )
            axes[0][1].plot(r, np.real(fr_interp), "x", label="after interpolation")
            axes[0][1].legend()
            axes[0][1].set_ylim(0, max(np.real(fr_interp)[fr_interp.size // 2 + 1 :])),
            axes[0][1].set_xscale("log"),
            axes[0][1].set_title("Interpolation")
            axes[0][1].set_xlabel("$r$")
            axes[0][1].set_ylabel("$f(r)$")

            axes[1][0].plot(q, np.real(fq), label="real part")
            axes[1][0].plot(q, np.imag(fq), label="imaginary part")
            axes[1][0].plot(q, abs(fq), label="absolute")
            axes[1][0].legend()
            axes[1][0].set_xlim(q[q.size // 2 - 1000], q[q.size // 2 + 1000]),
            axes[1][0].set_title("Transform")
            axes[1][0].set_xlabel("$q$ $(2\\pi\\cdot[r^{-1}])$")
            axes[1][0].set_ylabel("$f(q)$")

            axes[1][1].plot(q, np.real(fq), label="real part")
            axes[1][1].legend()
            axes[1][1].set_ylim(
                min(np.real(fq[fq.size // 2 :])), max(np.real(fq[fq.size // 2 :])) * 1.1
            ),
            axes[1][1].set_xscale("log"),
            axes[1][1].set_title("Transform")
            axes[1][1].set_xlabel("$q$ $(2\\pi\\cdot[r^{-1}])$")
            axes[1][1].set_ylabel("$f(q)$")

    return r, fr_interp, q, fq


def hankel_transform(
    x: np.ndarray,
    y: np.ndarray,
    should_inverse: bool = False,
    n_robust: int = 3,
    should_do_gaussian_interpolation: bool = False,
    dr=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Doc."""

    n = x.size

    # prepare the Hankel transformation matrix C
    c0 = scipy.special.jn_zeros(0, n)
    bessel_j0 = scipy.special.j0
    bessel_j1 = scipy.special.j1

    j_n, j_m = np.meshgrid(c0, c0)

    C = (
        (2 / c0[n - 1])
        * bessel_j0(j_n * j_m / c0[n - 1])
        / (abs(bessel_j1(j_n)) * abs(bessel_j1(j_m)))
    )

    if not should_inverse:

        r_max = max(x)
        q_max = c0[n - 1] / (2 * np.pi * r_max)  # Maximum frequency

        r = c0.T * r_max / c0[n - 1]  # Radius vector
        q = c0.T / (2 * np.pi * r_max)  # Frequency vector

        m1 = (abs(bessel_j1(c0)) / r_max).T  # m1 prepares input vector for transformation
        m2 = m1 * r_max / q_max  # m2 prepares output vector for display

        # end preparations for Hankel transform

        if n_robust:  # use robust interpolation
            if should_do_gaussian_interpolation:  # Gaussian
                fr_interp = np.exp(robust_interpolation(r ** 2, x ** 2, np.log(y), n_robust))
                fr_interp[
                    r > x[-n_robust]
                ] = 0  # zero pad last points that do not have full interpolation
            else:  # linear
                fr_interp = robust_interpolation(r, x, y, n_robust)

            fr_interp = fr_interp.ravel()

        else:
            if should_do_gaussian_interpolation:
                interpolator = scipy.interpolate.interp1d(
                    x ** 2,
                    np.log(y),
                    fill_value="extrapolate",
                    assume_sorted=True,
                )
                fr_interp = np.exp(interpolator(r ** 2))
            else:  # linear
                interpolator = scipy.interpolate.interp1d(
                    x,
                    y,
                    fill_value="extrapolate",
                    assume_sorted=True,
                )
                fr_interp = interpolator(r)

        return 2 * np.pi * q, C @ (fr_interp / m1) * m2

    else:  # inverse transform

        if dr is not None:
            q_max = 1 / (2 * dr)
        else:
            q_max = max(x) / (2 * np.pi)

        r_max = c0[n - 1] / (2 * np.pi * q_max)  # Maximum radius

        r = c0.T * r_max / c0[n - 1]  # Radius vector
        q = c0.T / (2 * np.pi * r_max)  # Frequency vector
        q = 2 * np.pi * q

        m1 = (abs(bessel_j1(c0)) / r_max).T  # m1 prepares input vector for transformation
        m2 = m1 * r_max / q_max  # m2 prepares output vector for display
        # end preparations for Hankel transform

        interpolator = scipy.interpolate.interp1d(
            x,
            y,
            fill_value="extrapolate",
            assume_sorted=True,
        )

        return r, C @ (interpolator(q) / m2) * m1


def unify_length(vec_in: np.ndarray, out_len: int) -> np.ndarray:
    """Either trims or zero-pads the tail of a 1D array to match 'out_len'"""

    if len(vec_in) >= out_len:
        return vec_in[:out_len]
    else:
        return np.hstack((vec_in, np.zeros(out_len - len(vec_in))))


def xcorr(a, b):
    """Does correlation similar to Matlab xcorr, cuts positive lags, normalizes properly"""

    c = np.correlate(a, b, mode="full")
    c = c[c.size // 2 :]
    c = c / np.arange(c.size, 0, -1)
    lags = np.arange(c.size, dtype=np.uint16)

    return c, lags


def can_float(value: Any) -> bool:
    """Checks if 'value' can be turned into a float"""

    try:
        float(value)
        return True
    except (ValueError, TypeError):
        if value == "-":  # consider hyphens part of float (minus sign)
            return True
        return False


def str_to_num(x):
    """Attempts to convert 'x' into an integer, or a float if that fails."""

    try:
        return int(x)
    except (ValueError, OverflowError):
        return float(x)


def generate_numbers_from_string(source_str):
    """A generator function for getting numbers out of strings."""

    i = 0
    while i < len(source_str):
        j = i + 1
        while (j < len(source_str) + 1) and can_float(source_str[i:j]):
            j += 1
        with suppress(TypeError, ValueError):
            yield str_to_num(source_str[i : j - 1])
        i = j


def center_of_mass(arr: np.ndarray) -> Tuple[float, ...]:
    """Returns the center of mass coordinates of a Numpy ndarray as a tuple"""

    def center_of_mass_of_dimension(arr: np.ndarray, dim: int = 0) -> float:
        """
        Returns the center of mass of an Numpy along a specific axis.
        Default axis is 0 (easier calling for 1d arrays)

        Based on the COM formula: 1/M * \\Sigma_i (m_i * x_i)
        """

        total_mass = arr.sum()
        displacements = np.arange(arr.shape[dim])
        masses_at_displacements = np.atleast_2d(arr).sum(axis=dim)
        return 1 / total_mass * np.dot(displacements, masses_at_displacements)

    return tuple(center_of_mass_of_dimension(arr, dim) for dim in range(len(arr.shape)))


def list_to_file(file_path, lines: List[str]) -> None:
    """Accepts a list of strings 'lines' and writes them to 'file_path'."""

    with open(file_path, "w") as f:
        f.write("\n".join(lines))


def file_to_list(file_path) -> List[str]:
    """Doc."""

    with open(file_path, "r") as f:
        return f.read().splitlines()


def bool_str(str_: str):
    """A strict bool() for strings"""

    if str_ == "True":
        return True
    elif str_ == "False":
        return False
    else:
        raise ValueError(f"'{str_}' is neither 'True' nor 'False'.")


def deep_getattr(obj, deep_attr_name: str, default=None):
    """
    Get deep attribute of obj. Useful for dynamically-set deep attributes.
    Example usage: a = deep_getattr(obj, "sobj.ssobj.a")
    """

    for attr_name in deep_attr_name.split("."):
        obj = getattr(obj, attr_name, default)
    return obj


def div_ceil(x: int, y: int) -> int:
    """Returns x divided by y rounded towards positive infinity"""

    return int(-(-x // y))


def reverse_dict(dict_: dict) -> dict:
    """Return a new dict for which keys and values are switched"""

    return {val: key for key, val in dict_.items()}


def translate_dict_values(original_dict: dict, trans_dict: dict) -> dict:
    """
    Updates values of dict according to another dict:
    val_trans_dct.keys() are the values to update,
    and val_trans_dct.values() are the new values.
    """

    return {
        key: (trans_dict[val] if val in trans_dict.keys() else val)
        for key, val in original_dict.items()
    }


def file_last_line(file_path) -> str:
    """
    Return the last line of a text file, quickly (seeks from end)
    (https://stackoverflow.com/questions/46258499/read-the-last-line-of-a-file-in-python)
    """

    try:
        with open(file_path, "rb") as f:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode()
    except OSError:
        return None
    else:
        return last_line


def dir_date_parts(data_path: str, sub_dir: str = "", month: int = None, year: int = None) -> list:
    """
    Inputs:
        data_path - string containing the path to the main data folder.
        month - integer month to search for. if None, will return months matching the year.
        year - integer year to search for. If None, will return years of all subfolders.

    Returns:
        a sorted list of strings containing all relevant dates parts in the folder.

    Examples:
        say in folder 'main_data_path' we have the folders: 11_01_2019, 15_01_2019, 20_02_2019, 05_08_2018.
        get_folder_dates(main_data_path, month=1, year=2019) will return ['11', '15'].
        get_folder_dates(main_data_path, year=2019) will return ['1', '2'].
        get_folder_dates(main_data_path) will return ['2018', '2019'].

    """

    if month and year is None:
        raise ValueError("Month was supplied while year was not.")

    # list all non-empty directories in 'data_path'
    dir_name_list = [
        item
        for item in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, item, sub_dir))
        and os.listdir(os.path.join(data_path, item, sub_dir))
    ]

    dir_date_dict_list = [
        {
            date_key: date_val.lstrip("0")
            for date_key, date_val in zip(("day", "month", "year"), dir_name.split("_"))
        }
        for dir_name in dir_name_list
    ]

    # get matching days
    if year and month:
        date_item_list = [
            dir_date_dict["day"]
            for dir_date_dict in dir_date_dict_list
            if (dir_date_dict.get("month") == month) and (dir_date_dict.get("year") == year)
        ]

    # get matching months
    elif year:
        date_item_list = [
            dir_date_dict["month"]
            for dir_date_dict in dir_date_dict_list
            if dir_date_dict.get("year") == year
        ]

    # get all existing years
    else:
        date_item_list = [
            dir_date_dict.get("year")
            for dir_date_dict in dir_date_dict_list
            if dir_date_dict.get("year") is not None
        ]

    # return unique date parts, sorted in descending order
    return sorted(set(date_item_list), key=lambda item: int(item), reverse=True)
