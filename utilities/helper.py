"""Miscellaneous helper functions/classes"""

from __future__ import annotations

import asyncio
import functools
import math
import os
import sys
import time
from collections import Counter
from contextlib import suppress
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, List, Tuple, TypeVar

import numpy as np
import scipy
from sklearn import linear_model
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import utilities.display as display

# import time # TESTING
# tic = time.perf_counter() # TESTING
# print(f"part 1 timing: {(time.perf_counter() - tic)*1e3:0.4f} ms") # TESTING

EPS = sys.float_info.epsilon
Number = TypeVar("Number", int, float)


@dataclass
class Vector:
    """Doc."""

    _x: float
    _y: float
    units: str

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __repr__(self):
        return f"Vector(x={self.x:.2f}, y={self.y:.2f}, units={self.units})"

    def __iter__(self):
        yield from (self.x, self.y)

    def __getitem__(self, idx):
        return tuple(self.x, self.y)[idx]

    def __len__(self):
        return 2

    def __neg__(self):
        return Vector(-self.x, -self.y, self.units)

    def __add__(self, other):
        if self.units == other.units:
            return Vector(self.x + other.x, self.y + other.y, self.units)
        else:
            raise TypeError(f"Vectors are of different units! ({self.units}, {other.units})")

    def __sub__(self, other):
        if self.units == other.units:
            return Vector(self.x - other.x, self.y - other.y, self.units)
        else:
            raise TypeError(f"Vectors are of different units! ({self.units}, {other.units})")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector(self.x * other, self.y * other, self.units)
        else:
            raise NotImplementedError(f"Can only multiply {type(self)} with scalars, for now")

    __rmul__ = __mul__

    def __eq__(self, other):
        try:
            if isinstance(other, tuple) or self.units == other.units:
                try:
                    return tuple(self) == other
                except TypeError:
                    raise TypeError("Can only compare Vectors to other instances or tuples")
            else:
                raise TypeError(f"Vectors are of different units! ({self.units}, {other.units})")
        except AttributeError:
            # other has no 'units'
            raise TypeError("Can only compare Vectors to other instances or tuples")

    def __round__(self, ndigits=None):
        return Vector(round(self.x, ndigits=ndigits), round(self.y, ndigits=ndigits), self.units)

    def convert_units(self, mul_factor: float, units_str: str) -> Vector:
        """Return a units-converted copy of a Vector, based on supplied multiplication factor"""

        return Vector(self.x * mul_factor, self.y * mul_factor, units_str)


class Limits:
    """Doc."""

    def __init__(
        self,
        limits=(-np.inf, np.inf),
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
                    raise ValueError("None is not a valid Limits argument!")
            else:
                if None in limits:
                    raise ValueError("None is not a valid Limits argument!")

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
        else:
            raise TypeError("'other' must be a scalar")

    def __mul__(self, other: Number):
        if isinstance(other, (int, float)):
            self_copy = copy(self)
            self_copy.lower *= other
            self_copy.upper *= other
            return self_copy
        else:
            raise TypeError("'other' must be a scalar")

    __rmul__ = __mul__

    def __pow__(self, other: Number):
        self_copy = copy(self)
        self_copy.lower = self.lower**other
        self_copy.upper = self.upper**other
        return self_copy

    def __round__(self, ndigits=None):
        self_copy = copy(self)
        self_copy.lower = round(self.lower, ndigits=ndigits)
        self_copy.upper = round(self.upper, ndigits=ndigits)
        return self_copy

    def __bool__(self):
        return not ((self.lower == -np.inf) and (self.upper == np.inf))

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
            try:
                arr = tuple(arr)
            except TypeError:
                raise TypeError(f"Argument 'arr' (type {type(arr)}) must be iterable!")
            else:
                if not as_bool:
                    raise NotImplementedError(
                        f"Can only return boolean values for 'arr' input of type {type(arr)}."
                    )
                else:
                    return [self.lower <= elem <= self.upper for elem in arr]

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
        """Get a Python 'range' (generator), rounding first to have integers"""

        return range(round(self.lower), round(self.upper))


class Gate(Limits):
    """
    Convenience class for defining time-gates in ns using the Limits class.
    Note that 'hard gates' are not the same as TDC gates; lower/upper values
    of a hard gate represent actual times while TDC values must be added the pulse
    delay time to match the hard gate.
    """

    # TODO: perhaps I should change this class such that tdc gates and hard gates are attributes (inheriting Limits)?
    # This would have to involve proper handling of older measurements!

    def __init__(self, *args, hard_gate=None, units: str = "ns", **kwargs):
        if not args:
            args = (0, np.inf)
        super().__init__(*args, **kwargs)  # initialize self as Limits
        self.hard_gate = Gate(hard_gate) if hard_gate else None
        self.units = units

        if self.lower < 0:
            print(
                "WARNING: LOWER GATE IS LOWER THAN 0! CLIPPING TO ZERO IN CASE THIS WAS INTERNTIONAL"
            )
            self.lower = 0  # TESTESTEST
        #            raise ValueError(f"Gating limits {self} must be between 0 and positive infinity.")

        if self.hard_gate is not None and self.hard_gate.upper == np.inf:
            raise ValueError("Hardware gating must have a finite upper limit.")

        if self.lower > self.upper:
            raise ValueError(
                f"Lower limit ({self.lower}) must be lower than upper limit ({self.upper})"
            )

    def __bool__(self):
        return not ((self.lower == 0) and (self.upper == np.inf))

    def __repr__(self):
        return f"Gate(lower={self.lower}, upper={self.upper}" + (
            f", hard_gate={self.hard_gate})" if self.hard_gate is not None else ")"
        )

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


@dataclass
class InterpExtrap1D:
    """Holds the results of some one-dimensional interpolation/extrapolation."""

    interp_type: str
    x_interp: np.ndarray
    y_interp: np.ndarray
    x_sample: np.ndarray
    y_sample: np.ndarray
    x_data: np.ndarray  # original data
    y_data: np.ndarray  # original data
    interp_idxs: np.ndarray
    x_lims: Limits

    def plot(self, label_prefix="", **kwargs):
        """Display the interpolation."""

        # TODO: once hierarchical plotting is applied, use the first (confocal) max(self.x_data[self.interp_idxs]) as the upper x limit for plotting
        #        kwargs["xlim"] = kwargs.get("xlim", (0, max(self.x_data[self.interp_idxs])))
        kwargs["xlim"] = kwargs.get("xlim", (0, 1))  # TESTESTEST - temporary fix
        kwargs["ylim"] = kwargs.get("ylim", (5e-3, 1.3))

        with display.Plotter(
            super_title=f"{self.interp_type.capitalize()} Interpolation",
            xlabel="$x$",
            ylabel="$y$",
            **kwargs,
        ) as ax:
            line2d = ax.plot(
                self.x_data,
                self.y_data,
                "o",
                label=f"_{label_prefix}data",
                alpha=0.3,
                markerfacecolor="none",
            )
            color = line2d[0].get_color()  # get the color from the first plotted line
            ax.plot(
                self.x_sample,
                self.y_sample,
                "o",
                label=f"_{label_prefix}sample",
                color=color,
                markerfacecolor="none",
            )
            ax.plot(
                self.x_interp,
                self.y_interp,
                "-",
                markersize=4,
                label=label_prefix,
                color=color,
            )
            ax.axvline(x=self.x_lims.lower, color=color, lw=1, ls="--")
            ax.axvline(x=self.x_lims.upper, color=color, lw=1, ls="--")
            ax.legend()


def exclude_elements_by_indices_1d(input_arr, idxs_to_exclude):
    """Doc."""
    # TODO: look for lines where I filter indices using list comprehensions and replace with this function

    all_idxs = np.arange(len(input_arr))
    idxs_to_keep = np.setdiff1d(all_idxs, idxs_to_exclude)
    return input_arr[idxs_to_keep]


def normalize_scan_img_rows(img: np.ndarray, mask=None):
    """Normalize an image to the median of the maximum row. Optionally use a supplied mask first."""

    if mask is None:
        mask = np.full(img.shape, True, dtype=np.bool)
        mask[-1, :] = False  # last row is always empty

    temp_img = img.copy().astype(np.float64)
    temp_img[~mask] = 0
    max_row = np.argmax(temp_img.sum(axis=1))
    max_row_median = np.median(img[max_row][mask[max_row]])
    norm_masked_img = img.astype(np.float64)
    for row_idx in np.unique(norm_masked_img.nonzero()[0]):
        if mask[row_idx].any():
            norm_masked_img[row_idx] *= max_row_median / np.median(img[row_idx][mask[row_idx]])
    return norm_masked_img


def moving_average(arr, window_size, keep_size=True):
    """
    Calculate the moving average of a 1D NumPy array.

    Parameters:
        arr (numpy.ndarray): The input array.
        window_size (int): The size of the moving window.
        keep_size (bool): If True, return an interpolated version of the averaged array to keep the same size.

    Returns:
        numpy.ndarray: The moving average of the input array with the same size (or interpolated if keep_size is True).
    """

    cumsum = np.cumsum(arr)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    avg_arr = cumsum[window_size - 1 :] / window_size

    if keep_size:
        return np.interp(np.arange(len(arr)), np.arange(window_size - 1, len(arr)), avg_arr)
    else:
        return avg_arr


def most_common(list_):
    """
    Returns the most common element in a list/array.
    Adapted from: https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
    """

    data = Counter(list_)
    try:
        return data.most_common(1)[0][0]
    except IndexError:
        # list was empty
        return None


def dbscan_noise_thresholding(
    X, min_noise_thresh=0.25, eps0=0.5, eps_inc=0.1, max_tries=500, label="", **kwargs
):
    """Doc."""

    # Standardize the data
    X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + EPS)

    eps = eps0
    for _ in range(max_tries):
        clustering = DBSCAN(eps=eps, min_samples=5)
        y_clusters = clustering.fit_predict(X_std)
        dbscan_noise_mask = y_clusters == -1
        largest_cluster_label = most_common(y_clusters[y_clusters != -1])
        # calculate the noise percentage out of the largest cluster
        noise_perc = sum(dbscan_noise_mask) / sum(
            dbscan_noise_mask | (y_clusters == largest_cluster_label)
        )
        #        print(f"Percentage of 'noise' points: {noise_perc:.2%}")
        if noise_perc <= min_noise_thresh:
            #         print(f"Selected eps={eps:.2f}")
            break
        else:
            eps += eps_inc

    # get the largest cluster label. the rest of the clusters will be considered part of the "noise"
    noise_mask = (y_clusters == -1) | (y_clusters != largest_cluster_label)

    # do PCA to view results
    if kwargs.get("should_plot"):
        pca = PCA(n_components=2)
        X_std_pca = pca.fit_transform(X_std)
        # display the PCA (2D) using the cluster labels
        display.display_dim_reduction(X_std_pca, noise_mask, f"{label}\nPCA", figsize=(8, 6))

    return noise_mask


def batch_mean_rows(arr: np.ndarray, batch_size: int = None, batch_sizes: List[int] = None):
    """
    Compute the mean of batches of rows from a 2D numpy array.

    This function takes a 2D numpy array and an integer `n_rows` as input. It divides the array's rows
    into approximately equal-sized batches and computes the mean along each column for each batch.
    The resulting mean values are stacked vertically to form a new 2D numpy array.

    Parameters:
        arr (np.ndarray): A 2D numpy array with shape (num_rows, num_columns).
        n_rows (int): The desired number of batches for computing the mean.

    Returns:
        np.ndarray: A new 2D numpy array containing the mean values of each batch of rows.

    Example:
        input_array = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9],
                                [10, 11, 12],
                                [13, 14, 15]])
        n_rows = 2
        output_array = batch_mean_rows(input_array, n_rows)
        print(output_array)
        # Output:
        # array([[ 4. ,  5. ,  6. ],
        #        [11.5, 12.5, 13.5]])
    """

    def split_array_by_batch(arr, batch_sizes):
        start = 0
        result = []
        for size in batch_sizes:
            result.append(arr[start : start + size])
            start += size
        return result

    if batch_size:
        return np.vstack(
            [
                arr_batch.mean(axis=0)
                for arr_batch in np.array_split(arr, batch_size)
                if arr_batch.any()
            ]
        )
    elif batch_sizes:
        return np.vstack(
            [
                arr_batch.mean(axis=0)
                for arr_batch in split_array_by_batch(arr, batch_sizes)
                if arr_batch.any()
            ]
        )
    else:
        raise ValueError("Either 'batch_size' or 'batch_sizes' must be supplied!")


def get_encompassing_rectangle_dims(
    dims: Tuple[float, float], angle_deg: float
) -> Tuple[float, float]:
    """
    Given dimensions of a rectangle (width, height) and an angle of rotation (clockwise from negative X-axis),
    return the encompassing cartesian rectangle's dimensions.
    """

    w, h = dims
    angle_rad = math.radians(angle_deg)

    h_enc = h * math.sin(angle_rad - math.pi / 2) + w * math.sin(math.pi - angle_rad)
    w_enc = h * math.cos(angle_rad - math.pi / 2) + w * math.cos(math.pi - angle_rad)
    return (w_enc, h_enc)


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


def chunks(arr, n: int):
    """
    Generate n-sized chunks from arr.
    https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks?page=1&tab=votes#tab-top
    """

    for i in range(0, len(arr), n):
        yield arr[i : i + n]


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


def robust_interpolation(
    x,  # x-values to interpolate onto
    xi,  # real x vals
    yi,  # real y vals
    n_pnts=3,  # number of points to include in each intepolation
) -> np.ndarray:
    """Doc."""

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


def get_noise_start_idx(arr: np.ndarray, **kwargs):
    """
    Get the index where noise begins (according to standard deviation from local linear fits).
    Assumes valid (no NaNs or <=0) values of 'arr' are supplied.
    """

    def std_from_local_linear_fit(arr: np.ndarray, kernel_size=8, **kwargs):
        """
        Given an array and an even kernel (1D) size, returns an array of equal length where each kernel-sized subarray
        contains the local standard deviation of that part of the original array from a local linear fit of that part.
        """

        ransac = linear_model.RANSACRegressor()

        local_std_arr = np.zeros(arr.shape, dtype=np.float64)
        for idx in range(kernel_size, arr.size - kernel_size):
            # prepare kernel-sized sub-array
            min_idx = idx - kernel_size // 2
            max_idx = idx + kernel_size // 2 + 1
            xs = np.arange(min_idx, max_idx)
            ys = arr[min_idx:max_idx]

            # fit linear model to subarray
            ransac.fit(xs[:, np.newaxis], ys)
            p0, p1 = ransac.estimator_.intercept_, ransac.estimator_.coef_[0]
            fitted_ys = p0 + p1 * xs

            # add fitted line to data, then calculate std
            local_std_arr[min_idx:max_idx] = (arr[min_idx:max_idx] - fitted_ys).std()

        return local_std_arr

    log_arr = np.log(arr)
    local_std = std_from_local_linear_fit(log_arr, **kwargs)
    normalized_local_std = local_std / local_std.max()
    diff_normalized_local_std = np.diff(normalized_local_std)

    # get the max jumps, and return the minimum idx among the highest 10 jumps
    higherst_diff_normalized_local_std_sorted = np.argsort(diff_normalized_local_std)[::-1]
    return higherst_diff_normalized_local_std_sorted[:10].min()


def extrapolate_over_noise(
    x,
    y,
    x_interp=None,
    x_lims: Limits = Limits(),
    y_lims: Limits = Limits(),
    n_bins=2**17,
    n_robust=3,
    interp_type="gaussian",
    extrap_x_lims=Limits(-np.inf, np.inf),
    should_interactively_set_upper_x=True,
    **kwargs,
) -> InterpExtrap1D:
    """Doc."""

    INTERP_TYPES = ("gaussian", "linear")

    # unify length of y to x (assumes y decays to zero)
    y = unify_length(y, (len(x),))

    if x_interp is None:
        extrap_x_lims = extrap_x_lims.clamp(Limits(min(x), max(x)))
        initial_x_interp = np.linspace(*extrap_x_lims, n_bins)
    else:
        initial_x_interp = x_interp

    # heuristic auto-determination of limits by noise inspection
    valid_idxs = Limits(0.05, 10).valid_indices(x) & y_lims.valid_indices(y)
    noise_start_idx = get_noise_start_idx(y[valid_idxs], **kwargs)
    x_noise = x[valid_idxs][noise_start_idx]
    if kwargs.get("should_auto_determine_upper_x"):
        # allow manual limiting of auto choice
        if x_lims.upper > x_noise:
            x_lims.upper = x_noise

    # interactively choose the noise level
    elif should_interactively_set_upper_x:
        with display.Plotter(
            super_title="Interactive Selection of Noise Level\n(One point)",
            selection_limits=x_lims,
            selection_type="upper",
            should_close_after_selection=True,
            xlim=(0, min(x[(y < 1e-3) & (x > 0.1)][0], x[(x < 2)][-1])),
            x_scale="quadratic",
            ylim=(1e-4, 1),
            y_scale="log",
        ) as ax:
            ax.plot(x, y, "ok", label=kwargs["name"] if kwargs.get("name") else "data")
            ax.axvline(x=x_noise, color="red", lw=1, ls="--", label="heuristic estimate")
            ax.legend()

    # choose interpolation range (extrapolate the noisy parts)
    interp_idxs = x_lims.valid_indices(x) & y_lims.valid_indices(y)
    x_samples = x[interp_idxs]
    y_samples = y[interp_idxs]

    if interp_type == "gaussian":
        if n_robust:
            y_interp = robust_interpolation(
                initial_x_interp**2, x_samples**2, np.log(y_samples), n_robust
            )
        else:
            interpolator = scipy.interpolate.interp1d(
                x_samples**2,
                np.log(y_samples),
                fill_value="extrapolate",
                assume_sorted=True,
            )
            y_interp = interpolator(initial_x_interp**2)

    elif interp_type == "linear":
        if n_robust:
            y_interp = robust_interpolation(initial_x_interp, x_samples, y_samples, n_robust)
        else:
            interpolator = scipy.interpolate.interp1d(
                x_samples,
                y_samples,
                fill_value="extrapolate",
                assume_sorted=True,
            )
            y_interp = interpolator(initial_x_interp)

    else:
        raise ValueError(f"Unknown interpolation type '{interp_type}'. Choose from {INTERP_TYPES}.")

    if interp_type == "gaussian":
        y_interp = np.exp(y_interp)
    elif interp_type == "linear":
        #  zero-pad the linear interpolation
        y_interp[x_interp > x[y_lims.valid_indices(y)][-1]] = 0

    return InterpExtrap1D(
        interp_type,
        x_interp,
        y_interp,
        x_samples,
        y_samples,
        x,
        y,
        interp_idxs,
        x_lims,
    )


def fourier_transform_1d(
    x: np.ndarray,
    y: np.ndarray,
    should_inverse: bool = False,
    is_input_symmetric=False,
    n_bins=2**17,
    bin_size=None,
    should_normalize: bool = False,
    **kwargs,
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

    if kwargs.get("should_plot"):
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


def simple_slice(arr: np.ndarray, idxs, axis: int):
    """
    Adapted from:
    https://stackoverflow.com/questions/24398708/slicing-a-numpy-array-along-a-dynamically-specified-axis
    """
    sl = [slice(None)] * arr.ndim
    sl[axis] = idxs
    return arr[tuple(sl)]


def unify_length(arr: np.ndarray, req_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Returns a either a zero-padded array or a trimmed one, according to the requested shape.
    passing None as one of the shape axes ignores that axis.
    """

    if arr.ndim != len(req_shape):
        raise ValueError(f"Dimensionallities do not match {len(arr.shape)}, {len(req_shape)}")

    out_arr = np.copy(arr)

    # assume >=2D array
    try:
        for ax, req_length in enumerate(req_shape):
            if req_length is None:
                # no change required
                continue
            if (arr_len := arr.shape[ax]) >= req_shape[ax]:
                out_arr = simple_slice(out_arr, slice(req_length), ax)
            else:
                pad_width = tuple(
                    [(0, 0)] * ax + [(0, req_length - arr_len)] + [(0, 0)] * (arr.ndim - (ax + 1))
                )
                out_arr = np.pad(out_arr, pad_width)

        return out_arr

    # 1D array
    except IndexError:
        out_length = req_shape[0]
        if (arr_len := arr.size) >= out_length:
            return arr[:out_length]
        else:
            return np.pad(arr, (0, out_length - arr_len))


def xcorr(a, b):
    """Does correlation similar to Matlab xcorr, cuts positive lags, normalizes properly"""

    c = scipy.signal.correlate(a, b)  # Uses FFT for large arrays - much faster!
    c = c[c.size // 2 :]
    c = c / np.arange(c.size, 0, -1)

    # normalize
    c = c / (a.mean() * b.mean()) - 1

    # subtract shot noise
    c[0] -= 1 / np.sqrt(a.mean() * b.mean())

    if c.size <= np.iinfo(np.uint16).max:
        lags = np.arange(c.size, dtype=np.uint16)
    else:
        lags = np.arange(c.size, dtype=np.uint32)

    return c, lags


def can_float(value: Any) -> bool:
    """Checks if 'value' can be turned into a float"""

    try:
        float(value)
        return True
    except (ValueError, TypeError):
        try:
            if value == "-":  # consider hyphens part of float (minus sign)
                return True
            return False
        except TypeError:
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
        masses_at_displacements = np.atleast_2d(arr).sum(axis=dim)
        displacements = np.arange(masses_at_displacements.size)
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


def reverse_dict(dict_: dict, ignore_unhashable=False) -> dict:
    """
    Return a new dict for which keys and values are switched.
    Raises a TypeError if a value is unhashable, unless 'ignore_unhashable' is True,
    in which case it will drop from the new dict.
    """

    if ignore_unhashable:
        return {val: key for key, val in dict_.items() if isinstance(val, Hashable)}
    else:
        return {val: key for key, val in dict_.items()}


def update_attributes(obj, val_dict: dict):
    """Update the (existing) attributes of an object according to a supplied dictionary"""

    rev_attr_dict = reverse_dict(vars(obj), ignore_unhashable=True)
    for old_val, new_val in val_dict.items():
        if old_val in rev_attr_dict:
            setattr(obj, rev_attr_dict[old_val], new_val)


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


def dir_date_parts(data_path: str, sub_dir: str = "", month: str = None, year: str = None) -> list:
    """
    Inputs:
        data_path - string containing the path to the main data folder.
        month - str month to search for. if None, will return months matching the year.
        year - str year to search for. If None, will return years of all subfolders.

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
