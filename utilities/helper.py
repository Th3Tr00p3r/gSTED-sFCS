"""Miscellaneous helper functions/classes"""

import asyncio
import functools
import logging
import math
import os
import time
from contextlib import suppress
from typing import Any, Callable, List, Tuple

import numpy as np
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from scipy.special import j0, j1, jn_zeros
from sklearn import linear_model

# import time # TESTING
# tic = time.perf_counter() # TESTING
# print(f"part 1 timing: {(time.perf_counter() - tic)*1e3:0.4f} ms") # TESTING


def timer(threshold_ms: int = 0) -> Callable:
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
                    print(
                        f"***TIMER*** Function '{func.__name__}()' took {elapsed_time_ms:.2f} ms (threshold: {threshold_ms:d} ms).\n"
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
                    print(
                        f"***TIMER*** Function '{func.__name__}()' took {elapsed_time_ms:.2f} ms (threshold: {threshold_ms:d} ms).\n"
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


def fourier_transform_1d(
    x: np.ndarray,
    y: np.ndarray,
    should_inverse: bool = False,
    n_bins=1024,
    bin_size=None,
    lag_units_factor=1,
    should_force_zero_to_one: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Doc."""

    # keep the zero value
    y0 = y[0]

    # interpolate to ensure equal spacing and set bins_size or number of bins
    if bin_size:
        r = np.arange(min(x), max(x), bin_size)
    else:
        r = np.linspace(min(x), max(x), n_bins)
        bin_size = r[1] - r[0]

    x = np.hstack((-np.flip(x[1:]), x[1:]))
    y = np.hstack((np.flip(y[1:]), y[1:]))
    r = np.hstack((-np.flip(r[1:]), r[0], r[1:]))
    n_bins = r.size

    q = 2 * np.pi * np.arange(n_bins) / (bin_size * n_bins)

    # Linear interpolation
    interpolator = interp1d(
        x,
        y,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    fr_interp = interpolator(r)

    # normalize
    fr_interp *= bin_size * lag_units_factor

    # handle center (0)
    if should_force_zero_to_one:
        y0 = 1
    else:  # normalize according to the units and bin size
        y0 *= bin_size * lag_units_factor
    fr_interp[fr_interp.size // 2] = y0

    # Transform
    if should_inverse:
        fr_interp = fr_interp.astype(np.complex) * np.exp(+1j * q * (min(r)))
        fq = ifft(fr_interp)
    else:
        fq = fft(fr_interp)
    fq *= np.exp(-1j * q * (min(r)))

    # symmetrize q
    k1 = int(n_bins / 2 + 0.9)  # trick to deal with even and odd values of 'n'
    k2 = math.ceil(n_bins / 2 + 1)
    q = np.hstack((q[k2:] - 2 * np.pi / bin_size, q[:k1]))
    fq = np.hstack((fq[k2:], fq[:k1]))

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
    c0 = jn_zeros(0, n)
    bessel_j0 = j0
    bessel_j1 = j1

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
                interpolator = interp1d(
                    x ** 2,
                    np.log(y),
                    fill_value="extrapolate",
                    assume_sorted=True,
                )
                fr_interp = np.exp(interpolator(r ** 2))
            else:  # linear
                interpolator = interp1d(
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

        interpolator = interp1d(
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


def chunks(list_: list, n: int):
    """
    Yield successive n-sized chunks from list_.
    https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks?page=1&tab=votes#tab-top
    """

    for i in range(0, len(list_), n):
        yield list_[i : i + n]


def can_float(value: Any) -> bool:
    """Checks if 'value' can be turned into a float"""

    try:
        float(value)
        return True
    except (ValueError, TypeError):
        if value == "-":  # consider hyphens part of float (minus sign)
            return True
        return False


def number(x):
    """Attempts to convert 'x' into an integer, a float if that fails."""

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
            yield number(source_str[i : j - 1])
        i = j


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
        lower = max(self[0], other[0])
        upper = min(self[1], other[1])
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

    def valid_indices(self, arr: np.ndarray):
        """
        Checks whether each element is contained and returns a boolean array of same shape.
        __contains__ must return a single boolean array, otherwise would be included there.
        """
        if isinstance(arr, np.ndarray):
            return (arr >= self.lower) & (arr <= self.upper)
        else:
            raise TypeError("Argument 'arr' must be a Numpy ndarray!")

    def as_dict(self):
        if self.dict_labels is not None:
            return {key: val for key, val in zip(self.dict_labels, (self.lower, self.upper))}
        else:
            return {key: val for key, val in zip(("lower", "upper"), (self.lower, self.upper))}

    def interval(self):
        return abs(self.upper - self.lower)

    def center(self):
        """Get the center of the range"""

        return sum(self) * 0.5

    def clamp(self, n):
        """Force limit range on number n"""

        return max(min(self.upper, n), self.lower)

    def as_range(self) -> range:
        """Get a Python 'range' (generator)"""

        return range(self.lower, self.upper)


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


def center_of_mass(arr: np.ndarray) -> Tuple:
    """Returns the center of mass coordinates of a Numpy ndarray"""

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


def deep_getattr(obj, deep_attr_name: str, default=None, recursion=False):
    """
    Get deep attribute of obj. Useful for dynamically-set deep attributes.
    Example usage: a = deep_getattr(obj, "sobj.ssobj.a")
    """

    if recursion:
        try:
            next_attr_name, deep_attr_name = deep_attr_name.split(".", maxsplit=1)
        except ValueError:
            # end condition - only one level of attributes left
            return getattr(obj, deep_attr_name, default)
        else:
            # recursion
            return deep_getattr(getattr(obj, next_attr_name), deep_attr_name, default)

    else:
        # loop version, faster
        for attr_name in deep_attr_name.split("."):
            obj = getattr(obj, attr_name, default)
        return obj


def div_ceil(x: int, y: int) -> int:
    """Returns x divided by y rounded towards positive infinity"""

    return -(-x // y)  # TODO: test me!


#    return int(x // y + (x % y > 0))


def reverse_dict(dict_: dict) -> dict:
    """Return a new dict for which keys and values are switched"""

    return {val: key for key, val in dict_.items()}


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
        # File empty and was just created
        logging.info("Log file initialized.")
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
