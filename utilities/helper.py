"""Miscellaneous helper functions/classes"""

import asyncio
import functools
import logging
import os
import time
from contextlib import suppress
from typing import Any, Callable, List, Tuple

import numpy as np

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
                print(f"Funtion '{func.__name__}()' was called...", end=" ")
                tic = time.perf_counter()
                value = await func(*args, **kwargs)
                toc = time.perf_counter()
                elapsed_time_ms = (toc - tic) * 1e3
                if elapsed_time_ms > threshold_ms:
                    print(f"It took {elapsed_time_ms:.2f} ms ({threshold_ms:d} ms threshold).")
                return value

        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                print(f"Funtion '{func.__name__}()' was called...", end=" ")
                tic = time.perf_counter()
                value = func(*args, **kwargs)
                toc = time.perf_counter()
                elapsed_time_ms = (toc - tic) * 1e3
                if elapsed_time_ms > threshold_ms:
                    print(f"It took {elapsed_time_ms:.2f} ms ({threshold_ms:d} ms threshold).")
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


def _can_float(value: Any) -> bool:
    "Checks if 'value' can be turned into a float"

    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def number(x):
    try:
        return int(x)
    except ValueError:
        return float(x)


def _generate_numbers_from_string(source_str):
    """A generator function for getting numbers out of strings."""

    i = 0
    while i < len(source_str):
        j = i + 1
        while (j < len(source_str) + 1) and _can_float(source_str[i:j]):
            j += 1
        with suppress(TypeError, ValueError):
            yield number(source_str[i : j - 1])
        i = j


class Limits:
    """Doc."""

    def __init__(
        self, lower=np.NINF, upper=np.inf, dict_labels: Tuple[str, str] = None, from_string=False
    ):
        if from_string:
            source_str = lower
            self.lower, self.upper = _generate_numbers_from_string(source_str)
        elif isinstance(lower, tuple):
            tuple_ = lower
            self.lower, self.upper = tuple_
        elif isinstance(lower, Limits):
            limit_range_ = lower
            self.lower, self.upper = limit_range_.lower, limit_range_.upper
        elif upper is not None:
            self.lower, self.upper = lower, upper
        else:
            raise TypeError(f"Expected a tuple or 2 numbers, got '{lower}' and '{upper}'.")
        self.dict_labels = dict_labels

    def __call__(self, *args, **kwargs):
        self.__init__(*args, **kwargs)

    def __str__(self):
        lower_frmt = "d" if (int(self.lower) == float(self.lower)) else ".2f"
        upper_frmt = "d" if (int(self.upper) == float(self.upper)) else ".2f"
        return f"({self.lower:{lower_frmt}}, {self.upper:{upper_frmt}})"

    def __iter__(self):
        yield from (self.lower, self.upper)

    def __getitem__(self, idx):
        return tuple(self)[idx]

    def __eq__(self, other):
        if isinstance(other, tuple) or hasattr(other, "clamp"):
            return tuple(self) == other
        else:
            raise TypeError("Can only compare Limits to other instances or tuples")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return other < self.upper

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return other > self.lower

    def __contains__(self, other):
        """
        Checks if 'other' is in 'Limits'.
        If:
        other is a tuple/Limits: checks if full range is contained and returns bool
        other is number: checks if number is contained in range and returns bool
        """
        if (isinstance(other, tuple) and len(other) == 2) or hasattr(other, "clamp"):
            return (self[0] <= other[0]) and (self[1] >= other[1])

        elif isinstance(other, (int, float)):
            return self.lower <= other <= self.upper

        else:
            raise TypeError("Can only compare Limits to other instances or tuples")

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


def write_list_to_file(file_path, lines: List[str]) -> None:
    """Accepts a list of strings 'lines' and writes them to 'file_path'."""

    with open(file_path, "w") as f:
        f.write("\n".join(lines))


def read_file_to_list(file_path) -> List[str]:
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


def deep_getattr(obj, deep_attr_name: str, recursion=False):
    """
    Get deep attribute of obj. Useful for dynamically-set deep attributes.
    Example usage: a = deep_getattr(obj, "sobj.ssobj.a")
    """

    if recursion:
        try:
            next_attr_name, deep_attr_name = deep_attr_name.split(".", maxsplit=1)
        except ValueError:
            # end condition - only one level of attributes left
            return getattr(obj, deep_attr_name)
        else:
            # recursion
            return deep_getattr(getattr(obj, next_attr_name), deep_attr_name)

    else:
        # loop version, faster
        for attr_name in deep_attr_name.split("."):
            obj = getattr(obj, attr_name)
        return obj


def div_ceil(x: int, y: int) -> int:
    """Returns x divided by y rounded towards positive infinity"""

    return int(x // y + (x % y > 0))


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
