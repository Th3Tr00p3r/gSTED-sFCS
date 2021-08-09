"""Miscellaneous helper functions/classes"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import re
import time
from types import SimpleNamespace
from typing import Callable, List, Union

import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtGui import QIcon


def timer(func) -> Callable:
    """
    Meant to be used as a decorator (@timer)
    for quickly setting up function timing for testing.
    """

    if asyncio.iscoroutinefunction(func):
        # timing async funcitons
        @functools.wraps(func)
        async def wrapper_timer(*args, **kwargs):
            tic = time.perf_counter()
            value = await func(*args, **kwargs)
            toc = time.perf_counter()
            elapsed_time_ms = (toc - tic) * 1e3
            print(f"{func.__name__}() took {elapsed_time_ms:0.4f} ms")
            return value

    else:

        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            tic = time.perf_counter()
            value = func(*args, **kwargs)
            toc = time.perf_counter()
            elapsed_time_ms = (toc - tic) * 1e3
            print(f"{func.__name__}() took {elapsed_time_ms:0.4f} ms")
            return value

    return wrapper_timer


# import time # TESTING
# tic = time.perf_counter() # TESTING
# print(f"part 1 timing: {(time.perf_counter() - tic)*1e3:0.4f} ms") # TESTING


def force_aspect(ax, aspect=1) -> None:
    """
    See accepted answer here:
    https://stackoverflow.com/questions/7965743/how-can-i-set-the-aspect-ratio-in-matplotlib
    """

    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def paths_to_icons(paths_dict) -> dict:
    """Doc."""

    return {key: QIcon(val) for key, val in paths_dict.items()}


async def sync_to_thread(func) -> None:
    """
    This is a workaround -
    asyncio.to_thread() must be awaited, but toggle_video needs to be
    a regular function to keep the rest of the code as it is. by creating this
    async helper function I can make it work. A lambda would be better here
    but there's no async lambda (yet?).
    """

    await asyncio.to_thread(func)


def bool_str(str_: str):
    """A strict bool() for strings"""
    if str_ == "True":
        return True
    elif str_ == "False":
        return False
    else:
        raise ValueError(f"'{str_}' is neither 'True' or 'False'.")


# What to do with each widget class
getter_setter_type_dict = {
    "QComboBox": ("currentText", "setCurrentText", str),
    "QTabWidget": ("currentIndex", "setCurrentIndex", int),
    "QCheckBox": ("isChecked", "setChecked", bool_str),
    "QRadioButton": ("isChecked", "setChecked", bool_str),
    "QSlider": ("value", "setValue", int),
    "QSpinBox": ("value", "setValue", int),
    "QDoubleSpinBox": ("value", "setValue", float),
    "QLineEdit": ("text", "setText", str),
    "QPlainTextEdit": ("toPlainText", "setPlainText", str),
    "QButtonGroup": ("checkedButton", None, None),
    "QTimeEdit": ("time", "setTime", None),
    "QIcon": ("icon", "setIcon", None),
    "QStackedWidget": ("currentIndex", "setCurrentIndex", int),
}


def widget_getter_setter_type(widget_class: str) -> tuple:
    """
    Returns a tuple of strings, where the first element
    is the name of the getter method of the widget,
    the second is the setter method and third is the type
    """

    try:
        return getter_setter_type_dict[widget_class]
    except KeyError:
        return (None,) * 3


def wdgt_items_to_text_lines(parent_wdgt) -> List[str]:
    """Doc."""

    wdgt_types = [
        "QLineEdit",
        "QSpinBox",
        "QDoubleSpinBox",
        "QComboBox",
        "QStackedWidget",
        "QRadioButton",
        "QSlider",
        "QTabWidget",
        "QCheckBox",
    ]
    children_class_lists = [
        parent_wdgt.findChildren(getattr(QtWidgets, wdgt_type)) for wdgt_type in wdgt_types
    ]
    children_list = [child for child_list in children_class_lists for child in child_list]

    lines = []
    for child in children_list:
        child_class = child.__class__.__name__
        child_name = child.objectName()
        getter, _, _ = widget_getter_setter_type(child_class)

        if (
            (hasattr(child, "isReadOnly") and not child.isReadOnly())
            or not hasattr(child, "isReadOnly")
        ) and child_name not in {"qt_spinbox_lineedit", "qt_tabwidget_stackedwidget"}:
            val = getattr(child, getter)()
        else:
            # ignore read-only and weird auto-widgets
            continue

        lines.append(f"{child_name},{val}")

    return lines


def write_gui_to_file(parent_wdgt, file_path):
    """Doc."""

    write_list_to_file(file_path, wdgt_items_to_text_lines(parent_wdgt))


def write_list_to_file(file_path, lines: List[str]) -> None:
    """Accepts a list of strings 'lines' and writes them to 'file_path'."""

    with open(file_path, "w") as f:
        f.write("\n".join(lines))


def read_file_to_list(file_path) -> List[str]:
    """Doc."""

    with open(file_path, "r") as f:
        return f.read().splitlines()


def read_file_to_gui(file_path, gui_parent):
    """Doc."""

    lines = read_file_to_list(file_path)

    for line in lines:
        wdgt_name, val = re.split(",", line, maxsplit=1)
        child = gui_parent.findChild(QtWidgets.QWidget, wdgt_name)
        _, setter, type_func = widget_getter_setter_type(child.__class__.__name__)

        if type_func not in {None, str}:
            val = type_func(val)

        try:
            getattr(child, setter)(val)
        except TypeError:
            logging.warning(
                f"Child widget '{wdgt_name}' was not found in parent widget '{gui_parent.objectName()}' - probably removed from GUI. Overwrite the defaults to stop seeing this warning."
            )


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


def namespace_to_dict(ns) -> dict:
    """Doc."""

    return vars(ns).copy()


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


def reverse_dict(dict_: dict) -> dict:
    """Return a new dict for which keys and values are switched"""

    return {val: key for key, val in dict_.items()}


def file_extension(file_path: str) -> str:
    """Returns the file extension as a string, i.e. file_extension('Path/file.ext') -> '.ext'."""

    return re.split("(\\.[a-z]{3})$", file_path, maxsplit=1)[1]


def sort_file_paths_by_file_number(file_paths: List[str]) -> List[str]:
    """
    Returns a path list, sorted according to file number (ascending).
    Works for file paths of the following format:
    "PATH_TO_FILE_DIRECTORY/file_template_*.ext"
    where the important part is '*.ext' (* is a any number, ext is any 3 letter file extension)
    """

    return sorted(
        file_paths,
        key=lambda file_path: int(re.split(f"(\\d+){file_extension(file_paths[0])}", file_path)[1]),
    )


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


class QtWidgetAccess:
    """Doc."""

    def __init__(self, obj_name: str, widget_class: str, gui_parent_name: str, does_hold_obj: bool):
        self.obj_name = obj_name
        self.getter, self.setter, _ = widget_getter_setter_type(widget_class)
        self.gui_parent_name = gui_parent_name
        self.does_hold_obj = does_hold_obj

    def hold_obj(self, parent_gui) -> QtWidgetAccess:
        """Save the actual widget object as an attribute"""

        if self.does_hold_obj:
            self.obj = getattr(parent_gui, self.obj_name)
        return self

    def get(self, parent_gui=None) -> Union[int, float, str]:
        """Get widget value"""

        if self.getter is not None:
            wdgt = self.obj if parent_gui is None else getattr(parent_gui, self.obj_name)
            return getattr(wdgt, self.getter)()
        else:
            return None

    def set(self, arg, parent_gui=None) -> None:
        """Set widget property"""

        wdgt = self.obj if parent_gui is None else getattr(parent_gui, self.obj_name)
        getattr(wdgt, self.setter)(arg)


class QtWidgetCollection:
    """Doc."""

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, QtWidgetAccess(*val))

    def hold_objects(self, app) -> QtWidgetCollection:
        """Stores the actual GUI object in all widgets (for which does_hold_obj is True)."""

        for wdgt in vars(self).values():
            parent_gui = getattr(app.gui, wdgt.gui_parent_name)
            wdgt.hold_obj(parent_gui)

        return self

    def write_to_gui(self, app, new_vals) -> None:
        """
        Fill widget collection with values from dict/list, or a single value for all.
        if new_vals is a list, the values will be inserted in the order of vars(self).keys().
        """

        if isinstance(new_vals, list):
            new_vals = dict(zip(vars(self).keys(), new_vals))

        if isinstance(new_vals, dict):
            for attr_name, val in new_vals.items():
                wdgt = getattr(self, attr_name)
                parent_gui = getattr(app.gui, wdgt.gui_parent_name)
                wdgt.set(val, parent_gui)
        else:
            for wdgt in vars(self).values():
                parent_gui = getattr(app.gui, wdgt.gui_parent_name)
                wdgt.set(new_vals, parent_gui)

    def read_dict_from_gui(self, app) -> dict:
        """
        Read values from QtWidgetAccess objects, which are the attributes of self and return a dict.
        If a QtWidgetAccess object holds the actual GUI object, the dict will contain the
        QtWidgetAccess object itself instead of the value (for getting/setting live values)
        """

        wdgt_val_dict = {}
        for attr_name, wdgt in vars(self).items():
            parent_gui = getattr(app.gui, wdgt.gui_parent_name)
            if hasattr(wdgt, "obj"):
                wdgt_val_dict[attr_name] = wdgt
            else:
                wdgt_val_dict[attr_name] = wdgt.get(parent_gui)
        return wdgt_val_dict

    def read_namespace_from_gui(self, app) -> SimpleNamespace:
        """
        Same as 'read_dict_from_gui' but returns an object
        instead of a dictionary.
        """

        wdgt_val_ns = SimpleNamespace()
        for attr_name, wdgt in vars(self).items():
            parent_gui = getattr(app.gui, wdgt.gui_parent_name)
            if hasattr(wdgt, "obj"):
                setattr(wdgt_val_ns, attr_name, wdgt)
            else:
                setattr(wdgt_val_ns, attr_name, wdgt.get(parent_gui))
        return wdgt_val_ns
