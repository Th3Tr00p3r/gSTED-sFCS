"""Miscellaneous helper functions/classes"""

from __future__ import annotations

import asyncio
import csv
import functools
import os
import time
from types import SimpleNamespace
from typing import Callable, List, Union

import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtGui import QIcon

import logic.app


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


# print(f"part 1 timing: {(time.perf_counter() - tic)*1e3:0.4f} ms") # TESTING
# tic = time.perf_counter() # TESTING


def paths_to_icons(paths_dict) -> dict:
    """Doc."""

    return {key: QIcon(val) for key, val in paths_dict.items()}


async def sync_to_thread(func) -> None:
    """
    This is a workaround -
    asyncio.to_thread() must be awaited, but toggle_video needs to be
    a regular function to keep the rest of the code as it is. by creating this
    async helper function I can make it work. A lambda would be better here
    but there's no async lambda yet.
    """
    await asyncio.to_thread(func)


def wdgt_children_as_row_list(parent_wdgt) -> List[List[str, str]]:
    """Doc."""

    l1 = parent_wdgt.findChildren(QtWidgets.QLineEdit)
    l2 = parent_wdgt.findChildren(QtWidgets.QSpinBox)
    l3 = parent_wdgt.findChildren(QtWidgets.QDoubleSpinBox)
    l4 = parent_wdgt.findChildren(QtWidgets.QComboBox)
    l5 = parent_wdgt.findChildren(QtWidgets.QCheckBox)
    l6 = parent_wdgt.findChildren(QtWidgets.QRadioButton)
    l7 = parent_wdgt.findChildren(QtWidgets.QSlider)
    children_list = l1 + l2 + l3 + l4 + l5 + l6 + l7

    rows = []
    for child in children_list:

        child_class = child.__class__.__name__

        if child_class == "QComboBox":
            val = child.currentText()
        elif child_class in {"QCheckBox", "QRadioButton"}:
            val = child.isChecked()
        elif child_class == "QSlider":
            val = child.value()
        elif child_class in {"QSpinBox", "QDoubleSpinBox"} and not child.isReadOnly():
            val = child.value()
        elif (
            child_class == "QLineEdit"
            and not child.isReadOnly()
            and child.objectName() != "qt_spinbox_lineedit"
        ):
            val = child.text()
        else:
            continue

        rows.append((child.objectName(), str(val)))

    return rows


def gui_to_csv(parent_wdgt, file_path):
    """Doc."""

    rows = wdgt_children_as_row_list(parent_wdgt)

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def csv_rows_as_list(file_path) -> List[tuple]:
    """Doc."""

    rows = []
    with open(file_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(tuple(row))
    return rows


def csv_to_gui(file_path, gui_parent):
    """Doc."""

    rows = csv_rows_as_list(file_path)

    for row in rows:
        wdgt_name, val = row

        child = gui_parent.findChild(QtWidgets.QWidget, wdgt_name)
        child_class = child.__class__.__name__

        if child_class == "QSlider":
            child.setValue(int(val))
        elif child_class in {"QSpinBox", "QDoubleSpinBox"}:
            child.setValue(float(val))
        elif child_class == "QLineEdit":
            child.setText(val)
        elif child_class == "QComboBox":
            child.setCurrentText(val)
        elif child_class in {"QCheckBox", "QRadioButton"}:
            child.setChecked(bool(val))


def deep_getattr(object, deep_attr_name: str, recursion=False):
    """
    Get deep attribute of object. Useful for dynamically-set deep attributes.
    Example usage: a = deep_getattr(obj, "sobj.ssobj.a")
    """

    if recursion:
        try:
            next_attr_name, deep_attr_name = deep_attr_name.split(".", maxsplit=1)
        except ValueError:
            # end condition - only one level of attributes left
            return getattr(object, deep_attr_name)
        else:
            # recursion
            return deep_getattr(getattr(object, next_attr_name), deep_attr_name)

    else:
        # loop version, faster
        for attr_name in deep_attr_name.split("."):
            object = getattr(object, attr_name)
        return object


def div_ceil(x: int, y: int) -> int:
    """Returns x divided by y rounded towards positive infinity"""

    return int(x // y + (x % y > 0))


def translate_dict(original_dict: dict, trans_dict: dict) -> dict:
    """
    Updates values of dict according to another dict:
    val_trans_dct.keys() are the values to update,
    and val_trans_dct.values() are the new values.
    """

    return {
        key: (trans_dict[val] if val in trans_dict.keys() else val)
        for key, val in original_dict.items()
    }


def dir_date_parts(data_path, month: int = None, year: int = None) -> list:
    """
    Inputs:
        main_data_path - string containing the path to the main data folder.
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

    dir_name_list = [
        item for item in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, item))
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
            if (dir_date_dict["month"] == month) and (dir_date_dict["year"] == year)
        ]

    # get matching months
    elif year:
        date_item_list = [
            dir_date_dict["month"]
            for dir_date_dict in dir_date_dict_list
            if dir_date_dict["year"] == year
        ]

    # get all existing years
    else:
        date_item_list = [dir_date_dict["year"] for dir_date_dict in dir_date_dict_list]

    # return unique date parts, sorted in descending order
    return sorted(set(date_item_list), reverse=True)


class QtWidgetAccess:
    """Doc."""

    def __init__(self, obj_name: str, getter: str, gui_parent_name: str = "settings"):
        self.obj_name = obj_name
        if getter is not None:
            self.getter = getter
            self.setter = self._get_setter()
        else:
            self.getter = None
            self.setter = None
        self.gui_parent_name = gui_parent_name

    def _get_setter(self) -> str:
        """Doc."""

        if self.getter == "isChecked":
            return "setChecked"
        else:
            first_getter_letter, *rest_getter_str = self.getter
            return "set" + first_getter_letter.upper() + "".join(rest_getter_str)

    def hold_obj(self, parent_gui) -> QtWidgetAccess:
        """Save the actual widget object as an attribute"""

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

    def hold_objects(
        self, app: logic.app.App, wdgt_name_list: List[str] = None, hold_all=False
    ) -> QtWidgetCollection:
        """Stores the actual GUI object in the listed (or all) widgets."""

        if wdgt_name_list is not None:
            for wdgt_name in wdgt_name_list:
                try:
                    wdgt = getattr(self, wdgt_name)
                except AttributeError:
                    # collection has no widget 'wdgt_name'
                    pass
                else:
                    parent_gui = getattr(app.gui, wdgt.gui_parent_name)
                    wdgt.hold_obj(parent_gui)

        elif hold_all:
            for wdgt in vars(self).values():
                parent_gui = getattr(app.gui, wdgt.gui_parent_name)
                wdgt.hold_obj(parent_gui)

        return self

    def write_to_gui(self, app: logic.app.App, new_vals) -> None:
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

    def read_dict_from_gui(self, app: logic.app.App) -> dict:
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

    def read_namespace_from_gui(self, app: logic.app.App) -> SimpleNamespace:
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
