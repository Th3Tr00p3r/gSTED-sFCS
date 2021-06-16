"""Miscellaneous helper functions/classes"""

from __future__ import annotations

import asyncio
import csv
import datetime
import functools
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, List, Union

import numpy as np
import PyQt5.QtWidgets as QtWidgets
import pyqtgraph as pg
from PyQt5.QtGui import QIcon

import logic.app


def timer(func) -> Callable:
    """
    Meant to be used as a decorator (@timer)
    for quickly setting up function timing for testing.
    """

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
    children_list = l1 + l2 + l3 + l4 + l5

    rows = []
    for child in children_list:
        if not child.objectName() == "qt_spinbox_lineedit":
            try:
                if not child.isReadOnly():
                    # only save editable QSpinBox/QLineEdit
                    if hasattr(child, "value"):
                        # QSpinBox
                        val = child.value()
                    else:
                        # QLineEdit
                        val = child.text()
                    rows.append((child.objectName(), str(val)))
            except AttributeError:
                # if doesn't have a isReadOnly attribute (QComboBox/QCheckBox)
                if hasattr(child, "currentIndex"):
                    # QComboBox
                    val = child.currentIndex()
                elif hasattr(child, "isChecked"):
                    # QCheckBox
                    val = child.isChecked()
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
        if not child == "nullptr":
            if hasattr(child, "value"):  # QSpinBox
                child.setValue(float(val))
            elif hasattr(child, "currentIndex"):  # QComboBox
                child.setCurrentIndex(int(val))
            elif hasattr(child, "isChecked"):  # QCheckBox
                child.setChecked(bool(val))
            elif hasattr(child, "text"):  # QLineEdit
                child.setText(val)


def deep_getattr(deep_object, deep_name, default=None):
    """
    Get deep attribute of object. Useful for dynamically-set deep attributes.
    Example usage: a = deep_getattr(obj, "sobj.ssobj.a")
    """

    for attr in deep_name.split("."):
        deep_object = getattr(deep_object, attr)
    return deep_object


def div_ceil(x: int, y: int) -> int:
    """Returns x divided by y rounded towards positive infinity"""

    return int(x // y + (x % y > 0))


def get_datetime_str() -> str:
    """Return current date and time string in the format DDMMYY_HHMMSS"""

    return datetime.datetime.now().strftime("%d%m_%H%M%S")


def inv_dict(dct: dict) -> dict:
    """Inverts a Python dictionary. Expects mapping to be 1-to-1"""

    return {val: key for key, val in dct.items()}


def trans_dict(dct: dict, val_trans_dct: dict) -> dict:
    """
    Updates values of dict according to another dict:
    val_trans_dct.keys() are the values to update,
    and val_trans_dct.values() are the new values.
    """

    for org_key, org_val in dct.items():
        if org_val in val_trans_dct.keys():
            dct[org_key] = val_trans_dct[org_val]
    return dct


@dataclass
class DeviceAttrs:

    class_name: str
    log_ref: str
    param_widgets: QtWidgetCollection
    led_icon: QIcon
    cls_xtra_args: List[str] = None


@dataclass
class ImageData:

    pic1: np.ndarray
    norm1: np.ndarray
    pic2: np.ndarray
    norm2: np.ndarray
    line_ticks_V: np.ndarray
    row_ticks_V: np.ndarray


class QtWidgetAccess:
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


class ImageDisplay:
    """Doc."""

    def __init__(self, layout):
        glw = pg.GraphicsLayoutWidget()
        self.vb = glw.addViewBox()
        self.hist = pg.HistogramLUTItem()
        glw.addItem(self.hist)
        layout.addWidget(glw)

    def add_image(self, image: np.ndarray, limit_zoomout=True, crosshair=True):
        """Doc."""

        image_item = pg.ImageItem(image)
        self.vb.addItem(image_item)
        self.hist.setImageItem(image_item)

        if limit_zoomout:
            self.vb.setLimits(
                xMin=0,
                xMax=image.shape[0],
                minXRange=0,
                maxXRange=image.shape[0],
                yMin=0,
                yMax=image.shape[1],
                minYRange=0,
                maxYRange=image.shape[1],
            )
        if crosshair:
            self.vLine = pg.InfiniteLine(angle=90, movable=True)
            self.hLine = pg.InfiniteLine(angle=0, movable=True)
            self.vb.addItem(self.vLine)
            self.vb.addItem(self.hLine)
            try:
                # keep crosshair at last position
                x_pos, y_pos = self.last_roi
                self.vLine.setPos(x_pos)
                self.hLine.setPos(y_pos)
            except AttributeError:
                # case first image since application loaded (no last_roi)
                pass
            self.vb.scene().sigMouseClicked.connect(self.mouseClicked)

    def mouseClicked(self, evt):
        """Doc."""
        # TODO: selected position is not accurate for some reason.

        try:
            pos = evt.pos()
        except AttributeError:
            # outside image
            pass
        else:
            if self.vb.sceneBoundingRect().contains(pos):
                mousePoint = self.vb.mapSceneToView(pos)
                self.vLine.setPos(mousePoint.x())
                self.hLine.setPos(mousePoint.y())
                self.last_roi = (mousePoint.x(), mousePoint.y())
