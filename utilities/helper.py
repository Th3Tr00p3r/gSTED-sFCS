# -*- coding: utf-8 -*-
"""Miscellaneous helper functions/classes"""

from __future__ import annotations

import csv
import datetime
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, NoReturn, Union

import numpy as np
import PyQt5.QtWidgets as QtWidgets
import pyqtgraph as pg

import gui.icons.icon_paths as icon_path
import logic.app as app_module


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
            wdgt = (
                self.obj if parent_gui is None else getattr(parent_gui, self.obj_name)
            )
            return getattr(wdgt, self.getter)()
        else:
            return None

    def set(self, arg, parent_gui=None) -> NoReturn:
        """Set widget property"""

        wdgt = self.obj if parent_gui is None else getattr(parent_gui, self.obj_name)
        getattr(wdgt, self.setter)(arg)


class QtWidgetCollection:
    """Doc."""

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, QtWidgetAccess(*val))

    def hold_objects(
        self, app: app_module.App, wdgt_name_list: List[str] = None, hold_all=False
    ) -> QtWidgetCollection:
        """Stores the actual GUI object in the listed, or all, widgets."""

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

    def write_to_gui(self, app: app_module.App, new_vals) -> NoReturn:
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

    def read_dict_from_gui(self, app: app_module.App) -> dict:
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

    def read_namespace_from_gui(self, app: app_module.App) -> SimpleNamespace:
        """Doc."""

        wdgt_val_ns = SimpleNamespace()
        for attr_name, wdgt in vars(self).items():
            parent_gui = getattr(app.gui, wdgt.gui_parent_name)
            if hasattr(wdgt, "obj"):
                setattr(wdgt_val_ns, attr_name, wdgt)
            else:
                setattr(wdgt_val_ns, attr_name, wdgt.get(parent_gui))
        return wdgt_val_ns


def wdgt_children_as_row_list(parent_wdgt) -> List[List[str, str]]:
    """Doc."""

    l1 = parent_wdgt.findChildren(QtWidgets.QLineEdit)
    l2 = parent_wdgt.findChildren(QtWidgets.QSpinBox)
    l3 = parent_wdgt.findChildren(QtWidgets.QDoubleSpinBox)
    l4 = parent_wdgt.findChildren(QtWidgets.QComboBox)
    children_list = l1 + l2 + l3 + l4

    rows = []
    for child in children_list:
        if not child.objectName() == "qt_spinbox_lineedit":
            # if QComboBox or QSpinBox/QLineEdit (combobox doesn't have readonly property)
            if hasattr(child, "currentIndex") or not child.isReadOnly():
                if hasattr(child, "value"):  # QSpinBox
                    val = child.value()
                elif hasattr(child, "currentIndex"):  # QComboBox
                    val = child.currentIndex()
                else:  # QLineEdit
                    val = child.text()
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
    # TODO: add exception handeling for when a .ui file was changed since loadout was saved

    rows = csv_rows_as_list(file_path)

    for row in rows:
        wdgt_name, val = row
        child = gui_parent.findChild(QtWidgets.QWidget, wdgt_name)
        if not child == "nullptr":
            if hasattr(child, "value"):  # QSpinBox
                child.setValue(float(val))
            elif hasattr(child, "currentIndex"):  # QComboBox
                child.setCurrentIndex(int(val))
            elif hasattr(child, "text"):  # QLineEdit
                child.setText(val)


def deep_getattr(object, deep_name, default=None):
    """
    Get deep attribute of object. Useful for dynamically-set deep attributes.

    Example usage: a = deep_getattr(obj, "sobj.ssobj.a")
    """

    deep_obj = object
    for attr in deep_name.split("."):
        deep_obj = getattr(deep_obj, attr)
    return deep_obj


def div_ceil(x: int, y: int) -> int:
    """Returns x divided by y rounded towards positive infinity"""

    return int(x // y + (x % y > 0))


def limit(val: float, min: float, max: float) -> float:

    if min <= val <= max:
        return val
    elif val < min:
        return min
    return max


def get_datetime_str() -> str:
    """Return a date and time string in the format DDMMYY_HHMMSS"""

    return datetime.datetime.now().strftime("%d%m_%H%M%S")


def inv_dict(dct: dict) -> dict:
    """Inverts a Python dictionary. Expects mapping to be 1-to-1"""

    return {val: key for key, val in dct.items()}


@dataclass
class DeviceAttrs:

    class_name: str
    log_ref: str
    param_widgets: QtWidgetCollection
    cls_xtra_args: List[str] = None
    led_icon_path: str = icon_path.LED_GREEN


@dataclass
class ImageData:

    pic1: np.ndarray
    norm1: np.ndarray
    pic2: np.ndarray
    norm2: np.ndarray
    line_ticks_V: np.ndarray
    row_ticks_V: np.ndarray


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
                self.vLine.setPos(mousePoint.x() + 0.5)
                self.hLine.setPos(mousePoint.y())
