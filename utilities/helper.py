# -*- coding: utf-8 -*-
"""Miscellaneous helper functions/classes"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, NoReturn, Union

import pandas as pd
import PyQt5.QtWidgets as QtWidgets

import gui.icons.icon_paths as icon_path
import logic.app as app_module


class QtWidgetAccess:
    def __init__(self, obj_name: str, getter: str, gui_parent_name: str = "settings"):
        self.obj_name = obj_name
        self.getter = getter
        first_char, *rest_of_str = self.getter
        self.setter = "set" + first_char.upper() + "".join(rest_of_str)
        self.gui_parent_name = gui_parent_name

    def hold_obj(self, parent_gui) -> QtWidgetAccess:
        """Save the actual widget object as an attribute"""

        self.obj = getattr(parent_gui, self.obj_name)
        return self

    def get(self, parent_gui=None) -> Union[int, float, str]:
        """Get widget value"""

        wdgt = self.obj if parent_gui is None else getattr(parent_gui, self.obj_name)
        return getattr(wdgt, self.getter)()

    def set(self, arg, parent_gui=None) -> NoReturn:
        """Set widget property"""
        wdgt = self.obj if parent_gui is None else getattr(parent_gui, self.obj_name)
        getattr(wdgt, self.setter)(arg)


class QtWidgetCollection:
    """Doc."""

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

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

    def read_dict_from_gui(
        self, app: app_module.App
    ) -> Dict[Union[int, float, str, QtWidgetAccess]]:
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

    def hold_objects(
        self, app: app_module.App, wdgt_name_list: List[str] = None, hold_all=False
    ) -> QtWidgetCollection:
        """Stores the actual GUI object in the listed, or all, widgets."""

        if wdgt_name_list is not None:
            for wdgt_name in wdgt_name_list:
                wdgt = getattr(self, wdgt_name)
                parent_gui = getattr(app.gui, wdgt.gui_parent_name)
                wdgt.hold_obj(parent_gui)

        elif hold_all:
            for wdgt in vars(self).values():
                parent_gui = getattr(app.gui, wdgt.gui_parent_name)
                wdgt.hold_obj(parent_gui)

        return self


@dataclass
class DeviceAttrs:

    cls_name: str
    log_ref: str
    led_widget: QtWidgetAccess
    param_widgets: QtWidgetCollection
    cls_xtra_args: List[str] = None
    led_icon_path: str = icon_path.LED_GREEN
    switch_widget: QtWidgetAccess = None


def gui_to_csv(gui_parent, file_path):
    """Doc."""

    with open(file_path, "w") as f:
        # get all names of fields in settings window as lists (for file saving/loading)
        l1 = gui_parent.findChildren(QtWidgets.QLineEdit)
        l2 = gui_parent.findChildren(QtWidgets.QSpinBox)
        l3 = gui_parent.findChildren(QtWidgets.QDoubleSpinBox)
        l4 = gui_parent.findChildren(QtWidgets.QComboBox)
        children_list = l1 + l2 + l3 + l4

        obj_names = []
        for child in children_list:
            if not child.objectName() == "qt_spinbox_lineedit":
                if hasattr(child, "currentIndex"):  # QComboBox
                    obj_names.append(child.objectName())
                elif not child.isReadOnly():  # QSpinBox, QLineEdit
                    obj_names.append(child.objectName())

        writer = csv.writer(f)
        for i in range(len(obj_names)):
            child = gui_parent.findChild(QtWidgets.QWidget, obj_names[i])
            if hasattr(child, "value"):  # QSpinBox
                val = child.value()
            elif hasattr(child, "currentIndex"):  # QComboBox
                val = child.currentIndex()
            else:  # QLineEdit
                val = child.text()
            writer.writerow([obj_names[i], val])


def csv_to_gui(file_path, gui_parent):
    """Doc."""
    # TODO: add exception handeling for when a .ui file was changed since loadout was saved

    df = pd.read_csv(
        file_path,
        header=None,
        delimiter=",",
        keep_default_na=False,
        error_bad_lines=False,
    )

    for i in range(len(df)):
        obj_name, obj_val = df.iloc[i, 0], df.iloc[i, 1]
        child = gui_parent.findChild(QtWidgets.QWidget, obj_name)
        if not child == "nullptr":
            if hasattr(child, "value"):  # QSpinBox
                child.setValue(float(obj_val))
            elif hasattr(child, "currentIndex"):  # QComboBox
                child.setCurrentIndex(int(obj_val))
            elif hasattr(child, "text"):  # QLineEdit
                child.setText(obj_val)


def deep_getattr(object, deep_name, default=None):
    """
    Get deep attribute of object.
    Example usage: a = deep_getattr(obj, "sobj.ssobj.a")
    """

    deep_obj = object
    for attr in deep_name.split("."):
        deep_obj = getattr(deep_obj, attr)
    return deep_obj
