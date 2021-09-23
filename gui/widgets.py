""" Widget Collections """

from __future__ import annotations

import logging
import re
from contextlib import suppress
from types import SimpleNamespace
from typing import List, Union

import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtGui import QIcon

from gui.icons import icon_paths_dict
from utilities import helper


class QtWidgetAccess:
    """Doc."""

    def __init__(self, obj_name: str, widget_class: str, gui_parent_name: str, does_hold_obj: bool):
        self.obj_name = obj_name
        self.getter, self.setter, _ = _getter_setter_type_dict.get(widget_class, (None,) * 3)
        self.gui_parent_name = gui_parent_name
        self.does_hold_obj = does_hold_obj

    def hold_widget(self, parent_gui) -> QtWidgetAccess:
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

    def hold_widgets(self, app) -> QtWidgetCollection:
        """Stores the actual GUI object in all widgets (for which does_hold_obj is True)."""

        for wdgt_access in vars(self).values():
            parent_gui = getattr(app.gui, wdgt_access.gui_parent_name)
            wdgt_access.hold_widget(parent_gui)
        return self

    def clear_all_objects(self) -> None:
        """Issues a clear() command for all widgets held (clears the GUI)"""

        for wdgt_access in vars(self).values():
            if wdgt_access.does_hold_obj:
                with suppress(AttributeError):
                    wdgt_access.obj.clear()

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

    def read_gui(self, app_obj, out="namespace") -> SimpleNamespace:
        """
        Read values from QtWidgetAccess objects, which are the attributes of self and return a namespace.
        If a QtWidgetAccess object holds the actual GUI object, the dict will contain the
        QtWidgetAccess object itself instead of the value (for getting/setting live values).
        """

        wdgt_val = SimpleNamespace()
        for attr_name, wdgt in vars(self).items():
            if hasattr(wdgt, "obj"):
                setattr(wdgt_val, attr_name, wdgt)
            else:
                parent_gui = getattr(app_obj.gui, wdgt.gui_parent_name)
                setattr(wdgt_val, attr_name, wdgt.get(parent_gui))

        if out == "namespace":
            return wdgt_val
        elif out == "dict":
            return vars(wdgt_val).copy()
        else:
            raise ValueError(
                f"Keyword argument 'out' ({out}) accepts either 'namespace' or 'dict'."
            )


def wdgt_items_to_text_lines(parent_wdgt, widget_types: list) -> List[str]:
    """Doc."""

    children_class_lists = [
        parent_wdgt.findChildren(getattr(QtWidgets, wdgt_type)) for wdgt_type in widget_types
    ]
    children_list = [child for child_list in children_class_lists for child in child_list]

    lines = []
    for child in children_list:
        child_class = child.__class__.__name__
        child_name = child.objectName()
        getter, _, _ = _getter_setter_type_dict.get(child_class, (None,) * 3)

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


def write_gui_to_file(parent_wdgt, widget_types, file_path):
    """Doc."""

    helper.write_list_to_file(file_path, wdgt_items_to_text_lines(parent_wdgt, widget_types))


def read_file_to_gui(file_path, gui_parent):
    """Doc."""

    lines = helper.read_file_to_list(file_path)

    for line in lines:
        wdgt_name, val = re.split(",", line, maxsplit=1)
        child = gui_parent.findChild(QtWidgets.QWidget, wdgt_name)
        _, setter, type_func = _getter_setter_type_dict.get(child.__class__.__name__, (None,) * 3)

        if type_func not in {None, str}:
            val = type_func(val)

        try:
            getattr(child, setter)(val)
        except TypeError:
            logging.warning(
                f"read_file_to_gui(): Child widget '{wdgt_name}' was not found in parent widget '{gui_parent.objectName()}' - probably removed from GUI. Overwrite the defaults to stop seeing this warning."
            )


def get_icon_paths():
    return {key: QIcon(val) for key, val in icon_paths_dict.items()}


MAIN_TYPES = [
    "QLineEdit",
    "QSpinBox",
    "QDoubleSpinBox",
    "QComboBox",
    "QStackedWidget",
    "QRadioButton",
    "QSlider",
    "QTabWidget",
    "QCheckBox",
    "QToolBox",
    "QDial",
]

SETTINGS_TYPES = [
    "QLineEdit",
    "QSpinBox",
    "QDoubleSpinBox",
    "QComboBox",
    "QRadioButton",
    "QCheckBox",
]

# What to do with each widget class
_getter_setter_type_dict = {
    "QLabel": ("text", "setText", str),
    "QComboBox": ("currentText", "setCurrentText", str),
    "QTabWidget": ("currentIndex", "setCurrentIndex", int),
    "QCheckBox": ("isChecked", "setChecked", helper.bool_str),
    "QRadioButton": ("isChecked", "setChecked", helper.bool_str),
    "QSlider": ("value", "setValue", int),
    "QSpinBox": ("value", "setValue", int),
    "QDoubleSpinBox": ("value", "setValue", float),
    "QLineEdit": ("text", "setText", str),
    "QPlainTextEdit": ("toPlainText", "setPlainText", str),
    "QButtonGroup": ("checkedButton", None, None),
    "QTimeEdit": ("time", "setTime", None),
    "QIcon": ("icon", "setIcon", None),
    "QStackedWidget": ("currentIndex", "setCurrentIndex", int),
    "QToolBox": ("currentIndex", "setCurrentIndex", int),
    "QDial": ("value", "setValue", int),
}

# ------------------------------
# Devices
# ------------------------------

LED_COLL = QtWidgetCollection(
    exc=("ledExc", "QIcon", "main", True),
    dep=("ledDep", "QIcon", "main", True),
    shutter=("ledShutter", "QIcon", "main", True),
    stage=("ledStage", "QIcon", "main", True),
    tdc=("ledTdc", "QIcon", "main", True),
    cam=("ledCam", "QIcon", "main", True),
    scnrs=("ledScn", "QIcon", "main", True),
    cntr=("ledCounter", "QIcon", "main", True),
    um232=("ledUm232h", "QIcon", "main", True),
    pxl_clk=("ledPxlClk", "QIcon", "main", True),
)

SWITCH_COLL = QtWidgetCollection(
    exc=("excOnButton", "QIcon", "main", True),
    dep=("depEmissionOn", "QIcon", "main", True),
    shutter=("depShutterOn", "QIcon", "main", True),
    stage=("stageOn", "QIcon", "main", True),
)

# ----------------------------------------------
# Analysis Widget Collections
# ----------------------------------------------
# TODO: change unneeded widgets to 'False', and implement using "read widgets" instead of manually reading each one using .get()
# Widgets that need .obj should be 'True'

DATA_IMPORT_COLL = QtWidgetCollection(
    is_image_type=("imageDataImport", "QRadioButton", "main", False),
    is_solution_type=("solDataImport", "QRadioButton", "main", False),
    data_days=("dataDay", "QComboBox", "main", True),
    data_months=("dataMonth", "QComboBox", "main", True),
    data_years=("dataYear", "QComboBox", "main", True),
    data_templates=("dataTemplate", "QComboBox", "main", True),
    n_files=("dataNumFiles", "QLabel", "main", True),
    new_template=("newTemplate", "QLineEdit", "main", False),
    import_stacked=("importStacked", "QStackedWidget", "main", True),
    log_text=("dataDirLog", "QPlainTextEdit", "main", True),
    img_preview_disp=("imgScanPreviewDisp", None, "main", False),
    sol_save_processed=("solImportSaveProcessed", "QCheckBox", "main", False),
    sol_use_processed=("solImportLoadProcessed", "QCheckBox", "main", False),
    sol_file_dicrimination=("fileSelectionGroup", "QButtonGroup", "main", False),
    sol_file_use_or_dont=("solImportUseDontUse", "QComboBox", "main", False),
    sol_file_selection=("solImportFileSelectionPattern", "QLineEdit", "main", False),
)

SOL_ANALYSIS_COLL = QtWidgetCollection(
    fix_shift=("solDataFixShift", "QCheckBox", "main", False),
    external_plotting=("solDataExtPlot", "QCheckBox", "main", False),
    scan_image_disp=("solScanImgDisp", None, "main", True),
    row_dicrimination=("rowDiscriminationGroup", "QButtonGroup", "main", False),
    remove_over=("solAnalysisRemoveOverSpinner", "QDoubleSpinBox", "main", False),
    remove_worst=("solAnalysisRemoveWorstSpinner", "QDoubleSpinBox", "main", False),
    plot_spatial=("solAveragingPlotSpatial", "QRadioButton", "main", False),
    plot_temporal=("solAveragingPlotTemporal", "QRadioButton", "main", False),
    row_acf_disp=("solScanAcfDisp", None, "main", True),
    imported_templates=("importedSolDataTemplates", "QComboBox", "main", True),
    scan_img_file_num=("scanImgFileNum", "QSpinBox", "main", True),
    scan_settings=("solAnalysisScanSettings", "QPlainTextEdit", "main", True),
    scan_duration_min=("solAnalysisDur", "QDoubleSpinBox", "main", True),
    n_files=("solAnalysisNumFiles", "QSpinBox", "main", True),
    avg_cnt_rate_khz=("solAnalysisCountRate", "QDoubleSpinBox", "main", True),
    mean_g0=("solAnalysisMeanG0", "QDoubleSpinBox", "main", True),
    mean_tau=("solAnalysisMeanTau", "QDoubleSpinBox", "main", True),
    n_good_rows=("solAnalysisGoodRows", "QSpinBox", "main", True),
    n_bad_rows=("solAnalysisBadRows", "QSpinBox", "main", True),
)
# ----------------------------------------------
# Measurement Widget Collections
# ----------------------------------------------

SOL_ANG_SCAN_COLL = QtWidgetCollection(
    max_line_len_um=("maxLineLen", "QDoubleSpinBox", "main", False),
    ao_samp_freq_Hz=("angAoSampFreq", "QDoubleSpinBox", "main", False),
    angle_deg=("angle", "QSpinBox", "main", False),
    lin_frac=("solLinFrac", "QDoubleSpinBox", "main", False),
    line_shift_um=("lineShift", "QSpinBox", "main", False),
    speed_um_s=("solAngScanSpeed", "QSpinBox", "main", False),
    min_lines=("minNumLines", "QSpinBox", "main", False),
    max_scan_freq_Hz=("maxScanFreq", "QSpinBox", "main", False),
)

SOL_CIRC_SCAN_COLL = QtWidgetCollection(
    ao_samp_freq_Hz=("circAoSampFreq", "QSpinBox", "main", False),
    diameter_um=("circDiameter", "QDoubleSpinBox", "main", False),
    speed_um_s=("circSpeed", "QSpinBox", "main", False),
)

SOL_MEAS_COLL = QtWidgetCollection(
    scan_type=("solScanType", "QComboBox", "main", False),
    file_template=("solScanFileTemplate", "QLineEdit", "main", False),
    regular=("regularSolMeas", "QRadioButton", "main", False),
    repeat=("repeatSolMeas", "QRadioButton", "main", False),
    final=("finalSolMeas", "QRadioButton", "main", False),
    save_path=("dataPath", "QLineEdit", "settings", False),
    sub_dir_name=("solSubdirName", "QLineEdit", "settings", False),
    max_file_size_mb=("solScanMaxFileSize", "QDoubleSpinBox", "main", False),
    duration=("solScanDur", "QDoubleSpinBox", "main", False),
    duration_units=("solScanDurUnits", "QComboBox", "main", False),
    prog_bar_wdgt=("solScanProgressBar", "QSlider", "main", True),
    start_time_wdgt=("solScanStartTime", "QTimeEdit", "main", True),
    end_time_wdgt=("solScanEndTime", "QTimeEdit", "main", True),
    time_left_wdgt=("solScanTimeLeft", "QSpinBox", "main", True),
    file_num_wdgt=("solScanFileNo", "QSpinBox", "main", True),
    pattern_wdgt=("solScanPattern", None, "main", True),
    g0_wdgt=("g0", "QDoubleSpinBox", "main", True),
    tau_wdgt=("decayTime", "QDoubleSpinBox", "main", True),
    plot_wdgt=("solScanAcf", None, "main", True),
    fit_led=("ledFit", "QIcon", "main", True),
)

IMG_SCAN_COLL = QtWidgetCollection(
    scan_plane=("imgScanType", "QComboBox", "main", False),
    dim1_lines_um=("imgScanDim1", "QDoubleSpinBox", "main", False),
    dim2_col_um=("imgScanDim2", "QDoubleSpinBox", "main", False),
    dim3_um=("imgScanDim3", "QDoubleSpinBox", "main", False),
    n_lines=("imgScanNumLines", "QSpinBox", "main", False),
    ppl=("imgScanPPL", "QSpinBox", "main", False),
    line_freq_Hz=("imgScanLineFreq", "QSpinBox", "main", False),
    lin_frac=("imgScanLinFrac", "QDoubleSpinBox", "main", False),
    n_planes=("imgScanNumPlanes", "QSpinBox", "main", False),
    curr_ao_x=("xAOVint", "QDoubleSpinBox", "main", False),
    curr_ao_y=("yAOVint", "QDoubleSpinBox", "main", False),
    curr_ao_z=("zAOVint", "QDoubleSpinBox", "main", False),
    auto_cross=("autoCrosshair", "QCheckBox", "main", False),
)

IMG_MEAS_COLL = QtWidgetCollection(
    file_template=("imgScanFileTemplate", "QLineEdit", "main", False),
    save_path=("dataPath", "QLineEdit", "settings", False),
    sub_dir_name=("imgSubdirName", "QLineEdit", "settings", False),
    prog_bar_wdgt=("imgScanProgressBar", "QSlider", "main", True),
    curr_plane_wdgt=("currPlane", "QSpinBox", "main", True),
    plane_shown=("numPlaneShown", "QSpinBox", "main", True),
    plane_choice=("numPlaneShownChoice", "QSlider", "main", True),
    image_wdgt=("imgScanPlot", None, "main", True),
    pattern_wdgt=("imgScanPattern", None, "main", True),
    scale_image=("scaleImgScan", "QDial", "main", True),
    image_method=("imgShowMethod", "QComboBox", "main", False),
)