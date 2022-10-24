""" Widget Collections """

from __future__ import annotations

import logging
import re
from contextlib import suppress
from typing import Any, Dict, List, Tuple, Union

import PyQt5.QtWidgets as QtWidgets

from utilities import helper


class QtWidgetAccess:
    """Doc."""

    def __init__(self, obj_name: str, widget_class: str, gui_parent_name: str, does_hold_obj: bool):
        self.obj_name = obj_name
        self.getter: str
        self.setter: str
        self.getter, self.setter, _ = _getter_setter_type_dict.get(widget_class, (None,) * 3)
        self.gui_parent_name = gui_parent_name
        self.does_hold_obj = does_hold_obj

    def hold_widget(self, parent_gui) -> QtWidgetAccess:
        """Save the actual widget object as an attribute"""

        if self.does_hold_obj:
            self.obj = getattr(parent_gui, self.obj_name)
        return self

    def get(self, parent_gui=None) -> Union[helper.Number, str]:
        """Get widget value"""

        if self.getter is not None:
            wdgt = self.obj if parent_gui is None else getattr(parent_gui, self.obj_name)
            return getattr(wdgt, self.getter)()
        else:
            return None

    def set(self, arg, parent_gui=None) -> None:
        """Set widget property"""

        wdgt = self.obj if self.does_hold_obj else getattr(parent_gui, self.obj_name)
        try:
            getattr(wdgt, self.setter)(arg)
        except TypeError:
            raise TypeError(
                f"widget '{self.obj_name}' was attempted to be set for bad value '{arg}'"
            )


class QtWidgetCollection:
    """Doc."""

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, QtWidgetAccess(*val))

    def __getattr__(self, name: str) -> QtWidgetAccess:
        ...

    def hold_widgets(self, gui) -> QtWidgetCollection:
        """Stores the actual GUI object in all widgets (for which does_hold_obj is True)."""

        for wdgt_access in vars(self).values():
            parent_gui = getattr(gui, wdgt_access.gui_parent_name)
            wdgt_access.hold_widget(parent_gui)
        return self

    def clear_all_objects(self) -> None:
        """Issues a clear() command for all widgets held (clears the GUI)"""

        for wdgt_access in vars(self).values():
            if wdgt_access.does_hold_obj:
                with suppress(AttributeError):
                    wdgt_access.obj.clear()

    def obj_to_gui(self, gui, obj) -> None:
        """
        Fill widget collection with values from namespace/dict/list, or a single value for all.
        if obj is a list, the values will be inserted in the order of vars(self).keys().
        """

        with suppress(TypeError):
            obj = vars(obj)

        if isinstance(obj, list):
            # convert list to dict
            obj = dict(zip(vars(self).keys(), obj))

        if isinstance(obj, dict):
            for attr_name, val in obj.items():
                with suppress(
                    AttributeError
                ):  # obj contains key with no matching wdgt in coll (should this pass silently?)
                    wdgt = getattr(self, attr_name)
                    parent_gui = getattr(gui, wdgt.gui_parent_name)
                    if not isinstance(val, QtWidgetAccess):
                        wdgt.set(val, parent_gui)
        else:
            for wdgt in vars(self).values():
                parent_gui = getattr(gui, wdgt.gui_parent_name)
                wdgt.set(obj, parent_gui)

    def gui_to_dict(self, gui) -> Dict[str, Any]:
        """
        Read values from QtWidgetAccess objects, which are the attributes of self and return a dictionary.
        If a QtWidgetAccess object holds the actual GUI object, the dict will contain the
        QtWidgetAccess object itself instead of the value (for getting/setting live values).
        """

        wdgt_val = {}
        for attr_name, wdgt in vars(self).items():
            if hasattr(wdgt, "obj"):
                wdgt_val[attr_name] = wdgt
            else:
                parent_gui = getattr(gui, wdgt.gui_parent_name)
                wdgt_val[attr_name] = wdgt.get(parent_gui)

        return wdgt_val


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

    helper.list_to_file(file_path, wdgt_items_to_text_lines(parent_wdgt, widget_types))


def read_file_to_gui(file_path, gui_parent):
    """Doc."""

    lines = helper.file_to_list(file_path)

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
_getter_setter_type_dict: Dict[str, Tuple[str, str, type]] = {
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
    cam1=("ledCam1", "QIcon", "main", True),
    cam2=("ledCam2", "QIcon", "main", True),
    scnrs=("ledScn", "QIcon", "main", True),
    pxl_clk=("ledPxlClk", "QIcon", "main", True),
    cntr=("ledCounter", "QIcon", "main", True),
    um232=("ledUm232h", "QIcon", "main", True),
)

SWITCH_COLL = QtWidgetCollection(
    exc=("excOnButton", "QIcon", "main", True),
    dep=("depEmissionOn", "QIcon", "main", True),
    shutter=("depShutterOn", "QIcon", "main", True),
    stage=("stageOn", "QIcon", "main", True),
    cam1=("videoSwitch1", "QIcon", "main", True),
    cam2=("videoSwitch2", "QIcon", "main", True),
)

# ----------------------------------------------
# Analysis Widget Collections
# ----------------------------------------------

DATA_IMPORT_COLL = QtWidgetCollection(
    is_image_type=("imageDataImport", "QRadioButton", "main", False),
    is_solution_type=("solDataImport", "QRadioButton", "main", False),
    data_days=("dataDay", "QComboBox", "main", True),
    data_months=("dataMonth", "QComboBox", "main", True),
    data_years=("dataYear", "QComboBox", "main", True),
    data_templates=("dataTemplate1", "QComboBox", "main", True),
    n_files=("dataNumFiles", "QLabel", "main", True),
    new_template=("newTemplate", "QLineEdit", "main", False),
    import_stacked=("importStacked", "QStackedWidget", "main", True),
    analysis_stacked=("analysisStacked", "QStackedWidget", "main", True),
    log_text=("dataDirLog", "QPlainTextEdit", "main", True),
    img_preview_disp=("imgScanPreviewDisp", None, "main", True),
    sol_file_dicrimination=("fileSelectionGroup", "QButtonGroup", "main", False),
    sol_file_use_or_dont=("solImportUseDontUse", "QComboBox", "main", False),
    sol_file_selection=("solImportFileSelectionPattern", "QLineEdit", "main", False),
    fix_shift=("fixShift", "QCheckBox", "options", False),
    should_auto_roi=("autoRoi", "QCheckBox", "options", False),
    should_subtract_bg_corr=("subtractBgCorr", "QCheckBox", "options", False),
    afterpulsing_method=("afterpulsingRemovalMethod", "QComboBox", "options", False),
    hist_norm_factor=("histNormFactor", "QDoubleSpinBox", "options", False),
    gating_mechanism=("gatingMechanism", "QComboBox", "options", False),
    override_system_info=("useDefaultSysInfo", "QCheckBox", "options", False),
    should_re_correlate=("solImportReCorrelate", "QCheckBox", "main", False),
    auto_load_processed=("solImportAutoLoadProcessed", "QCheckBox", "main", False),
)

SOL_MEAS_ANALYSIS_COLL = QtWidgetCollection(
    scan_image_disp=("solScanImgDisp", None, "main", True),
    pattern_wdgt=("solScanAnalysisPattern", None, "main", True),
    row_dicrimination=("rowDiscriminationGroup", "QButtonGroup", "main", False),
    remove_over=("solAnalysisRemoveOverSpinner", "QDoubleSpinBox", "main", False),
    remove_worst=("solAnalysisRemoveWorstSpinner", "QDoubleSpinBox", "main", True),
    plot_spatial=("solAveragingPlotSpatial", "QRadioButton", "main", False),
    plot_temporal=("solAveragingPlotTemporal", "QRadioButton", "main", False),
    row_acf_disp=("solScanAcfDisp", None, "main", True),
    imported_templates=("importedSolDataTemplates1", "QComboBox", "main", True),
    scan_img_file_num=("scanImgFileNum", "QSpinBox", "main", True),
    scan_settings=("solAnalysisScanSettings", "QPlainTextEdit", "main", True),
    scan_duration_min=("solAnalysisDur", "QDoubleSpinBox", "main", True),
    n_files=("solAnalysisNumFiles", "QSpinBox", "main", True),
    avg_cnt_rate_khz=("solAnalysisAvgCountRate", "QDoubleSpinBox", "main", True),
    std_cnt_rate_khz=("solAnalysisStdCountRate", "QDoubleSpinBox", "main", True),
    mean_g0=("solAnalysisMeanG0", "QDoubleSpinBox", "main", True),
    mean_tau=("solAnalysisMeanTau", "QDoubleSpinBox", "main", True),
    n_good_rows=("solAnalysisGoodRows", "QSpinBox", "main", True),
    n_bad_rows=("solAnalysisBadRows", "QSpinBox", "main", True),
)

SOL_EXP_ANALYSIS_COLL = QtWidgetCollection(
    loaded_experiments=("loadedExperimentNames", "QComboBox", "main", True),
    # loading
    gui_display_loading=("solExpLoading", None, "main", True),
    should_assign_loaded=("assignLoadedMeas", "QRadioButton", "main", False),
    imported_templates=("importedSolDataTemplates2", "QComboBox", "main", True),
    should_assign_raw=("assignRawData", "QRadioButton", "main", False),
    data_templates=("dataTemplate1", "QComboBox", "main", True),
    assigned_confocal_template=("assignedExpConfMeas", "QLineEdit", "main", True),
    assigned_sted_template=("assignedExpStedMeas", "QLineEdit", "main", True),
    experiment_name=("experimentName", "QLineEdit", "main", True),
    # TDC calibration
    calibration_gating=("tdcCalibGating", "QSpinBox", "main", False),
    gui_display_tdc_cal=("solExpTDC1", None, "main", True),
    gui_display_comp_lifetimes=("solExpTDC2", None, "main", True),
    # gating
    custom_gate_to_assign_lt=("customGateLifetimes", "QDoubleSpinBox", "main", True),
    assigned_gates=("assignedGates", "QComboBox", "main", True),
    available_gates=("availableGates", "QComboBox", "main", True),
    gui_display_sted_gating=("solExpGSTED", None, "main", True),
    # structure factor
    g_min=("expStrctFctrGMin", "QDoubleSpinBox", "main", False),
    n_robust=("expStrctFctrRbstIntrpPnts", "QSpinBox", "main", False),
    # properties
    g0_ratio=("expPropRatio", "QDoubleSpinBox", "main", True),
    fluoresence_lifetime=("expPropLifetime", "QDoubleSpinBox", "main", True),
    laser_pulse_delay=("expPropLaserPulseDelay", "QDoubleSpinBox", "main", True),
    gate_ns=("expPropGstedGate", "QComboBox", "main", True),
    resolution_nm=("expPropGstedResolution", "QDoubleSpinBox", "main", True),
)
# -----------------------------------------------
# MeasurementProcedure Widget Collections
# -----------------------------------------------

SOL_ANG_SCAN_COLL = QtWidgetCollection(
    max_line_len_um=("maxLineLen", "QDoubleSpinBox", "main", False),
    ao_sampling_freq_hz=("angAoSampFreq", "QDoubleSpinBox", "main", False),
    angle_deg=("angle", "QSpinBox", "main", False),
    linear_fraction=("solLinFrac", "QDoubleSpinBox", "main", False),
    line_shift_um=("lineShift", "QSpinBox", "main", False),
    speed_um_s=("solAngScanSpeed", "QSpinBox", "main", False),
    min_lines=("minNumLines", "QSpinBox", "main", False),
    max_scan_freq_Hz=("maxScanFreq", "QSpinBox", "main", False),
    ppl=("solPointsPerLine", "QSpinBox", "main", True),
    samples_per_line=("solTotPointsPerLine", "QSpinBox", "main", True),
    eff_speed_um_s=("solAngActualSpeed", "QDoubleSpinBox", "main", True),
    n_lines=("solNumLines", "QSpinBox", "main", True),
    line_freq_hz=("solActualScanFreq", "QDoubleSpinBox", "main", True),
)

SOL_CIRC_SCAN_COLL = QtWidgetCollection(
    ao_sampling_freq_hz=("circAoSampFreq", "QSpinBox", "main", False),
    diameter_um=("circDiameter", "QDoubleSpinBox", "main", False),
    speed_um_s=("circSpeed", "QSpinBox", "main", False),
    n_circles=("numOfCircles", "QSpinBox", "main", True),
)

SOL_MEAS_COLL = QtWidgetCollection(
    file_template=("solScanFileTemplate", "QLineEdit", "main", False),
    save_path=("dataPath", "QLineEdit", "settings", False),
    sub_dir_name=("solSubdirName", "QLineEdit", "settings", False),
    prog_bar_wdgt=("solScanProgressBar", "QSlider", "main", True),
    scan_type=("solScanType", "QComboBox", "main", False),
    floating_z_amplitude_um=("solScanFloatingAmp", "QSpinBox", "main", False),
    regular=("regularSolMeas", "QRadioButton", "main", False),
    repeat=("repeatSolMeas", "QRadioButton", "main", False),
    final=("finalSolMeas", "QRadioButton", "main", False),
    max_file_size_mb=("solScanMaxFileSize", "QDoubleSpinBox", "main", False),
    duration=("solScanDur", "QDoubleSpinBox", "main", False),
    duration_units=("solScanDurUnits", "QComboBox", "main", False),
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
    plane_orientation=("imgScanType", "QComboBox", "main", False),
    dim1_um=("imgScanDim1", "QDoubleSpinBox", "main", False),
    dim2_um=("imgScanDim2", "QDoubleSpinBox", "main", False),
    dim3_um=("imgScanDim3", "QDoubleSpinBox", "main", False),
    n_lines=("imgScanNumLines", "QSpinBox", "main", False),
    ppl=("imgScanPPL", "QSpinBox", "main", False),
    line_freq_hz=("imgScanLineFreq", "QSpinBox", "main", False),
    linear_fraction=("imgScanLinFrac", "QDoubleSpinBox", "main", False),
    n_planes=("imgScanNumPlanes", "QSpinBox", "main", False),
    auto_cross=("autoCrosshair", "QCheckBox", "main", False),
)

IMG_MEAS_COLL = QtWidgetCollection(
    file_template=("imgScanFileTemplate", "QLineEdit", "main", False),
    save_path=("dataPath", "QLineEdit", "settings", False),
    sub_dir_name=("imgSubdirName", "QLineEdit", "settings", False),
    prog_bar_wdgt=("imgScanProgressBar", "QSlider", "main", True),
    always_save=("alwaysSaveImg", "QCheckBox", "main", False),
    curr_plane_wdgt=("currPlane", "QSpinBox", "main", True),
    plane_shown=("numPlaneShown", "QSpinBox", "main", True),
    plane_choice=("numPlaneShownChoice", "QSlider", "main", True),
    image_wdgt=("imgScanPlot", None, "main", True),
    pattern_wdgt=("imgScanPattern", None, "main", True),
    scale_image=("scaleImgScan", "QDial", "main", True),
    image_method=("imgShowMethod", "QComboBox", "main", False),
)
