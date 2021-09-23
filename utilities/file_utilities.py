"""Data-File Loading Utilities"""

import glob
import os
import pickle
import re
from contextlib import suppress
from typing import List, Set

import numpy as np
import scipy.io as spio

from utilities.helper import reverse_dict

legacy_matlab_trans_dict = {
    # Solution Scan
    "Setup": "setup",
    "AfterPulseParam": "after_pulse_param",
    "AI_ScalingXYZ": "ai_scaling_xyz",
    "XYZ_um_to_V": "xyz_um_to_v",
    "SystemInfo": "system_info",
    "FullData": "full_data",
    "X": "x",
    "Y": "y",
    "ActualSpeed": "actual_speed_um_s",
    "ScanFreq": "scan_freq_hz",
    "SampleFreq": "sample_freq_hz",
    "PointsPerLineTotal": "points_per_line_total",
    "PointsPerLine": "points_per_line",
    "NofLines": "n_lines",
    "LineLength": "line_length",
    "LengthTot": "total_length",
    "LineLengthMax": "max_line_length_um",
    "LineShift": "line_shift_um",
    "AngleDegrees": "angle_degrees",
    "LinFrac": "linear_frac",
    "LinearPart": "linear_part",
    "Xlim": "x_lim",
    "Ylim": "y_lim",
    "Data": "data",
    "AvgCnt": "avg_cnt_rate_khz",
    "CircleSpeed_um_sec": "circle_speed_um_s",
    "AnglularScanSettings": "angular_scan_settings",
    # Image Scan
    "Cnt": "cnt",
    "PID": "pid",
    "SP": "scan_param",
    "LinesOdd": "lines_odd",
    "FastScan": "is_fast_scan",
    "TdcScanData": "tdc_scan_data",
    "Plane": "plane",
    "ScanParam": "scan_param",
    "Dimension1_lines_um": "dim1_lines_um",
    "Dimension2_col_um": "dim2_col_um",
    "Dimension3_um": "dim3_um",
    "Line": "lines",
    "Planes": "planes",
    "Line_freq_Hz": "line_freq_hz",
    "Points_per_Line": "ppl",
    "ScanType": "scan_type",
    "Offset_AOX": "offset_aox",
    "Offset_AOY": "offset_aoy",
    "Offset_AOZ": "offset_aoz",
    "Offset_AIX": "offset_aix",
    "Offset_AIY": "offset_aiy",
    "Offset_AIZ": "offset_aiz",
    "whatStage": "what_stage",
    "LinFrac": "linear_frac",
    # General
    "Version": "version",
    "AI": "ai",
    "AO": "ao",
    "FpgaFreq": "fpga_freq_mhz",
    "DataVersion": "data_version",
    "PixelFreq": "pix_clk_freq_mhz",
    "PixClockFreq": "pix_clk_freq_mhz",
    "LaserFreq": "laser_freq_mhz",
}

legacy_python_trans_dict = {
    # Solution
    "fpga_freq": "fpga_freq_mhz",
    "pix_clk_freq": "pix_clk_freq_mhz",
    "laser_freq": "laser_freq_mhz",
    "circle_speed_um_sec": "circle_speed_um_s",
    "actual_speed": "actual_speed_um_s",
    "scan_freq": "scan_freq_hz",
    "sample_freq": "sample_freq_hz",
    "max_line_length": "max_line_length_um",
    "line_shift": "line_shift_um",
    # Image
    "cnt": "ci",
    "lines": "n_lines",
    "planes": "n_planes",
    "points_per_line": "ppl",
    "scan_type": "scan_plane",
    "lines_odd": "set_pnts_lines_odd",  # moved to scan_param
    # General
}

default_system_info = {
    "setup": "STED with galvos",
    "after_pulse_param": (
        "multi_exponent_fit",
        1e5
        * np.array(
            [
                0.183161051158731,
                0.021980256326163,
                6.882763042785681,
                0.154790280034295,
                0.026417532300439,
                0.004282749744374,
                0.001418363840077,
                0.000221275818533,
            ]
        ),
    ),
    "ai_scaling_xyz": (1.243, 1.239, 1),
    "xyz_um_to_v": (70.81, 82.74, 10.0),
}


def estimate_disk_size(obj) -> float:
    """Returns the estimated size in bytes."""

    return len(pickle.dumps(obj))


def save_processed_solution_meas(full_data, dir_path) -> None:
    """
    Save a processed measurement, lacking any raw data.
    The template may then be loaded much more quickly.
    """

    # lower size if possible
    for p in full_data.data:
        if p.runtime.max() <= np.iinfo(np.int32).max:
            p.runtime = p.runtime.astype(np.int32)

    #    print(f"full_data size estimate: {round(estimate_disk_size(full_data) / 1e6)} Mb")
    #    [
    #        print(f"{key}: {size} Mb ({val.dtype})")
    #        for key, val in full_data.data[0].__dict__.items()
    #        if (size := round(estimate_disk_size(val) / 1e6)) > 0
    #    ]

    os.makedirs(os.path.join(dir_path, "processed"), exist_ok=True)
    file_path = os.path.join(dir_path, "processed", re.sub("_[*]", "", full_data.template))
    with open(file_path, "wb") as f:
        pickle.dump(full_data, f)


def load_file_dict(file_path: str):
    """
    Load files according to extension,
    Allow backwards compatibility with legacy dictionary keys (relevant for both .mat and .pkl files),
    use defaults for legacy files where 'system_info' or 'after_pulse_param' is not iterable (therefore old).
    """

    ext = file_extension(file_path)
    if ext == ".pkl":
        file_dict = translate_dict_keys(load_pkl(file_path), legacy_python_trans_dict)
    elif ext == ".mat":
        file_dict = translate_dict_keys(load_mat(file_path), legacy_matlab_trans_dict)
    else:
        raise NotImplementedError(f"Unknown file extension: '{ext}'.")

    if not file_dict.get("system_info"):
        print("'system_info' is missing, using defaults...", end=" ")
        file_dict["system_info"] = default_system_info
    else:
        with suppress(KeyError):
            if not isinstance(file_dict["system_info"]["after_pulse_param"], tuple):
                file_dict["system_info"]["after_pulse_param"] = (
                    "multi_exponent_fit",
                    file_dict["system_info"]["after_pulse_param"],
                )
                # TODO: check somehow that this is indeed meant foor "multi_exponent_fit"? ask Oleg.

    return file_dict


def translate_dict_keys(original_dict: dict, translation_dict: dict) -> dict:
    """
    Updates keys of dict according to another dict:
    trans_dct.keys() are the keys to update,
    and trans_dct.values() are the new keys.
    Key, value pairs that do not appear in translation_dict will remain unchanged.
    """

    translated_dict = {}
    # iterate over key, val pairs of original dict
    for org_key, org_val in original_dict.items():
        if isinstance(org_val, dict):
            # if the original val is a dict
            if org_key in translation_dict.keys():
                # if the dict name needs translation, translate it and then translate the sub_dict
                translated_dict[translation_dict[org_key]] = translate_dict_keys(
                    org_val, translation_dict
                )
            else:
                # translate the sub_dict
                translated_dict[org_key] = translate_dict_keys(org_val, translation_dict)
        else:
            # translate the key if needed
            if org_key in translation_dict.keys():
                translated_dict[translation_dict[org_key]] = org_val
            else:
                translated_dict[org_key] = org_val
    return translated_dict


def convert_types_to_matlab_format(obj, key_name=None):
    """
    Recursively converts any list/tuple in dictionary and any sub-dictionaries to Numpy ndarrays.
    Converts integers to floats.
    """

    # stop condition
    if not isinstance(obj, dict):
        if isinstance(obj, (list, tuple)):
            return np.array(obj, dtype=np.double)
        elif isinstance(obj, int):
            return float(obj)
        else:
            return obj

    # recursion
    return {key: convert_types_to_matlab_format(val, key_name=str(key)) for key, val in obj.items()}


def save_mat(file_dict: dict, file_path: str) -> None:
    """
    Saves 'file_dict' as 'file_path', after converting all keys to legacy naming,
    takes care of converting 'AfterPulseParam' to old style (array only, no type),
    and all lists to Numpy arrays."""

    file_dict = translate_dict_keys(file_dict, reverse_dict(legacy_matlab_trans_dict))
    file_dict["SystemInfo"]["AfterPulseParam"] = file_dict["SystemInfo"]["AfterPulseParam"][1]
    file_dict = convert_types_to_matlab_format(file_dict)
    spio.savemat(file_path, file_dict)


def load_mat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    adapted from: https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    """

    def _check_keys(d):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs nested dictionaries from matobjects
        """
        d = {}
        for name in matobj._fieldnames:
            elem = matobj.__dict__[name]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[name] = _todict(elem)
            elif isinstance(elem, np.ndarray) and name not in {"Data", "data"}:
                d[name] = _tolist(elem)
            else:
                d[name] = elem
        return d

    def _tolist(ndarray):
        """
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def load_pkl(file_path: str) -> dict:
    """Short cut for opening .pkl files"""

    with open(file_path, "rb") as f:
        return pickle.load(f)


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


def file_selection_str_to_list(file_selection: str) -> (List[int], bool):
    """Doc."""

    def range_str_to_list(range_str: str) -> Set[int]:
        """
        Accept a range string and returns a list of integers, lowered by 1 to match indices.
        Examples:
            range_string_to_list('1-3') -> [0, 1, 2]
            range_string_to_list('2') -> [1]
        """

        try:
            range_edges = [int(s) if int(s) != 0 else 1 for s in range_str.split("-")]
            if len(range_edges) > 2:
                raise ValueError
            elif len(range_edges) == 2:
                range_start, range_end = range_edges
                return list(range(range_start - 1, range_end))
            else:  # len(range_edges) == 1
                single_int, *_ = range_edges
                return [single_int - 1]
        except (ValueError, IndexError):
            raise ValueError(f"Bad range format: '{range_str}'.")

    _, choice, file_selection = re.split("(Use|Don't Use)", file_selection)
    file_idxs = {
        file_num
        for range_str in file_selection.split(",")
        for file_num in range_str_to_list(range_str)
    }

    return file_idxs, choice


def prepare_file_paths(file_path_template: str, file_selection: str) -> List[str]:
    """Doc."""

    file_paths = sort_file_paths_by_file_number(glob.glob(file_path_template))
    if not file_paths:
        raise FileNotFoundError(f"File template path ('{file_path_template}') does not exist!")

    if file_selection:
        try:
            file_idxs, choice = file_selection_str_to_list(file_selection)
            if choice == "Use":
                file_paths = [file_paths[i] for i in file_idxs]
            else:
                file_paths = [file_paths[i] for i in range(len(file_paths)) if i not in file_idxs]
            print(f'(files: "{file_selection}")')
        except IndexError:
            raise ValueError(
                f"Bad file selection string: '{file_selection}'. Try file numbers between 1 and {len(file_paths)}."
            )
    else:
        print("(all files)")

    return file_paths
