"""Data-File Loading Utilities"""

from __future__ import annotations

import copy
import functools
import gzip
import logging
import pickle
import re

# from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import bloscpack
import numpy as np
import scipy.io as spio

from utilities.helper import Limits, reverse_dict

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
    "Data": "byte_data",
    "AvgCnt": "avg_cnt_rate_khz",
    "CircleSpeed_um_sec": "circle_speed_um_s",
    "AnglularScanSettings": "angular_scan_settings",
    # Image Scan
    "Cnt": "cnt",
    "PID": "pid",
    "SP": "scan_params",
    "LinesOdd": "lines_odd",
    "FastScan": "is_fast_scan",
    "TdcScanData": "tdc_scan_data",
    "Plane": "plane",
    "ScanParam": "scan_params",
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
    "data": "byte_data",
    # Image
    "cnt": "ci",
    "lines": "n_lines",
    "planes": "n_planes",
    "points_per_line": "ppl",
    "scan_type": "plane_orientation",
    "scan_plane": "plane_orientation",
    "lines_odd": "set_pnts_lines_odd",  # moved to scan_params
    # General
    "scan_param": "scan_params",
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


def estimate_bytes(obj) -> int:
    """Returns the estimated size in bytes."""

    return len(pickle.dumps(obj, protocol=-1))


def deep_size_estimate(obj, level=np.inf, indent=0, threshold_mb=0.01, name=None) -> None:
    """
     Print a cascading size description of object 'obj' up to level 'level'
    for objects (and subobjects) requiring an estimated disk space over 'threshold_mb'.

    for ease of use, copy these lines to somewhere in the code:
    from utilities.file_utilities import deep_size_estimate as dse # TESTESTEST
    dse(obj) # TESTESTEST
    """

    size_mb = estimate_bytes(obj) / 1e6
    if (size_mb > threshold_mb) and (level >= 0):
        if name is None:
            name = obj.__class__.__name__
            print(
                f"Displaying object of class '{name}' size tree up to level={level} with a threshold of {threshold_mb} Mb:"
            )
        try:
            type_str = str(obj.dtype)
        except AttributeError:
            type_str = type(obj).__name__
        print(f"{indent * ' '}{name} ({type_str}): {size_mb:.2f} Mb")

        if isinstance(obj, (list, tuple, set)):
            for idx, elem in enumerate(obj):
                deep_size_estimate(elem, level - 1, indent + 4, threshold_mb, f"{name}[{idx}]")
        if hasattr(obj, "__dict__"):
            # convert namespaces into dicts
            obj = copy.deepcopy(vars(obj))
        if isinstance(obj, dict):
            for key, val in obj.items():
                deep_size_estimate(val, level - 1, indent + 4, threshold_mb, key)
        else:
            return
    elif name is None:
        name = obj.__class__.__name__
        print(f"Object '{name}' size is below the threshold ({threshold_mb} Mb).")
    else:
        return


def save_object_to_disk(
    obj,
    file_path: Path,
    size_limits_mb: Limits = None,
    compression_method: str = None,  # "gzip" / "blosc"
) -> bool:
    """
    Save a pickle-serialized and optionally gzip/blosc-compressed object to disk, if estimated size is within the limits.
    Returns 'True' if saved, 'False' otherwise.
    """

    if size_limits_mb is not None:
        disk_size_mb = estimate_bytes(obj) / 1e6
        if disk_size_mb not in size_limits_mb:
            logging.debug(
                f"Object of class '{obj.__class__.__name__}' was not saved (estimated size ({disk_size_mb}) is not in {size_limits_mb})"
            )
            return False

    dir_path = file_path.parent
    Path.mkdir(dir_path, parents=True, exist_ok=True)

    #    print("\npre-compression size evaluation:") # TESTESTEST
    #    deep_size_estimate(obj) # TESTESTEST

    # pickle/serialize the object
    obj = pickle.dumps(obj, protocol=-1)

    if compression_method == "blosc":
        # blosc compress the serialized object
        blosc_args = bloscpack.BloscArgs(typesize=4, clevel=1, cname="zlib")
        obj = bloscpack.pack_bytes_to_bytes(obj, blosc_args=blosc_args)

    elif compression_method == "gzip":
        # gzip compress the serialized object
        obj = gzip.compress(obj, compresslevel=1)

    # save the object
    with open(file_path, "wb") as f:
        f.write(obj)

    #    if compression_method is not None: # TESTESTEST
    #        print("\npost-compression size evaluation:") # TESTESTEST
    #        deep_size_estimate(obj) # TESTESTEST

    logging.debug(f"Object of class '{obj.__class__.__name__}' saved as: {file_path}")
    return True


def load_file(file_path: Union[str, Path]) -> Any:
    """
    Short cut for opening (possibly 'gzip' and/or 'blosc' compressed) .pkl files
    Returns the saved object.
    """

    try:  # gzip decompression
        with gzip.open(file_path, "rb") as f_cmprsd:
            obj = f_cmprsd.read()
    except gzip.BadGzipFile:  # in case gzip decompression fails
        try:  # blosc decompression
            obj, _ = bloscpack.unpack_bytes_from_file(file_path)
        except ValueError:  # in case blosc decompression fails
            with open(file_path, "rb") as f:
                return pickle.load(f)  # deserialize only
    except OSError:  # not fully downloaded from cloud
        raise OSError(
            "File was not fully downloaded from cloud! Check that cloud is synchronizing."
        )

    # if decompressed, still needs deserialization
    return pickle.loads(obj)


def save_processed_solution_meas(tdc_obj, dir_path: Path) -> None:
    """
    Save a processed measurement, including the '.data' attribute.
    The template may then be loaded much more quickly.
    """

    original_data = copy.deepcopy(tdc_obj.data)

    # lower size if possible
    for p in tdc_obj.data:
        if p.runtime.max() <= np.iinfo(np.int32).max:
            p.runtime = p.runtime.astype(np.int32)

    dir_path = dir_path / "processed"
    file_name = re.sub("_[*]", "", tdc_obj.template)
    file_path = dir_path / file_name
    save_object_to_disk(tdc_obj, file_path, compression_method="blosc")

    tdc_obj.data = original_data


def load_processed_solution_measurement(file_path: Path):
    """Doc."""

    tdc_obj = load_file(file_path)
    # Load runtimes as int64 if they are not already of that type
    for p in tdc_obj.data:
        p.runtime = p.runtime.astype(np.int64, copy=False)

    return tdc_obj


def load_file_dict(file_path: Path):
    """
    Load files according to extension,
    Allow backwards compatibility with legacy dictionary keys (relevant for both .mat and .pkl files),
    use defaults for legacy files where 'system_info' or 'after_pulse_param' is not iterable (therefore old).
    """

    if file_path.suffix == ".pkl":
        file_dict = translate_dict_keys(load_file(file_path), legacy_python_trans_dict)
    elif file_path.suffix == ".mat":
        file_dict = translate_dict_keys(load_mat(file_path), legacy_matlab_trans_dict)
    else:
        raise NotImplementedError(f"Unknown file extension '{file_path.suffix}'.")

    # patch for legacy Python files (almost non-existant)
    if not file_dict.get("system_info"):
        print("'system_info' is missing, using defaults...", end=" ")
        file_dict["system_info"] = default_system_info

    # patch MATLAB files
    elif not isinstance(file_dict["system_info"]["after_pulse_param"], tuple):
        if file_dict.get("python_converted"):
            file_dict["system_info"]["after_pulse_param"] = (
                "multi_exponent_fit",  # modern MATLAB format
                file_dict["system_info"]["after_pulse_param"],
            )
        else:
            file_dict["system_info"]["after_pulse_param"] = (
                "exponent_of_polynom_of_log",  # legacy MATLAB format
                file_dict["system_info"]["after_pulse_param"],
            )

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
            return np.array(obj, dtype=np.int64)
        elif isinstance(obj, int):
            return float(obj)
        else:
            return obj

    # recursion
    return {key: convert_types_to_matlab_format(val, key_name=str(key)) for key, val in obj.items()}


def save_mat(file_dict: dict, file_path: Path) -> None:
    """
    Saves 'file_dict' as 'file_path', after converting all keys to legacy naming,
    takes care of converting 'AfterPulseParam' to old style (array only, no type),
    and all lists to Numpy arrays."""

    file_dict = translate_dict_keys(file_dict, reverse_dict(legacy_matlab_trans_dict))
    file_dict["SystemInfo"]["AfterPulseParam"] = file_dict["SystemInfo"]["AfterPulseParam"][1]
    file_dict = convert_types_to_matlab_format(file_dict)
    file_dict["python_converted"] = True  # mark as converted-from-Python MATLAB format
    spio.savemat(file_path, file_dict)


def load_mat(file_path):
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
            elif isinstance(elem, np.ndarray) and name not in {"Data", "data", "LinearPart"}:
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

    data = spio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def sort_file_paths_by_file_number(file_paths: List[Path]) -> List[Path]:
    """
    Returns a path list, sorted according to file number (ascending).
    Works for file paths of the following format:
    "PATH_TO_FILE_DIRECTORY/file_template_*.ext"
    where the important part is '*.ext' (* is a any number, ext is any 3 letter file extension)
    """

    return sorted(
        file_paths,
        key=lambda file_path: int(re.split("(\\d+)$", file_path.stem)[1]),
    )


def file_selection_str_to_list(file_selection: str) -> Tuple[List[int], str]:
    """Doc."""

    def range_str_to_list(range_str: str) -> List[int]:
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
    file_idxs = [
        file_num
        for range_str in file_selection.split(",")
        for file_num in range_str_to_list(range_str)
    ]

    return file_idxs, choice


def prepare_file_paths(file_path_template: Path, file_selection: str = None) -> List[Path]:
    """Doc."""

    dir_path, file_template = file_path_template.parent, file_path_template.name
    unsorted_paths = list(dir_path.glob(file_template))
    file_paths = sort_file_paths_by_file_number(unsorted_paths)
    if not file_paths:
        raise FileNotFoundError(f"File template path ('{file_path_template}') does not exist!")

    if file_selection is not None:
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


def rotate_data_to_disk(method) -> Callable:
    """
    Loads 'self.data' object from disk prior to calling the method 'method',
    and dumps (saves and deletes the attribute) 'self.data' afterwards.
    """

    @functools.wraps(method)
    def method_wrapper(self, *args, **kwargs):
        self.dump_or_load_data(should_load=True, method_name=method.__name__)
        value = method(self, *args, **kwargs)
        self.dump_or_load_data(should_load=False, method_name=method.__name__)
        return value

    return method_wrapper
