"""Data-File Loading Utilities"""

from __future__ import annotations

import copy
import functools
import gzip
import logging
import pickle
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, Tuple, Union

import bloscpack
import numpy as np
import scipy.io as spio

from utilities.helper import Limits, reverse_dict, timer

DUMP_PATH = Path("C:/temp_sfcs_data/")

# TODO: this should be defined elsewhere (devices? app?)
with open("FastGatedSPAD_AP.pkl", "rb") as f:
    beta = pickle.load(f)

# NEW DETECTOR AFTERPULSING (Excess Bias 7 V, Hold-Off 81 ns, Avalanch Threshold 12 mV)
FAST_GATED_SPAD_AP = ("multi_exponent_fit", beta)

# OLD DETECTOR AFTERPULSING
PDM_SPAD_AP = (
    "multi_exponent_fit",
    np.array(
        [
            114424.39560026,
            10895.53707084,
            12817.86449556,
            1766.32335809,
            119012.66649389,
            10895.66339894,
            1518.68623068,
            315.70074808,
        ]
    ),
)

# TODO: this should be defined elsewhere (devices? app?)
default_system_info = {
    "setup": "STED with galvos",
    "afterpulse_params": FAST_GATED_SPAD_AP,  # PDM_SPAD_AP
    "ai_scaling_xyz": (1.243, 1.239, 1),
    "xyz_um_to_v": (70.81, 82.74, 10.0),
}

legacy_matlab_trans_dict = {
    # Solution Scan
    "Setup": "setup",
    "AfterPulseParam": "afterpulse_params",
    "AI_ScalingXYZ": "ai_scaling_xyz",
    "XYZ_um_to_V": "xyz_um_to_v",
    "SystemInfo": "system_info",
    "FullData": "full_data",
    "X": "x",
    "Y": "y",
    "ActualSpeed": "speed_um_s",
    "ScanFreq": "line_freq_hz",
    "SampleFreq": "ao_sampling_freq_hz",
    "PointsPerLineTotal": "samples_per_line",
    "PointsPerLine": "points_per_line",
    "NofLines": "n_lines",
    "LineLengthMax": "max_line_length_um",
    "LineShift": "line_shift_um",
    "AngleDegrees": "angle_degrees",
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
    "Dimension1_lines_um": "dim1_um",
    "Dimension2_col_um": "dim2_um",
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
    # General
    "LinFrac": "linear_fraction",
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
    "actual_speed": "speed_um_s",
    "actual_speed_um_s": "speed_um_s",
    "scan_freq": "line_freq_hz",
    "sample_freq": "ao_sampling_freq_hz",
    "max_line_length": "max_line_length_um",
    "line_shift": "line_shift_um",
    "data": "byte_data",
    "points_per_line_total": "samples_per_line",
    "line_length": "linear_len_um",
    "linear_len": "linear_len_um",
    # Image
    "cnt": "ci",
    "lines": "n_lines",
    "planes": "n_planes",
    "points_per_line": "ppl",
    "scan_type": "plane_orientation",
    "scan_plane": "plane_orientation",
    "lines_odd": "set_pnts_lines_odd",  # moved to scan_params
    "dim1_lines_um": "dim1_um",
    "dim2_col_um": "dim2_um",
    "linear_frac": "linear_fraction",
    # General
    "sample_freq_hz": "ao_sampling_freq_hz",
    "scan_param": "scan_settings",
    "scan_params": "scan_settings",
    "after_pulse_param": "afterpulse_params",
}


def _chunks(list_: list, n: int):
    """
    Generate n-sized chunks from list_.
    https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks?page=1&tab=votes#tab-top
    """

    for i in range(0, len(list_), n):
        yield list_[i : i + n]


def estimate_bytes(obj) -> int:
    """Returns the estimated size in bytes."""

    try:
        return len(pickle.dumps(obj, protocol=-1))
    except MemoryError:
        raise MemoryError("Object is too big!")


def deep_size_estimate(obj, level=np.inf, indent=4, threshold_mb=0, name=None) -> None:
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


@timer(threshold_ms=10000)
def save_object(
    obj,
    file_path: Path,
    compression_method: str = None,  # "gzip" / "blosc"
    obj_name: str = None,
    element_size_estimate_mb: float = None,
    should_track_progress=False,
) -> bool:
    """
    Save a pickle-serialized and optionally gzip/blosc-compressed object to disk, if estimated size is within the limits.
    Returns 'True' if saved, 'False' otherwise.
    """

    # create parent directory if needed
    dir_path = file_path.parent
    Path.mkdir(dir_path, parents=True, exist_ok=True)

    # split iterables to chunks if possible
    MAX_CHUNK_MB = 500
    if element_size_estimate_mb is not None:
        n_elem_per_chunk = max(int(MAX_CHUNK_MB / element_size_estimate_mb), 1)
        chunked_obj = list(_chunks(obj, n_elem_per_chunk))
    else:
        try:  # attempt to estimate using the first object
            element_size_estimate_mb = estimate_bytes(obj[0])
            n_elem_per_chunk = max(int(MAX_CHUNK_MB / element_size_estimate_mb), 1)
            chunked_obj = list(_chunks(obj, n_elem_per_chunk))
        except TypeError:  # obj isn't iterable - treat as a single chunk
            chunked_obj = [obj]

    should_track_progress = should_track_progress and len(chunked_obj) > 1
    if should_track_progress:
        print(f"Saving '{file_path.name}' in {len(chunked_obj)} chunks:", end=" ")

    if compression_method == "gzip":
        with gzip.open(file_path, "wb", compresslevel=1) as f_gzip:
            for chunk_ in chunked_obj:
                pickle.dump(chunk_, f_gzip, protocol=-1)  # type: ignore
                if should_track_progress:
                    print("O", end="")

    elif compression_method == "blosc":
        blosc_args = bloscpack.BloscArgs(typesize=4, clevel=1, cname="zlib")
        with open(file_path, "wb") as f_blosc:
            for chunk_ in chunked_obj:
                p_chunk = pickle.dumps(chunk_, protocol=-1)
                cmprsd_chunk = bloscpack.pack_bytes_to_bytes(p_chunk, blosc_args=blosc_args)
                pickle.dump(cmprsd_chunk, f_blosc, protocol=-1)
                if should_track_progress:
                    print("O", end="")

    else:  # save uncompressed
        with open(file_path, "wb") as f:
            for chunk_ in chunked_obj:
                pickle.dump(chunk_, f, protocol=-1)
                if should_track_progress:
                    print("O", end="")

    if should_track_progress:
        print(" - Done.")

    logging.debug(
        f"Object '{obj_name}' of class '{obj.__class__.__name__}' ({compression_method}-compressed) saved as: {file_path}"
    )
    return True


@timer(threshold_ms=10000)
def load_object(file_path: Union[str, Path], should_track_progress=False, **kwargs) -> Any:
    """
    Short cut for opening (possibly 'gzip' and/or 'blosc' compressed) .pkl files
    Returns the saved object.
    """

    file_path = Path(file_path)

    if should_track_progress:
        print(f"Loading '{file_path.name}':", end=" ")

    try:
        try:  # gzip decompression
            with gzip.open(file_path, "rb") as f_gzip_cmprsd:
                loaded_data = []
                while True:
                    loaded_data.append(pickle.load(f_gzip_cmprsd))  # type: ignore
                    if should_track_progress:
                        print("O", end="")

        except gzip.BadGzipFile:  # blosc decompression
            with open(file_path, "rb") as f_blosc_cmprsd:
                loaded_data = []
                try:
                    while True:
                        cmprsd_bytes = pickle.load(f_blosc_cmprsd)
                        p_bytes, _ = bloscpack.unpack_bytes_from_bytes(cmprsd_bytes)
                        loaded_data.append(pickle.loads(p_bytes))
                        if should_track_progress:
                            print("O", end="")

                except (ValueError, TypeError):  # uncompressed file
                    with open(file_path, "rb") as f_uncompressed:
                        loaded_data = []
                        while True:
                            loaded_data.append(pickle.load(f_uncompressed))
                            if should_track_progress:
                                print("O", end="")

    except EOFError:
        if should_track_progress:
            print(f" - Done ({len(loaded_data)} chunks)")

        if len(loaded_data) == 1:  # extract non-chunked loaded data
            return loaded_data[0]
        else:  # flatten iterable chunked data
            return [item for chunk_ in loaded_data for item in chunk_]


def save_processed_solution_meas(meas, dir_path: Path, should_force=False) -> bool:
    """
    Save a processed measurement, including the '.data' attribute.
    The template may then be loaded much more quickly.
    """

    # init return value as false
    has_saved_meas = False

    # save the measurement
    dir_path = dir_path / "processed"
    file_name = re.sub("_[*]", "", meas.template)
    file_path = dir_path / file_name
    if not file_path.is_file() or should_force:

        # load the data first, to save it as well
        meas.dump_or_load_data("load")
        # copy it
        data_copy = copy.deepcopy(meas.data)
        # clear the data from the original
        meas.dump_or_load_data("clear")

        # lower size if possible
        for p in data_copy:
            if p.correlation.pulse_runtime.max() <= np.iinfo(np.int32).max:
                p.correlation.pulse_runtime = p.correlation.pulse_runtime.astype(np.int32)

        has_saved_meas = save_object(
            meas, file_path, compression_method="blosc", obj_name="processed measurement"
        )

        # save the data separately
        data_path = file_path.parent / Path(str(file_path.stem) + "_data" + str(file_path.suffix))
        # TODO: find out what is causes the PermissionError (notebooks)
        has_saved_data = save_object(
            data_copy,
            data_path,
            compression_method="blosc",
            element_size_estimate_mb=data_copy[0].general.size_estimate_mb,
            obj_name="dumped data array",
            should_track_progress=True,
        )

    return has_saved_meas and has_saved_data


def load_processed_solution_measurement(file_path: Path, file_template: str, should_load_data=True):
    """Doc."""

    # load the measurement
    meas = load_object(file_path)

    # define the template # TODO: why is this needed?
    meas.template = file_template

    # load separately the data
    if should_load_data:
        data_path = file_path.parent / Path(str(file_path.stem) + "_data" + str(file_path.suffix))
        data = load_object(data_path, should_track_progress=True)
        for p in data:
            # Load runtimes as int64 if they are not already of that type
            p.correlation.pulse_runtime = p.correlation.pulse_runtime.astype(np.int64, copy=False)
        meas.data = data

    meas.is_data_dumped = not should_load_data

    return meas


def _handle_legacy_file_dict(file_dict, override_system_info=False, **kwargs):
    """Fixes data saved in varios legacy formats in-place"""

    # patches for legacy Python files
    if (
        file_dict.get("system_info") is None or override_system_info
    ):  # missing/overriden system_info
        file_dict["system_info"] = default_system_info
    if full_data := file_dict.get("full_data"):  # legacy full_data object structures
        if full_data.get("circle_speed_um_s"):
            full_data.pop("circle_speed_um_s")
            full_data["scan_settings"] = {
                "pattern": "circle",
                "speed_um_s": 6000,
                "ao_sampling_freq_hz": int(1e4),
                "diameter_um": 50,
            }
        if scan_settings := full_data.get("scan_settings"):
            if (
                scan_settings.get("pattern") == "circle"
                and scan_settings.get("circle_freq_hz") is None
            ):
                circumference = np.pi * scan_settings["diameter_um"]
                scan_settings["circle_freq_hz"] = scan_settings["speed_um_s"] / circumference
        if scan_settings := full_data.get("circular_scan_settings"):
            scan_pattern = "circle"
            circumference = np.pi * scan_settings["diameter_um"]
            circle_freq_hz = scan_settings["speed_um_s"] / circumference
            full_data["scan_settings"] = {
                "pattern": scan_pattern,
                "circle_freq_hz": circle_freq_hz,
                **scan_settings,
            }
        elif scan_settings := full_data.get("angular_scan_settings"):
            scan_pattern = "angular"
            full_data["scan_settings"] = {"pattern": scan_pattern, **scan_settings}
        if full_data.get("ao") is not None:
            n_rows_ao, n_cols_ao = full_data["ao"].shape
            n_rows_ai, n_cols_ai = full_data["ai"].shape
            full_data["scan_settings"].update(
                ao=full_data["ao"].T if n_rows_ao < n_cols_ao else full_data["ao"],
                ai=full_data["ai"].T if n_rows_ai < n_cols_ai else full_data["ai"],
            )
            full_data["scan_settings"].pop("x", None)
            full_data["scan_settings"].pop("y", None)
            full_data.pop("ao")
            full_data.pop("ai")

        # legacy detector/delayer settings placement
        if file_dict["system_info"].get("detector_settings") is not None:
            full_data["detector_settings"] = file_dict["system_info"]["detector_settings"].__dict__
            full_data["delayer_settings"] = file_dict["system_info"]["delayer_settings"].__dict__
        elif not full_data.get("detector_settings"):  # OLD DETECTOR
            full_data["detector_settings"] = dict(model="PDM", gate_width_ns=100)
        # namespaces to dicts
        if isinstance(full_data.get("detector_settings"), SimpleNamespace):
            full_data["detector_settings"] = full_data.get("detector_settings").__dict__
        if isinstance(full_data.get("delayer_settings"), SimpleNamespace):
            full_data["delayer_settings"] = full_data.get("delayer_settings").__dict__

    if scan_settings := file_dict.get("scan_settings"):  # legacy image scan
        if scan_settings.get("plane_orientation") and not scan_settings.get("dim_order"):
            if scan_settings["plane_orientation"] == "XY":
                scan_settings["dim_order"] = (0, 1, 2)
            elif scan_settings["plane_orientation"] == "YZ":
                scan_settings["dim_order"] = (1, 2, 0)
            elif scan_settings["plane_orientation"] == "XZ":
                scan_settings["dim_order"] = (0, 2, 1)

    # patch MATLAB files
    elif not isinstance(file_dict["system_info"]["afterpulse_params"], tuple):
        if file_dict.get("python_converted"):
            file_dict["system_info"]["afterpulse_params"] = (
                "multi_exponent_fit",  # modern MATLAB format
                file_dict["system_info"]["afterpulse_params"],
            )
        else:
            file_dict["system_info"]["afterpulse_params"] = (
                "exponent_of_polynom_of_log",  # legacy MATLAB format
                file_dict["system_info"]["afterpulse_params"],
            )
            # completing AI to 6 channes (X,Y,Z and their internals)
            n_samples, _ = full_data["scan_settings"]["ai"].shape
            full_data["scan_settings"]["ai"] = np.hstack(
                (full_data["scan_settings"]["ai"], np.full((n_samples, 4), np.nan))
            )


def load_file_dict(file_path: Path):
    """
    Load files according to extension.
    Allows backwards compatibility with legacy dictionary keys (relevant for both .mat and .pkl files).
    Uses defaults for legacy files where 'system_info' or 'afterpulse_params' is not iterable (therefore old).
    """

    try:
        if file_path.suffix == ".pkl":
            file_dict = _translate_dict_keys(load_object(file_path), legacy_python_trans_dict)
        elif file_path.suffix == ".mat":
            file_dict = _translate_dict_keys(_load_mat(file_path), legacy_matlab_trans_dict)
        else:
            raise NotImplementedError(f"Unknown file extension '{file_path.suffix}'.")

    except OSError as exc:
        raise OSError(
            f"File was not fully downloaded from cloud (check that cloud is synchronizing), or is missing. [{exc}]"
        )

    _handle_legacy_file_dict(file_dict)

    return file_dict


def _translate_dict_keys(original_dict: dict, translation_dict: dict) -> dict:
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
                translated_dict[translation_dict[org_key]] = _translate_dict_keys(
                    org_val, translation_dict
                )
            else:
                # translate the sub_dict
                translated_dict[org_key] = _translate_dict_keys(org_val, translation_dict)
        else:
            # translate the key if needed
            if org_key in translation_dict.keys():
                translated_dict[translation_dict[org_key]] = org_val
            else:
                translated_dict[org_key] = org_val

    return translated_dict


def save_mat(file_path: Path) -> None:
    """
    Re-saves raw data at 'file_path' as .mat, in order to be loaded by old MATLAB analysis tools.
    This takes care of converting all keys to legacy naming, converting 'AfterPulseParam' to old style (array only, no type),
    and all lists to Numpy arrays.
    """

    def _convert_types_to_matlab_format(obj, key_name=None):
        """
        Recursively converts any list/tuple in dictionary and any sub-dictionaries to Numpy ndarrays.
        Converts integers to floats.
        """

        # stop condition
        if not isinstance(obj, dict):
            if isinstance(obj, (list, tuple, Limits)):
                return np.array(obj, dtype=np.int64)
            elif isinstance(obj, int):
                return float(obj)
            elif isinstance(obj, SimpleNamespace):
                obj = obj.__dict__
            else:
                return obj

        # recursion
        return {
            key: _convert_types_to_matlab_format(val, key_name=str(key))
            for key, val in obj.items()
            if val is not None
        }

    file_dict = load_file_dict(file_path)
    mat_file_path_str = re.sub("\\.pkl", ".mat", str(file_path))
    if "solution" in mat_file_path_str:
        mat_file_path = Path(re.sub("solution", r"solution\\matlab", mat_file_path_str))
    elif "image" in mat_file_path_str:
        mat_file_path = Path(re.sub("image", r"image\\matlab", mat_file_path_str))

    file_dict = _translate_dict_keys(file_dict, reverse_dict(legacy_matlab_trans_dict))
    file_dict["SystemInfo"]["AfterPulseParam"] = file_dict["SystemInfo"]["AfterPulseParam"][1]
    file_dict = _convert_types_to_matlab_format(file_dict)
    file_dict["python_converted"] = True  # mark as converted-from-Python MATLAB format
    spio.savemat(mat_file_path, file_dict)


def _load_mat(file_path):
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
            elif isinstance(elem, np.ndarray) and name not in {
                "Data",
                "data",
                "LinearPart",
                "AO",
                "AI",
            }:
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


def prepare_file_paths(
    file_path_template: Path,
    file_selection: str = "Use All",
    should_unite_start_times=False,
    **kwargs,
) -> List[Path]:
    """Doc."""

    dir_path, file_template = file_path_template.parent, file_path_template.name
    if should_unite_start_times:  # ignore start time in template
        file_template = "_".join(file_template.split("_")[:-2]) + "_*.pkl"
    unsorted_paths = list(dir_path.glob(file_template))
    file_paths = sort_file_paths_by_file_number(unsorted_paths)
    if not file_paths:
        raise FileNotFoundError(f"File template path ('{file_path_template}') does not exist!")

    if file_selection != "Use All":
        file_idxs, choice = file_selection_str_to_list(file_selection)
        if choice == "Use":
            file_paths = [file_paths[i] for i in range(len(file_paths)) if i in file_idxs]
        else:
            file_paths = [file_paths[i] for i in range(len(file_paths)) if i not in file_idxs]

    return file_paths


def rotate_data_to_disk(does_modify_data: bool = False, **kwargs) -> Callable:
    """
    Loads 'self.data' object from disk prior to calling the method 'method',
    and dumps (saves and deletes the attribute) 'self.data' afterwards.
    """

    def outer_wrapper(method) -> Callable:
        @functools.wraps(method)
        def method_wrapper(self, *args, should_dump_data: bool = True, **kwargs):
            # load the data (if needed)
            self.dump_or_load_data("load", method_name=method.__name__, **kwargs)
            # call the method with data loaded
            value = method(self, *args, **kwargs)
            # re-save and dump the data if was changed and not skipping dumping (e.g. skipping when doing multiple data-related actions in series)
            if should_dump_data:
                if does_modify_data or not self.dump_path.exists():
                    self.dump_or_load_data("dump", method_name=method.__name__, **kwargs)
                else:  # no change in Any
                    self.dump_or_load_data("clear", method_name=method.__name__, **kwargs)
            # return what the method returns
            return value

        return method_wrapper

    return outer_wrapper
