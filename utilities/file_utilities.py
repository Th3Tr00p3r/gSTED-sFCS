"""Data-File Loading Utilities"""

from __future__ import annotations

import copy
import gzip
import io
import logging
import pickle
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Union

import bloscpack
import numpy as np
import scipy.io as spio

from utilities.helper import Gate, Limits, chunks, reverse_dict, timer

# use different paths for different types systems (windows, mac) - check for darwin or win32
if sys.platform == "darwin":
    DUMP_ROOT = Path("/Users/ido.michealovich/tmp/")
else:  # win32
    DUMP_ROOT = Path("D:/temp_sfcs_data/")

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
default_system_info: Dict[str, Any] = {
    "afterpulse_params": FAST_GATED_SPAD_AP,  # PDM_SPAD_AP
    "ai_scaling_xyz": (1.243, 1.239, 1),
    "xyz_um_to_v": (70.81, 82.74, 10.0),
}

legacy_matlab_trans_dict = {
    # Solution Scan
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
    "Lines": "n_lines",
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
    "PixClkFreq": "pix_clk_freq_mhz",
    "Cnt": "ci",
    "PID": "pid",
    "SP": "sp",
    "LinesOdd": "set_pnts_lines_odd",
    "FastScan": "is_fast_scan",
    "TdcScanData": "tdc_scan_data",
    "Plane": "plane",
    "ScanParam": "scan_settings",
    "Dimension1_lines_um": "dim1_um",
    "Dimension2_col_um": "dim2_um",
    "Dimension3_um": "dim3_um",
    "Line": "n_lines",
    "Planes": "n_planes",
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
    "Version": "version",
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


class PathFixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name in ("WindowsPath", "PosixPath"):
            return Path
        return super().find_class(module, name)


class MemMapping:
    """
    A convenience class working with Numpy memory-mapping.
    Can be used as a context manager to ensure deletion of on-disk file (single-use).
    """

    def __init__(self, arr: np.ndarray):
        # Create a temporary file
        self._temp_file = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        self._dump_file_path = Path(self._temp_file.name)
        # dump
        np.save(
            self._dump_file_path,
            arr,
            allow_pickle=False,
            fix_imports=False,
        )
        self._temp_file.close()  # Close the file handle

        # keep some useful attributes for quick access
        self.shape = arr.shape
        self.size = arr.size
        self.max = arr.max()
        self.min = arr.min()

    def read(self):
        """
        Access the data from disk by memory-mapping.
        """
        return np.load(
            self._dump_file_path,
            mmap_mode="r",
            allow_pickle=True,
            fix_imports=False,
        )

    def write1d(self, arr: np.ndarray, start_idx=0, stop_idx=None):
        """
        Write to the memory-mapped array.
        """
        if stop_idx is None:
            stop_idx = arr.size

        mmap_sub_arr = np.load(
            self._dump_file_path,
            mmap_mode="r+",
            allow_pickle=False,
            fix_imports=False,
        )
        mmap_sub_arr[start_idx:stop_idx] = arr
        mmap_sub_arr.flush()

        # update useful attributes for quick access
        self.shape = mmap_sub_arr.shape
        self.size = mmap_sub_arr.size
        self.max = mmap_sub_arr.max()
        self.min = mmap_sub_arr.min()

    def chunked_bincount(self, max_val=None, n_chunks=10):
        """Perform 'bincount' in series on disk-loaded array chunks."""
        max_val = max_val or self.max
        bins = np.zeros(max_val + 1)
        for arr_chunk in chunks(self.read(), int(self.size / n_chunks)):
            arr_chunk = arr_chunk[arr_chunk <= max_val]
            bins += np.bincount(arr_chunk, minlength=max_val + 1)

        return bins

    def __del__(self):
        """Ensure the temporary file is deleted when the object is no longer in use."""
        self._dump_file_path.unlink(missing_ok=True)

    def delete(self):
        """Delete the on-disk array explicitly if needed before the object is garbage collected."""
        self.__del__()


def search_database(data_root: Path, str_list: List[str]) -> str:
    """
    Search the database for templates containing all strings in `str_list`
    and print them along with their dates.
    """

    # get all unique templates by using their .log files from all directories in DATA_ROOT. filter using the SEARCH_LIST.
    try:
        template_date_dict = {
            path_.stem[:-2]: datetime.strptime(path_.parent.parent.name, "%d_%m_%Y").date()
            for path_ in data_root.rglob("*_1.*")
            if all([str_.lower() in str(path_).lower() for str_ in str_list])
            and path_.parent.name != "data"
        }
    except ValueError:
        # Issue with date format
        return "Issue with date format..."

    # sort by date in reverse order (newest first)
    template_date_dict = dict(
        sorted(template_date_dict.items(), key=lambda item: item[1], reverse=True)
    )

    # return the findings (date first, though the key is the template)
    if template_date_dict:
        text = f"Found {len(template_date_dict)} matching templates:\n" + "\n".join(
            [
                f"{date.strftime('%d_%m_%Y')}: {template}_"
                for template, date in template_date_dict.items()
            ]
        )
        return text
    else:
        return "No matches found!"


def estimate_bytes(obj) -> int:
    """Returns the estimated size in bytes."""

    try:
        return len(pickle.dumps(obj, protocol=-1))
    except MemoryError:
        raise MemoryError("Object is too big!")


def deep_size_estimate(obj, threshold_mb=1e-6, level=np.inf, indent=4, name=None) -> None:
    """
    Print a cascading size description of object 'obj' up to level 'level'
    for objects (and subobjects) requiring an estimated disk space over 'threshold_mb'.

    for ease of use, copy these lines to somewhere in the code:
    from utilities.file_utilities import deep_size_estimate as dse # TESTESTEST
    dse(obj) # TESTESTEST
    """

    # get object size
    try:
        size_mb = (obj.nbytes if isinstance(obj, np.ndarray) else estimate_bytes(obj)) / 1e6
    except MemoryError:
        # Object is too big!
        size_mb = np.inf

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
                deep_size_estimate(elem, threshold_mb, level - 1, indent + 4, f"{name}[{idx}]")
        if hasattr(obj, "__dict__"):
            # convert namespaces into dicts
            obj = copy.deepcopy(vars(obj))
        if isinstance(obj, dict):
            for key, val in obj.items():
                deep_size_estimate(val, threshold_mb, level - 1, indent + 4, key)
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
    compression_method: str = "no compression",  # "gzip" / "blosc"
    obj_name: str = None,
    element_size_estimate_mb: float = None,
    should_track_progress=False,
) -> bool:
    """
    Save a pickle-serialized and optionally gzip/blosc-compressed object to disk.
    Returns 'True' if saved, 'False' otherwise.
    """
    # TODO: file suffix (.pkl, .blosc, .gzip) should be determined here! (with_suffix...) - this would simplify the load_object function, too, since it would know which file to expect. (but it's nice that it finds out on its own)
    # TODO: compressed objects take up less space, and so the chunksize should (if possible) be set to a higher value when estimating the chunksize of a soon to be compressed object (but how??)

    # create parent directory if needed
    dir_path = file_path.parent
    Path.mkdir(dir_path, parents=True, exist_ok=True)

    # split iterables to chunks if possible/relevant
    if isinstance(obj, np.ndarray):
        chunked_obj = [obj]
    else:
        MAX_CHUNK_MB = 500
        if element_size_estimate_mb is not None:
            n_elem_per_chunk = max(int(MAX_CHUNK_MB / element_size_estimate_mb), 1)
            chunked_obj = list(chunks(obj, n_elem_per_chunk))
        else:
            try:  # attempt to estimate using the first object
                element_size_estimate_mb = estimate_bytes(obj[0])
                n_elem_per_chunk = max(int(MAX_CHUNK_MB / element_size_estimate_mb), 1)
                chunked_obj = list(chunks(obj, n_elem_per_chunk))
            except (TypeError, KeyError):  # obj isn't iterable - treat as a single chunk
                chunked_obj = [obj]

    if should_track_progress := should_track_progress and len(chunked_obj) > 1:
        print(
            f"Saving '{file_path.name}' in {len(chunked_obj)} chunks ({compression_method}): ",
            end="",
        )

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
        print(f"Loading '{file_path.name}': ", end="")
        compression_method = "no"

    # load pure numpy arrays
    if file_path.suffix == "npy":
        return np.load(file_path)

    try:
        try:  # gzip decompression
            with gzip.open(file_path, "rb") as f_gzip_cmprsd:
                loaded_data = []
                while True:
                    loaded_data.append(PathFixUnpickler(f_gzip_cmprsd).load())  # type: ignore
                    if should_track_progress:
                        compression_method = "gzip"
                        print("O", end="")

        except gzip.BadGzipFile:  # blosc decompression
            with open(file_path, "rb") as f_blosc_cmprsd:
                loaded_data = []
                try:
                    while True:
                        cmprsd_bytes = pickle.load(f_blosc_cmprsd)
                        p_bytes, _ = bloscpack.unpack_bytes_from_bytes(cmprsd_bytes)
                        loaded_data.append(PathFixUnpickler(io.BytesIO(p_bytes)).load())
                        if should_track_progress:
                            compression_method = "blosc"
                            print("O", end="")

                except (ValueError, TypeError):  # uncompressed file
                    with open(file_path, "rb") as f_uncompressed:
                        loaded_data = []
                        while True:
                            loaded_data.append(PathFixUnpickler(f_uncompressed).load())
                            if should_track_progress:
                                print("O", end="")

    except EOFError:
        if should_track_progress:
            print(
                f" - Done ({(n_chunks := len(loaded_data))} {'chunks' if n_chunks > 1 else 'chunk'}, {compression_method} compression)"
            )

        if len(loaded_data) == 1:  # extract non-chunked loaded data
            return loaded_data[0]
        else:  # flatten iterable chunked data
            return [item for chunk_ in loaded_data for item in chunk_]


def load_processed_solution_measurement(
    dir_path: Path, dump_root: Path = DUMP_ROOT, should_load_data=True, **proc_options
):
    """Doc."""
    # TODO: this should be a method of SolutionSFCSExperiment class

    meas_file_path = dir_path / "SolutionSFCSMeasurement.blosc"

    # load the measurement
    print("Loading SolutionSFCSMeasurement object... ", end="")
    meas = load_object(meas_file_path)
    print("Done.")

    # load separately the data, but only if not already in temp folder (to avoid long decompressing)
    if should_load_data:
        print(
            f"Loading (decompressing and memory-mapping) {len(meas.data):,} processed data files: ",
            end="",
        )
        for idx, p in enumerate(meas.data):
            if idx > 0:
                print(", ", end="")
            # redefining the dump path to the temp folder of the current system
            p.dump_path = dump_root / p.dump_path.parts[-1]
            p.raw._dump_path = p.dump_path
            if not p.raw.dump_path.exists():
                try:
                    p.raw.load_compressed(dir_path / "data")
                except FileNotFoundError:
                    meas.dump_path = p.dump_path
                    meas.data_processor.dump_path = p.dump_path
                    meas.file_path_template = dir_path.parent.parent / meas.template
                    file_paths = prepare_file_paths(meas.file_path_template, **proc_options)
                    data_filepath = file_paths[idx]
                    print("Compressed data file not found! ", end="")
                    p = meas.process_data_file(data_filepath, **proc_options)
                    p.raw.dump()
            print(f"{idx+1}", end="")
        print(".")
    return meas


def _handle_legacy_file_dict(file_dict, override_system_info=False, **kwargs):  # NOQA C901
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

    # legacy Python image scan
    if scan_settings := file_dict.get("scan_settings"):
        if scan_settings.get("plane_orientation") and not scan_settings.get("dim_order"):
            if scan_settings["plane_orientation"] == "XY":
                scan_settings["dim_order"] = (0, 1, 2)
            elif scan_settings["plane_orientation"] == "YZ":
                scan_settings["dim_order"] = (1, 2, 0)
            elif scan_settings["plane_orientation"] == "XZ":
                scan_settings["dim_order"] = (0, 2, 1)

    # patches for legacy image files
    if "full_data" not in file_dict:
        file_dict["full_data"] = file_dict.pop("tdc_scan_data")
        file_dict["full_data"]["fpga_freq_hz"] = int(400e6)
        file_dict["full_data"]["ci"] = file_dict.pop("ci")
        file_dict["full_data"]["scan_settings"] = file_dict.pop("scan_settings")
        file_dict["full_data"]["scan_settings"]["ai"] = file_dict.pop("ai")
        file_dict["full_data"]["scan_settings"]["ao"] = file_dict.pop("ao").T
        file_dict["full_data"]["scan_settings"]["pattern"] = "image"
        file_dict["full_data"]["scan_settings"]["ao_sampling_freq_hz"] = int(1e4)
        file_dict["full_data"]["scan_settings"]["pattern"] = "image"
        file_dict["full_data"]["detector_settings"] = {}
        file_dict["full_data"]["detector_settings"]["mode"] = "free running"
        file_dict["full_data"]["detector_settings"]["gate_ns"] = Gate()
        file_dict["full_data"]["laser_mode"] = file_dict.pop("laser_mode", None)
        # legacy MATLAB
        if "pid" in file_dict:
            # get rid of useless values
            file_dict.pop("pid")
            file_dict.pop("sp")
            file_dict.pop("is_fast_scan")
            file_dict["full_data"]["scan_settings"].pop("is_fast_scan")
            file_dict.pop("__version__")
            file_dict.pop("__globals__")
            # move stuff around
            file_dict["full_data"]["version"] = file_dict.pop("version")
            file_dict["full_data"]["pix_clk_freq_mhz"] = file_dict.pop("pix_clk_freq_mhz")
            file_dict["full_data"]["scan_settings"]["set_pnts_lines_odd"] = file_dict.pop(
                "set_pnts_lines_odd"
            )
            file_dict["full_data"]["byte_data"] = np.array(file_dict["full_data"].pop("plane"))
            # modify and patch
            if (stage := file_dict["full_data"]["scan_settings"]["what_stage"]) == "Galvanometers":
                file_dict["full_data"]["scan_settings"]["dim_order"] = (0, 1, 2)
                file_dict["full_data"]["scan_settings"]["plane_orientation"] = "XY"
            else:
                raise NotImplementedError(f"Handle this (dim_order) for '{stage}' stage.")
            ao_center = file_dict["full_data"]["scan_settings"]["ao"].mean()
            file_dict["full_data"]["scan_settings"]["initial_ao"] = (ao_center, None, None)
            ao_1d = file_dict["full_data"]["scan_settings"]["ao"]
            file_dict["full_data"]["scan_settings"]["ao"] = np.vstack(
                (ao_1d, np.empty_like(ao_1d), np.empty_like(ao_1d))
            ).T

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
        elif file_path.suffix == ".npy":
            file_dict = dict(byte_data=np.load(file_path, allow_pickle=False, fix_imports=False))
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


def save_mat(*file_paths: Path) -> None:
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
                try:
                    return np.array(obj, dtype=np.int64)
                except OverflowError:  # TESTESTEST
                    return np.array(obj, dtype=np.float64)
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

    mat_file_path_str = re.sub("\\.pkl", ".mat", str(file_paths[0]))
    if "solution" in mat_file_path_str:
        mat_file_path = Path(re.sub("solution", r"solution\\matlab", mat_file_path_str))
    elif "image" in mat_file_path_str:
        mat_file_path = Path(re.sub("image", r"image\\matlab", mat_file_path_str))

    # translate to Matlab format
    file_dicts = [load_file_dict(file_path) for file_path in file_paths]
    combined_file_dict = {k: v for file_dict in file_dicts for k, v in file_dict.items()}
    combined_file_dict = _translate_dict_keys(
        combined_file_dict, reverse_dict(legacy_matlab_trans_dict)
    )
    combined_file_dict["SystemInfo"]["AfterPulseParam"] = combined_file_dict["SystemInfo"][
        "AfterPulseParam"
    ][1]
    combined_file_dict = _convert_types_to_matlab_format(combined_file_dict)
    combined_file_dict["FullData"]["Data"] = combined_file_dict.pop("Data")
    combined_file_dict["python_converted"] = True  # mark as converted-from-Python MATLAB format
    spio.savemat(mat_file_path, combined_file_dict)


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
        file_template = "_".join(file_template.split("_")[:-2]) + f"_*{file_path_template.suffix}"
    unsorted_paths = list(dir_path.glob(file_template))
    file_paths = sort_file_paths_by_file_number(unsorted_paths)
    if not file_paths:
        raise FileNotFoundError(f"File template path ('{file_path_template}') does not exist!")

    if file_selection != "Use All":
        file_idxs, choice = file_selection_str_to_list(file_selection)
        if choice == "Use":
            file_paths = [
                file_path
                for file_path in file_paths
                if any(
                    f"_{idx + 1}{file_path_template.suffix}" in str(file_path) for idx in file_idxs
                )
            ]
        else:
            file_paths = [
                file_path
                for file_path in file_paths
                if not any(
                    f"_{idx + 1}{file_path_template.suffix}" in str(file_path) for idx in file_idxs
                )
            ]

    return file_paths
