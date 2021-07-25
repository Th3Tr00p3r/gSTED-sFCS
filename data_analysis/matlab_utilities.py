"""Data-File Loading Utilities"""

import numpy as np
import scipy.io as spio

legacy_keys_trans_dict = {
    # legacy MATLAB naming
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
    "DataVersion": "data_version",
    "FpgaFreq": "fpga_freq_mhz",
    "PixelFreq": "pix_clk_freq_mhz",
    "PixClockFreq": "pix_clk_freq_mhz",
    "LaserFreq": "laser_freq_mhz",
    "Version": "version",
    "AI": "ai",
    "AO": "ao",
    "AvgCnt": "avg_cnt_rate_khz",
    "CircleSpeed_um_sec": "circle_speed_um_s",
    "AnglularScanSettings": "angular_scan_settings",
    # legacy Python naming
    "fpga_freq": "fpga_freq_mhz",
    "pix_clk_freq": "pix_clk_freq_mhz",
    "laser_freq": "laser_freq_mhz",
    "circle_speed_um_sec": "circle_speed_um_s",
    "actual_speed": "actual_speed_um_s",
    "scan_freq": "scan_freq_hz",
    "sample_freq": "sample_freq_hz",
    "max_line_length": "max_line_length_um",
    "line_shift": "line_shift_um",
}


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


def loadmat(filename):
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
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray) and strg not in {"Data", "data"}:
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
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
