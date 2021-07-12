#!/usr/bin/env python3
"""
Created on Tue Mar 23 19:16:16 2021

@author: copied from: https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
"""
import numpy as np
import scipy.io as spio

legacy_matlab_naming_angular_scan_settings_trans_dict = {
    "X": "x",
    "Y": "y",
    "ActualSpeed": "actual_speed",
    "ScanFreq": "scan_freq",
    "SampleFreq": "sample_freq",
    "PointsPerLineTotal": "points_per_line_total",
    "PointsPerLine": "points_per_line",
    "NofLines": "n_lines",
    "LineLength": "line_length",
    "LengthTot": "total_length",
    "LineLengthMax": "max_line_length",
    "LineShift": "line_shift",
    "AngleDegrees": "angle_degrees",
    "LinFrac": "linear_frac",
    "LinearPart": "linear_part",
    "Xlim": "x_lim",
    "Ylim": "y_lim",
}

legacy_matlab_naming_full_data_trans_dict = {
    "Data": "data",
    "DataVersion": "data_version",
    "FpgaFreq": "fpga_freq",
    "PixelFreq": "pix_clk_freq",
    "LaserFreq": "laser_freq",
    "Version": "version",
    "AI": "ai",
    "AO": "ao",
    "AvgCnt": "avg_cnt_rate",
    "CircleSpeed_um_sec": "circle_speed_um_sec",
    "AnglularScanSettings": (
        "angular_scan_settings",
        legacy_matlab_naming_angular_scan_settings_trans_dict,
    ),
}

legacy_matlab_naming_system_info_trans_dict = {
    "Setup": "setup",
    "AfterPulseParam": "after_pulse_param",
    "AI_ScalingXYZ": "ai_scaling_xyz",
    "XYZ_um_to_V": "xyz_um_to_v",
}

legacy_matlab_naming_trans_dict = {
    "SystemInfo": ("system_info", legacy_matlab_naming_system_info_trans_dict),
    "FullData": ("full_data", legacy_matlab_naming_full_data_trans_dict),
}


def translate_dict_keys(original_dict: dict, trans_dict: dict) -> dict:
    """
    Updates keys of dict according to another dict:
    trans_dct.keys() are the keys to update,
    and trans_dct.values() are the new keys.
    Key, value pairs that do not appear in trans_dict will remain unchanged.
    """

    new_dict = {}
    for org_key, org_val in original_dict.items():
        if org_key in trans_dict.keys():
            if isinstance(org_val, dict):
                new_key, sub_trans_dict = trans_dict[org_key]
                new_dict[new_key] = translate_dict_keys(org_val, sub_trans_dict)
            else:
                new_dict[trans_dict[org_key]] = org_val
        else:
            new_dict[org_key] = org_val
    return new_dict


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
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
            elif isinstance(elem, np.ndarray):
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
