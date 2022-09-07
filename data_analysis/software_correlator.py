"""Software correlator"""

import sys
from collections import namedtuple
from ctypes import CDLL, c_double, c_int, c_long
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from numpy.ctypeslib import ndpointer


class CorrelatorType:
    PH_DELAY_CORRELATOR = 1
    PH_COUNT_CORRELATOR = 2
    PH_DELAY_CROSS_CORRELATOR = 3
    PH_COUNT_CROSS_CORRELATOR = 4  # does not seem to be implemented check! 1st column is photon arrival times, 2nd column boolean vector with 1s for photons arriving on the A channel and  0s otherwise, and the 3rd column is 1s for photons arriving on B channel and 0s otherwise
    PH_DELAY_CORRELATOR_LINES = 5  # additional column with 1s for photons that have to be counted, on 0s the delay chain is reset; used in image correlation and in angular scan
    PH_DELAY_CROSS_CORRELATOR_LINES = 6  # as PH_COUNT_CROSS_CORRELATOR with an additional column with 1s for photons that have to be counted, on 0s the delay chain is reset; used in image correlation and in angular scan
    PH_DELAY_LIFETIME_CROSS_CORRELATOR = 7
    PH_DELAY_LIFETIME_CORRELATOR = 8
    PH_DELAY_LIFETIME_CROSS_CORRELATOR_LINES = 9
    PH_DELAY_LIFETIME_CORRELATOR_LINES = 10


@dataclass
class SoftwareCorrelatorOutput:

    lag: np.ndarray
    corrfunc: np.ndarray
    weights: np.ndarray
    countrate: float


@dataclass
class SoftwareCorrelatorListOutput:

    lag_list: list
    corrfunc_list: list
    weights_list: list
    countrate_list: list


CrossCorrCountRates = namedtuple("CrossCorrCountRates", "a b")


class SoftwareCorrelator:
    """Doc."""

    LIB_DIR_PATH = Path("./SoftCorrelatorDynamicLib/")

    def __init__(self):
        if sys.platform == "darwin":
            LIB_NAME = "SoftCorrelatorDynamicLib.so"
        else:  # win32
            LIB_NAME = "SoftCorrelatorDynamicLib_win32.so"
        self.LIB_PATH = self.LIB_DIR_PATH / LIB_NAME

        soft_corr_dynamic_lib = CDLL(str(self.LIB_PATH), winmode=0)
        get_corr_params = soft_corr_dynamic_lib.getCorrelatorParams
        get_corr_params.restype = None
        get_corr_params.argtypes = [ndpointer(c_int), ndpointer(c_int)]
        doub_size = np.zeros(1, dtype=np.int32)
        num_corr = np.zeros(1, dtype=np.int32)
        get_corr_params(doub_size, num_corr)
        self.doubling_size = doub_size[0]
        self.num_of_correlators = num_corr[0]
        self.get_corr_params = get_corr_params
        self.tot_corr_chan_len = doub_size[0] * (num_corr[0] + 1) + 1

        # Initialize correlators (C-lang)

        # regular correlator
        soft_corr = soft_corr_dynamic_lib.softwareCorrelator
        soft_corr.restype = None
        soft_corr.argtypes = [
            c_int,  # correlator_option
            c_long,  # n_entries
            ndpointer(c_long, ndim=1, flags="C_CONTIGUOUS"),  # ph_hist
            ndpointer(c_int),  # n_corr_channels
            ndpointer(c_double, ndim=2, flags="C_CONTIGUOUS"),  # corr_py
        ]
        self.soft_corr = soft_corr

        # life-time correlator
        lt_soft_corr = soft_corr_dynamic_lib.softwareLifeTimeCorrelator
        lt_soft_corr.restype = None
        lt_soft_corr.argtypes = [
            c_int,  # correlator_option
            c_long,  # n_entries
            ndpointer(c_long, ndim=1, flags="C_CONTIGUOUS"),  # ph_hist
            ndpointer(c_double, ndim=1, flags="C_CONTIGUOUS"),  # filter
            ndpointer(c_int),  # n_corr_channels
            ndpointer(c_double, ndim=2, flags="C_CONTIGUOUS"),  # corr_py
        ]
        self.lt_soft_corr = lt_soft_corr

        # Initialize numpy array for holding correlator outputs
        self.corr_py = np.zeros((3, self.tot_corr_chan_len), dtype=float)

    def correlate_list(
        self,
        list_of_photon_arrays: List[np.ndarray],
        *args,
        is_verbose=False,
        list_of_filter_arrays=None,
        **kwargs,
    ) -> SoftwareCorrelatorListOutput:
        """Doc."""

        #        # TODO: parallel correlation isn't operational. SoftwareCorrelator objects contain ctypes containing pointers, which cannot be pickled for seperate processes.
        #        # See: https://stackoverflow.com/questions/18976937/multiprocessing-and-ctypes-with-pointers
        #        if should_parallel_process and len(time_stamp_split_list) > 1:  # parallel-correlate
        #            N_CORES = mp.cpu_count() // 2 - 1  # division by 2 due to hyperthreading in intel CPUs
        #            func = partial(
        #                SC.correlate, c_type=correlator_type, timebase_ms=1000 / laser_freq_hz
        #            )
        #            print(f"Parallel processing using {N_CORES} CPUs/processes.")
        #            with mp.get_context("spawn").Pool(N_CORES) as pool:
        #                correlator_output_list = list(pool.imap(func, time_stamp_split_list))

        correlator_output_list = []
        for idx, ts_split in enumerate(list_of_photon_arrays):
            if ts_split.size != 0:
                if list_of_filter_arrays is not None:
                    kwargs["filter_array"] = list_of_filter_arrays[idx]
                corr_output = self.correlate(ts_split, *args, **kwargs)
                correlator_output_list.append(corr_output)
                if is_verbose:
                    print(idx + 1, end=(", " if idx < len(list_of_photon_arrays) - 1 else ""))

        return self.list_output(correlator_output_list)

    def correlate(  # NOQA C901
        self, photon_array, correlator_option, filter_array=None, timebase_ms=1, **kwargs
    ) -> SoftwareCorrelatorOutput:
        """Doc."""

        self.correlator_option = correlator_option

        # TODO: this could possibly be avoided by creating/casting photon_array to platform independent type 'int' (i.e. dtype=int)
        if sys.platform == "darwin":  # fix operation for Mac users
            photon_array = photon_array.astype(np.int64)

        try:
            _, n_entries = photon_array.shape
        except ValueError:  # TODO: when would this happen (1D array?)
            n_entries = photon_array.size

        ph_hist = photon_array.reshape(-1)  # convert to 1D array
        if filter_array is not None:
            filter = filter_array.reshape(-1)  # convert to 1D array

        n_corr_channels = np.zeros(1, dtype=np.int32)

        if correlator_option == CorrelatorType.PH_DELAY_CORRELATOR:
            if len(photon_array.shape) != 1:
                raise RuntimeError(
                    f"Photon array {photon_array.shape} should be 1D for this correlator option!"
                )
            countrate = n_entries / photon_array.sum() / timebase_ms * 1000

        elif correlator_option == CorrelatorType.PH_COUNT_CORRELATOR:
            if len(photon_array.shape) != 1:
                raise RuntimeError(
                    f"Photon array {photon_array.shape} should be 1D for this correlator option!"
                )
            countrate = photon_array.sum() / n_entries / timebase_ms * 1000

        elif correlator_option == CorrelatorType.PH_DELAY_CROSS_CORRELATOR:
            if (len(photon_array.shape) == 1) or (photon_array.shape[0] != 3):
                raise RuntimeError(
                    f"Photon array {photon_array.shape} should have 3 rows for this correlator option! 0th row with photon time-stamps, 1st (2nd)  row contains 1s for photons in channel A (B) and 0s for photons in channel B(A)"
                )
            duration_s = photon_array[0, :].sum() * timebase_ms / 1000
            countrate_a = photon_array[2, :].sum() / duration_s
            countrate_b = photon_array[1, :].sum() / duration_s
            countrate = CrossCorrCountRates(countrate_a, countrate_b)

        elif correlator_option == CorrelatorType.PH_DELAY_CORRELATOR_LINES:
            if (len(photon_array.shape) == 1) or (photon_array.shape[0] != 2):
                raise RuntimeError(
                    f"Photon array {photon_array.shape} should have 2 rows for this correlator option! 0th row with photon time-stamps, 1st row is 1 for valid lines"
                )
            valid = (photon_array[1, :] == 1) | (photon_array[1, :] == -2)
            duration_s = photon_array[0, valid].sum() * timebase_ms / 1000
            countrate = np.sum(photon_array[1, :] == 1) / duration_s

        elif correlator_option == CorrelatorType.PH_DELAY_CROSS_CORRELATOR_LINES:
            if (len(photon_array.shape) == 1) or (photon_array.shape[0] != 4):
                raise RuntimeError(
                    f"Photon array {photon_array.shape} should have 4 rows for this correlator option! 0th row with photon time-stamps, 1st (2nd) row contains 1s for photons in channel A (B) and 0s for photons in channel B(A), and 4th row is 1s for valid lines"
                )
            valid_timestamps = (photon_array[3, :] == 1) | (photon_array[3, :] == -2)
            valid_photons = photon_array[3, :] == 1
            duration_s = photon_array[0, valid_timestamps].sum() * timebase_ms / 1000
            countrate_a = np.sum(photon_array[2, valid_photons] == 1) / duration_s
            countrate_b = np.sum(photon_array[1, valid_photons] == 1) / duration_s
            countrate = CrossCorrCountRates(countrate_a, countrate_b)

        elif correlator_option == CorrelatorType.PH_DELAY_LIFETIME_CROSS_CORRELATOR:
            if len(photon_array.shape) != 1:
                raise RuntimeError(
                    f"Photon array {photon_array.shape} should be 1D for this correlator option!"
                )
            if (len(filter_array.shape) == 1) or (filter_array.shape[0] != 2):
                raise RuntimeError(
                    f"Filter array {filter_array.shape} should have 2 rows for this correlator option! 0th & 1st rows should contain float filter values for photons in channels A & B (respectively)"
                )
            duration_s = photon_array[0, :].sum() * timebase_ms / 1000
            countrate_a = photon_array[2, :].sum() / duration_s
            countrate_b = photon_array[1, :].sum() / duration_s
            countrate = CrossCorrCountRates(countrate_a, countrate_b)

        elif correlator_option == CorrelatorType.PH_DELAY_LIFETIME_CORRELATOR:
            if len(photon_array.shape) != 1:
                raise RuntimeError(
                    f"Photon array {photon_array.shape} should be 1D for this correlator option!"
                )
            if len(filter_array.shape) != 1:
                raise RuntimeError(
                    f"Filter array {filter_array.shape} should be 1D for this correlator option! It should contain float filter values for fluorescence photons (as in not afterpulsing photons)"
                )
            countrate = n_entries / photon_array.sum() / timebase_ms * 1000

        elif correlator_option == CorrelatorType.PH_DELAY_LIFETIME_CROSS_CORRELATOR_LINES:
            if (len(photon_array.shape) == 1) or (photon_array.shape[0] != 4):
                raise RuntimeError(
                    f"Photon array {photon_array.shape} should have 4 rows for this correlator option! 0th row with photon time-stamps, 1st (2nd) row contains 1s for photons in channel A (B) and 0s for photons in channel B(A), and 4th row is 1s for valid lines"
                )
            if (len(filter_array.shape) == 1) or (filter_array.shape[0] != 2):
                raise RuntimeError(
                    f"Filter array {filter_array.shape} should have 2 rows for this correlator option! 0th & 1st rows should contain float filter values for photons in channels A & B (respectively)"
                )
            valid_timestamps = (photon_array[3, :] == 1) | (photon_array[3, :] == -2)
            valid_photons = photon_array[3, :] == 1
            duration_s = photon_array[0, valid_timestamps].sum() * timebase_ms / 1000
            countrate_a = np.sum(photon_array[2, valid_photons] == 1) / duration_s
            countrate_b = np.sum(photon_array[1, valid_photons] == 1) / duration_s
            countrate = CrossCorrCountRates(countrate_a, countrate_b)

        elif correlator_option == CorrelatorType.PH_DELAY_LIFETIME_CORRELATOR_LINES:
            if (len(photon_array.shape) == 1) or (photon_array.shape[0] != 2):
                raise RuntimeError(
                    f"Photon array {photon_array.shape} should have 2 rows for this correlator option! 0th row with photon time-stamps, 1st row is 1 for valid lines"
                )
            if len(filter_array.shape) != 1:
                raise RuntimeError(
                    f"Filter array {filter_array.shape} should be 1D for this correlator option! It should contain float filter values for fluorescence photons (as in not afterpulsing photons)"
                )
            valid = (photon_array[1, :] == 1) | (photon_array[1, :] == -2)
            duration_s = photon_array[0, valid].sum() * timebase_ms / 1000
            countrate = np.sum(photon_array[1, :] == 1) / duration_s

        else:
            raise ValueError("Invalid correlator type!")

        # Perform software correlation
        if correlator_option in {
            CorrelatorType.PH_DELAY_LIFETIME_CROSS_CORRELATOR,
            CorrelatorType.PH_DELAY_LIFETIME_CORRELATOR,
            CorrelatorType.PH_DELAY_LIFETIME_CROSS_CORRELATOR_LINES,
            CorrelatorType.PH_DELAY_LIFETIME_CORRELATOR_LINES,
        }:  # lifetime corralation
            self.lt_soft_corr(
                correlator_option, n_entries, ph_hist, filter, n_corr_channels, self.corr_py
            )

        else:  # non-lifetime correlation
            self.soft_corr(correlator_option, n_entries, ph_hist, n_corr_channels, self.corr_py)

        if n_corr_channels[0] != self.tot_corr_chan_len:
            raise ValueError("Number of correlator channels inconsistent!")

        # collect outputs
        valid_corr = self.corr_py[2, :] > 0  # weights > 0
        lag = (self.corr_py[1, :] * timebase_ms)[valid_corr]
        corrfunc = self.corr_py[0, :][valid_corr]
        weights = self.corr_py[2, :][valid_corr]

        return SoftwareCorrelatorOutput(
            lag,
            corrfunc,
            weights,
            countrate,
        )

    def list_output(self, correlator_output_list: list) -> SoftwareCorrelatorListOutput:
        """Accumulate all software correlator outputs in lists"""

        lag_list = [output.lag for output in correlator_output_list]
        corrfunc_list = [output.corrfunc for output in correlator_output_list]
        weights_list = [output.weights for output in correlator_output_list]
        countrate_list = [output.countrate for output in correlator_output_list]

        return SoftwareCorrelatorListOutput(lag_list, corrfunc_list, weights_list, countrate_list)
