"""Software correlator"""

import sys
from ctypes import CDLL, c_double, c_int, c_long
from dataclasses import dataclass
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


class SoftwareCorrelator:
    """Doc."""

    LIB_PATH = "./SoftCorrelatorDynamicLib/SoftCorrelatorDynamicLib_win32.so"

    def __init__(self):
        if sys.platform == "darwin":
            self.LIB_PATH = "/Users/oleg/Documents/Python programming/Scanning setups Lab/gSTED-sFCS/SoftCorrelatorDynamicLib/SoftCorrelatorDynamicLib/SoftCorrelatorDynamicLib.so"

        soft_corr_dynamic_lib = CDLL(self.LIB_PATH, winmode=0)
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
        self.corr_py = np.zeros((3, self.tot_corr_chan_len), dtype=float)

        soft_corr = soft_corr_dynamic_lib.softwareCorrelator
        soft_corr.restype = None
        soft_corr.argtypes = [
            c_int,
            c_long,
            ndpointer(c_long, ndim=1, flags="C_CONTIGUOUS"),
            ndpointer(c_int),
            ndpointer(c_double, ndim=2, flags="C_CONTIGUOUS"),
        ]
        self.soft_corr = soft_corr

        self.corr_py = np.zeros((3, self.tot_corr_chan_len), dtype=float)

    def correlate_list(
        self, list_of_photon_arrays: List[np.ndarray], *args, is_verbose=False, **kwargs
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
                self.correlate(ts_split, *args, **kwargs)
                correlator_output_list.append(self.output())
                if is_verbose:
                    print(idx + 1, end=", ")

        return self.list_output(correlator_output_list)

    def correlate(self, photon_array, corr_type, timebase_ms=1):
        """Doc."""

        self.corr_type = corr_type

        if sys.platform == "darwin":  # fix operation for Mac users
            photon_array = photon_array.astype(np.int64)

        try:
            _, n_entries = photon_array.shape
        except ValueError:  # TODO: when would this happen (1D array?)
            n_entries = photon_array.size

        ph_hist = photon_array.reshape(-1)  # convert to 1D array
        n_corr_channels = np.zeros(1, dtype=np.int32)

        if corr_type == CorrelatorType.PH_DELAY_CORRELATOR:
            if len(photon_array.shape) != 1:
                raise RuntimeError(
                    f"Photon array {photon_array.shape} should be 1D for this correlator option!"
                )
            self.countrate = n_entries / photon_array.sum() / timebase_ms * 1000

        elif corr_type == CorrelatorType.PH_COUNT_CORRELATOR:
            if len(photon_array.shape) != 1:
                raise RuntimeError(
                    f"Photon array {photon_array.shape} should be 1D for this correlator option!"
                )
            self.countrate = photon_array.sum() / n_entries / timebase_ms * 1000

        elif corr_type == CorrelatorType.PH_DELAY_CROSS_CORRELATOR:
            if (len(photon_array.shape) == 1) or (photon_array.shape[0] != 3):
                raise RuntimeError(
                    f"Photon array {photon_array.shape} should have 3 rows for this correlator option! 0th row with photon delay times, 1st (2nd)  row contains 1s for photons in channel A (B) and 0s for photons in channel B(A)"
                )
            duration_s = photon_array[0, :].sum() * timebase_ms / 1000
            countrate_a = photon_array[1, :].sum() / duration_s
            countrate_b = photon_array[2, :].sum() / duration_s
            self.countrate = (countrate_a, countrate_b)

        elif corr_type == CorrelatorType.PH_DELAY_CORRELATOR_LINES:
            if (len(photon_array.shape) == 1) or (photon_array.shape[0] != 2):
                raise RuntimeError(
                    f"Photon array {photon_array.shape} should have 2 rows for this correlator option! 0th row with photon delay times, 1st row is 1 for valid lines"
                )
            valid = (photon_array[1, :] == 1) | (photon_array[1, :] == -2)
            duration_s = photon_array[0, valid].sum() * timebase_ms / 1000
            self.countrate = np.sum(photon_array[1, :] == 1) / duration_s

        elif corr_type == CorrelatorType.PH_DELAY_CROSS_CORRELATOR_LINES:
            if (len(photon_array.shape) == 1) or (photon_array.shape[0] != 4):
                raise RuntimeError(
                    f"Photon array {photon_array.shape} should have 4 rows for this correlator option! 0th row with photon delay times, 1st (2nd)  row contains 1s for photons in channel A (B) and 0s for photons in channel B(A), and 4th row is 1s for valid lines"
                )
            valid = (photon_array[3, :] == 1) | (photon_array[3, :] == -2)
            duration_s = photon_array[0, valid].sum() * timebase_ms / 1000
            countrate_a = np.sum(photon_array[1, :] == 1) / duration_s
            countrate_b = np.sum(photon_array[2, :] == 1) / duration_s
            self.countrate = (countrate_a, countrate_b)
        else:
            raise ValueError("Invalid correlator type!")

        self.soft_corr(corr_type, n_entries, ph_hist, n_corr_channels, self.corr_py)
        if n_corr_channels[0] != self.tot_corr_chan_len:
            raise ValueError("Number of correlator channels inconsistent!")

        valid_corr = self.corr_py[2, :] > 0  # weights > 0
        self.lag = (self.corr_py[1, :] * timebase_ms)[valid_corr]
        self.corrfunc = self.corr_py[0, :][valid_corr]
        self.weights = self.corr_py[2, :][valid_corr]

    def list_output(self, correlator_output_list: list) -> SoftwareCorrelatorListOutput:
        """Accumulate all software correlator outputs in lists"""

        lag_list = [output.lag for output in correlator_output_list]
        corrfunc_list = [output.corrfunc for output in correlator_output_list]
        weights_list = [output.weights for output in correlator_output_list]
        countrate_list = [output.countrate for output in correlator_output_list]

        return SoftwareCorrelatorListOutput(lag_list, corrfunc_list, weights_list, countrate_list)

    def output(self) -> SoftwareCorrelatorOutput:
        """Doc."""

        return SoftwareCorrelatorOutput(self.lag, self.corrfunc, self.weights, self.countrate)
