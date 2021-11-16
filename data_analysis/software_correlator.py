"""Software correlator"""

import sys
from ctypes import CDLL, c_double, c_int, c_long

import numpy as np
from numpy.ctypeslib import ndpointer


class CorrelatorType:
    PH_DELAY_CORRELATOR = 1
    PH_COUNT_CORRELATOR = 2
    PH_DELAY_CROSS_CORRELATOR = 3
    PH_COUNT_CROSS_CORRELATOR = 4  # does not seem to be implemented check! 1st column is photon arrival times, 2nd column boolean vector with 1s for photons arriving on the A channel and  0s otherwise, and the 3rd column is 1s for photons arriving on B channel and 0s otherwise
    PH_DELAY_CORRELATOR_LINES = 5  # additional column with 1s for photons that have to be counted, on 0s the delay chain is reset; used in image correlation and in angular scan
    PH_DELAY_CROSS_CORRELATOR_LINES = 6  # as PH_COUNT_CROSS_CORRELATOR with an additional column with 1s for photons that have to be counted, on 0s the delay chain is reset; used in image correlation and in angular scan


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

    def correlate(self, photon_array, c_type=CorrelatorType.PH_DELAY_CORRELATOR, timebase_ms=1):
        """Doc."""

        if sys.platform == "darwin":  # fix operation for Mac users
            photon_array = photon_array.astype(np.int64)

        if len(photon_array.shape) == 1:
            n_entries = photon_array.size
        else:
            n_entries = photon_array.shape[1]
        ph_hist = photon_array.reshape(-1)  # convert to 1D array
        n_corr_channels = np.zeros(1, dtype=np.int32)

        if c_type == CorrelatorType.PH_DELAY_CORRELATOR:
            if len(photon_array.shape) != 1:
                raise RuntimeError("Photon Array should be 1D for this correlator option!")
            self.countrate = n_entries / photon_array.sum() / timebase_ms * 1000

        elif c_type == CorrelatorType.PH_COUNT_CORRELATOR:
            if len(photon_array.shape) != 1:
                raise RuntimeError("Photon Array should be 1D for this correlator option!")
            self.countrate = photon_array.sum() / n_entries / timebase_ms * 1000

        elif c_type == CorrelatorType.PH_DELAY_CROSS_CORRELATOR:
            if (len(photon_array.shape) == 1) or (photon_array.shape[0] != 3):
                raise RuntimeError(
                    "Photon Array should have 3 rows for this correlator option! 0th row with photon delay times, 1st (2nd)  row contains 1s for photons in channel A (B) and 0s for photons in channel B(A)"
                )
            duration_s = photon_array[0, :].sum() * timebase_ms / 1000
            self.countrate_a = photon_array[1, :].sum() / duration_s
            self.countrate_b = photon_array[2, :].sum() / duration_s

        elif c_type == CorrelatorType.PH_DELAY_CORRELATOR_LINES:
            if (len(photon_array.shape) == 1) or (photon_array.shape[0] != 2):
                raise RuntimeError(
                    "Photon Array should have 2 rows for this correlator option! 0th row with photon delay times, 1st row is 1 for valid lines"
                )
            valid = (photon_array[1, :] == 1) | (photon_array[1, :] == -2)
            duration_s = photon_array[0, valid].sum() * timebase_ms / 1000
            self.countrate = np.sum(photon_array[1, :] == 1) / duration_s

        elif c_type == CorrelatorType.PH_DELAY_CROSS_CORRELATOR_LINES:
            if (len(photon_array.shape) == 1) or (photon_array.shape[0] != 4):
                raise RuntimeError(
                    "Photon Array should have 3 rows for this correlator option! 0th row with photon delay times, 1st (2nd)  row contains 1s for photons in channel A (B) and 0s for photons in channel B(A), and 3rd column is 1s for valid lines"
                )
            valid = (photon_array[3, :] == 1) | (photon_array[3, :] == -2)
            duration_s = photon_array[0, valid].sum() * timebase_ms / 1000
            self.countrate_a = np.sum(photon_array[1, :] == 1) / duration_s
            self.countrate_b = np.sum(photon_array[2, :] == 1) / duration_s
        else:
            raise ValueError("Invalid correlator type!")

        self.soft_corr(c_type, n_entries, ph_hist, n_corr_channels, self.corr_py)
        if n_corr_channels[0] != self.tot_corr_chan_len:
            raise ValueError("Number of correlator channels inconsistent!")

        [print(f"{weight:.0f}", end=", ") for weight in self.corr_py[2, :]]  # TESTESTEST
        print("\n")  # TESTESTEST

        valid_corr = self.corr_py[2, :] > 0  # weights > 0
        self.lag = (self.corr_py[1, :] * timebase_ms)[valid_corr]
        self.corrfunc = self.corr_py[0, :][valid_corr]
        self.weights = self.corr_py[2, :][valid_corr]
