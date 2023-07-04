"""Data Processing."""

import multiprocessing as mp
from collections import deque
from contextlib import suppress
from copy import copy
from dataclasses import InitVar, dataclass
from functools import partial
from itertools import count as infinite_range
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import scipy
import skimage

from data_analysis.workers import N_CPU_CORES, get_xcorr_input_dict
from utilities.display import Plotter
from utilities.file_utilities import load_object, save_object
from utilities.fit_tools import FitParams, curve_fit_lims
from utilities.helper import (
    Gate,
    Limits,
    MemMapping,
    chunked_bincount,
    div_ceil,
    nan_helper,
    xcorr,
)

NAN_PLACEBO = -100  # marks starts/ends of lines
LINE_END_ADDER = 1000
ZERO_LINE_START_ADDER = 2000


class CircularScanDataMixin:
    """Doc."""

    def _sum_scan_circles(self, pulse_runtime, laser_freq_hz, ao_sampling_freq_hz, circle_freq_hz):
        """Doc."""

        # calculate the number of samples obtained at each photon arrival, since beginning of file
        sample_runtime = pulse_runtime * ao_sampling_freq_hz // laser_freq_hz
        # calculate to which pixel each photon belongs (possibly many samples per pixel)
        samples_per_circle = int(ao_sampling_freq_hz / circle_freq_hz)
        pixel_num = sample_runtime % samples_per_circle

        # build the 'line image'
        bins = np.arange(-0.5, samples_per_circle)
        img, _ = np.histogram(pixel_num, bins=bins)

        return img, samples_per_circle


class AngularScanDataMixin:
    """Doc."""

    def convert_angular_scan_to_image(
        self,
        pulse_runtime,
        laser_freq_hz,
        ao_sampling_freq_hz,
        samples_per_line,
        n_lines,
        should_fix_shift=True,
        pix_shift=0,
        **kwargs,
    ):
        """Converts angular scan pulse_runtime into an image."""

        # calculate the number of samples obtained at each photon arrival, since beginning of file
        sample_runtime = pulse_runtime * ao_sampling_freq_hz // laser_freq_hz
        # calculate to which pixel each photon belongs (possibly many samples per pixel)
        pixel_num = sample_runtime % samples_per_line
        # calculate to which line each photon belongs (global, not considering going back to the first line)
        line_num_tot = sample_runtime // samples_per_line
        # calculate to which line each photon belongs (extra line is for returning to starting position)
        line_num = (line_num_tot % (n_lines + 1)).astype(np.int16)

        # build the image
        img = np.empty((n_lines + 1, samples_per_line), dtype=np.uint16)
        bins = np.arange(-0.5, samples_per_line)
        for j in range(n_lines + 1):
            img[j, :], _ = np.histogram(pixel_num[line_num == j], bins=bins)

        if should_fix_shift:
            if kwargs.get("is_verbose"):
                print("Fixing data shift...", end=" ")
            pix_shift = self._get_data_shift(img, **kwargs)
            if kwargs.get("is_verbose"):
                print(f"shifting {pix_shift} pixels...", end=" ")
            shifted_pulse_runtime = pulse_runtime + pix_shift * round(
                self.laser_freq_hz / ao_sampling_freq_hz
            )
            return self.convert_angular_scan_to_image(
                shifted_pulse_runtime,
                self.laser_freq_hz,
                ao_sampling_freq_hz,
                samples_per_line,
                n_lines,
                pix_shift=pix_shift,
                should_fix_shift=False,
            )

        else:
            # flip every second row to get an image (back and forth scan)
            img[1::2, :] = np.flip(img[1::2, :], 1)
            return img, sample_runtime, pixel_num, line_num, pix_shift

    def _get_data_shift(self, cnt: np.ndarray, median_factor=1.5, **kwargs) -> int:
        """Doc."""

        def get_best_pix_shift(img: np.ndarray, min_shift, max_shift, step) -> int:
            """Doc."""

            pix_shifts = np.arange(min_shift, max_shift, step)
            score = np.empty(pix_shifts.shape, dtype=np.uint32)
            for idx, pix_shift in enumerate(range(min_shift, max_shift, step)):
                rolled_img = np.roll(img, pix_shift).astype(np.int32)
                score[idx] = (abs(rolled_img[:-1:2, :] - np.fliplr(rolled_img[1::2, :]))).sum()
            return pix_shifts[score.argmin()]

        cnt = cnt.copy()
        height, width = cnt.shape
        step = div_ceil(width, 1000)  # handling "slow" scans with many points per line

        # replacing outliers with median value
        med = np.median(cnt)
        cnt[cnt > med * 1.5] = med

        # limit initial attempt to the width of the image
        min_pix_shift = -round(width / 2)
        max_pix_shift = min_pix_shift + width + 1
        pix_shift = get_best_pix_shift(cnt, min_pix_shift, max_pix_shift, step)

        # Test if not stuck in local minimum. Either 'outer_half_sum > inner_half_sum'
        # Or if the 'return row' (the empty one) is not at the bottom after shift
        rolled_cnt = np.roll(cnt, pix_shift)
        inner_half_sum = rolled_cnt[:, int(width * 0.25) : int(width * 0.75)].sum()
        outer_half_sum = rolled_cnt.sum() - inner_half_sum
        return_row_idx = rolled_cnt.sum(axis=1).argmin()

        # in case initial attempt fails, limit shift to the flattened size of the image
        if (outer_half_sum > inner_half_sum) or return_row_idx != height - 1:
            if return_row_idx != height - 1 and kwargs.get("is_verbose"):
                print("Data is heavily shifted, check it out!", end=" ")
            min_pix_shift = -round(cnt.size / 2)
            max_pix_shift = min_pix_shift + cnt.size + 1
            pix_shift = get_best_pix_shift(cnt, min_pix_shift, max_pix_shift, step)

        return pix_shift

    def _threshold_and_smooth(
        self, img, otsu_classes=4, n_bins=256, disk_radius=2, median_factor=1.1, **kwargs
    ) -> np.ndarray:
        """Doc."""

        # global filtering of outliers (replace bright pixels with median of central area)
        img = img.copy()
        _, width = img.shape
        median = np.median(img[:, int(width * 0.25) : int(width * 0.75)])
        img[img > median * median_factor] = median

        # minor local filtering of outliers then thresholding
        thresh = skimage.filters.threshold_multiotsu(
            skimage.filters.median(img).astype(np.float32), otsu_classes, nbins=n_bins
        )
        cnt_dig = np.digitize(img, bins=thresh)

        plateau_lvl = np.median(img[cnt_dig == (otsu_classes - 1)])
        std_plateau = scipy.stats.median_abs_deviation(img[cnt_dig == (otsu_classes - 1)])
        dev_cnt = img - plateau_lvl
        bw = dev_cnt >= -std_plateau

        bw = scipy.ndimage.binary_fill_holes(bw)
        disk_open = skimage.morphology.disk(radius=disk_radius)
        bw = skimage.morphology.opening(bw, footprint=disk_open)

        return bw

    def _get_line_markers_and_roi(
        self,
        bw_mask: np.ndarray,
        pulse_runtime: np.ndarray,
        laser_freq_hz: int,
        ao_sampling_freq_hz: int,
    ):
        """Doc."""

        line_starts = []
        line_stops = []
        line_start_labels = []
        line_stop_labels = []
        roi: Dict[str, deque] = {"row": deque([]), "col": deque([])}

        sample_runtime = pulse_runtime * ao_sampling_freq_hz // laser_freq_hz

        for row_idx in np.unique(bw_mask.nonzero()[0]):
            nonzero_row_idxs = bw_mask[row_idx, :].nonzero()[0]
            left_edge, right_edge = nonzero_row_idxs[0], nonzero_row_idxs[-1]
            # add row to ROI
            roi["row"].appendleft(row_idx)
            roi["col"].appendleft(left_edge)
            roi["row"].append(row_idx)
            roi["col"].append(right_edge)

            line_starts_new_idx = np.ravel_multi_index((row_idx, left_edge), bw_mask.shape)
            line_starts_new = list(
                range(sample_runtime[0] + line_starts_new_idx, sample_runtime[-1], bw_mask.size)
            )
            line_stops_new_idx = np.ravel_multi_index((row_idx, right_edge), bw_mask.shape)
            line_stops_new = list(
                range(sample_runtime[0] + line_stops_new_idx, sample_runtime[-1], bw_mask.size)
            )

            try:
                line_starts_new, line_stops_new = [
                    list(tup) for tup in zip(*zip(line_starts_new, line_stops_new))
                ]
            except ValueError:
                continue
            else:
                line_start_labels += [
                    (-ZERO_LINE_START_ADDER if row_idx == 0 else -row_idx)
                    for elem in range(len(line_starts_new))
                ]
                line_stop_labels += [
                    (-row_idx - LINE_END_ADDER) for elem in range(len(line_stops_new))
                ]
                line_starts += line_starts_new
                line_stops += line_stops_new

        # repeat first point to close the polygon
        roi["row"].append(roi["row"][0])
        roi["col"].append(roi["col"][0])

        # convert lists/deques to numpy arrays
        roi = {key: np.array(val, dtype=np.uint16) for key, val in roi.items()}
        line_start_labels = np.array(line_start_labels, dtype=np.int16)
        line_stop_labels = np.array(line_stop_labels, dtype=np.int16)
        line_starts = np.array(line_starts, dtype=np.int64)
        line_stops = np.array(line_stops, dtype=np.int64)

        line_starts_prt = line_starts * round(laser_freq_hz / ao_sampling_freq_hz)
        scan_line_stops_prt = line_stops * round(laser_freq_hz / ao_sampling_freq_hz)

        return line_starts_prt, scan_line_stops_prt, line_start_labels, line_stop_labels, roi

    def get_bright_pixels(self, img: np.ndarray, mask: np.ndarray, thresh_factor=1000, **kwargs):
        """Get a mask for 'bright pixels' of an image using a histogram-based heuristic method."""

        # select number of bins based on number of unique values in ROI
        n_unique = len(np.unique(img[mask]))
        n_bins = max(10, round(n_unique / 3.5))

        scan_hist, bin_edges = np.histogram(img[mask], bins=n_bins)
        bin_num = np.arange(len(scan_hist))
        bg_part = round(n_bins / 10)

        # fit Gaussian to histogram
        FP = curve_fit_lims(
            "gaussian_1d_fit",
            (
                scan_hist[bg_part:].max() / 10,
                np.mean(bin_num[bg_part:]),
                len(bin_num[bg_part:]) / 10,
                0,
            ),
            bin_num[bg_part:],
            scan_hist[bg_part:],
            bounds=(  # A, mu, sigma, bg
                [0, 0, 0, -10],
                [1e4, bin_num.max(), bin_num.max(), 0],
            ),
        )

        # get bin where fitted_y drops to 1/N_TRESH of value and use as threshold
        try:
            thresh_bin = FP.x[
                (FP.x > FP.beta["mu"]) & (FP.fitted_y < FP.beta["A"] / thresh_factor)
            ][0]
        except IndexError:
            thresh_bin = bin_num.max()

        # get the bad pixels
        d = np.digitize(img, bin_edges)
        return d > thresh_bin

    def _bg_line_correlations(
        self,
        image1: np.ndarray,
        bw_mask: np.ndarray,
        valid_lines: np.ndarray,
        sampling_freq_Hz,
        image2: np.ndarray = None,
        **kwargs,
    ) -> list:
        """Returns a list of auto-correlations of the lines of an image."""

        is_doing_xcorr = image2 is not None

        line_corr_list = []
        for j in valid_lines:
            line1 = image1[j][bw_mask[j]].astype(np.float64)
            if is_doing_xcorr:
                line2 = image2[j][bw_mask[j]].astype(np.float64)
            try:
                if not is_doing_xcorr:
                    c, lags = xcorr(line1, line1)
                else:
                    c, lags = xcorr(line1, line2)
            except ValueError:
                if kwargs.get("is_verbose"):
                    print(f"Correlation of line No.{j} has failed. Skipping.", end=" ")
            else:
                if not is_doing_xcorr:  # Autocorrelation
                    c = c / line1.mean() ** 2 - 1
                    c[0] -= 1 / line1.mean()  # subtracting shot noise, small stuff really
                else:  # Cross-Correlation
                    c = c / (line1.mean() * line2.mean()) - 1
                    c[0] -= 1 / np.sqrt(
                        line1.mean() * line2.mean()
                    )  # subtracting shot noise, small stuff really

                line_corr_list.append(
                    {
                        "lag": lags * 1e3 / sampling_freq_Hz,  # in ms
                        "corrfunc": c,
                    }
                )
        return line_corr_list

    def normalize_scan_img_rows(self, img: np.ndarray, mask=None):
        """Normalize an image to the median of the maximum row. Optionally use a supplied mask first."""

        if mask is None:
            mask = np.full(img.shape, True, dtype=np.bool)
            mask[-1, :] = False  # last row is always empty

        temp_img = img.copy().astype(np.float64)
        temp_img[~mask] = 0
        max_row = np.argmax(temp_img.sum(axis=1))
        max_row_median = np.median(img[max_row][mask[max_row]])
        norm_masked_img = img.astype(np.float64)
        for row_idx in np.unique(norm_masked_img.nonzero()[0]):
            if mask[row_idx].any():
                norm_masked_img[row_idx] *= max_row_median / np.median(img[row_idx][mask[row_idx]])
        return norm_masked_img


class RawFileData:
    """
    Holds a single file's worth of processed, TDC-based, time-tagged photon data which is used in turn for photon delay-time calibration
    and analysis of the entire measurement data.
    This version uses Numpy.mmap (memory-mapping) for getting the initially save 'raw' data (which is unchanging after initial processing of byte data.
    Therefore, no actual data is ever kept in the object (unless should_dump=False)
    """

    def __init__(
        self,
        idx: int,
        dump_path: Path,
        coarse: np.ndarray,
        coarse2: np.ndarray,
        fine: np.ndarray,
        pulse_runtime: np.ndarray,
        delay_time: np.ndarray,
        line_num: np.ndarray = None,
        should_avoid_dumping=False,
        **kwargs,
    ):
        """
        Unite the (equal shape) arrays into a single 2D array and immediately dump to disk.
        From now on will be accessed via memory mapping.
        """

        self._idx = idx
        self._file_name = f"raw_data_{idx}.npy"
        self._dump_path = dump_path
        self.dump_file_path = self._dump_path / self._file_name
        self.should_avoid_dumping = should_avoid_dumping

        # keep data until dumped
        self._coarse = coarse
        self._coarse2 = coarse2
        self._fine = fine
        self._pulse_runtime = pulse_runtime
        self._delay_time = delay_time
        self._line_num = line_num

        # keep track
        self._was_data_dumped = False
        self.compressed_file_path: Path = None

    @property
    def coarse(self):
        if self._was_data_dumped:
            return self.read_mmap(0).astype(np.int16)
        else:
            return self._coarse

    @coarse.setter
    def coarse(self, new: np.ndarray):
        if self._was_data_dumped:
            return RuntimeError("RawFileData attributes are read-only.")
        else:
            self._coarse = new

    @property
    def coarse2(self):
        if self._was_data_dumped:
            return self.read_mmap(1).astype(np.int16)
        else:
            return self._coarse2

    @coarse2.setter
    def coarse2(self, new: np.ndarray):
        if self._was_data_dumped:
            return RuntimeError("RawFileData attributes are read-only.")
        else:
            self._coarse2 = new

    @property
    def fine(self):
        if self._was_data_dumped:
            return self.read_mmap(2).astype(np.int16)
        else:
            return self._fine

    @fine.setter
    def fine(self, new: np.ndarray):
        if self._was_data_dumped:
            return RuntimeError("RawFileData attributes are read-only.")
        else:
            self._fine = new

    @property
    def pulse_runtime(self):
        if self._was_data_dumped:
            return self.read_mmap(3).astype(np.int64)
        else:
            return self._pulse_runtime

    @pulse_runtime.setter
    def pulse_runtime(self, new: np.ndarray):
        if self._was_data_dumped:
            return RuntimeError("RawFileData attributes are read-only.")
        else:
            self._pulse_runtime = new

    @property
    def delay_time(self):
        if self._was_data_dumped:
            return self.read_mmap(4).astype(np.float16)
        else:
            return self._delay_time

    @delay_time.setter
    def delay_time(self, new: np.ndarray):
        # NOTE: this does not allow for slicing! must assign a full delay_time array of same size
        if self._was_data_dumped:
            return self.write_mmap_row(4, new)
        else:
            self._delay_time = new

    @property
    def line_num(self):
        if self._was_data_dumped:
            try:
                return self.read_mmap(5).astype(np.int16)
            except IndexError:
                return None
        else:
            # TODO: IS THIS A MISTAKE??
            return self._pulse_runtime

    #            return self._line_num

    @line_num.setter
    def line_num(self, new: np.ndarray):
        if self._was_data_dumped:
            return RuntimeError("RawFileData attributes are read-only.")
        else:
            self._line_num = new

    def read_mmap(self, row_idx: Union[int, slice] = slice(None), file_path=None):
        """
        Access the data from disk by memory-mapping, and get the 'row_idx' row.
        If row_idx was not supplied, retrieve the whole array.
        each read should take about 1 ms, therefore unnoticeable.
        """

        return np.load(
            file_path if file_path is not None else self.dump_file_path,
            mmap_mode="r",
            allow_pickle=True,
            fix_imports=False,
        )[row_idx]

    def write_mmap_row(self, row_idx: int, new_row: np.ndarray):
        """
        Access the data from disk by memory-mapping, get the 'row_idx' row and write to it.
        each write should take about ???, therefore unnoticeable.
        """

        data = np.load(
            self.dump_file_path,
            mmap_mode="r+",
            allow_pickle=False,
            fix_imports=False,
        )
        data[row_idx] = new_row
        data.flush()

    def dump(self, is_verbose=False):
        """Dump the data to disk. Should be called right after initialization (only needed once). Returns whether the data was dumped."""

        # cancel dumping if not needed
        if (
            not self._was_data_dumped
            and not (self.dump_file_path.exists() and self.should_avoid_dumping)
            or not self.dump_file_path.exists()
        ):

            if is_verbose:
                print(f"Dumping data to '{self.dump_file_path}' (Memory-Mapping).")

            # prepare data ndarray
            if self._line_num is None:
                data = np.vstack(
                    (self._coarse, self._coarse2, self._fine, self._pulse_runtime, self._delay_time)
                )
            else:
                data = np.vstack(
                    (
                        self._coarse,
                        self._coarse2,
                        self._fine,
                        self._pulse_runtime,
                        self._delay_time,
                        self._line_num,
                    )
                )

            # save
            Path.mkdir(self._dump_path, parents=True, exist_ok=True)
            np.save(
                self.dump_file_path,
                data,
                allow_pickle=False,
                fix_imports=False,
            )

        # keep track
        if self.dump_file_path.exists():
            self._was_data_dumped = True

        # clear the RAM
        self._coarse = None
        self._coarse2 = None
        self._fine = None
        self._pulse_runtime = None
        self._delay_time = None
        self._line_num = None

    def save_compressed(self, dir_path: Path = None):
        """Compress (blosc) and save the memory-mapped data in a provided folder"""

        # load the data from dump path
        data = self.read_mmap()

        # compress and save to processed folder ('dir_path')
        Path.mkdir(dir_path, parents=True, exist_ok=True)  # creating the folder if needed
        self.compressed_file_path = (dir_path / self._file_name).with_suffix(".blosc")
        save_object(
            data,
            self.compressed_file_path,
            compression_method="blosc",
            element_size_estimate_mb=data.nbytes / 1e6,
            obj_name="dumped data array",
            should_track_progress=True,
        )

    def load_compressed(self, dir_path: Path):
        """Decompress and re-save the processed data in the dump path for analysis"""

        # load data from processed folder
        data = load_object(
            (dir_path / self._file_name).with_suffix(".blosc"), should_track_progress=True
        )

        # resave (dump), creating the dumpt folder first if needed
        Path.mkdir(self._dump_path, parents=True, exist_ok=True)
        np.save(
            self.dump_file_path,
            data,
            allow_pickle=False,
            fix_imports=False,
        )


@dataclass
class GeneralFileData:
    """
    Holds miscellaneous information of a single meausrement file.
    Not heavy and therefore stays in RAM during analysis (is not dumped).
    """

    # general
    laser_freq_hz: int
    section_runtime_edges: list
    size_estimate_mb: float
    duration_s: float
    skipped_duration: float
    avg_cnt_rate_khz: float = None
    image: np.ndarray = None
    bg_line_corr: List[Dict[str, Any]] = None

    # continuous scan
    all_section_edges: np.ndarray = None

    # angular scan
    valid_lines: np.ndarray = None
    samples_per_line: int = None
    n_lines: int = None
    roi: Dict[str, deque] = None
    bw_mask: np.ndarray = None
    image_bw_mask: np.ndarray = None
    pix_shift: int = None
    single_scan_edges: List[Tuple[int, int]] = None
    normalized_masked_alleviated_image: np.ndarray = None


@dataclass
class AfterpulsingFilter:
    """Doc."""

    t_hist: np.ndarray
    all_hist_norm: np.ndarray
    baseline: float
    I_j: np.ndarray
    norm_factor: float
    valid_limits: Limits
    M: np.ndarray
    filter: np.ndarray
    fine_bins: np.ndarray

    def get_split_filter_input(
        self,
        split_dt: np.ndarray,
        get_afterpulsing=False,
        **kwargs,
    ) -> np.ndarray:
        """Doc."""

        # prepare filter input if afterpulsing filter is supplied
        filter = self.filter[int(get_afterpulsing)]
        # create a filter for genuine fluorscene (ignoring afterpulsing)
        bin_num = np.digitize(split_dt, self.fine_bins)
        # adding a final zero value for NaNs (which are put in the last bin by np.digitize)
        filter = np.hstack((filter, [0]))  # TODO: should the filter be created like this?
        # add the relevent filter values to the correlator filter input list
        filter_input = filter[bin_num - 1]
        return filter_input

    def plot(self, parent_ax=None, **plot_kwargs):
        """Doc."""

        valid_idxs = self.valid_limits.valid_indices(self.t_hist)

        with Plotter(
            parent_ax=parent_ax,
            subplots=(1, 2),
            **plot_kwargs,
        ) as axes:
            axes[0].set_title("Filter Ingredients")
            axes[0].set_yscale("log")
            axes[0].plot(
                self.t_hist, self.all_hist_norm / self.norm_factor, label="norm. raw histogram"
            )
            axes[0].plot(self.t_hist[valid_idxs], self.I_j / self.norm_factor, label="norm. I_j")
            axes[0].plot(
                self.t_hist[valid_idxs],
                self.baseline / self.norm_factor * np.ones(self.t_hist[valid_idxs].shape),
                label="norm. baseline",
            )
            axes[0].plot(
                self.t_hist[valid_idxs],
                self.M.T[0],
                label="M_j1 (ideal fluorescence decay curve)",
            )
            axes[0].plot(
                self.t_hist[valid_idxs],
                self.M.T[1],
                label="M_j2 (ideal afterpulsing 'decay' curve)",
            )
            axes[0].legend()
            axes[0].set_ylim(self.baseline / self.norm_factor / 10, None)

            axes[1].set_title("Filter")
            try:  # TODO: fix this at the filter-building level!
                axes[1].plot(
                    self.t_hist, self.filter.T, label=["F_1j (signal)", "F_2j (afterpulsing)"]
                )
                axes[1].plot(self.t_hist, self.filter.sum(axis=0), label="F.sum(axis=0)")
            except ValueError as exc:
                print(exc)
                print("^ ignoring the last filter element...")
                axes[1].plot(
                    self.t_hist, self.filter.T[:-1], label=["F_1j (signal)", "F_2j (afterpulsing)"]
                )
                axes[1].plot(self.t_hist, self.filter.sum(axis=0)[:-1], label="F.sum(axis=0)")

            axes[1].legend()

            if self.valid_limits.upper != np.inf:
                axes[0].set_xlim(*self.valid_limits)
                axes[1].set_xlim(*self.valid_limits)


@dataclass
class TDCPhotonFileData:
    """Holds the total processed data of a single measurement file."""

    idx: int
    general: GeneralFileData
    raw: RawFileData
    dump_path: Path

    def __repr__(self):
        return f"TDCPhotonFileData(idx={self.idx}, dump_path={self.dump_path})"

    def import_raw(self, raw_data: np.ndarray, **kwargs):
        """Load RawFileData using an existing ndarray"""

        self.raw = RawFileData(
            self.idx,
            self.dump_path,
            *[line for line in raw_data],
            **kwargs,
        )

    def get_xcorr_input_dict(
        self, xcorr_types: List[str], gate1_ns=Gate(), gate2_ns=Gate(), **kwargs
    ):
        """Return a list of SoftwareCorrelator input units (splits) from a measurement data in a single file (self)"""

        if self.raw.line_num is not None:  # line data
            return self._get_line_xcorr_input_dict(xcorr_types, gate1_ns, gate2_ns, **kwargs)
        else:  # continuous data
            return self._get_continuous_xcorr_input_dict(xcorr_types, gate1_ns, gate2_ns, **kwargs)

    def _get_line_xcorr_input_dict(
        self,
        xcorr_types: List[str],
        gate1_ns,
        gate2_ns,
        **kwargs,
    ):
        """Splits are all photons belonging to each scan line."""

        # Gating is performed here
        dt = self.raw.delay_time
        # NOTE: NaNs mark line starts/ends (used to create valid = -1/-2 needed in C code)
        nan_idxs = np.isnan(dt)
        if "A" in "".join(xcorr_types):
            gate1_idxs = gate1_ns.valid_indices(dt)
            valid_idxs1 = gate1_idxs | nan_idxs
            dt1 = dt[valid_idxs1]
            prt1 = self.raw.pulse_runtime[valid_idxs1]
            ts1 = np.hstack(([0], np.diff(prt1)))
            dt_ts1 = np.vstack((dt1, ts1))
            line_num1 = self.raw.line_num[valid_idxs1]
        if "B" in "".join(xcorr_types):
            gate2_idxs = gate2_ns.valid_indices(dt)
            valid_idxs2 = gate2_idxs | nan_idxs
            dt2 = dt[valid_idxs2]
            prt2 = self.raw.pulse_runtime[valid_idxs2]
            ts2 = np.hstack(([0], np.diff(prt2)))
            dt_ts2 = np.vstack((dt2, ts2))
            line_num2 = self.raw.line_num[valid_idxs2]
        if "AB" in xcorr_types or "BA" in xcorr_types:
            # NOTE: # gate2 is first in line to match how software correlator C-code is written
            dt_ts12 = np.vstack(
                (
                    dt,
                    self.raw.pulse_runtime,
                    valid_idxs2,
                    valid_idxs1,
                )
            )[:, valid_idxs1 | valid_idxs2]
            dt_ts12[0] = np.hstack(([0], np.diff(dt_ts12[0])))
            line_num12 = self.raw.line_num[valid_idxs1 | valid_idxs2]

        xcorr_input_dict: Dict[str, List[np.ndarray]] = {xx: [] for xx in xcorr_types}
        for line_idx in self.general.valid_lines:
            for xx in xcorr_types:
                if xx == "AA":
                    xcorr_input_dict[xx].append(
                        self._add_validity(dt_ts1, line_idx, line_num1, **kwargs)
                    )
                if xx == "BB":
                    xcorr_input_dict[xx].append(
                        self._add_validity(dt_ts2, line_idx, line_num2, **kwargs)
                    )
                if xx == "AB":
                    xcorr_input_AB = self._add_validity(
                        dt_ts12,
                        line_idx,
                        line_num12,
                        **kwargs,
                    )
                    xcorr_input_dict[xx].append(xcorr_input_AB)
                if xx == "BA":
                    if "AB" in xcorr_types:
                        xcorr_input_dict[xx].append(xcorr_input_AB[[0, 2, 1, 3], :])
                    else:
                        dt_ts21 = dt_ts12[[0, 2, 1, 3], :]
                        xcorr_input_dict[xx].append(
                            self._add_validity(
                                dt_ts21,
                                line_idx,
                                line_num12,
                                **kwargs,
                            )
                        )

        print(".", end="")  # TESTESTEST
        return xcorr_input_dict

    def _get_continuous_xcorr_input_dict(
        self,
        xcorr_types: List[str],
        gate1_ns,
        gate2_ns,
        *args,
        n_splits_requested=10,
        **kwargs,
    ):
        """Continuous scan/static measurement - splits are arbitrarily cut along the measurement"""

        dt = self.raw.delay_time

        # TODO: split duration (in bytes! not time) should be decided upon according to how well the correlator performs with said split size.
        # Currently it is arbitrarily decided by 'n_splits_requested' which causes inconsistent processing times for each split
        split_duration = self.general.duration_s / n_splits_requested
        for se_idx, (se_start, se_end) in enumerate(self.general.all_section_edges):
            # split into sections of approx time of run_duration
            section_time = (
                self.raw.pulse_runtime[se_end] - self.raw.pulse_runtime[se_start]
            ) / self.general.laser_freq_hz

            section_pulse_runtime = self.raw.pulse_runtime[se_start : se_end + 1]
            section_delay_time = dt[se_start : se_end + 1]

            # split the data into parts A/B according to gates
            if "A" in "".join(xcorr_types):
                gate1_idxs = gate1_ns.valid_indices(section_delay_time)
                section_prt1 = section_pulse_runtime[gate1_idxs]
                section_dt1 = section_delay_time[gate1_idxs]
                section_dt_prt1 = np.vstack((section_dt1, section_prt1))

            if "B" in "".join(xcorr_types):
                gate2_idxs = gate2_ns.valid_indices(section_delay_time)
                section_prt2 = section_pulse_runtime[gate2_idxs]
                section_dt2 = section_delay_time[gate2_idxs]
                section_dt_prt2 = np.vstack((section_dt2, section_prt2))
            if "AB" in xcorr_types or "BA" in xcorr_types:
                section_prt12 = section_pulse_runtime
                section_dt_prt12 = np.vstack(
                    (section_delay_time, section_prt12, gate2_idxs, gate1_idxs)
                )[:, gate1_idxs | gate2_idxs]

            xcorr_input_dict: Dict[str, List[np.ndarray]] = {xx: [] for xx in xcorr_types}
            for split_idx in range(n_splits := div_ceil(section_time, split_duration)):
                for xx in xcorr_types:
                    if xx == "AA":
                        xcorr_input_dict[xx].append(
                            self._split_continuous_section(
                                section_dt_prt1,
                                split_idx,
                                n_splits,
                                **kwargs,
                            )
                        )
                    if xx == "BB":
                        xcorr_input_dict[xx].append(
                            self._split_continuous_section(
                                section_dt_prt2,
                                split_idx,
                                n_splits,
                                **kwargs,
                            )
                        )
                    if xx == "AB":
                        xcorr_input_AB = self._split_continuous_section(
                            section_dt_prt12,
                            split_idx,
                            n_splits,
                            **kwargs,
                        )
                        xcorr_input_dict[xx].append(xcorr_input_AB)
                    if xx == "BA":
                        if "AB" in xcorr_types:
                            xcorr_input_dict[xx].append(xcorr_input_AB[[0, 2, 1], :])
                        else:
                            section_dt_prt21 = section_dt_prt12[[0, 2, 1], :]
                            xcorr_input_dict[xx].append(
                                self._split_continuous_section(
                                    section_dt_prt21,
                                    split_idx,
                                    n_splits,
                                    **kwargs,
                                )
                            )

        print(".", end="")  # TESTESTEST
        return xcorr_input_dict

    def _split_continuous_section(
        self,
        dt_prt_in,
        idx,
        n_splits: int,
        **kwargs,
    ) -> np.ndarray:
        """Doc."""

        splits = np.linspace(0, dt_prt_in.shape[1], n_splits + 1, dtype=np.int32)
        return dt_prt_in[:, splits[idx] : splits[idx + 1]]

    def _add_validity(
        self,
        dt_ts_in,
        idx,
        line_num,
        **kwargs,
    ) -> np.ndarray:
        """Doc."""

        valid = (line_num == idx).astype(np.int8)
        if idx != 0:
            valid[line_num == -idx] = -1
        else:
            valid[line_num == -ZERO_LINE_START_ADDER] = -1
        valid[line_num == -idx - LINE_END_ADDER] = -2

        #  remove photons from wrong lines
        dt_ts_out = dt_ts_in[:, valid != 0]
        valid = valid[valid != 0]

        if valid.any():
            # the first photon in line measures the time from line start and the line end (-2) finishes the duration of the line
            # check that we start with the line beginning and not its end
            if valid[0] != -1:
                # remove photons till the first found beginning
                j_start = np.where(valid == -1)[0]

                if len(j_start) > 0:
                    dt_ts_out = dt_ts_out[:, j_start[0] :]
                    valid = valid[j_start[0] :]

            # check that we stop with the line ending and not its beginning
            if valid[-1] != -2:
                # remove photons after the last found ending
                j_end = np.where(valid == -2)[0]

                if len(j_end) > 0:
                    *_, j_end_last = j_end
                    dt_ts_out = dt_ts_out[:, : j_end_last + 1]
                    valid = valid[: j_end_last + 1]

            # back to prt
            prt = dt_ts_out[1:].cumsum()
            dt_prt_out = np.vstack((dt_ts_out[0], prt, valid))

        else:
            dt_prt_out = np.vstack(([], []))

        return dt_prt_out


class TDCPhotonMeasurementData(list):
    """
    Holds lists of the entire meausrement data (all files).
    Capable of rotation to/from disk of specific data type.
    """

    def __init__(self):
        super().__init__()

    def prepare_xcorr_input(
        self, xcorr_types: List[str], should_parallel_process=False, **kwargs
    ) -> Dict[str, List]:
        """
        Prepare SoftwareCorrelator input from complete measurement data.
        Gates are meant to divide the data into 2 parts (A&B), each having its own splits.
        To perform autocorrelation, only one ("AA") is used, and in the default (0, inf) limits, with actual gating done later in 'correlate_data' method.
        """

        # parallel processing
        if should_parallel_process and (N_FILES := len(self)) > 20:
            N_PROCESSES = N_CPU_CORES - 1
            CHUNKSIZE = N_FILES // N_PROCESSES
            kwargs["xcorr_types"] = xcorr_types
            partial_get_splits_dict = partial(get_xcorr_input_dict, **kwargs)
            print(
                f"(Parallel processing using {N_PROCESSES} processes, with chunksize {CHUNKSIZE}) ",
                end="",
            )
            with mp.get_context().Pool(N_PROCESSES) as pool:
                file_xcorr_input_dict_list = list(
                    pool.imap_unordered(partial_get_splits_dict, self, CHUNKSIZE)
                )

        # serial (regular) processing
        else:
            #            print() # TESTESTEST
            file_xcorr_input_dict_list = [
                p.get_xcorr_input_dict(xcorr_types, **kwargs) for p in self
            ]

        # TODO: explain this in a comment (uniting the dicts)
        xcorr_input_dict = {
            xx: [
                xcorr_input
                for xcorr_input_dict in file_xcorr_input_dict_list
                for xcorr_input in xcorr_input_dict[xx]
            ]
            for xx in xcorr_types
        }

        return xcorr_input_dict


@dataclass
class TDCCalibration:
    """Doc."""

    x_all: np.ndarray
    h_all: np.ndarray
    coarse_bins: np.ndarray
    h: np.ndarray
    max_j: int
    coarse_calib_bins: Any
    fine_bins: Any
    l_quarter_tdc: Any
    r_quarter_tdc: Any
    t_calib: Any
    hist_weight: np.ndarray
    delay_times: Any
    total_laser_pulses: int
    h_tdc_calib: Any
    t_hist: Any
    all_hist: Any
    all_hist_norm: Any
    error_all_hist_norm: Any
    t_weight: Any

    def plot(self, **kwargs) -> None:
        """Doc."""

        try:
            x_all = self.x_all
            h_all = self.h_all
            coase_bins = self.coarse_bins
            h = self.h
            coarse_calib_bins = self.coarse_calib_bins
            t_calib = self.t_calib
            t_hist = self.t_hist
            all_hist_norm = self.all_hist_norm
        except AttributeError:
            return

        with Plotter(
            subplots=(2, 2),
            **kwargs,
        ) as axes:
            axes[0, 0].semilogy(x_all, h_all, "-o", label="All Hist")
            axes[0, 0].semilogy(coase_bins, h, "o", label="Valid Bins")
            axes[0, 0].semilogy(
                coase_bins[np.isin(coase_bins, coarse_calib_bins)],
                h[np.isin(coase_bins, coarse_calib_bins)],
                "o",
                markersize=4,
                label="Calibration Bins",
            )
            axes[0, 0].legend()

            axes[0, 1].plot(self.h_tdc_calib, "-o", label="Calibration Photon Histogram")
            axes[0, 1].legend()

            axes[1, 0].semilogy(t_hist, all_hist_norm, "-o", label="Photon Lifetime Histogram")
            axes[1, 0].legend()

            axes[1, 1].plot(t_calib, "-o", label="TDC Calibration")
            axes[1, 1].set_ylabel("Time (ns)")
            axes[1, 1].legend()

    def fit_lifetime_hist(
        self,
        x_field="t_hist",
        y_field="all_hist_norm",
        y_error_field="error_all_hist_norm",
        fit_param_estimate=[0.1, 4, 0.001],
        fit_range=(3.5, 30),
        x_scale="linear",
        y_scale="log",
        **kwargs,
    ) -> FitParams:
        """Doc."""

        is_finite_y = np.isfinite(getattr(self, y_field))

        return curve_fit_lims(
            "exponent_with_background_fit",
            fit_param_estimate,
            xs=getattr(self, x_field)[is_finite_y],
            ys=getattr(self, y_field)[is_finite_y],
            ys_errors=getattr(self, y_error_field)[is_finite_y],
            x_limits=Limits(fit_range),
            plot_kwargs=dict(x_scale=x_scale, y_scale=y_scale),
            **kwargs,
        )

    def calculate_afterpulsing_filter(
        self,
        gate_ns: Gate,
        meas_type: str,
        baseline_tail_perc=0.3,
        **kwargs,
    ) -> np.ndarray:
        """Doc."""

        # make copies so that originals are preserved
        all_hist_norm = copy(self.all_hist_norm)
        t_hist = copy(self.t_hist)

        # define valid bins to work with
        peak_bin = max(np.nanargmax(all_hist_norm) - 2, 0)
        peak_to_end_limits = Limits(t_hist[peak_bin], np.inf)
        valid_limits = peak_to_end_limits & gate_ns
        valid_idxs = valid_limits.valid_indices(t_hist)

        # interpolate over NaNs
        nans, x = nan_helper(all_hist_norm)  # get nans and a way to interpolate over them later
        #        print("nans: ", nans.sum()) # TESTESTEST - why always 23?
        all_hist_norm[nans] = np.interp(x(nans), x(~nans), all_hist_norm[~nans])

        # calculate the baseline using the mean of the tail of the (hard-gate-limited) histogram
        if gate_ns.hard_gate:
            in_hard_gate_idxs = gate_ns.hard_gate.valid_indices(t_hist)
        else:
            # use all indices
            in_hard_gate_idxs = slice(None)
        baseline = all_hist_norm[in_hard_gate_idxs][
            -round(len(t_hist) * baseline_tail_perc) :
        ].mean()

        # normalization factor
        norm_factor = (all_hist_norm[valid_idxs] - baseline).sum()

        # define matrices and calculate F
        M_j1 = (all_hist_norm[valid_idxs] - baseline) / norm_factor  # p1
        M_j2 = 1 / len(t_hist[valid_idxs]) * np.ones(t_hist[valid_idxs].shape)  # p2
        M = np.vstack((M_j1, M_j2)).T

        I_j = all_hist_norm[valid_idxs]
        I = np.diag(I_j)  # NOQA E741
        inv_I = np.linalg.pinv(I)

        F = np.linalg.pinv(M.T @ inv_I @ M) @ M.T @ inv_I

        # Return the filter to original dimensions by adding zeros in the detector-gated zone
        # prepare padding for the gated-out/noisy part of the histogram
        lower_idx = round(valid_limits.lower * 10)
        F_pad_before = np.zeros((2, lower_idx))
        if gate_ns.upper != np.inf:
            upper_idx = int(valid_limits.upper * 10)
            F_pad_after = np.zeros((2, len(t_hist) - upper_idx))
        else:
            F_pad_after = np.zeros((2, 0))
        F = np.hstack((F_pad_before, F, F_pad_after))

        if F.shape[1] != self.fine_bins.size - 1:  # TESTESTEST
            print(
                f"\nWARNING: Filter length ({F.shape[1]}) should be exactly fine_bins.size - 1 ({self.fine_bins.size - 1})!"
            )  # TESTESTEST

        ap_filter = AfterpulsingFilter(
            t_hist,
            all_hist_norm,
            baseline,
            I_j,
            norm_factor,
            valid_limits,
            M,
            F,
            self.fine_bins,
        )

        if kwargs.get("should_plot"):
            ap_filter.plot()

        return ap_filter


class TDCPhotonDataProcessor(AngularScanDataMixin, CircularScanDataMixin):
    """For processing raw bytes data"""

    GROUP_LEN: int = 7
    MAX_VAL: int = 256**3

    def __init__(
        self, dump_path: Path, laser_freq_hz: int, fpga_freq_hz: int, detector_gate_ns: Gate
    ):
        self.dump_path = dump_path
        self.laser_freq_hz = laser_freq_hz
        self.fpga_freq_hz = fpga_freq_hz
        self.detector_gate_ns = detector_gate_ns

    def process_data(self, idx, full_data, should_dump=True, **proc_options) -> TDCPhotonFileData:
        """Doc."""

        # sFCS
        if scan_settings := full_data.get("scan_settings"):
            if (scan_type := scan_settings["pattern"]) == "circle":  # Circular sFCS
                p = self._process_circular_scan_data_file(
                    idx, full_data, should_dump=should_dump, **proc_options
                )
            elif scan_type == "angular":  # Angular sFCS
                p = self._process_angular_scan_data_file(
                    idx, full_data, should_dump=should_dump, **proc_options
                )
            elif scan_type == "image":  # image sFCS
                p = self._process_image_scan_plane_data(
                    idx, full_data, should_dump=should_dump, **proc_options
                )
        # FCS
        else:
            scan_type = "static"
            p = self._convert_fpga_data_to_photons(
                idx,
                is_scan_continuous=True,
                **proc_options,
            )
            if should_dump:  # False when multiprocessing
                p.raw.dump()

        # add general properties
        with suppress(AttributeError):
            # AttributeError: p is None
            p.general.avg_cnt_rate_khz = full_data.get("avg_cnt_rate_khz")

        return p

    def _convert_fpga_data_to_photons(
        self,
        idx,
        byte_data_path=None,
        byte_data=None,
        is_scan_continuous=False,
        should_use_all_sections=True,
        len_factor=0.01,
        byte_data_slice=None,
        **proc_options,
    ) -> TDCPhotonFileData:
        """Doc."""

        if proc_options.get("is_verbose"):
            print("Converting raw data to photons...", end=" ")

        # getting byte data by memory-mapping (unless supplied - alignment measurements and pre-conversion data only)
        if byte_data is None:
            # NOTE - WARNING! byte data is memory mapped - mode must always be kept to 'r' to avoid writing over byte_data!!!
            byte_data = np.load(byte_data_path, "r")

        # option to use only certain parts of data (for testing)
        if byte_data_slice is not None:
            byte_data = byte_data[byte_data_slice]

        section_edges, tot_single_errors = self._find_all_section_edges(byte_data)
        section_lengths = [edge_stop - edge_start for (edge_start, edge_stop) in section_edges]

        if should_use_all_sections:
            photon_idxs_list: List[int] = []
            section_runtime_edges = []
            for start_idx, end_idx in section_edges:
                if end_idx - start_idx > sum(section_lengths) * len_factor:
                    section_runtime_start = len(photon_idxs_list)
                    section_photon_indxs = list(range(start_idx, end_idx, self.GROUP_LEN))
                    section_runtime_end = section_runtime_start + len(section_photon_indxs)
                    photon_idxs_list += section_photon_indxs
                    section_runtime_edges.append((section_runtime_start, section_runtime_end))

            photon_idxs = np.array(photon_idxs_list)

        else:  # using largest section only
            max_sec_start_idx, max_sec_end_idx = section_edges[np.argmax(section_lengths)]
            photon_idxs = np.arange(max_sec_start_idx, max_sec_end_idx, self.GROUP_LEN)
            section_runtime_edges = [(0, len(photon_idxs))]

        if proc_options.get("is_verbose"):
            if len(section_edges) > 1:
                print(
                    f"Found {len(section_edges)} sections of lengths: {', '.join(map(str, section_lengths))}.",
                    end=" ",
                )
                if should_use_all_sections:
                    print(
                        f"Using all valid (> {len_factor:.0%}) sections ({len(section_runtime_edges)}/{len(section_edges)}).",
                        end=" ",
                    )
                else:  # Use largest section only
                    print(f"Using largest (section num.{np.argmax(section_lengths)+1}).", end=" ")
            else:
                print(f"Found a single section of length: {section_lengths[0]}.", end=" ")
            if tot_single_errors > 0:
                print(f"Encountered {tot_single_errors} ignoreable single errors.", end=" ")

        # calculate the global pulse_runtime (the number of laser pulses at each photon arrival since the beginning of the file)
        # each index in pulse_runtime represents a photon arrival into the TDC
        pulse_runtime = (
            byte_data[photon_idxs + 1] * 256**2
            + byte_data[photon_idxs + 2] * 256
            + byte_data[photon_idxs + 3]
        ).astype(np.int64)

        time_stamps = np.diff(pulse_runtime)

        # find simple "inversions": the data with a missing byte
        # decrease in pulse_runtime on data j+1, yet the next pulse_runtime data (j+2) is higher than j.
        inv_idxs = np.where((time_stamps[:-1] < 0) & ((time_stamps[:-1] + time_stamps[1:]) > 0))[0]
        if (inv_idxs.size) != 0:
            if proc_options.get("is_verbose"):
                print(
                    f"Found {inv_idxs.size} instances of missing byte data, ad hoc fixing...",
                    end=" ",
                )
            temp = (time_stamps[inv_idxs] + time_stamps[inv_idxs + 1]) / 2
            time_stamps[inv_idxs] = np.floor(temp).astype(np.int64)
            time_stamps[inv_idxs + 1] = np.ceil(temp).astype(np.int64)
            pulse_runtime[inv_idxs + 1] = pulse_runtime[inv_idxs + 2] - time_stamps[inv_idxs + 1]

        # repairing drops in pulse_runtime (happens when number of laser pulses passes 'MAX_VAL')
        neg_time_stamp_idxs = np.where(time_stamps < 0)[0]
        for i in neg_time_stamp_idxs + 1:
            pulse_runtime[i:] += self.MAX_VAL

        # handling coarse and fine times (for gating)
        coarse = byte_data[photon_idxs + 4].astype(np.int16)
        fine = byte_data[photon_idxs + 5].astype(np.int16)

        # some fix due to an issue in FPGA
        coarse_mod64 = np.mod(coarse, 64)
        coarse2 = coarse_mod64 - np.mod(coarse_mod64, 4) + (coarse // 64)
        coarse = coarse_mod64

        # Duration calculation
        time_stamps = np.diff(pulse_runtime).astype(np.int32)
        duration_s = (time_stamps / self.laser_freq_hz).sum()

        if is_scan_continuous:  # relevant for static/circular data
            all_section_edges, skipped_duration = self._section_continuous_data(
                pulse_runtime, time_stamps, **proc_options
            )
        else:  # angular scan
            all_section_edges = None
            skipped_duration = 0

        return TDCPhotonFileData(
            # TODO: using properties, I can ignore the fact that some attributes are 'raw' and some are 'general' and get less code
            idx,
            GeneralFileData(
                # general
                laser_freq_hz=self.laser_freq_hz,
                section_runtime_edges=section_runtime_edges,
                size_estimate_mb=max(section_lengths) / 1e6,
                duration_s=duration_s,
                skipped_duration=skipped_duration,
                # continuous scan
                all_section_edges=all_section_edges,
            ),
            RawFileData(
                idx=idx,
                dump_path=self.dump_path,
                coarse=coarse,
                coarse2=coarse2,
                fine=fine,
                pulse_runtime=pulse_runtime,
                delay_time=np.full(
                    pulse_runtime.shape, self.detector_gate_ns.lower, dtype=np.float16
                ),
                **proc_options,
            ),
            self.dump_path,
        )

    def _section_continuous_data(
        self,
        pulse_runtime,
        time_stamps,
        max_outlier_prob=1e-5,
        n_splits_requested=10,
        min_time_frac=0.5,
        **kwargs,
    ):
        """Find outliers and create sections seperated by them. Short sections are discarded"""

        mu = np.median(time_stamps) / np.log(2)
        duration_estimate_s = mu * len(pulse_runtime) / self.laser_freq_hz

        # find additional outliers (improbably large time_stamps) and break into
        # additional sections if they exist.
        # for exponential distribution MEDIAN and MAD are the same, but for
        # biexponential MAD seems more sensitive
        mu = max(np.median(time_stamps), np.abs(time_stamps - time_stamps.mean()).mean()) / np.log(
            2
        )
        max_time_stamp = scipy.stats.expon.ppf(1 - max_outlier_prob / len(time_stamps), scale=mu)
        sec_edges = (time_stamps > max_time_stamp).nonzero()[0].tolist()
        if (n_outliers := len(sec_edges)) > 0 and kwargs.get("is_verbose"):
            print(f"found {n_outliers} outliers.", end=" ")
        sec_edges = [0] + sec_edges + [len(time_stamps)]
        all_section_edges = np.array([sec_edges[:-1], sec_edges[1:]]).T

        # Filter short sections
        # TODO: duration limitation is unclear
        split_duration = duration_estimate_s / n_splits_requested
        skipped_duration = 0
        all_section_edges_valid = []
        # Ignore short sections (default is below half the run_duration)
        for se_idx, (se_start, se_end) in enumerate(all_section_edges):
            section_time = (pulse_runtime[se_end] - pulse_runtime[se_start]) / self.laser_freq_hz
            if section_time < min_time_frac * split_duration:
                if kwargs.get("is_verbose"):
                    print(
                        f"Skipping section {se_idx} - too short ({section_time * 1e3:.2f} ms).",
                        end=" ",
                    )
                skipped_duration += section_time
            else:  # use section
                all_section_edges_valid.append((se_start, se_end))

        return np.array(all_section_edges_valid), skipped_duration

    def _find_all_section_edges(
        self,
        byte_data: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], int]:
        """Doc."""

        section_edges = []
        data_end = False
        last_edge_stop = 0
        total_single_errors = 0
        while not data_end:
            remaining_byte_data = byte_data[last_edge_stop:]
            new_edge_start, new_edge_stop, data_end, n_single_errors = self._find_section_edges(
                remaining_byte_data,
            )
            new_edge_start += last_edge_stop
            new_edge_stop += last_edge_stop
            section_edges.append((new_edge_start, new_edge_stop))
            last_edge_stop = new_edge_stop
            total_single_errors += n_single_errors

        return section_edges, total_single_errors

    def _find_section_edges(self, byte_data: np.ndarray):  # NOQA C901
        """Doc."""

        # find index of first complete photon (where 248 and 254 bytes are spaced exatly (GROUP_LEN -1) bytes apart)
        if (edge_start := self._first_full_photon_idx(byte_data)) is None:
            raise RuntimeError("No byte data found! Check detector and FPGA.")

        # slice byte_data where assumed to be 248 and 254 (photon brackets)
        data_presumed_248 = byte_data[edge_start :: self.GROUP_LEN]
        data_presumed_254 = byte_data[(edge_start + self.GROUP_LEN - 1) :: self.GROUP_LEN]

        # find indices where this assumption breaks
        missed_248_idxs = np.where(data_presumed_248 != 248)[0]
        tot_248s_missed = missed_248_idxs.size
        missed_254_idxs = np.where(data_presumed_254 != 254)[0]
        tot_254s_missed = missed_254_idxs.size

        data_end = False
        n_single_errors = 0
        count = 0
        for count, missed_248_idx in enumerate(missed_248_idxs):

            # hold byte_data idx of current missing photon starting bracket
            data_idx_of_missed_248 = edge_start + missed_248_idx * self.GROUP_LEN

            # condition for ignoring single photon error (just test the most significant bytes of the pulse_runtime are close)
            curr_runtime_msb = int(byte_data[data_idx_of_missed_248 + 1])
            prev_runtime_msb = int(byte_data[data_idx_of_missed_248 - (self.GROUP_LEN - 1)])
            ignore_single_error_cond = abs(curr_runtime_msb - prev_runtime_msb) < 3

            # check that this is not a case of a singular mistake in 248 byte: happens very rarely but does happen
            if tot_254s_missed < count + 1:
                # no missing ending brackets, at least one missing starting bracket
                if missed_248_idx == (len(data_presumed_248) - 1):
                    # problem in the last photon in the file
                    # Found single error in last photon of file, ignoring and finishing...
                    edge_stop = data_idx_of_missed_248
                    break

                elif (tot_248s_missed == count + 1) or (
                    np.diff(missed_248_idxs[count : (count + 2)]) > 1
                ):
                    if ignore_single_error_cond:
                        # f"Found single photon error (byte_data[{data_idx_of_missed_248}]), ignoring and continuing..."
                        n_single_errors += 1
                    else:
                        raise RuntimeError("Check byte data for strange section edges!")

                else:
                    raise RuntimeError(
                        "Bizarre problem in byte data: 248 byte out of registry while 254 is in registry!"
                    )

            else:  # (tot_254s_missed >= count + 1)
                # if (missed_248_idxs[count] == missed_254_idxs[count]), # likely a real section
                if np.isin(missed_248_idx, missed_254_idxs):
                    # Found a section, continuing...
                    edge_stop = data_idx_of_missed_248
                    if byte_data[edge_stop - 1] != 254:
                        edge_stop = edge_stop - self.GROUP_LEN
                    break

                elif missed_248_idxs[count] == (
                    missed_254_idxs[count] + 1
                ):  # np.isin(missed_248_idx, (missed_254_idxs[count]+1)): # likely a real section ? why np.isin?
                    # Found a section, continuing...
                    edge_stop = data_idx_of_missed_248
                    if byte_data[edge_stop - 1] != 254:
                        edge_stop = edge_stop - self.GROUP_LEN
                    break

                elif missed_248_idx < missed_254_idxs[count]:  # likely a singular error on 248 byte
                    if ignore_single_error_cond:
                        # f"Found single photon error (byte_data[{data_idx_of_missed_248}]), ignoring and continuing..."
                        n_single_errors += 1
                        continue
                    else:
                        raise RuntimeError("Check byte data for strange section edges!")

                else:  # likely a signular mistake on 254 byte
                    if ignore_single_error_cond:
                        # f"Found single photon error (byte_data[{data_idx_of_missed_248}]), ignoring and continuing..."
                        n_single_errors += 1
                        continue
                    else:
                        raise RuntimeError("Check byte data for strange section edges!")

        # did reach the end of the loop without breaking?
        were_no_breaks = not tot_248s_missed or (count == (missed_248_idxs.size - 1))
        # case there were no problems
        if were_no_breaks:
            edge_stop = edge_start + (data_presumed_254.size - 1) * self.GROUP_LEN
            data_end = True

        return edge_start, edge_stop, data_end, n_single_errors

    def _first_full_photon_idx(self, byte_data: np.ndarray) -> int:
        """
        Return the starting index of the first intact photon - a sequence of 'GROUP_LEN'
        bytes starting with 248 and ending with 254. If no intact photons are found, returns 'None'
        """

        for idx in infinite_range():
            try:
                if (byte_data[idx] == 248) and (byte_data[idx + (self.GROUP_LEN - 1)] == 254):
                    return idx
            except IndexError:
                break
        return None

    def _process_image_scan_plane_data(
        self,
        idx,
        full_data,
        should_dump=True,
        **kwargs,
    ) -> TDCPhotonFileData:
        """
        Processes a single plane image sFCS data ('full_data').
        Returns the processed results as a 'TDCPhotonData' object.
        '"""

        try:
            p = self._convert_fpga_data_to_photons(
                idx,
                byte_data=full_data["byte_data"],
                **kwargs,
            )
        except RuntimeError as exc:
            print(f"{exc} Skipping plane.")
            return None

        return p

    def _process_circular_scan_data_file(
        self, idx, full_data, should_dump=True, **proc_options
    ) -> TDCPhotonFileData:
        """
        Processes a single circular sFCS data file ('full_data').
        Returns the processed results as a 'TDCPhotonData' object.
        '"""
        # TODO: can this method be moved to the appropriate Mixin class?

        p = self._convert_fpga_data_to_photons(
            idx,
            is_scan_continuous=True,
            **proc_options,
        )
        if should_dump:  # False when multiprocessing
            p.raw.dump()

        scan_settings = full_data["scan_settings"]
        ao_sampling_freq_hz = int(scan_settings["ao_sampling_freq_hz"])

        if proc_options.get("is_verbose"):
            print("Converting circular scan to image...", end=" ")
        cnt, _ = self._sum_scan_circles(
            p.raw.pulse_runtime,
            self.laser_freq_hz,
            ao_sampling_freq_hz,
            scan_settings["circle_freq_hz"],
        )

        p.general.image = cnt

        # get background correlation
        cnt = cnt.astype(np.float64)
        c, lags = xcorr(cnt, cnt)
        c = c / cnt.mean() ** 2 - 1
        c[0] -= 1 / cnt.mean()  # subtracting shot noise, small stuff really
        p.general.bg_line_corr = [
            {
                "lag": lags * 1e3 / ao_sampling_freq_hz,  # in ms
                "corrfunc": c,
            }
        ]

        return p

    def _process_angular_scan_data_file(  # NOQA C901
        self,
        idx,
        full_data,
        roi_selection="auto",
        should_dump=True,
        **kwargs,
    ) -> TDCPhotonFileData:
        """
        Processes a single angular sFCS data file ('full_data').
        Returns the processed results as a 'TDCPhotonData' object.
        '"""
        # TODO: can this method be moved to the appropriate Mixin class?

        try:
            p = self._convert_fpga_data_to_photons(
                idx,
                **kwargs,
            )
        except RuntimeError as exc:
            # TODO: this should just be an error, and handled as such up the call chain
            print(f"{exc} Skipping file.")
            return None

        scan_settings = full_data["scan_settings"]
        # TODO: why is the .round() needed? (probably for old data - should move to file_utilities)
        linear_part = scan_settings["linear_part"].round().astype(np.uint16)
        ao_sampling_freq_hz = int(scan_settings["ao_sampling_freq_hz"])
        p.general.samples_per_line = int(scan_settings["samples_per_line"])
        p.general.n_lines = int(scan_settings["n_lines"])

        if kwargs.get("is_verbose"):
            print("Converting angular scan to image...", end=" ")

        pulse_runtime = np.empty(p.raw.pulse_runtime.shape, dtype=np.int64)
        img = np.zeros((p.general.n_lines + 1, p.general.samples_per_line), dtype=np.uint16)
        sample_runtime = np.empty(pulse_runtime.shape, dtype=np.int64)
        pixel_num = np.empty(pulse_runtime.shape, dtype=np.int64)
        line_num = np.empty(pulse_runtime.shape, dtype=np.int16)
        for sec_idx, (start_idx, end_idx) in enumerate(p.general.section_runtime_edges):
            sec_pulse_runtime = p.raw.pulse_runtime[start_idx:end_idx]
            (
                sec_img,
                sec_sample_runtime,
                sec_pixel_num,
                sec_line_num,
                pix_shift,
            ) = self.convert_angular_scan_to_image(
                sec_pulse_runtime,
                self.laser_freq_hz,
                ao_sampling_freq_hz,
                p.general.samples_per_line,
                p.general.n_lines,
                **kwargs,
            )

            pulse_runtime[start_idx:end_idx] = sec_pulse_runtime
            img += sec_img
            sample_runtime[start_idx:end_idx] = sec_sample_runtime
            pixel_num[start_idx:end_idx] = sec_pixel_num
            line_num[start_idx:end_idx] = sec_line_num

        # work with entire scan image, then return to ROI lines start/stops after
        prt_shift = pix_shift * round(self.laser_freq_hz / ao_sampling_freq_hz)
        (
            whole_line_starts_prt,
            whole_line_stops_prt,
            whole_line_start_labels,
            whole_line_stop_labels,
            _,
        ) = self._get_line_markers_and_roi(
            np.full(img.shape, True),
            pulse_runtime + prt_shift,
            self.laser_freq_hz,
            ao_sampling_freq_hz,
        )

        # keeping the single-scan, in ROI, start/stop runtimes
        single_scan_edges = list(
            zip(
                whole_line_starts_prt[whole_line_start_labels == whole_line_start_labels[0]],
                whole_line_stops_prt[whole_line_stop_labels == whole_line_stop_labels[-1]],
            )
        )

        if roi_selection == "auto":
            if kwargs.get("is_verbose"):
                print("Thresholding and smoothing...", end=" ")
            try:
                img_bw = self._threshold_and_smooth(img.copy())
            except ValueError:
                raise RuntimeError("Automatic ROI selection: Thresholding failed")
        elif roi_selection == "all":
            img_bw = np.full(img.shape, True)
        else:
            raise ValueError(
                f"roi_selection='{roi_selection}' is not supported. Only 'auto' is, at the moment."
            )

        # cut edges
        bw_temp = np.full(img_bw.shape, False, dtype=bool)
        bw_temp[:, linear_part] = img_bw[:, linear_part]
        img_bw = bw_temp

        # discard short rows, then fill long rows
        m2 = np.sum(img_bw, axis=1)
        img_bw[m2 < 0.5 * m2.max(), :] = False
        true_vals = np.argwhere(img_bw)
        for row_idx in np.unique(img_bw.nonzero()[0]):
            true_idxs = true_vals[true_vals[:, 0] == row_idx][:, 1]
            img_bw[row_idx, true_idxs.min() : true_idxs.max() + 1] = True

        # create a copy and reverse 2nd rows again to get a "data mask"
        bw = img_bw.copy()
        bw[1::2, :] = np.flip(bw[1::2, :], 1)

        if kwargs.get("is_verbose"):
            print("Building ROI...", end=" ")

        # TODO: this is an experimantal feature - it should be fixed so that only the "best" (least brightest spots) of each file are used
        if kwargs.get("should_alleviate_bright_pixels"):
            if kwargs.get("is_verbose"):
                print("Getting rid of rows with bright spots on the single-scan level...", end=" ")

            n_lines_removed = 0  # TESTESTEST
            n_total_lines = p.general.n_lines * len(single_scan_edges)  # TESTESTEST

            line_list = img_bw.sum(axis=1).nonzero()[0]  # initialize once

            # calculate the pulse runtime shift from the pixel shift
            prt_shift = pix_shift * round(self.laser_freq_hz / ao_sampling_freq_hz)
            # initialize the valid indices to be all False, then set the photons in good rows (which are in ful scans) to True
            total_valid_photon_idxs = np.full(len(line_num), False, dtype=bool)
            scan_line_starts_prt_list = []
            scan_line_stops_prt_list = []
            scan_line_start_labels_list = []
            scan_line_stop_labels_list = []
            for scan_idx, (scan_start_prt, scan_stop_prt) in enumerate(single_scan_edges):
                # get the indices of in-scan photons
                in_scan_idxs = Limits(scan_start_prt, scan_stop_prt).valid_indices(pulse_runtime)
                scan_prt = pulse_runtime[in_scan_idxs]

                # Handle faulty scans
                if scan_prt.size < 100:
                    raise RuntimeError(
                        f"One of the single scans (index {scan_idx}) is faulty. Ignoring whole file."
                    )

                # prepare scan image to discriminate bad rows (with bright pixels)
                scan_img, _, scan_pixel_num, scan_line_num, _ = self.convert_angular_scan_to_image(
                    scan_prt + prt_shift,
                    self.laser_freq_hz,
                    ao_sampling_freq_hz,
                    p.general.samples_per_line,
                    p.general.n_lines,
                )
                bright_pixels_img_bw = self.get_bright_pixels(scan_img, img_bw, **kwargs)
                scan_bad_row_labels = np.unique(bright_pixels_img_bw.nonzero()[0])

                # get indices of all valid photons (full rows withought bright spots) in the scan
                total_valid_photon_idxs[in_scan_idxs] = np.in1d(
                    scan_line_num, scan_bad_row_labels, invert=True
                )

                # also get the valid line start/stop and labels indices and ignore the bad lines
                valid_scan_line_idxs = np.in1d(
                    line_list, scan_bad_row_labels, assume_unique=True, invert=True
                )

                #                # Check out line discrimination in each scan image # TESTESTEST
                #                with Plotter(should_force_aspect=True) as ax: # TESTESTEST
                #                    ax.imshow(scan_img * scan_img_bw, vmin=0, vmax=128, interpolation="none") # TESTESTEST
                #                    for bad_row_idx in set([label for label in scan_bad_row_labels if label >= 0]): # TESTESTEST
                #                        ax.axhline(y=bad_row_idx, color="r", lw=1, ls="--") # TESTESTEST

                # get in-scan line starts/stops/labels (necessary in this case since the ROI "changes" each scan - lines starts/stops change)
                (
                    scan_line_starts_prt,
                    scan_line_stops_prt,
                    scan_line_start_labels,
                    scan_line_stop_labels,
                    _,
                ) = self._get_line_markers_and_roi(
                    img_bw, scan_prt + prt_shift, self.laser_freq_hz, ao_sampling_freq_hz
                )

                try:
                    scan_line_starts_prt = scan_line_starts_prt[valid_scan_line_idxs]
                    scan_line_stops_prt = scan_line_stops_prt[valid_scan_line_idxs]
                    scan_line_start_labels = scan_line_start_labels[valid_scan_line_idxs]
                    scan_line_stop_labels = scan_line_stop_labels[valid_scan_line_idxs]
                except IndexError:
                    # scan_line_starts_prt is empty
                    ...
                else:
                    # add to total list of start/stops
                    scan_line_starts_prt_list.append(scan_line_starts_prt)
                    scan_line_stops_prt_list.append(scan_line_stops_prt)
                    scan_line_start_labels_list.append(scan_line_start_labels)
                    scan_line_stop_labels_list.append(scan_line_stop_labels)

                n_lines_removed += len(scan_bad_row_labels)

            print(f"removed: {n_lines_removed:.0f}/{n_total_lines} lines.")

            # Now, to actually discriminate the bad rows using the accumulated indices
            pulse_runtime = pulse_runtime[total_valid_photon_idxs]
            coarse = p.raw.coarse[total_valid_photon_idxs]
            coarse2 = p.raw.coarse2[total_valid_photon_idxs]
            fine = p.raw.fine[total_valid_photon_idxs]

            # create a new scan image as well
            (p.general.image, _, pixel_num, line_num, _,) = self.convert_angular_scan_to_image(
                pulse_runtime + prt_shift,
                self.laser_freq_hz,
                ao_sampling_freq_hz,
                p.general.samples_per_line,
                p.general.n_lines,
            )

            # Now cut crop the filtered lines using the ROI
            # TODO: get the ROI separately - all I need is the 'img_bw' anyway...
            *_, roi = self._get_line_markers_and_roi(
                img_bw, pulse_runtime + prt_shift, self.laser_freq_hz, ao_sampling_freq_hz
            )

            # remove bad lines from line starts/stops/labels (sorting them by pulse runtime first so they match the scan line order)
            # unite all line starts/stops/labels
            line_starts_prt = np.hstack(tuple(scan_line_starts_prt_list))
            line_stops_prt = np.hstack(tuple(scan_line_stops_prt_list))
            line_start_labels = np.hstack(tuple(scan_line_start_labels_list))
            line_stop_labels = np.hstack(tuple(scan_line_stop_labels_list))

        else:
            try:
                (
                    line_starts_prt,
                    line_stops_prt,
                    line_start_labels,
                    line_stop_labels,
                    roi,
                ) = self._get_line_markers_and_roi(
                    img_bw, pulse_runtime + prt_shift, self.laser_freq_hz, ao_sampling_freq_hz
                )
            except IndexError:
                # TODO: this should just be an error, and handled as such up the call chain
                if kwargs.get("is_verbose"):
                    print("ROI is empty (need to figure out the cause). Skipping file.\n")
                return None

            coarse = p.raw.coarse
            coarse2 = p.raw.coarse2
            fine = p.raw.fine
            p.general.image = img

        # Inserting the line start/stop markers into the arrays
        pulse_runtime = np.hstack((line_starts_prt, line_stops_prt, pulse_runtime))
        sorted_idxs = np.argsort(pulse_runtime)
        new_pulse_runtime = pulse_runtime[sorted_idxs]
        # Set an 'out-of-bounds' line number to all photons outside mask (these are later ignored by '_prepare_correlator_input')
        line_num[~bw[line_num, pixel_num]] = -3000  # TODO: give this a constant

        new_line_num = np.hstack(
            (
                line_start_labels,
                line_stop_labels,
                line_num,
            )
        )[sorted_idxs]
        line_starts_nans = np.full(line_starts_prt.size, NAN_PLACEBO, dtype=np.int16)
        line_stops_nans = np.full(line_stops_prt.size, NAN_PLACEBO, dtype=np.int16)
        new_coarse = np.hstack((line_starts_nans, line_stops_nans, coarse))[sorted_idxs]
        new_coarse2 = np.hstack((line_starts_nans, line_stops_nans, coarse2))[sorted_idxs]
        new_fine = np.hstack((line_starts_nans, line_stops_nans, fine))[sorted_idxs]

        # initialize delay times with lower detector gate (nans at line edges) - filled-in during TDC calibration
        delay_time = np.full(new_pulse_runtime.shape, self.detector_gate_ns.lower, dtype=np.float16)
        line_edge_idxs = new_fine == NAN_PLACEBO
        delay_time[line_edge_idxs] = np.nan

        # replace the raw data after angular scan changes made
        p.import_raw(
            np.vstack(
                (new_coarse, new_coarse2, new_fine, new_pulse_runtime, delay_time, new_line_num)
            ),
            **kwargs,
        )
        if (
            should_dump
        ):  # False only if multiprocessing or if manually set to avoid re-dumping many existing identical files
            if kwargs.get("is_verbose"):
                print("Dumping raw data file to disk (if needed)...", end=" ")
            p.raw.dump()

        p.general.image_bw_mask = img_bw
        p.general.roi = roi
        p.general.pix_shift = pix_shift
        p.general.single_scan_edges = single_scan_edges

        # get background correlation
        if kwargs.get("is_verbose"):
            print("Getting background correlation...", end=" ")
        p.general.valid_lines = np.unique(new_line_num[new_line_num >= 0])
        p.general.bg_line_corr = self._bg_line_correlations(
            p.general.image,
            img_bw,
            p.general.valid_lines,
            ao_sampling_freq_hz,
        )

        return p

    def calibrate_tdc(  # NOQA C901
        self,
        data: list[TDCPhotonFileData],
        scan_type,
        tdc_chain_length=128,
        pick_valid_bins_according_to=None,
        sync_coarse_time_to=None,
        pick_calib_bins_according_to=None,
        external_calib=None,
        calib_range_ns: Union[Limits, tuple] = Limits(40, 80),
        n_zeros_for_fine_bounds=10,
        time_bins_for_hist_ns=0.1,
        **kwargs,
    ) -> TDCCalibration:
        """Doc."""
        # TODO: move to 'TDCPhotonMeasurementData' class

        if kwargs.get("is_verbose"):
            print("Uniting coarse/fine data... ", end="")
        coarse_mmap, fine_mmap = self._unite_coarse_fine_data(data, scan_type)

        if kwargs.get("is_verbose"):
            print("Binning coarse data... ", end="")
        h_all = chunked_bincount(coarse_mmap).astype(np.uint32)
        x_all = np.arange(coarse_mmap.max + 1, dtype=np.uint8)

        if pick_valid_bins_according_to is None:
            h_all = h_all[coarse_mmap.min :]
            x_all = np.arange(coarse_mmap.min, coarse_mmap.max + 1, dtype=np.uint8)
            coarse_bins = x_all
            h = h_all
        elif isinstance(pick_valid_bins_according_to, np.ndarray):
            coarse_bins = pick_valid_bins_according_to
            h = h_all[coarse_bins]
        elif isinstance(pick_valid_bins_according_to, TDCCalibration):
            coarse_bins = pick_valid_bins_according_to.coarse_bins
            h = h_all[coarse_bins]
        else:
            raise TypeError(
                f"Unknown type '{type(pick_valid_bins_according_to)}' for picking valid coarse_bins!"
            )

        # rearranging the coarse_bins
        if sync_coarse_time_to is None:
            if (lower_detector_gate_ns := self.detector_gate_ns.lower) > 0:  # detector gating
                max_j = np.argmax(h) - round((lower_detector_gate_ns * 1e-9) * self.fpga_freq_hz)
            else:
                max_j = np.argmax(h)
        elif isinstance(sync_coarse_time_to, int):
            max_j = sync_coarse_time_to
        elif isinstance(sync_coarse_time_to, TDCCalibration):
            max_j = sync_coarse_time_to.max_j
        else:
            raise TypeError(
                f"sync_coarse_time_to must be either a number or a 'TDCCalibration' object! Type '{type(sync_coarse_time_to).name}' was supplied."
            )

        j_shift = np.roll(np.arange(len(h)), -max_j + 2)

        if pick_calib_bins_according_to is None:
            # pick data at more than 'calib_time_ns' delay from peak maximum
            calib_range_bins = (
                ((Limits(calib_range_ns) & self.detector_gate_ns) + self.detector_gate_ns.lower)
                * 1e-9
                * self.fpga_freq_hz
            )
            j = calib_range_bins.valid_indices(coarse_bins)
            if not j.any():
                raise ValueError(f"Gate width is too narrow for calib_time_ns={calib_range_ns}!")
            j_calib = j_shift[j]
            coarse_calib_bins = coarse_bins[j_calib]
        elif isinstance(pick_calib_bins_according_to, (list, np.ndarray)):
            coarse_calib_bins = pick_calib_bins_according_to
        elif isinstance(pick_calib_bins_according_to, TDCCalibration):
            coarse_calib_bins = pick_calib_bins_according_to.coarse_calib_bins
        else:
            raise TypeError(
                f"Unknown type '{type(pick_calib_bins_according_to)}' for picking calibration coarse_bins!"
            )

        # Don't use 'empty' coarse_bins for calibration
        valid_cal_bins = np.nonzero(h_all > (np.median(h_all) / 100))[0]
        coarse_calib_bins = np.array(
            [bin_idx for bin_idx in coarse_calib_bins if bin_idx in valid_cal_bins]
        )

        if isinstance(external_calib, TDCCalibration):
            if kwargs.get("is_verbose"):
                print("Using external calibration... ", end="")
            max_j = external_calib.max_j
            coarse_calib_bins = external_calib.coarse_calib_bins
            fine_bins = external_calib.fine_bins
            t_calib = external_calib.t_calib
            h_tdc_calib = external_calib.h_tdc_calib
            t_weight = external_calib.t_weight
            l_quarter_tdc = external_calib.l_quarter_tdc
            r_quarter_tdc = external_calib.r_quarter_tdc
            h = external_calib.h
            delay_times = external_calib.delay_times

        else:
            if kwargs.get("is_verbose"):
                print("Binning fine data... ", end="")
            fine_calib = fine_mmap.read()[np.isin(coarse_mmap.read(), coarse_calib_bins)]
            h_tdc_calib = np.bincount(fine_calib, minlength=tdc_chain_length).astype(np.uint32)

            # find effective range of TDC by, say, finding 10 zeros in a row
            cumusum_h_tdc_calib = np.cumsum(h_tdc_calib)
            diff_cumusum_h_tdc_calib = (
                cumusum_h_tdc_calib[(n_zeros_for_fine_bounds - 1) :]
                - cumusum_h_tdc_calib[: (-n_zeros_for_fine_bounds + 1)]
            )
            zeros_tdc = np.where(diff_cumusum_h_tdc_calib == 0)[0]
            # find middle of TDC
            mid_tdc = np.mean(fine_calib)

            # left TDC edge: zeros stretch closest to mid_tdc
            left_tdc = max(zeros_tdc[zeros_tdc < mid_tdc]) + n_zeros_for_fine_bounds - 1
            right_tdc = min(zeros_tdc[zeros_tdc > mid_tdc]) + 1

            l_quarter_tdc = round(left_tdc + (right_tdc - left_tdc) / 4)
            r_quarter_tdc = round(right_tdc - (right_tdc - left_tdc) / 4)

            # zero those out of TDC (happened at least once, for old detector data from 10/05/2018)
            if sum(h_tdc_calib[:left_tdc]) or sum(h_tdc_calib[right_tdc:]):
                h_tdc_calib[:left_tdc] = 0
                h_tdc_calib[right_tdc:] = 0

            t_calib = (
                (1 - np.cumsum(h_tdc_calib) / np.sum(h_tdc_calib)) / self.fpga_freq_hz * 1e9
            )  # invert and to ns

            coarse_len = coarse_bins.size

            t_weight = np.tile(h_tdc_calib / np.mean(h_tdc_calib), coarse_len)
            coarse_times = (
                np.tile(np.arange(coarse_len), [t_calib.size, 1]) / self.fpga_freq_hz * 1e9
            )
            delay_times = np.tile(t_calib, coarse_len) + coarse_times.flatten("F")
            j_sorted = np.argsort(delay_times)
            delay_times = delay_times[j_sorted]

            t_weight = t_weight[j_sorted]

        # assign time delays to all photons
        if kwargs.get("is_verbose"):
            print("Assigning delay times... ", end="")
        total_laser_pulses = 0
        first_coarse_bin, *_, last_coarse_bin = coarse_bins
        delay_time_list = []
        for p in data:
            _delay_time = np.full(p.raw.coarse.shape, np.nan, dtype=np.float64)
            crs = np.minimum(p.raw.coarse, last_coarse_bin) - coarse_bins[max_j - 1]
            crs[crs < 0] = crs[crs < 0] + last_coarse_bin - first_coarse_bin + 1

            delta_coarse = p.raw.coarse2 - p.raw.coarse
            delta_coarse[delta_coarse == -3] = 1  # 2bit limitation

            # in the TDC midrange use "coarse" counter
            in_mid_tdc = (p.raw.fine >= l_quarter_tdc) & (p.raw.fine <= r_quarter_tdc)
            delta_coarse[in_mid_tdc] = 0

            # on the right of TDC use "coarse2" counter (no change in delta)
            # on the left of TDC use "coarse2" counter decremented by 1
            on_left_tdc = p.raw.fine < l_quarter_tdc
            delta_coarse[on_left_tdc] = delta_coarse[on_left_tdc] - 1

            photon_idxs = p.raw.fine != NAN_PLACEBO  # self.NAN_PLACEBO are starts/ends of lines
            _delay_time[photon_idxs] = (
                t_calib[p.raw.fine[photon_idxs]]
                + (crs[photon_idxs] + delta_coarse[photon_idxs]) / self.fpga_freq_hz * 1e9
            )
            p.raw.delay_time = _delay_time
            total_laser_pulses += p.raw.pulse_runtime[-1]

            delay_time_list.append(_delay_time[photon_idxs])

        delay_time = np.hstack(delay_time_list)

        fine_bins = np.arange(
            -time_bins_for_hist_ns / 2,
            np.max(delay_time) + time_bins_for_hist_ns,
            time_bins_for_hist_ns,
            dtype=np.float16,
        )

        t_hist = (fine_bins[:-1] + fine_bins[1:]) / 2
        k = np.digitize(delay_times, fine_bins)

        hist_weight = np.empty(t_hist.shape, dtype=np.float64)
        for i in range(len(hist_weight)):
            j = k == (i + 1)
            hist_weight[i] = np.sum(t_weight[j])

        all_hist = np.histogram(delay_time, bins=fine_bins)[0].astype(np.uint32)

        all_hist_norm = np.full(all_hist.shape, np.nan, dtype=np.float64)
        error_all_hist_norm = np.full(all_hist.shape, np.nan, dtype=np.float64)
        nonzero = hist_weight > 0
        all_hist_norm[nonzero] = (
            all_hist[nonzero] / hist_weight[nonzero] * np.mean(hist_weight) / total_laser_pulses
        )
        error_all_hist_norm[nonzero] = (
            np.sqrt(all_hist[nonzero])
            / hist_weight[nonzero]
            * np.mean(hist_weight)
            / total_laser_pulses
        )

        # delete temp files
        coarse_mmap.delete()
        fine_mmap.delete()

        if kwargs.get("is_verbose"):
            print("Done.")

        return TDCCalibration(
            x_all=x_all,
            h_all=h_all,
            coarse_bins=coarse_bins,
            h=h,
            max_j=max_j,
            coarse_calib_bins=coarse_calib_bins,
            fine_bins=fine_bins,
            l_quarter_tdc=l_quarter_tdc,
            r_quarter_tdc=r_quarter_tdc,
            h_tdc_calib=h_tdc_calib,
            t_calib=t_calib,
            t_hist=t_hist,
            hist_weight=hist_weight,
            all_hist=all_hist,
            all_hist_norm=all_hist_norm,
            error_all_hist_norm=error_all_hist_norm,
            delay_times=delay_times,
            t_weight=t_weight,
            total_laser_pulses=total_laser_pulses,
        )

    def _unite_coarse_fine_data(self, data, scan_type: str):
        """Doc."""

        # keep pulse_runtime elements of each file for array size allocation
        if scan_type == "angular":
            # remove line starts/ends from angular scan data sizes
            n_elem = np.cumsum([0] + [p.raw.fine[p.raw.fine > NAN_PLACEBO].size for p in data])
        else:
            n_elem = np.cumsum([0] + [p.raw.fine.size for p in data])

        # unite coarse and fine times from all files
        coarse = np.empty(shape=(n_elem[-1],), dtype=np.int16)
        fine = np.empty(shape=(n_elem[-1],), dtype=np.int16)
        for i, p in enumerate(data):
            if scan_type == "angular":
                # remove line starts/ends from angular scan data
                photon_idxs = p.raw.fine > NAN_PLACEBO
            else:
                # otherwise, treat all elements as photons
                photon_idxs = slice(None)
            coarse[n_elem[i] : n_elem[i + 1]] = p.raw.coarse[photon_idxs]
            fine[n_elem[i] : n_elem[i + 1]] = p.raw.fine[photon_idxs]
        coarse_mmap = MemMapping(coarse, "coarse.npy")
        fine_mmap = MemMapping(fine, "fine.npy")

        # return the MemMapping instances
        return coarse_mmap, fine_mmap


@dataclass
class ImageStackData:
    """
    Holds a stack of images along with some relevant scan data,
    built from CI (counter input) NIDAQmx data.
    """

    image_stack: np.ndarray
    effective_idxs: InitVar[np.ndarray]
    pxls_per_line: InitVar[int]
    line_ticks_v: np.ndarray
    row_ticks_v: np.ndarray
    plane_ticks_v: np.ndarray
    n_planes: int
    plane_orientation: str
    dim_order: Tuple[int, int, int]
    image_stack_forward: np.ndarray = None
    norm_stack_forward: np.ndarray = None
    image_stack_backward: np.ndarray = None
    norm_stack_backward: np.ndarray = None

    def __post_init__(self, effective_idxs, pxls_per_line):
        self.effective_binned_size = np.unique(effective_idxs).size
        turn_idx = self.image_stack.shape[1] // 2
        self.image_stack_forward, self.norm_stack_forward = self._rebin_and_normalize_image_stack(
            self.image_stack[:, :turn_idx, :],
            effective_idxs[:turn_idx],
            pxls_per_line,
        )
        self.image_stack_backward, self.norm_stack_backward = self._rebin_and_normalize_image_stack(
            self.image_stack[:, -1 : (turn_idx - 1) : -1, :],
            effective_idxs[-1 : (turn_idx - 1) : -1],
            pxls_per_line,
        )

    def _rebin_and_normalize_image_stack(self, image_stack, eff_idxs, pxls_per_line):
        """Doc."""

        n_lines, _, n_planes = image_stack.shape
        binned_image_stack = np.empty((n_lines, pxls_per_line, n_planes), dtype=image_stack.dtype)
        #        binned_image_stack = np.empty((n_lines, pxls_per_line, n_planes), dtype=np.int32)
        binned_norm_stack = np.empty((n_lines, pxls_per_line, n_planes), dtype=image_stack.dtype)
        #        binned_norm_stack = np.empty((n_lines, pxls_per_line, n_planes), dtype=np.int32)

        for i in range(pxls_per_line):
            binned_image_stack[:, i, :] = image_stack[:, eff_idxs == i, :].sum(axis=1)
            binned_norm_stack[:, i, :] = image_stack[:, eff_idxs == i, :].shape[1]

        return binned_image_stack, binned_norm_stack

    def construct_plane_image(self, method: str, plane_idx: int = None, **kwargs) -> np.ndarray:
        """Doc."""

        if plane_idx is None:
            plane_idx = self.image_stack_forward.shape[2] // 2

        if method == "forward":
            img = self.image_stack_forward[:, :, plane_idx]
        elif method == "forward normalization":
            img = self.norm_stack_forward[:, :, plane_idx]
        elif method == "forward normalized":
            img = np.zeros_like(self.image_stack_forward[:, :, plane_idx], dtype=np.float64)
            norm = self.norm_stack_forward[:, :, plane_idx]
            valid_mask = norm > 0
            img[valid_mask] = (
                self.image_stack_forward[:, :, plane_idx][valid_mask] / norm[valid_mask]
            )
        elif method == "backward":
            img = self.image_stack_backward[:, :, plane_idx]
        elif method == "backward normalization":
            img = self.norm_stack_backward[:, :, plane_idx]
        elif method == "backward normalized":
            img = np.zeros_like(self.image_stack_backward[:, :, plane_idx], dtype=np.float64)
            norm = self.norm_stack_backward[:, :, plane_idx]
            valid_mask = norm > 0
            img[valid_mask] = (
                self.image_stack_backward[:, :, plane_idx][valid_mask] / norm[valid_mask]
            )
        elif method == "interlaced":
            p1_norm = (
                self.image_stack_forward[:, :, plane_idx] / self.norm_stack_forward[:, :, plane_idx]
            )
            p2_norm = (
                self.image_stack_backward[:, :, plane_idx]
                / self.norm_stack_backward[:, :, plane_idx]
            )
            n_lines = p1_norm.shape[0] + p2_norm.shape[0]
            img = np.zeros(p1_norm.shape)
            img[:n_lines:2, :] = p1_norm[:n_lines:2, :]
            img[1:n_lines:2, :] = p2_norm[1:n_lines:2, :]
        elif method == "averaged":
            p1 = self.image_stack_forward[:, :, plane_idx]
            p2 = self.image_stack_backward[:, :, plane_idx]
            norm1 = self.norm_stack_forward[:, :, plane_idx]
            norm2 = self.norm_stack_backward[:, :, plane_idx]
            img = (p1 + p2) / (norm1 + norm2)
        else:
            raise ValueError(f"'{method}' is not a valid counts image construction method.")

        return img
