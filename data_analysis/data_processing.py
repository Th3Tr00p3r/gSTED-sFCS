"""Data Processing."""

import multiprocessing
import sys
from collections import deque
from contextlib import suppress
from copy import copy
from dataclasses import InitVar, dataclass, field
from itertools import count as infinite_range
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple, Union

import numpy as np
import scipy
import skimage

from data_analysis.workers import N_CPU_CORES
from utilities.display import Plotter
from utilities.file_utilities import load_object, save_object
from utilities.fit_tools import (
    FitParams,
    curve_fit_lims,
    exponent_with_background_fit,
    gaussian_1d_fit,
)
from utilities.helper import (
    Gate,
    Limits,
    MemMapping,
    chunked_bincount,
    exclude_elements_by_indices_1d,
    nan_helper,
    unify_length,
    xcorr,
)

# line markers
NAN_PLACEBO = -100  # marks starts/ends of lines
LINE_END_ADDER = 1000
ZERO_LINE_START_ADDER = 2000
OUT_OF_SAMPLE_LINE = -3000
IGNORED_LINE = -4000


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
        **kwargs,
    ):
        """Converts angular scan pulse_runtime into an image."""
        # TODO: check out rolling initally made image instead of recursing

        # calculate the sample index at each photon arrival, since beginning of file/section
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
            # TODO: (low priority) if there are multiple sections in the raw data, perhpas the pixel shift can be calculated only for the first section and the same shift used for the rest.
            if kwargs.get("is_verbose"):
                print("Fixing data shift...", end=" ")
            pix_shift = self._get_data_shift(img, **kwargs)
            if kwargs.get("is_verbose"):
                print(f"shifting {pix_shift} pixels...", end=" ")
            img = np.roll(img, pix_shift)
        else:
            pix_shift = 0

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
        step = int(np.ceil(width / 1000))  # handling "slow" scans with many points per line

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
            min_pix_shift = -round(cnt.size / 2)
            max_pix_shift = min_pix_shift + cnt.size + 1
            pix_shift = get_best_pix_shift(cnt, min_pix_shift, max_pix_shift, step)

        return pix_shift

    def _threshold_and_smooth(
        self, image, otsu_classes=4, num_bins=256, disk_radius=2, median_factor=1.1, **kwargs
    ) -> np.ndarray:
        """
        Apply thresholding and smoothing operations to an input image.

        Parameters:
        -----------
        image : np.ndarray
            Input image data.

        otsu_classes : int, optional
            Number of classes for Otsu's thresholding method. Default is 4.

        num_bins : int, optional
            Number of bins for histogram computation. Default is 256.

        disk_radius : int, optional
            Radius of the disk-shaped structuring element. Default is 2.

        median_factor : float, optional
            Factor to adjust the threshold for median filtering. Default is 1.1.

        Returns:
        --------
        np.ndarray
            Processed binary image after thresholding and smoothing operations.
        """

        # Make a copy of the input image to avoid modifying the original data
        img = image.copy()

        # Global filtering of outliers (replace bright pixels with median of central area)
        _, width = img.shape
        central_area_median = np.median(img[:, int(width * 0.25) : int(width * 0.75)])
        median_threshold = central_area_median * median_factor
        img[img > median_threshold] = central_area_median

        # Minor local filtering of outliers, then thresholding using Otsu's method
        median_filtered_img = skimage.filters.median(img).astype(np.float32)
        otsu_thresholds = skimage.filters.threshold_multiotsu(
            median_filtered_img, otsu_classes, nbins=num_bins
        )
        digitized_img = np.digitize(img, bins=otsu_thresholds)

        # Determine plateau level and standard deviation of plateau pixels
        plateau_level = np.median(img[digitized_img == (otsu_classes - 1)])
        std_dev_plateau = scipy.stats.median_abs_deviation(img[digitized_img == (otsu_classes - 1)])
        deviation_from_plateau = img - plateau_level
        binary_image = deviation_from_plateau >= -std_dev_plateau

        # Fill holes in the binary image and perform morphological opening
        binary_image = scipy.ndimage.binary_fill_holes(binary_image)
        disk_structure = skimage.morphology.disk(radius=disk_radius)
        binary_image = skimage.morphology.opening(binary_image, footprint=disk_structure)

        return binary_image

    def _get_line_markers_and_roi(
        self,
        image_mask: np.ndarray,
        data_mask: np.ndarray,
        sample_runtime: np.ndarray,
        pixel_shift: int,
        laser_freq_hz: int,
        ao_sampling_freq_hz: int,
    ):
        """Doc."""

        line_starts = []
        line_stops = []
        line_start_labels = []
        line_stop_labels = []
        roi: Dict[str, deque] = {"row": deque([]), "col": deque([])}

        # get all line starts/stops and corresponding labels
        for row_idx in np.unique(image_mask.nonzero()[0]):
            #        for row_idx in masked_row_idxs:
            nonzero_data_row_idxs = data_mask[row_idx, :].nonzero()[0]
            # the the left and right edges for the row
            left_edge, right_edge = nonzero_data_row_idxs[0], nonzero_data_row_idxs[-1] + 1
            # get the flat index of the line start for row
            row_start_flat_idx = np.ravel_multi_index((row_idx, left_edge), image_mask.shape)
            # get all line start indices for that line index by adding integer whole scans
            line_starts_new = list(
                range(
                    row_start_flat_idx + pixel_shift,
                    sample_runtime[-1] + pixel_shift,
                    image_mask.size,
                )
            )
            if (np.array(line_starts_new) < 0).any():  # TESTESTEST
                print("Warning: There are line starts lower than 0!")
            # get the flat index of the line stop for row
            row_stop_flat_idx = np.ravel_multi_index((row_idx, right_edge), image_mask.shape)
            # get all line stops indices for that line index by adding integer whole scans
            line_stops_new = list(
                range(
                    row_stop_flat_idx + pixel_shift,
                    sample_runtime[-1] + pixel_shift,
                    image_mask.size,
                )
            )

            try:
                line_starts_new, line_stops_new = [
                    list(tup) for tup in zip(*zip(line_starts_new, line_stops_new))
                ]
            except ValueError:
                continue
            else:
                # mark line starts with '-row_idx' except for zero, where 'ZERO_LINE_START_ADDER' is used instead
                line_start_labels += [
                    (-ZERO_LINE_START_ADDER if row_idx == 0 else -row_idx)
                    for elem in range(len(line_starts_new))
                ]
                # mark line ends with '-row_idx - LINE_END_ADDER'
                line_stop_labels += [
                    (-row_idx - LINE_END_ADDER) for elem in range(len(line_stops_new))
                ]
                line_starts += line_starts_new
                line_stops += line_stops_new

            # add row to ROI
            nonzero_image_row_idxs = image_mask[row_idx, :].nonzero()[0]
            left_edge, right_edge = nonzero_image_row_idxs[0], nonzero_image_row_idxs[-1]
            roi["row"].appendleft(row_idx)
            roi["col"].appendleft(left_edge)
            roi["row"].append(row_idx)
            roi["col"].append(right_edge)

        # repeat first point to close the polygon
        roi["row"].append(roi["row"][0])
        roi["col"].append(roi["col"][0])

        # convert to Numpy arrays
        line_starts_prt: np.ndarray = (
            np.array(line_starts, dtype=np.int64) * laser_freq_hz // ao_sampling_freq_hz
        )
        line_stops_prt = np.array(line_stops, dtype=np.int64) * laser_freq_hz // ao_sampling_freq_hz
        line_start_labels_arr = np.array(line_start_labels, dtype=np.int16)
        line_stop_labels_arr = np.array(line_stop_labels, dtype=np.int16)

        # filter out line markers which do not belong in the section
        in_section_bool_idxs = line_starts >= sample_runtime[0]
        line_starts_prt = line_starts_prt[in_section_bool_idxs]
        line_stops_prt = line_stops_prt[in_section_bool_idxs]
        line_start_labels_arr = line_start_labels_arr[in_section_bool_idxs]
        line_stop_labels_arr = line_stop_labels_arr[in_section_bool_idxs]

        return (
            line_starts_prt,
            line_stops_prt,
            line_start_labels_arr,
            line_stop_labels_arr,
            {key: np.array(val, dtype=np.uint16) for key, val in roi.items()},
        )

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
                line_corr_list.append(
                    {
                        "lag": lags * 1e3 / sampling_freq_Hz,  # in ms
                        "corrfunc": c,
                    }
                )
        return line_corr_list

    def get_bright_pixels(self, img: np.ndarray, mask: np.ndarray, thresh_factor=1000, **kwargs):
        """Get a mask for 'bright pixels' of an image using a histogram-based heuristic method."""
        # TODO: generalize so that it also catches "dark" pixels

        # select number of bins based on number of unique values in ROI
        n_unique = len(np.unique(img[mask]))
        n_bins = max(10, round(n_unique / 3.5))

        scan_hist, bin_edges = np.histogram(img[mask], bins=n_bins)
        bin_num = np.arange(len(scan_hist))
        bg_part = round(n_bins / 10)

        # fit Gaussian to histogram
        FP = curve_fit_lims(
            gaussian_1d_fit,
            (
                scan_hist[bg_part:].max() / 10,
                np.mean(bin_num[bg_part:]),
                len(bin_num[bg_part:]) / 10,
                0,
            ),
            bin_num[bg_part:],
            scan_hist[bg_part:],
            curve_fit_kwargs=dict(
                bounds=(  # A, mu, sigma, bg
                    [0, 0, 0, -10],
                    [1e4, bin_num.max(), bin_num.max(), 0],
                )
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


class RawFileData:
    """
    Holds a single file's worth of processed, TDC-based, time-tagged photon data which is used in turn for photon delay-time calibration
    and analysis of the entire measurement data.
    This version uses Numpy.mmap (memory-mapping) for getting the initially save 'raw' data (which is unchanging after initial processing of byte data.
    Therefore, no actual data is ever kept in the object (unless should_dump=False)
    """

    def __init__(
        self,
        file_num: int,
        dump_path: Path,
        coarse: np.ndarray = None,
        coarse2: np.ndarray = None,
        fine: np.ndarray = None,
        pulse_runtime: np.ndarray = None,
        delay_time: np.ndarray = None,
        line_num: np.ndarray = None,
        should_avoid_dumping=False,
        was_data_dumped: bool = False,
        **kwargs,
    ):
        """
        Unite the (equal shape) arrays into a single 2D array and immediately dump to disk.
        From now on will be accessed via memory mapping.
        """

        self._file_num = file_num
        self._file_name = f"raw_data_{file_num}.npy"
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
        self._was_data_dumped = was_data_dumped
        self.compressed_file_path: Path = None

    @property
    def coarse(self):
        if self._was_data_dumped:
            return self._read_mmap(0).astype(np.int16)
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
            return self._read_mmap(1).astype(np.int16)
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
            return self._read_mmap(2).astype(np.int16)
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
            return self._read_mmap(3).astype(np.int64)
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
            return self._read_mmap(4).astype(np.float16)
        else:
            return self._delay_time

    @delay_time.setter
    def delay_time(self, new: np.ndarray):
        # NOTE: this does not allow for slicing! must assign a full delay_time array of same size
        if self._was_data_dumped:
            return self._write_mmap_row(4, new)
        else:
            self._delay_time = new

    @property
    def line_num(self):
        if self._was_data_dumped:
            try:
                return self._read_mmap(5).astype(np.int16)
            except IndexError:
                return None
        else:
            return self._line_num

    @line_num.setter
    def line_num(self, new: np.ndarray):
        if self._was_data_dumped:
            return RuntimeError("RawFileData attributes are read-only.")
        else:
            self._line_num = new

    def _read_mmap(self, row_idx: Union[int, slice] = slice(None), file_path=None):
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

    def _write_mmap_row(self, row_idx: int, new_row: np.ndarray):
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
        data = self._read_mmap()

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
    size_estimate_mb: float
    duration_s: float
    skipped_duration: float
    all_section_slices: np.ndarray = None
    avg_cnt_rate_khz: float = None
    image: np.ndarray = None
    bg_line_corr: List[Dict[str, Any]] = None

    # continuous scan
    split_duration_s: float = None
    # TODO: 'split_duration_s' should be an attribute of the measurement, not a specific data file - it is uniform anyway

    # angular scan
    valid_lines: np.ndarray = None
    samples_per_line: int = None
    n_lines: int = None
    sec_roi_list: List[Dict[str, deque]] = None
    bw_mask: np.ndarray = None
    image_bw_mask: np.ndarray = None
    normalized_masked_alleviated_image: np.ndarray = None

    @property
    def n_corr_splits(self):
        # angular scan
        if self.valid_lines:
            return sum(
                [len(self.valid_lines[sec_idx]) for sec_idx in range(len(self.all_section_slices))]
            )
        else:
            return "Implement this - find number of splits for continuous/static scans..."


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
        # create a filter for genuine fluorscence (ignoring afterpulsing)
        bin_num = np.digitize(split_dt, self.fine_bins)
        # adding a final zero value for NaNs (which are put in the last bin by np.digitize)
        filter = np.hstack(([0], filter, [0]))  # TODO: should the filter be created like this?
        # add the relevent filter values to the correlator filter input list
        filter_input = filter[bin_num]
        return filter_input

    def plot(self, parent_ax=None, **plot_kwargs):
        """Doc."""

        valid_idxs = self.valid_limits.valid_indices(self.t_hist)

        with Plotter(
            parent_ax=parent_ax,
            subplots=(1, 2),
            xlabel="time (ns)",
            **plot_kwargs,
        ) as axes:
            axes[0].set_title("Filter Ingredients")
            axes[0].set_yscale("log")
            axes[0].plot(
                self.t_hist[valid_idxs],
                self.I_j / self.norm_factor,
                label="Normalized $I_j$ (raw histogram)",
            )
            axes[0].plot(
                self.t_hist[valid_idxs],
                self.baseline / self.norm_factor * np.ones(self.t_hist[valid_idxs].shape),
                label="Normalized baseline",
            )
            axes[0].plot(
                self.t_hist[valid_idxs],
                self.M.T[0],
                label="$M_{j1}$ (ideal fluorescence decay curve)",
            )
            axes[0].plot(
                self.t_hist[valid_idxs],
                self.M.T[1],
                label="$M_{j2}$ (ideal afterpulsing 'decay' curve)",
            )
            axes[0].set_ylabel("Norm. Photon Count")
            axes[0].set_ylim(self.baseline / self.norm_factor / 10, None)
            axes[0].legend()

            axes[1].set_title("Filter")
            try:  # TODO: fix this error at the filter-building level!
                axes[1].plot(
                    self.t_hist,
                    self.filter.T,
                    label=["$F_{1j}$ (signal)", "$F_{2j}$ (afterpulsing)"],
                )
                axes[1].plot(self.t_hist, self.filter.sum(axis=0), label="$F_{1j}+F_{2j}=1$")
            except ValueError as exc:
                print(exc)
                print("^ unifying lengh of filter to t_hist...")
                axes[1].plot(
                    self.t_hist,
                    unify_length(self.filter.T, (self.t_hist.size, 2)),
                    label=["$F_{1j}$ (signal)", "$F_{2j}$ (afterpulsing)"],
                )
                axes[1].plot(
                    self.t_hist,
                    unify_length(self.filter.sum(axis=0), self.t_hist.shape),
                    label="$F_{1j}+F_{2j}=1$",
                )
            axes[1].set_ylabel("Filter Value")
            axes[1].legend()

            # focus on valid limits
            plot_lims = Limits(self.valid_limits)
            if plot_lims.upper == np.inf:
                plot_lims.upper = max(self.t_hist)
            axes[0].set_xlim(*plot_lims)
            axes[1].set_xlim(*plot_lims)


@dataclass
class TDCPhotonFileData:
    """Holds the total processed data of a single measurement file."""

    file_num: int
    general: GeneralFileData
    raw: RawFileData
    dump_path: Path

    def __repr__(self):
        return f"TDCPhotonFileData(idx={self.file_num}, dump_path={self.dump_path})"

    def __lt__(self, other):
        return self.file_num < other.file_num

    def get_section_line_splits(
        self,
        sec_slice: slice,
        sec_idx: int,
        gate_ns,
        **kwargs,
    ):
        """Splits are all photons belonging to each scan line."""

        # get valid indices
        sec_dt = self.raw.delay_time[sec_slice]
        # NOTE: NaNs mark line starts/ends (used to create valid = -1/-2 needed in C code)
        nan_idxs = np.isnan(sec_dt)
        in_gate_idxs = gate_ns.valid_indices(sec_dt)
        valid_idxs = in_gate_idxs | nan_idxs
        # create split components
        sec_dt = sec_dt[valid_idxs]
        sec_prt = self.raw.pulse_runtime[sec_slice][valid_idxs]
        sec_ts = np.hstack(([0], np.diff(sec_prt)))
        sec_dt_ts = np.vstack((sec_dt, sec_ts))
        sec_line_num = self.raw.line_num[sec_slice][valid_idxs]
        # add validity to each photon (based on line attribution)
        sec_valid_lines = self.general.valid_lines[sec_idx]
        for line_idx in sec_valid_lines:
            line_split = self._add_validity(sec_dt_ts, line_idx, sec_line_num, **kwargs)
            yield line_split

    def get_section_continuous_splits(
        self,
        sec_slice: slice,
        gate_ns: Gate,
        *args,
        **kwargs,
    ):
        """Continuous scan/static measurement - splits are arbitrarily cut along the measurement"""

        sec_dt = self.raw.delay_time[sec_slice]
        gate_idxs = gate_ns.valid_indices(sec_dt)
        sec_dt = sec_dt[gate_idxs]
        sec_prt = self.raw.pulse_runtime[sec_slice][gate_idxs]
        sec_dt_prt = np.vstack((sec_dt, sec_prt))

        # add all splits for each requested xcorr type
        yield from self._split_continuous_section(sec_dt_prt)

    def _split_continuous_section(
        self,
        dt_prt_in: np.ndarray,
    ) -> np.ndarray:
        """Doc."""

        # get photon timings
        t_photon = dt_prt_in[1] / self.general.laser_freq_hz

        # trim the split duration to maximum possible (minus 5%), and warn user about it
        if self.general.split_duration_s > t_photon[-1]:
            print(
                f"File #{self.file_num}: Split duration ({self.general.split_duration_s:.2f} s) is longer than the entire section ({t_photon[-1]:.2f} s)... reducing split duration... ",
                end="",
            )
            self.general.split_duration_s = t_photon[-1] * 0.95

        # get split end indices
        split_end_idxs = (
            np.nonzero(np.diff(t_photon % self.general.split_duration_s) < 0)[0] + 1
        ).tolist()

        # get a list of split limits (dropping the last underfilled split)
        split_edges = [0] + split_end_idxs
        all_split_edges = np.array([split_edges[:-1], split_edges[1:]]).T

        # return the splits as a generator
        ts_out = np.hstack(([0], np.diff(dt_prt_in[1])))
        dt_ts_out = np.vstack((dt_prt_in[0], ts_out))
        return (dt_ts_out[:, start_idx:end_idx] for start_idx, end_idx in all_split_edges)

    def _add_validity(
        self,
        dt_ts_in,
        line_idx,
        line_num,
        **kwargs,
    ) -> np.ndarray:
        """Doc."""

        valid = (line_num == line_idx).astype(np.int8)
        if line_idx != 0:
            valid[line_num == -line_idx] = -1
        else:
            valid[line_num == -ZERO_LINE_START_ADDER] = -1
        valid[line_num == -line_idx - LINE_END_ADDER] = -2

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

            # combine delay times, time-stamps and validity
            dt_ts_valid_out = np.vstack((dt_ts_out, valid))

        else:
            dt_ts_valid_out = np.vstack(([], [], []))

        return dt_ts_valid_out


class TDCPhotonMeasurementData(list):
    """
    Holds lists of the entire meausrement data (all files).
    Capable of rotation to/from disk of specific data type.
    """

    def __init__(self):
        super().__init__()

    def sort(self):
        """Sort in-place according to file numbers"""

        # create a sorted list (by file number)
        sorted_list = sorted(self)
        # replace with sorted list
        self.clear()
        self += sorted_list

    def generate_splits(self, gate_ns=Gate(), **kwargs) -> Generator[np.ndarray, None, None]:
        """
        Prepare SoftwareCorrelator input from complete measurement data.
        """

        # iterate over all files
        for p in self:
            # get splits separately for each section (the main reason is that the line numbers in different sections are not neccessarily identical)
            # TODO: I could possibly define 'global' line numbers for each file, which could allow me to combine matching photon line numbers of different sections into the same splits
            for sec_idx, sec_slice in enumerate(p.general.all_section_slices):
                if p.raw.line_num is not None:
                    # line data
                    yield from p.get_section_line_splits(sec_slice, sec_idx, gate_ns, **kwargs)
                else:
                    # continuous data
                    yield from p.get_section_continuous_splits(sec_slice, gate_ns, **kwargs)


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
            coarse_bins = self.coarse_bins
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
            axes[0, 0].semilogy(coarse_bins, h, "o", label="Valid Bins")
            axes[0, 0].semilogy(
                coarse_bins[np.isin(coarse_bins, coarse_calib_bins)],
                h[np.isin(coarse_bins, coarse_calib_bins)],
                "o",
                markersize=4,
                label="Calibration Bins",
            )
            axes[0, 0].set_xlabel("Coarse Bin")
            axes[0, 0].set_ylabel("Photons")
            axes[0, 0].legend()

            axes[0, 1].plot(self.h_tdc_calib, "-o", label="Calibration Photon Histogram")
            axes[0, 1].set_xlabel("TDC Bin (Adder)")
            axes[0, 1].set_ylabel("Photons")
            axes[0, 1].legend()

            axes[1, 0].semilogy(t_hist, all_hist_norm, "-o", label="Photon Lifetime Histogram")
            axes[1, 0].set_xlabel("Time (ns)")
            axes[1, 0].set_ylabel("Photons (Normalized)")
            axes[1, 0].legend()

            axes[1, 1].plot(t_calib, "-o", label="TDC Calibration")
            axes[1, 1].set_xlabel("TDC Bin (Adder)")
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
            exponent_with_background_fit,
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
        hard_gate_ns: Gate,
        meas_type: str,
        baseline_tail_perc=0.3,
        hard_gate_max_bin=-40,
        **kwargs,
    ) -> np.ndarray:
        """Doc."""

        # make copies so that originals are preserved
        all_hist_norm = copy(self.all_hist_norm)
        t_hist = copy(self.t_hist)

        # define valid bins to work with
        peak_bin = max(np.nanargmax(all_hist_norm) - 2, 0)
        if hard_gate_ns:
            # in case of a hard gate, make the max bin 'hard_gate_max_bin' / 10  (4 ns defualt) smaller
            valid_limits = Limits(t_hist[peak_bin], t_hist[hard_gate_max_bin]) & hard_gate_ns
        else:
            # otherwise, leave the upper limit open
            valid_limits = Limits(t_hist[peak_bin], np.inf)
        valid_idxs = valid_limits.valid_indices(t_hist)

        # interpolate over NaNs
        nans, x = nan_helper(all_hist_norm)  # get nans and a way to interpolate over them later
        #        print("nans: ", nans.sum()) # TESTESTEST - why always 23?
        all_hist_norm[nans] = np.interp(x(nans), x(~nans), all_hist_norm[~nans])

        # calculate the baseline using the mean of the tail of the (hard-gate-limited) histogram
        baseline = all_hist_norm[valid_idxs][-round(len(t_hist) * baseline_tail_perc) :].mean()

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
        #        lower_idx = round(valid_limits.lower * 10)
        lower_idx = int(np.ceil(valid_limits.lower * 10))  # TESTESTEST
        F_pad_before = np.zeros((2, lower_idx))
        if valid_limits.upper != np.inf:
            upper_idx = int(valid_limits.upper * 10)
            F_pad_after = np.zeros((2, max(0, len(t_hist) - upper_idx)))
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

    def process_data(
        self, file_num, full_data, should_dump=True, force_processing: bool = True, **proc_options
    ) -> TDCPhotonFileData:
        """Doc."""

        # if is allowed and able to use dumped pre-processed data
        if (
            not force_processing
            and (dumped_fpath := self.dump_path / f"raw_data_{file_num}.npy").exists()
        ):
            raise NotImplementedError(
                f"I still need to save/dump the GeneralFileData to '{dumped_fpath.parent}', alongside the raw data (used for memory-mapping arrays) so that I can load it as well, as some of its properties are necessary downstream (e.g. correlation.)"
            )
        #            p = TDCPhotonFileData(
        #                idx,
        #                GeneralFileData(
        #                    # general
        #                    laser_freq_hz=self.laser_freq_hz,
        #                    size_estimate_mb=max(section_lengths) / 1e6,
        #                    duration_s=duration_s,
        #                    skipped_duration=skipped_duration,
        #                    all_section_slices=all_section_slices,
        #                    # continuous scan
        #                    split_duration_s=split_duration_s,
        #                ),
        #                RawFileData(
        #                idx,
        #                self.dump_path,
        #                should_avoid_dumping=True,
        #                was_data_dumped=True,
        #                ),
        #                self.dump_path
        #            )
        #            if proc_options.get("is_verbose"):
        #                print(f"Pre-processed file used: {dumped_fpath}", end=" ")
        # process file
        else:
            # sFCS
            if scan_settings := full_data.get("scan_settings"):
                if (scan_type := scan_settings["pattern"]) == "circle":  # Circular sFCS
                    p = self._process_circular_scan_data_file(file_num, full_data, **proc_options)
                elif scan_type == "angular":  # Angular sFCS
                    p = self._process_angular_scan_data_file(file_num, full_data, **proc_options)
                elif scan_type == "image":  # image sFCS
                    p = self._process_image_scan_plane_data(file_num, full_data, **proc_options)
            # FCS
            else:
                scan_type = "static"
                p = self._convert_fpga_data_to_photons(
                    file_num,
                    **proc_options,
                )

        with suppress(AttributeError):
            # AttributeError: p is None
            # dumping of raw data to disk
            # False only if manually set to avoid re-dumping many existing identical files
            if should_dump:
                if proc_options.get("is_verbose"):
                    print("Dumping raw data file to disk (if needed)...", end=" ")
                    # TODO: this isn't transparent enough. the should/shouldn't dump flag should be determined from the file (p) and notified to user
                p.raw.dump()

            # add general properties
            p.general.avg_cnt_rate_khz = full_data.get("avg_cnt_rate_khz")

        return p

    def _convert_fpga_data_to_photons(
        self,
        file_num,
        byte_data_path=None,
        byte_data=None,
        len_factor=0.01,
        byte_data_slice=None,
        split_duration_s: float = 1.0,
        **proc_options,
    ) -> TDCPhotonFileData:
        """Doc."""

        if proc_options.get("is_verbose"):
            print("Converting raw data to photons... ", end="")

        # getting byte data by memory-mapping (unless supplied - alignment measurements and pre-conversion data only)
        if byte_data is None:
            # NOTE - WARNING! byte data is memory mapped - mode must always be kept to 'r' to avoid writing over byte_data!!!
            byte_data = np.load(byte_data_path, "r")

        # option to use only certain parts of data (for testing)
        if byte_data_slice is not None:
            if proc_options.get("is_verbose"):
                print(
                    f"Using {byte_data_slice} of data ({byte_data[byte_data_slice].size/byte_data.size:.2%} used) ",
                    end="",
                )
            byte_data = byte_data[byte_data_slice]

        section_slices, tot_single_errors = self._find_all_section_edges(byte_data)
        section_lengths = [sec_slice.stop - sec_slice.start for sec_slice in section_slices]

        photon_idxs_list: List[int] = []
        section_runtime_slices = []
        for sec_slice in section_slices:
            if sec_slice.stop - sec_slice.start > sum(section_lengths) * len_factor:
                section_runtime_start = len(photon_idxs_list)
                section_photon_indxs = list(range(sec_slice.start, sec_slice.stop, self.GROUP_LEN))
                section_runtime_end = section_runtime_start + len(section_photon_indxs)
                photon_idxs_list += section_photon_indxs
                section_runtime_slices.append(slice(section_runtime_start, section_runtime_end))

        photon_idxs = np.array(photon_idxs_list)

        if proc_options.get("is_verbose"):
            if len(section_slices) > 1:
                print(
                    f"Found {len(section_slices) - 1:,} removeable discontinuities, potentially causing sections of lengths: {', '.join([str(round(sec_len * 1e-3)) for sec_len in section_lengths])} Kb.",
                    end=" ",
                )
                print(
                    f"Using all valid (> {len_factor:.1%}) sections ({len(section_runtime_slices)}/{len(section_slices)}).",
                    end=" ",
                )
            else:
                print(
                    f"Found a single section of length: {section_lengths[0]*1e-6:.1f} Mb.", end=" "
                )
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
        time_stamps = np.hstack(([0], np.diff(pulse_runtime).astype(np.int32)))

        # Break each section into sub-setions according to simple outlier timestamp detection
        all_section_slices, skipped_duration = self._section_continuous_data(
            pulse_runtime, time_stamps, **proc_options
        )

        # handling coarse and fine times (for gating)
        coarse = byte_data[photon_idxs + 4].astype(np.int16)
        fine = byte_data[photon_idxs + 5].astype(np.int16)

        # some fix due to an issue in FPGA
        coarse_mod64 = np.mod(coarse, 64)
        coarse2 = coarse_mod64 - np.mod(coarse_mod64, 4) + (coarse // 64)
        coarse = coarse_mod64

        # Duration calculation
        duration_s = (time_stamps / self.laser_freq_hz).sum()

        return TDCPhotonFileData(
            # TODO: using properties, I can ignore the fact that some attributes are 'raw' and some are 'general' and get less code
            file_num,
            GeneralFileData(
                # general
                laser_freq_hz=self.laser_freq_hz,
                size_estimate_mb=max(section_lengths) / 1e6,
                duration_s=duration_s,
                skipped_duration=skipped_duration,
                all_section_slices=all_section_slices,
                # continuous scan
                split_duration_s=split_duration_s,
            ),
            RawFileData(
                file_num=file_num,
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
        pulse_runtime: np.ndarray,
        time_stamps: np.ndarray,
        max_time_stamp=300_000,  # 75_000,  # TODO: this could be found heuristically based on the timestamp histogram (where it drops to 0)
        max_outliers=100,  # TESTESTEST
        len_factor=0.01,
        should_use_all_sections=True,
        **kwargs,
    ):
        """Find outliers and create sections seperated by them. Short sections are discarded"""

        jump_idxs = (time_stamps > max_time_stamp).nonzero()[0]
        if n_outliers := len(jump_idxs):
            if n_outliers > max_outliers:
                raise RuntimeError(
                    f"Too many jumps in runtime (found: {n_outliers:,}, allowed: {max_outliers:,})."
                )
            if kwargs.get("is_verbose"):
                print(f"Found {n_outliers + 1:,} sections caused by jumps in runtime. ", end="")

            # find all section edges
            sec_edges = np.hstack(([0], jump_idxs, [pulse_runtime.size]))
            all_section_slices = [
                slice(*edges) for edges in np.array([sec_edges[:-1], sec_edges[1:]]).T
            ]

            # Filter short sections
            sec_lens = [
                pulse_runtime[sec_slice.stop - 1] - pulse_runtime[sec_slice.start]
                for sec_slice in all_section_slices
            ]
            valid_sec_idxs = [
                idx for idx, sec_len in enumerate(sec_lens) if sec_len > sum(sec_lens) * len_factor
            ]
            all_section_slices = [all_section_slices[idx] for idx in valid_sec_idxs]
            if (n_outliers + 1) > len(all_section_slices) and kwargs.get("is_verbose"):
                # TODO: when this happens, it causes an issue with line markers. this sometimes happens in the first timestamps of a file, resulting in a short section - needs to be solved generally
                # currently, we are just ignoring the first few data elements using 'byte_data_slice' kwarg.
                print(
                    f"Using all valid (> {len_factor:.1%}) sections: {', '.join([str(idx+1) for idx in valid_sec_idxs])} ({len(all_section_slices)}/{n_outliers + 1}). ",
                    end="",
                )

            # find lost duration
            skipped_duration = pulse_runtime.size - sum(sec_lens)

        else:
            skipped_duration = 0
            all_section_slices = [slice(0, pulse_runtime.size)]

        if not should_use_all_sections:  # using largest section only
            all_section_slices = [all_section_slices[(largest_sec_idx := np.argmax(sec_lens))]]
            skipped_duration = max(sec_lens)
            print(f"Using largest (section num.{largest_sec_idx+1}).", end=" ")

        return all_section_slices, skipped_duration

    def _find_all_section_edges(
        self,
        byte_data: np.ndarray,
    ) -> Tuple[List[slice], int]:
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
            section_edges.append(slice(new_edge_start, new_edge_stop))
            last_edge_stop = section_edges[-1].stop
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

    def _process_circular_scan_data_file(self, idx, full_data, **proc_options) -> TDCPhotonFileData:
        """
        Processes a single circular sFCS data file ('full_data').
        Returns the processed results as a 'TDCPhotonData' object.
        '"""
        # TODO: can this method be moved to the appropriate Mixin class?

        p = self._convert_fpga_data_to_photons(
            idx,
            **proc_options,
        )

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
        mask_short_rows_fac=0.5,
        roi_selection="auto",
        should_alleviate_bright_pixels=False,
        agg_ratio=0.001,
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

        # initialize the total scan image and the ROI mask
        img = np.zeros((p.general.n_lines + 1, p.general.samples_per_line), dtype=np.uint16)
        img_bw = np.full(img.shape, False, dtype=bool)
        roi_list = []

        pulse_runtime_list = []
        coarse_list = []
        coarse2_list = []
        fine_list = []
        pixel_num_list = []
        line_num_list = []
        delay_time_list = []
        new_section_slices = []
        p.general.valid_lines = []
        p.general.bg_line_corr = []
        last_idx = 0
        for sec_slice in p.general.all_section_slices:
            # after concatenating the sections, these will be the new section slices to use for creating splits for correlator
            new_section_slices.append(
                slice(last_idx, last_idx + (sec_slice.stop - sec_slice.start))
            )
            last_idx = new_section_slices[-1].stop

            sec_pulse_runtime = p.raw.pulse_runtime[sec_slice]
            (
                sec_image,
                sec_sample_runtime,
                sec_pixel_num,
                sec_line_num,
                sec_pixel_shift,
            ) = self.convert_angular_scan_to_image(
                sec_pulse_runtime,
                self.laser_freq_hz,
                ao_sampling_freq_hz,
                p.general.samples_per_line,
                p.general.n_lines,
                **kwargs,
            )
            sec_prt_shift = sec_pixel_shift * (self.laser_freq_hz // ao_sampling_freq_hz)
            sec_pulse_runtime_shifted = sec_pulse_runtime + sec_prt_shift

            # get BW section image
            if roi_selection == "auto":
                if kwargs.get("is_verbose"):
                    print("Thresholding and smoothing...", end=" ")
                try:
                    sec_image_mask = self._threshold_and_smooth(sec_image.copy())
                except ValueError:
                    raise RuntimeError("Automatic ROI selection: Thresholding failed")
            elif roi_selection == "all":
                sec_image_mask = np.full(sec_image.shape, True)
            else:
                raise ValueError(
                    f"roi_selection='{roi_selection}' is not supported. Only 'auto' is, at the moment."
                )

            # cut edges
            bw_temp = np.full(sec_image_mask.shape, False, dtype=bool)
            bw_temp[:, linear_part] = sec_image_mask[:, linear_part]
            sec_image_mask = bw_temp

            # discard short rows, then fill long rows
            m2 = np.sum(sec_image_mask, axis=1)
            sec_image_mask[m2 < mask_short_rows_fac * m2.max(), :] = False
            true_vals = np.argwhere(sec_image_mask)
            for row_idx in np.unique(sec_image_mask.nonzero()[0]):
                true_idxs = true_vals[true_vals[:, 0] == row_idx][:, 1]
                sec_image_mask[row_idx, true_idxs.min() : true_idxs.max() + 1] = True

            # Getting line markers and ROI
            if kwargs.get("is_verbose"):
                print("Building ROI...", end=" ")

            # get flat indices of in-ROI photons (need to invert every second row since scan is "snake-like")
            sec_data_mask = sec_image_mask.copy()
            sec_data_mask[1::2, :] = np.flip(sec_data_mask[1::2, :], 1)
            try:
                (
                    sec_line_starts_prt,
                    sec_line_stops_prt,
                    sec_line_start_labels,
                    sec_line_stop_labels,
                    sec_roi,
                ) = self._get_line_markers_and_roi(
                    sec_image_mask,
                    sec_data_mask,
                    sec_sample_runtime,
                    sec_pixel_shift,
                    self.laser_freq_hz,
                    ao_sampling_freq_hz,
                )
            except IndexError:
                # TODO: this should be allowed to propagate and handled up the call chain
                if kwargs.get("is_verbose"):
                    print("ROI is empty (need to figure out the cause). Skipping file.\n")
                return None

            # Set an 'out-of-bounds' line number to all photons outside mask (these are later ignored by '_prepare_correlator_input')
            sec_valid_logical_idxs = sec_data_mask[sec_line_num, sec_pixel_num]
            sec_line_num[~sec_valid_logical_idxs] = OUT_OF_SAMPLE_LINE

            # ignore lines with outlier pixels (bright or dark)
            # gather the start/stop indices of bad lines
            bad_line_idxs = []
            if should_alleviate_bright_pixels:
                # sort the line starts/stops/labels so that they can be filtered properly
                lines_starts_sorted_idxs = np.argsort(sec_line_starts_prt)
                sec_line_starts_prt = sec_line_starts_prt[lines_starts_sorted_idxs]
                sec_line_stops_prt = sec_line_stops_prt[lines_starts_sorted_idxs]
                sec_line_start_labels = sec_line_start_labels[lines_starts_sorted_idxs]
                sec_line_stop_labels = sec_line_stop_labels[lines_starts_sorted_idxs]

                # rebuild the section image from the single scans minus the bad rows
                sec_image = np.zeros_like(sec_image)
                scan_min_row, *_, scan_max_row = np.unique(sec_image_mask.nonzero()[0])
                n_rows_in_scan = scan_max_row - scan_min_row
                total_scans = sec_line_starts_prt.size / n_rows_in_scan
                # TODO: The least amount o data is lost when using a sinle scan. The only issue might be that in low CR scans,
                # a single scan will not have enough photons - figure out what the minimum amount of photons is and only agregate as necessary
                # Use the mean countrate to know in advance?
                n_scans_per_image = max(1, int(agg_ratio * total_scans))  # TODO: see above
                lines_per_image = n_scans_per_image * n_rows_in_scan

                if kwargs.get("is_verbose"):
                    print(
                        f"Ignoring lines with bright pixels, aggregating per {n_scans_per_image} scans ({sec_line_starts_prt.size // lines_per_image} images)... ",
                        end="",
                    )

                # Filter only the first and last lines of every "image" (aggregated scans), since only knowing the starts and ends is needed.
                # This can be achieved by filtering the line start/stpos
                image_edges_line_edges_prt = np.vstack(
                    (
                        sec_line_starts_prt[::lines_per_image],
                        unify_length(
                            sec_line_stops_prt[lines_per_image - 1 :: lines_per_image],
                            sec_line_starts_prt[::lines_per_image].shape,
                        ),
                    )
                )
                image_edges_line_edges_prt = image_edges_line_edges_prt[
                    :, image_edges_line_edges_prt[1] != 0
                ]

                for image_idx, (image_first_line_start_prt, image_last_line_stop_prt) in enumerate(
                    zip(*image_edges_line_edges_prt)
                ):
                    # get the line indices and update the minimum and maximum scan indices based on them
                    image_idxs = Limits(
                        image_first_line_start_prt, image_last_line_stop_prt
                    ).valid_indices(sec_pulse_runtime_shifted, as_bool=False)

                    # build single-scan image using the section runtime with the line indices with convert_angular_scan_to_image()
                    scan_image, *_ = self.convert_angular_scan_to_image(
                        sec_pulse_runtime_shifted[image_idxs[0] : image_idxs[-1]],
                        self.laser_freq_hz,
                        ao_sampling_freq_hz,
                        p.general.samples_per_line,
                        p.general.n_lines,
                        should_fix_shift=False,
                    )

                    # find the bad rows
                    bright_pixels_img_bw = self.get_bright_pixels(
                        scan_image, sec_image_mask, **kwargs
                    )
                    image_bad_row_idxs = np.unique(
                        bright_pixels_img_bw.nonzero()[0]
                    )  # this is supposed to get the indices of the bad lines
                    # add indices of all aggregated scans
                    all_scan_idxs = image_bad_row_idxs.tolist()
                    for scan_idx in range(1, n_scans_per_image):
                        all_scan_idxs += (image_bad_row_idxs + n_rows_in_scan * scan_idx).tolist()

                    # add valid rows of scan to the total section image
                    scan_image[image_bad_row_idxs] = 0
                    sec_image += scan_image

                    bad_line_idxs += (
                        np.array(all_scan_idxs) + (image_idx * lines_per_image)
                    ).tolist()

                # TODO: why does this happen? Possibly due to jumps in the runtime?
                bad_line_idxs = sorted(bad_line_idxs)
                print(
                    f"GETTING RID OF ALL INDICES ABOVE 'sec_line_starts_prt.size - 1' ({sec_line_starts_prt.size}):"
                )  # TESTESTEST
                while bad_line_idxs and bad_line_idxs[-1] >= sec_line_starts_prt.size:
                    print(f"last bad index: {bad_line_idxs[-1]}")  # TESTESTEST
                    bad_line_idxs.pop()

                # label all photon in the bad lines as IGNORED_LINE
                if kwargs.get("is_verbose"):
                    print(
                        f"marking {len(bad_line_idxs)}/{(n_lines_total := int(n_rows_in_scan * total_scans))} ({len(bad_line_idxs)/n_lines_total:.1%}) 'bad lines'... ",
                        end="",
                    )

                for bad_line_edges in zip(
                    sec_line_starts_prt[bad_line_idxs], sec_line_stops_prt[bad_line_idxs]
                ):
                    single_bad_line_idxs = Limits(bad_line_edges).valid_indices(
                        sec_pulse_runtime_shifted
                    )
                    sec_line_num[single_bad_line_idxs] = IGNORED_LINE

                # use the gathered indices to filter line starts/stops and their corresponding labels
                sec_line_starts_prt = exclude_elements_by_indices_1d(
                    sec_line_starts_prt, bad_line_idxs
                )
                sec_line_stops_prt = exclude_elements_by_indices_1d(
                    sec_line_stops_prt, bad_line_idxs
                )
                sec_line_start_labels = exclude_elements_by_indices_1d(
                    sec_line_start_labels, bad_line_idxs
                )
                sec_line_stop_labels = exclude_elements_by_indices_1d(
                    sec_line_stop_labels, bad_line_idxs
                )

            # TODO: COMMENT HERE
            sec_coarse = p.raw.coarse[sec_slice]
            sec_coarse2 = p.raw.coarse2[sec_slice]
            sec_fine = p.raw.fine[sec_slice]

            # Inserting the line start/stop markers into the arrays
            sec_pulse_runtime = np.hstack(
                (sec_line_starts_prt, sec_line_stops_prt, sec_pulse_runtime_shifted)
            )
            sec_sorted_idxs = np.argsort(sec_pulse_runtime)
            sec_pulse_runtime = sec_pulse_runtime[sec_sorted_idxs]
            sec_line_num = np.hstack(
                (
                    sec_line_start_labels,
                    sec_line_stop_labels,
                    sec_line_num,
                )
            )[sec_sorted_idxs]

            # keep the valid lines for each section
            sec_valid_lines = np.unique((sec_image * sec_image_mask).nonzero()[0])
            p.general.valid_lines.append(sec_valid_lines)

            # Getting section background correlation
            if kwargs.get("is_verbose"):
                print("Getting background correlation...", end=" ")
            p.general.bg_line_corr += self._bg_line_correlations(
                sec_image,
                sec_image_mask,
                sec_valid_lines,
                ao_sampling_freq_hz,
            )

            sec_line_starts_nans = np.full(sec_line_starts_prt.size, NAN_PLACEBO, dtype=np.int16)
            sec_line_stops_nans = np.full(sec_line_stops_prt.size, NAN_PLACEBO, dtype=np.int16)
            sec_coarse = np.hstack((sec_line_starts_nans, sec_line_stops_nans, sec_coarse))[
                sec_sorted_idxs
            ]
            sec_coarse2 = np.hstack((sec_line_starts_nans, sec_line_stops_nans, sec_coarse2))[
                sec_sorted_idxs
            ]
            sec_fine = np.hstack((sec_line_starts_nans, sec_line_stops_nans, sec_fine))[
                sec_sorted_idxs
            ]

            # initialize delay times with lower detector gate (nans at line edges) - filled-in during TDC calibration
            sec_delay_time = np.full(
                sec_pulse_runtime.shape, self.detector_gate_ns.lower, dtype=np.float16
            )
            sec_line_edge_idxs = sec_fine == NAN_PLACEBO
            sec_delay_time[sec_line_edge_idxs] = np.nan

            # keep all section stuff in lists, to be combined into arrays afterwards
            pulse_runtime_list.append(sec_pulse_runtime)
            pixel_num_list.append(sec_pixel_num)
            line_num_list.append(sec_line_num)
            delay_time_list.append(sec_delay_time)
            coarse_list.append(sec_coarse)
            coarse2_list.append(sec_coarse2)
            fine_list.append(sec_fine)

            # combine section images
            img += sec_image
            img_bw += sec_image_mask

            # keep all section ROIs
            roi_list.append(sec_roi)

        # combine sections
        pulse_runtime = np.hstack(pulse_runtime_list)
        line_num = np.hstack(line_num_list)
        delay_time = np.hstack(delay_time_list)
        coarse = np.hstack(coarse_list)
        coarse2 = np.hstack(coarse2_list)
        fine = np.hstack(fine_list)
        # update the section slices after leaving only the in-section indices
        p.general.all_section_slices = new_section_slices

        # replace the raw data after angular scan changes made
        p.raw = RawFileData(
            p.file_num,
            p.dump_path,
            coarse,
            coarse2,
            fine,
            pulse_runtime,
            delay_time,
            line_num,
            **kwargs,
        )
        p.general.image = img
        p.general.image_bw_mask = img_bw
        p.general.sec_roi_list = roi_list

        return p

    def get_delay_time(self, args):
        """Doc."""

        (
            p,
            coarse_bins,
            first_coarse_bin,
            last_coarse_bin,
            max_j,
            l_quarter_tdc,
            r_quarter_tdc,
            t_calib,
            fpga_freq_hz,
            NAN_PLACEBO,
        ) = args

        _delay_time = np.full(p.raw.coarse.shape, np.nan, dtype=np.float64)
        crs = np.minimum(p.raw.coarse, last_coarse_bin) - coarse_bins[max_j - 1]
        crs[crs < 0] += last_coarse_bin - first_coarse_bin + 1

        delta_coarse = p.raw.coarse2 - p.raw.coarse
        delta_coarse[delta_coarse == -3] = 1  # 2bit limitation

        # in the TDC midrange use "coarse" counter
        in_mid_tdc = (p.raw.fine >= l_quarter_tdc) & (p.raw.fine <= r_quarter_tdc)
        delta_coarse[in_mid_tdc] = 0

        # on the right of TDC use "coarse2" counter (no change in delta)
        # on the left of TDC use "coarse2" counter decremented by 1
        on_left_tdc = p.raw.fine < l_quarter_tdc
        delta_coarse[on_left_tdc] -= 1

        photon_idxs = p.raw.fine != NAN_PLACEBO
        try:
            _delay_time[photon_idxs] = (
                t_calib[p.raw.fine[photon_idxs]]
                + (crs[photon_idxs] + delta_coarse[photon_idxs]) / fpga_freq_hz * 1e9
            )
        except IndexError as exc:
            print(
                f"\n[WORKER] Error in file {p.file_num}: {exc}. Please remove/fix the file, and re-run the processing."
            )
            sys.stdout.flush()
        p.raw.delay_time = _delay_time
        return _delay_time[photon_idxs]

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
        coarse_mmap, fine_mmap, coarse_lims = self._unite_coarse_fine_data(data, scan_type)

        if kwargs.get("is_verbose"):
            print("Binning coarse data... ", end="")
        h_all = chunked_bincount(coarse_mmap, max_val=coarse_lims.upper, n_chunks=len(data)).astype(
            np.uint32
        )
        x_all = np.arange(coarse_mmap.max + 1, dtype=np.uint8)

        if pick_valid_bins_according_to is None:
            h_all = h_all[coarse_lims.lower :]
            x_all = np.arange(coarse_lims.lower, coarse_lims.upper + 1, dtype=np.uint8)
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
            print(
                f"Assigning delay times (multiprocessing using {N_CPU_CORES-1}/{N_CPU_CORES} available cores)... ",
                end="",
            )
        total_laser_pulses = 0
        first_coarse_bin, *_, last_coarse_bin = coarse_bins

        # Create a pool of worker processes
        with multiprocessing.Pool(N_CPU_CORES - 1) as pool:
            # Create argument tuples for each process
            args_list = [
                (
                    p,
                    coarse_bins,
                    first_coarse_bin,
                    last_coarse_bin,
                    max_j,
                    l_quarter_tdc,
                    r_quarter_tdc,
                    t_calib,
                    self.fpga_freq_hz,
                    NAN_PLACEBO,
                )
                for p in data
            ]
            # Use the pool to parallelize the processing of data
            delay_time_list = pool.map(self.get_delay_time, args_list)

        if kwargs.get("is_verbose"):
            print("Collecting total pulses... ", end="")

        # get total laser pulses
        total_laser_pulses = sum(p.raw.pulse_runtime[-1] for p in data)

        if kwargs.get("is_verbose"):
            print("Defining 'fine_bins'... ", end="")

        fine_bins = np.arange(
            -time_bins_for_hist_ns / 2,
            max(delay_time.max() for delay_time in delay_time_list) + time_bins_for_hist_ns,
            time_bins_for_hist_ns,
            dtype=np.float16,
        )

        if kwargs.get("is_verbose"):
            print("Binning 'delay_times' into 'fine_bins'...  ", end="")

        t_hist = (fine_bins[:-1] + fine_bins[1:]) / 2
        dt_fbins_idxs = np.digitize(delay_times, fine_bins)

        if kwargs.get("is_verbose"):
            print("Calculating 'hist_weight'...  ", end="")

        hist_weight = np.array(
            [t_weight[dt_fbins_idxs == i].sum() for i in range(1, len(t_hist) + 1)]
        )

        if kwargs.get("is_verbose"):
            print("Calculating 'all_hist'...  ", end="")

        all_hist = np.zeros(fine_bins.size - 1, dtype=np.float64)
        for delay_time in delay_time_list:
            all_hist += np.histogram(delay_time, bins=fine_bins)[0]

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
            n_elem = np.cumsum(
                [0] + [(p.raw.fine > NAN_PLACEBO).sum() for p in data], dtype=np.int64
            )
        else:
            n_elem = np.cumsum([0] + [p.raw.fine.size for p in data])

        # unite coarse and fine times from all files
        coarse = np.empty(shape=(n_elem[-1],), dtype=np.int16)
        fine = np.empty(shape=(n_elem[-1],), dtype=np.int16)
        coarse_lims = Limits()
        for i, p in enumerate(data):
            if scan_type == "angular":
                # remove line starts/ends from angular scan data
                photon_idxs = p.raw.fine > NAN_PLACEBO
            else:
                # otherwise, treat all elements as photons
                photon_idxs = slice(None)
            coarse[n_elem[i] : n_elem[i + 1]] = (valid_file_coarse := p.raw.coarse[photon_idxs])
            fine[n_elem[i] : n_elem[i + 1]] = p.raw.fine[photon_idxs]
            # keep the minimal coarse range to ensure all files use the same coarse bins
            coarse_lims = coarse_lims & Limits(valid_file_coarse.min(), valid_file_coarse.max())
        coarse_mmap = MemMapping(coarse, "coarse.npy")
        fine_mmap = MemMapping(fine, "fine.npy")

        # return the MemMapping instances
        return coarse_mmap, fine_mmap, coarse_lims


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
    gate_ns: Gate = field(default_factory=Gate)
    is_lifetime_img: bool = False
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

        plane_idx = plane_idx or self.image_stack_forward.shape[2] // 2

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

        # subtract the lower gate in case of a lifetime image - this is so the pixel values will represent the actual delay times (not including the pulse delay)
        if self.is_lifetime_img:
            img -= self.gate_ns.lower

        return img
