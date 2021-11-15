"""Data organization and manipulation."""

import logging
import re
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Union

import numpy as np
import scipy
import skimage

from data_analysis.photon_data import (
    CountsImageMixin,
    TDCPhotonData,
    TDCPhotonDataMixin,
)
from data_analysis.software_correlator import CorrelatorType, SoftwareCorrelator
from utilities import file_utilities, fit_tools
from utilities.display import Plotter
from utilities.helper import LimitRange, div_ceil


class CorrFunc:
    """Doc."""

    def __init__(self, gate_ns):
        self.gate_ns = gate_ns
        self.lag = []
        self.countrate_list = []
        self.fit_param = dict()
        self.run_duration: float
        self.skipped_duration = 0
        self.total_duration: float
        self.afterpulse: np.ndarray
        self.vt_um: np.ndarray

    def average_correlation(
        self,
        rejection=2,
        reject_n_worst=None,
        norm_range=LimitRange(1e-3, 2e-3),
        delete_list=[],
        should_plot=False,
        plot_kwargs={},
        **kwargs,
    ):
        """Doc."""

        self.rejection = rejection
        self.norm_range = norm_range
        self.delete_list = delete_list
        self.average_all_cf_cr = (self.cf_cr * self.weights).sum(0) / self.weights.sum(0)
        self.median_all_cf_cr = np.median(self.cf_cr, 0)
        jj = LimitRange(norm_range.upper, 100).idxs_in_limits(self.lag)  # work in the relevant part
        self.score = (
            (1 / np.var(self.cf_cr[:, jj], 0))
            * (self.cf_cr[:, jj] - self.median_all_cf_cr[jj]) ** 2
            / len(jj)
        ).sum(axis=1)

        total_n_rows, _ = self.cf_cr.shape

        if reject_n_worst not in {None, 0}:
            delete_list = np.argsort(self.score)[-reject_n_worst:]
        elif rejection is not None:
            delete_list = np.where(self.score >= self.rejection)[0]
            if len(delete_list) == total_n_rows:
                raise RuntimeError(
                    "All rows are in 'delete_list'! Increase the rejection limit. Ignoring."
                )

        # if 'reject_n_worst' and 'rejection' are both None, use supplied delete list.
        # if no delete list is supplied, use all rows.
        self.j_bad = delete_list
        self.j_good = [row for row in range(total_n_rows) if row not in delete_list]

        self.avg_cf_cr, self.error_cf_cr = _calculate_weighted_avg(
            self.cf_cr[self.j_good, :], self.weights[self.j_good, :]
        )

        j_t = norm_range.idxs_in_limits(self.lag)
        self.g0 = (self.avg_cf_cr[j_t] / self.error_cf_cr[j_t] ** 2).sum() / (
            1 / self.error_cf_cr[j_t] ** 2
        ).sum()
        self.normalized = self.avg_cf_cr / self.g0
        self.error_normalized = self.error_cf_cr / self.g0

        if should_plot:
            self.plot_correlation_function(plot_kwargs=plot_kwargs)

    def plot_correlation_function(
        self,
        parent_ax=None,
        x_field="lag",
        y_field="avg_cf_cr",
        x_scale="log",
        y_scale="linear",
        xlim=None,
        ylim=None,
        plot_kwargs={},
        **kwargs,
    ):

        x = getattr(self, x_field)
        y = getattr(self, y_field)
        if x_scale == "log":  # remove zero point data
            x, y = x[1:], y[1:]

        #        print(f"final plot {x_field} size: {x.size}") # TESTESTEST
        #        print(f"final plot {y_field} size: {y.size}") # TESTESTEST

        with Plotter(parent_ax=parent_ax, xlim=xlim, ylim=ylim, should_autoscale=True) as ax:
            ax.set_xlabel(x_field)
            ax.set_ylabel(y_field)
            ax.set_xscale(x_scale)
            ax.set_yscale(y_scale)
            ax.plot(x, y, "-", **plot_kwargs)

    def fit_correlation_function(
        self,
        x_field="lag",
        y_field="avg_cf_cr",
        y_error_field="error_cf_cr",
        fit_name="diffusion_3d_fit",
        fit_param_estimate=None,
        fit_range=(1e-3, 100),
        x_scale="log",
        y_scale="linear",
        should_plot=False,
    ):

        if fit_param_estimate is None:
            fit_param_estimate = [self.g0, 0.035, 30.0]

        x = getattr(self, x_field)
        y = getattr(self, y_field)
        error_y = getattr(self, y_error_field)
        if x_scale == "log":
            x = x[1:]  # remove zero point data
            y = y[1:]
            error_y = error_y[1:]

        fit_param = fit_tools.curve_fit_lims(
            fit_name,
            fit_param_estimate,
            x,
            y,
            error_y,
            x_limits=fit_range,
            should_plot=should_plot,
            x_scale=x_scale,
            y_scale=y_scale,
        )

        self.fit_param[fit_param["func_name"]] = fit_param


@dataclass
class CorrFuncAccumulator:
    """
    A convecience class for accumulating the outputs of SoftwareCorrelator
    for a CorrFunc during correlation of measurement data.
    """

    cf: CorrFunc
    corrfunc_list: list[np.ndarray] = field(default_factory=list)
    weights_list: list[np.ndarray] = field(default_factory=list)
    cf_cr_list: list[np.ndarray] = field(default_factory=list)
    n_corrfuncs: int = 0

    def __enter__(self):
        """Initiate a temporary structure for accumulating SoftwareCorrelator outputs."""

        return self

    def __exit__(self, *exc):
        """Create padded 2D ndarrays from the accumulated lists and delete the accumulator."""

        lag_len = len(self.cf.lag)
        self.cf.corrfunc, self.cf.weights, self.cf.cf_cr = self.join_and_pad(lag_len)

    def accumulate(self, SC: SoftwareCorrelator):
        """Doc."""

        self.corrfunc_list.append(SC.corrfunc)
        self.weights_list.append(SC.weights)
        self.cf_cr_list.append(SC.cf_cr)
        self.n_corrfuncs += 1

    def join_and_pad(self, lag_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Doc."""

        shape = (self.n_corrfuncs, lag_len)
        corrfunc = np.empty(shape=shape, dtype=np.float64)
        weights = np.empty(shape=shape, dtype=np.float64)
        cf_cr = np.empty(shape=shape, dtype=np.float64)
        for idx in range(self.n_corrfuncs):
            pad_len = lag_len - len(self.corrfunc_list[idx])
            corrfunc[idx] = np.pad(self.corrfunc_list[idx], (0, pad_len))
            weights[idx] = np.pad(self.weights_list[idx], (0, pad_len))
            cf_cr[idx] = np.pad(self.cf_cr_list[idx], (0, pad_len))

        return corrfunc, weights, cf_cr


class SolutionSFCSMeasurement(TDCPhotonDataMixin):
    """Doc."""

    NAN_PLACEBO = -100
    DUMP_PATH = Path("C:/temp_sfcs_data/")
    SIZE_LIMITS_MB = (100, 1e4)

    def __init__(self, name=None):
        self.name = name
        self.data = []  # list to hold the data of each file
        self.cf = dict()
        self.is_data_dumped = False
        self.scan_type: str
        self.duration_min: float = None

    @file_utilities.rotate_data_to_disk
    def read_fpga_data(
        self,
        file_path_template: Union[str, Path],
        file_selection: str = None,
        should_plot=False,
        **kwargs,
    ) -> None:
        """Processes a complete FCS measurement (multiple files)."""

        print("\nLoading FPGA data from hard drive:", end=" ")

        file_paths = file_utilities.prepare_file_paths(Path(file_path_template), file_selection)
        n_files = len(file_paths)
        *_, self.template = Path(file_path_template).parts
        self.name_on_disk = re.sub("\\*", "", re.sub("_[*]", "", self.template))

        for idx, file_path in enumerate(file_paths):
            # Loading file from disk
            print(f"Loading file No. {idx+1}/{n_files}: '{file_path}'...", end=" ")
            try:
                file_dict = file_utilities.load_file_dict(file_path)
            except OSError:
                print(f"File '{idx+1}' has not been fully downloaded from cloud. Skipping...\n")
                continue
            else:
                print("Done.")

            # Processing data
            p = self.process_data(file_dict, idx, verbose=True, **kwargs)

            # Appending data to self
            if p is not None:
                p.file_path = file_path
                self.data.append(p)
                print(f"Finished processing file No. {idx+1}\n")

        # count the files and ensure there's at least one file
        self.n_files = len(self.data)
        if not self.n_files:
            raise RuntimeError(
                f"Loading FPGA data catastrophically failed ({n_files}/{n_files} files skipped)."
            )

        if self.scan_type == "angular_scan":
            # aggregate images and ROIs for angular sFCS
            self.scan_images_dstack = np.dstack(tuple(p.image for p in self.data))
            self.roi_list = [p.roi for p in self.data]

        # calculate average count rate
        self.avg_cnt_rate_khz = sum([p.avg_cnt_rate_khz for p in self.data]) / len(self.data)

        if self.duration_min is None:
            # calculate duration if not supplied
            self.duration_min = (
                np.mean([np.diff(p.runtime).sum() for p in self.data]) / self.laser_freq_hz / 60
            )
            print(f"Calculating duration (not supplied): {self.duration_min:.1f} min\n")

        print(f"Finished loading FPGA data ({len(self.data)}/{n_files} files used).\n")

        # plotting of scan image and ROI
        if should_plot:
            print("Displaying scan images...", end=" ")
            with Plotter(subplots=(1, n_files), should_force_aspect=True) as axes:
                if not hasattr(
                    axes, "size"
                ):  # if axes is not an ndarray (only happens if reding just one file)
                    axes = np.array([axes])
                for idx, (ax, image, roi) in enumerate(
                    zip(axes, np.moveaxis(self.scan_images_dstack, -1, 0), self.roi_list)
                ):
                    ax.set_title(f"file #{idx+1} of '{self.name}' measurement")
                    ax.set_xlabel("Pixel Number")
                    ax.set_ylabel("Line Number")
                    ax.imshow(image)
                    ax.plot(roi["col"], roi["row"], color="white")
            print("Done.\n")

    def process_data(self, file_dict: dict, idx: int = 0, **kwargs) -> TDCPhotonData:
        """Doc."""

        full_data = file_dict["full_data"]

        if idx == 0:
            self.after_pulse_param = file_dict["system_info"]["after_pulse_param"]
            self.laser_freq_hz = int(full_data["laser_freq_mhz"] * 1e6)
            self.fpga_freq_hz = int(full_data["fpga_freq_mhz"] * 1e6)
            with suppress(KeyError):
                self.duration_min = full_data["duration_s"] / 60

        # Circular sFCS
        if full_data.get("circle_speed_um_s"):
            self.scan_type = "circular_scan"
            self.v_um_ms = full_data["circle_speed_um_s"] * 1e-3  # to um/ms
            raise NotImplementedError("Circular scan analysis not yet implemented...")

        # Angular sFCS
        elif full_data.get("angular_scan_settings"):
            if idx == 0:
                self.scan_type = "angular_scan"
                self.angular_scan_settings = full_data["angular_scan_settings"]
                self.LINE_END_ADDER = 1000
            return self.process_angular_scan_data(full_data, idx, **kwargs)

        # FCS
        else:
            self.scan_type = "static"
            return self.process_static_data(full_data, idx, **kwargs)

    def process_static_data(self, full_data, idx, **kwargs) -> TDCPhotonData:
        """
        Processes a single static FCS data file ('full_data').
        Returns the processed results as a 'TDCPhotonData' object.
        '"""

        p = self.convert_fpga_data_to_photons(
            full_data["byte_data"],
            version=full_data["version"],
            locate_outliers=True,
            **kwargs,
        )

        p.file_num = idx + 1
        p.avg_cnt_rate_khz = full_data["avg_cnt_rate_khz"]

        return p

    def process_angular_scan_data(
        self,
        full_data,
        idx,
        should_fix_shift=True,
        roi_selection="auto",
        **kwargs,
    ) -> TDCPhotonData:
        """
        Processes a single angular sFCS data file ('full_data').
        Returns the processed results as a 'TDCPhotonData' object.
        '"""

        p = self.convert_fpga_data_to_photons(
            full_data["byte_data"], version=full_data["version"], verbose=True
        )

        p.file_num = idx + 1
        p.avg_cnt_rate_khz = full_data["avg_cnt_rate_khz"]

        angular_scan_settings = full_data["angular_scan_settings"]
        linear_part = angular_scan_settings["linear_part"].round().astype(np.uint16)
        self.v_um_ms = angular_scan_settings["actual_speed_um_s"] * 1e-3
        sample_freq_hz = int(angular_scan_settings["sample_freq_hz"])
        ppl_tot = int(angular_scan_settings["points_per_line_total"])
        n_lines = int(angular_scan_settings["n_lines"])

        print("Converting angular scan to image...", end=" ")

        runtime = p.runtime
        cnt, n_pix_tot, n_pix, line_num = _convert_angular_scan_to_image(
            runtime, self.laser_freq_hz, sample_freq_hz, ppl_tot, n_lines
        )

        if should_fix_shift:
            pix_shift = _fix_data_shift(cnt)
            runtime = p.runtime + pix_shift * round(self.laser_freq_hz / sample_freq_hz)
            cnt, n_pix_tot, n_pix, line_num = _convert_angular_scan_to_image(
                runtime, self.laser_freq_hz, sample_freq_hz, ppl_tot, n_lines
            )
            print(f"Shifted by {pix_shift} pixels. Done.")
        else:
            print("Done.")

        # invert every second line
        cnt[1::2, :] = np.flip(cnt[1::2, :], 1)

        print("ROI selection: ", end=" ")

        if roi_selection == "auto":
            print("automatic. Thresholding and smoothing...", end=" ")
            try:
                bw = _threshold_and_smooth(cnt)
            except ValueError:
                print("Thresholding failed, skipping file.")
                return None
        else:
            raise ValueError(
                f"roi_selection='{roi_selection}' is not supported. Only 'auto' is, at the moment."
            )

        # cut edges
        bw_temp = np.full(bw.shape, False, dtype=bool)
        bw_temp[:, linear_part] = bw[:, linear_part]
        bw = bw_temp

        # discard short and fill long rows
        m2 = np.sum(bw, axis=1)
        bw[m2 < 0.5 * m2.max(), :] = False

        while self.LINE_END_ADDER < bw.shape[0]:
            self.LINE_END_ADDER *= 10

        print("Building ROI...", end=" ")

        line_starts = []
        line_stops = []
        line_start_lables = []
        line_stop_labels = []
        roi: Dict[str, deque] = {"row": deque([]), "col": deque([])}

        bw_rows, _ = bw.shape
        for row_idx in range(bw_rows):
            nonzero_row_idxs = bw[row_idx, :].nonzero()[0]
            if nonzero_row_idxs.size != 0:  # if bw row has nonzero elements
                # set mask to True between non-zero edges of row
                left_edge, right_edge = nonzero_row_idxs[0], nonzero_row_idxs[-1]
                bw[row_idx, left_edge:right_edge] = True
                # add row to ROI
                roi["row"].appendleft(row_idx)
                roi["col"].appendleft(left_edge)
                roi["row"].append(row_idx)
                roi["col"].append(right_edge)

                line_starts_new_idx = np.ravel_multi_index((row_idx, left_edge), bw.shape)
                line_starts_new = list(range(line_starts_new_idx, n_pix_tot[-1], bw.size))
                line_stops_new_idx = np.ravel_multi_index((row_idx, right_edge), bw.shape)
                line_stops_new = list(range(line_stops_new_idx, n_pix_tot[-1], bw.size))

                line_start_lables.extend([-row_idx for elem in range(len(line_starts_new))])
                line_stop_labels.extend(
                    [(-row_idx - self.LINE_END_ADDER) for elem in range(len(line_stops_new))]
                )
                line_starts.extend(line_starts_new)
                line_stops.extend(line_stops_new)

        try:
            # repeat first point to close the polygon
            roi["row"].append(roi["row"][0])
            roi["col"].append(roi["col"][0])
        except IndexError:
            print("ROI is empty (need to figure out the cause). Skipping file.", end=" ")
            return None

        # convert lists/deques to numpy arrays
        roi = {key: np.array(val, dtype=np.uint16) for key, val in roi.items()}
        line_start_lables = np.array(line_start_lables, dtype=np.int16)
        line_stop_labels = np.array(line_stop_labels, dtype=np.int16)
        line_starts = np.array(line_starts, dtype=np.int64)
        line_stops = np.array(line_stops, dtype=np.int64)

        print("Done.")

        runtime_line_starts: np.ndarray = line_starts * round(self.laser_freq_hz / sample_freq_hz)
        runtime_line_stops: np.ndarray = line_stops * round(self.laser_freq_hz / sample_freq_hz)

        runtime = np.hstack((runtime_line_starts, runtime_line_stops, runtime))
        sorted_idxs = np.argsort(runtime)
        p.runtime = runtime[sorted_idxs]
        p.line_num = np.hstack(
            (
                line_start_lables,
                line_stop_labels,
                line_num * bw[line_num, n_pix].flatten(),
            )
        )[sorted_idxs]
        p.coarse = np.hstack(
            (
                np.full(runtime_line_starts.size, self.NAN_PLACEBO, dtype=np.int16),
                np.full(runtime_line_stops.size, self.NAN_PLACEBO, dtype=np.int16),
                p.coarse,
            )
        )[sorted_idxs]
        p.coarse2 = np.hstack(
            (
                np.full(runtime_line_starts.size, self.NAN_PLACEBO, dtype=np.int16),
                np.full(runtime_line_stops.size, self.NAN_PLACEBO, dtype=np.int16),
                p.coarse2,
            )
        )[sorted_idxs]
        p.fine = np.hstack(
            (
                np.full(runtime_line_starts.size, self.NAN_PLACEBO, dtype=np.int16),
                np.full(runtime_line_stops.size, self.NAN_PLACEBO, dtype=np.int16),
                p.fine,
            )
        )[sorted_idxs]

        p.image = cnt
        p.roi = roi

        # reverse rows again
        bw[1::2, :] = np.flip(bw[1::2, :], 1)
        p.bw_mask = bw

        # get image line correlation to subtract trends
        img = p.image * p.bw_mask

        p.image_line_corr = _line_correlations(img, p.bw_mask, roi, sample_freq_hz)

        return p

    def correlate_and_average(self, **kwargs):
        """High level function for correlating and averaging any data."""

        CF = self.correlate_data(**kwargs)
        CF.average_correlation(**kwargs)

    @file_utilities.rotate_data_to_disk
    def correlate_data(self, cf_name=None, **kwargs):
        """
        High level function for correlating any type of data (e.g. static, angular scan...)
        Returns a 'CorrFunc' object.
        Data attribute is possibly rotated from/to disk.
        """

        if self.scan_type == "angular_scan":
            CF = self.correlate_angular_scan_data(**kwargs)
        elif self.scan_type == "circular_scan":
            CF = self.correlate_circular_scan_data(**kwargs)
        elif self.scan_type == "static":
            CF = self.correlate_static_data(**kwargs)
        else:
            raise NotImplementedError(
                f"Correlating data of type '{self.scan_type}' is not implemented."
            )

        if cf_name is not None:
            self.cf[cf_name] = CF
        else:
            self.cf[f"gate {CF.gate_ns}"] = CF

        return CF

    def correlate_static_data(
        self,
        gate_ns=LimitRange(0, np.inf),
        run_duration=None,
        min_time_frac=0.5,
        n_runs_requested=60,
        subtract_afterpulse=True,
        verbose=False,
        **kwargs,
    ) -> CorrFunc:
        """Correlates data for static FCS. Returns a CorrFunc object"""

        if verbose:
            print(f"Correlating {self.template}:", end=" ")

        if run_duration is None:  # auto determination of run duration
            if len(self.cf) > 0:  # read run_time from the last calculated correlation function
                run_duration = next(reversed(self.cf.values())).run_duration
            else:  # auto determine
                total_duration_estimate = 0
                for p in self.data:
                    time_stamps = np.diff(p.runtime).astype(np.int32)
                    mu = np.median(time_stamps) / np.log(2)
                    total_duration_estimate = (
                        total_duration_estimate + mu * len(p.runtime) / self.laser_freq_hz
                    )
                run_duration = total_duration_estimate / n_runs_requested

        self.min_duration_frac = min_time_frac
        duration = []

        SC = SoftwareCorrelator()
        CF = CorrFunc(gate_ns)
        CF.run_duration = run_duration

        with CorrFuncAccumulator(CF) as Accumulator:
            for p in self.data:

                if verbose:
                    print(f"({p.file_num})", end=" ")

                # Ignore short segments (default is below half the run_duration)
                for se_idx, (se_start, se_end) in enumerate(p.all_section_edges):
                    segment_time = (p.runtime[se_end] - p.runtime[se_start]) / self.laser_freq_hz
                    if segment_time < min_time_frac * run_duration:
                        if verbose:
                            print(
                                f"Skipping segment {se_idx} of file {p.file_num} - too short ({segment_time:.2f}s).",
                                end=" ",
                            )
                        CF.skipped_duration += segment_time
                        continue

                    runtime = p.runtime[se_start : se_end + 1]

                    # Gating
                    if hasattr(p, "delay_time"):
                        delay_time = p.delay_time[se_start : se_end + 1]
                        j_gate = gate_ns.idxs_in_limits(delay_time)
                        runtime = runtime[j_gate]
                    #                        delay_time = delay_time[j_gate]  # TODO: why is this not used anywhere?
                    elif gate_ns != (0, np.inf):
                        raise RuntimeError("For gating, TDC must first be calibrated!")

                    # split into segments of approx time of run_duration
                    n_splits = div_ceil(segment_time, run_duration)
                    splits = np.linspace(0, runtime.size, n_splits + 1, dtype=np.int32)
                    time_stamps = np.diff(runtime).astype(np.int32)

                    for k in range(n_splits):

                        ts_split = time_stamps[splits[k] : splits[k + 1]]
                        duration.append(ts_split.sum() / self.laser_freq_hz)
                        SC.correlate(
                            ts_split,
                            CorrelatorType.PH_DELAY_CORRELATOR,
                            timebase_ms=1000 / self.laser_freq_hz,
                        )  # time base of 20MHz to ms

                        if len(CF.lag) < len(SC.lag):
                            CF.lag = SC.lag
                            CF.afterpulse = self.calculate_afterpulse(gate_ns, CF.lag)

                        # subtract afterpulse
                        if subtract_afterpulse:
                            SC.cf_cr = (
                                SC.countrate * SC.corrfunc - CF.afterpulse[: SC.corrfunc.size]
                            )
                        else:
                            SC.cf_cr = SC.countrate * SC.corrfunc

                        # Append new correlation functions
                        Accumulator.accumulate(SC)

        CF.total_duration = sum(duration)

        if verbose:
            if CF.skipped_duration:
                try:
                    skipped_ratio = CF.skipped_duration / CF.total_duration
                except RuntimeWarning:
                    # CF.total_duration = 0
                    print("Whole measurement was skipped! Something's wrong...", end=" ")
                else:
                    print(f"Skipped/total duration: {skipped_ratio:.1%}", end=" ")
            print("- Done.")

        return CF

    def correlate_angular_scan_data(
        self,
        gate_ns=LimitRange(0, np.inf),
        min_time_frac=0.5,
        subtract_bg_corr=True,
        subtract_afterpulse=True,
        **kwargs,
    ) -> CorrFunc:
        """Correlates data for angular scans. Returns a CorrFunc object"""

        print(f"Correlating angular scan data '{self.template}':", end=" ")

        self.min_duration_frac = min_time_frac  # TODO: not used?
        duration = []

        SC = SoftwareCorrelator()
        CF = CorrFunc(gate_ns)

        with CorrFuncAccumulator(CF) as Accumulator:
            for p in self.data:
                print(f"({p.file_num})", end=" ")
                line_num = p.line_num
                min_line, max_line = line_num[line_num > 0].min(), line_num.max()
                if hasattr(p, "delay_time"):  # if measurement quacks as gated
                    j_gate = gate_ns.idxs_in_limits(p.delay_time) | (p.delay_time == np.nan)
                    runtime = p.runtime[j_gate]
                    #                    delay_time = delay_time[j_gate] # TODO: not used?
                    line_num = p.line_num[j_gate]

                    # TESTESTEST -  recalculate (now gated) background correlation
                    gated_image, *_ = _convert_angular_scan_to_image(
                        runtime,
                        self.laser_freq_hz,
                        self.angular_scan_settings["sample_freq_hz"],
                        self.angular_scan_settings["points_per_line_total"],
                        self.angular_scan_settings["n_lines"],
                    )
                    gated_image[1::2, :] = np.flip(gated_image[1::2, :], 1)
                    gated_image *= p.bw_mask
                    p.gated_image_line_corr = _line_correlations(
                        gated_image, p.bw_mask, p.roi, self.angular_scan_settings["sample_freq_hz"]
                    )

                elif gate_ns != (0, np.inf):
                    raise RuntimeError(
                        f"A gate '{gate_ns}' was specified for uncalibrated TDC data."
                    )
                else:
                    runtime = p.runtime

                time_stamps = np.diff(runtime).astype(np.int32)
                for line_idx, j in enumerate(range(min_line, max_line + 1)):
                    valid = (line_num == j).astype(np.int8)
                    valid[line_num == -j] = -1
                    valid[line_num == -j - self.LINE_END_ADDER] = -2
                    # both photons separated by time-stamp should belong to the line
                    valid = valid[1:]

                    # remove photons from wrong lines
                    timest = time_stamps[valid != 0]
                    valid = valid[valid != 0]

                    if not valid.size:
                        print(f"No valid photons in line {j}. Skipping.")
                        continue

                    # check that we start with the line beginning and not its end
                    if valid[0] != -1:
                        # remove photons till the first found beginning
                        j_start = np.where(valid == -1)[0]
                        if len(j_start) > 0:
                            timest = timest[j_start[0] :]
                            valid = valid[j_start[0] :]

                        # check that we stop with the line ending and not its beginning
                    if valid[-1] != -2:
                        # remove photons till the last found ending
                        j_start = np.where(valid == -1)[0]
                        if len(j_start) > 0:
                            timest = timest[j_start[0] :]
                            valid = valid[j_start[0] :]

                    # the first photon in line measures the time from line start and the line end (-2) finishes the duration of the line
                    dur = timest[(valid == 1) | (valid == -2)].sum() / self.laser_freq_hz
                    duration.append(dur)
                    ts_split = np.vstack((timest, valid))

                    SC.correlate(
                        ts_split,
                        CorrelatorType.PH_DELAY_CORRELATOR_LINES,
                        # time base of 20MHz to ms
                        timebase_ms=1000 / self.laser_freq_hz,
                    )

                    # remove background correlation
                    if subtract_bg_corr:
                        # TESTESTEST -  recalculate (now gated) background correlation
                        if hasattr(p, "delay_time") and kwargs.get(
                            "should_use_gated_background", False
                        ):  # if measurement quacks as gated
                            image_line_corr = p.gated_image_line_corr
                        else:
                            image_line_corr = p.image_line_corr
                        bg_corr = np.interp(
                            SC.lag,
                            image_line_corr[line_idx]["lag"],
                            image_line_corr[line_idx]["corrfunc"],
                            right=0,
                        )
                    else:
                        bg_corr = 0
                    SC.corrfunc -= bg_corr

                    if len(CF.lag) < len(SC.lag):
                        CF.lag = SC.lag
                        CF.afterpulse = self.calculate_afterpulse(gate_ns, CF.lag)

                    # subtract afterpulse
                    if subtract_afterpulse:
                        SC.cf_cr = SC.countrate * SC.corrfunc - CF.afterpulse[: SC.corrfunc.size]
                    else:
                        SC.cf_cr = SC.countrate * SC.corrfunc

                    # Append new correlation functions
                    Accumulator.accumulate(SC)

        CF.vt_um = self.v_um_ms * CF.lag
        CF.total_duration = sum(duration)

        print("- Done.")

        return CF

    def correlate_circular_scan_data(
        self, gate_ns=LimitRange(0, np.inf), min_time_frac=0.5, subtract_bg_corr=True, **kwargs
    ) -> CorrFunc:
        """Correlates data for circular scans. Returns a CorrFunc object"""

        raise NotImplementedError("Correlation of circular scan data not yet implemented.")

    def plot_correlation_functions(
        self,
        x_field="lag",
        y_field="normalized",
        x_scale="log",
        y_scale="linear",
        parent_figure=None,
        parent_ax=None,
        ylim=(-0.20, 1.4),
        plot_kwargs={},
        **kwargs,
    ) -> List[str]:
        """Doc."""

        with Plotter(
            parent_figure=parent_figure,
            parent_ax=parent_ax,
            super_title=f"'{self.name}' - ACFs",
        ) as ax:
            legend_labels = []
            for cf_name, cf in self.cf.items():
                cf.plot_correlation_function(
                    parent_ax=ax,
                    x_field=x_field,
                    y_field=y_field,
                    x_scale=x_scale,
                    y_scale=y_scale,
                    ylim=ylim,
                    plot_kwargs=plot_kwargs,
                    **kwargs,
                )
                legend_labels.append(cf_name)
            ax.legend(legend_labels)

        return legend_labels

    def calculate_afterpulse(self, gate_ns: LimitRange, lag: np.ndarray) -> np.ndarray:
        """Doc."""

        gate_to_laser_pulses = min([1.0, gate_ns.interval() * self.laser_freq_hz / 1e9])
        if self.after_pulse_param[0] == "multi_exponent_fit":
            # work with any number of exponents
            beta = self.after_pulse_param[1]
            afterpulse = gate_to_laser_pulses * np.dot(
                beta[::2], np.exp(-np.outer(beta[1::2], lag))
            )
        elif self.after_pulse_param[0] == "exponent_of_polynom_of_log":  # for old MATLAB files
            beta = self.after_pulse_param[1]
            if lag[0] == 0:
                lag[0] = np.nan
            afterpulse = gate_to_laser_pulses * np.exp(np.polyval(beta, np.log(lag)))

        return afterpulse

    def dump_or_load_data(self, should_load: bool):
        """
        Load or save the 'data' attribute.
        (relieve RAM - important during multiple-experiment analysis)
        """

        with suppress(AttributeError):
            # AttributeError - name_on_disk is not defined (happens when doing alignment, for example)
            if should_load:  # loading data
                if self.is_data_dumped:
                    logging.debug(
                        f"Loading dumped data '{self.name_on_disk}' from '{self.DUMP_PATH}'."
                    )
                    with suppress(FileNotFoundError):
                        self.data = file_utilities.load_file(self.DUMP_PATH / self.name_on_disk)
                        self.is_data_dumped = False
            else:  # dumping data
                is_saved = file_utilities.save_object_to_disk(
                    self.data,
                    self.DUMP_PATH / self.name_on_disk,
                    size_limits_mb=self.SIZE_LIMITS_MB,
                    compression_method=None,
                )
                if is_saved:
                    self.data = []
                    self.is_data_dumped = True
                    logging.debug(f"Dumped data '{self.name_on_disk}' to '{self.DUMP_PATH}'.")


class ImageSFCSMeasurement(TDCPhotonDataMixin, CountsImageMixin):
    """Doc."""

    def __init__(self):
        pass

    def read_image_data(self, file_path, **kwargs) -> None:
        """Doc."""

        file_dict = file_utilities.load_file_dict(file_path)
        self.process_data(file_dict, **kwargs)

    def process_data(self, file_dict: dict, **kwargs) -> None:
        """Doc."""

        # store relevant attributes (add more as needed)
        self.laser_mode = file_dict.get("laser_mode")
        self.scan_params = file_dict.get("scan_params")

        # Get ungated image (excitation or sted)
        self.image_data = self.create_image_stack_data(file_dict)

        # gating stuff (TDC) - not yet implemented
        self.data = None


class SFCSExperiment:
    """Doc."""

    def __init__(self, name=None):
        self.confocal = SolutionSFCSMeasurement(name="confocal")
        self.sted = SolutionSFCSMeasurement(name="STED")
        self.name = name

    def load_experiment(
        self,
        confocal_template: Union[str, Path] = None,
        sted_template: Union[str, Path] = None,
        should_plot=False,
        **kwargs,
    ):
        """Doc."""

        if confocal_template is None and sted_template is None:
            raise RuntimeError("Can't load experiment with no measurements!")

        if confocal_template is not None:
            self.load_measurement(
                meas_type="confocal",
                file_path_template=confocal_template,
                should_plot=should_plot,
                **kwargs,
            )
        if sted_template is not None:
            self.load_measurement(
                meas_type="sted",
                file_path_template=sted_template,
                should_plot=should_plot,
                **kwargs,
            )

        if should_plot:
            super_title = f"Experiment '{self.name}' - All ACFs"
            with Plotter(subplots=(1, 2), super_title=super_title) as axes:
                self.plot_correlation_functions(
                    parent_ax=axes[0],
                    x_field="vt_um",
                    y_field="avg_cf_cr",
                    x_scale="log",
                    y_scale="linear",
                    xlim=None,  # autoscale x axis
                )

                self.plot_correlation_functions(
                    parent_ax=axes[1],
                    x_field="vt_um",
                    y_field="normalized",
                    x_scale="linear",
                    y_scale="linear",
                    xlim=(0, 1),
                )

    def load_measurement(
        self,
        meas_type: str,
        file_path_template: Union[str, Path],
        should_plot: bool = False,
        plot_kwargs: dict = {},
        **kwargs,
    ):

        measurement = getattr(self, meas_type)
        if "should_fix_shift" in kwargs:
            should_fix_shift = kwargs["should_fix_shift"]
            kwargs.pop("should_fix_shift")
        else:
            should_fix_shift = True

        if "cf_name" not in kwargs:
            if meas_type == "confocal":
                kwargs["cf_name"] = "Confocal"
            else:  # sted
                kwargs["cf_name"] = "CW STED"
        cf_name = kwargs["cf_name"]

        file_selection = kwargs.get(f"{meas_type}_file_selection")

        measurement.read_fpga_data(
            file_path_template,
            should_fix_shift=should_fix_shift,
            should_plot=should_plot,
            file_selection=file_selection,
            **kwargs,
        )

        if (x_field := kwargs.get("x_field")) is None:
            if measurement.scan_type == "static":
                x_field = "lag"
            else:  # angular or circular scan
                x_field = "vt_um"

        measurement.correlate_and_average(**kwargs)

        ignoresForPlot = {"cf_name"}
        for paramName in ignoresForPlot:
            kwargs.pop(paramName)

        if should_plot:
            super_title = f"'{self.name}' Experiment\n'{measurement.name}' Measurement - ACFs"
            with Plotter(super_title=super_title) as ax:
                measurement.cf[cf_name].plot_correlation_function(
                    parent_ax=ax,
                    y_field="average_all_cf_cr",
                    x_field=x_field,
                    plot_kwargs=plot_kwargs,
                )
                measurement.cf[cf_name].plot_correlation_function(
                    parent_ax=ax, y_field="avg_cf_cr", x_field=x_field, plot_kwargs=plot_kwargs
                )
                ax.legend(["average_all_cf_cr", "avg_cf_cr"])

    def calibrate_tdc(self, should_plot=False, **kwargs):
        """Doc."""

        if hasattr(self.confocal, "scan_type"):  # if confocal maesurement quacks as if loaded
            super_title = f"'{self.name}' Experiment\nTDC Calibration"
            with Plotter(subplots=(2, 4), super_title=super_title) as axes:
                self.confocal.calibrate_tdc(
                    should_plot=should_plot, parent_axes=axes[:, :2], **kwargs
                )
                kwargs["sync_coarse_time_to"] = self.confocal  # sync sted to confocal
                if hasattr(self.sted, "scan_type"):  # if sted maesurement quacks as if loaded
                    self.sted.calibrate_tdc(
                        should_plot=should_plot, parent_axes=axes[:, 2:], **kwargs
                    )

                if should_plot:
                    self.compare_lifetimes(**kwargs)

        else:
            raise RuntimeError(
                "Cannot calibrate TDC if confocal measurement is not loaded to the experiment!"
            )

    def compare_lifetimes(
        self,
        normalization_type="Per Time",
        **kwargs,
    ):
        """Doc."""

        if hasattr(self.confocal, "tdc_calib"):  # if TDC calibration performed
            super_title = f"'{self.name}' Experiment\nLifetime Comparison"
            with Plotter(super_title=super_title) as ax:
                self.confocal.compare_lifetimes(
                    "confocal",
                    compare_to=dict(STED=self.sted),
                    normalization_type=normalization_type,
                    parent_ax=ax,
                )

    def get_lifetime_parameters(
        self,
        sted_field="paraboloid",
        drop_idxs=[],
        fit_range=None,
        param_estimates=None,
        bg_range=LimitRange(40, 60),
        **kwargs,
    ):
        """Doc."""
        # TODO: WORK IN PROGRESS HERE

        conf = self.confocal
        sted = self.sted

        conf_hist = conf.TDCcalib.allHistNorm
        conf_t = conf.TDCcalib.t_Hist
        conf_t = conf_t(np.isfinite(conf_hist))
        conf_hist = conf_hist(np.isfinite(conf_hist))

        sted_hist = sted.TDCcalib.allHistNorm
        sted_t = sted.TDCcalib.t_Hist
        sted_t = sted_t(np.isfinite(sted_hist))
        sted_hist = sted_hist(np.isfinite(sted_hist))

        h_max, j_max = max(conf_hist)
        t_max = conf_t(j_max)

        beta0 = (h_max, 4, h_max * 1e-3)

        if fit_range is None:
            fit_range = LimitRange(t_max, 40)

        if param_estimates is None:
            param_estimates = beta0

        ConfParam = conf.DoFitLifeTimeHist("FitRange", fit_range, "fit param estimate", beta0)

        # remove background

        sted_bg = np.mean(sted_hist(bg_range.idxs_in_limits(sted_t)))
        conf_bg = np.mean(conf_hist(bg_range.idxs_in_limits(conf_t)))
        sted_hist = sted_hist - sted_bg
        conf_hist = conf_hist - conf_bg

        j = conf_t < 20
        t = conf_t(j)
        hist_ratio = conf_hist(j) / np.interp(t, sted_t, sted_hist, right=0)
        # TODO: test this translation of MATLAB's interp1d
        # hist_ratio = ExponentBGfit(ConfParam.beta, t)./sted_hist(J);
        if drop_idxs:
            j = np.setdiff1d(np.arange(1, len(t)), drop_idxs)
            t = t(j)
            hist_ratio = hist_ratio(j)

        # TODO: select relevant data points using rectangle selector from plot: https://matplotlib.org/stable/gallery/widgets/rectangle_selector.html
        #        h = figure;
        #        figure(h);
        #        plot(t, hist_ratio);
        #        title('Zoom into linear range and hit any key');
        #        figure(h);
        #        pause;
        j_selected = []

        LifeTimeParam = SimpleNamespace()
        if sted_field == "symmetric":
            # TODO: see this :https://medium.com/swlh/robust-regression-all-you-need-to-know-an-example-in-python-878081bafc0
            # and this: https://scikit-learn.org/stable/auto_examples/linear_model/plot_robust_fit.html
            # for robust linear regression

            pass

        #            p = robustfit(t(j_selected), hist_ratio(j_selected)); # to minimize effect of outliers
        #            p = (p(2), p(1)) # reversed order in robust fit
        #            plot(t, hist_ratio, 'o', t(j_selected), polyval(p, t(j_selected)));
        #
        #
        #            LifeTimeParam.LifeTime_ns = ConfParam.beta(2);
        #            LifeTimeParam.SigOverTau_STED = p(1);
        #            LifeTimeParam.Sig_STED = p(1)*LifeTimeParam.LifeTime_ns;
        #            LifeTimeParam.LaserPulse = (1 - p(2))/p(1);
        else:
            fit_param = fit_tools.curve_fit_lims(
                fit_name="ratio_of_lifetime_histograms",
                param_estimates=(2, 1, 1),
                xs=t(j_selected),
                ys=hist_ratio(j_selected),
                ys_errors=np.ones(j_selected.size),
                should_plot=True,
            )

            LifeTimeParam.LifeTime_ns = ConfParam["beta"][2]
            LifeTimeParam.Sig1overTau_STED = fit_param["beta"][1]
            LifeTimeParam.Sig2overTau_STED = fit_param["beta"][2]
            LifeTimeParam.Sig1_STED = fit_param["beta"][1] * LifeTimeParam.LifeTime_ns
            LifeTimeParam.Sig2_STED = fit_param["beta"][2] * LifeTimeParam.LifeTime_ns
            LifeTimeParam.LaserPulse = fit_param["beta"][3]

    def add_gate(self, gate_ns: Tuple[float, float], should_plot=False, **kwargs):
        """Doc."""

        if hasattr(self.sted, "scan_type"):  # if sted maesurement quacks as if loaded
            self.sted.correlate_and_average(gate_ns=gate_ns, **kwargs)
            if should_plot:
                self.plot_correlation_functions(**kwargs)
        else:
            raise RuntimeError(
                "Cannot add a gate if there's no STED measurement loaded to the experiment!"
            )

    def add_gates(self, gate_list: List[Tuple[float, float]], should_plot=False, **kwargs):
        """A convecience method for adding multiple gates."""

        if hasattr(self.sted, "scan_type"):  # if sted maesurement quacks as if loaded
            for gate_ns in gate_list:
                self.add_gate(gate_ns, **kwargs)

            if should_plot:
                self.plot_correlation_functions(**kwargs)
        else:
            raise RuntimeError(
                "Cannot add a gate if there's no STED measurement loaded to the experiment!"
            )

    def plot_correlation_functions(
        self,
        x_field="vt_um",
        y_field="normalized",
        x_scale="linear",
        y_scale="linear",
        parent_figure=None,
        parent_ax=None,
        xlim=(0, 1),
        ylim=(-0.20, 1.4),
        plot_kwargs={},
        **kwargs,
    ):
        if self.confocal.scan_type == "static":
            x_field = "lag"

        if y_field in {"average_all_cf_cr", "avg_cf_cr"}:
            ylim = None  # will autoscale y

        super_title = f"'{self.name}' Experiment - All ACFs"
        with Plotter(
            parent_figure=parent_figure, parent_ax=parent_ax, super_title=super_title
        ) as ax:
            confocal_legend_labels = self.confocal.plot_correlation_functions(
                parent_ax=ax,
                x_field=x_field,
                y_field=y_field,
                x_scale=x_scale,
                y_scale=y_scale,
                xlim=xlim,
                ylim=ylim,
                plot_kwargs=plot_kwargs,
            )
            sted_legend_labels = self.sted.plot_correlation_functions(
                parent_ax=ax,
                x_field=x_field,
                y_field=y_field,
                x_scale=x_scale,
                y_scale=y_scale,
                xlim=xlim,
                ylim=ylim,
                plot_kwargs=plot_kwargs,
            )
            ax.legend(confocal_legend_labels + sted_legend_labels)


def _convert_angular_scan_to_image(runtime, laser_freq_hz, sample_freq_hz, ppl_tot, n_lines):
    """utility function for opening Angular Scans"""

    n_pix_tot = runtime * sample_freq_hz // laser_freq_hz
    # to which pixel photon belongs
    n_pix = np.mod(n_pix_tot, ppl_tot)
    line_num_tot = np.floor_divide(n_pix_tot, ppl_tot)
    # one more line is for return to starting positon
    line_num = np.mod(line_num_tot, n_lines + 1).astype(np.int16)

    img = np.empty((n_lines + 1, ppl_tot), dtype=np.uint16)
    bins = np.arange(-0.5, ppl_tot)
    for j in range(n_lines + 1):
        img[j, :], _ = np.histogram(n_pix[line_num == j], bins=bins)

    return img, n_pix_tot, n_pix, line_num


def _get_best_pix_shift(img: np.ndarray, min_shift, max_shift) -> int:
    """Doc."""

    score = np.empty(shape=(max_shift - min_shift), dtype=np.uint32)
    pix_shifts = np.arange(min_shift, max_shift)
    for idx, pix_shift in enumerate(range(min_shift, max_shift)):
        rolled_img = np.roll(img, pix_shift).astype(np.uint32)
        score[idx] = ((rolled_img[:-1:2, :] - np.fliplr(rolled_img[1::2, :])) ** 2).sum()
    return pix_shifts[score.argmin()]


def _fix_data_shift(cnt) -> int:
    """Doc."""

    print("Fixing line shift...", end=" ")

    height, width = cnt.shape

    min_pix_shift = -round(width / 2)
    max_pix_shift = min_pix_shift + width + 1
    pix_shift = _get_best_pix_shift(cnt, min_pix_shift, max_pix_shift)

    # Test if not stuck in local minimum (outer_half_sum > inner_half_sum)
    # OR if the 'return row' (the empty one) is not at the bottom for some reason
    # TODO: ask Oleg how the latter can happen
    rolled_cnt = np.roll(cnt, pix_shift)
    inner_half_sum = rolled_cnt[:, int(width * 0.25) : int(width * 0.75)].sum()
    outer_half_sum = rolled_cnt.sum() - inner_half_sum
    return_row_idx = rolled_cnt.sum(axis=1).argmin()

    if (outer_half_sum > inner_half_sum) or return_row_idx != height - 1:
        if return_row_idx != height - 1:
            print("Data is heavily shifted, check it out!", end=" ")
        min_pix_shift = -round(cnt.size / 2)
        max_pix_shift = min_pix_shift + cnt.size + 1
        pix_shift = _get_best_pix_shift(cnt, min_pix_shift, max_pix_shift)

    return pix_shift


def _threshold_and_smooth(img, otsu_classes=4, n_bins=256, disk_radius=2) -> np.ndarray:
    """Doc."""

    thresh = skimage.filters.threshold_multiotsu(
        skimage.filters.median(img).astype(np.float32), otsu_classes, nbins=n_bins
    )  # minor filtering of outliers
    cnt_dig = np.digitize(img, bins=thresh)
    plateau_lvl = np.median(img[cnt_dig == (otsu_classes - 1)])
    std_plateau = scipy.stats.median_absolute_deviation(img[cnt_dig == (otsu_classes - 1)])
    dev_cnt = img - plateau_lvl
    bw = dev_cnt > -std_plateau
    bw = scipy.ndimage.binary_fill_holes(bw)
    disk_open = skimage.morphology.selem.disk(radius=disk_radius)
    bw = skimage.morphology.opening(bw, selem=disk_open)
    return bw


def _line_correlations(image, bw_mask, roi, sampling_freq) -> list:
    """Returns a list of auto-correlations of the lines of an image"""

    image_line_corr = []
    for j in range(roi["row"].min(), roi["row"].max() + 1):
        line = image[j][bw_mask[j] > 0]
        try:
            c, lags = _auto_corr(line.astype(np.float64))
        except ValueError:
            print(f"Auto correlation of line #{j} has failed. Skipping.", end=" ")
        else:
            c = c / line.mean() ** 2 - 1
            c[0] -= 1 / line.mean()  # subtracting shot noise, small stuff really
            image_line_corr.append(
                {
                    "lag": lags * 1e3 / sampling_freq,  # in ms
                    "corrfunc": c,
                }
            )
    return image_line_corr


def _auto_corr(a):
    """Does correlation similar to Matlab xcorr, cuts positive lags, normalizes properly"""

    c = np.correlate(a, a, mode="full")
    c = c[c.size // 2 :]
    c = c / np.arange(c.size, 0, -1)
    lags = np.arange(c.size, dtype=np.uint16)

    return c, lags


def _calculate_weighted_avg(cf_cr, weights):
    """Doc."""

    tot_weights = weights.sum(0)
    # TODO: error handeling for the row below (zero division) - can/should it be detected beforehand?
    avg_cf_cr = (cf_cr * weights).sum(0) / tot_weights
    error_cf_cr = np.sqrt((weights ** 2 * (cf_cr - avg_cf_cr) ** 2).sum(0)) / tot_weights

    return avg_cf_cr, error_cf_cr
