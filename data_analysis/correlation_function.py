"""Data organization and manipulation."""

import logging
import multiprocessing as mp
import re
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Union

import numpy as np
import scipy
from sklearn import linear_model

from data_analysis.data_processing import (
    CountsImageMixin,
    TDCCalibration,
    TDCCalibrationGenerator,
    TDCPhotonData,
    TDCPhotonDataProcessor,
)
from data_analysis.software_correlator import CorrelatorType, SoftwareCorrelator
from utilities import file_utilities
from utilities.display import Plotter
from utilities.fit_tools import (
    FIT_NAME_DICT,
    FitParams,
    curve_fit_lims,
    multi_exponent_fit,
)
from utilities.helper import (
    EPS,
    Limits,
    extrapolate_over_noise,
    hankel_transform,
    unify_length,
)

# laser_pulse_tdc_calib = file_utilities.load_object("./data_analysis/laser_pulse_tdc_calib.pkl")


@dataclass
class StructureFactor:
    """Holds structure factor data"""

    # parameters
    n_interp_pnts: int
    r_max: float
    r_min: float
    g_min: float

    r: np.ndarray
    fr: np.ndarray
    fr_linear_interp: np.ndarray

    q: np.ndarray
    sq: np.ndarray
    sq_lin_intrp: np.ndarray
    sq_error: np.ndarray


@dataclass
class LifeTimeParams:
    """Doc."""

    lifetime_ns: float
    sigma_sted: Union[float, Tuple[float, float]]
    laser_pulse_delay_ns: float


class CorrFunc:
    """Doc."""

    # Initialize the software correlator once (for all instances)
    SC = SoftwareCorrelator()

    afterpulse: np.ndarray
    vt_um: np.ndarray
    cf_cr: np.ndarray
    g0: float

    def __init__(self, name: str, correlator_type: int, laser_freq_hz):
        self.name = name
        self.correlator_type = correlator_type
        self.laser_freq_hz = laser_freq_hz
        self.fit_params: Dict[str, FitParams] = dict()

    def correlate_measurement(
        self,
        time_stamp_split_list: List[np.ndarray],
        *args,
        is_verbose: bool = False,
        **kwargs,
    ) -> None:
        """Doc."""

        if is_verbose:
            print(f"Correlating {len(time_stamp_split_list)} splits -", end=" ")

        output = self.SC.correlate_list(
            time_stamp_split_list,
            self.correlator_type,
            timebase_ms=1000 / self.laser_freq_hz,
            is_verbose=is_verbose,
            **kwargs,
        )
        self.lag = max(output.lag_list, key=len)

        if is_verbose:
            print(". Processing correlator output...", end=" ")

        self._process_correlator_list_output(output, *args, **kwargs)

    def _process_correlator_list_output(
        self,
        corr_output,
        afterpulse_params,
        bg_corr_list,
        gate_ns=Limits(0, np.inf),
        should_subtract_afterpulsing: bool = False,
        external_afterpulsing: np.ndarray = None,
        **kwargs,
    ) -> None:
        """Doc."""

        # subtract background correlation
        for idx, bg_corr_dict in enumerate(bg_corr_list):
            bg_corr = np.interp(
                corr_output.lag_list[idx],
                bg_corr_dict["lag"],
                bg_corr_dict["corrfunc"],
                right=0,
            )
            corr_output.corrfunc_list[idx] -= bg_corr

        # subtract afterpulsing
        if should_subtract_afterpulsing:
            if external_afterpulsing is not None:
                self.afterpulse = external_afterpulsing
            else:
                self.afterpulse = calculate_calibrated_afterpulse(
                    self.lag, afterpulse_params, gate_ns, self.laser_freq_hz
                )

        # zero-pad and generate cf_cr
        lag_len = len(self.lag)
        n_corrfuncs = len(corr_output.corrfunc_list)
        shape = (n_corrfuncs, lag_len)
        self.corrfunc = np.empty(shape=shape, dtype=np.float64)
        self.weights = np.empty(shape=shape, dtype=np.float64)
        self.cf_cr = np.empty(shape=shape, dtype=np.float64)
        for idx in range(n_corrfuncs):
            pad_len = lag_len - len(corr_output.corrfunc_list[idx])
            self.corrfunc[idx] = np.pad(corr_output.corrfunc_list[idx], (0, pad_len))
            self.weights[idx] = np.pad(corr_output.weights_list[idx], (0, pad_len))
            try:
                self.cf_cr[idx] = corr_output.countrate_list[idx] * self.corrfunc[idx]
            except ValueError:  # Cross-correlation - countrate is a 2-tuple
                self.cf_cr[idx] = corr_output.countrate_list[idx].a * self.corrfunc[idx]
            with suppress(AttributeError):  # no .afterpulse attribute
                # ext. afterpulse might be shorter/longer
                self.cf_cr[idx] -= unify_length(self.afterpulse, lag_len)

        self.countrate_list = corr_output.countrate_list
        try:  # xcorr
            self.countrate_a = np.mean([countrate_pair.a for countrate_pair in self.countrate_list])
            self.countrate_b = np.mean([countrate_pair.b for countrate_pair in self.countrate_list])
        except AttributeError:  # autocorr
            self.countrate = np.mean([countrate for countrate in self.countrate_list])

    def average_correlation(
        self,
        rejection=2,
        reject_n_worst=None,
        norm_range=(1e-3, 2e-3),
        delete_list=[],
        should_plot=False,
        plot_kwargs={},
        **kwargs,
    ) -> None:
        """Doc."""

        self.rejection = rejection
        self.norm_range = Limits(norm_range)
        self.delete_list = delete_list
        self.average_all_cf_cr = (self.cf_cr * self.weights).sum(0) / self.weights.sum(0)
        self.median_all_cf_cr = np.median(self.cf_cr, axis=0)
        jj = Limits(self.norm_range.upper, 100).valid_indices(self.lag)  # work in the relevant part

        try:
            self.score = (
                (1 / np.var(self.cf_cr[:, jj], 0))
                * (self.cf_cr[:, jj] - self.median_all_cf_cr[jj]) ** 2
                / len(jj)
            ).sum(axis=1)
        except RuntimeWarning:  # division by zero
            # TODO: why does this happen?
            self.score = (
                (1 / (np.var(self.cf_cr[:, jj], 0) + EPS))
                * (self.cf_cr[:, jj] - self.median_all_cf_cr[jj]) ** 2
                / len(jj)
            ).sum(axis=1)
            print(
                f"Division by zero avoided by adding EPSILON={EPS}. Why does this happen (zero in variance)?"
            )

        total_n_rows, _ = self.cf_cr.shape

        if reject_n_worst not in {None, 0}:
            delete_list = np.argsort(self.score)[-reject_n_worst:]
        elif rejection is not None:
            delete_list = np.where(self.score >= self.rejection)[0]
            if len(delete_list) == total_n_rows:
                raise RuntimeError(
                    "All rows are in 'delete_list'! Increase the rejection limit. Ignoring."
                )

        # if 'reject_n_worst' and 'rejection' are both None, use supplied delete list. If no delete list is supplied, use all rows.
        self.j_bad = delete_list
        self.j_good = [row for row in range(total_n_rows) if row not in delete_list]

        self.avg_cf_cr, self.error_cf_cr = self._calculate_weighted_avg(
            self.cf_cr[self.j_good, :], self.weights[self.j_good, :]
        )

        self.avg_corrfunc, self.error_corrfunc = self._calculate_weighted_avg(
            self.corrfunc[self.j_good, :], self.weights[self.j_good, :]
        )

        j_t = self.norm_range.valid_indices(self.lag)

        try:
            self.g0 = (self.avg_cf_cr[j_t] / self.error_cf_cr[j_t] ** 2).sum() / (
                1 / self.error_cf_cr[j_t] ** 2
            ).sum()
        except RuntimeWarning:  # division by zero
            # TODO: why does this happen?
            self.g0 = (self.avg_cf_cr[j_t] / (self.error_cf_cr[j_t] + EPS) ** 2).sum() / (
                1 / (self.error_cf_cr[j_t] + EPS) ** 2
            ).sum()
            print(
                f"Division by zero avoided by adding EPSILON={EPS}. Why does this happen (zero in variance)?"
            )

        self.normalized = self.avg_cf_cr / self.g0
        self.error_normalized = self.error_cf_cr / self.g0

        if should_plot:
            self.plot_correlation_function(plot_kwargs=plot_kwargs)

    def _calculate_weighted_avg(
        self, cf_cr: np.ndarray, weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates weighted average and standard error of a 2D array (rows are correlation functions)."""

        tot_weights = weights.sum(0)
        try:
            avg_cf_cr = (cf_cr * weights).sum(0) / tot_weights
        except RuntimeWarning:  # division by zero - caused by zero total weights element/s
            # TODO: why does this happen?
            tot_weights += EPS
            avg_cf_cr = (cf_cr * weights).sum(0) / tot_weights
            print(
                f"Division by zero avoided by adding epsilon={EPS}. Why does this happen (zero total weight)?"
            )
        finally:
            error_cf_cr = np.sqrt((weights ** 2 * (cf_cr - avg_cf_cr) ** 2).sum(0)) / tot_weights

        return avg_cf_cr, error_cf_cr

    def plot_correlation_function(
        self,
        x_field="lag",
        y_field="avg_cf_cr",
        x_scale="log",
        y_scale="linear",
        **kwargs,
    ) -> None:

        x = getattr(self, x_field)
        y = getattr(self, y_field)
        if x_scale == "log":  # remove zero point data
            x, y = x[1:], y[1:]

        with Plotter(
            x_scale=x_scale,
            y_scale=y_scale,
            should_autoscale=True,
            **kwargs,
        ) as ax:
            ax.set_xlabel(x_field)
            ax.set_ylabel(y_field)
            ax.plot(x, y, "-", **kwargs.get("plot_kwargs", {}))

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
        bounds=None,
        max_nfev=int(1e4),
        should_plot=False,
        **kwargs,
    ) -> None:

        if fit_param_estimate is None:
            fit_param_estimate = (self.g0, 0.035, 30.0)

        x = getattr(self, x_field)
        y = getattr(self, y_field)
        error_y = getattr(self, y_error_field)
        if x_scale == "log":
            x = x[1:]  # remove zero point data
            y = y[1:]
            error_y = error_y[1:]

        # set bounds for parameters
        if fit_name == "diffusion_3d_fit":
            bounds = (  # a, tau, w_sq
                [0, 0, 0],
                [1e7, 10, np.inf],
            )

        self.fit_params[fit_name] = curve_fit_lims(
            FIT_NAME_DICT[fit_name],
            fit_param_estimate,
            x,
            y,
            error_y,
            x_limits=Limits(fit_range),
            should_plot=should_plot,
            plot_kwargs=dict(x_scale=x_scale, y_scale=y_scale),
            bounds=bounds,
            max_nfev=max_nfev,
            **kwargs,
        )

    def calculate_structure_factor(
        self,
        n_interp_pnts: int = 2048,
        r_max: float = 10.0,
        r_min: float = 0.05,
        g_min: float = 1e-2,
        n_robust: int = 2,
        **kwargs,
    ) -> StructureFactor:
        """Doc."""

        print(f"Calculating '{self.name}' structure factor...", end=" ")

        c0 = scipy.special.jn_zeros(0, n_interp_pnts)  # Bessel function zeros
        r = (r_max / c0[n_interp_pnts - 1]) * c0  # Radius vector

        # interpolation (extrapolate the noisy parts)
        #  Gaussian
        gauss_interp = extrapolate_over_noise(
            self.vt_um,
            self.normalized,
            r,
            Limits(r_min, r_max),
            Limits(g_min, np.inf),
            n_robust=n_robust,
            interp_type="gaussian",
        )

        #  linear
        lin_interp = extrapolate_over_noise(
            self.vt_um,
            self.normalized,
            r,
            Limits(r_min, r_max),
            Limits(g_min, np.inf),
            n_robust=n_robust,
            interp_type="linear",
        )

        # plot interpolations for testing
        with Plotter(
            super_title=f"{self.name.capitalize()}: Interpolation Testing",
            xlim=(0, self.vt_um[max(lin_interp.interp_idxs) + 5] ** 2),
            ylim=(1e-1, 1.3),
        ) as ax:
            ax.semilogy(self.vt_um ** 2, self.normalized, "x", label="Normalized")
            ax.semilogy(
                lin_interp.x_samples ** 2, lin_interp.y_samples, "o", label="Interpolation Sample"
            )
            ax.semilogy(r ** 2, gauss_interp.y_interp, label="Gaussian Intep/Extrap")
            ax.semilogy(r ** 2, lin_interp.y_interp, label="Linear Intep/Extrap")
            ax.legend()
            ax.set_xlabel("Displacement $(\\mu m^2)$")
            ax.set_ylabel("ACF (Normalized)")

        # Fourier Transform
        q, fq = hankel_transform(r, gauss_interp.y_interp)

        #  linear interpolated Fourier Transform
        _, fq_linear_interp = hankel_transform(r, lin_interp.y_interp)

        # TODO: show this to Oleg - can't figure out what's wrong
        #        #  Estimating error of transform
        #        print("Estimating error of transform...") # TESTESTEST
        #        fq_allfunc = np.empty(shape=(len(self.j_good), q.size))
        #        for idx, j in enumerate(self.j_good):
        #            sample_cf_cr = self.cf_cr[j, j_intrp]
        #            sample_cf_cr[sample_cf_cr <= 0] = 1e-3  # TODO: is that the best way to solve this?
        #            if n_robust and False: # TESTESTEST - CANELLED FOR NOW
        #                log_fr = robust_interpolation(
        #                    r ** 2, sample_vt_um ** 2, np.log(sample_cf_cr), n_robust
        #                )
        #            else:
        #                log_fr = np.interp(r ** 2, sample_vt_um ** 2, np.log(sample_cf_cr))
        #            fr = np.exp(log_fr)
        #            fr[np.nonzero(fr < 0)[0]] = 0
        #
        #            _, fq_allfunc[idx] = hankel_transform(r, fr)
        #
        #        fq_error = np.std(fq_allfunc, axis=0, ddof=1) / np.sqrt(len(self.j_good)) / self.g0
        fq_error = None

        self.structure_factor = StructureFactor(
            n_interp_pnts,
            r_max,
            r_min,
            g_min,
            r,
            gauss_interp.y_interp,
            lin_interp.y_interp,
            q,
            fq,
            fq_linear_interp,
            fq_error,
        )

        return self.structure_factor


class SolutionSFCSMeasurement:
    """Doc."""

    tdc_calib: TDCCalibration

    def __init__(self, name=""):
        self.name = name
        self.data: list = []  # list to hold the data of each file
        self.cf: dict = dict()
        self.xcf: dict = dict()
        self.is_data_dumped = False
        self.scan_type: str
        self.duration_min: float = None

    @file_utilities.rotate_data_to_disk(does_modify_data=True)
    def read_fpga_data(
        self,
        file_path_template: Union[str, Path],
        file_selection: str = "Use All",
        should_plot=False,
        **kwargs,
    ) -> None:
        """Processes a complete FCS measurement (multiple files)."""

        file_paths = file_utilities.prepare_file_paths(
            Path(file_path_template), file_selection, **kwargs
        )
        self.n_paths = len(file_paths)
        self.file_path_template = file_path_template
        *_, self.template = Path(file_path_template).parts
        self.name_on_disk = re.sub("\\*", "", re.sub("_[*]", "", self.template))

        print("\nLoading FPGA data from disk -")
        print(f"Template path: '{file_path_template}'")
        print(f"Files: {self.n_paths}, Selection: '{file_selection}'\n")

        # actual data processing
        self.data = self._process_all_data(file_paths, **kwargs)

        # count the files and ensure there's at least one file
        self.n_files = len(self.data)
        if not self.n_files:
            raise RuntimeError(
                f"Loading FPGA data catastrophically failed ({self.n_paths}/{self.n_paths} files skipped)."
            )

        # aggregate images and ROIs for sFCS
        if self.scan_type == "circle":
            self.scan_image = np.vstack(tuple(p.image for p in self.data))
            bg_corr_array = np.empty((len(self.data), self.scan_settings["samples_per_circle"]))
            for idx, p in enumerate(self.data):
                bg_corr_array[idx] = p.bg_line_corr[0]["corrfunc"]
            avg_bg_corr = bg_corr_array.mean(axis=0)
            self.bg_line_corr_list = [dict(lag=p.bg_line_corr[0]["lag"], corrfunc=avg_bg_corr)]

        if self.scan_type == "angular":
            # aggregate images and ROIs for angular sFCS
            self.scan_images_dstack = np.dstack(tuple(p.image for p in self.data))
            self.roi_list = [p.roi for p in self.data]
            self.bg_line_corr_list = [
                bg_line_corr
                for bg_file_corr in [p.bg_line_corr for p in self.data]
                for bg_line_corr in bg_file_corr
            ]

        else:  # static
            self.bg_line_corr_list = []

        # calculate average count rate
        self.avg_cnt_rate_khz = np.mean([p.avg_cnt_rate_khz for p in self.data])
        try:
            self.std_cnt_rate_khz = np.std([p.avg_cnt_rate_khz for p in self.data], ddof=1)
        except RuntimeWarning:  # single file
            self.std_cnt_rate_khz = 0.0

        calc_duration_mins = sum([p.duration_s for p in self.data]) / 60
        if self.duration_min is not None:
            if abs(calc_duration_mins - self.duration_min) > self.duration_min * 0.05:
                print(
                    f"Attention! calculated duration ({calc_duration_mins} mins) is significantly different than the set duration ({self.duration_min} min). Using calculated.\n"
                )
        else:
            print(f"Calculating duration (not supplied): {calc_duration_mins:.1f} mins\n")
        self.requested_duration_min = self.duration_min
        self.duration_min = calc_duration_mins

        # done with loading
        print(f"Finished loading FPGA data ({self.n_files}/{self.n_paths} files used).\n")

        # plotting of scan image and ROI
        if should_plot:
            print("Displaying scan images...", end=" ")
            if self.scan_type == "angular":
                with Plotter(
                    subplots=(1, self.n_files), fontsize=8, should_force_aspect=True
                ) as axes:
                    if not hasattr(
                        axes, "size"
                    ):  # if axes is not an ndarray (only happens if reading just one file)
                        axes = np.array([axes])
                    for file_idx, (ax, image, roi) in enumerate(
                        zip(axes, np.moveaxis(self.scan_images_dstack, -1, 0), self.roi_list)
                    ):
                        ax.set_title(f"file #{file_idx+1} of\n'{self.name}' measurement")
                        ax.set_xlabel("Pixel Number")
                        ax.set_ylabel("Line Number")
                        ax.imshow(image, interpolation="none")
                        ax.plot(roi["col"], roi["row"], color="white")
            elif self.scan_type == "circle":
                # TODO: FILL ME IN (plotting in jupyter notebook, same as above angular scan stuff)
                pass
            print("Done.\n")

    def _process_all_data(
        self,
        file_paths: List[Path],
        should_parallel_process: bool = False,
        **kwargs,
    ) -> List[TDCPhotonData]:
        """Doc."""

        self._get_general_properties(file_paths[0], **kwargs)

        # parellel processing
        if should_parallel_process and len(file_paths) > 20:
            N_CORES = mp.cpu_count() // 2 - 1  # /2 due to hyperthreading, -1 to leave one free
            func = partial(self.process_data_file, is_verbose=True, **kwargs)
            print(f"Parallel processing using {N_CORES} CPUs/processes.")
            with mp.get_context("spawn").Pool(N_CORES) as pool:
                data = list(pool.map(func, file_paths))

        # serial processing (default)
        else:
            data = []
            for file_path in file_paths:
                # Processing data
                p = self.process_data_file(file_path, is_verbose=True, **kwargs)
                print("Done.\n")
                # Appending data to self
                if p is not None:
                    data.append(p)

        # auto determination of run duration
        self.run_duration = sum([p.duration_s for p in data])

        return data

    def _get_general_properties(
        self, file_path: Path = None, file_dict: dict = None, **kwargs
    ) -> None:
        """Get general measurement properties from the first data file"""

        if file_path is not None:  # Loading file from disk
            file_dict = file_utilities.load_file_dict(file_path, **kwargs)

        full_data = file_dict["full_data"]

        self.afterpulse_params = file_dict["system_info"]["afterpulse_params"]
        self.detector_settings = full_data.get("detector_settings")
        self.delayer_settings = full_data.get("delayer_settings")
        self.laser_freq_hz = int(full_data["laser_freq_mhz"] * 1e6)
        self.pulse_period_ns = 1 / self.laser_freq_hz * 1e9
        self.fpga_freq_hz = int(full_data["fpga_freq_mhz"] * 1e6)
        with suppress(KeyError):
            self.duration_min = full_data["duration_s"] / 60

        # Detector Gate (calibrated - actual gate can be inferred from TDC calibration and compared)
        if self.delayer_settings is not None and getattr(self.detector_settings, "is_gated", False):
            lower_detector_gate_ns = (
                self.delayer_settings.effective_delay_ns - self.delayer_settings.sync_delay_ns
            )
        else:
            lower_detector_gate_ns = 0
        if self.detector_settings is not None and getattr(
            self.detector_settings, "is_gated", False
        ):
            self.gate_width_ns = self.detector_settings.gate_width_ns
            self.detector_gate_ns = Limits(
                lower_detector_gate_ns, lower_detector_gate_ns + self.gate_width_ns
            )
        else:
            self.gate_width_ns = 100
            self.detector_gate_ns = Limits(0, np.inf)

        # sFCS
        if scan_settings := full_data.get("scan_settings"):
            self.scan_type = scan_settings["pattern"]
            self.scan_settings = scan_settings
            self.v_um_ms = self.scan_settings["speed_um_s"] * 1e-3
            if self.scan_type == "circle":  # Circular sFCS
                self.ao_sampling_freq_hz = self.scan_settings.get("ao_sampling_freq_hz", int(1e4))
                self.diameter_um = self.scan_settings.get("diameter_um", 50)

        # FCS
        else:
            self.scan_type = "static"

    def process_data_file(
        self, file_path: Path = None, file_dict: dict = None, **kwargs
    ) -> TDCPhotonData:
        """Doc."""

        if not file_path and not file_dict:
            raise ValueError("Must supply either a valid path or a ")

        # if using existing file_dict - usually during alignment measurements
        if file_dict is not None:
            self._get_general_properties(file_dict=file_dict, **kwargs)
            file_idx = 1

        # File Data Loading
        if file_path is not None:  # Loading file from disk
            *_, template = file_path.parts
            try:
                file_idx = int(re.split("_(\\d+)\\.", template)[1])
            except IndexError:  # legacy template style
                file_idx = int(re.split("(\\d+)\\.", template)[1])
            print(
                f"Loading and processing file No. {file_idx} ({self.n_paths} files): '{template}'...",
                end=" ",
            )
            try:
                file_dict = file_utilities.load_file_dict(file_path)
            except FileNotFoundError:
                print(f"File '{file_path}' not found. Ignoring.")

        # File Processing
        data_processor = TDCPhotonDataProcessor(self.laser_freq_hz)
        return data_processor.process_data(file_dict["full_data"])

    @file_utilities.rotate_data_to_disk(does_modify_data=True)
    def calibrate_tdc(
        self, force_processing=True, should_plot=False, is_verbose=False, **kwargs
    ) -> None:
        """Doc."""

        if not force_processing and hasattr(self, "tdc_calib"):
            print(f"\n{self.name}: TDC calibration exists, skipping.")
            if should_plot:
                self.tdc_calib.plot_tdc_calibration()
            return

        if is_verbose:
            print(f"\n{self.name}: Calibrating TDC...", end=" ")

        # perform actual TDC calibration
        tdc_calib_gen = TDCCalibrationGenerator(
            self.laser_freq_hz,
            self.fpga_freq_hz,
        )
        self.tdc_calib = tdc_calib_gen.calibrate_tdc(self.data, self.scan_type, **kwargs)

        if should_plot:
            self.tdc_calib.plot_tdc_calibration()

        if is_verbose:
            print("Done.")

    def correlate_and_average(self, **kwargs) -> CorrFunc:
        """High level function for correlating and averaging any data."""

        CF = self.correlate_data(**kwargs)
        CF.average_correlation(**kwargs)
        return CF

    @file_utilities.rotate_data_to_disk()
    def correlate_data(  # NOQA C901
        self,
        cf_name=None,
        gate_ns=Limits(0, np.inf),
        afterpulsing_method="none",
        gating_mechanism="removal",
        external_afterpulse_params=None,
        external_afterpulsing=None,
        # TODO: gates should not be fixed - gate1 should start at the peak (or right before it) and the second should end before the detector width ends - these choices are important for normalization
        inherent_afterpulsing_gates=(Limits(2.5, 10), Limits(30, 90)),
        should_subtract_bg_corr=True,
        is_verbose=False,
        **kwargs,
    ) -> CorrFunc:
        """
        High level function for correlating any type of data (e.g. static, angular scan, circular scan...)
        Returns a 'CorrFunc' object.
        Data attribute is possibly rotated from/to disk.
        """

        if is_verbose:
            print("Preparing split data for correlator...", end=" ")

        # the following afterpulsing removal methods require TDC calibration before creating splits
        if afterpulsing_method in {"subtract inherent (xcorr)", "filter (lifetime)"}:
            if not hasattr(self, "tdc_calib"):  # calibrate TDC (if not already calibrated)
                if is_verbose:
                    print("(Calibrating TDC first...)", end=" ")
                self.calibrate_tdc(
                    should_dump_data=False, is_verbose=is_verbose
                )  # abort data rotation decorator

        # create list of split data for correlator
        dt_ts_split_list = self._prepare_xcorr_splits_dict(["AA"], is_verbose=is_verbose)["AA"]

        # Gating
        # Unite TDC gate and detector gate
        tdc_gate_ns = Limits(gate_ns)
        if tdc_gate_ns != Limits(0, np.inf) or self.detector_gate_ns != Limits(0, np.inf):
            if not hasattr(self, "tdc_calib"):  # calibrate TDC (if not already calibrated)
                self.calibrate_tdc(
                    should_dump_data=False, is_verbose=is_verbose
                )  # abort data rotation decorator
            effective_lower_gate_ns = max(tdc_gate_ns.lower, self.detector_gate_ns.lower)
            effective_upper_gate_ns = min(tdc_gate_ns.upper, self.detector_gate_ns.upper)
            gate_ns = Limits(effective_lower_gate_ns, effective_upper_gate_ns)

        # Apply gate and/or filter to correlator input
        is_filtered = (
            afterpulsing_method == "filter (lifetime)" or gating_mechanism == "binary filter"
        )
        corr_input_list = []
        if is_filtered:
            if afterpulsing_method == "filter (lifetime)":
                self.filter_matrix = self.tdc_calib.calculate_afterpulsing_filter(**kwargs)
            filter_input_list = []
        for dt_ts_split in dt_ts_split_list:
            split_delay_time = dt_ts_split[0]
            in_gate_idxs = gate_ns.valid_indices(split_delay_time)
            if is_filtered:
                split_filter = np.zeros((dt_ts_split.shape[1],), dtype=float)
                if gating_mechanism == "binary filter":
                    # TESTESTEST - testing the new filtering approach by applying it to gating
                    split_filter[in_gate_idxs] = 0.1
                if afterpulsing_method == "filter (lifetime)":
                    # create a filter for genuine fluorscene (ignoring afterpulsing)
                    bin_num = np.digitize(split_delay_time, self.tdc_calib.fine_bins)
                    valid_indxs = (
                        (bin_num > 0) & (bin_num < len(self.tdc_calib.fine_bins)) & in_gate_idxs
                    )
                    split_filter[valid_indxs] = self.filter_matrix[0][bin_num[valid_indxs] - 1]
                corr_input_list.append(np.squeeze(dt_ts_split[1:].astype(np.int32)))
                filter_input_list.append(split_filter)
            else:  # gating_mechanism == "removal"
                corr_input_list.append(np.squeeze(dt_ts_split[1:, in_gate_idxs].astype(np.int32)))

        kwargs["should_subtract_afterpulsing"] = afterpulsing_method in {
            "subtract calibrated",
            "subtract inherent (xcorr)",
        }

        if afterpulsing_method == "subtract inherent (xcorr)":
            # Calculate inherent afterpulsing by cross-correlating the fluorscent photons (peak) with the white-noise ones (tail)
            if external_afterpulsing is None:
                external_afterpulsing, _ = self.calculate_inherent_afterpulsing(
                    *inherent_afterpulsing_gates,
                    **kwargs,
                )
            else:
                print(
                    "Warning: external_afterpulsing was given and afterpulsing_method == 'subtract inherent (xcorr)'! Using external_afterpulsing...",
                    end=" ",
                )

        # choose correct correlator type
        if self.scan_type in {"static", "circle"}:
            if is_filtered:
                correlator_option = CorrelatorType.PH_DELAY_LIFETIME_CORRELATOR
            else:
                correlator_option = CorrelatorType.PH_DELAY_CORRELATOR
        elif self.scan_type == "angular":
            if is_filtered:
                correlator_option = CorrelatorType.PH_DELAY_LIFETIME_CORRELATOR_LINES
            else:
                correlator_option = CorrelatorType.PH_DELAY_CORRELATOR_LINES

        if is_verbose:
            print(
                f"Correlating {self.scan_type} data ({self.name} [{gate_ns} ns gating]):", end=" "
            )

        # Correlate data
        CF = CorrFunc(cf_name, correlator_option, self.laser_freq_hz)
        CF.correlate_measurement(
            corr_input_list,
            external_afterpulse_params
            if external_afterpulse_params is not None
            else self.afterpulse_params,
            getattr(self, "bg_line_corr_list", []) if should_subtract_bg_corr else [],
            is_verbose=is_verbose,
            external_afterpulsing=external_afterpulsing,
            gate_ns=gate_ns,
            list_of_filter_arrays=filter_input_list if is_filtered else None,
            **kwargs,
        )

        try:  # temporal to spatial conversion, if scanning
            CF.vt_um = self.v_um_ms * CF.lag
        except AttributeError:
            CF.vt_um = CF.lag  # static

        if is_verbose:
            print("- Done.")

        # name the Corrfunc object
        if cf_name is not None:
            self.cf[cf_name] = CF
        else:
            self.cf[f"gSTED {gate_ns}"] = CF

        return CF

    @file_utilities.rotate_data_to_disk()
    def cross_correlate_data(
        self,
        xcorr_types=["AB", "BA"],
        cf_name=None,
        gate1_ns=Limits(0, np.inf),
        gate2_ns=Limits(0, np.inf),
        afterpulse_params=None,
        should_subtract_bg_corr=True,
        is_verbose=False,
        should_add_to_xcf_dict=True,
        **kwargs,
    ) -> Dict[str, CorrFunc]:
        """Doc."""

        if is_verbose:
            print("Preparing split data for correlator...", end=" ")

        # Unite TDC gates and detector gates
        gates = []
        for i in (1, 2):
            gate_ns = locals()[f"gate{i}_ns"]
            tdc_gate_ns = Limits(gate_ns)
            if tdc_gate_ns != Limits(0, np.inf) or self.detector_gate_ns != Limits(0, np.inf):
                if not hasattr(self, "tdc_calib"):  # calibrate TDC (if not already calibrated)
                    self.calibrate_tdc(
                        should_dump_data=False, is_verbose=is_verbose
                    )  # abort data rotation decorator
                effective_lower_gate_ns = max(tdc_gate_ns.lower, self.detector_gate_ns.lower)
                effective_upper_gate_ns = min(tdc_gate_ns.upper, self.detector_gate_ns.upper)
                gates.append(Limits(effective_lower_gate_ns, effective_upper_gate_ns))
        gate1_ns, gate2_ns = gates

        # create list of split data for correlator
        dt_ts_split_dict = self._prepare_xcorr_splits_dict(
            xcorr_types,
            gate1_ns=gate1_ns,
            gate2_ns=gate2_ns,
            is_verbose=is_verbose,
        )

        # disregard the first line of each split (delay_time, used for gating in autocorrelation) and convert to int32 for correlator
        corr_input_dict: Dict[str, List] = {xx: [] for xx in xcorr_types}
        for xx in xcorr_types:
            for split in dt_ts_split_dict[xx]:
                corr_input_dict[xx].append(np.squeeze(split[1:].astype(np.int32)))

        # correlate data
        if is_verbose:
            print(
                f"Correlating ({', '.join(xcorr_types)}) {self.scan_type} data ({self.name} [{gate1_ns} ns vs. {gate2_ns} ns]):",
                end=" ",
            )

        if self.scan_type in {"static", "circle"}:
            correlator_option = SimpleNamespace(
                auto=CorrelatorType.PH_DELAY_CORRELATOR,
                cross=CorrelatorType.PH_DELAY_CROSS_CORRELATOR,
            )
        elif self.scan_type == "angular":
            correlator_option = SimpleNamespace(
                auto=CorrelatorType.PH_DELAY_CORRELATOR_LINES,
                cross=CorrelatorType.PH_DELAY_CROSS_CORRELATOR_LINES,
            )

        # create a dictionary with instantiated CorrFunc objects according to the needed 'xcorr_types'
        CF_dict = {
            xx: CorrFunc(
                f"{cf_name}_{xx} ({gate1_ns} vs. {gate2_ns} ns)"
                if cf_name is not None
                else f"{xx} ({gate1_ns} vs. {gate2_ns} ns)",
                correlator_option.auto if xx in {"AA", "BB"} else correlator_option.cross,
                self.laser_freq_hz,
            )
            for xx in xcorr_types
        }

        # cross/auto-correlate
        for xx in xcorr_types:
            CF = CF_dict[xx]
            CF.correlate_measurement(
                corr_input_dict[xx],
                afterpulse_params if afterpulse_params is not None else self.afterpulse_params,
                getattr(self, "bg_line_corr_list", []) if should_subtract_bg_corr else [],
                is_verbose=is_verbose,
                **kwargs,
            )

            try:  # temporal to spatial conversion, if scanning
                CF.vt_um = self.v_um_ms * CF.lag
            except AttributeError:
                CF.vt_um = CF.lag  # static

            if should_add_to_xcf_dict:
                # name the Corrfunc object
                self.xcf[CF.name] = CF

        if is_verbose:
            print("- Done.")

        return CF_dict

    def _prepare_xcorr_splits_dict(
        self, xcorr_types: List[str], is_verbose=True, **kwargs
    ) -> Dict[str, List]:
        """
        Gates are meant to divide the data into 2 parts (A&B), each having its own splits.
        To perform autocorrelation, only one ("AA") is used, and in the default (0, inf) limits, with actual gating done later in 'correlate_data' method.
        """

        if is_verbose:
            print("Preparing files for software correlator...", end=" ")

        file_splits_dict_list = [
            p.get_xcorr_splits_dict(self.scan_type, xcorr_types, self.laser_freq_hz, **kwargs)
            for p in self.data
        ]
        dt_ts_splits_dict = {
            xx: [
                dt_ts_split
                for splits_dict in file_splits_dict_list
                for dt_ts_split in splits_dict[xx]
            ]
            for xx in xcorr_types
        }

        if is_verbose:
            print("Done.")

        return dt_ts_splits_dict

    def plot_correlation_functions(
        self,
        x_field="lag",
        y_field="normalized",
        x_scale="log",
        y_scale="linear",
        ylim=(-0.20, 1.4),
        plot_kwargs={},
        **kwargs,
    ) -> List[str]:
        """Doc."""

        with Plotter(super_title=f"'{self.name.capitalize()}' - ACFs", **kwargs) as ax:
            legend_labels = []
            kwargs["parent_ax"] = ax
            for cf_name, cf in {**self.cf, **self.xcf}.items():
                cf.plot_correlation_function(
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

    def compare_lifetimes(
        self,
        legend_label: str,
        compare_to: dict = None,
        normalization_type="Per Time",
        parent_ax=None,
    ):
        """
        Plots a comparison of lifetime histograms. 'kwargs' is a dictionary, where keys are to be used as legend labels
        and values are 'TDCPhotonDataMixin'-inheriting objects which are supposed to have their own TDC calibrations.
        """

        # add self (first) to compared TDC calibrations
        compared = {**{legend_label: self}, **compare_to}

        h = []
        for label, measurement in compared.items():
            with suppress(AttributeError):
                # AttributeError - assume other tdc_objects that have tdc_calib structures
                x = measurement.tdc_calib.t_hist
                if normalization_type == "NO":
                    y = measurement.tdc_calib.all_hist / measurement.tdc_calib.t_weight
                elif normalization_type == "Per Time":
                    y = measurement.tdc_calib.all_hist_norm
                elif normalization_type == "By Sum":
                    y = measurement.tdc_calib.all_hist_norm / np.sum(
                        measurement.tdc_calib.all_hist_norm[
                            np.isfinite(measurement.tdc_calib.all_hist_norm)
                        ]
                    )
                else:
                    raise ValueError(f"Unknown normalization type '{normalization_type}'.")
                h.append((x, y, label))

        with Plotter(parent_ax=parent_ax, super_title="Life Time Comparison") as ax:
            labels = []
            for tuple_ in h:
                x, y, label = tuple_
                labels.append(label)
                ax.semilogy(x, y, "-o", label=label)
            ax.set_xlabel("Life Time (ns)")
            ax.set_ylabel("Frequency")
            ax.legend(labels)

    def calculate_structure_factors(self, plot_kwargs={}, **kwargs) -> None:
        """Doc."""

        #  calculation and plotting
        with Plotter(
            #            xlim=Limits(q[1], np.pi / min(w_xy)),
            super_title=f"{self.name.capitalize()}: Structure Factor ($S(q)$)",
            **kwargs,
        ) as ax:
            legend_labels = []
            for name, cf in self.cf.items():
                s = cf.calculate_structure_factor(**kwargs)
                ax.set_title("Gaussian vs. linear\nInterpolation")
                ax.loglog(
                    s.q,
                    np.vstack((s.sq / s.sq[0], s.sq_lin_intrp / s.sq_lin_intrp[0])).T,
                    **plot_kwargs,
                )
                legend_labels += [
                    f"{name}: Gaussian Interpolation",
                    f"{name}: Linear Interpolation",
                ]
            ax.set_xlabel("$q$ $(\\mu m^{-1})$")
            ax.set_ylabel("$S(q)$")
            ax.legend(legend_labels)

    def calculate_inherent_afterpulsing(
        self,
        gate1_ns,
        gate2_ns,
        is_verbose=False,
        **kwargs,
    ) -> Tuple[np.ndarray, Tuple[CorrFunc, CorrFunc]]:
        """Calculate inherent afterpulsing by cross-correlating the fluorscent photons (peak) with the white-noise ones (tail)"""

        if is_verbose:
            print("Calculating inherent afterpulsing from cross-correlation...", end=" ")
        cf_dict = self.cross_correlate_data(
            xcorr_types=["AB", "BA"],
            cf_name="Afterpulsing",
            gate1_ns=gate1_ns,
            gate2_ns=gate2_ns,
            should_subtract_afterpulsing=False,
            #                should_subtract_bg_corr=False,
            should_dump_data=False,  # abort data rotation decorator
            should_add_to_xcf_dict=False,  # avoid saving to self
            is_verbose=is_verbose,
        )
        XCF_AB, XCF_BA = cf_dict["AB"], cf_dict["BA"]

        # Averaging and normalizing properly the subtraction of BA from AB
        sbtrct_AB_BA_arr = np.empty(XCF_AB.corrfunc.shape)
        for idx, (corrfunc_AB, corrfunc_BA, countrate_pair) in enumerate(
            zip(XCF_AB.corrfunc, XCF_BA.corrfunc, XCF_AB.countrate_list)
        ):
            norm_factor = self.pulse_period_ns / (
                gate2_ns.interval() / countrate_pair.b - gate1_ns.interval() / countrate_pair.a
            )
            sbtrct_AB_BA_arr[idx] = norm_factor * (corrfunc_AB - corrfunc_BA)

        return sbtrct_AB_BA_arr.mean(axis=0), (XCF_AB, XCF_BA)

    def dump_or_load_data(self, should_load: bool, method_name=None) -> None:
        """
        Load or save the 'data' attribute.
        (relieve RAM - important during multiple-experiment analysis)
        """

        DUMP_PATH = file_utilities.DUMP_PATH

        with suppress(AttributeError):
            # AttributeError - name_on_disk is not defined (happens when doing alignment, for example)
            if should_load:  # loading data
                if self.is_data_dumped:
                    logging.debug(
                        f"{method_name}: Loading dumped data '{self.name_on_disk}' from '{DUMP_PATH}'."
                    )
                    with suppress(FileNotFoundError):
                        self.data = file_utilities.load_object(DUMP_PATH / self.name_on_disk)
                        self.is_data_dumped = False
            else:  # saving data
                is_saved = file_utilities.save_object(
                    self.data,
                    DUMP_PATH / self.name_on_disk,
                    obj_name="dumped data array",
                    element_size_estimate_mb=self.data[0].size_estimate_mb,
                )
                if is_saved:
                    self.data = []
                    self.is_data_dumped = True
                    logging.debug(
                        f"{method_name}: Dumped data '{self.name_on_disk}' to '{DUMP_PATH}'."
                    )


class ImageSFCSMeasurement(CountsImageMixin):
    """Doc."""

    def __init__(self):
        pass

    def read_image_data(self, file_path, **kwargs) -> None:
        """Doc."""

        file_dict = file_utilities.load_file_dict(file_path)
        self.process_data_file(file_dict, **kwargs)

    def process_data_file(self, file_dict: dict, **kwargs) -> None:
        """Doc."""

        # store relevant attributes (add more as needed)
        self.laser_mode = file_dict.get("laser_mode")
        self.scan_params = file_dict.get("scan_params")

        # Get ungated image (excitation or sted)
        self.image_data = self.create_image_stack_data(file_dict)

        # gating stuff (TDC) - not yet implemented
        self.data = None


class SolutionSFCSExperiment:
    """Doc."""

    UPPERֹ_ֹGATE_NS = 20

    def __init__(self, name):
        self.name = name
        self.confocal: SolutionSFCSMeasurement
        self.sted: SolutionSFCSMeasurement

    def load_experiment(
        self,
        confocal_template: Union[str, Path] = None,
        sted_template: Union[str, Path] = None,
        confocal=None,
        sted=None,
        should_plot=True,
        should_plot_meas=True,
        confocal_kwargs={},
        sted_kwargs={},
        **kwargs,
    ):
        """Doc."""

        if (
            (confocal_template is None)
            and (sted_template is None)
            and (confocal is None)
            and (sted is None)
        ):  # check if at least one measurement is available for laoding
            raise RuntimeError("Can't load experiment with no measurements!")

        # load measurements
        for meas_type in ("confocal", "sted"):
            measurement = locals()[meas_type]
            meas_template = locals()[f"{meas_type}_template"]
            meas_kwargs = locals()[f"{meas_type}_kwargs"]
            if measurement is None:

                if meas_template is not None:
                    self.load_measurement(
                        meas_type=meas_type,
                        file_path_template=meas_template,
                        should_plot=should_plot_meas,
                        **meas_kwargs,
                        **kwargs,
                    )
                else:  # Use empty measuremnt by default
                    setattr(self, meas_type, SolutionSFCSMeasurement(name=meas_type))
            else:  # use supllied measurement
                setattr(self, meas_type, measurement)
                getattr(self, meas_type).name = meas_type  # remame supplied measurement

        if should_plot:
            super_title = f"Experiment '{self.name.capitalize()}' - All ACFs"
            with Plotter(subplots=(1, 2), super_title=super_title, **kwargs) as axes:
                self.plot_correlation_functions(
                    parent_ax=axes[0],
                    y_field="avg_cf_cr",
                    x_scale="log",
                    xlim=None,  # autoscale x axis
                )

                self.plot_correlation_functions(
                    parent_ax=axes[1],
                )

    def load_measurement(
        self,
        meas_type: str,
        file_path_template: Union[str, Path],
        should_plot: bool = False,
        plot_kwargs: dict = {},
        force_processing=True,
        should_re_correlate=False,
        **kwargs,
    ):

        if "cf_name" not in kwargs:
            if meas_type == "confocal":
                kwargs["cf_name"] = "confocal"
            else:  # sted
                kwargs["cf_name"] = "sted"
        cf_name = kwargs["cf_name"]

        if kwargs.get(f"{meas_type}_file_selection"):
            kwargs["file_selection"] = kwargs[f"{meas_type}_file_selection"]

        measurement = SolutionSFCSMeasurement(name=meas_type)

        if not force_processing:  # Use pre-processed
            try:
                file_path_template = Path(file_path_template)  # str-> Path if needed
                dir_path, file_template = file_path_template.parent, file_path_template.name
                # load pre-processed
                file_path = dir_path / "processed" / re.sub("_[*]", "", file_template)
                measurement = file_utilities.load_processed_solution_measurement(
                    file_path,
                    file_template,
                    should_load_data=should_re_correlate,
                )
                measurement.name = meas_type
                print(f"Loaded pre-processed {meas_type} measurement: '{file_path}'")
            except OSError:
                print(
                    f"Pre-processed {meas_type} measurement not found at: '{file_path}'. Processing data regularly."
                )

        if not measurement.cf:  # Process data
            measurement.read_fpga_data(
                file_path_template,
                should_plot=should_plot,
                **kwargs,
            )
        if not measurement.cf or should_re_correlate:  # Correlate and average data
            measurement.cf = {}
            measurement.correlate_and_average(is_verbose=True, **kwargs)

        if should_plot:

            if (x_field := kwargs.get("x_field")) is None:
                if measurement.scan_type == "static":
                    x_field = "lag"
                else:  # angular or circular scan
                    x_field = "vt_um"

            with Plotter(
                super_title=f"'{self.name.capitalize()}' Experiment\n'{measurement.name.capitalize()}' Measurement - ACFs",
                ylim=(-100, list(measurement.cf.values())[0].g0 * 1.5),
            ) as ax:
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

        setattr(self, meas_type, measurement)

    def save_processed_measurements(self, **kwargs):
        """Doc."""

        print("Saving processed measurements to disk...", end=" ")
        if hasattr(self.confocal, "scan_type"):
            if file_utilities.save_processed_solution_meas(
                self.confocal, self.confocal.file_path_template.parent, **kwargs
            ):
                print("Confocal saved...", end=" ")
            else:
                print(
                    "Not saving - processed measurement already exists (set 'should_force = True' to override.)",
                    end=" ",
                )
        if hasattr(self.sted, "scan_type"):
            if file_utilities.save_processed_solution_meas(
                self.sted, self.sted.file_path_template.parent, **kwargs
            ):
                print("STED saved...", end=" ")
            else:
                print(
                    "Not saving - processed measurement already exists (set 'should_force = True' to override.)",
                    end=" ",
                )
        print("Done.\n")

    def calibrate_tdc(self, should_plot=True, **kwargs):
        """Doc."""

        # calibrate excitation measurement first, and sync STED to it if STED exists
        if hasattr(self.confocal, "scan_type"):
            self.confocal.calibrate_tdc(**kwargs)
            if hasattr(self.sted, "scan_type"):  # if both measurements quack as if loaded
                self.sted.calibrate_tdc(sync_coarse_time_to=self.confocal.tdc_calib, **kwargs)

        # calibrate STED only
        else:
            self.sted.calibrate_tdc(**kwargs)

        if should_plot:
            super_title = f"'{self.name.capitalize()}' Experiment\nTDC Calibration"
            # Both measurements loaded
            if hasattr(self.confocal, "scan_type") and hasattr(
                self.sted, "scan_type"
            ):  # if both measurements quack as if loaded
                with Plotter(subplots=(2, 4), super_title=super_title, **kwargs) as axes:
                    self.confocal.tdc_calib.plot_tdc_calibration(parent_axes=axes[:, :2])
                    self.sted.tdc_calib.plot_tdc_calibration(parent_axes=axes[:, 2:])

            # Confocal only
            elif hasattr(self.confocal, "scan_type"):  # STED measurement not loaded
                self.confocal.tdc_calib.plot_tdc_calibration(super_title=super_title, **kwargs)

            # STED only
            else:
                self.sted.tdc_calib.plot_tdc_calibration(super_title=super_title, **kwargs)

    def get_lifetime_parameters(
        self,
        sted_field="symmetric",
        drop_idxs=[],
        fit_range=None,
        param_estimates=None,
        bg_range=Limits(40, 60),
        **kwargs,
    ) -> LifeTimeParams:
        """Doc."""

        conf = self.confocal
        sted = self.sted

        conf_hist = conf.tdc_calib.all_hist_norm
        conf_t = conf.tdc_calib.t_hist
        conf_t = conf_t[np.isfinite(conf_hist)]
        conf_hist = conf_hist[np.isfinite(conf_hist)]

        sted_hist = sted.tdc_calib.all_hist_norm
        sted_t = sted.tdc_calib.t_hist
        sted_t = sted_t[np.isfinite(sted_hist)]
        sted_hist = sted_hist[np.isfinite(sted_hist)]

        h_max, j_max = conf_hist.max(), conf_hist.argmax()
        t_max = conf_t[j_max]

        beta0 = (h_max, 4, h_max * 1e-3)

        if fit_range is None:
            fit_range = Limits(t_max, 40)

        if param_estimates is None:
            param_estimates = beta0

        conf_params = conf.tdc_calib.fit_lifetime_hist(
            fit_range=fit_range, fit_param_estimate=beta0
        )

        # remove background
        sted_bg = np.mean(sted_hist[Limits(bg_range).valid_indices(sted_t)])
        conf_bg = np.mean(conf_hist[Limits(bg_range).valid_indices(conf_t)])
        sted_hist = sted_hist - sted_bg
        conf_hist = conf_hist - conf_bg

        j = conf_t < 20
        t = conf_t[j]
        hist_ratio = conf_hist[j] / np.interp(t, sted_t, sted_hist, right=0)
        if drop_idxs:
            j = np.setdiff1d(np.arange(1, len(t)), drop_idxs)
            t = t[j]
            hist_ratio = hist_ratio[j]

        title = "Robust Linear Fit of the Linear Part of the Histogram Ratio"
        with Plotter(figsize=(11.25, 7.5), super_title=title, **kwargs) as ax:

            # Using inner Plotter for manual selection
            title = "Use the mouse to place 2 markers\nlimiting the linear range:"
            linear_range = Limits()
            with Plotter(
                parent_ax=ax, super_title=title, selection_limits=linear_range, **kwargs
            ) as ax:
                ax.plot(t, hist_ratio)
                ax.legend(["hist_ratio"])

            j_selected = linear_range.valid_indices(t)

            if sted_field == "symmetric":
                # Robustly fit linear model with RANSAC algorithm
                ransac = linear_model.RANSACRegressor()
                ransac.fit(t[j_selected][:, np.newaxis], hist_ratio[j_selected])
                p0, p1 = ransac.estimator_.intercept_, ransac.estimator_.coef_[0]

                ax.plot(t[j_selected], hist_ratio[j_selected], "oy")
                ax.plot(t[j_selected], np.polyval([p1, p0], t[j_selected]), "r")
                ax.legend(["hist_ratio", "linear range", "robust fit"])

                lifetime_ns = conf_params.beta[1]
                sigma_sted = p1 * lifetime_ns
                try:
                    laser_pulse_delay_ns = (1 - p0) / p1
                except RuntimeWarning:
                    laser_pulse_delay_ns = None

            elif sted_field == "paraboloid":
                fit_params = curve_fit_lims(
                    FIT_NAME_DICT["ratio_of_lifetime_histograms"],
                    param_estimates=(2, 1, 1),
                    xs=t[j_selected],
                    ys=hist_ratio[j_selected],
                    ys_errors=np.ones(j_selected.sum()),
                    should_plot=True,
                )

                lifetime_ns = conf_params.beta[1]
                sigma_sted = (fit_params.beta[0] * lifetime_ns, fit_params.beta[1] * lifetime_ns)
                laser_pulse_delay_ns = fit_params.beta[2]

        self.lifetime_params = LifeTimeParams(lifetime_ns, sigma_sted, laser_pulse_delay_ns)
        return self.lifetime_params

    def compare_lifetimes(
        self,
        normalization_type="Per Time",
        **kwargs,
    ):
        """Doc."""

        if hasattr(self.confocal, "tdc_calib"):  # if TDC calibration performed
            super_title = f"'{self.name.capitalize()}' Experiment\nLifetime Comparison"
            with Plotter(super_title=super_title, **kwargs) as ax:
                self.confocal.compare_lifetimes(
                    "confocal",
                    compare_to=dict(STED=self.sted),
                    normalization_type=normalization_type,
                    parent_ax=ax,
                )

    def add_gate(
        self,
        gate_ns: Tuple[float, float],
        should_plot=True,
        should_re_correlate=False,
        is_verbose=True,
        **kwargs,
    ):
        """Doc."""

        if not should_re_correlate and self.sted.cf.get(f"gSTED {gate_ns}"):
            print(f"gSTED gate {gate_ns} already exists. Skipping...")
            return

        try:
            self.sted.correlate_and_average(
                cf_name=f"gSTED {gate_ns}", gate_ns=gate_ns, is_verbose=is_verbose, **kwargs
            )
            if should_plot:
                super_title = f"Experiment '{self.name.capitalize()}' - All ACFs"
                with Plotter(subplots=(1, 2), super_title=super_title, **kwargs) as axes:
                    self.plot_correlation_functions(
                        parent_ax=axes[0],
                        y_field="avg_cf_cr",
                        x_scale="log",
                        xlim=None,  # autoscale x axis
                    )

                    self.plot_correlation_functions(
                        parent_ax=axes[1],
                    )
        except AttributeError:
            # STED measurement not loaded
            raise RuntimeError(
                "Cannot add a gate if there's no STED measurement loaded to the experiment!"
            )

    def add_gates(self, gate_list: List[Tuple[float, float]], should_plot=True, **kwargs):
        """A convecience method for adding multiple gates."""

        for gate_ns in gate_list:
            self.add_gate(gate_ns, should_plot=False, **kwargs)
        if should_plot:
            super_title = f"Experiment '{self.name.capitalize()}' - All ACFs"
            with Plotter(subplots=(1, 2), super_title=super_title, **kwargs) as axes:
                self.plot_correlation_functions(
                    parent_ax=axes[0],
                    y_field="avg_cf_cr",
                    x_scale="log",
                    xlim=None,  # autoscale x axis
                )

                self.plot_correlation_functions(
                    parent_ax=axes[1],
                )

    def plot_correlation_functions(self, **kwargs):
        """Doc."""

        if hasattr(self.confocal, "scan_type"):
            ref_meas = self.confocal
        else:
            ref_meas = self.sted

        if ref_meas.scan_type == "static":
            kwargs["x_field"] = "lag"

        if kwargs.get("ylim") is None:
            if kwargs.get("y_field") in {"average_all_cf_cr", "avg_cf_cr"}:
                kwargs["ylim"] = Limits(-1e3, ref_meas.cf[ref_meas.name].g0 * 1.5)

        with Plotter(
            super_title=f"'{self.name.capitalize()}' Experiment - All ACFs",
            **kwargs,
        ) as ax:
            legend_label_lists = [
                getattr(self, meas_type).plot_correlation_functions(
                    parent_ax=ax,
                    x_field=kwargs.get("x_field", "vt_um"),
                    y_field=kwargs.get("y_field", "normalized"),
                    x_scale=kwargs.get("x_scale", "linear"),
                    y_scale=kwargs.get("y_scale", "linear"),
                    xlim=kwargs.get("xlim", (0, 1)),
                    ylim=kwargs.get("ylim", (-0.20, 1.4)),
                    plot_kwargs=kwargs.get("plot_kwargs", {}),
                )
                for meas_type in ("confocal", "sted")
            ]
            confocal_legend_labels, sted_legend_labels = legend_label_lists
            ax.legend(confocal_legend_labels + sted_legend_labels)

    def calculate_structure_factors(self, **kwargs) -> None:
        """Doc."""

        print(
            f"Calculating all structure factors for '{self.name.capitalize()}' experiment...",
            end=" ",
        )

        # calculate all structure factors
        for meas_type in ("confocal", "sted"):
            getattr(self, meas_type).calculate_structure_factors(**kwargs)

        # plot them
        with Plotter(
            subplots=(1, 2),
            super_title=f"Experiment '{self.name.capitalize()}':\nStructure Factors",
        ) as axes:
            axes[0].set_title("Gaussian Interpolation")
            axes[1].set_title("Linear Interpolation")
            legend_labels = []

            for meas_type in ("confocal", "sted"):
                meas = getattr(self, meas_type)
                for cf_name, cf in meas.cf.items():
                    s = cf.structure_factor
                    axes[0].loglog(s.q, s.sq / s.sq[0], **kwargs.get("plot_kwargs", {}))
                    axes[1].loglog(
                        s.q, s.sq_lin_intrp / s.sq_lin_intrp[0], **kwargs.get("plot_kwargs", {})
                    )
                    legend_labels.append(cf_name)

            for ax in axes:
                ax.set_xlabel("$q$ $(\\mu m^{-1})$")
                ax.set_ylabel("$S(q)$")
                ax.legend(legend_labels)

    def fit_structure_factors(self, model: str):
        """Doc."""
        # TODO:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.dawsn.html

        raise NotImplementedError


def calculate_calibrated_afterpulse(
    lag: np.ndarray,
    afterpulse_params: tuple = file_utilities.default_system_info["afterpulse_params"],
    gate_ns: Union[tuple, Limits] = (0, np.inf),
    laser_freq_hz: float = 1e7,
) -> np.ndarray:
    """Doc."""

    gate_pulse_period_ratio = min(1.0, Limits(gate_ns).interval() / 1e9 * laser_freq_hz)
    fit_name, beta = afterpulse_params
    if fit_name == "multi_exponent_fit":
        # work with any number of exponents
        afterpulse = gate_pulse_period_ratio * multi_exponent_fit(lag, *beta)
    elif fit_name == "exponent_of_polynom_of_log":  # for old MATLAB files
        if lag[0] == 0:
            lag[0] = np.nan
        afterpulse = gate_pulse_period_ratio * np.exp(np.polyval(beta, np.log(lag)))

    return afterpulse
