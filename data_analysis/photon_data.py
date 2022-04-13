"""Raw data handling."""

from contextlib import suppress
from dataclasses import dataclass
from itertools import count as infinite_range
from typing import Any, List, Tuple, Union

import numpy as np
import scipy
from sklearn import linear_model

from utilities.display import Plotter
from utilities.file_utilities import rotate_data_to_disk
from utilities.fit_tools import FitParams, curve_fit_lims
from utilities.helper import Limits, div_ceil


@dataclass
class TDCPhotonData:
    """Holds a single file's worth of processed, TDC-based, temporal photon data"""

    version: int
    section_runtime_edges: list
    coarse: np.ndarray
    coarse2: np.ndarray
    fine: np.ndarray
    pulse_runtime: np.ndarray
    all_section_edges: np.ndarray
    size_estimate_mb: float


@dataclass
class TDCCalibration:
    """Doc."""

    x_all: np.ndarray
    h_all: np.ndarray
    bins: np.ndarray
    h: np.ndarray
    max_j: int
    coarse_bins: Any
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


@dataclass
class LifeTimeParams:
    """Doc."""

    lifetime_ns: float
    sigma_sted: Union[float, Tuple[float, float]]
    laser_pulse_delay_ns: float


class TDCPhotonDataMixin:
    """Methods for creating and analyzing TDCPhotonData objects"""

    # NOTE: These are for mypy to be silent. The proper way to do it would be using abstract base classes (ABCs)
    name: str
    data: List
    scan_type: str
    NAN_PLACEBO: int
    fpga_freq_hz: int
    template: str
    tdc_calib: TDCCalibration
    confocal: Any
    sted: Any

    def convert_fpga_data_to_photons(
        self,
        byte_data,
        ignore_coarse_fine=False,
        version=3,
        locate_outliers=False,
        max_outlier_prob=1e-5,
        should_use_all_sections=True,
        len_factor=0.01,
        is_verbose=False,
        byte_data_slice=None,
        **kwargs,
    ) -> TDCPhotonData:
        """Doc."""

        if is_verbose:
            print("Converting raw data to photons...", end=" ")

        if version >= 2:
            group_len = 7
            maxval = 256 ** 3
        else:
            raise ValueError(f"Version ({version}) must be greater than 2")

        # option to use only certain parts of data (for testing)
        if byte_data_slice is not None:
            byte_data = byte_data[byte_data_slice]

        section_edges, tot_single_errors = _find_all_section_edges(byte_data, group_len)
        section_lengths = [edge_stop - edge_start for (edge_start, edge_stop) in section_edges]

        if should_use_all_sections:
            photon_idxs_list: List[int] = []
            section_runtime_edges = []
            for start_idx, end_idx in section_edges:
                if end_idx - start_idx > sum(section_lengths) * len_factor:
                    section_runtime_start = len(photon_idxs_list)
                    section_photon_indxs = list(range(start_idx, end_idx, group_len))
                    section_runtime_end = section_runtime_start + len(section_photon_indxs)
                    photon_idxs_list += section_photon_indxs
                    section_runtime_edges.append((section_runtime_start, section_runtime_end))

            photon_idxs = np.array(photon_idxs_list)

        else:  # using largest section only
            max_sec_start_idx, max_sec_end_idx = section_edges[np.argmax(section_lengths)]
            photon_idxs = np.arange(max_sec_start_idx, max_sec_end_idx, group_len)
            section_runtime_edges = [(0, len(photon_idxs))]

        if is_verbose:
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
            byte_data[photon_idxs + 1] * 256 ** 2
            + byte_data[photon_idxs + 2] * 256
            + byte_data[photon_idxs + 3]
        ).astype(np.int64)

        time_stamps = np.diff(pulse_runtime)

        # find simple "inversions": the data with a missing byte
        # decrease in pulse_runtime on data j+1, yet the next pulse_runtime data (j+2) is higher than j.
        inv_idxs = np.where((time_stamps[:-1] < 0) & ((time_stamps[:-1] + time_stamps[1:]) > 0))[0]
        if (inv_idxs.size) != 0:
            if is_verbose:
                print(
                    f"Found {inv_idxs.size} instances of missing byte data, ad hoc fixing...",
                    end=" ",
                )
            temp = (time_stamps[inv_idxs] + time_stamps[inv_idxs + 1]) / 2
            time_stamps[inv_idxs] = np.floor(temp).astype(np.int64)
            time_stamps[inv_idxs + 1] = np.ceil(temp).astype(np.int64)
            pulse_runtime[inv_idxs + 1] = pulse_runtime[inv_idxs + 2] - time_stamps[inv_idxs + 1]

        # repairing drops in pulse_runtime (happens when number of laser pulses passes 'maxval')
        neg_time_stamp_idxs = np.where(time_stamps < 0)[0]
        for i in neg_time_stamp_idxs + 1:
            pulse_runtime[i:] += maxval

        # handling coarse and fine times (for gating)
        if not ignore_coarse_fine:
            coarse = byte_data[photon_idxs + 4].astype(np.int16)
            fine = byte_data[photon_idxs + 5].astype(np.int16)

            # some fix due to an issue in FPGA
            if version >= 3:
                coarse_mod64 = np.mod(coarse, 64)
                coarse2 = coarse_mod64 - np.mod(coarse_mod64, 4) + (coarse // 64)
                coarse = coarse_mod64
            else:
                coarse = coarse
                coarse2 = None
        else:
            coarse = None
            coarse2 = None
            fine = None

        time_stamps = np.diff(pulse_runtime).astype(np.int32)

        # find additional outliers (improbably large time_stamps) and break into
        # additional segments if they exist.
        # for exponential distribution MEDIAN and MAD are the same, but for
        # biexponential MAD seems more sensitive
        if locate_outliers:  # relevent for static data
            mu = max(
                np.median(time_stamps), np.abs(time_stamps - time_stamps.mean()).mean()
            ) / np.log(2)
            max_time_stamp = scipy.stats.expon.ppf(
                1 - max_outlier_prob / len(time_stamps), scale=mu
            )
            sec_edges = (time_stamps > max_time_stamp).nonzero()[0].tolist()
            if (n_outliers := len(sec_edges)) > 0:
                print(f"found {n_outliers} outliers.", end=" ")
            sec_edges = [0] + sec_edges + [len(time_stamps)]
            all_section_edges = np.array([sec_edges[:-1], sec_edges[1:]]).T
        else:
            all_section_edges = None

        return TDCPhotonData(
            version=version,
            section_runtime_edges=section_runtime_edges,
            coarse=coarse,
            coarse2=coarse2,
            fine=fine,
            pulse_runtime=pulse_runtime,
            all_section_edges=all_section_edges,
            size_estimate_mb=max(section_lengths) / 1e6,
        )

    @rotate_data_to_disk(does_modify_data=True)
    def calibrate_tdc(
        self,
        tdc_chain_length=128,
        pick_valid_bins_according_to=None,
        sync_coarse_time_to=None,
        pick_calib_bins_according_to=None,
        external_calib=None,
        calib_time_ns=40,
        n_zeros_for_fine_bounds=10,
        time_bins_for_hist_ns=0.1,
        should_plot=False,
        parent_axes=None,
        **kwargs,
    ) -> None:
        """Doc."""

        print(f"\n{self.name}: Calibrating TDC...", end=" ")

        coarse, fine = self._unite_coarse_fine_data()

        h_all = np.bincount(coarse).astype(np.uint32)
        x_all = np.arange(coarse.max() + 1, dtype=np.uint8)

        if pick_valid_bins_according_to is None:
            h_all = h_all[coarse.min() :]
            x_all = np.arange(coarse.min(), coarse.max() + 1, dtype=np.uint8)
            bins = x_all
            h = h_all
        elif isinstance(pick_valid_bins_according_to, np.ndarray):
            bins = sync_coarse_time_to
            h = h_all[bins]
        elif isinstance(pick_valid_bins_according_to, TDCCalibration):
            bins = sync_coarse_time_to.coarse_bins
            h = h_all[bins]
        else:
            raise TypeError(
                f"Unknown type '{type(pick_valid_bins_according_to)}' for picking valid bins!"
            )

        # rearranging the bins
        if sync_coarse_time_to is None:
            max_j = np.argmax(h)
        elif isinstance(sync_coarse_time_to, int):
            max_j = sync_coarse_time_to
        elif isinstance(sync_coarse_time_to, TDCCalibration):
            max_j = sync_coarse_time_to.max_j
        else:
            raise ValueError(
                "Syncing coarse time is possible to either a number or a 'TDCCalibration' object!"
            )

        j_shift = np.roll(np.arange(len(h)), -max_j + 2)

        if pick_calib_bins_according_to is None:
            # pick data at more than 'calib_time_ns' delay from peak maximum
            j = np.where(bins >= ((calib_time_ns * 1e-9) * self.fpga_freq_hz + 2))[0]
            if not j.any():
                raise ValueError(f"Gate width is too narrow for calib_time_ns={calib_time_ns}!")
            j_calib = j_shift[j]
            coarse_bins = bins[j_calib]
        elif isinstance(pick_calib_bins_according_to, (list, np.ndarray)):
            coarse_bins = pick_calib_bins_according_to
        elif isinstance(pick_calib_bins_according_to, TDCCalibration):
            coarse_bins = pick_calib_bins_according_to.coarse_bins
        else:
            raise TypeError(
                f"Unknown type '{type(pick_calib_bins_according_to)}' for picking calibration bins!"
            )

        # Don't use 'empty' bins for calibration # TESTESTEST
        valid_cal_bins = np.nonzero(h_all > (np.median(h_all) / 100))[0]  # TESTESTEST
        coarse_bins = np.array(
            [bin_idx for bin_idx in coarse_bins if bin_idx in valid_cal_bins]
        )  # TESTESTEST

        if isinstance(external_calib, TDCCalibration):
            max_j = external_calib.max_j
            coarse_bins = external_calib.coarse_bins
            fine_bins = external_calib.fine_bins
            t_calib = external_calib.t_calib
            h_tdc_calib = external_calib.h_tdc_calib
            t_weight = external_calib.t_weight
            l_quarter_tdc = external_calib.l_quarter_tdc
            r_quarter_tdc = external_calib.r_quarter_tdc
            h = external_calib.h
            delay_times = external_calib.delay_times

            with suppress(AttributeError):
                l_quarter_tdc = self.tdc_calib.l_quarter_tdc
                r_quarter_tdc = self.tdc_calib.r_quarter_tdc

        else:

            fine_calib = fine[np.isin(coarse, coarse_bins)]
            fine_bins = np.arange(tdc_chain_length, dtype=np.uint8)
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
            left_tdc = np.max(zeros_tdc[zeros_tdc < mid_tdc]) + n_zeros_for_fine_bounds - 1
            right_tdc = np.min(zeros_tdc[zeros_tdc > mid_tdc]) + 1

            l_quarter_tdc = round(left_tdc + (right_tdc - left_tdc) / 4)
            r_quarter_tdc = round(right_tdc - (right_tdc - left_tdc) / 4)

            # zero those out of TDC: I think h_tdc_calib[left_tdc] = 0, so does not actually need to be set to 0
            h_tdc_calib[:left_tdc] = 0
            h_tdc_calib[right_tdc:] = 0

            t_calib = (
                (1 - np.cumsum(h_tdc_calib) / np.sum(h_tdc_calib)) / self.fpga_freq_hz * 1e9
            )  # invert and to ns

            coarse_len = bins.size

            t_weight = np.tile(h_tdc_calib / np.mean(h_tdc_calib), coarse_len)
            coarse_times = (
                np.tile(np.arange(coarse_len), [t_calib.size, 1]) / self.fpga_freq_hz * 1e9
            )
            delay_times = np.tile(t_calib, coarse_len) + coarse_times.flatten("F")
            # initially delay times are piece wise inverted. After "flip" in line 323 there should be no need in sorting:
            j_sorted = np.argsort(delay_times)
            delay_times = delay_times[j_sorted]

            t_weight = t_weight[j_sorted]

        # assign time delays to all photons
        total_laser_pulses = 0
        first_coarse_bin, *_, last_coarse_bin = bins
        delay_time_list = []
        for p in self.data:
            p.delay_time = np.empty(p.coarse.shape, dtype=np.float64)
            crs = np.minimum(p.coarse, last_coarse_bin) - bins[max_j - 1]
            crs[crs < 0] = crs[crs < 0] + last_coarse_bin - first_coarse_bin + 1

            delta_coarse = p.coarse2 - p.coarse
            delta_coarse[delta_coarse == -3] = 1  # 2bit limitation

            # in the TDC midrange use "coarse" counter
            in_mid_tdc = (p.fine >= l_quarter_tdc) & (p.fine <= r_quarter_tdc)
            delta_coarse[in_mid_tdc] = 0

            # on the right of TDC use "coarse2" counter (no change in delta)
            # on the left of TDC use "coarse2" counter decremented by 1
            on_left_tdc = p.fine < l_quarter_tdc
            delta_coarse[on_left_tdc] = delta_coarse[on_left_tdc] - 1

            photon_idxs = p.fine > self.NAN_PLACEBO  # self.NAN_PLACEBO are starts/ends of lines
            p.delay_time[photon_idxs] = (
                t_calib[p.fine[photon_idxs]]
                + (crs[photon_idxs] + delta_coarse[photon_idxs]) / self.fpga_freq_hz * 1e9
            )
            total_laser_pulses += p.pulse_runtime[-1]
            p.delay_time[~photon_idxs] = np.nan  # line ends/starts

            delay_time_list.append(p.delay_time[photon_idxs])

        delay_time = np.hstack(delay_time_list)

        bin_edges = np.arange(
            -time_bins_for_hist_ns / 2,
            np.max(delay_time) + time_bins_for_hist_ns,
            time_bins_for_hist_ns,
            dtype=np.float16,
        )

        t_hist = (bin_edges[:-1] + bin_edges[1:]) / 2
        k = np.digitize(delay_times, bin_edges)  # starts from 1

        hist_weight = np.empty(t_hist.shape, dtype=np.float64)
        for i in range(len(t_hist)):
            j = k == (i + 1)
            hist_weight[i] = np.sum(t_weight[j])

        all_hist = np.histogram(delay_time, bins=bin_edges)[0].astype(np.uint32)

        all_hist_norm = np.full(all_hist.shape, np.nan, dtype=np.float64)
        error_all_hist_norm = np.full(all_hist.shape, np.nan, dtype=np.float64)
        nonzero = hist_weight > 0
        all_hist_norm[nonzero] = all_hist[nonzero] / hist_weight[nonzero] / total_laser_pulses
        error_all_hist_norm[nonzero] = (
            np.sqrt(all_hist[nonzero]) / hist_weight[nonzero] / total_laser_pulses
        )

        if should_plot:
            self.plot_tdc_calibration()

        print("Done.")

        self.tdc_calib = TDCCalibration(
            x_all=x_all,
            h_all=h_all,
            bins=bins,
            h=h,
            max_j=max_j,
            coarse_bins=coarse_bins,
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

    def plot_tdc_calibration(self, parent_axes=None, **plot_kwargs) -> None:
        """Doc."""

        try:
            x_all = self.tdc_calib.x_all
            h_all = self.tdc_calib.h_all
            bins = self.tdc_calib.bins
            h = self.tdc_calib.h
            coarse_bins = self.tdc_calib.coarse_bins
            t_calib = self.tdc_calib.t_calib
            t_hist = self.tdc_calib.t_hist
            all_hist_norm = self.tdc_calib.all_hist_norm
        except AttributeError:
            return

        with Plotter(
            parent_ax=parent_axes,
            subplots=(2, 2),
            **plot_kwargs,
        ) as axes:
            axes[0, 0].semilogy(x_all, h_all, "-o", label="All Hist")
            axes[0, 0].semilogy(bins, h, "o", label="Valid Bins")
            axes[0, 0].semilogy(
                bins[np.isin(bins, coarse_bins)],
                h[np.isin(bins, coarse_bins)],
                "o",
                label="Calibration Bins",
            )
            axes[0, 0].legend()

            axes[0, 1].plot(t_calib, "-o", label="TDC Calibration")
            axes[0, 1].legend()

            axes[1, 0].semilogy(t_hist, all_hist_norm, "-o", label="Photon Lifetime Histogram")
            axes[1, 0].legend()

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

    def fit_lifetime_hist(
        self,
        fit_name="exponent_with_background_fit",
        x_field="t_hist",
        y_field="all_hist_norm",
        y_error_field="error_all_hist_norm",
        fit_param_estimate=[0.1, 4, 0.001],
        fit_range=(3.5, 30),
        x_scale="linear",
        y_scale="log",
        max_iter=3,
        should_plot=False,
    ) -> FitParams:
        """Doc."""

        is_finite_y = np.isfinite(getattr(self.tdc_calib, y_field))

        return curve_fit_lims(
            fit_name,
            fit_param_estimate,
            xs=getattr(self.tdc_calib, x_field)[is_finite_y],
            ys=getattr(self.tdc_calib, y_field)[is_finite_y],
            ys_errors=getattr(self.tdc_calib, y_error_field)[is_finite_y],
            x_limits=Limits(fit_range),
            should_plot=should_plot,
            x_scale=x_scale,
            y_scale=y_scale,
        )

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

        conf_params = conf.fit_lifetime_hist(fit_range=fit_range, fit_param_estimate=beta0)

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
                    fit_name="ratio_of_lifetime_histograms",
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

    def _unite_coarse_fine_data(self):
        """Doc."""

        # keep pulse_runtime elements of each file for array size allocation
        n_elem = np.cumsum([0] + [p.pulse_runtime.size for p in self.data])

        # unite coarse and fine times from all files
        coarse = np.empty(shape=(n_elem[-1],), dtype=np.int16)
        fine = np.empty(shape=(n_elem[-1],), dtype=np.int16)
        for i, p in enumerate(self.data):
            coarse[n_elem[i] : n_elem[i + 1]] = p.coarse
            fine[n_elem[i] : n_elem[i + 1]] = p.fine

        # remove line starts/ends from angular scan data
        if self.scan_type == "angular":
            photon_idxs = fine > self.NAN_PLACEBO
            coarse = coarse[photon_idxs]
            fine = fine[photon_idxs]

        return coarse, fine


@dataclass
class CountsImageStackData:
    """
    Holds a stack of images along with some relevant scan data,
    built from CI (counter input) NIDAQmx data.
    """

    image_stack_forward: np.ndarray
    norm_stack_forward: np.ndarray
    image_stack_backward: np.ndarray
    norm_stack_backward: np.ndarray
    line_ticks_v: np.ndarray
    row_ticks_v: np.ndarray
    plane_ticks_v: np.ndarray
    n_planes: int
    plane_orientation: str
    dim_order: Tuple[int, int, int]

    def get_image(self, method: str, plane_idx: int = None) -> np.ndarray:
        """Doc."""

        if plane_idx is None:
            plane_idx = self.n_planes // 2

        if method == "forward":
            img = self.image_stack_forward[:, :, plane_idx]
        elif method == "forward normalization":
            img = self.norm_stack_forward[:, :, plane_idx]
        elif method == "forward normalized":
            img = (
                self.image_stack_forward[:, :, plane_idx] / self.norm_stack_forward[:, :, plane_idx]
            )
        elif method == "backward":
            img = self.image_stack_backward[:, :, plane_idx]
        elif method == "backward normalization":
            img = self.norm_stack_backward[:, :, plane_idx]
        elif method == "backward normalized":
            img = (
                self.image_stack_backward[:, :, plane_idx]
                / self.norm_stack_backward[:, :, plane_idx]
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

        return img


class CountsImageMixin:
    """Doc."""

    def create_image_stack_data(self, file_dict: dict) -> CountsImageStackData:
        """Doc."""

        scan_settings = file_dict["scan_settings"]
        um_v_ratio = file_dict["system_info"]["xyz_um_to_v"]
        ao = file_dict["ao"]
        counts = file_dict["ci"]

        n_planes = scan_settings["n_planes"]
        n_lines = scan_settings["n_lines"]
        pxl_size_um = scan_settings["dim2_um"] / n_lines
        pxls_per_line = div_ceil(scan_settings["dim1_um"], pxl_size_um)
        dim_order = scan_settings["dim_order"]
        ppl = scan_settings["ppl"]
        ppp = n_lines * ppl
        turn_idx = ppl // 2

        first_dim = dim_order[0]
        dim1_center = scan_settings["initial_ao"][first_dim]
        um_per_v = um_v_ratio[first_dim]

        line_len_v = scan_settings["dim1_um"] / um_per_v
        dim1_min = dim1_center - line_len_v / 2

        pxl_size_v = pxl_size_um / um_per_v
        pxls_per_line = div_ceil(scan_settings["dim1_um"], pxl_size_um)

        # prepare to remove counts from outside limits
        dim1_ao_single = ao[0][:ppl]
        eff_idxs = ((dim1_ao_single - dim1_min) // pxl_size_v + 1).astype(np.int16)
        eff_idxs_forward = eff_idxs[:turn_idx]
        eff_idxs_backward = eff_idxs[-1 : (turn_idx - 1) : -1]

        # create counts stack shaped (n_lines, ppl, n_planes) - e.g. 80 x 1000 x 1
        j0 = ppp * np.arange(n_planes)[:, np.newaxis]
        J = np.tile(np.arange(ppp), (n_planes, 1)) + j0
        counts_stack = np.diff(np.concatenate((j0, counts[J]), axis=1))
        counts_stack = counts_stack.T.reshape(n_lines, ppl, n_planes)
        counts_stack_forward = counts_stack[:, :turn_idx, :]
        counts_stack_backward = counts_stack[:, -1 : (turn_idx - 1) : -1, :]

        # calculate the images and normalization separately for the forward/backward parts of the scan
        image_stack_forward, norm_stack_forward = self._calculate_plane_image_stack(
            counts_stack_forward, eff_idxs_forward, pxls_per_line
        )
        image_stack_backward, norm_stack_backward = self._calculate_plane_image_stack(
            counts_stack_backward, eff_idxs_backward, pxls_per_line
        )

        return CountsImageStackData(
            image_stack_forward=image_stack_forward,
            norm_stack_forward=norm_stack_forward,
            image_stack_backward=image_stack_backward,
            norm_stack_backward=norm_stack_backward,
            line_ticks_v=dim1_min + np.arange(pxls_per_line) * pxl_size_v,
            row_ticks_v=scan_settings["set_pnts_lines_odd"],
            plane_ticks_v=scan_settings.get("set_pnts_planes"),  # doesn't exist in older versions
            n_planes=n_planes,
            plane_orientation=scan_settings["plane_orientation"],
            dim_order=dim_order,
        )

    def _calculate_plane_image_stack(self, counts_stack, eff_idxs, pxls_per_line):
        """Doc."""

        n_lines, _, n_planes = counts_stack.shape
        image_stack = np.empty((n_lines, pxls_per_line, n_planes), dtype=np.int32)
        norm_stack = np.empty((n_lines, pxls_per_line, n_planes), dtype=np.int32)

        for i in range(pxls_per_line):
            image_stack[:, i, :] = counts_stack[:, eff_idxs == i, :].sum(axis=1)
            norm_stack[:, i, :] = counts_stack[:, eff_idxs == i, :].shape[1]

        return image_stack, norm_stack


def _find_section_edges(byte_data: np.ndarray, group_len: int):  # NOQA C901
    """
    group_len: bytes per photon
    """

    # find index of first complete photon (where 248 and 254 bytes are spaced exatly (group_len -1) bytes apart)
    if (edge_start := _first_full_photon_idx(byte_data, group_len)) is None:
        raise RuntimeError("No byte data found! Check detector and FPGA.")

    # slice byte_data where assumed to be 248 and 254 (photon brackets)
    data_presumed_248 = byte_data[edge_start::group_len]
    data_presumed_254 = byte_data[(edge_start + group_len - 1) :: group_len]

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
        data_idx_of_missed_248 = edge_start + missed_248_idx * group_len

        # condition for ignoring single photon error (just test the most significant bytes of the pulse_runtime are close)
        curr_runtime_msb = int(byte_data[data_idx_of_missed_248 + 1])
        prev_runtime_msb = int(byte_data[data_idx_of_missed_248 - (group_len - 1)])
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
                    edge_stop = edge_stop - group_len
                break

            elif missed_248_idxs[count] == (
                missed_254_idxs[count] + 1
            ):  # np.isin(missed_248_idx, (missed_254_idxs[count]+1)): # likely a real section ? why np.isin?
                # Found a section, continuing...
                edge_stop = data_idx_of_missed_248
                if byte_data[edge_stop - 1] != 254:
                    edge_stop = edge_stop - group_len
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
        edge_stop = edge_start + (data_presumed_254.size - 1) * group_len
        data_end = True

    return edge_start, edge_stop, data_end, n_single_errors


def _first_full_photon_idx(byte_data: np.ndarray, group_len: int) -> int:
    """
    Return the starting index of the first intact photon - a sequence of 'group_len'
    bytes starting with 248 and ending with 254. If no intact photons are found, returns 'None'
    """

    for idx in infinite_range():
        try:
            if (byte_data[idx] == 248) and (byte_data[idx + (group_len - 1)] == 254):
                return idx
        except IndexError:
            break
    return None


def _find_all_section_edges(
    byte_data: np.ndarray, group_len: int
) -> Tuple[List[Tuple[int, int]], int]:
    """Doc."""

    section_edges = []
    data_end = False
    last_edge_stop = 0
    total_single_errors = 0
    while not data_end:
        remaining_byte_data = byte_data[last_edge_stop:]
        new_edge_start, new_edge_stop, data_end, n_single_errors = _find_section_edges(
            remaining_byte_data, group_len
        )
        new_edge_start += last_edge_stop
        new_edge_stop += last_edge_stop
        section_edges.append((new_edge_start, new_edge_stop))
        last_edge_stop = new_edge_stop
        total_single_errors += n_single_errors

    return section_edges, total_single_errors
