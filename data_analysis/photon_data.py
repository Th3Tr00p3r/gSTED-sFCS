"""Raw data handling."""

from contextlib import suppress
from types import SimpleNamespace

import numpy as np
from scipy import stats

from utilities import display, fit_tools


class TDCPhotonData:
    """Doc."""

    def convert_fpga_data_to_photons(
        self,
        fpga_data,
        ignore_coarse_fine=False,
        version=3,
        locate_outliers=False,
        max_outlier_prob=1e-5,
        verbose=False,
    ):
        """Doc."""

        p = SimpleNamespace()

        if version >= 2:
            group_len = 7
            maxval = 256 ** 3
        else:
            raise ValueError(f"Version ({version}) must be greater than 2")

        p.version = version

        section_edges, tot_single_errors = find_all_section_edges(fpga_data, group_len)

        section_lengths = [edge_stop - edge_start for (edge_start, edge_stop) in section_edges]
        if verbose:
            if len(section_edges) > 1:
                print(
                    f"Found {len(section_edges)} sections of lengths: {', '.join(map(str, section_lengths))}. Using largest.",
                    end=" ",
                )
            else:
                print(f"Found a single section of length: {section_lengths[0]}.", end=" ")
            if tot_single_errors > 0:
                print(
                    f"Encountered a total of {tot_single_errors} ignoreable single errors.", end=" "
                )

        # using the largest section only
        largest_section_start_idx, largest_section_end_idx = section_edges[
            np.argmax(section_lengths)
        ]
        idxs = np.arange(largest_section_start_idx, largest_section_end_idx, group_len)

        # calculate the runtime in terms of the number of laser pulses since the beginning of the file
        runtime = (
            fpga_data[idxs + 1] * 256 ** 2 + fpga_data[idxs + 2] * 256 + fpga_data[idxs + 3]
        ).astype(np.int64)

        time_stamps = np.diff(runtime)

        # find simple "inversions": the data with a missing byte
        # decrease in runtime on data j+1, yet the next runtime data (j+2) is higher than j.
        inv_idxs = np.where((time_stamps[:-1] < 0) & ((time_stamps[:-1] + time_stamps[1:]) > 0))[0]
        if (inv_idxs.size) != 0:
            if verbose:
                print(
                    f"Found {inv_idxs.size} instances of missing byte data, ad hoc fixing...",
                    end=" ",
                )
            temp = (time_stamps[inv_idxs] + time_stamps[inv_idxs + 1]) / 2
            time_stamps[inv_idxs] = np.floor(temp, dtype="int")
            time_stamps[inv_idxs + 1] = np.ceil(temp, dtype="int")
            runtime[inv_idxs + 1] = runtime[inv_idxs + 2] - time_stamps[inv_idxs + 1]

        # repairing drops in runtime (happens when number of laser pulses passes 'maxval')
        neg_time_stamp_idxs = np.where(time_stamps < 0)[0]
        for i in neg_time_stamp_idxs + 1:
            runtime[i:] += maxval

        # handling coarse and fine times (for gating)
        if not ignore_coarse_fine:
            coarse = fpga_data[idxs + 4].astype(np.int16)
            p.fine = fpga_data[idxs + 5].astype(np.int16)

            # some fix due to an issue in FPGA
            if p.version >= 3:
                p.coarse = np.mod(coarse, 64)
                p.coarse2 = p.coarse - np.mod(p.coarse, 4) + (coarse // 64)
            else:
                p.coarse = coarse

        p.runtime = runtime
        p.time_stamps = np.diff(runtime).astype(np.int32)

        # find additional outliers (improbably large time_stamps) and break into
        # additional segments if they exist.
        # for exponential distribution MEDIAN and MAD are the same, but for
        # biexponential MAD seems more sensitive
        if locate_outliers:  # relevent for static data
            mu = max(
                np.median(p.time_stamps), np.abs(p.time_stamps - p.time_stamps.mean()).mean()
            ) / np.log(2)
            max_time_stamp = stats.expon.ppf(1 - max_outlier_prob / len(p.time_stamps), scale=mu)
            sec_edges = (p.time_stamps > max_time_stamp).nonzero()[0].tolist()
            if (n_outliers := len(sec_edges)) > 0:
                print(f"found {n_outliers} outliers.", end=" ")
            sec_edges = [0] + sec_edges + [len(p.time_stamps)]
            p.all_section_edges = np.array([sec_edges[:-1], sec_edges[1:]]).T

        return p

    def calibrate_tdc(  # NOQA C901
        self,
        tdc_chain_length=128,
        pick_valid_bins_method="auto",
        pick_calib_bins_method="auto",
        calib_time_s=40e-9,
        n_zeros_for_fine_bounds=10,
        fine_shift=0,
        time_bins_for_hist_ns=0.1,
        exmpl_photon_data=None,
        sync_coarse_time_to=None,
        forced_valid_coarse_bins=np.arange(19),
        forced_calibration_coarse_bins=np.arange(3, 12),
        should_plot=True,
    ):
        """Doc."""

        self.tdc_calib = dict()

        # keep runtime elements of each file for array size allocation
        n_elem = np.cumsum([0] + [p.runtime.size for p in self.data])

        # unite coarse and fine times from all files
        coarse = np.empty(shape=(n_elem[-1],), dtype=np.int16)
        fine = np.empty(shape=(n_elem[-1],), dtype=np.int16)
        for i, p in enumerate(self.data):
            coarse[n_elem[i] : n_elem[i + 1]] = p.coarse
            fine[n_elem[i] : n_elem[i + 1]] = p.fine

        if self.type == "angular_scan":
            photon_idxs = fine > self.NAN_PLACEBO  # remove line starts/ends
            coarse = coarse[photon_idxs]
            fine = fine[photon_idxs]

        h_all = np.bincount(coarse)
        x_all = np.arange(coarse.max() + 1)

        if pick_valid_bins_method == "auto":
            h_all = h_all[coarse.min() :]
            x_all = np.arange(coarse.min(), coarse.max() + 1)
            j = (h_all > (np.median(h_all) / 100)).nonzero()[0]
            x = x_all[j]
            h = h_all[j]
        elif pick_valid_bins_method == "forced":
            x = forced_valid_coarse_bins
            h = h_all[x]
        elif pick_valid_bins_method == "by example":
            x = exmpl_photon_data.coarse["bins"]
            h = h_all[x]
        elif pick_valid_bins_method == "interactive":
            raise NotImplementedError("'interactive' valid bins selection is not yet implemented.")
        else:
            raise ValueError(f"Unknown method '{pick_valid_bins_method}' for picking valid bins!")

        self.coarse = dict(bins=x, h=h)

        # rearranging the bins
        if sync_coarse_time_to is None:
            max_j = np.argmax(h)
        elif isinstance(sync_coarse_time_to, int):
            max_j = sync_coarse_time_to
        elif isinstance(sync_coarse_time_to, dict) and hasattr(sync_coarse_time_to, "tdc_calib"):
            max_j = sync_coarse_time_to.tdc_calib["max_j"]
        else:
            raise ValueError(
                "Syncing coarse time is possible to either a number or to an object that has the attribute 'tdc_calib'!"
            )

        j_shift = np.roll(np.arange(len(h)), -max_j + 2)

        if pick_calib_bins_method == "auto":
            # pick data at more than 20ns delay from maximum
            j = np.where(j >= (calib_time_s * self.fpga_freq_hz + 2))[0]
            j_calib = j_shift[j]
            x_calib = x[j_calib]
        elif pick_calib_bins_method == "forced":
            x_calib = forced_calibration_coarse_bins
        elif (
            pick_calib_bins_method == "by example"
            or pick_calib_bins_method == "External calibration"
        ):
            x_calib = exmpl_photon_data.tdc_calib["coarse_bins"]
        elif pick_valid_bins_method == "interactive":
            raise NotImplementedError(
                "'interactive' calibration bins selection is not yet implemented."
            )
        else:
            raise ValueError(
                f"Unknown method '{pick_calib_bins_method}' for picking calibration bins!"
            )

        if pick_calib_bins_method == "External calibration":
            self.tdc_calib = exmpl_photon_data.tdc_calib
            max_j = exmpl_photon_data.tdc_calib["max_j"]

            if "l_quarter_tdc" in self.tdc_calib:
                l_quarter_tdc = self.tdc_calib["l_quarter_tdc"]
                r_quarter_tdc = self.tdc_calib["r_quarter_tdc"]

        else:
            self.tdc_calib["coarse_bins"] = x_calib

            fine_calib = fine[np.isin(coarse, x_calib)]

            self.tdc_calib["fine_bins"] = np.arange(tdc_chain_length)
            # x_tdc_calib_nonzero, h_tdc_calib_nonzero = np.unique(fine_calib, return_counts=True) #histogram check also np.bincount
            # h_tdc_calib = np.zeros(x_tdc_calib.shape, dtype = h_tdc_calib_nonzero.dtype)
            # h_tdc_calib[x_tdc_calib_nonzero] = h_tdc_calib_nonzero
            h_tdc_calib = np.bincount(fine_calib, minlength=tdc_chain_length)

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

            self.tdc_calib["l_quarter_tdc"] = l_quarter_tdc
            self.tdc_calib["r_quarter_tdc"] = r_quarter_tdc

            # zero those out of TDC: I think h_tdc_calib[left_tdc] = 0, so does not actually need to be set to 0
            h_tdc_calib[:left_tdc] = 0
            h_tdc_calib[right_tdc:] = 0

            t_calib = (
                (1 - np.cumsum(h_tdc_calib) / np.sum(h_tdc_calib)) / self.fpga_freq_hz * 1e9
            )  # invert and to ns

            self.tdc_calib["h"] = h_tdc_calib
            self.tdc_calib["t_calib"] = t_calib

            coarse_len = self.coarse["bins"].size

            t_weight = np.tile(self.tdc_calib["h"] / np.mean(self.tdc_calib["h"]), coarse_len)
            # t_weight = np.flip(t_weight)
            coarse_times = (
                np.tile(np.arange(coarse_len), [t_calib.size, 1]) / self.fpga_freq_hz * 1e9
            )
            delay_times = np.tile(t_calib, coarse_len) + coarse_times.flatten("F")
            # initially delay times are piece wise inverted. After "flip" in line 323 there should be no need in sorting:
            j_sorted = np.argsort(delay_times)
            self.tdc_calib["delay_times"] = delay_times[j_sorted]

            self.tdc_calib["t_weight"] = t_weight[j_sorted]

            self.tdc_calib["max_j"] = max_j

        # assign time delays to all photons
        self.tdc_calib["total_laser_pulses"] = 0
        last_coarse_bin = self.coarse["bins"][-1]
        max_j_m1 = max_j - 1
        if max_j_m1 == -1:
            max_j_m1 = last_coarse_bin

        delay_time = np.empty((0,), dtype=np.float64)
        for p in self.data:
            p.delay_time = np.empty(p.coarse.shape, dtype=np.float64)
            crs = np.minimum(p.coarse, last_coarse_bin) - self.coarse["bins"][max_j_m1]
            crs[crs < 0] = crs[crs < 0] + last_coarse_bin - self.coarse["bins"][0] + 1

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
                self.tdc_calib["t_calib"][p.fine[photon_idxs]]
                + (crs[photon_idxs] + delta_coarse[photon_idxs]) / self.fpga_freq_hz * 1e9
            )
            self.tdc_calib["total_laser_pulses"] += p.runtime[-1]
            p.delay_time[~photon_idxs] = np.nan  # line ends/starts

            delay_time = np.append(delay_time, p.delay_time[photon_idxs])

        bin_edges = np.arange(
            -time_bins_for_hist_ns / 2,
            np.max(delay_time) + time_bins_for_hist_ns,
            time_bins_for_hist_ns,
        )

        self.tdc_calib["t_hist"] = (bin_edges[:-1] + bin_edges[1:]) / 2
        k = np.digitize(self.tdc_calib["delay_times"], bin_edges)  # starts from 1

        self.tdc_calib["hist_weight"] = np.empty(self.tdc_calib["t_hist"].shape, dtype=np.float64)
        for i in range(len(self.tdc_calib["t_hist"])):
            j = k == (i + 1)
            self.tdc_calib["hist_weight"][i] = np.sum(self.tdc_calib["t_weight"][j])

        self.tdc_calib["all_hist"] = np.histogram(delay_time, bins=bin_edges)[0]
        self.tdc_calib["all_hist_norm"] = np.empty(
            self.tdc_calib["all_hist"].shape, dtype=np.float64
        )
        self.tdc_calib["error_all_hist_norm"] = np.empty(
            self.tdc_calib["all_hist"].shape, dtype=np.float64
        )
        nonzero = self.tdc_calib["hist_weight"] > 0
        self.tdc_calib["all_hist_norm"][nonzero] = (
            self.tdc_calib["all_hist"][nonzero]
            / self.tdc_calib["hist_weight"][nonzero]
            / self.tdc_calib["total_laser_pulses"]
        )
        self.tdc_calib["error_all_hist_norm"][nonzero] = (
            np.sqrt(self.tdc_calib["all_hist"][nonzero])
            / self.tdc_calib["hist_weight"][nonzero]
            / self.tdc_calib["total_laser_pulses"]
        )

        self.tdc_calib["all_hist_norm"][~nonzero] = np.nan
        self.tdc_calib["error_all_hist_norm"][~nonzero] = np.nan

        if should_plot:
            with display.show_external_axes(
                subplots=(2, 2), super_title=f"TDC Calibration - '{self.template}'"
            ) as axes:
                # TODO: shouldn't these (x, h, x_all, h_all, x_calib...) be saved to enable
                # plotting later on?
                axes[0, 0].semilogy(
                    x_all,
                    h_all,
                    "-o",
                    x,
                    h,
                    "-o",
                    x[np.isin(x, x_calib)],
                    h[np.isin(x, x_calib)],
                    "-o",
                )
                axes[0, 0].legend(["all hist", "valid bins", "calibration bins"])

                axes[0, 1].plot(self.tdc_calib["t_calib"], "-o")
                axes[0, 1].legend(["TDC calibration"])

                axes[1, 0].semilogy(self.tdc_calib["t_hist"], self.tdc_calib["all_hist_norm"], "-o")
                axes[1, 0].legend(["Photon lifetime histogram"])

    def compare_lifetimes(
        self,
        normalization_type="Per Time",
        legend_label=None,
        fontsize=14,
        **kwargs,
    ):
        """
        Plots a comparison of lifetime histograms.
        'kwargs' is a dictionary, where keys are to be used as legend labels and values are 'full_data'
        objects which are supposed to have their own TDC calibrations.
        """

        if legend_label is None:
            legend_label = self.template

        # add self to compared TDC calibrations
        kwargs.update([(legend_label, self)])

        h = []
        for label, full_data in kwargs.items():
            with suppress(AttributeError):
                # AttributeError - assume other objects that have TDCcalib structures
                x = full_data.tdc_calib["t_hist"]
                if normalization_type == "NO":
                    y = full_data.tdc_calib["all_hist"] / full_data.tdc_calib["t_weight"]
                elif normalization_type == "Per Time":
                    y = full_data.tdc_calib["all_hist_norm"]
                elif normalization_type == "By Sum":
                    y = full_data.tdc_calib["all_hist_norm"] / np.sum(
                        full_data.tdc_calib["all_hist_norm"][
                            np.isfinite(full_data.tdc_calib["all_hist_norm"])
                        ]
                    )
                else:
                    raise ValueError(f"Unknown normalization type '{normalization_type}'.")
                h.append((x, y, label))

        with display.show_external_axes(
            super_title="Life Time Comparison", fontsize=fontsize
        ) as ax:
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
        should_plot=True,
    ):
        """Doc."""

        is_finite_y = np.isfinite(self.tdc_calib[y_field])

        fit_param = fit_tools.curve_fit_lims(
            fit_name,
            fit_param_estimate,
            xs=self.tdc_calib[x_field][is_finite_y],
            ys=self.tdc_calib[y_field][is_finite_y],
            ys_errors=self.tdc_calib[y_error_field][is_finite_y],
            x_limits=fit_range,
            should_plot=should_plot,
            x_scale=x_scale,
            y_scale=y_scale,
        )

        try:
            self.tdc_calib["fit_param"][fit_param["func_name"]] = fit_param
        except KeyError:
            self.tdc_calib["fit_param"] = dict()
            self.tdc_calib["fit_param"][fit_param["func_name"]] = fit_param


def find_section_edges(data, group_len):  # NOQA C901
    """
    group_len: bytes per photon
    """

    # find index of first complete photon (where 248 and 254 bytes are spaced exatly (group_len -1) bytes apart)
    edge_start = first_full_photon_idx(data, group_len)
    if edge_start is None:
        raise RuntimeError("No data found! Check detector and FPGA.")

    # slice data where assumed to be 248 and 254 (photon brackets)
    data_presumed_248 = data[edge_start::group_len]
    data_presumed_254 = data[(edge_start + group_len - 1) :: group_len]

    # find indices where this assumption breaks
    missed_248_idxs = np.where(data_presumed_248 != 248)[0]
    tot_missed_248s = len(missed_248_idxs)
    missed_254_idxs = np.where(data_presumed_254 != 254)[0]
    tot_missed_254s = len(missed_254_idxs)

    data_end = False
    n_single_errors = 0
    for count, missed_248_idx in enumerate(missed_248_idxs):

        # hold data idx of current missing photon starting bracket
        data_idx_of_missed_248 = edge_start + missed_248_idx * group_len

        # condition for ignoring single photon error (just test the most significant bytes of the runtime are close)
        ignore_single_error_cond = (
            abs(
                int(data[data_idx_of_missed_248 + 1])
                - int(data[data_idx_of_missed_248 - (group_len - 1)])
            )
            < 3
        )

        # check that this is not a case of a singular mistake in 248 byte: happens very rarely but does happen
        if tot_missed_254s < count + 1:
            # no missing ending brackets, at least one missing starting bracket
            if missed_248_idx == (len(data_presumed_248) - 1):
                # problem in the last photon in the file
                # Found single error in last photon of file, ignoring and finishing...
                edge_stop = data_idx_of_missed_248
                break

            elif (tot_missed_248s == count + 1) or (
                np.diff(missed_248_idxs[count : (count + 2)]) > 1
            ):
                if ignore_single_error_cond:
                    # f"Found single photon error (data[{data_idx_of_missed_248}]), ignoring and continuing..."
                    n_single_errors += 1
                else:
                    raise RuntimeError("Check data for strange section edges!")

            else:
                raise RuntimeError(
                    "Bizarre problem in data: 248 byte out of registry while 254 is in registry!"
                )

        else:  # (tot_missed_254s >= count + 1)
            # if (missed_248_idxs[count] == missed_254_idxs[count]), # likely a real section
            if np.isin(missed_248_idx, missed_254_idxs):
                # Found a section, continuing...
                edge_stop = data_idx_of_missed_248
                if data[edge_stop - 1] != 254:
                    edge_stop = edge_stop - group_len
                break

            elif missed_248_idxs[count] == (
                missed_254_idxs[count] + 1
            ):  # np.isin(missed_248_idx, (missed_254_idxs[count]+1)): # likely a real section ? why np.isin?
                # Found a section, continuing...
                edge_stop = data_idx_of_missed_248
                if data[edge_stop - 1] != 254:
                    edge_stop = edge_stop - group_len
                break

            elif missed_248_idx < missed_254_idxs[count]:  # likely a singular error on 248 byte
                if ignore_single_error_cond:
                    # f"Found single photon error (data[{data_idx_of_missed_248}]), ignoring and continuing..."
                    n_single_errors += 1
                    continue
                else:
                    raise RuntimeError("Check data for strange section edges!")

            else:  # likely a signular mistake on 254 byte
                if ignore_single_error_cond:
                    # f"Found single photon error (data[{data_idx_of_missed_248}]), ignoring and continuing..."
                    n_single_errors += 1
                    continue
                else:
                    raise RuntimeError("Check data for strange section edges!")

    if tot_missed_248s > 0:
        if count == missed_248_idxs.size - 1:  # reached the end of the loop without breaking
            edge_stop = edge_start + (data_presumed_254.size - 1) * group_len
            data_end = True
    else:
        edge_stop = edge_start + (data_presumed_254.size - 1) * group_len
        data_end = True

    return edge_start, edge_stop, data_end, n_single_errors


def first_full_photon_idx(data, group_len) -> int:
    """Doc."""

    for idx in range(data.size):
        if (data[idx] == 248) and (data[idx + (group_len - 1)] == 254):
            return idx
    return None


def find_all_section_edges(data, group_len):
    """Doc."""

    section_edges = []
    data_end = False
    last_edge_stop = 0
    total_single_errors = 0
    while not data_end:
        remaining_data = data[last_edge_stop:]
        new_edge_start, new_edge_stop, data_end, n_single_errors = find_section_edges(
            remaining_data, group_len
        )
        new_edge_start += last_edge_stop
        new_edge_stop += last_edge_stop
        section_edges.append((new_edge_start, new_edge_stop))
        last_edge_stop = new_edge_stop
        total_single_errors += n_single_errors

    return section_edges, total_single_errors
