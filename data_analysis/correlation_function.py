#!/usr/bin/env python3
"""
Created on Fri Apr 23 12:11:40 2021

@author: oleg
"""

import glob
import logging
import os
import re  # regular expressions
import sys

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy import ndimage, stats
from skimage import filters as skifilt
from skimage import morphology

from data_analysis.fit_tools import curve_fit_lims

# loads matfiles with structures (scipy.io does not do structures)
from data_analysis.matlab_utilities import loadmat
from data_analysis.photon_data import PhotonData
from data_analysis.software_correlator import CorrelatorType, SoftwareCorrelator

data_analysis_parent_folder, temp = os.path.split(os.path.dirname(__file__))
sys.path.append(data_analysis_parent_folder)

sys.path.append(os.path.dirname(__file__))
# sys.path.append('//Users/oleg/Documents/Python programming/FCS Python/')


class CorrFuncData:
    """Doc."""

    def do_average_corr(
        self,
        rejection=2,
        norm_range=np.array([1e-3, 2e-3]),
        delete_list=[],
        no_plot=False,
        use_numba=False,
    ):
        """Doc."""

        # enable 'use_numba' for speed-up
        def calc_weighted_avg(cf_cr, weights):
            """Doc."""

            tot_weights = weights.sum(0)

            average_cf_cr = (cf_cr * weights).sum(0) / tot_weights

            error_cf_cr = (
                np.sqrt((weights ** 2 * (cf_cr - average_cf_cr) ** 2).sum(0)) / tot_weights
            )

            return average_cf_cr, error_cf_cr

        self.rejection = rejection
        self.norm_range = norm_range
        self.delete_list = delete_list
        self.average_all_cf_cr = (self.cf_cr * self.weights).sum(0) / self.weights.sum(0)
        self.median_all_cf_cr = np.median(self.cf_cr, 0)
        JJ = np.logical_and(
            (self.lag > norm_range[1]), (self.lag < 100)
        )  # work in the relevant part
        self.score = (
            (1 / np.var(self.cf_cr[:, JJ], 0))
            * (self.cf_cr[:, JJ] - self.median_all_cf_cr[JJ]) ** 2
            / len(JJ)
        )
        if len(delete_list) == 0:
            self.j_good = np.where(self.score < self.rejection)[0]
            self.j_bad = np.where(self.score >= self.rejection)[0]
        else:
            self.j_bad = delete_list
            self.j_good = np.array(
                [i for i in range(self.cf_cr.shape[0]) if i not in delete_list]
            ).astype("int")

        if use_numba:
            func = nb.njit(calc_weighted_avg, cache=True)
        else:
            func = calc_weighted_avg

        self.average_cf_cr, self.error_cf_cr = func(
            self.cf_cr[self.j_good, :], self.weights[self.j_good, :]
        )

        Jt = np.logical_and((self.lag > self.norm_range[0]), (self.lag < self.norm_range[1]))
        self.g0 = (self.average_cf_cr[Jt] / self.error_cf_cr[Jt] ** 2).sum() / (
            1 / self.error_cf_cr[Jt] ** 2
        ).sum()
        self.normalized = self.average_cf_cr / self.g0
        self.error_normalized = self.error_cf_cr / self.g0
        if not no_plot:
            self.do_plot_corr_func()

    def do_plot_corr_func(
        self, x_field="lag", y_field="average_cf_cr", x_scale="log", y_scale="linear"
    ):
        x = getattr(self, x_field)
        y = getattr(self, y_field)
        if x_scale == "log":
            x = x[1:]  # remove zero point data
            y = y[1:]
        plt.plot(x, y, "o")  # skip 0 lag time
        plt.xlabel(x_field)
        plt.ylabel(y_field)
        plt.gca().set_x_scale(x_scale)
        plt.gca().set_y_scale(y_scale)
        plt.show()

    def do_fit(
        self,
        x_field="lag",
        y_field="average_cf_cr",
        y_error_field="error_cf_cr",
        fit_func="diffusion_3d_fit",
        constant_param={},
        fit_param_estimate=None,
        fit_range=[1e-3, 100],
        no_plot=False,
        x_scale="log",
        y_scale="linear",
    ):

        # [DynRange, IsDynRangeInput] = ParseInputs('Dynamic Range', [], varargin); % variable fit_range: give either a scalar or a vector of two
        # DynRangeBetaParam = ParseInputs('Dynamic Range parameter', 2, varargin); %the parameter in beta on which to do dynamic range
        # MaxIter = 3;
        # lsqcurvefit_param = ParseInputs('lsqcurvefit_param', {}, varargin);

        if not fit_param_estimate:
            fit_param_estimate = [self.g0, 0.035, 30]

        x = getattr(self, x_field)
        y = getattr(self, y_field)
        errorY = getattr(self, y_error_field)
        if x_scale == "log":
            x = x[1:]  # remove zero point data
            y = y[1:]
            errorY = errorY[1:]

        FP = curve_fit_lims(
            fit_func,
            fit_param_estimate,
            x,
            y,
            errorY,
            x_lim=fit_range,
            no_plot=no_plot,
            x_scale=x_scale,
            y_scale=y_scale,
        )

        # if ~isempty(DynRange), % do dynamic range iterations
        #     for iter = 1:MaxIter;
        #         fit_range(1) = FP.beta(DynRangeBetaParam)/DynRange(1);
        #         fit_range(2) = FP.beta(DynRangeBetaParam)*DynRange(end);
        #         J = find((X>fit_range(1)) & (X < fit_range(2)));
        #         FP = nlinfitWeight2(X(J), Y(J), fit_func, FP.beta, errorY(J), param, lsqcurvefit_param{:});
        #     end
        # end

        # FP.fit_range = fit_range;
        # FP.constParam = param;

        # self.do_plot_corr_func()
        if not hasattr(self, "fit_param"):
            self.fit_param = dict()

        self.fit_param[FP["fit_func"]] = FP


class CorrFuncTDC(CorrFuncData):
    """Doc."""

    def __init__(self):
        self.data = {
            "version": 2,
            "line_end_adder": 1000,
            "data": list(),
        }  # dictionary to hold the data

        self.tdc_calib = dict()  # dictionary for TDC calibration

        self.is_data_on_disk = False  # saving data on disk to free RAM
        self.data_filename_on_disk = ""
        self.data_name_on_disk = ""
        self.after_pulse_param = (
            "multi_exponent_fit",
            1e5
            * np.array(
                [
                    0.183161051158731,
                    0.021980256326163,
                    6.882763042785681,
                    0.154790280034295,
                    0.026417532300439,
                    0.004282749744374,
                    0.001418363840077,
                    0.000221275818533,
                ]
            ),
        )

    def read_fpga_data(
        self,
        fpathtmpl,
        fix_shift=False,
        line_end_adder=1000,
        roi_selection="auto",
        verbose=False,
    ):
        """Doc."""

        if verbose:
            print("Loading FPGA data...")
        # fpathtmpl : template of complete path to data

        if not (fpathes := glob.glob(fpathtmpl)):
            logging.warning("No files found! Check file template!")

        # order filenames -
        # splits in folderpath and filename template
        folderpath_fnametmpl = os.path.split(fpathtmpl)
        # splits filename template into the name template proper and its extension
        fnametmpl, file_extension = os.path.splitext(folderpath_fnametmpl[1])

        # Nfile = list()
        # for fpath in fpathes:
        #     folderpath_fname = os.path.split(fpath) #splits in folderpath and filename
        #     fname, file_extension = os.path.splitext(folderpath_fname[1]) # splits filename into the name proper and its extension
        #     temp = re.split(fnametmpl, fname)
        #     Nfile.append(int(temp[1])) #file number
        # Nfile = np.array(Nfile)
        # J = np.argsort(Nfile)
        # fpathes = fpathes[J]
        fpathes.sort(key=lambda fpath: re.split(fpathtmpl[:-4], os.path.splitext(fpath)[0])[1])

        for fpath in fpathes:
            if verbose:
                print(f"Loading {fpath}")
            # splits in folderpath and filename
            folderpath_fname = os.path.split(fpath)
            # splits filename into the name proper and its extension
            fname, file_extension = os.path.splitext(folderpath_fname[1])

            # test type of file and open accordingly
            if file_extension == ".mat":
                filedict = loadmat(fpath)
                if "system_info" in filedict.keys():
                    system_info = filedict["system_info"]
                    if "after_pulse_param" in system_info.keys():
                        self.after_pulse_param = system_info["after_pulse_param"]
                # TODO: add python loading

                # patching for new parameters

                if isinstance(self.after_pulse_param, np.ndarray):
                    self.after_pulse_param = (
                        "multi_exponent_fit",
                        np.array(
                            1e5
                            * [
                                0.183161051158731,
                                0.021980256326163,
                                6.882763042785681,
                                0.154790280034295,
                                0.026417532300439,
                                0.004282749744374,
                                0.001418363840077,
                                0.000221275818533,
                            ]
                        ),
                    )

                full_data = filedict["full_data"]
                p = PhotonData()
                p.convert_fpga_data_to_photons(
                    np.array(full_data["data"]).astype("B"), version=full_data["version"]
                )
                self.laser_freq = full_data["laser_freq"] * 1e6
                self.fpga_freq = full_data["fpga_freq"] * 1e6

                if "circle_speed_um_sec" in full_data.keys():
                    self.v_um_ms = full_data["circle_speed_um_sec"] / 1000  # to um/ms

                if "angular_scan_settings" in full_data.keys():  # angular scan
                    angular_scan_settings = full_data["angular_scan_settings"]
                    angular_scan_settings["linear_part"] = np.array(
                        angular_scan_settings["linear_part"]
                    )
                    p.line_end_adder = line_end_adder
                    self.v_um_ms = angular_scan_settings["actual_speed"] / 1000  # to um/ms
                    runtime = p.runtime
                    cnt, n_pix_tot, n_pix, line_num = self.convert_angular_scan_to_image(
                        runtime, angular_scan_settings
                    )

                    # n_pix_tot = np.floor(runtime*angular_scan_settings['sample_freq']/self.laser_freq)
                    # n_pix = mod(n_pix_tot, angular_scan_settings['points_per_line_total']) #to which pixel photon belongs
                    # line_num_tot = np.floor(n_pix_tot/angular_scan_settings['points_per_line_total'])
                    # line_num = np.mod(line_num_tot, angular_scan_settings['n_lines']+1) # one more line is for return to starting positon
                    # cnt = np.empty((angular_scan_settings['n_lines']+1, angular_scan_settings['points_per_line_total']), dtype = 'int')
                    # for j in range(angular_scan_settings['n_lines']+1):
                    #     belong_to_line = (line_num.astype('int') == j)
                    #     cnt_line, bins = np.histogram(n_pix[belong_to_line].astype('int'),
                    #                         bins = np.arange(-0.5, angular_scan_settings['points_per_line_total']))
                    #     cnt[j, :] = cnt_line

                    if fix_shift:
                        score = list()
                        pix_shifts = list()
                        min_pix_shift = -np.round(cnt.shape[1] / 2)
                        max_pix_shift = min_pix_shift + cnt.shape[1] + 1
                        for pix_shift in np.arange(min_pix_shift, max_pix_shift).astype(int):
                            cnt2 = np.roll(cnt, pix_shift)
                            diff_cnt2 = (cnt2[:-1:2, :] - np.flip(cnt2[1::2, :], 1)) ** 2
                            score.append(diff_cnt2.sum())
                            pix_shifts.append(pix_shift)
                        score = np.array(score)
                        pix_shifts = np.array(pix_shifts)
                        pix_shift = pix_shifts[score.argmin()]

                        runtime = (
                            p.runtime
                            + pix_shift * self.laser_freq / angular_scan_settings["sample_freq"]
                        )
                        (
                            cnt,
                            n_pix_tot,
                            n_pix,
                            line_num,
                        ) = self.convert_angular_scan_to_image(runtime, angular_scan_settings)

                    else:
                        pix_shift = 0

                    # invert every second line
                    cnt[1::2, :] = np.flip(cnt[1::2, :], 1)

                    if roi_selection == "auto":
                        classes = 4
                        thresh = skifilt.threshold_multiotsu(
                            skifilt.median(cnt), classes
                        )  # minor filtering of outliers
                        cnt_dig = np.digitize(cnt, bins=thresh)
                        plateau_lvl = np.median(cnt[cnt_dig == (classes - 1)])
                        std_plateau = stats.median_absolute_deviation(cnt[cnt_dig == (classes - 1)])
                        dev_cnt = cnt - plateau_lvl
                        bw = dev_cnt > -std_plateau
                        bw = ndimage.binary_fill_holes(bw)
                        disk_open = morphology.selem.disk(radius=3)
                        bw = morphology.opening(bw, selem=disk_open)

                    # elif  roi_selection == 'manual'):
                    #      subplot(1, 1, 1)
                    #     imagesc(cnt)
                    #     title('Use polygon tool to make selection', 'FontSize', 20)
                    #     figure(gcf)

                    #     if exist('roi'),
                    #         hPoly = impoly(gca, roi);
                    #     else
                    #         hPoly = impoly;
                    #     end;
                    #     bw = hPoly.createMask;
                    #     roi = hPoly.getPosition;

                    bw[1::2, :] = np.flip(bw[1::2, :], 1)
                    # cut edges
                    bw_temp = np.zeros(bw.shape)
                    bw_temp[:, angular_scan_settings["linear_part"].astype("int")] = bw[
                        :, angular_scan_settings["linear_part"].astype("int")
                    ]
                    bw = bw_temp
                    # discard short and fill long rows
                    m2 = np.sum(bw, axis=1)
                    bw[m2 < 0.5 * m2.max(), :] = False
                    line_starts = np.array([], dtype="int")
                    line_stops = np.array([], dtype="int")
                    line_start_lables = np.array([], dtype="int")
                    line_stop_labels = np.array([], dtype="int")

                    if not (line_end_adder > bw.shape[0]):
                        logging.warning(
                            "Number of lines larger than line_end_adder! Increase line_end_adder."
                        )

                    roi = {"row": list(), "col": list()}
                    for j in range(bw.shape[0]):
                        k = bw[j, :].nonzero()
                        if k[0].size > 0:  # k is a tuple
                            bw[j, k[0][0] : k[0][-1]] = True
                            roi["row"].insert(0, j)
                            roi["col"].insert(0, k[0][0])
                            roi["row"].append(j)
                            roi["col"].append(k[0][-1])

                            line_starts_new_index = np.ravel_multi_index(
                                (j, k[0][0]), bw.shape, mode="raise", order="C"
                            )
                            line_starts_new = np.arange(
                                line_starts_new_index, n_pix_tot[-1], bw.size
                            )
                            line_stops_new_index = np.ravel_multi_index(
                                (j, k[0][-1]), bw.shape, mode="raise", order="C"
                            )
                            line_stops_new = np.arange(line_stops_new_index, n_pix_tot[-1], bw.size)
                            line_start_lables = np.append(
                                line_start_lables, (-j * np.ones(line_starts_new.shape))
                            )
                            line_stop_labels = np.append(
                                line_stop_labels,
                                ((-j - line_end_adder) * np.ones(line_stops_new.shape)),
                            )
                            line_starts = np.append(line_starts, line_starts_new)
                            line_stops = np.append(line_stops, line_stops_new)

                    # repeat first point to close the polygon
                    roi["row"].append(roi["row"][0])
                    roi["col"].append(roi["col"][0])

                    line_start_lables = np.array(line_start_lables)
                    line_stop_labels = np.array(line_stop_labels)
                    line_starts = np.array(line_starts)
                    line_stops = np.array(line_stops)
                    runtime_line_starts = np.round(
                        line_starts * self.laser_freq / angular_scan_settings["sample_freq"]
                    )
                    runtime_line_stops = np.round(
                        line_stops * self.laser_freq / angular_scan_settings["sample_freq"]
                    )

                    runtime = np.hstack((runtime_line_starts, runtime_line_stops, runtime))
                    j_sort = np.argsort(runtime)
                    runtime = runtime[j_sort]
                    line_num = np.hstack(
                        (
                            line_start_lables,
                            line_stop_labels,
                            line_num * bw[line_num, n_pix].flatten(),
                        )
                    )
                    p.line_num = line_num[j_sort]
                    coarse = np.hstack(
                        (
                            np.full(runtime_line_starts.shape, np.nan),
                            np.full(runtime_line_stops.shape, np.nan),
                            p.coarse,
                        )
                    )
                    p.coarse = coarse[j_sort]
                    coarse = np.hstack(
                        (
                            np.full(runtime_line_starts.shape, np.nan),
                            np.full(runtime_line_stops.shape, np.nan),
                            p.coarse2,
                        )
                    )
                    p.coarse2 = coarse[j_sort]
                    fine = np.hstack(
                        (
                            np.full(runtime_line_starts.shape, np.nan),
                            np.full(runtime_line_stops.shape, np.nan),
                            p.fine,
                        )
                    )
                    p.fine = fine[j_sort]
                    p.runtime = runtime

                    # add line starts/stops to photon runtimes

                    self.angular_scan_settings = angular_scan_settings
                    plt.subplot(1, 1, 1)
                    plt.imshow(cnt)
                    p.image = cnt
                    folderpath_fname = os.path.split(fpath)  # splits in folderpath and filename
                    plt.title(folderpath_fname[1])
                    # temporarily reverse rows again
                    bw[1::2, :] = np.flip(bw[1::2, :], 1)
                    p.bw_mask = bw
                    # roi1 = segmentation.find_boundaries(bw)
                    roi1 = {"row": np.array(roi["row"]), "col": np.array(roi["col"])}
                    plt.plot(roi1["col"], roi1["row"], color="white")
                    # now back
                    bw[1::2, :] = np.flip(bw[1::2, :], 1)
                    roi2 = {"row": np.array(roi["row"]), "col": np.array(roi["col"])}
                    plt.plot(roi2["col"], roi2["row"], color="white")

                    # get image line correlation to subtract trends
                    img = p.image * p.bw_mask
                    # lineNos = find(sum(p.bw_mask, 2));
                    p.image_line_corr = list()
                    for j in range(roi1["row"].min(), roi1["row"].max() + 1):
                        prof = img[j]
                        prof = prof[p.bw_mask[j] > 0]
                        c, lags = do_x_corr(prof, prof)
                        c = c / prof.mean() ** 2 - 1
                        c[0] = c[0] - 1 / prof.mean()  # subtracting shot noise, small stuff really
                        p.image_line_corr.append(
                            {
                                "lag": lags * 1000 / angular_scan_settings["sample_freq"],  # in ms
                                "corrfunc": c,
                            }
                        )  # c/mean(prof).^2-1;

                    plt.show()
                    plt.figure()

            # for key in self.data:
            #     if not hasattr(p, key):
            #         setattr(p, key, self.data[key])

            #                p = orderfields(p, obj.data);
            p.fname = fname
            p.fpath = fpath
            self.data["data"].append(p)

    def convert_angular_scan_to_image(self, runtime, angular_scan_settings):
        """utility function for opening Angular Scans"""

        n_pix_tot = np.floor(
            runtime * angular_scan_settings["sample_freq"] / self.laser_freq
        ).astype("int")
        n_pix = np.mod(n_pix_tot, angular_scan_settings["points_per_line_total"]).astype(
            "int"
        )  # to which pixel photon belongs
        line_num_tot = np.floor(n_pix_tot / angular_scan_settings["points_per_line_total"]).astype(
            "int"
        )
        line_num = np.mod(line_num_tot, angular_scan_settings["n_lines"] + 1).astype(
            "int"
        )  # one more line is for return to starting positon
        cnt = np.empty(
            (
                angular_scan_settings["n_lines"] + 1,
                angular_scan_settings["points_per_line_total"],
            ),
            dtype="int",
        )
        for j in range(angular_scan_settings["n_lines"] + 1):
            belong_to_line = line_num.astype("int") == j
            cnt_line, bins = np.histogram(
                n_pix[belong_to_line].astype("int"),
                bins=np.arange(-0.5, angular_scan_settings["points_per_line_total"]),
            )
            cnt[j, :] = cnt_line

        return cnt, n_pix_tot, n_pix, line_num

    def correlate_regular_data(
        self,
        run_duration=-1,
        min_time_frac=0.5,
        max_outlier_prob=1e-5,
        n_runs_requested=60,
        verbose=False,
    ):

        # if self.is_data_on_disk:
        #     self.DoLoadDataFromDisk
        #     DumpDataToDiskAfter = True
        # else:
        #     DumpDataToDiskAfter = False

        cf = SoftwareCorrelator()

        if run_duration < 0:  # auto determination of run duration
            total_duration_estimate = 0
            for p in self.data["data"]:
                time_stamps = np.diff(p.runtime)
                mu = np.median(time_stamps) / np.log(2)
                total_duration_estimate = (
                    total_duration_estimate + mu * len(p.runtime) / self.laser_freq
                )

            run_duration = total_duration_estimate / n_runs_requested
            if verbose:
                print(f"Auto determination of run duration: {run_duration} s")

        self.requested_duration = run_duration
        self.min_duration_frac = min_time_frac
        self.duration = np.array([])
        self.lag = np.array([])
        self.corrfunc = list()
        self.weights = list()
        self.cf_cr = list()
        self.countrate = np.array([])

        self.total_duration_skipped = 0

        for p in self.data["data"]:

            if verbose:
                print(f"Correlating {p.fname}...")
            # find additional outliers
            time_stamps = np.diff(p.runtime)
            # for exponential distribution MEDIAN and MAD are the same, but for
            # biexponential MAD seems more sensitive
            mu = np.maximum(
                np.median(time_stamps), np.abs(time_stamps - time_stamps.mean()).mean()
            ) / np.log(2)
            max_time_stamp = stats.expon.ppf(1 - max_outlier_prob / len(time_stamps), scale=mu)
            sec_edges = np.asarray(time_stamps > max_time_stamp).nonzero()[0]
            no_outliers = len(sec_edges)
            if no_outliers > 0:
                if verbose:
                    print(f"{no_outliers} of all outliers")

            sec_edges = np.append(np.insert(sec_edges, 0, 0), len(time_stamps))
            p.all_section_edges = np.array([sec_edges[:-1], sec_edges[1:]]).T

            for j, se in enumerate(p.all_section_edges):

                # split into segments of approx time of run_duration
                segment_time = (p.runtime[se[1]] - p.runtime[se[0]]) / self.laser_freq
                if segment_time < min_time_frac * run_duration:
                    if verbose:
                        print(
                            f"Duration of segment No. {j} of file {p.fname} is {segment_time}s: too short. Skipping..."
                        )
                    self.total_duration_skipped = self.total_duration_skipped + segment_time
                    continue

                n_splits = np.ceil(segment_time / run_duration).astype("int")
                splits = np.linspace(0, np.diff(se)[0].astype("int"), n_splits + 1, dtype="int")
                ts = time_stamps[se[0] : se[1]]

                for k in range(n_splits):

                    ts_split = ts[splits[k] : splits[k + 1]]
                    self.duration = np.append(self.duration, ts_split.sum() / self.laser_freq)
                    cf.do_soft_cross_correlator(
                        ts_split,
                        CorrelatorType.PH_DELAY_CORRELATOR,
                        timebase_ms=1000 / self.laser_freq,
                    )  # time base of 20MHz to ms
                    if len(self.lag) < len(cf.lag):
                        self.lag = cf.lag
                        if self.after_pulse_param[0] == "multi_exponent_fit":
                            # work with any number of exponents
                            # y = beta(1)*exp(-beta(2)*t) + beta(3)*exp(-beta(4)*t) + beta(5)*exp(-beta(6)*t);
                            beta = self.after_pulse_param[1]
                            self.after_pulse = np.dot(
                                beta[::2], np.exp(-np.outer(beta[1::2], self.lag))
                            )

                    self.corrfunc.append(cf.corrfunc)
                    self.weights.append(cf.weights)
                    self.countrate = np.append(self.countrate, cf.countrate)
                    self.cf_cr.append(
                        cf.countrate * cf.corrfunc - self.after_pulse[: cf.corrfunc.size]
                    )

        # zero pad
        for ind in range(len(self.corrfunc)):
            pad_len = len(self.lag) - len(self.corrfunc[ind])
            self.corrfunc[ind] = np.pad(self.corrfunc[ind], (0, pad_len), "constant")
            self.weights[ind] = np.pad(self.weights[ind], (0, pad_len), "constant")
            self.cf_cr[ind] = np.pad(self.cf_cr[ind], (0, pad_len), "constant")

        self.corrfunc = np.array(self.corrfunc)
        self.weights = np.array(self.weights)
        self.cf_cr = np.array(self.cf_cr)

        self.total_duration = self.duration.sum()
        if verbose:
            print(f"{self.total_duration_skipped}s skipped out of {self.total_duration}s.")

        # if ~isempty(obj.v_um_ms)
        #     obj.DoSetVelocity(obj.v_um_ms);
        # end

        # if DumpDataToDiskAfter
        #     obj.DoDumpDataToDisk,
        # end


def do_x_corr(a, b):
    """Does correlation similar to Matlab xcorr, cuts positive lags, normalizes properly"""

    if a.size != b.size:
        logging.warning("For unequal lengths of a, b the meaning of lags is not clear!")
    c = np.correlate(a, b, mode="full")
    c = c[np.floor(c.size / 2).astype("int") :]
    lags = np.arange(c.size)

    return c, lags