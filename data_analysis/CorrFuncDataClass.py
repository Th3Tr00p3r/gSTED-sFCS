#!/usr/bin/env python3
"""
Created on Fri Apr 23 12:11:40 2021

@author: oleg
"""
import os
import sys

import matplotlib.pyplot as plt
import numba as nb
import numpy as np

from data_analysis.FitTools import curvefitLims

sys.path.append(os.path.dirname(__file__))


class CorrFuncDataClass:
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
        fit_func="Diffusion3Dfit",
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

        FP = curvefitLims(
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
