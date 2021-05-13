#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    def DoAverageCorr(
        self,
        Rejection=2,
        NormRange=np.array([1e-3, 2e-3]),
        DeleteList=[],
        NoPlot=False,
        use_numba=False,
    ):

        # enable 'use_numba' for speed-up
        def calc_weighted_avg(CF_CR, weights):
            """Doc."""

            tot_weights = weights.sum(0)

            AverageCF_CR = (CF_CR * weights).sum(0) / tot_weights

            errorCF_CR = (
                np.sqrt((weights ** 2 * (CF_CR - AverageCF_CR) ** 2).sum(0))
                / tot_weights
            )

            return AverageCF_CR, errorCF_CR

        self.Rejection = Rejection
        self.NormRange = NormRange
        self.DeleteList = DeleteList
        self.AverageAllCF_CR = (self.CF_CR * self.weights).sum(0) / self.weights.sum(0)
        self.MedianAllCF_CR = np.median(self.CF_CR, 0)
        JJ = np.logical_and(
            (self.lag > NormRange[1]), (self.lag < 100)
        )  # work in the relevant part
        self.score = (
            (1 / np.var(self.CF_CR[:, JJ], 0))
            * (self.CF_CR[:, JJ] - self.MedianAllCF_CR[JJ]) ** 2
            / len(JJ)
        )
        if len(DeleteList) == 0:
            self.Jgood = np.where(self.score < self.Rejection)[0]
            self.Jbad = np.where(self.score >= self.Rejection)[0]
        else:
            self.Jbad = DeleteList
            self.Jgood = np.array(
                [i for i in range(self.CF_CR.shape[0]) if i not in DeleteList]
            ).astype("int")

        if use_numba:
            func = nb.njit(calc_weighted_avg, cache=True)
        else:
            func = calc_weighted_avg

        self.AverageCF_CR, self.errorCF_CR = func(
            self.CF_CR[self.Jgood, :], self.weights[self.Jgood, :]
        )

        Jt = np.logical_and(
            (self.lag > self.NormRange[0]), (self.lag < self.NormRange[1])
        )
        self.G0 = (self.AverageCF_CR[Jt] / self.errorCF_CR[Jt] ** 2).sum() / (
            1 / self.errorCF_CR[Jt] ** 2
        ).sum()
        self.Normalized = self.AverageCF_CR / self.G0
        self.errorNormalized = self.errorCF_CR / self.G0
        if not NoPlot:
            self.DoPlotCorrFunc()

    def DoPlotCorrFunc(
        self, Xfield="lag", Yfield="AverageCF_CR", Xscale="log", Yscale="linear"
    ):
        X = getattr(self, Xfield)
        Y = getattr(self, Yfield)
        if Xscale == "log":
            X = X[1:]  # remove zero point data
            Y = Y[1:]
        plt.plot(X, Y, "o")  # skip 0 lag time
        plt.xlabel(Xfield)
        plt.ylabel(Yfield)
        plt.gca().set_xscale(Xscale)
        plt.gca().set_yscale(Yscale)
        plt.show()

    def DoFit(
        self,
        Xfield="lag",
        Yfield="AverageCF_CR",
        YerrorField="errorCF_CR",
        FitFunc="Diffusion3Dfit",
        ConstantParam={},
        FitParamEstimate=[5000, 0.04, 30],
        FitRange=[1e-3, 100],
        NoPlot=False,
        Xscale="log",
        Yscale="linear",
    ):

        # [DynRange, IsDynRangeInput] = ParseInputs('Dynamic Range', [], varargin); % variable FitRange: give either a scalar or a vector of two
        # DynRangeBetaParam = ParseInputs('Dynamic Range parameter', 2, varargin); %the parameter in beta on which to do dynamic range
        # MaxIter = 3;
        # lsqcurvefitParam = ParseInputs('lsqcurvefitParam', {}, varargin);

        X = getattr(self, Xfield)
        Y = getattr(self, Yfield)
        errorY = getattr(self, YerrorField)
        if Xscale == "log":
            X = X[1:]  # remove zero point data
            Y = Y[1:]
            errorY = errorY[1:]

        FP = curvefitLims(
            FitFunc,
            X,
            Y,
            errorY,
            XLim=FitRange,
            NoPlot=NoPlot,
            XScale=Xscale,
            YScale=Yscale,
            paramEstimates=FitParamEstimate,
        )

        # if ~isempty(DynRange), % do dynamic range iterations
        #     for iter = 1:MaxIter;
        #         FitRange(1) = FP.beta(DynRangeBetaParam)/DynRange(1);
        #         FitRange(2) = FP.beta(DynRangeBetaParam)*DynRange(end);
        #         J = find((X>FitRange(1)) & (X < FitRange(2)));
        #         FP = nlinfitWeight2(X(J), Y(J), FitFunc, FP.beta, errorY(J), param, lsqcurvefitParam{:});
        #     end
        # end

        # FP.FitRange = FitRange;
        # FP.constParam = param;

        # self.DoPlotCorrFunc()
        if not hasattr(self, "FitParam"):
            self.FitParam = dict()

        self.FitParam[FP["FitFunc"]] = FP
