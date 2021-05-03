#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 17:20:00 2018

@author: Oleg Krichevsky okrichev@bgu.ac.il
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

# from CellTrackModule import CellTrackClass

warnings.simplefilter("error", OptimizeWarning)
warnings.simplefilter("error", RuntimeWarning)


def curvefitLims(
    fitName,
    xs,
    ys,
    ysErrs,
    XLim=[],
    YLim=[],
    NoPlot=False,
    XScale="log",
    YScale="linear",
    paramEstimates=None,
):
    """Doc."""

    if len(YLim) == 0:
        YLim = np.array([np.NINF, np.Inf])
    elif len(YLim) == 1:
        YLim = np.array([YLim, np.Inf])

    if len(XLim) == 0:
        XLim = np.array([np.NINF, np.Inf])
    elif len(XLim) == 1:
        XLim = np.array([XLim, np.Inf])

    inLims = (xs >= XLim[0]) & (xs <= XLim[1]) & (ys >= YLim[0]) & (ys <= YLim[1])
    isfiniteErr = (ysErrs > 0) & np.isfinite(ysErrs)
    # print(inLims)
    x = xs[inLims & isfiniteErr]
    #   print(x)
    y = ys[inLims & isfiniteErr]
    yErr = ysErrs[inLims & isfiniteErr]
    #    print(yErr)
    fitFunc = globals()[fitName]
    #    print(fitName)
    bta, covM = curve_fit(
        fitFunc, x, y, p0=paramEstimates, sigma=yErr, absolute_sigma=True
    )
    #    print(bta)
    FitParam = dict()
    FitParam["beta"] = bta
    FitParam["errorBeta"] = np.sqrt(np.diag(covM))
    chiSqArray = np.square((fitFunc(x, *bta) - y) / yErr)
    FitParam["chiSqNorm"] = chiSqArray.sum() / x.size
    FitParam["x"] = x
    FitParam["y"] = y
    FitParam["XLim"] = XLim
    FitParam["YLim"] = YLim
    FitParam["FitFunc"] = fitName
    # print(FitParam)

    if not NoPlot:
        plt.errorbar(xs, ys, ysErrs, fmt=".")
        plt.plot(xs[inLims], fitFunc(xs[inLims], *bta), zorder=10)
        plt.xscale(XScale)
        plt.yscale(YScale)
        plt.gcf().canvas.draw()
        plt.autoscale()
        plt.show()

    return FitParam


# Fit functions
def LinearFit(t, a, b):
    return a * t + b


def PowerFit(t, a, n):
    return a * t ** n


def Diffusion3Dfit(t, A, tau, wSq):
    return A / (1 + t / tau) / np.sqrt(1 + t / tau / wSq)


def WeightedAverage(ListOfDataVectors, ListOfDataErrors):
    """Doc."""

    DataVectors = np.array(ListOfDataVectors)
    DataErrors = np.array(ListOfDataErrors)
    weights = DataErrors ** (-2)
    WA = np.sum(DataVectors * weights, 0) / np.sum(weights, 0)
    WE = np.sum(weights, 0) ** (-2)
    return WA, WE
