#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 12:11:40 2021

@author: oleg
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(__file__))
# sys.path.append('//Users/oleg/Documents/Python programming/FCS Python/')

# from PhotonDataClass import PhotonDataClass


class CorrFuncDataClass:
    #    def __init__(self):
    def DoAverageCorr(
        self, Rejection=2, NormRange=np.array([1e-3, 2e-3]), DeleteList=[], NoPlot=False
    ):
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

        self.AverageCF_CR = (
            self.CF_CR[self.Jgood, :] * self.weights[self.Jgood, :]
        ).sum(0) / self.weights[self.Jgood, :].sum(0)
        sig = self.CF_CR[self.Jgood, :] - self.AverageCF_CR
        self.errorCF_CR = np.sqrt(
            (self.weights[self.Jgood, :] ** 2 * sig ** 2).sum(0)
        ) / self.weights[self.Jgood, :].sum(0)
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

    def DoPlotCorrFunc(self):
        plt.semilogx(self.lag[1:], self.AverageCF_CR[1:])  # skip 0 lag time
        plt.xlabel("lag (ms)")
        plt.ylabel("G (cps)")
        plt.autoscale()
        plt.show()
