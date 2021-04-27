#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:38:27 2021

@author: oleg
"""
import glob
import os
import re  # regular expressions
import sys
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, stats
from skimage import filters as skifilt
from skimage import morphology

from data_analysis.CorrFuncDataClass import CorrFuncDataClass
from data_analysis.MatlabUtilities import (
    loadmat,  # loads matfiles with structures (scipy.io does not do structures)
)
from data_analysis.PhotonDataClass import PhotonDataClass
from data_analysis.SoftwareCorrelatorModule import (
    CorrelatorType,
    SoftwareCorrelatorClass,
)

sys.path.append(os.path.dirname(__file__))
# sys.path.append('//Users/oleg/Documents/Python programming/FCS Python/')


class CorrFuncTDCclass(CorrFuncDataClass):
    def __init__(self):
        self.data = {
            "Version": 2,
            "LineEndAdder": 1000,
            "Data": list(),
        }  # dictionary to hold the data

        self.TDCcalib = dict()  # dictionary for TDC calibration

        self.IsDataOnDisk = False  # saving data on disk to free RAM
        self.DataFileNameOnDisk = ""
        self.DataNameOnDisk = ""
        self.AfterPulseParam = (
            "MultiExponentFit",
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

    def DoReadFPGAdata(
        self, fpathtmpl, FixShift=False, LineEndAdder=1000, ROISelection="auto"
    ):
        print("Loading FPGA data...")
        # fpathtmpl : template of complete path to data

        fpathes = glob.glob(fpathtmpl)
        assert fpathes, "No files found! Check file template!"

        # order filenames
        folderpath_fnametmpl = os.path.split(
            fpathtmpl
        )  # splits in folderpath and filename template
        fnametmpl, file_extension = os.path.splitext(
            folderpath_fnametmpl[1]
        )  # splits filename template into the name template proper and its extension

        # Nfile = list()
        # for fpath in fpathes:
        #     folderpath_fname = os.path.split(fpath) #splits in folderpath and filename
        #     fname, file_extension = os.path.splitext(folderpath_fname[1]) # splits filename into the name proper and its extension
        #     temp = re.split(fnametmpl, fname)
        #     Nfile.append(int(temp[1])) #file number
        # Nfile = np.array(Nfile)
        # J = np.argsort(Nfile)
        # fpathes = fpathes[J]
        fpathes.sort(
            key=lambda fpath: re.split(fpathtmpl[:-4], os.path.splitext(fpath)[0])[1]
        )

        for fpath in fpathes:
            print("Loading " + fpath)
            folderpath_fname = os.path.split(fpath)  # splits in folderpath and filename
            fname, file_extension = os.path.splitext(
                folderpath_fname[1]
            )  # splits filename into the name proper and its extension

            # test type of file and open accordingly
            if file_extension == ".mat":
                filedict = loadmat(fpath)
                if "SystemInfo" in filedict.keys():
                    SystemInfo = filedict["SystemInfo"]
                    if "AfterPulseParam" in SystemInfo.keys():
                        self.AfterPulseParam = SystemInfo["AfterPulseParam"]

                # patching for new parameters

                if isinstance(self.AfterPulseParam, np.ndarray):
                    self.AfterPulseParam = (
                        "MultiExponentFit",
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

                FullData = filedict["FullData"]
                P = PhotonDataClass()
                P.DoConvertFPGAdataToPhotons(
                    np.array(FullData["Data"]).astype("B"), Version=FullData["Version"]
                )
                self.LaserFreq = FullData["LaserFreq"] * 1e6
                self.FPGAfreq = FullData["FpgaFreq"] * 1e6

                if "CircleSpeed_um_sec" in FullData.keys():
                    self.V_um_ms = FullData["CircleSpeed_um_sec"] / 1000  # to um/ms

                if "AnglularScanSettings" in FullData.keys():  # angular scan
                    AnglularScanSettings = FullData["AnglularScanSettings"]
                    AnglularScanSettings["LinearPart"] = np.array(
                        AnglularScanSettings["LinearPart"]
                    )
                    P.LineEndAdder = LineEndAdder
                    self.V_um_ms = (
                        AnglularScanSettings["ActualSpeed"] / 1000
                    )  # to um/ms
                    rntime = P.runtime
                    Cnt, PixNoTot, pixNumber, LineNo = self.DoConvertAngularScanToImage(
                        rntime, AnglularScanSettings
                    )

                    # PixNoTot = np.floor(rntime*AnglularScanSettings['SampleFreq']/self.LaserFreq)
                    # pixNumber = mod(PixNoTot, AnglularScanSettings['PointsPerLineTotal']) #to which pixel photon belongs
                    # LineNoTot = np.floor(PixNoTot/AnglularScanSettings['PointsPerLineTotal'])
                    # LineNo = np.mod(LineNoTot, AnglularScanSettings['NofLines']+1) # one more line is for return to starting positon
                    # Cnt = np.empty((AnglularScanSettings['NofLines']+1, AnglularScanSettings['PointsPerLineTotal']), dtype = 'int')
                    # for j in range(AnglularScanSettings['NofLines']+1):
                    #     belongToLine = (LineNo.astype('int') == j)
                    #     CntLine, bins = np.histogram(pixNumber[belongToLine].astype('int'),
                    #                         bins = np.arange(-0.5, AnglularScanSettings['PointsPerLineTotal']))
                    #     Cnt[j, :] = CntLine

                    if FixShift:
                        score = list()
                        pixShifts = list()
                        minPixShift = -np.round(Cnt.shape[1] / 2)
                        maxPixShift = minPixShift + Cnt.shape[1] + 1
                        for pixShift in np.arange(minPixShift, maxPixShift).astype(int):
                            Cnt2 = np.roll(Cnt, pixShift)
                            diffCnt2 = (Cnt2[:-1:2, :] - np.flip(Cnt2[1::2, :], 1)) ** 2
                            score.append(diffCnt2.sum())
                            pixShifts.append(pixShift)
                        score = np.array(score)
                        pixShifts = np.array(pixShifts)
                        pixShift = pixShifts[score.argmin()]

                        rntime = (
                            P.runtime
                            + pixShift
                            * self.LaserFreq
                            / AnglularScanSettings["SampleFreq"]
                        )
                        (
                            Cnt,
                            PixNoTot,
                            pixNumber,
                            LineNo,
                        ) = self.DoConvertAngularScanToImage(
                            rntime, AnglularScanSettings
                        )

                    else:
                        pixShift = 0

                    # invert every second line
                    Cnt[1::2, :] = np.flip(Cnt[1::2, :], 1)

                    if ROISelection == "auto":
                        classes = 4
                        thresh = skifilt.threshold_multiotsu(
                            skifilt.median(Cnt), classes
                        )  # minor filtering of outliers
                        CntDig = np.digitize(Cnt, bins=thresh)
                        plateauLevel = np.median(Cnt[CntDig == (classes - 1)])
                        stdPlateau = stats.median_absolute_deviation(
                            Cnt[CntDig == (classes - 1)]
                        )
                        devCnt = Cnt - plateauLevel
                        BW = devCnt > -stdPlateau
                        BW = ndimage.binary_fill_holes(BW)
                        Ropen = 3
                        diskOpen = morphology.selem.disk(Ropen)
                        BW = morphology.opening(BW, selem=diskOpen)

                    # elif  ROISelection == 'manual'):
                    #      subplot(1, 1, 1)
                    #     imagesc(Cnt)
                    #     title('Use polygon tool to make selection', 'FontSize', 20)
                    #     figure(gcf)

                    #     if exist('ROI'),
                    #         hPoly = impoly(gca, ROI);
                    #     else
                    #         hPoly = impoly;
                    #     end;
                    #     BW = hPoly.createMask;
                    #     ROI = hPoly.getPosition;

                    BW[1::2, :] = np.flip(BW[1::2, :], 1)
                    # cut edges
                    BWtemp = np.zeros(BW.shape)
                    BWtemp[:, AnglularScanSettings["LinearPart"].astype("int")] = BW[
                        :, AnglularScanSettings["LinearPart"].astype("int")
                    ]
                    BW = BWtemp
                    # discard short and fill long rows
                    M2 = np.sum(BW, axis=1)
                    BW[M2 < 0.5 * M2.max(), :] = False
                    LineStarts = np.array([], dtype="int")
                    LineStops = np.array([], dtype="int")
                    LineStartLabels = np.array([], dtype="int")
                    LineStopLabels = np.array([], dtype="int")

                    assert (
                        LineEndAdder > BW.shape[0]
                    ), "Number of lines larger than LineEndAdder! Increase LineEndAdder."

                    ROI = {"row": list(), "col": list()}
                    for j in range(BW.shape[0]):
                        k = BW[j, :].nonzero()
                        if k[0].size > 0:  # k is a tuple
                            BW[j, k[0][0] : k[0][-1]] = True
                            ROI["row"].insert(0, j)
                            ROI["col"].insert(0, k[0][0])
                            ROI["row"].append(j)
                            ROI["col"].append(k[0][-1])

                            LineStartsNewIndex = np.ravel_multi_index(
                                (j, k[0][0]), BW.shape, mode="raise", order="C"
                            )
                            LineStartsNew = np.arange(
                                LineStartsNewIndex, PixNoTot[-1], BW.size
                            )
                            LineStopsNewIndex = np.ravel_multi_index(
                                (j, k[0][-1]), BW.shape, mode="raise", order="C"
                            )
                            LineStopsNew = np.arange(
                                LineStopsNewIndex, PixNoTot[-1], BW.size
                            )
                            LineStartLabels = np.append(
                                LineStartLabels, (-j * np.ones(LineStartsNew.shape))
                            )
                            LineStopLabels = np.append(
                                LineStopLabels,
                                ((-j - LineEndAdder) * np.ones(LineStopsNew.shape)),
                            )
                            LineStarts = np.append(LineStarts, LineStartsNew)
                            LineStops = np.append(LineStops, LineStopsNew)

                    ROI["row"].append(
                        ROI["row"][0]
                    )  # repeat first point to close the polygon
                    ROI["col"].append(ROI["col"][0])
                    LineStartLabels = np.array(LineStartLabels)
                    LineStopLabels = np.array(LineStopLabels)
                    LineStarts = np.array(LineStarts)
                    LineStops = np.array(LineStops)
                    rntimeLineStarts = np.round(
                        LineStarts * self.LaserFreq / AnglularScanSettings["SampleFreq"]
                    )
                    rntimeLineStops = np.round(
                        LineStops * self.LaserFreq / AnglularScanSettings["SampleFreq"]
                    )

                    rntime = np.hstack((rntimeLineStarts, rntimeLineStops, rntime))
                    Jsort = np.argsort(rntime)
                    rntime = rntime[Jsort]
                    LineNo = np.hstack(
                        (
                            LineStartLabels,
                            LineStopLabels,
                            LineNo * BW[LineNo, pixNumber].flatten(),
                        )
                    )
                    P.LineNo = LineNo[Jsort]
                    Coarse = np.hstack(
                        (
                            np.full(rntimeLineStarts.shape, np.nan),
                            np.full(rntimeLineStops.shape, np.nan),
                            P.coarse,
                        )
                    )
                    P.coarse = Coarse[Jsort]
                    Coarse = np.hstack(
                        (
                            np.full(rntimeLineStarts.shape, np.nan),
                            np.full(rntimeLineStops.shape, np.nan),
                            P.coarse2,
                        )
                    )
                    P.coarse2 = Coarse[Jsort]
                    Fine = np.hstack(
                        (
                            np.full(rntimeLineStarts.shape, np.nan),
                            np.full(rntimeLineStops.shape, np.nan),
                            P.fine,
                        )
                    )
                    P.fine = Fine[Jsort]
                    P.runtime = rntime

                    # add line starts/stops to photon runtimes

                    self.AnglularScanSettings = AnglularScanSettings
                    plt.subplot(1, 1, 1)
                    plt.imshow(Cnt)
                    P.Image = Cnt
                    folderpath_fname = os.path.split(
                        fpath
                    )  # splits in folderpath and filename
                    plt.title(folderpath_fname[1])
                    # temporarily reverse rows again
                    BW[1::2, :] = np.flip(BW[1::2, :], 1)
                    P.BWmask = BW
                    # ROI1 = segmentation.find_boundaries(BW)
                    ROI1 = {"row": np.array(ROI["row"]), "col": np.array(ROI["col"])}
                    plt.plot(ROI1["col"], ROI1["row"], color="white")
                    # now back
                    BW[1::2, :] = np.flip(BW[1::2, :], 1)
                    ROI2 = {"row": np.array(ROI["row"]), "col": np.array(ROI["col"])}
                    plt.plot(ROI2["col"], ROI2["row"], color="white")

                    # get image line correlation to subtract trends
                    Img = P.Image * P.BWmask
                    # lineNos = find(sum(P.BWmask, 2));
                    P.ImageLineCorr = list()
                    for j in range(ROI1["row"].min(), ROI1["row"].max() + 1):
                        prof = Img[j]
                        prof = prof[P.BWmask[j] > 0]
                        C, lags = DoXcorr(prof, prof)
                        C = C / prof.mean() ** 2 - 1
                        C[0] = (
                            C[0] - 1 / prof.mean()
                        )  # subtracting shot noise, small stuff really
                        P.ImageLineCorr.append(
                            {
                                "lag": lags
                                * 1000
                                / AnglularScanSettings["SampleFreq"],  # in ms
                                "corrfunc": C,
                            }
                        )  # C/mean(prof).^2-1;

                    plt.show()
                    plt.figure()

            # for key in self.data:
            #     if not hasattr(P, key):
            #         setattr(P, key, self.data[key])

            #                P = orderfields(P, obj.data);
            P.fname = fname
            P.fpath = fpath
            self.data["Data"].append(P)

    def DoConvertAngularScanToImage(
        self, rntime, AnglularScanSettings
    ):  # utility function for opening Angular Scans
        PixNoTot = np.floor(
            rntime * AnglularScanSettings["SampleFreq"] / self.LaserFreq
        ).astype("int")
        pixNumber = np.mod(PixNoTot, AnglularScanSettings["PointsPerLineTotal"]).astype(
            "int"
        )  # to which pixel photon belongs
        LineNoTot = np.floor(
            PixNoTot / AnglularScanSettings["PointsPerLineTotal"]
        ).astype("int")
        LineNo = np.mod(LineNoTot, AnglularScanSettings["NofLines"] + 1).astype(
            "int"
        )  # one more line is for return to starting positon
        Cnt = np.empty(
            (
                AnglularScanSettings["NofLines"] + 1,
                AnglularScanSettings["PointsPerLineTotal"],
            ),
            dtype="int",
        )
        for j in range(AnglularScanSettings["NofLines"] + 1):
            belongToLine = LineNo.astype("int") == j
            CntLine, bins = np.histogram(
                pixNumber[belongToLine].astype("int"),
                bins=np.arange(-0.5, AnglularScanSettings["PointsPerLineTotal"]),
            )
            Cnt[j, :] = CntLine

        return Cnt, PixNoTot, pixNumber, LineNo

    def DoCorrelateRegularData(
        self, RunDuration=-1, MinTimeFrac=0.5, MaxOutlierProb=1e-5, NrunsRequested=60
    ):

        # if self.IsDataOnDisk:
        #     self.DoLoadDataFromDisk
        #     DumpDataToDiskAfter = True
        # else:
        #     DumpDataToDiskAfter = False

        CF = SoftwareCorrelatorClass()

        if RunDuration < 0:  # auto determination of run duration
            TotalDurationEstimate = 0
            for P in self.data["Data"]:
                time_stamps = np.diff(P.runtime)
                mu = np.median(time_stamps) / np.log(2)
                TotalDurationEstimate = (
                    TotalDurationEstimate + mu * len(P.runtime) / self.LaserFreq
                )

            RunDuration = TotalDurationEstimate / NrunsRequested
            print("Auto determination of run duration = " + str(RunDuration))

        self.RequestedDuration = RunDuration
        self.MinDurationFraction = MinTimeFrac
        self.duration = np.array([])
        self.lag = np.array([])
        self.corrfunc = list()
        self.weights = list()
        self.CF_CR = list()
        self.countrate = np.array([])

        self.TotalDurationSkipped = 0

        for P in self.data["Data"]:
            print("Correlating " + P.fname)
            # find additional outliers
            time_stamps = np.diff(P.runtime)
            mu = np.maximum(
                np.median(time_stamps), stats.median_abs_deviation(time_stamps)
            ) / np.log(
                2
            )  # for exponential distribution MEDIAN and MAD are the same, but for
            # biexponential MAD seems more sensitive
            maxTimeStamp = stats.expon.ppf(
                1 - MaxOutlierProb / len(time_stamps), scale=mu
            )
            secEdges = np.asarray(time_stamps > maxTimeStamp).nonzero()[0]
            NoOutliers = len(secEdges)
            if NoOutliers > 0:
                print(str(NoOutliers) + " of all outliers")

            secEdges = np.append(np.insert(secEdges, 0, 0), len(time_stamps))
            P.AllSectionEdges = np.array([secEdges[:-1], secEdges[1:]]).T

            for j, sE in enumerate(P.AllSectionEdges):
                # split into segments of approx time of RunDuration
                SegmentTime = (P.runtime[sE[1]] - P.runtime[sE[0]]) / self.LaserFreq
                if SegmentTime < MinTimeFrac * RunDuration:
                    print(
                        "Duration of segment No. "
                        + str(j)
                        + " of file "
                        + P.fname
                        + "is "
                        + str(SegmentTime)
                        + "s: too short. Skipping..."
                    )
                    self.TotalDurationSkipped = self.TotalDurationSkipped + SegmentTime
                    continue

                NoSplits = np.ceil(SegmentTime / RunDuration).astype("int")
                Splits = np.linspace(
                    0, np.diff(sE)[0].astype("int"), NoSplits + 1, dtype="int"
                )
                ts = time_stamps[sE[0] : sE[1]]

                for k in range(NoSplits):
                    ts_split = ts[Splits[k] : Splits[k + 1]]
                    self.duration = np.append(
                        self.duration, ts_split.sum() / self.LaserFreq
                    )
                    CF.DoSoftCrossCorrelator(
                        ts_split,
                        CorrelatorType.PhDelayCorrelator,
                        timebase_ms=1000 / self.LaserFreq,
                    )  # time base of 20MHz to ms
                    if len(self.lag) < len(CF.lag):
                        self.lag = CF.lag
                        if self.AfterPulseParam[0] == "MultiExponentFit":
                            # work with any number of exponents
                            # y = beta(1)*exp(-beta(2)*t) + beta(3)*exp(-beta(4)*t) + beta(5)*exp(-beta(6)*t);
                            beta = self.AfterPulseParam[1]
                            self.AfterPulse = np.dot(
                                beta[::2], np.exp(-np.outer(beta[1::2], self.lag))
                            )

                    self.corrfunc.append(CF.corrfunc)
                    self.weights.append(CF.weights)
                    self.countrate = np.append(self.countrate, CF.countrate)
                    self.CF_CR.append(
                        CF.countrate * CF.corrfunc - self.AfterPulse[: CF.corrfunc.size]
                    )

        for ind in range(len(self.corrfunc)):  # zero pad
            padLen = len(self.lag) - len(self.corrfunc[ind])
            self.corrfunc[ind] = np.pad(self.corrfunc[ind], (0, padLen), "constant")
            self.weights[ind] = np.pad(self.weights[ind], (0, padLen), "constant")
            self.CF_CR[ind] = np.pad(self.CF_CR[ind], (0, padLen), "constant")

        self.corrfunc = np.array(self.corrfunc)
        self.weights = np.array(self.weights)
        self.CF_CR = np.array(self.CF_CR)

        self.TotalDuration = self.duration.sum()
        print(
            str(self.TotalDurationSkipped)
            + "s skipped out of "
            + str(self.TotalDuration)
            + "s."
        )

        # if ~isempty(obj.V_um_ms)
        #     obj.DoSetVelocity(obj.V_um_ms);
        # end

        # if DumpDataToDiskAfter
        #     obj.DoDumpDataToDisk,
        # end


def DoXcorr(
    a, b
):  # does correlation similar to Matlab xcorr, cuts positive lags, normalizes properly
    if a.size != b.size:
        warn("For unequal lengths of a, b the meaning of lags is not clear!")
    C = np.correlate(a, b, mode="full")
    C = C[np.floor(C.size / 2).astype("int") :]
    lags = np.arange(C.size)

    return C, lags
