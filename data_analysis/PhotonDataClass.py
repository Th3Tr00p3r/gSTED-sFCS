#!/usr/bin/env python3
"""
Created on Mon Dec 21 22:54:26 2020

@author: oleg
"""
import logging

import numpy as np


class PhotonDataClass:

    # def __init__(self):

    def DoConvertFPGAdataToPhotons(self, FPGAdata, Version=3, verbose=False):
        if Version >= 2:
            GroupLen = 7
            maxval = 256 ** 3
        self.Version = Version

        sectionEdges = list()
        edgeStart, edgeStop, dataEnd = DoFindSectionEdge(FPGAdata, 0, GroupLen)
        sectionEdges.append(np.array([edgeStart, edgeStop]))

        while not dataEnd:
            edgeStart, edgeStop, dataEnd = DoFindSectionEdge(FPGAdata, edgeStop + 1, GroupLen)
            sectionEdges.append(np.array([edgeStart, edgeStop]))

        SectionLength = np.array([np.diff(SE)[0] for SE in sectionEdges])
        if verbose:
            print(f"Found {len(sectionEdges)} sections of lengths:\n{SectionLength}")

        # patching: look at the largest section only
        SecInd = np.argmax(SectionLength)
        Ind = np.arange(sectionEdges[SecInd][0], sectionEdges[SecInd][1], GroupLen)
        counter = FPGAdata[Ind + 1] * 256 ** 2 + FPGAdata[Ind + 2] * 256 + FPGAdata[Ind + 3]
        time_stamps = np.diff(counter.astype(int))

        # find simple "inversions": the data with a missing byte
        # decrease in counter on data j+1, yet the next counter data (j+2) is
        # higher than j.
        J = np.where(
            np.logical_and((time_stamps[:-1] < 0), (time_stamps[:-1] + time_stamps[1:] > 0))
        )[0]
        if J.size != 0:
            if verbose:
                print(f"Found {J.size} of missing bit data: ad hoc fixing...")
            temp = (time_stamps[J] + time_stamps[J + 1]) / 2
            time_stamps[J] = np.floor(temp).astype(int)
            time_stamps[J + 1] = np.ceil(temp).astype(int)
            counter[J + 1] = counter[J + 2] - time_stamps[J + 1]

        J = np.where(time_stamps < 0)[0]
        time_stamps[J] = time_stamps[J] + maxval
        for i in J + 1:
            counter[i:] = counter[i:] + maxval

        coarse = FPGAdata[Ind + 4].astype(int)
        fine = FPGAdata[Ind + 5].astype(int)
        self.coarse = coarse
        if self.Version >= 3:
            twobit1 = np.floor(coarse / 64).astype(int)
            self.coarse = self.coarse - twobit1 * 64
            self.coarse2 = self.coarse - np.mod(self.coarse, 4) + twobit1

        self.runtime = counter.astype(int)
        #       self.time_stamps = time_stamps
        self.fname = ""
        self.fine = fine
        #       if self.runtime(end) > flintmax:
        #           error('Runtime overrun intmax!')

        self.sectionEdges = sectionEdges
        self.AllSectionEdges = np.array([])
        self.counterEnds = np.array([counter[0], counter[-1]])


# P.NoOfInversionProblems = NoOfInversionProblems;


def DoFindSectionEdge(data, sectionStart, GroupLen):
    dataEnd = False
    # find brackets
    I_248 = np.where(data[sectionStart:] == 248)[0]
    I_254 = np.where(data[sectionStart:] == 254)[0]
    II = np.intersect1d(I_248, I_254 - GroupLen + 1, assume_unique=True)  # find those in registry
    if not II.size:
        raise RuntimeError("No data found! Check detector and FPGA.")
    edgeStart = sectionStart + II[0]
    p248 = data[edgeStart::GroupLen]
    p254 = data[(edgeStart + GroupLen - 1) :: GroupLen]
    kk248 = np.where(p248 != 248)[0]
    kk254 = np.where(p254 != 254)[0]

    for ii, kk in enumerate(kk248):
        # check that this is not a case of a singular mistake in 248
        # byte: happens very rarely but does happen
        if len(kk254) < ii + 1:
            if kk == len(p248) - 1:  # problem in the last photon in the file
                edgeStop = edgeStart + kk * GroupLen
                logging.warning("if (len(kk254) < ii+1):")
                break
            elif len(kk248) == ii + 1:  # likely a singular problem
                # just test the most significant bytes of the runtime are
                # close
                if abs(np.diff(data(edgeStart + kk * GroupLen + np.array([-GroupLen + 1, 1])))) < 3:
                    # single error: move on
                    logging.warning("(len(kk248) == ii+1):")
                    continue
                else:
                    logging.warning("Check data for strange section edges!")

            elif (
                np.diff(kk248[ii : (ii + 2)]) > 1
            ):  # likely a singular problem since further data are in registry
                # just test the most significant bytes of the runtime are
                # close
                if abs(np.diff(data(edgeStart + kk * GroupLen + np.array([-GroupLen + 1, 1])))) < 3:
                    # single error: move on
                    logging.warning("elif np.diff(kk248[ii:(ii+2)]) > 1: ")
                    continue
                else:
                    logging.warning("Check data for strange section edges!")

            else:
                logging.warning(
                    "Bizarre problem in data: 248 byte out of registry while 254 is in registry!"
                )

        else:  # (length(kk254) >= ii + 1)
            # if (kk248(ii) == kk254(ii)), # likely a real section
            if np.isin(kk, kk254):
                edgeStop = edgeStart + kk * GroupLen
                if data[edgeStop - 1] != 254:
                    edgeStop = edgeStop - GroupLen
                logging.warning("if np.isin(kk, kk254):")
                break
            elif kk248[ii] == (
                kk254[ii] + 1
            ):  # np.isin(kk, (kk254[ii]+1)): # likely a real section ? why np.isin?
                edgeStop = edgeStart + kk * GroupLen
                if data[edgeStop - 1] != 254:
                    edgeStop = edgeStop - GroupLen
                logging.warning("elif  np.isin(kk, (kk254[ii]+1)):")
                break
            elif kk < kk254[ii]:  # likely a singular error on 248 byte
                # just test the most significant bytes of the runtime are
                # close
                if abs(np.diff(data[edgeStart + kk * GroupLen + np.array([-GroupLen + 1, 1])])) < 3:
                    # single error: move on
                    logging.warning("elif (kk < kk254[ii]):")
                    continue
                else:
                    logging.warning("Check data for strange section edges!")

            else:  # likely a signular mistake on 254 byte
                # just test the most significant bytes of the runtime are
                # close
                if abs(np.diff(data[edgeStart + kk * GroupLen + np.array([-GroupLen + 1, 1])])) < 3:
                    # single error: move on
                    logging.warning("else :")
                    continue
                else:
                    logging.warning("Check data for strange section edges!")

    if len(kk248) > 0:
        if ii == kk248.size - 1:  # reached the end of the loop without breaking
            edgeStop = edgeStart + (p254.size - 1) * GroupLen
            dataEnd = True
    else:
        edgeStop = edgeStart + (p254.size - 1) * GroupLen
        dataEnd = True

    return edgeStart, edgeStop, dataEnd
