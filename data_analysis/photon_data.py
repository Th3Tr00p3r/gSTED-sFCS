#!/usr/bin/env python3
"""
Created on Mon Dec 21 22:54:26 2020

@author: oleg
"""
import logging

import numpy as np


class PhotonData:
    """Doc."""

    def convert_fpga_data_to_photons(self, fpga_data, version=3, verbose=False):
        """Doc."""

        if version >= 2:
            group_len = 7
            maxval = 256 ** 3
        self.version = version

        section_edges = list()
        edge_start, edge_stop, data_end = find_section_edge(fpga_data, 0, group_len)
        section_edges.append(np.array([edge_start, edge_stop]))

        while not data_end:
            edge_start, edge_stop, data_end = find_section_edge(fpga_data, edge_stop + 1, group_len)
            section_edges.append(np.array([edge_start, edge_stop]))

        SectionLength = np.array([np.diff(SE)[0] for SE in section_edges])
        if verbose:
            print(f"Found {len(section_edges)} sections of lengths:\n{SectionLength}")

        # patching: look at the largest section only
        SecInd = np.argmax(SectionLength)
        Ind = np.arange(section_edges[SecInd][0], section_edges[SecInd][1], group_len)
        counter = fpga_data[Ind + 1] * 256 ** 2 + fpga_data[Ind + 2] * 256 + fpga_data[Ind + 3]
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

        coarse = fpga_data[Ind + 4].astype(int)
        fine = fpga_data[Ind + 5].astype(int)
        self.coarse = coarse
        if self.version >= 3:
            twobit1 = np.floor(coarse / 64).astype(int)
            self.coarse = self.coarse - twobit1 * 64
            self.coarse2 = self.coarse - np.mod(self.coarse, 4) + twobit1

        self.runtime = counter.astype(int)
        #       self.time_stamps = time_stamps
        self.fname = ""
        self.fine = fine
        #       if self.runtime(end) > flintmax:
        #           error('Runtime overrun intmax!')

        self.section_edges = section_edges
        self.all_section_edges = np.array([])
        self.counter_ends = np.array([counter[0], counter[-1]])


# P.NoOfInversionProblems = NoOfInversionProblems;


def find_section_edge(data, section_start, group_len):
    data_end = False
    # find brackets
    i_248 = np.where(data[section_start:] == 248)[0]
    i_254 = np.where(data[section_start:] == 254)[0]
    ii = np.intersect1d(i_248, i_254 - group_len + 1, assume_unique=True)  # find those in registry
    if not ii.size:
        raise RuntimeError("No data found! Check detector and FPGA.")
    edge_start = section_start + ii[0]
    p248 = data[edge_start::group_len]
    p254 = data[(edge_start + group_len - 1) :: group_len]
    kk248 = np.where(p248 != 248)[0]
    kk254 = np.where(p254 != 254)[0]

    for ii, kk in enumerate(kk248):
        # check that this is not a case of a singular mistake in 248
        # byte: happens very rarely but does happen
        if len(kk254) < ii + 1:
            if kk == len(p248) - 1:  # problem in the last photon in the file
                edge_stop = edge_start + kk * group_len
                logging.warning("if (len(kk254) < ii+1):")
                break
            elif len(kk248) == ii + 1:  # likely a singular problem
                # just test the most significant bytes of the runtime are
                # close
                if (
                    abs(np.diff(data(edge_start + kk * group_len + np.array([-group_len + 1, 1]))))
                    < 3
                ):
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
                if (
                    abs(np.diff(data(edge_start + kk * group_len + np.array([-group_len + 1, 1]))))
                    < 3
                ):
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
                edge_stop = edge_start + kk * group_len
                if data[edge_stop - 1] != 254:
                    edge_stop = edge_stop - group_len
                logging.warning("if np.isin(kk, kk254):")
                break
            elif kk248[ii] == (
                kk254[ii] + 1
            ):  # np.isin(kk, (kk254[ii]+1)): # likely a real section ? why np.isin?
                edge_stop = edge_start + kk * group_len
                if data[edge_stop - 1] != 254:
                    edge_stop = edge_stop - group_len
                logging.warning("elif  np.isin(kk, (kk254[ii]+1)):")
                break
            elif kk < kk254[ii]:  # likely a singular error on 248 byte
                # just test the most significant bytes of the runtime are
                # close
                if (
                    abs(np.diff(data[edge_start + kk * group_len + np.array([-group_len + 1, 1])]))
                    < 3
                ):
                    # single error: move on
                    logging.warning("elif (kk < kk254[ii]):")
                    continue
                else:
                    logging.warning("Check data for strange section edges!")

            else:  # likely a signular mistake on 254 byte
                # just test the most significant bytes of the runtime are
                # close
                if (
                    abs(np.diff(data[edge_start + kk * group_len + np.array([-group_len + 1, 1])]))
                    < 3
                ):
                    # single error: move on
                    logging.warning("else :")
                    continue
                else:
                    logging.warning("Check data for strange section edges!")

    if len(kk248) > 0:
        if ii == kk248.size - 1:  # reached the end of the loop without breaking
            edge_stop = edge_start + (p254.size - 1) * group_len
            data_end = True
    else:
        edge_stop = edge_start + (p254.size - 1) * group_len
        data_end = True

    return edge_start, edge_stop, data_end
