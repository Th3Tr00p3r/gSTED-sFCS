"""Raw data handling."""

import numpy as np


class PhotonData:
    """Doc."""

    def convert_fpga_data_to_photons(self, fpga_data, version=3, verbose=False):
        """Doc."""

        type_ = np.int64

        if version >= 2:
            group_len = 7
            maxval = 256 ** 3
        self.version = version

        section_edges = []
        edge_start, edge_stop, data_end = find_section_edge(fpga_data, group_len, verbose)
        section_edges.append((edge_start, edge_stop))

        while not data_end:
            remaining_data = fpga_data[edge_stop + 1 :]
            edge_start, edge_stop, data_end = find_section_edge(remaining_data, group_len, verbose)
            section_edges.append((edge_start, edge_stop))

        section_lengths = np.array(
            [edge_stop - edge_start for (edge_start, edge_stop) in section_edges]
        )
        if verbose:
            if (n_sec_edges := len(section_edges)) > 1:
                print(
                    f"Found {n_sec_edges} sections of lengths: {', '.join(map(str, section_lengths))}. Using largest.",
                    end=" ",
                )
            else:
                print(f"Found a single section of length: {section_lengths[0]}.", end=" ")

        # using the largest section only
        largest_sec_start_idx, largest_sec_end_idx = section_edges[np.argmax(section_lengths)]
        idxs = np.arange(largest_sec_start_idx, largest_sec_end_idx, group_len)
        counter = (
            fpga_data[idxs + 1] * 256 ** 2 + fpga_data[idxs + 2] * 256 + fpga_data[idxs + 3]
        ).astype(type_)
        time_stamps = np.diff(counter)

        # find simple "inversions": the data with a missing byte
        # decrease in counter on data j+1, yet the next counter data (j+2) is
        # higher than j.
        inv_idxs = np.where((time_stamps[:-1] < 0) & ((time_stamps[:-1] + time_stamps[1:]) > 0))[0]
        if (n_invs := inv_idxs.size) != 0:
            if verbose:
                print(f"Found {n_invs} of missing bit data, ad hoc fixing...", end=" ")
            temp = (time_stamps[inv_idxs] + time_stamps[inv_idxs + 1]) / 2
            time_stamps[inv_idxs] = np.floor(temp).astype(type_)
            time_stamps[inv_idxs + 1] = np.ceil(temp).astype(type_)
            counter[inv_idxs + 1] = counter[inv_idxs + 2] - time_stamps[inv_idxs + 1]

        # repairing drops in counter (idomic note)
        neg_time_stamp_idxs = np.where(time_stamps < 0)[0]
        time_stamps[neg_time_stamp_idxs] += maxval
        for i in neg_time_stamp_idxs + 1:
            counter[i:] += maxval

        # saving coarse and fine times
        self.coarse = fpga_data[idxs + 4].astype(type_)
        self.fine = fpga_data[idxs + 5].astype(type_)

        # some fix due to an issue in FPGA
        if self.version >= 3:
            twobit1 = np.floor(self.coarse / 64).astype(type_)
            self.coarse = self.coarse - twobit1 * 64
            self.coarse2 = self.coarse - np.mod(self.coarse, 4) + twobit1

        self.runtime = counter


def find_section_edge(data, group_len, verbose=False):  # noqa c901
    """
    group_len: bytes per photon
    """

    data_end = False
    # find brackets (photon starts and ends)
    idx_248 = np.where(data == 248)[0]
    idx_254 = np.where(data == 254)[0]

    try:
        # find index of first complete photon (where 248 and 254 bytes are spaced exatly (group_len -1) bytes apart)
        photon_start_idxs = np.intersect1d(idx_248, idx_254 - (group_len - 1), assume_unique=True)
        edge_start = photon_start_idxs[0]
    except IndexError:
        raise RuntimeError("No data found! Check detector and FPGA.")

    # slice data where assumed to be 248 and 254 (photon brackets)
    data_presumed_248 = data[edge_start::group_len]
    data_presumed_254 = data[(edge_start + group_len - 1) :: group_len]

    # find indices where this assumption breaks
    missed_248_idxs = np.where(data_presumed_248 != 248)[0]
    tot_missed_248s = len(missed_248_idxs)
    missed_254_idxs = np.where(data_presumed_254 != 254)[0]
    tot_missed_254s = len(missed_254_idxs)

    for count, missed_248_idx in enumerate(missed_248_idxs):

        # hold data idx of current missing photon starting bracket
        data_idx_of_missed_248 = edge_start + missed_248_idx * group_len

        single_error_msg = f"Found single photon error (data[{data_idx_of_missed_248}]), ignoring and continuing..."

        # condition for ignoring single photon error (just test the most significant bytes of the runtime are close)
        ignore_single_error_cond = (
            abs(data[data_idx_of_missed_248 + 1] - data[data_idx_of_missed_248 - (group_len - 1)])
            < 3
        )

        # check that this is not a case of a singular mistake in 248 byte: happens very rarely but does happen
        if tot_missed_254s < count + 1:
            # no missing ending brackets, at least one missing starting bracket
            if missed_248_idx == (len(data_presumed_248) - 1):
                # problem in the last photon in the file
                if verbose:
                    print(
                        "Found single error in last photon of file, ignoring and finishing...",
                        end=" ",
                    )
                edge_stop = data_idx_of_missed_248
                break

            elif (tot_missed_248s == count + 1) or (
                np.diff(missed_248_idxs[count : (count + 2)]) > 1
            ):
                if ignore_single_error_cond:
                    if verbose:
                        print(single_error_msg, end=" ")
                    continue
                else:
                    raise RuntimeError("Check data for strange section edges!")

            else:
                raise RuntimeError(
                    "Bizarre problem in data: 248 byte out of registry while 254 is in registry!"
                )

        else:  # (tot_missed_254s >= count + 1)
            # if (missed_248_idxs[count] == missed_254_idxs[count]), # likely a real section
            if np.isin(missed_248_idx, missed_254_idxs):
                if verbose:
                    print("Found a section, continuing...", end=" ")
                edge_stop = data_idx_of_missed_248
                if data[edge_stop - 1] != 254:
                    edge_stop = edge_stop - group_len
                break

            elif missed_248_idxs[count] == (
                missed_254_idxs[count] + 1
            ):  # np.isin(missed_248_idx, (missed_254_idxs[count]+1)): # likely a real section ? why np.isin?
                if verbose:
                    print("Found a section, continuing...", end=" ")
                edge_stop = data_idx_of_missed_248
                if data[edge_stop - 1] != 254:
                    edge_stop = edge_stop - group_len
                break

            elif missed_248_idx < missed_254_idxs[count]:  # likely a singular error on 248 byte
                if ignore_single_error_cond:
                    if verbose:
                        print(single_error_msg, end=" ")
                    continue
                else:
                    raise RuntimeError("Check data for strange section edges!")

            else:  # likely a signular mistake on 254 byte
                if ignore_single_error_cond:
                    if verbose:
                        print(single_error_msg, end=" ")
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

    return edge_start, edge_stop, data_end
