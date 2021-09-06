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
        ).astype(type_)

        time_stamps = np.diff(runtime)

        # find simple "inversions": the data with a missing byte
        # decrease in runtime on data j+1, yet the next runtime data (j+2) is
        # higher than j.
        inv_idxs = np.where((time_stamps[:-1] < 0) & ((time_stamps[:-1] + time_stamps[1:]) > 0))[0]
        if (inv_idxs.size) != 0:
            if verbose:
                print(
                    f"Found {inv_idxs.size} instances of missing byte data, ad hoc fixing...",
                    end=" ",
                )
            temp = (time_stamps[inv_idxs] + time_stamps[inv_idxs + 1]) / 2
            time_stamps[inv_idxs] = np.floor(temp).astype(type_)
            time_stamps[inv_idxs + 1] = np.ceil(temp).astype(type_)
            runtime[inv_idxs + 1] = runtime[inv_idxs + 2] - time_stamps[inv_idxs + 1]

        # repairing drops in runtime (happens when number of laser pulses passes 'maxval')
        neg_time_stamp_idxs = np.where(time_stamps < 0)[0]
        time_stamps[neg_time_stamp_idxs] += maxval
        for i in neg_time_stamp_idxs + 1:
            runtime[i:] += maxval

        # saving coarse and fine times
        coarse = fpga_data[idxs + 4]
        self.fine = fpga_data[idxs + 5]

        # some fix due to an issue in FPGA
        if self.version >= 3:
            self.coarse = np.mod(coarse, 64)
            self.coarse2 = coarse - np.mod(coarse, 4) + (coarse // 64)
        else:
            self.coarse = coarse

        self.runtime = runtime


def find_section_edges(data, group_len):  # NOQA C901
    """
    group_len: bytes per photon
    """

    def first_full_photon_idx(data, group_len) -> int:
        """Doc."""

        for idx in range(data.size):
            if (data[idx] == 248) and (data[idx + (group_len - 1)] == 254):
                return idx
        return None

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
