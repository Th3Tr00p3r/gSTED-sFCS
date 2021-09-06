""" Image data handeling """

from dataclasses import dataclass

import numpy as np

import utilities.helper as helper


@dataclass
class ImageData:

    image_forward: np.ndarray
    norm_forward: np.ndarray
    image_backward: np.ndarray
    norm_backward: np.ndarray
    line_ticks_v: np.ndarray
    row_ticks_v: np.ndarray


def convert_counts_to_images(counts, ao, scan_params: dict, um_v_ratio) -> ImageData:
    """Doc."""

    def calc_plane_image_stack(counts_stack, eff_idxs, pxls_per_line):
        """Doc."""

        n_lines, _, n_planes = counts_stack.shape
        image_stack = np.empty((n_lines, pxls_per_line, n_planes))
        norm_stack = np.empty((n_lines, pxls_per_line, n_planes))

        for i in range(pxls_per_line):
            image_stack[:, i, :] = counts_stack[:, eff_idxs == i, :].sum(axis=1)
            norm_stack[:, i, :] = counts_stack[:, eff_idxs == i, :].shape[1]

        return image_stack, norm_stack

    n_planes = scan_params["n_planes"]
    n_lines = scan_params["n_lines"]
    pxl_size_um = scan_params["dim2_col_um"] / n_lines
    pxls_per_line = helper.div_ceil(scan_params["dim1_lines_um"], pxl_size_um)
    scan_plane = scan_params["scan_plane"]
    ppl = scan_params["ppl"]
    ppp = n_lines * ppl
    turn_idx = ppl // 2

    if scan_plane in {"XY", "XZ"}:
        dim1_center = scan_params["initial_ao"][0]
        um_per_v = um_v_ratio[0]

    elif scan_plane == "YZ":
        dim1_center = scan_params["initial_ao"][1]
        um_per_v = um_v_ratio[1]

    line_len_v = scan_params["dim1_lines_um"] / um_per_v
    dim1_min = dim1_center - line_len_v / 2

    pxl_size_v = pxl_size_um / um_per_v
    pxls_per_line = helper.div_ceil(scan_params["dim1_lines_um"], pxl_size_um)

    # prepare to remove counts from outside limits
    dim1_ao_single = ao[0][:ppl]
    eff_idxs = ((dim1_ao_single - dim1_min) // pxl_size_v + 1).astype(np.int32)
    eff_idxs_forward = eff_idxs[:turn_idx]
    eff_idxs_backward = eff_idxs[-1 : (turn_idx - 1) : -1]

    # create counts stack shaped (n_lines, ppl, n_planes) - e.g. 80 x 1000 x 1
    j0 = ppp * np.arange(n_planes)[:, np.newaxis]
    J = np.tile(np.arange(ppp), (n_planes, 1)) + j0
    counts_stack = np.diff(np.concatenate((j0, counts[J]), axis=1))
    counts_stack = counts_stack.T.reshape(n_lines, ppl, n_planes)
    counts_stack_forward = counts_stack[:, :turn_idx, :]
    counts_stack_backward = counts_stack[:, -1 : (turn_idx - 1) : -1, :]

    # calculate the images and normalization separately for the forward/backward parts of the scan
    image_stack_forward, norm_stack_forward = calc_plane_image_stack(
        counts_stack_forward, eff_idxs_forward, pxls_per_line
    )
    image_stack_backward, norm_stack_backward = calc_plane_image_stack(
        counts_stack_backward, eff_idxs_backward, pxls_per_line
    )

    line_scale_v = dim1_min + np.arange(pxls_per_line) * pxl_size_v
    row_scale_v = scan_params["set_pnts_lines_odd"]

    return ImageData(
        image_stack_forward,
        norm_stack_forward,
        image_stack_backward,
        norm_stack_backward,
        line_scale_v,
        row_scale_v,
    )
