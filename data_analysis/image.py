""" Image data handeling """

import numpy as np

import utilities.helper as helper


class ImageScanData:
    """Doc."""

    def __init__(self, *args):
        self._counts_to_image_stack(*args)

    def _counts_to_image_stack(self, counts, ao, scan_params: dict, um_v_ratio) -> None:
        """Doc."""

        def calc_plane_image_stack(counts_stack, eff_idxs, pxls_per_line):
            """Doc."""

            n_lines, _, n_planes = counts_stack.shape
            image_stack = np.empty((n_lines, pxls_per_line, n_planes), dtype=np.int32)
            norm_stack = np.empty((n_lines, pxls_per_line, n_planes), dtype=np.int32)

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
        eff_idxs = ((dim1_ao_single - dim1_min) // pxl_size_v + 1).astype(np.int16)
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
        self.image_stack_forward, self.norm_stack_forward = calc_plane_image_stack(
            counts_stack_forward, eff_idxs_forward, pxls_per_line
        )
        self.image_stack_backward, self.norm_stack_backward = calc_plane_image_stack(
            counts_stack_backward, eff_idxs_backward, pxls_per_line
        )

        self.line_ticks_v = dim1_min + np.arange(pxls_per_line) * pxl_size_v
        self.row_ticks_v = scan_params["set_pnts_lines_odd"]
        self.n_planes = n_planes

    def build_image(self, method, plane_idx):
        """Doc."""

        if method == "forward":
            img = self.image_stack_forward[:, :, plane_idx]

        elif method == "forward normalization":
            img = self.norm_stack_forward[:, :, plane_idx]

        elif method == "forward normalized":
            img = (
                self.image_stack_forward[:, :, plane_idx] / self.norm_stack_forward[:, :, plane_idx]
            )

        elif method == "backward":
            img = self.image_stack_backward[:, :, plane_idx]

        elif method == "backward normalization":
            img = self.norm_stack_backward[:, :, plane_idx]

        elif method == "backward normalized":
            img = (
                self.image_stack_backward[:, :, plane_idx]
                / self.norm_stack_backward[:, :, plane_idx]
            )

        elif method == "interlaced":
            p1_norm = (
                self.image_stack_forward[:, :, plane_idx] / self.norm_stack_forward[:, :, plane_idx]
            )
            p2_norm = (
                self.image_stack_backward[:, :, plane_idx]
                / self.norm_stack_backward[:, :, plane_idx]
            )
            n_lines = p1_norm.shape[0] + p2_norm.shape[0]
            img = np.zeros(p1_norm.shape)
            img[:n_lines:2, :] = p1_norm[:n_lines:2, :]
            img[1:n_lines:2, :] = p2_norm[1:n_lines:2, :]

        elif method == "averaged":
            p1 = self.image_stack_forward[:, :, plane_idx]
            p2 = self.image_stack_backward[:, :, plane_idx]
            norm1 = self.norm_stack_forward[:, :, plane_idx]
            norm2 = self.norm_stack_backward[:, :, plane_idx]
            img = (p1 + p2) / (norm1 + norm2)

        return img
