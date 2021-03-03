# -*- coding: utf-8 -*-
""" scan patterns module. """

from typing import SimpleNameSpace

import numpy as np


class ScanPatternAO:
    def __init__(self, pattern: str, scan_params, um_V_ratio):
        self.pattern = pattern
        self.scan_params = scan_params
        self.um_V_ratio = um_V_ratio

    def calculate_ao(self):
        """Doc."""

        if self.pattern == "image":
            return self.calc_image_pattern(self.scan_params, self.um_V_ratio)
        if self.pattern == "angular":
            return self.calc_angular_pattern(self.scan_params, self.um_V_ratio)
        if self.pattern == "circle":
            return self.calc_circle_pattern(self.scan_params, self.um_V_ratio)

    def calc_image_pattern(self, params, um_V_ratio):
        # TODO: this function needs better documentation, starting with some comments
        """Doc."""

        dt = 1 / (params.line_freq_Hz * params.ppl)

        # order according to relevant plane dimensions
        if params.scn_type == "XY":
            dim_conv = tuple(self.um_V_ratio[i] for i in (0, 1, 2))
            curr_ao = tuple(getattr(params, f"curr_ao_{ax.lower()}") for ax in "XYZ")
        if params.scn_type == "YZ":
            dim_conv = tuple(self.um_V_ratio[i] for i in (1, 2, 0))
            curr_ao = tuple(getattr(params, f"curr_ao_{ax.lower()}") for ax in "YZX")
        if params.scn_type == "XZ":
            dim_conv = tuple(self.um_V_ratio[i] for i in (0, 2, 1))
            curr_ao = tuple(getattr(params, f"curr_ao_{ax.lower()}") for ax in "XZY")

        T = params.ppl
        f = params.lin_frac
        t0 = T / 2 * (1 - f) / (2 - f)
        v = 1 / (T / 2 - 2 * t0)
        a = v / t0
        A = 1 / f

        t = np.arange(T)
        s = np.zeros(T)
        J = t <= t0
        s[J] = a * np.power(t[J], 2)

        J = np.logical_and((t > t0), (t <= (T / 2 - t0)))
        s[J] = v * t[J] - a * t0 ** 2 / 2

        J = np.logical_and((t > (T / 2 - t0)), (t <= (T / 2 + t0)))
        s[J] = A - a * np.power((t[J] - T / 2), 2) / 2

        J = np.logical_and((t > (T / 2 + t0)), (t <= (T - t0)))
        s[J] = A + a * t0 ** 2 / 2 - v * (t[J] - T / 2)

        J = t > (T - t0)
        s[J] = a * np.power((T - t[J]), 2) / 2

        s = s - 1 / (2 * f)

        ampl = params.dim1_um / dim_conv[0]
        single_line_ao = curr_ao[0] + ampl * s

        ampl = params.dim2_um / dim_conv[1]
        if params.n_lines > 1:
            vect = np.array(
                [(val / (params.n_lines - 1)) - 0.5 for val in range(params.n_lines)]
            )
        else:
            vect = 0
        set_pnts_lines_odd = curr_ao[1] + ampl * vect

        ampl = params.dim3_um / dim_conv[2]
        if params.n_planes > 1:
            vect = np.array(
                [(val / (params.n_planes - 1)) - 0.5 for val in range(params.n_planes)]
            )
        else:
            vect = 0
        set_pnts_planes = curr_ao[2] + ampl * vect

        set_pnts_lines_even = set_pnts_lines_odd[::-1]

        total_pnts = params.ppl * params.n_lines * params.n_planes

        # at this point we have the AO (1D) for a single row,
        # and the voltages corresponding to each row and plane.
        # now we build the full AO (2D):

        single_line_ao = single_line_ao.tolist()
        set_pnts_lines_odd = set_pnts_lines_odd.tolist()
        set_pnts_lines_even = set_pnts_lines_even.tolist()
        set_pnts_planes = np.asarray(set_pnts_planes).tolist()
        if not isinstance(set_pnts_planes, list):
            set_pnts_planes = [set_pnts_planes]

        dim1_ao = []
        dim2_ao = []
        for odd_line_set_pnt in set_pnts_lines_odd:
            dim1_ao += single_line_ao
            dim2_ao += [odd_line_set_pnt] * len(single_line_ao)

        ao_buffer = [dim1_ao, dim2_ao]

        return (
            ao_buffer,
            dt,
            set_pnts_lines_odd,
            set_pnts_lines_even,
            set_pnts_planes,
            total_pnts,
        )

    def calc_angular_pattern(self, params, um_V_ratio):
        # TODO: this function needs better documentation, starting with some comments
        """Doc."""

        max_y_freq_Hz = 20
        req_samp_freq_Hz = 10000

        angle_rad = params.angle_deg * np.pi / 180
        f = params.lin_frac

        if (params.angle_deg == 0) or (params.angle_deg == 90):
            tot_len = params.max_line_len
            n_lines = 2 * int((params.max_line_len / params.line_shift + 1) / 2)
        elif (params.angle_deg > 0) and (params.angle_deg < 90):
            if np.cos(2 * angle_rad) < np.sqrt(2) * np.cos(
                angle_rad + np.pi / 4
            ) * params.max_line_len / (params.line_shift * (params.min_n_lines - 1)):
                tot_len = min(
                    params.max_line_len,
                    (
                        params.max_line_len
                        - params.line_shift
                        * (params.min_n_lines - 1)
                        * np.sin(angle_rad)
                    )
                    / np.cos(angle_rad),
                )
                n_lines = max(
                    params.min_n_lines,
                    2
                    * int(
                        (
                            params.max_line_len
                            / params.line_shift
                            * (1 - abs(np.cos(angle_rad)))
                            / np.sin(angle_rad)
                            + 1
                        )
                        / 2
                    ),
                )
            else:
                tot_len = min(
                    params.max_line_len,
                    (
                        params.max_line_len
                        - params.line_shift
                        * (params.min_n_lines - 1)
                        * np.cos(angle_rad)
                    )
                    / np.sin(angle_rad),
                )
                n_lines = max(
                    params.min_n_lines,
                    2
                    * int(
                        (
                            params.max_line_len
                            / params.line_shift
                            * (1 - abs(np.sin(angle_rad)))
                            / np.cos(angle_rad)
                            + 1
                        )
                        / 2
                    ),
                )

        elif (params.angle_deg > 90) and (params.angle_deg < 180):
            if np.cos(2 * angle_rad) < np.sqrt(2) * np.cos(
                np.pi - angle_rad + np.pi / 4
            ) * params.max_line_len / (params.line_shift * (params.min_n_lines - 1)):
                tot_len = min(
                    params.max_line_len,
                    (
                        -params.max_line_len
                        + params.line_shift
                        * (params.min_n_lines - 1)
                        * np.sin(angle_rad)
                    )
                    / np.cos(angle_rad),
                )
                n_lines = max(
                    params.min_n_lines,
                    2
                    * int(
                        (
                            params.max_line_len
                            / params.line_shift
                            * (1 - abs(np.cos(angle_rad)))
                            / np.sin(angle_rad)
                            + 1
                        )
                        / 2
                    ),
                )
            else:
                tot_len = min(
                    params.max_line_len,
                    (
                        params.max_line_len
                        + params.line_shift
                        * (params.min_n_lines - 1)
                        * np.cos(angle_rad)
                    )
                    / np.sin(angle_rad),
                )
                n_lines = max(
                    params.min_n_lines,
                    2
                    * int(
                        (
                            params.max_line_len
                            / params.line_shift
                            * (1 - abs(np.sin(angle_rad)))
                            / abs(np.cos(angle_rad))
                            + 1
                        )
                        / 2
                    ),
                )

        else:
            # TODO: error('The scan angle should be in [0, 180] range!')
            pass

        line_len = f * tot_len
        scan_freq_Hz = params.speed_um_s / (2 * tot_len * (2 - f))

        samp_freq_Hz = params.pxl_clk_freq_Hz / round(
            params.pxl_clk_freq_Hz / req_samp_freq_Hz
        )  # should be a multiplier of pxl_clk_freq_Hz
        tot_ppl = round((samp_freq_Hz / scan_freq_Hz) / 2)
        scan_freq_Hz = samp_freq_Hz / tot_ppl / 2

        if scan_freq_Hz > max_y_freq_Hz:
            x_ao = []
            y_ao = []
            # TODO: error(['Maximal frequency exceeded: ', num2str(Freq)])

        T = tot_ppl
        ppl = T * f / (2 - f)  # make ppl in linear part

        t0 = T * (1 - f) / (2 - f)
        Vshift = params.line_shift / t0 / 2
        v = 1 / (T - 2 * t0)
        a = v / t0
        A = 1 / f

        T = round(T)
        t = np.arange(T)
        s = np.zeros(T)
        shift_vec = np.zeros(T)

        J = t <= t0
        s[J] = a * np.power(t[J], 2) / 2
        shift_vec[J] = Vshift * (t[J] - t0)

        J = (t > t0) & (t <= (T - t0))
        s[J] = v * t[J] - a * t0 ^ 2 / 2
        shift_vec[J] = 0

        J = t > (T - t0)  # & (t <= (T/2 + t0))
        s[J] = A - a * np.power((t[J] - T), 2) / 2
        shift_vec[J] = Vshift * (t[J] - T + t0)
        # centering
        s = s - 1 / (2 * f)

        X1 = line_len * s * np.cos(angle_rad) + shift_vec * np.sin(angle_rad)
        Y1 = line_len * s * np.sin(angle_rad) - shift_vec * np.cos(angle_rad)
        X2 = -X1 + 2 * shift_vec * np.sin(angle_rad)
        Y2 = -Y1 - 2 * shift_vec * np.cos(angle_rad)

        for i in range(n_lines):
            if i % 2 == 0:
                XT = X1
                YT = Y1
            else:
                XT = X2
                YT = Y2

            x_ao[:, i] = XT[:] + (i - (n_lines + 1) / 2) * params.line_shift * np.sin(
                angle_rad
            )
            y_ao[:, i] = YT[:] - (i - (n_lines + 1) / 2) * params.line_shift * np.cos(
                angle_rad
            )

        # add one more line to walk back slowly to the beginning
        Xend = x_ao[-1, -1]
        Yend = y_ao[-1, -1]
        x_ao[:, i + 1] = Xend + np.arange(1.0, T + 1) * (x_ao(1, 1) - Xend) / T
        y_ao[:, i + 1] = Yend + np.arange(1.0, T + 1) * (y_ao(1, 1) - Yend) / T

        x_ao = x_ao[:]
        y_ao = y_ao[:]

        # try circular connections
        J = t <= t0
        s[J] = a * np.power(t[J], 2) / 2

        eff_speed_um_s = v * line_len * samp_freq_Hz

        scan_settings = SimpleNameSpace()
        scan_settings.eff_speed_um_s = eff_speed_um_s
        scan_settings.scan_freq_Hz = scan_freq_Hz
        scan_settings.samp_freq_Hz = samp_freq_Hz
        scan_settings.tot_ppl = tot_ppl
        scan_settings.ppl = ppl
        scan_settings.n_lines = n_lines
        scan_settings.line_len = line_len
        scan_settings.tot_len = tot_len
        scan_settings.max_line_len = params.max_line_len
        scan_settings.line_shift = params.line_shift
        scan_settings.angle_deg = params.angle_deg
        scan_settings.lin_frac = params.lin_frac
        scan_settings.pxl_clk_freq_Hz = params.pxl_clk_freq_Hz
        scan_settings.lin_part = np.arange(t0, (T - t0) + 1)
        scan_settings.Xlim = [min(x_ao), max(x_ao)]
        scan_settings.Ylim = [min(y_ao), max(y_ao)]

        return x_ao, y_ao, scan_settings
