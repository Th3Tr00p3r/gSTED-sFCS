# -*- coding: utf-8 -*-
""" scan patterns module. """

from math import cos, pi, sin, sqrt

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
        if params.scan_plane == "XY":
            dim_conv = tuple(self.um_V_ratio[i] for i in (0, 1, 2))
            curr_ao = tuple(getattr(params, f"curr_ao_{ax.lower()}") for ax in "XYZ")
        if params.scan_plane == "YZ":
            dim_conv = tuple(self.um_V_ratio[i] for i in (1, 2, 0))
            curr_ao = tuple(getattr(params, f"curr_ao_{ax.lower()}") for ax in "YZX")
        if params.scan_plane == "XZ":
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
        )

    def calc_angular_pattern(self, params, um_V_ratio):
        # TODO: this function needs better documentation, starting with some comments
        """Doc."""

        # argument definitions (for better readability
        ang_deg = params.angle_deg
        ang_rad = ang_deg * (pi / 180)
        f = params.lin_frac
        max_line_len = params.max_line_len_um
        line_shift = params.line_shift_um
        min_n_lines = params.min_lines
        samp_freq_Hz = params.ao_samp_freq_Hz
        max_scan_freq_Hz = params.max_scan_freq_Hz

        if (ang_deg == 0) or (ang_deg == 90):
            tot_len = max_line_len
            n_lines = 2 * int((max_line_len / line_shift + 1) / 2)
        elif 0 < ang_deg < 90:
            if cos(2 * ang_rad) < sqrt(2) * cos(ang_rad + pi / 4) * max_line_len / (
                line_shift * (min_n_lines - 1)
            ):
                tot_len = min(
                    max_line_len,
                    (max_line_len - line_shift * (min_n_lines - 1) * sin(ang_rad))
                    / cos(ang_rad),
                )
                n_lines = max(
                    min_n_lines,
                    2
                    * int(
                        (
                            max_line_len
                            / line_shift
                            * (1 - abs(cos(ang_rad)))
                            / sin(ang_rad)
                            + 1
                        )
                        / 2
                    ),
                )
            else:
                tot_len = min(
                    max_line_len,
                    (max_line_len - line_shift * (min_n_lines - 1) * cos(ang_rad))
                    / sin(ang_rad),
                )
                n_lines = max(
                    min_n_lines,
                    2
                    * int(
                        (
                            max_line_len
                            / line_shift
                            * (1 - abs(sin(ang_rad)))
                            / cos(ang_rad)
                            + 1
                        )
                        / 2
                    ),
                )

        elif 90 < ang_deg < 180:
            if cos(2 * ang_rad) < sqrt(2) * cos(
                pi - ang_rad + pi / 4
            ) * max_line_len / (line_shift * (min_n_lines - 1)):
                tot_len = min(
                    max_line_len,
                    (-max_line_len + line_shift * (min_n_lines - 1) * sin(ang_rad))
                    / cos(ang_rad),
                )
                n_lines = max(
                    min_n_lines,
                    2
                    * int(
                        (
                            max_line_len
                            / line_shift
                            * (1 - abs(cos(ang_rad)))
                            / sin(ang_rad)
                            + 1
                        )
                        / 2
                    ),
                )
            else:
                tot_len = min(
                    max_line_len,
                    (max_line_len + line_shift * (min_n_lines - 1) * cos(ang_rad))
                    / sin(ang_rad),
                )
                n_lines = max(
                    min_n_lines,
                    2
                    * int(
                        (
                            max_line_len
                            / line_shift
                            * (1 - abs(sin(ang_rad)))
                            / abs(cos(ang_rad))
                            + 1
                        )
                        / 2
                    ),
                )

        else:
            # TODO: handle this error from measurements.py using needed decorator
            raise ValueError("The scan angle should be in [0, 180] range!")

        lin_len = f * tot_len
        scan_freq_Hz = params.speed_um_s / (2 * tot_len * (2 - f))

        tot_ppl = round((samp_freq_Hz / scan_freq_Hz) / 2)
        scan_freq_Hz = samp_freq_Hz / tot_ppl / 2

        # TODO: ask Oleg about this (max y freq?)
        if scan_freq_Hz > max_scan_freq_Hz:
            print(f"scan frequency is over {max_scan_freq_Hz}. ({scan_freq_Hz} Hz)")
            # raise ValueError("Why is this an error?")

        T = tot_ppl
        ppl = T * f / (2 - f)  # make ppl in linear part

        t0 = T * (1 - f) / (2 - f)
        Vshift = line_shift / t0 / 2
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
        s[J] = v * t[J] - a * t0 ** 2 / 2
        shift_vec[J] = 0

        J = t > (T - t0)
        s[J] = A - a * np.power((t[J] - T), 2) / 2
        shift_vec[J] = Vshift * (t[J] - T + t0)
        # centering
        s = s - 1 / (2 * f)

        x_ao = np.zeros(shape=(T, n_lines + 1))
        y_ao = np.zeros(shape=(T, n_lines + 1))
        X1 = lin_len * s * cos(ang_rad) + shift_vec * sin(ang_rad)
        Y1 = lin_len * s * sin(ang_rad) - shift_vec * cos(ang_rad)
        X2 = -X1 + 2 * shift_vec * sin(ang_rad)
        Y2 = -Y1 - 2 * shift_vec * cos(ang_rad)
        for line_idx in range(n_lines):
            if line_idx % 2 == 0:
                XT = X1
                YT = Y1
            else:
                XT = X2
                YT = Y2

            x_ao[:, line_idx] = XT[:] + (
                line_idx - (n_lines + 1) / 2
            ) * line_shift * sin(ang_rad)
            y_ao[:, line_idx] = YT[:] - (
                line_idx - (n_lines + 1) / 2
            ) * line_shift * cos(ang_rad)

        # add one more line to walk back slowly to the beginning
        Xend = x_ao[-1, -1]
        Yend = y_ao[-1, -1]
        x_ao[:, line_idx + 1] = Xend + np.arange(1.0, T + 1) * (x_ao[0, 0] - Xend) / T
        y_ao[:, line_idx + 1] = Yend + np.arange(1.0, T + 1) * (y_ao[0, 0] - Yend) / T

        # convert to voltages
        x_um_V_ratio, y_um_V_ratio, _ = um_V_ratio
        x_ao = x_ao / x_um_V_ratio
        y_ao = y_ao / y_um_V_ratio

        ao_buffer = [x_ao.flatten("F").tolist(), y_ao.flatten("F").tolist()]

        dt = 1 / samp_freq_Hz

        params.eff_speed_um_s = v * lin_len * samp_freq_Hz
        params.scan_freq_Hz = scan_freq_Hz
        params.tot_ppl = tot_ppl
        params.ppl = ppl
        params.n_lines = n_lines
        params.lin_len = lin_len
        params.tot_len = tot_len
        params.lin_part = np.arange(t0, (T - t0) + 1)
        params.Xlim = [np.min(x_ao), np.max(x_ao)]
        params.Ylim = [np.min(y_ao), np.max(y_ao)]

        return ao_buffer, dt, params

    def calc_circle_pattern(self, params, um_V_ratio):
        """Doc."""

        # argument definitions (for better readability
        samp_freq_Hz = params.ao_samp_freq_Hz
        R_um = params.diameter_um / 2
        speed = params.speed_um_s

        tot_len = 2 * pi * R_um
        scan_freq_Hz = speed / tot_len
        n_smpls = int(samp_freq_Hz / scan_freq_Hz)

        x_um_V_ratio, y_um_V_ratio, _ = um_V_ratio
        R_Vx = R_um / x_um_V_ratio
        R_Vy = R_um / y_um_V_ratio

        ao_buffer = [
            [R_Vx * sin(2 * pi * (i / n_smpls)) for i in range(n_smpls)],
            [R_Vy * cos(2 * pi * (i / n_smpls)) for i in range(n_smpls)],
        ]

        dt = 1 / scan_freq_Hz

        params.scan_freq_Hz = scan_freq_Hz

        return ao_buffer, dt, params
