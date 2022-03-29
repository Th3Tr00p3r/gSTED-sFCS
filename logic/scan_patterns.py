""" scan patterns module. """

import logging
from math import cos, pi, sin, sqrt
from types import SimpleNamespace
from typing import Tuple

import numpy as np


class ScanPatternAO:
    """Doc."""

    def __init__(
        self,
        pattern: str,
        um_v_ratio: Tuple[float, float, float],
        curr_ao_v: tuple,
        scan_params,
    ):
        self.pattern = pattern
        self.um_v_ratio = um_v_ratio
        self.curr_ao_v = curr_ao_v
        self.scan_params = scan_params

    def calculate_pattern(self) -> Tuple[np.ndarray, SimpleNamespace]:
        """Doc."""

        if self.pattern == "image":
            return self.calc_image_pattern()
        elif self.pattern == "angular":
            return self.calc_angular_pattern()
        elif self.pattern == "circle":
            return self.calc_circle_pattern()
        else:
            raise ValueError(f"{self.pattern} is not a valid pattern name!")

    def calc_image_pattern(
        self,
    ) -> Tuple[np.ndarray, SimpleNamespace]:
        """Doc."""

        line_freq_hz = self.scan_params.line_freq_hz
        ppl = self.scan_params.ppl
        f = self.scan_params.linear_fraction
        plane_orientation = self.scan_params.plane_orientation
        current_ao = self.curr_ao_v
        dim_um = tuple(getattr(self.scan_params, f"dim{i}_um") for i in (1, 2, 3))
        n_lines = self.scan_params.n_lines
        n_planes = self.scan_params.n_planes

        # order according to relevant plane dimensions
        if plane_orientation == "XY":
            dim_order = (0, 1, 2)
        elif plane_orientation == "YZ":
            dim_order = (1, 2, 0)
        elif plane_orientation == "XZ":
            dim_order = (0, 2, 1)
        dim_conv = tuple(self.um_v_ratio[i] for i in dim_order)
        current_ao = tuple(current_ao[i] for i in dim_order)

        t0 = ppl / 2 * (1 - f) / (2 - f)
        v = 1 / (ppl / 2 - 2 * t0)
        a = v / t0
        A = 1 / f

        s = np.zeros(ppl)

        t = np.arange(ppl)
        j = t <= t0
        s[j] = a * np.power(t[j], 2)

        j = (t > t0) & (t <= (ppl / 2 - t0))
        s[j] = v * t[j] - a * t0 ** 2 / 2

        j = (t > (ppl / 2 - t0)) & (t <= (ppl / 2 + t0))
        s[j] = A - a * np.power((t[j] - ppl / 2), 2) / 2

        j = (t > (ppl / 2 + t0)) & (t <= (ppl - t0))
        s[j] = A + a * t0 ** 2 / 2 - v * (t[j] - ppl / 2)

        j = t > (ppl - t0)
        s[j] = a * np.power((ppl - t[j]), 2) / 2

        s = s - 1 / (2 * f)

        ampl = dim_um[0] / dim_conv[0]
        single_line_ao = current_ao[0] + ampl * s

        ampl = dim_um[1] / dim_conv[1]
        if n_lines > 1:
            vect = np.array([(val / (n_lines - 1)) - 0.5 for val in range(n_lines)])
        else:
            vect = 0
        set_pnts_lines_odd = current_ao[1] + ampl * vect

        ampl = dim_um[2] / dim_conv[2]
        if n_planes > 1:
            vect = np.array([(val / (n_planes - 1)) - 0.5 for val in range(n_planes)])
        else:
            vect = 0
        set_pnts_planes = np.atleast_1d(current_ao[2] + ampl * vect)

        set_pnts_lines_even = set_pnts_lines_odd[::-1]

        # at this point we have the AO (1D) for a single row,
        # and the voltages corresponding to each row and plane.
        # now we build the full AO (2D):
        ao_buffer = calculate_image_ao(set_pnts_lines_odd, single_line_ao)

        self.scan_params.dt = 1 / (line_freq_hz * ppl)
        self.scan_params.dim_order = dim_order
        self.scan_params.set_pnts_lines_odd = set_pnts_lines_odd
        self.scan_params.set_pnts_lines_even = set_pnts_lines_even
        self.scan_params.set_pnts_planes = set_pnts_planes

        return ao_buffer, self.scan_params

    def calc_angular_pattern(
        self,
    ) -> Tuple[np.ndarray, SimpleNamespace]:
        """Doc."""

        # argument definitions (for better readability
        ang_deg = self.scan_params.angle_deg
        ang_rad = ang_deg * (pi / 180)
        f = self.scan_params.linear_fraction
        max_line_len = self.scan_params.max_line_len_um
        line_shift_um = self.scan_params.line_shift_um
        min_n_lines = self.scan_params.min_lines
        ao_sampling_freq_hz = self.scan_params.ao_sampling_freq_hz
        max_scan_freq_Hz = self.scan_params.max_scan_freq_Hz
        speed_um_s = self.scan_params.speed_um_s

        if not (0 <= ang_deg <= 180):
            ang_deg = ang_deg % 180
            logging.warning("The scan angle should be in [0, 180] range!")

        if (ang_deg == 0) or (ang_deg == 90):
            tot_len = max_line_len
            n_lines = 2 * int((max_line_len / line_shift_um + 1) / 2)
        elif 0 < ang_deg < 90:
            if cos(2 * ang_rad) < sqrt(2) * cos(ang_rad + pi / 4) * max_line_len / (
                line_shift_um * (min_n_lines - 1)
            ):
                tot_len = min(
                    max_line_len,
                    (max_line_len - line_shift_um * (min_n_lines - 1) * sin(ang_rad))
                    / cos(ang_rad),
                )
                n_lines = max(
                    min_n_lines,
                    2
                    * int(
                        (max_line_len / line_shift_um * (1 - abs(cos(ang_rad))) / sin(ang_rad) + 1)
                        / 2
                    ),
                )
            else:
                tot_len = min(
                    max_line_len,
                    (max_line_len - line_shift_um * (min_n_lines - 1) * cos(ang_rad))
                    / sin(ang_rad),
                )
                n_lines = max(
                    min_n_lines,
                    2
                    * int(
                        (max_line_len / line_shift_um * (1 - abs(sin(ang_rad))) / cos(ang_rad) + 1)
                        / 2
                    ),
                )

        elif 90 < ang_deg < 180:
            if cos(2 * ang_rad) < sqrt(2) * cos(pi - ang_rad + pi / 4) * max_line_len / (
                line_shift_um * (min_n_lines - 1)
            ):
                tot_len = min(
                    max_line_len,
                    (-max_line_len + line_shift_um * (min_n_lines - 1) * sin(ang_rad))
                    / cos(ang_rad),
                )
                n_lines = max(
                    min_n_lines,
                    2
                    * int(
                        (max_line_len / line_shift_um * (1 - abs(cos(ang_rad))) / sin(ang_rad) + 1)
                        / 2
                    ),
                )
            else:
                tot_len = min(
                    max_line_len,
                    (max_line_len + line_shift_um * (min_n_lines - 1) * cos(ang_rad))
                    / sin(ang_rad),
                )
                n_lines = max(
                    min_n_lines,
                    2
                    * int(
                        (
                            max_line_len
                            / line_shift_um
                            * (1 - abs(sin(ang_rad)))
                            / abs(cos(ang_rad))
                            + 1
                        )
                        / 2
                    ),
                )

        linear_len_um = f * tot_len
        line_freq_hz = speed_um_s / (2 * tot_len * (2 - f))

        samples_per_line = round((ao_sampling_freq_hz / line_freq_hz) / 2)
        line_freq_hz = ao_sampling_freq_hz / samples_per_line / 2

        # NOTE: ask Oleg about this (max y freq?)
        if line_freq_hz > max_scan_freq_Hz:
            logging.warning(
                f"scan frequency ({line_freq_hz:.2f} Hz) is over {max_scan_freq_Hz} Hz."
            )
            raise ValueError

        T = samples_per_line
        ppl = T * f / (2 - f)  # make ppl in linear part

        t0 = T * (1 - f) / (2 - f)
        v_shift = line_shift_um / t0 / 2
        v = 1 / (T - 2 * t0)
        a = v / t0
        A = 1 / f

        t = np.arange(T)
        s = np.zeros(T)
        shift_vec = np.zeros(T)

        j = t <= t0
        s[j] = a * np.power(t[j], 2) / 2
        shift_vec[j] = v_shift * (t[j] - t0)

        j = (t > t0) & (t <= (T - t0))
        s[j] = v * t[j] - a * t0 ** 2 / 2
        shift_vec[j] = 0

        j = t > (T - t0)
        s[j] = A - a * np.power((t[j] - T), 2) / 2
        shift_vec[j] = v_shift * (t[j] - T + t0)
        # centering
        s = s - 1 / (2 * f)

        x_ao = np.zeros(shape=(T, n_lines + 1))
        y_ao = np.zeros(shape=(T, n_lines + 1))
        X1 = linear_len_um * s * cos(ang_rad) + shift_vec * sin(ang_rad)
        Y1 = linear_len_um * s * sin(ang_rad) - shift_vec * cos(ang_rad)
        X2 = -X1 + 2 * shift_vec * sin(ang_rad)
        Y2 = -Y1 - 2 * shift_vec * cos(ang_rad)
        for line_idx in range(n_lines):
            if line_idx % 2 == 0:
                XT = X1
                YT = Y1
            else:
                XT = X2
                YT = Y2

            x_ao[:, line_idx] = XT[:] + (line_idx - (n_lines + 1) / 2) * line_shift_um * sin(
                ang_rad
            )
            y_ao[:, line_idx] = YT[:] - (line_idx - (n_lines + 1) / 2) * line_shift_um * cos(
                ang_rad
            )

        # add one more line to walk back slowly to the beginning
        Xend = x_ao[-1, -2]
        Yend = y_ao[-1, -2]
        x_ao[:, line_idx + 1] = Xend + np.arange(T) * (x_ao[0, 0] - Xend) / T
        y_ao[:, line_idx + 1] = Yend + np.arange(T) * (y_ao[0, 0] - Yend) / T

        # convert to voltages
        x_um_v_ratio, y_um_v_ratio, _ = self.um_v_ratio
        x_ao = x_ao / x_um_v_ratio
        y_ao = y_ao / y_um_v_ratio

        # combine AOX and AOY
        ao_buffer = np.vstack((x_ao.flatten("F"), y_ao.flatten("F")))

        # add current AO
        current_aox_v, current_aoy_v, _ = self.curr_ao_v
        curr_ao = np.array([[current_aox_v], [current_aoy_v]])
        ao_buffer += curr_ao

        self.scan_params.dt = 1 / (
            line_freq_hz * ppl
        )  # TODO: test this (instead of commented line below)
        #        self.scan_params.dt = 1 / ao_sampling_freq_hz
        self.scan_params.n_lines = n_lines
        self.scan_params.samples_per_line = T
        self.scan_params.linear_len_um = linear_len_um
        self.scan_params.ppl = ppl
        self.scan_params.line_freq_hz = line_freq_hz
        self.scan_params.eff_speed_um_s = v * linear_len_um * ao_sampling_freq_hz
        self.scan_params.linear_part = np.arange(t0, (T - t0) + 1, dtype=np.int32)
        self.scan_params.x_lim = [np.min(x_ao), np.max(x_ao)]
        self.scan_params.y_lim = [np.min(y_ao), np.max(y_ao)]

        return ao_buffer, self.scan_params

    def calc_circle_pattern(self) -> Tuple[np.ndarray, SimpleNamespace]:
        """Doc."""

        ao_sampling_freq_hz = self.scan_params.ao_sampling_freq_hz
        diameter_um = self.scan_params.diameter_um
        speed_um_s = self.scan_params.speed_um_s

        circumference = pi * diameter_um
        circle_freq_hz = speed_um_s / circumference
        samples_per_circle = int(ao_sampling_freq_hz / circle_freq_hz)
        n_circles = int(ao_sampling_freq_hz / samples_per_circle)

        x_um_v_ratio, y_um_v_ratio, _ = self.um_v_ratio
        R_Vx = diameter_um / 2 / x_um_v_ratio
        R_Vy = diameter_um / 2 / y_um_v_ratio

        ao_buffer = np.array(
            [
                [
                    R_Vx * sin(2 * pi * (idx / samples_per_circle))
                    for idx in range(samples_per_circle)
                ]
                * n_circles,
                [
                    R_Vy * cos(2 * pi * (idx / samples_per_circle))
                    for idx in range(samples_per_circle)
                ]
                * n_circles,
            ],
        )

        # add current AO
        current_aox_v, current_aoy_v, _ = self.curr_ao_v
        curr_ao = np.array([[current_aox_v], [current_aoy_v]])
        ao_buffer += curr_ao

        # edit params object
        self.scan_params.dt = 1 / (circle_freq_hz * samples_per_circle)
        self.scan_params.eff_speed_um_s = (
            (1 / samples_per_circle) * circumference * ao_sampling_freq_hz
        )
        self.scan_params.n_circles = n_circles
        self.scan_params.circle_freq_hz = circle_freq_hz

        return ao_buffer, self.scan_params


def calculate_image_ao(set_pnts_lines_odd, single_line_ao):
    """Doc."""

    ppl = single_line_ao.size

    dim1_ao = np.empty(shape=(set_pnts_lines_odd.size * ppl,), dtype=np.float64)
    dim2_ao = np.empty(shape=(set_pnts_lines_odd.size * ppl,), dtype=np.float64)
    for idx, odd_line_set_pnt in enumerate(set_pnts_lines_odd):
        dim1_ao[idx * ppl : idx * ppl + ppl] = single_line_ao
        dim2_ao[idx * ppl : idx * ppl + ppl] = [odd_line_set_pnt] * ppl

    return np.vstack((dim1_ao, dim2_ao))
