""" scan patterns module. """

import logging
from math import cos, pi, sin
from typing import Any, Dict, Tuple

import numpy as np

from utilities.helper import Limits, unify_length


class ScanPatternAO:
    """Doc."""

    def __init__(
        self,
        pattern: str,
        um_v_ratio: Tuple[float, float, float],
        origin_ao_v: tuple,
        scan_params,
    ):
        self.pattern = pattern
        self.um_v_ratio = um_v_ratio
        self.origin_ao_v = origin_ao_v
        self.scan_params = scan_params

    def calculate_pattern(self) -> Tuple[np.ndarray, Dict[str, Any]]:
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
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Doc."""

        line_freq_hz = self.scan_params["line_freq_hz"]
        ppl = self.scan_params["ppl"]
        f = self.scan_params["linear_fraction"]
        plane_orientation = self.scan_params["plane_orientation"]
        dim_um = tuple(self.scan_params[f"dim{i}_um"] for i in (1, 2, 3))
        n_lines = self.scan_params["n_lines"]
        n_planes = self.scan_params["n_planes"]

        # order according to relevant plane dimensions
        if plane_orientation == "XY":
            dim_order = (0, 1, 2)
        elif plane_orientation == "YZ":
            dim_order = (1, 2, 0)
        elif plane_orientation == "XZ":
            dim_order = (0, 2, 1)
        dim_conv = tuple(self.um_v_ratio[i] for i in dim_order)
        origin_ao_v = tuple(self.origin_ao_v[i] for i in dim_order)

        t0 = ppl / 2 * (1 - f) / (2 - f)
        v = 1 / (ppl / 2 - 2 * t0)
        a = v / t0
        A = 1 / f

        s = np.zeros(ppl)

        t = np.arange(ppl)
        j = t <= t0
        s[j] = a * np.power(t[j], 2)

        j = (t > t0) & (t <= (ppl / 2 - t0))
        s[j] = v * t[j] - a * t0**2 / 2

        j = (t > (ppl / 2 - t0)) & (t <= (ppl / 2 + t0))
        s[j] = A - a * np.power((t[j] - ppl / 2), 2) / 2

        j = (t > (ppl / 2 + t0)) & (t <= (ppl - t0))
        s[j] = A + a * t0**2 / 2 - v * (t[j] - ppl / 2)

        j = t > (ppl - t0)
        s[j] = a * np.power((ppl - t[j]), 2) / 2

        s = s - 1 / (2 * f)

        ampl = dim_um[0] / dim_conv[0]
        single_line_ao = origin_ao_v[0] + ampl * s

        ampl = dim_um[1] / dim_conv[1]
        if n_lines > 1:
            vect = np.array([(val / (n_lines - 1)) - 0.5 for val in range(n_lines)])
        else:
            vect = 0
        set_pnts_lines_odd = origin_ao_v[1] + ampl * vect

        ampl = dim_um[2] / dim_conv[2]
        if n_planes > 1:
            vect = np.array([(val / (n_planes - 1)) - 0.5 for val in range(n_planes)])
        else:
            vect = 0
        set_pnts_planes = np.atleast_1d(origin_ao_v[2] + ampl * vect)

        set_pnts_lines_even = set_pnts_lines_odd[::-1]

        # at this point we have the AO (1D) for a single row,
        # and the voltages corresponding to each row and plane.
        # now we build the full AO (2D):
        ao_buffer = calculate_image_ao(set_pnts_lines_odd, single_line_ao)

        self.scan_params["dt"] = 1 / (line_freq_hz * ppl)
        self.scan_params["dim_order"] = dim_order
        self.scan_params["set_pnts_lines_odd"] = set_pnts_lines_odd
        self.scan_params["set_pnts_lines_even"] = set_pnts_lines_even
        self.scan_params["set_pnts_planes"] = set_pnts_planes

        return ao_buffer, self.scan_params

    def calc_angular_pattern(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Doc."""

        # argument definitions (for better readability
        ang_deg = self.scan_params["angle_deg"]
        f = self.scan_params["linear_fraction"]
        tot_len = self.scan_params["max_line_len_um"]
        scan_width = self.scan_params["scan_width_um"]
        line_shift_um = self.scan_params["line_shift_um"]
        ao_sampling_freq_hz = self.scan_params["ao_sampling_freq_hz"]
        max_scan_freq_Hz = self.scan_params["max_scan_freq_Hz"]
        speed_um_s = self.scan_params["speed_um_s"]

        if not (0 <= ang_deg <= 180):
            ang_deg = ang_deg % 180
            logging.warning("The scan angle should be in [0, 180] range!")
        ang_rad = ang_deg * (pi / 180)

        n_lines = 2 * int((scan_width / line_shift_um + 1) / 2)  # ensure 'n_lines' is even
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
        s[j] = v * t[j] - a * t0**2 / 2
        shift_vec[j] = 0

        j = t > (T - t0)
        s[j] = A - a * np.power((t[j] - T), 2) / 2
        shift_vec[j] = v_shift * (t[j] - T + t0)
        # centering
        s = s - 1 / (2 * f)
        # TODO: need to center width as well?

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
        x_um_v_ratio, y_um_v_ratio, z_um_v_ratio = self.um_v_ratio
        x_ao = x_ao / x_um_v_ratio
        y_ao = y_ao / y_um_v_ratio

        # combine AOX and AOY
        ao_buffer = np.vstack((x_ao.flatten("F"), y_ao.flatten("F")))

        # add origin AO
        origin_aox_v, origin_aoy_v, origin_aoz_v = self.origin_ao_v
        ao_buffer += np.array([[origin_aox_v], [origin_aoy_v]])

        # extend to best fit the AO sampling rate (if needed)
        _, samples_per_scan = ao_buffer.shape
        # multiplied by 10 for floating Z speed # TODO: this can be better implemented by defining Z-speed...
        if samples_per_scan < ao_sampling_freq_hz:
            n_scans = int(ao_sampling_freq_hz / samples_per_scan) * 10
            ao_buffer = np.hstack([ao_buffer] * n_scans)

        # floating z - scan slowly in z-axis (one period during many xy circles)
        if (z_amp_um := self.scan_params.get("floating_z_amplitude_um", 0)) != 0:
            R_Vz = z_amp_um / z_um_v_ratio
            _, aoz_len = ao_buffer.shape
            aoz = [origin_aoz_v + R_Vz * sin(2 * pi * (idx / aoz_len)) for idx in range(aoz_len)]
            ao_buffer = np.vstack((ao_buffer, aoz))

        self.scan_params["dt"] = 1 / (line_freq_hz * ppl)
        self.scan_params["n_lines"] = n_lines
        self.scan_params["samples_per_line"] = T
        self.scan_params["linear_len_um"] = linear_len_um
        self.scan_params["ppl"] = round(ppl)  # TESTESTEST
        self.scan_params["line_freq_hz"] = line_freq_hz
        self.scan_params["eff_speed_um_s"] = v * linear_len_um * ao_sampling_freq_hz
        self.scan_params["linear_part"] = np.arange(t0, (T - t0) + 1, dtype=np.int32)
        self.scan_params["x_lim"] = Limits(np.min(x_ao), np.max(x_ao))
        self.scan_params["y_lim"] = Limits(np.min(y_ao), np.max(y_ao))

        return ao_buffer, self.scan_params

    def calc_circle_pattern(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Doc."""

        ao_sampling_freq_hz = self.scan_params["ao_sampling_freq_hz"]

        # Build AO for a large, fast circle
        diameter_um_large = self.scan_params["diameter_um"]
        speed_um_s_large = self.scan_params["speed_um_s"]

        circumference_large = pi * diameter_um_large
        circle_freq_hz_large = speed_um_s_large / circumference_large
        samples_per_circle_large = int(ao_sampling_freq_hz / circle_freq_hz_large)

        # multiplied by 10 for (optional) floating Z
        n_circles_large = int(ao_sampling_freq_hz / samples_per_circle_large * 10)

        x_um_v_ratio, y_um_v_ratio, z_um_v_ratio = self.um_v_ratio
        R_Vx_large = diameter_um_large / 2 / x_um_v_ratio
        R_Vy_large = diameter_um_large / 2 / y_um_v_ratio

        ao_buffer_large = np.array(
            [
                [
                    R_Vx_large * sin(2 * pi * (idx / samples_per_circle_large))
                    for idx in range(samples_per_circle_large)
                ]
                * n_circles_large,
                [
                    R_Vy_large * cos(2 * pi * (idx / samples_per_circle_large))
                    for idx in range(samples_per_circle_large)
                ]
                * n_circles_large,
            ],
        )

        # add a small, circle to the AO, to alleviate bleaching and spatial correlation in dense solutions
        if self.scan_params.get("should_precess", False):
            diameter_um_small = diameter_um_large / 10
            speed_um_s_small = speed_um_s_large / 1000

            circumference_small = pi * diameter_um_small
            circle_freq_hz_small = speed_um_s_small / circumference_small
            samples_per_circle_small = int(ao_sampling_freq_hz / circle_freq_hz_small)
            # the number of small circles is determined by the already prepared big circles AO length
            # generally, they won't be divisible and one of the circles (big or small) will not finish its last rounds
            n_circles_small = max(int(ao_buffer_large.size / samples_per_circle_small), 1)

            R_Vx_small = diameter_um_small / 2 / x_um_v_ratio
            R_Vy_small = diameter_um_small / 2 / y_um_v_ratio

            ao_buffer_small = np.array(
                [
                    [
                        R_Vx_small * sin(2 * pi * (idx / samples_per_circle_small))
                        for idx in range(samples_per_circle_small)
                    ]
                    * n_circles_small,
                    [
                        R_Vy_small * cos(2 * pi * (idx / samples_per_circle_small))
                        for idx in range(samples_per_circle_small)
                    ]
                    * n_circles_small,
                ],
            )

            # shorten the longer AO array to match the length of the shorter
            if ao_buffer_large.size < ao_buffer_small.size:
                ao_buffer_small = unify_length(ao_buffer_small, ao_buffer_large.shape)
            else:
                ao_buffer_large = unify_length(ao_buffer_large, ao_buffer_small.shape)
        else:
            ao_buffer_small = np.array([[0], [0]])

        # add origin AO vector
        origin_aox_v, origin_aoy_v, origin_aoz_v = self.origin_ao_v
        origin_ao = np.array([[origin_aox_v], [origin_aoy_v]])

        # Combine the large-fast circle, the small-slow circle and the static origin (vector addition)
        ao_buffer = ao_buffer_large + ao_buffer_small + origin_ao

        # floating z - scan slowly in z-axis (one period during many xy circles)
        z_amp_um = self.scan_params.get("floating_z_amplitude_um", 0)
        if z_amp_um != 0:
            R_Vz = z_amp_um / z_um_v_ratio
            aoz_len = ao_buffer.shape[1]
            aoz = [origin_aoz_v + R_Vz * sin(2 * pi * (idx / aoz_len)) for idx in range(aoz_len)]
            ao_buffer = np.vstack((ao_buffer, aoz))

        # edit params object
        self.scan_params["dt"] = 1 / (circle_freq_hz_large * samples_per_circle_large)
        self.scan_params["eff_speed_um_s"] = (
            (1 / samples_per_circle_large) * circumference_large * ao_sampling_freq_hz
        )
        self.scan_params["n_circles"] = n_circles_large
        self.scan_params["circle_freq_hz"] = circle_freq_hz_large

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
