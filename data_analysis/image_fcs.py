"""
Images from scans made with the scanning-FCS (angular) routines - "ImageFCS".

An ImageFCS measurement is an ordinary angular sFCS measurement whose lines are spaced
as closely as the lines of an image scan (and, optionally, repeated 'line_repeats' times
before stepping on). Everything is rebuilt from the photon time-stamps alone, so the same
raw file gives both the image and the per-line time series for the correlations.

Timing - what the hardware does and does not synchronise
---------------------------------------------------------
The AO sample clock is the FPGA's pixel clock, which is gated by the TDC enable, so the
scan pattern starts on the enable to within one AO sample. The photon time-stamps, however,
do NOT count from the enable: the FPGA starts its laser-pulse counter at the first detected
photon ('start_count' in Coarse_time.vhd). The time-stamps therefore lag the scan by however
long the first photon took to arrive, which is nothing when the beam is parked on a bright
spot and milliseconds when the light comes on late or the scan starts on a dark region. On
top of that the scanners follow the command with a constant lag (about 4.5 AO samples at
10 kHz in the 2025 alignment files).

Both delays are one number per file, 'offset_samples', and it is measurable from the data:
the scan runs back and forth, and with the wrong offset the forward and backward passes are
displaced in opposite directions. 'estimate_scan_offset' finds the offset that makes them
coincide.

Only NumPy is needed, so that the acquisition program can import this module cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Optional, Tuple

import numpy as np

FPGA_GROUP_LEN = 7  # bytes per photon: 248, 3 x pulse count, coarse, fine, 254
FPGA_MAX_COUNT = 256**3  # the FPGA's laser-pulse counter is 24 bits wide


# ----------------------------------------------------------------------------------------
# raw bytes -> photon times
# ----------------------------------------------------------------------------------------
def photon_pulse_counts(byte_data) -> np.ndarray:
    """
    Return the raw 24-bit laser-pulse count of every complete photon in 'byte_data'.

    A photon is a group of 7 bytes that begins with 248 and ends with 254. Groups that do
    not sit on the 7-byte grid of a neighbour (a 248 that is really part of a time-stamp)
    are ignored.
    """

    bd = np.asarray(byte_data, dtype=np.uint8)
    if bd.size < 2 * FPGA_GROUP_LEN:
        return np.empty((0,), dtype=np.int64)

    starts = np.flatnonzero(bd[: bd.size - (FPGA_GROUP_LEN - 1)] == 248)
    starts = starts[bd[starts + FPGA_GROUP_LEN - 1] == 254]
    if starts.size < 2:
        return np.empty((0,), dtype=np.int64)

    # keep groups that have a neighbour exactly one group away, and do not overlap the
    # group before them
    gaps = np.diff(starts)
    has_next = np.append(gaps == FPGA_GROUP_LEN, False)
    has_prev = np.insert(gaps == FPGA_GROUP_LEN, 0, False)
    overlaps_prev = np.insert(gaps < FPGA_GROUP_LEN, 0, False)
    starts = starts[(has_next | has_prev) & ~overlaps_prev]

    return (
        bd[starts + 1].astype(np.int64) * 256**2
        + bd[starts + 2].astype(np.int64) * 256
        + bd[starts + 3].astype(np.int64)
    )


def unwrap_pulse_counts(pulse_counts: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Turn the raw 24-bit pulse counts into a monotonic pulse runtime.

    The counter wraps every 2**24 laser pulses (1.68 s at 10 MHz). Only a drop of more than
    half the counter range is taken for a wrap; any other backward step is a corrupted
    photon and is dropped. (Adding 2**24 at *every* backward step, as a naive repair does,
    shifts the rest of the file by 1.68 s for each corrupted photon - which is not a whole
    number of scan periods, so the image jumps.)

    Returns the runtime of the photons that were kept, and the number dropped.
    """

    counts = np.asarray(pulse_counts, dtype=np.int64)
    if counts.size < 2:
        return counts.copy(), 0

    half = FPGA_MAX_COUNT // 2
    steps = np.diff(counts)
    # a photon that leaps far ahead and is immediately followed by a step back is itself
    # the corrupted one - drop it before looking for wraps
    lone_leap = np.flatnonzero((steps[:-1] > half) & (steps[1:] < 0) & (steps[1:] > -half)) + 1
    keep = np.ones(counts.size, dtype=bool)
    keep[lone_leap] = False
    counts = counts[keep]

    steps = np.diff(counts)
    runtime = counts + np.insert(np.cumsum(steps < -half), 0, 0) * FPGA_MAX_COUNT
    # whatever still runs backwards is out of order
    in_order = runtime >= np.maximum.accumulate(runtime)
    return runtime[in_order], int(pulse_counts.size - in_order.sum())


def photon_runtime(byte_data) -> np.ndarray:
    """Raw FPGA bytes -> laser-pulse runtime of each photon (counted from the first)."""

    runtime, _ = unwrap_pulse_counts(photon_pulse_counts(byte_data))
    return runtime


# ----------------------------------------------------------------------------------------
# scan geometry
# ----------------------------------------------------------------------------------------
@dataclass
class ScanGeometry:
    """
    What is needed to place a photon of an angular scan in the sample.

    A scan frame is 'n_passes' sweeps of 'samples_per_line' AO samples each, followed by one
    more sweep that walks back to the start. Sweeps alternate direction. Each line is swept
    'line_repeats' times: the first line_repeats - 1 sweeps stay on the line, half a line
    spacing before it, and the last one sits on it and steps on through its turnaround.
    """

    samples_per_line: int
    n_passes: int
    line_repeats: int
    linear_frac: float
    linear_len_um: float
    line_shift_um: float
    angle_deg: float
    ao_sampling_freq_hz: float
    laser_freq_hz: float
    slow_axis: str = "xy"  # "z": the lines are stepped along z (an X-Z section)

    @classmethod
    def from_scan_settings(cls, scan_settings: dict, laser_freq_hz: float) -> ScanGeometry:
        """
        Build from the 'scan_settings' of a saved file, or from the live 'scan_params' of a
        measurement (the two use slightly different key names).
        """

        ss = scan_settings

        def first(*keys):
            for key in keys:
                if ss.get(key) is not None:
                    return ss[key]
            raise KeyError(f"None of {keys} found in the scan settings")

        line_repeats = int(ss.get("line_repeats") or 1)
        # saved files written before 'line_repeats' existed have no 'n_passes', and their
        # 'n_lines' is the number of sweeps
        n_passes = int(ss.get("n_passes") or ss["n_lines"])
        return cls(
            samples_per_line=int(ss["samples_per_line"]),
            n_passes=n_passes,
            line_repeats=line_repeats,
            linear_frac=float(first("linear_frac", "linear_fraction")),
            linear_len_um=float(ss["linear_len_um"]),
            line_shift_um=float(ss["line_shift_um"]),
            angle_deg=float(first("angle_degrees", "angle_deg")),
            ao_sampling_freq_hz=float(ss["ao_sampling_freq_hz"]),
            laser_freq_hz=float(laser_freq_hz),
            slow_axis=str(ss.get("slow_axis") or "xy").lower(),
        )

    @property
    def n_lines(self) -> int:
        return self.n_passes // self.line_repeats

    @property
    def turn_samples(self) -> float:
        """Length of one turnaround (acceleration) segment, in AO samples."""
        f = self.linear_frac
        return self.samples_per_line * (1 - f) / (2 - f)

    @property
    def um_per_sample(self) -> float:
        """Distance travelled per AO sample in the linear part of a sweep."""
        return self.linear_len_um / (self.samples_per_line - 2 * self.turn_samples)

    @property
    def frame_samples(self) -> int:
        return (self.n_passes + 1) * self.samples_per_line

    @property
    def frame_time_s(self) -> float:
        return self.frame_samples / self.ao_sampling_freq_hz

    def pass_offsets_um(self) -> np.ndarray:
        """Perpendicular position of the linear part of every sweep (as in the pattern)."""

        n_lines, repeats = self.n_lines, self.line_repeats
        line_offsets = (np.arange(n_lines) - (n_lines + 1) / 2) * self.line_shift_um
        offsets = np.repeat(line_offsets, repeats).astype(float)
        is_repeat = np.tile(np.arange(repeats) < repeats - 1, n_lines)
        offsets[is_repeat] -= self.line_shift_um / 2
        return offsets

    def line_offsets_um(self) -> np.ndarray:
        """Mean perpendicular position of each line (averaged over its sweeps)."""

        return self.pass_offsets_um().reshape(self.n_lines, self.line_repeats).mean(axis=1)

    def to_xy_um(self, along_um, across_um) -> Tuple[np.ndarray, np.ndarray]:
        """(along the line, across the lines) -> (x, y), in um from the scan origin."""

        ang = self.angle_deg * pi / 180
        x_um = np.asarray(along_um) * cos(ang) + np.asarray(across_um) * sin(ang)
        y_um = np.asarray(along_um) * sin(ang) - np.asarray(across_um) * cos(ang)
        return x_um, y_um

    def locate(self, runtime: np.ndarray, offset_samples: float = 0.0):
        """
        Place photons in the scan: returns (sweep index, q), where q is the position along
        the sweep in AO samples, measured in the forward direction for every sweep (so a
        backward sweep is already mirrored). The sweep index 'n_passes' is the walk back.
        """

        T = self.samples_per_line
        phase = runtime * (self.ao_sampling_freq_hz / self.laser_freq_hz) + offset_samples
        phase = np.mod(phase, self.frame_samples)
        sweep = np.minimum((phase // T).astype(np.int64), self.n_passes)
        along = phase - sweep * T
        q = np.where(sweep % 2 == 1, T - along, along)
        return sweep, q


# ----------------------------------------------------------------------------------------
# the scan offset
# ----------------------------------------------------------------------------------------
def sweep_profiles(
    runtime: np.ndarray, geom: ScanGeometry, offset_samples: float = 0.0, bins_per_sample: int = 2
) -> np.ndarray:
    """Photon counts per (sweep, position along the sweep), summed over all frames."""

    n_bins = geom.samples_per_line * bins_per_sample
    sweep, q = geom.locate(runtime, offset_samples)
    q_bin = np.clip((q * bins_per_sample).astype(np.int64), 0, n_bins - 1)
    counts = np.bincount(sweep * n_bins + q_bin, minlength=(geom.n_passes + 1) * n_bins)
    return counts.reshape(geom.n_passes + 1, n_bins)[: geom.n_passes]


def _sweep_pairs(runtime, geom, offset_samples, bins_per_sample):
    """Forward sweeps and the backward sweeps that follow them, as float profiles."""

    profiles = sweep_profiles(runtime, geom, offset_samples, bins_per_sample).astype(np.float64)
    fwd, bwd = profiles[0::2], profiles[1::2]
    n_pairs = min(len(fwd), len(bwd))
    return fwd[:n_pairs], bwd[:n_pairs]


def _refine_peak(curve: np.ndarray, best: int) -> float:
    """Sub-step position of the minimum of 'curve' near index 'best' (parabolic)."""

    if 0 < best < curve.size - 1:
        left, mid, right = curve[best - 1 : best + 2]
        curvature = left - 2 * mid + right
        if curvature > 0:
            return best + 0.5 * (left - right) / curvature
    return float(best)


def estimate_scan_offset(
    runtime: np.ndarray,
    geom: ScanGeometry,
    max_offset_samples: Optional[float] = None,
    bins_per_sample: int = 2,
) -> Tuple[float, float]:
    """
    Find the offset (in AO samples) to add to the photon times so that forward and backward
    sweeps coincide.

    Each forward sweep is compared with the backward sweep that follows it (the same line
    when the line is repeated, the neighbouring line otherwise). With an offset error 'd'
    the two are displaced by 2d against each other. Two steps, because each has a blind
    spot the other covers:

    1. Coarse, over the whole range: the peak of the cross-correlation of the zero-padded
       sweeps. Right for an image of sparse features however late the photons started, but
       pulled towards zero by broad structure (a uniform solution with some vignetting).
    2. Fine, within a quarter sweep of the coarse answer: the squared difference between
       the two sweeps over a fixed central window, each displaced by half the trial value.
       Unbiased for broad structure; the fixed window is what stops a trial displacement
       from winning merely by pushing a bright feature out of the comparison.

    Returns (offset_samples, quality). 'quality' is the depth of the fine minimum in units
    of the point-to-point noise of that curve; below ~5 the sample has too little structure
    (or too few photons) and the offset should not be trusted.
    """

    bps = bins_per_sample
    fwd, bwd = _sweep_pairs(runtime, geom, 0.0, bps)
    n_bins = fwd.shape[1]
    if not fwd.sum() or not bwd.sum():
        return 0.0, 0.0

    # --- 1. coarse
    if max_offset_samples is None:
        max_offset_samples = geom.samples_per_line / 4
    max_lag = int(min(2 * max_offset_samples * bps, n_bins // 2))
    level = (fwd.sum(axis=1, keepdims=True) + bwd.sum(axis=1, keepdims=True)) / (2 * n_bins)
    size = 2 * n_bins
    cross = np.fft.irfft(
        (
            np.conj(np.fft.rfft(fwd - level, size, axis=1))
            * np.fft.rfft(bwd - level, size, axis=1)
        ).sum(axis=0),
        size,
    )
    lags = np.arange(-max_lag, max_lag + 1)
    curve = -cross[lags % size]
    # the offset moves forward photons to larger q and backward ones to smaller q, so an
    # offset that is too small by d leaves the backward sweep displaced by +2d
    coarse = (_refine_peak(curve, int(np.argmin(curve))) - max_lag) / (2 * bps)

    # --- 2. fine, about the coarse answer
    fwd, bwd = _sweep_pairs(runtime, geom, coarse, bps)
    half_range = n_bins // 8  # trial half-displacements, in bins: +/- an eighth of a sweep
    window = slice(half_range, n_bins - half_range)
    shifts = np.arange(-half_range, half_range + 1)
    mismatch = np.empty(shifts.size)
    for idx, shift in enumerate(shifts):
        # forward sweep taken 'shift' bins earlier, backward one 'shift' bins later
        diff = (
            fwd[:, window.start - shift : window.stop - shift]
            - bwd[:, window.start + shift : window.stop + shift]
        )
        mismatch[idx] = (diff**2).sum()
    best = int(np.argmin(mismatch))
    fine = (_refine_peak(mismatch, best) - half_range) / bps

    # depth of the minimum against the point-to-point noise of the curve (from its second
    # differences, to which a smooth curve does not contribute)
    noise = 1.4826 * np.median(np.abs(np.diff(mismatch, 2))) / np.sqrt(6)
    quality = float((np.median(mismatch) - mismatch[best]) / noise) if noise > 0 else 0.0
    if best in (0, shifts.size - 1):  # ran into the edge of the search: no minimum found
        quality = 0.0

    return coarse + fine, quality


# ----------------------------------------------------------------------------------------
# the image
# ----------------------------------------------------------------------------------------
@dataclass
class ScanImage:
    """An image rebuilt from an angular scan, and where its pixels are in the sample."""

    counts: np.ndarray  # (n_lines, n_pixels) photons
    pixel_um: float  # pixel size along the line
    geom: ScanGeometry
    offset_samples: float
    offset_quality: float
    duration_s: float

    @property
    def dwell_s(self) -> float:
        """Total time spent on one pixel."""

        n_frames = self.duration_s / self.geom.frame_time_s
        samples = self.pixel_um / self.geom.um_per_sample
        return n_frames * self.geom.line_repeats * samples / self.geom.ao_sampling_freq_hz

    @property
    def rate_khz(self) -> np.ndarray:
        """The image as a count rate."""

        return self.counts / max(self.dwell_s, 1e-12) * 1e-3

    def pixel_to_scan_um(self, col: float, row: float) -> Tuple[float, float]:
        """Image position (col, row), as a cursor reports it -> (along, across) in um."""

        along_um = (col + 0.5) * self.pixel_um - self.geom.linear_len_um / 2
        line_offsets = self.geom.line_offsets_um()
        across_um = float(np.interp(row, np.arange(line_offsets.size), line_offsets))
        return along_um, across_um

    def pixel_to_xy_um(self, col: float, row: float) -> Tuple[float, float]:
        """Image position (col, row) -> (x, y) in um from the origin of the scan."""

        x_um, y_um, _ = self.pixel_to_xyz_um(col, row)
        return x_um, y_um

    def pixel_to_xyz_um(self, col: float, row: float) -> Tuple[float, float, float]:
        """
        Image position (col, row) -> (x, y, z) in um from the origin of the scan. In an X-Z
        section the rows are z and the columns run along the scan angle in the XY plane.
        """

        along_um, across_um = self.pixel_to_scan_um(col, row)
        if self.geom.slow_axis == "z":
            x_um, y_um = self.geom.to_xy_um(along_um, 0.0)
            return float(x_um), float(y_um), float(across_um)
        x_um, y_um = self.geom.to_xy_um(along_um, across_um)
        return float(x_um), float(y_um), 0.0


def scan_image(
    runtime: np.ndarray,
    geom: ScanGeometry,
    offset_samples: Optional[float] = None,
    pixel_um: Optional[float] = None,
    min_quality: float = 5.0,
    fallback_offset_samples: float = 0.0,
) -> ScanImage:
    """
    Build the image of an angular scan from the photon runtime.

    'offset_samples' None finds the offset from the data (falling back to
    'fallback_offset_samples' when the data cannot decide). Rows are lines, columns run
    along the line over its linear part; 'pixel_um' defaults to the line spacing, for
    square pixels.
    """

    quality = float("nan")
    if offset_samples is None:
        offset_samples, quality = estimate_scan_offset(runtime, geom)
        if not quality >= min_quality:
            offset_samples = fallback_offset_samples

    if pixel_um is None:
        pixel_um = geom.line_shift_um
    pixel_um = max(pixel_um, geom.um_per_sample / 8)
    n_pixels = max(int(geom.linear_len_um / pixel_um), 1)

    sweep, q = geom.locate(runtime, offset_samples)
    col = np.floor((q - geom.turn_samples) * geom.um_per_sample / pixel_um).astype(np.int64)
    valid = (sweep < geom.n_passes) & (col >= 0) & (col < n_pixels)
    line = sweep[valid] // geom.line_repeats
    counts = np.bincount(line * n_pixels + col[valid], minlength=geom.n_lines * n_pixels)

    duration_s = float(runtime[-1] - runtime[0]) / geom.laser_freq_hz if runtime.size else 0.0
    return ScanImage(
        counts=counts.reshape(geom.n_lines, n_pixels),
        pixel_um=pixel_um,
        geom=geom,
        offset_samples=float(offset_samples),
        offset_quality=quality,
        duration_s=duration_s,
    )


def scan_image_from_bytes(byte_data, scan_settings: dict, laser_freq_hz: float, **kwargs):
    """Raw FPGA bytes of one file -> ScanImage. 'kwargs' go to 'scan_image'."""

    geom = ScanGeometry.from_scan_settings(scan_settings, laser_freq_hz)
    return scan_image(photon_runtime(byte_data), geom, **kwargs)
