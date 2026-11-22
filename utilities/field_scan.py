"""
Donut field scan: how the far-field donut changes with scanner position.

The ring around a vortex beam's dark core is described by its azimuthal harmonics
(fit_tools.ring_balance): the first (one side brighter) and the second (two lobes,
astigmatism). Measured at a grid of scanner positions, each harmonic is split into

    c_m(z) = constant + b_m * z + c_m * conj(z)

where z = (x + i y) is the spot's position on the camera relative to the grid centre,
scaled so that |z| = 1 at the grid's farthest point.

- The constant part is the same wherever the beam is scanned: the vortex plate's
  centring (first harmonic) and anything in the beam before the scanners, plus a tilt of
  the camera's own lens (19 Nov: moving that lens sideways changed nothing).
- The part linear in z comes from optics that the scanned beam crosses at a changing
  place or angle - a tilted or decentred scan/tube lens, or the scanners themselves.
  It reaches the sample, and it is what a tube-lens tilt should bring to zero.

All harmonics are reported as 2|c| in percent, as in the camera's 'Ring Balance' read-out.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def grid_offsets(half_x: float, half_y: float, n_per_axis: int) -> List[Tuple[float, float]]:
    """
    Scanner offsets (V) for an n x n grid, in a snake order so that the scanners never
    jump across the field. The centre is measured first and again last: the difference
    between the two is the drift during the scan.
    """

    if n_per_axis < 2:
        raise ValueError("The grid needs at least 2 points per axis.")
    xs = np.linspace(-half_x, half_x, n_per_axis)
    ys = np.linspace(-half_y, half_y, n_per_axis)
    points = []
    for row, y in enumerate(ys):
        row_xs = xs if row % 2 == 0 else xs[::-1]
        points.extend((float(x), float(y)) for x in row_xs)
    return [(0.0, 0.0)] + points + [(0.0, 0.0)]


@dataclass
class FieldPoint:
    """One grid position: the scanner offset, and what the camera saw there."""

    offset_v: Tuple[float, float]
    core_px: Optional[Tuple[float, float]] = None
    harmonics: Dict[int, complex] = field(default_factory=dict)  # frame-averaged
    n_frames: int = 0

    @property
    def ok(self) -> bool:
        return self.core_px is not None and bool(self.harmonics)


@dataclass
class FieldFit:
    """The split of one harmonic into a constant and a part linear in field position."""

    order: int
    constant: complex
    b: complex  # coefficient of z
    c: complex  # coefficient of conj(z)
    residual_rms: float  # of the per-point values about the fit
    n_points: int

    @property
    def constant_pct(self) -> float:
        return 200 * abs(self.constant)

    @property
    def constant_angle_deg(self) -> float:
        """Bright side (order 1) or lobe axis (order 2), image coordinates (y down)."""
        return float(np.degrees(np.angle(self.constant)) / self.order)

    @property
    def field_pct(self) -> float:
        """The field-dependent part at the grid's farthest point, rms over directions."""
        return 200 * float(np.hypot(abs(self.b), abs(self.c)))

    @property
    def field_vector(self) -> np.ndarray:
        """(b, c) - what a knob scan fits linearly to find the knob setting that nulls it."""
        return np.array([self.b, self.c], dtype=complex)


def fit_field_dependence(points: Sequence[FieldPoint], order: int) -> FieldFit:
    """Fit constant + b z + c conj(z) to one harmonic over the measured grid points."""

    good = [p for p in points if p.ok and order in p.harmonics]
    if len(good) < 3:
        raise ValueError("Fewer than 3 grid points with a donut - cannot separate the parts.")
    cores = np.array([p.core_px for p in good], dtype=float)
    # positions relative to the mean of the centre measurements (first and last), or of all
    centre_points = [p for p in good if p.offset_v == (0.0, 0.0)]
    centre = (
        np.mean([p.core_px for p in centre_points], axis=0)
        if centre_points
        else cores.mean(axis=0)
    )
    z = (cores[:, 0] - centre[0]) + 1j * (cores[:, 1] - centre[1])
    scale = np.abs(z).max()
    if scale <= 0:
        raise ValueError("The spot did not move on the camera - are the scanners connected?")
    z = z / scale
    values = np.array([p.harmonics[order] for p in good], dtype=complex)
    design = np.column_stack([np.ones_like(z), z, np.conj(z)])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coef
    return FieldFit(
        order=order,
        constant=complex(coef[0]),
        b=complex(coef[1]),
        c=complex(coef[2]),
        residual_rms=float(np.sqrt(np.mean(np.abs(residual) ** 2))),
        n_points=len(good),
    )


def best_knob_setting(knob_values: Sequence[float], field_vectors: Sequence[np.ndarray]):
    """
    From field scans at several settings of one knob (with everything else unchanged),
    the setting that makes the field-dependent part smallest.

    For small turns the field-dependent coefficients change linearly with the knob,
    v(k) = v0 + k v1, so |v(k)| is smallest at k = -Re(v0 . conj(v1)) / |v1|^2.
    Returns (best setting, predicted field_pct there, field_pct per unit of knob).
    """

    k = np.asarray(knob_values, dtype=float)
    if len(k) < 2 or np.ptp(k) == 0:
        raise ValueError("Need scans at two or more different settings of the knob.")
    vectors = np.array(field_vectors, dtype=complex)  # shape (n, 2)
    design = np.column_stack([np.ones_like(k), k])
    coef, *_ = np.linalg.lstsq(design, vectors, rcond=None)  # rows: v0, v1
    v0, v1 = coef
    norm1 = float(np.sum(np.abs(v1) ** 2))
    if norm1 == 0:
        raise ValueError("This knob does not change the field-dependent part at all.")
    k_best = -float(np.real(np.sum(v0 * np.conj(v1)))) / norm1
    v_best = v0 + k_best * v1
    return k_best, 200 * float(np.sqrt(np.sum(np.abs(v_best) ** 2))), 200 * np.sqrt(norm1)
