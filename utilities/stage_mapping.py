"""
Putting a bead where you want it, with the stepper stage.

A bead is wherever it happened to land. To measure a PSF, or to map the field across the
sample, it has to be moved to the centre of the scan - or to a chosen point in it. That
is a two-line problem once the stage is calibrated against the scan:

    d(image position, um) = J . d(stage, steps)

J is a 2x2 Jacobian. Its scale is the steps-per-micron the stage settings already carry,
but J also holds the two things a scalar cannot: which stage axis corresponds to which
image axis, and the sign of each. Those depend on how the stage is mounted, how the scan
is rastered and whether the image is transposed, so they are measured rather than assumed:
two moves, one per stage axis, with the bead's image position read before and after.

The stage is a stepper with backlash, so a position reached from the left is not the
position reached from the right. Every move planned here therefore ends with a fixed
approach vector: the stage is first sent PAST the target by the approach amount in the
opposite direction, then brought back along it. The last motion before a measurement is
then always the same direction and the same length, which is what makes a rough stage
repeatable even though it is not accurate.

Nothing here touches hardware; it returns the moves to make.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

AXIS_NAMES: Tuple[str, str] = ("stage_x", "stage_y")
# above this the two stage axes move the image along nearly the same line, so the
# inversion is unreliable - almost always a sign that one calibration move was too small
MAX_USABLE_CONDITION_NUMBER = 20.0
# how far past the target to go before the final approach, in steps
DEFAULT_BACKLASH_STEPS = 20


class StageMappingError(Exception):
    """Raised when a calibration cannot be built or used."""


@dataclass
class StageMove:
    """One calibration move: what was commanded, and what the image did."""

    steps: Tuple[float, float]  # commanded (stage_x, stage_y), in steps
    displacement_um: Tuple[float, float]  # the bead's image displacement (dx, dy), in um

    def as_dict(self) -> dict:
        return dict(steps=list(self.steps), displacement_um=list(self.displacement_um))


@dataclass
class StageScanCalibration:
    """The measured relation between stage steps and image microns."""

    jacobian: np.ndarray  # (2, 2) um of image motion per step; columns are stage axes
    backlash_steps: int = DEFAULT_BACKLASH_STEPS
    approach_sign: Tuple[int, int] = (1, 1)  # the direction every move finishes in
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    notes: str = ""

    def __post_init__(self):
        self.jacobian = np.asarray(self.jacobian, dtype=float).reshape(2, 2)
        if not np.isfinite(self.jacobian).all():
            raise StageMappingError("Jacobian contains non-finite values.")
        if np.linalg.matrix_rank(self.jacobian) < 2:
            raise StageMappingError(
                "The two stage axes move the image along the same line - "
                "the calibration moves were not independent."
            )

    # ------------------------------------------------------------------ quality

    @property
    def condition_number(self) -> float:
        return float(np.linalg.cond(self.jacobian))

    @property
    def is_well_conditioned(self) -> bool:
        return self.condition_number <= MAX_USABLE_CONDITION_NUMBER

    def steps_per_um(self) -> Dict[str, float]:
        """The scalar the stage settings carry, per axis - a check on the units."""

        return {
            name: float(1 / np.linalg.norm(self.jacobian[:, idx]))
            for idx, name in enumerate(AXIS_NAMES)
        }

    def describe_axes(self) -> str:
        """Plain words for what each stage axis does to the image."""

        lines = []
        for idx, name in enumerate(AXIS_NAMES):
            dx, dy = self.jacobian[:, idx]
            angle = np.degrees(np.arctan2(dy, dx))
            dominant = "image x" if abs(dx) >= abs(dy) else "image y"
            sign = "+" if (dx if abs(dx) >= abs(dy) else dy) > 0 else "-"
            crosstalk = min(abs(dx), abs(dy)) / max(abs(dx), abs(dy), 1e-12)
            lines.append(
                f"+{name} -> {sign}{dominant} "
                f"({np.hypot(dx, dy) * 1000:.0f} nm/step, {angle:+.0f} deg, "
                f"{100 * crosstalk:.0f}% cross-talk)"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ use

    def steps_for(self, displacement_um: Sequence[float]) -> Tuple[int, int]:
        """The stage move, in whole steps, that displaces the bead's image by this much."""

        displacement_um = np.asarray(displacement_um, dtype=float).ravel()
        if displacement_um.size != 2:
            raise StageMappingError("Displacement must be (dx, dy) in microns.")
        steps, *_ = np.linalg.lstsq(self.jacobian, displacement_um, rcond=None)
        return (int(round(steps[0])), int(round(steps[1])))

    def plan_move(self, displacement_um: Sequence[float]) -> Dict[str, Any]:
        """
        The full backlash-free plan: go past the target, then approach it.

        'moves' is the list of (x, y) step moves to make, in order; the last one is always
        the same approach vector, so consecutive positionings are repeatable even though
        the stage is not accurate.
        """

        total = np.array(self.steps_for(displacement_um), dtype=int)
        approach = np.array(self.approach_sign, dtype=int) * int(self.backlash_steps)
        overshoot = total - approach

        moves: List[Tuple[int, int]] = []
        if overshoot.any():
            moves.append((int(overshoot[0]), int(overshoot[1])))
        if approach.any():
            moves.append((int(approach[0]), int(approach[1])))
        if not moves:
            moves = [(0, 0)]

        predicted = self.jacobian @ total
        return dict(
            moves=moves,
            total_steps=(int(total[0]), int(total[1])),
            approach_steps=(int(approach[0]), int(approach[1])),
            requested_um=(float(displacement_um[0]), float(displacement_um[1])),
            predicted_um=(float(predicted[0]), float(predicted[1])),
            # rounding to whole steps is the only error the plan knows about
            rounding_error_um=float(np.linalg.norm(predicted - np.asarray(displacement_um))),
            condition_number=self.condition_number,
            is_well_conditioned=self.is_well_conditioned,
        )

    # ------------------------------------------------------------------ storage

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                dict(
                    jacobian=self.jacobian.tolist(),
                    backlash_steps=self.backlash_steps,
                    approach_sign=list(self.approach_sign),
                    timestamp=self.timestamp,
                    notes=self.notes,
                    axis_names=list(AXIS_NAMES),
                    units="um of image motion per stage step",
                ),
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path) -> "StageScanCalibration":
        raw = json.loads(Path(path).read_text())
        return cls(
            jacobian=np.asarray(raw["jacobian"], dtype=float),
            backlash_steps=int(raw.get("backlash_steps", DEFAULT_BACKLASH_STEPS)),
            approach_sign=tuple(raw.get("approach_sign", (1, 1))),
            timestamp=raw.get("timestamp", ""),
            notes=raw.get("notes", ""),
        )


def calibrate(moves: Sequence[StageMove], **kwargs) -> StageScanCalibration:
    """
    Build the calibration from recorded moves.

    Two moves - one per stage axis - are enough and are the usual case. More are allowed
    and are solved in the least-squares sense, which is worth doing if the stage is
    noisy: the fit then averages the backlash away instead of baking one instance of it
    into the matrix.
    """

    if len(moves) < 2:
        raise StageMappingError("At least two moves are needed (one per stage axis).")

    steps = np.array([m.steps for m in moves], dtype=float)  # (n, 2)
    observed = np.array([m.displacement_um for m in moves], dtype=float)  # (n, 2)
    if not np.isfinite(steps).all() or not np.isfinite(observed).all():
        raise StageMappingError("Recorded moves contain non-finite values.")
    if np.linalg.matrix_rank(steps) < 2:
        raise StageMappingError(
            "The recorded moves are not independent - move one stage axis at a time, "
            "and make each move large enough to see."
        )

    # observed = steps @ J^T, solved for J^T
    jacobian_t, *_ = np.linalg.lstsq(steps, observed, rcond=None)
    return StageScanCalibration(jacobian=jacobian_t.T, **kwargs)


def from_steps_per_um(
    steps_per_um: float, swap_axes=False, invert_x=False, invert_y=False, **kwargs
) -> StageScanCalibration:
    """
    A calibration from the scalar in the stage settings plus the axis conventions.

    Use it to get started before a real calibration - with the right swap and signs it
    positions a bead well enough to find it again, and a measured calibration then
    replaces it. The signs are easy to get wrong and cost nothing to check: move the
    stage, see which way the bead went.
    """

    if steps_per_um <= 0:
        raise StageMappingError("steps_per_um must be positive.")
    um_per_step = 1 / steps_per_um
    jacobian = np.diag(
        [um_per_step * (-1 if invert_x else 1), um_per_step * (-1 if invert_y else 1)]
    )
    if swap_axes:
        jacobian = jacobian[::-1]
    return StageScanCalibration(jacobian=jacobian, **kwargs)
