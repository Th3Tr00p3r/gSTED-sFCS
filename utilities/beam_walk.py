"""
Beam walking as a linear problem.

Walking two beams onto each other with two mirrors is usually done by iteration: move
mirror A until the spots coincide on camera 1, discover that camera 2 has moved, move
mirror B, and so on. It converges slowly because each mirror moves both cameras.

To first order, though, every spot position is linear in every mirror angle:

    p = p0 + J k

where p = (cam1_x, cam1_y, cam2_x, cam2_y) in pixels, k = the four knob coordinates
(two mirrors, two axes each) in whatever units the knobs are read in (turns, divisions,
volts), and J is a 4x4 Jacobian in pixels per knob unit. Measuring J costs one move per
knob; after that the correction that puts the beam on a target is one linear solve:

    dk = J^-1 (p_target - p_now)

so the alignment is a calibration pass plus a single move per knob - and the calibration
can be reused as long as nothing in the beam path has been changed.

Two calibration modes are supported:

  'full'      - four moves, one per knob. Measures the cross terms too (what mirror A's
                x knob does to the cameras' y coordinates), so it is exact for any mirror
                orientation.
  'decoupled' - two moves, one per mirror, each moving that mirror's x and y knobs by
                recorded amounts. Assumes x knobs move only x and y knobs only y. Twice
                as fast and usually good enough, but it is an assumption: mirrors at 45
                degrees, rotated mounts and image transposition all break it. Calibrate
                once in 'full' mode and read .crosstalk() to see whether it holds for
                your setup.

Nothing here touches hardware: the knob moves are reported for the experimenter to make.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# knob order used throughout: mirror 1 x, mirror 1 y, mirror 2 x, mirror 2 y
KNOB_NAMES: Tuple[str, ...] = ("mirror1_x", "mirror1_y", "mirror2_x", "mirror2_y")
# observable order: the two cameras' spot coordinates, in pixels
OBSERVABLE_NAMES: Tuple[str, ...] = ("cam1_x", "cam1_y", "cam2_x", "cam2_y")

# above this, the two cameras are not telling us independent things (see .condition_number)
MAX_USABLE_CONDITION_NUMBER = 50.0


class BeamWalkError(Exception):
    """Raised when a calibration cannot be built or used."""


@dataclass
class WalkMove:
    """One calibration move: the knob amounts turned, and where the spots ended up."""

    knobs: np.ndarray  # (4,) amounts turned, in knob units (zeros for untouched knobs)
    positions: np.ndarray  # (4,) spot positions after the move, in pixels

    def as_dict(self) -> dict:
        return dict(knobs=list(map(float, self.knobs)), positions=list(map(float, self.positions)))


@dataclass
class BeamWalkCalibration:
    """A measured Jacobian, with the information needed to trust it later."""

    jacobian: np.ndarray  # (4, 4) pixels per knob unit; columns are knobs, rows observables
    mode: str = "full"
    knob_unit: str = "turns"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    notes: str = ""

    # ------------------------------------------------------------------ quality

    @property
    def condition_number(self) -> float:
        """
        How well-posed the inversion is. It is large when the two cameras respond almost
        identically to the mirrors - i.e. when they are too close together, or when one
        of them sees mostly the same combination of beam position and angle as the other.
        A focused spot on the far camera (which then reports beam angle alone) lowers it.
        """

        return float(np.linalg.cond(self.jacobian))

    def crosstalk(self) -> float:
        """
        The fraction of each knob's effect that lands on the 'wrong' axis, as the largest
        over the four knobs. Small (say < 0.1) means 'decoupled' mode is safe to use.
        """

        x_rows, y_rows = (0, 2), (1, 3)
        worst = 0.0
        for knob_idx in range(4):
            column = self.jacobian[:, knob_idx]
            same = x_rows if knob_idx in (0, 2) else y_rows
            other = y_rows if knob_idx in (0, 2) else x_rows
            same_norm = float(np.linalg.norm(column[list(same)]))
            other_norm = float(np.linalg.norm(column[list(other)]))
            if same_norm > 0:
                worst = max(worst, other_norm / same_norm)
        return worst

    def sensitivity(self) -> Dict[str, float]:
        """Pixels of spot motion per knob unit, per knob - a sanity check on the units."""

        return {
            name: float(np.linalg.norm(self.jacobian[:, idx]))
            for idx, name in enumerate(KNOB_NAMES)
        }

    # ------------------------------------------------------------------ use

    def solve(self, current: Sequence[float], target: Sequence[float]) -> dict:
        """
        Return the knob moves that should bring the spots from 'current' to 'target'.

        'current' and 'target' are (cam1_x, cam1_y, cam2_x, cam2_y) in pixels. The result
        holds the four knob amounts, the error being corrected, and the residual error
        predicted by the fit itself (non-zero only if the Jacobian is singular in some
        direction, which is the honest signal that the two cameras cannot separate the
        remaining misalignment).
        """

        current = np.asarray(current, dtype=float).ravel()
        target = np.asarray(target, dtype=float).ravel()
        if current.size != 4 or target.size != 4:
            raise BeamWalkError("Both positions must have 4 components (2 cameras x 2 axes).")
        if not np.isfinite(current).all() or not np.isfinite(target).all():
            raise BeamWalkError("Positions contain non-finite values.")

        error_px = target - current
        knob_moves, *_ = np.linalg.lstsq(self.jacobian, error_px, rcond=None)
        predicted_residual = error_px - self.jacobian @ knob_moves

        return dict(
            knob_moves={name: float(knob_moves[idx]) for idx, name in enumerate(KNOB_NAMES)},
            knob_unit=self.knob_unit,
            error_px={name: float(error_px[idx]) for idx, name in enumerate(OBSERVABLE_NAMES)},
            error_norm_px=float(np.linalg.norm(error_px)),
            predicted_residual_px=float(np.linalg.norm(predicted_residual)),
            condition_number=self.condition_number,
            is_well_conditioned=self.condition_number <= MAX_USABLE_CONDITION_NUMBER,
        )

    def plan_steps(self, current, target) -> dict:
        """
        Split the correction into one move per mirror, and say where the spot must land
        after the first one - so the mirrors can be steered by eye, with no knob amounts.

        Each mirror's two knobs span a 2-D plane of possible spot displacements in the
        4-D space (cam1_x, cam1_y, cam2_x, cam2_y). If the two planes together span that
        space, the misalignment splits uniquely into a part only mirror 1 can make and a
        part only mirror 2 can make. Steering mirror 1 until its camera's spot reaches
        the returned waypoint therefore also puts the other camera where it needs to be,
        leaving exactly the part mirror 2 can null - at which point both cameras land on
        the target together.

        Returns the waypoint for each camera, which camera to watch during the first move
        (the one whose response to mirror 1 is better conditioned), the other camera's
        predicted position after that move - which is the check that the planes were
        measured correctly - and the knob coefficients, in units of the calibration moves.
        """

        current = np.asarray(current, dtype=float).ravel()
        target = np.asarray(target, dtype=float).ravel()
        if current.size != 4 or target.size != 4:
            raise BeamWalkError("Both positions must have 4 components (2 cameras x 2 axes).")

        error_px = target - current
        coefficients, *_ = np.linalg.lstsq(self.jacobian, error_px, rcond=None)
        mirror1_part = self.jacobian[:, :2] @ coefficients[:2]
        mirror2_part = self.jacobian[:, 2:] @ coefficients[2:]
        residual = error_px - (mirror1_part + mirror2_part)

        after_mirror1 = current + mirror1_part

        # steer the first move on whichever camera responds to mirror 1 least degenerately
        conditions = {
            1: float(np.linalg.cond(self.jacobian[:2, :2])),
            2: float(np.linalg.cond(self.jacobian[2:, :2])),
        }
        steer_camera = 1 if conditions[1] <= conditions[2] else 2
        other_camera = 2 if steer_camera == 1 else 1

        return dict(
            steer_camera=steer_camera,
            other_camera=other_camera,
            start={1: tuple(current[:2]), 2: tuple(current[2:])},
            waypoint={1: tuple(after_mirror1[:2]), 2: tuple(after_mirror1[2:])},
            predicted_after_mirror1=dict(
                cam1=tuple(after_mirror1[:2]), cam2=tuple(after_mirror1[2:])
            ),
            final_target=dict(cam1=tuple(target[:2]), cam2=tuple(target[2:])),
            mirror1_knobs={
                KNOB_NAMES[idx]: float(coefficients[idx]) for idx in (0, 1)
            },
            mirror2_knobs={
                KNOB_NAMES[idx]: float(coefficients[idx]) for idx in (2, 3)
            },
            knob_unit=self.knob_unit,
            error_norm_px=float(np.linalg.norm(error_px)),
            predicted_residual_px=float(np.linalg.norm(residual)),
            mirror1_conditions=conditions,
            is_well_conditioned=self.condition_number <= MAX_USABLE_CONDITION_NUMBER,
        )

    # ------------------------------------------------------------------ persistence

    def save(self, path) -> Path:
        """Doc."""

        path = Path(path)
        path.write_text(
            json.dumps(
                dict(
                    jacobian=[list(map(float, row)) for row in self.jacobian],
                    mode=self.mode,
                    knob_unit=self.knob_unit,
                    timestamp=self.timestamp,
                    notes=self.notes,
                    knob_names=list(KNOB_NAMES),
                    observable_names=list(OBSERVABLE_NAMES),
                ),
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path) -> "BeamWalkCalibration":
        """Doc."""

        d = json.loads(Path(path).read_text())
        jacobian = np.asarray(d["jacobian"], dtype=float)
        if jacobian.shape != (4, 4):
            raise BeamWalkError(f"Expected a 4x4 Jacobian, got {jacobian.shape}.")
        return cls(
            jacobian=jacobian,
            mode=d.get("mode", "full"),
            knob_unit=d.get("knob_unit", "turns"),
            timestamp=d.get("timestamp", ""),
            notes=d.get("notes", ""),
        )


def calibrate(
    baseline: Sequence[float],
    moves: List[WalkMove],
    mode: str = "full",
    knob_unit: str = "turns",
    notes: str = "",
) -> BeamWalkCalibration:
    """
    Build a Jacobian from a baseline reading and a list of calibration moves.

    Each move records the knob amounts turned (relative to the baseline, not to the
    previous move) and the resulting spot positions. In 'full' mode at least four moves
    spanning the four knobs are needed; in 'decoupled' mode two moves suffice, each
    moving one mirror in both axes.
    """

    baseline = np.asarray(baseline, dtype=float).ravel()
    if baseline.size != 4:
        raise BeamWalkError("The baseline must have 4 components (2 cameras x 2 axes).")
    if not moves:
        raise BeamWalkError("No calibration moves were recorded.")

    delta_knobs = np.column_stack([m.knobs for m in moves])  # (4, n_moves)
    delta_px = np.column_stack([np.asarray(m.positions, dtype=float) - baseline for m in moves])

    if np.allclose(delta_knobs, 0):
        raise BeamWalkError("All recorded knob amounts are zero.")

    if mode == "full":
        if np.linalg.matrix_rank(delta_knobs) < 4:
            raise BeamWalkError(
                "The calibration moves do not span all four knobs - move each knob at "
                "least once, or use 'decoupled' mode with one move per mirror."
            )
        # least-squares solution of  delta_px = J @ delta_knobs
        jacobian = delta_px @ np.linalg.pinv(delta_knobs)

    elif mode == "decoupled":
        # two independent 2x2 problems: x knobs -> x observables, y knobs -> y observables
        jacobian = np.zeros((4, 4))
        for knob_idxs, obs_idxs in (((0, 2), (0, 2)), ((1, 3), (1, 3))):
            sub_knobs = delta_knobs[list(knob_idxs), :]
            used = ~np.all(np.isclose(sub_knobs, 0), axis=0)  # moves that touched these knobs
            if np.linalg.matrix_rank(sub_knobs[:, used]) < 2:
                raise BeamWalkError(
                    "Each mirror must be moved in both axes for 'decoupled' mode "
                    f"(knobs {[KNOB_NAMES[i] for i in knob_idxs]} are not independent)."
                )
            sub_jacobian = delta_px[np.ix_(list(obs_idxs), np.flatnonzero(used))] @ np.linalg.pinv(
                sub_knobs[:, used]
            )
            jacobian[np.ix_(list(obs_idxs), list(knob_idxs))] = sub_jacobian

    else:
        raise BeamWalkError(f"Unknown calibration mode '{mode}' (use 'full' or 'decoupled').")

    if not np.isfinite(jacobian).all():
        raise BeamWalkError("The calibration produced a non-finite Jacobian.")

    return BeamWalkCalibration(
        jacobian=jacobian, mode=mode, knob_unit=knob_unit, notes=notes
    )
