"""
Lifetime-filtered correlation for CW-STED ("STED-FLCS"): shrinking the observation volume
with photon weights instead of a hard time gate.

Under a continuous depletion beam a molecule's fluorescence decays the faster the more
depletion light it sees: slowly (the natural lifetime) at the centre of the doughnut,
quickly on its crest. A time gate uses this by discarding all early photons. A lifetime
filter keeps every photon and gives it a weight f(t) that depends on its arrival time, the
weights being chosen so that - on average - they add up to the number of photons of the
slow, central component and cancel for the fast, peripheral one and for the flat
background (afterpulsing, dark counts). This is the afterpulsing filter the project already
uses, with one more decay pattern.

The decay is modelled as a sum of three patterns over the time channels:

  centre      the unquenched decay, measured with the depletion beam off
  periphery   what remains of the depleted decay after the largest centre contribution that
              the late channels allow, and the flat background, are subtracted
  flat        constant

and the filters are  F = (M^T D^-1 M)^-1 M^T D^-1,  M the patterns (columns, each summing
to 1), D the diagonal matrix of the measured decay - the standard FLCS expression.

Two species are an approximation of a continuum of decay rates across the focus, so the
'centre' filter does not select a sharply bounded region: it weights each position r by
W(r) = sum_t f(t) p(t | r), which is close to 1 at the centre, falls off across the
doughnut, and can dip slightly below zero on the crest. What matters for the analysis is
that W(r) is a property of the instrument and the dye - not of the sample - as long as THE
SAME FILTER (the same array of weights) is applied to the sample and to the calibration.
Recomputing the filter from each measurement's own decay would change W between them.
"""

from __future__ import annotations

from copy import copy
from typing import Optional, Tuple

import numpy as np

from data_analysis.data_processing import AfterpulsingFilter
from utilities.helper import Limits


def _clean_hist(tdc_calib) -> np.ndarray:
    """The normalised lifetime histogram with its NaN channels interpolated over."""

    hist = np.array(tdc_calib.all_hist_norm, dtype=np.float64)
    bad = ~np.isfinite(hist)
    channels = np.arange(hist.size)
    hist[bad] = np.interp(channels[bad], channels[~bad], hist[~bad])
    return hist


def _baseline(hist: np.ndarray, tail_fraction: float = 0.3) -> float:
    return float(hist[-round(hist.size * tail_fraction) :].mean())


def sted_flcs_filter(
    sted_meas,
    confocal_meas,
    tail_range_ns: Tuple[float, float] = (10.0, 20.0),
    centre_scale: float = 1.0,
    start_before_peak_ns: float = 0.2,
) -> AfterpulsingFilter:
    """
    Build the three-pattern filter for a depleted measurement.

    'sted_meas' and 'confocal_meas' are SolutionSFCSMeasurement objects with calibrated TDCs
    (calibrate_tdc() already called): the same sample and detector settings with the
    depletion beam on and off. 'tail_range_ns' (after the peak) is where the depleted decay
    is taken to be pure centre + flat; the centre contribution is the largest that fits
    there, times 'centre_scale' (lower it if the periphery pattern comes out negative).

    Returns an AfterpulsingFilter whose two rows are the CENTRE filter (row 0) and the
    PERIPHERY filter (row 1), so that the existing machinery can use it:

        sted_meas.afterpulsing_filter = flt
        cf_centre = sted_meas.correlate_and_average(cf_name=..., afterpulsing_method="filter")
        cf_periph = sted_meas.correlate_and_average(cf_name=..., afterpulsing_method="filter",
                                                    get_afterpulsing=True)

    The patterns and the decomposition are kept on the object as 'patterns' (channels x 3),
    'fractions' (photon fractions of centre, periphery, flat) and 'filters3' (3 x channels).
    """

    calib, conf_calib = sted_meas.tdc_calib, confocal_meas.tdc_calib
    t_hist = np.asarray(calib.t_hist, dtype=np.float64)
    sted = _clean_hist(calib)
    conf = _clean_hist(conf_calib)

    # the two measurements are calibrated separately: put the confocal decay on the depleted
    # one's time axis by aligning the peaks
    t_peak = t_hist[np.argmax(sted)]
    conf_t = np.asarray(conf_calib.t_hist, dtype=np.float64)
    conf = np.interp(t_hist, conf_t - conf_t[np.argmax(conf)] + t_peak, conf)

    valid_limits = Limits(t_peak - start_before_peak_ns, np.inf)
    valid = valid_limits.valid_indices(t_hist)
    t, sted_v, conf_v = t_hist[valid], sted[valid], conf[valid]

    sted_base, conf_base = _baseline(sted_v), _baseline(conf_v)
    centre = np.clip(conf_v - conf_base, 0, None)
    centre /= centre.sum()
    decay = sted_v - sted_base

    in_tail = Limits(t_peak + tail_range_ns[0], t_peak + tail_range_ns[1]).valid_indices(t)
    if not in_tail.any():
        raise ValueError(f"No channels in the tail range {tail_range_ns} ns after the peak")
    centre_amp = centre_scale * decay[in_tail].sum() / centre[in_tail].sum()
    periphery = np.clip(decay - centre_amp * centre, 0, None)
    # after the tail range begins there is, by construction, no periphery left
    periphery[t >= t_peak + tail_range_ns[0]] = 0
    periphery_amp = periphery.sum()
    if periphery_amp <= 0:
        raise ValueError("Nothing left for the periphery pattern - is this a depleted decay?")
    periphery /= periphery_amp
    flat = np.full(t.size, 1 / t.size)

    patterns = np.vstack((centre, periphery, flat)).T
    weights = 1 / np.clip(sted_v, sted_v[sted_v > 0].min(), None)
    filters = (
        np.linalg.inv(patterns.T @ (patterns * weights[:, None])) @ (patterns * weights[:, None]).T
    )

    # back to the full channel range, zeros where the filter is not defined
    full = np.zeros((3, t_hist.size))
    full[:, valid] = filters

    flt = AfterpulsingFilter(
        t_hist,
        sted,
        sted_base,
        sted_v,
        decay.sum(),
        valid_limits,
        patterns,
        full[:2],
        calib.fine_bins,
    )
    total = centre_amp + periphery_amp + sted_base * t.size
    flt.patterns = patterns
    flt.filters3 = full
    flt.fractions = np.array([centre_amp, periphery_amp, sted_base * t.size]) / total
    return flt


def transfer_filter(flt: AfterpulsingFilter, to_meas, from_peak_ns: Optional[float] = None):
    """
    The same weights for another measurement (e.g. computed on the calibration, applied to
    the sample), shifted in time if the two decays peak in different channels.
    """

    calib = to_meas.tdc_calib
    t_hist = np.asarray(calib.t_hist, dtype=np.float64)
    if from_peak_ns is None:
        from_peak_ns = flt.t_hist[np.argmax(flt.all_hist_norm)]
    shift = t_hist[np.argmax(_clean_hist(calib))] - from_peak_ns
    moved = copy(flt)
    moved.t_hist = t_hist
    moved.fine_bins = calib.fine_bins
    moved.filter = np.vstack(
        [np.interp(t_hist, flt.t_hist + shift, row, left=0, right=0) for row in flt.filter]
    )
    return moved


def effective_weight(flt: AfterpulsingFilter, tau0_ns: float, xi: np.ndarray) -> np.ndarray:
    """
    How strongly the CENTRE filter weights a molecule that sees a depletion intensity
    xi = I/I_sat: sum over channels of f(t) * (its normalised decay), relative to xi = 0.
    A check of what the filter does in space - multiply by the excitation profile and the
    doughnut profile xi(r) to get the effective observation function.
    """

    valid = flt.valid_limits.valid_indices(flt.t_hist)
    t = flt.t_hist[valid] - flt.t_hist[valid][0]
    f_centre = flt.filter[0][valid]
    xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    # photons per excitation, per channel: the decay rate is (1 + xi)/tau0, the radiative
    # part 1/tau0, so the total yield falls as 1/(1 + xi)
    decays = np.exp(-np.outer(1 + xi, t) / tau0_ns)
    decays *= (t[1] - t[0]) / tau0_ns
    weight = decays @ f_centre
    return weight / weight[0] if xi[0] == 0 else weight
