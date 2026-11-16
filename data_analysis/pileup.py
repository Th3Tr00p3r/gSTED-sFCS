"""
Pile-up (detector dead time) in lifetime histograms: how large it is, and a correction.

After each detected photon the SPAD is blind for its hold-off time (81 ns in the 2025-26
files, against a 100 ns laser period), so in practice only the FIRST photon of a laser
period is recorded. A second photon of the same pulse - always a later one - is lost, which
removes photons from the tail of the decay: the lifetime looks shorter and the count rate
lower. What matters is the number of photons per pulse WHILE THE SAMPLE IS BRIGHT, not the
average count rate - a scan that crosses a bright object piles up in the crossing and
nowhere else.

Three tools, NumPy only:

  pileup_parameter()       how big the effect is, measured from the photon stream itself
  dead_time_correct()      corrects one lifetime histogram taken at a steady rate
  rate_resolved_correct()  the same for bursty data: photons are sorted by the local count
                           rate first, each class is corrected, and the classes are added.
                           A single correction with the average rate under-corrects,
                           because the correction is not linear in the rate.

Rules of thumb (single-exponential decay, mu = detected photons per laser pulse, small mu):
  fraction of photons lost      ~ mu / 2
  apparent lifetime shortened   ~ mu / 4
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def pileup_parameter(
    runtime: np.ndarray, lag_range_pulses: Tuple[int, int] = (30, 100)
) -> Dict[str, float]:
    """
    Measure the photon-weighted mean number of photons per laser pulse, 'mu_eff'.

    'runtime' is the laser-pulse index of every photon (p.raw.pulse_runtime). The number of
    photon pairs that are exactly k pulses apart is sum(mu_i * mu_i+k) over the pulses,
    which for k shorter than any change of the brightness is sum(mu_i^2). Divided by the
    number of photons, sum(mu_i), it is the mean of mu as the photons see it. The pairs are
    counted a few microseconds apart (default 3-10 us) to stay clear of afterpulsing, which
    inflates the shortest lags.

    Returns mu_eff, the first-order losses that follow from it, and the steady count rate
    that would pile up as much. 'laser pulses' are taken to be the unit of 'runtime'.
    """

    runtime = np.asarray(runtime, dtype=np.int64)
    k_min, k_max = lag_range_pulses
    pair_counts = np.zeros(k_max + 1)
    for neighbour in range(1, runtime.size):
        lags = runtime[neighbour:] - runtime[:-neighbour]
        in_range = (lags >= 0) & (lags <= k_max)
        if not in_range.any():
            break
        pair_counts += np.bincount(lags[in_range], minlength=k_max + 1)[: k_max + 1]

    mu_eff = pair_counts[k_min:].mean() / max(runtime.size, 1)
    return dict(
        mu_eff=float(mu_eff),
        fraction_lost=float(mu_eff / 2),
        lifetime_shortening=float(mu_eff / 4),
        pairs_same_pulse=float(pair_counts[0]),
        pairs_per_lag=float(pair_counts[k_min:].mean()),
    )


def dead_time_correct(
    hist: np.ndarray, n_pulses: float, bin_ns: float, dead_time_ns: float
) -> np.ndarray:
    """
    Correct a lifetime histogram for a fixed (non-extending) detector dead time.

    'hist' holds the photons detected in each time bin of the laser period (the bins must
    cover the whole period), accumulated over 'n_pulses' laser pulses at a steady
    brightness. In a given bin the detector was live in all pulses except those in which it
    had fired during the preceding 'dead_time_ns' - a window that may reach back into the
    previous pulse. With L_i live pulses and N_i detections the mean number of photons
    arriving in the bin is -ln(1 - N_i / L_i) per live pulse (Poisson), so the corrected
    histogram is

        n_pulses * -ln(1 - N_i / L_i),      L_i = n_pulses - sum of N over the dead window.

    For a dead time longer than the rest of the period this is the classical Coates
    correction for 'first photon only'.
    """

    hist = np.asarray(hist, dtype=np.float64)
    n_bins = hist.size
    dead_bins = int(min(round(dead_time_ns / bin_ns), n_bins - 1))
    # detections in the 'dead_bins' bins before each bin, cyclically
    doubled = np.concatenate((hist, hist))
    cumulative = np.concatenate(([0.0], np.cumsum(doubled)))
    idx = np.arange(n_bins) + n_bins
    blocked = cumulative[idx] - cumulative[idx - dead_bins]
    live = np.maximum(n_pulses - blocked, 1.0)
    return -n_pulses * np.log1p(-np.clip(hist / live, 0, 1 - 1e-12))


def rate_resolved_correct(
    runtime: np.ndarray,
    time_bin: np.ndarray,
    n_bins: int,
    bin_ns: float,
    dead_time_ns: float,
    window_pulses: int = 2000,
    n_classes: int = 12,
    total_pulses: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Dead-time correction for data whose brightness changes (bursts, scans over objects).

    The measurement is cut into windows of 'window_pulses' laser pulses (0.2 ms by default
    at 10 MHz - shorter than a burst, long enough to hold several photons). Windows are
    sorted into 'n_classes' classes by their photon count, the photons of each class are
    histogrammed over 'time_bin' (the bin index of each photon in the laser period,
    0 .. n_bins - 1), each class is corrected with 'dead_time_correct' and the classes are
    summed.

    Returns the raw and corrected histograms, both in total and per class ('class_raw',
    'class_corrected', rows are classes), and the mean detected photons per pulse of each
    class ('class_mu').
    """

    runtime = np.asarray(runtime, dtype=np.int64)
    time_bin = np.asarray(time_bin, dtype=np.int64)
    valid = (time_bin >= 0) & (time_bin < n_bins)
    runtime, time_bin = runtime[valid], time_bin[valid]

    window = (runtime - runtime[0]) // window_pulses
    if total_pulses is None:
        total_pulses = int(runtime[-1] - runtime[0]) + 1
    n_windows = max(int(np.ceil(total_pulses / window_pulses)), int(window[-1]) + 1)
    per_window = np.bincount(window, minlength=n_windows)

    # classes of equal width in log(photons per window); empty windows form class 0
    top = max(per_window.max(), 2)
    edges = np.unique(np.round(np.geomspace(1, top + 1, n_classes)).astype(np.int64))
    window_class = np.searchsorted(edges, per_window, side="right")  # 0: empty windows
    n_cls = edges.size + 1

    photon_class = window_class[window]
    class_raw = np.bincount(photon_class * n_bins + time_bin, minlength=n_cls * n_bins)
    class_raw = class_raw.reshape(n_cls, n_bins).astype(np.float64)
    class_pulses = np.bincount(window_class, minlength=n_cls) * float(window_pulses)

    class_corrected = np.zeros_like(class_raw)
    for cls in range(n_cls):
        if class_pulses[cls] and class_raw[cls].any():
            class_corrected[cls] = dead_time_correct(
                class_raw[cls], class_pulses[cls], bin_ns, dead_time_ns
            )

    return dict(
        raw=class_raw.sum(axis=0),
        corrected=class_corrected.sum(axis=0),
        class_raw=class_raw,
        class_corrected=class_corrected,
        class_mu=class_raw.sum(axis=1) / np.maximum(class_pulses, 1),
        class_pulses=class_pulses,
    )
