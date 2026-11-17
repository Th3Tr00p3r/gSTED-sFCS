"""
STED-FLCS: correlation functions of a CW-STED measurement with lifetime filters, next to
the hard-gated ones.

Needs two measurements of the same kind of sample with the same detector settings: one
with the depletion beam off (its decay is the 'centre' pattern) and one with it on. Works
for static and for scanning measurements - for scanning ones compare against cf.vt_um.

What you get, in results:
    conf          CorrFunc of the excitation-only measurement
    sted          CorrFunc of the depleted measurement, all photons (afterpulsing filtered)
    gated[g]      the same with a TDC gate starting g ns after the peak, g in GATES_NS
    flcs_centre   lifetime-filtered: the slow (central) component
    flcs_periph   lifetime-filtered: the fast (peripheral) component
and 'flt', the filter (flt.patterns, flt.filters3, flt.fractions).

For a structure-factor analysis treat 'flcs_centre' exactly like a gated correlation
function: it needs its own calibration measurement, correlated WITH THE SAME FILTER
(sted_flcs.transfer_filter) - see the docstring of data_analysis/sted_flcs.py.
"""

import os
from pathlib import Path

import numpy as np

###############################################
# Move to project root to easily import modules
###############################################
PROJECT_ROOT = Path("/Users/oleg/Documents/GitHub/gSTED-sFCS")
if PROJECT_ROOT.exists():
    os.chdir(PROJECT_ROOT)

from matplotlib import pyplot as plt  # noqa: E402

from data_analysis import sted_flcs  # noqa: E402
from data_analysis.correlation_function import SolutionSFCSMeasurement  # noqa: E402
from utilities.helper import Gate  # noqa: E402

######################
# what to process
######################
DAY = Path("/Users/oleg/Documents/Experiments/tmp/12_11_2026")
CONFOCAL_TEMPLATE = "alignment_static_exc_*.pkl"
STED_TEMPLATE = "alignment_static_sted_*.pkl"
GATES_NS = (1.0, 2.0, 4.0)  # hard gates to compare with, in ns AFTER THE PEAK of the decay
TAIL_RANGE_NS = (10.0, 20.0)  # after the peak: where the depleted decay is centre + flat
READ_OPTIONS = dict(byte_data_slice=slice(21, None))  # add roi_selection etc. for scans


def half_decay(cf, x):
    """Lag (or distance) at which the normalised correlation falls to one half."""

    use = x > x[x > 0][2]
    below = np.flatnonzero(cf.normalized[use] < 0.5)
    return float(x[use][below[0]]) if below.size else np.nan


def process() -> dict:
    conf = SolutionSFCSMeasurement("confocal")
    conf.read_fpga_data(DAY / CONFOCAL_TEMPLATE, **READ_OPTIONS)
    conf.calibrate_tdc()
    sted = SolutionSFCSMeasurement("sted")
    sted.read_fpga_data(DAY / STED_TEMPLATE, **READ_OPTIONS)
    sted.calibrate_tdc()

    results = dict(conf_meas=conf, sted_meas=sted)
    results["conf"] = conf.correlate_and_average(cf_name="confocal")
    results["sted"] = sted.correlate_and_average(cf_name="sted")
    calib = sted.tdc_calib
    t_peak = float(calib.t_hist[np.nanargmax(calib.all_hist_norm)])
    results["gated"] = {
        gate: sted.correlate_and_average(cf_name="sted", tdc_gate_ns=Gate(t_peak + gate, np.inf))
        for gate in GATES_NS
    }

    # the lifetime filter takes the place of the measurement's afterpulsing filter
    standard_filter = sted.afterpulsing_filter
    flt = sted_flcs.sted_flcs_filter(sted, conf, tail_range_ns=TAIL_RANGE_NS)
    sted.afterpulsing_filter = flt
    results["flcs_centre"] = sted.correlate_and_average(cf_name="sted FLCS centre")
    results["flcs_periph"] = sted.correlate_and_average(
        cf_name="sted FLCS periphery", get_afterpulsing=True
    )
    sted.afterpulsing_filter = standard_filter
    results["flt"] = flt
    return results


def report(results: dict) -> None:
    flt = results["flt"]
    print(
        "\nPhoton fractions in the depleted decay: centre {:.1%}, periphery {:.1%}, "
        "flat {:.1%}".format(*flt.fractions)
    )
    rows = [("confocal", results["conf"]), ("sted, all photons", results["sted"])]
    rows += [(f"sted, gate {g} ns", cf) for g, cf in results["gated"].items()]
    rows += [("sted FLCS centre", results["flcs_centre"])]
    rows += [("sted FLCS periphery", results["flcs_periph"])]
    print(f"{'':24s}{'half-decay':>12s}{'G0 (cf.g0)':>14s}{'G0 error':>11s}")
    for label, cf in rows:
        in_norm = cf.norm_range.valid_indices(cf.lag)
        g0_err = np.sqrt(1 / (1 / cf.error_cf_cr[in_norm] ** 2).sum())
        print(f"{label:24s}{half_decay(cf, cf.vt_um):12.4g}{cf.g0:14.4g}{g0_err:11.3g}")


def plot(results: dict):
    flt = results["flt"]
    valid = flt.valid_limits.valid_indices(flt.t_hist)
    t = flt.t_hist[valid]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    amplitudes = flt.fractions * flt.I_j.sum()
    ax.semilogy(t, flt.I_j, "k.", ms=2, label="depleted decay")
    for column, (label, amp) in enumerate(zip(("centre", "periphery", "flat"), amplitudes)):
        ax.semilogy(t, amp * flt.patterns[:, column], label=label)
    ax.set_xlim(t[0], t[0] + 30)
    ax.set_ylim(flt.baseline / 3, None)
    ax.set_xlabel("time (ns)")
    ax.set_title("Decomposition of the depleted decay")
    ax.legend()

    ax = axes[1]
    for row, label in enumerate(("centre", "periphery", "flat")):
        ax.plot(flt.t_hist, flt.filters3[row], label=label)
    ax.set_xlim(t[0] - 1, t[0] + 30)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("photon weight")
    ax.set_title("Filters")
    ax.legend()

    ax = axes[2]
    curves = [("confocal", results["conf"]), ("sted, all photons", results["sted"])]
    curves += [(f"sted, gate {g} ns", cf) for g, cf in results["gated"].items()]
    curves += [("sted FLCS centre", results["flcs_centre"])]
    for label, cf in curves:
        use = cf.vt_um > 0
        ax.semilogx(cf.vt_um[use], cf.normalized[use], ".", ms=3, label=label)
    ax.set_ylim(-0.2, 1.4)
    ax.set_xlim(1e-3, 10)
    ax.set_xlabel("lag (ms) - or distance (um) for a scanning measurement")
    ax.set_ylabel("normalised correlation")
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    results = process()
    report(results)
    figure = plot(results)
    plt.show()
