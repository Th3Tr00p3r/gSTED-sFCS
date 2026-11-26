"""
Static FCS measurements: raw bytes -> photons -> lifetime histogram (with fits) -> ACF.

Run from anywhere (Spyder, a notebook cell, or 'python <this file>'). For every measurement
found in DAY it leaves, in the dictionary 'results[label]':

    raw_bytes   the FPGA byte stream exactly as saved (uint8, 7 bytes per photon)
    meta        the 'full_data' dictionary of the .pkl file (laser mode, detector, ...)
    meas        the SolutionSFCSMeasurement object
    photons     meas.data[0].raw - .pulse_runtime (laser pulses since the start),
                .coarse / .fine (TDC counters), .delay_time (ns after the laser pulse)
    t_ns, hist, hist_err    the lifetime histogram
    lifetime_fit            FitParams of the fit to it (.beta, .beta_error, .fit_func)
    cf          the CorrFunc object - .lag (ms), .avg_cf_cr, .error_cf_cr, .normalized,
                .g0, and .cf_cr (one row per split)

and draws two figures: the lifetime histograms with their fits (one axes), and the ACFs.

Lifetime fits
-------------
* excitation only: a single exponential with background (the program's own
  TDCCalibration.fit_lifetime_hist). Its lifetime, tau0, is then held fixed in:
* excitation + depletion: fit_tools.fit_gaussian_depletion_hist - the decay under a
  co-aligned GAUSSIAN depletion beam,

      H(t) = A/tau0 * exp(-t/tau0) * Gamma(a) * P(a, xi*t/tau0) / (xi*t/tau0)^a + bg

  with a = (w_dep/w_det)^2 and xi = I0/I_sat. This is the model for the vortex plate moved
  OUT of the beam. It is not the model for a doughnut - with the plate in, read the fitted
  xi as a description of the curve, not as I0/I_sat.

  The model counts t from the excitation pulse, so the histogram's time axis is shifted to
  put t = 0 at the peak of the excitation-only histogram before fitting.
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

from data_analysis.correlation_function import SolutionSFCSMeasurement  # noqa: E402
from utilities import fit_tools  # noqa: E402
from utilities.file_utilities import load_file_dict  # noqa: E402

######################
# what to process
######################
DAY = Path("/Users/oleg/Documents/Experiments/tmp/12_11_2026")

# label -> (file name without '_0.pkl', measurement type for the analysis classes)
MEASUREMENTS = {
    "exc": ("alignment_static_exc", "confocal"),
    "sted": ("alignment_static_sted", "sted"),
    "dep": ("alignment_static_dep", "confocal"),  # depletion beam alone: a background check
}

# skip the first 3 photons of each file (bad time-stamps there, sometimes); None keeps all
BYTE_DATA_SLICE = slice(21, None)

# lifetime fits. Times are in ns AFTER the peak of the excitation-only histogram.
EXC_FIT_RANGE_NS = (0.3, 30)  # single exponential
STED_FIT_RANGE_NS = (0.3, 30)  # Gaussian-depletion model
GEOMETRY_A = 1.296  # (w_dep/w_det)^2 - the diffraction-limited value; use the measured one

SHOULD_CORRELATE = True
ACF_MIN_LAG_MS = 1e-3  # shortest lag shown in the ACF figure


def process(label: str, name: str, meas_type: str) -> dict:
    """Load one measurement and compute its lifetime histogram (and its ACF)."""

    print(f"\n=================== {label}: {name} ===================")
    out = dict(
        raw_bytes=np.load(DAY / f"{name}_0_byte_data.npy", mmap_mode="r"),
        meta=load_file_dict(DAY / f"{name}_0.pkl")["full_data"],
    )

    meas = SolutionSFCSMeasurement(meas_type)
    proc_options = dict(is_verbose=True)
    if BYTE_DATA_SLICE is not None:
        proc_options["byte_data_slice"] = BYTE_DATA_SLICE
    meas.read_fpga_data(DAY / f"{name}_*.pkl", **proc_options)
    out["meas"] = meas
    out["photons"] = meas.data[0].raw

    # lifetime histogram (this also fills photons.delay_time)
    meas.calibrate_tdc()
    is_finite = np.isfinite(meas.tdc_calib.all_hist_norm)
    out["t_ns"] = meas.tdc_calib.t_hist[is_finite]
    out["hist"] = meas.tdc_calib.all_hist_norm[is_finite]
    out["hist_err"] = meas.tdc_calib.error_all_hist_norm[is_finite]

    n_photons = out["photons"].pulse_runtime.size
    print(
        f"{label}: {n_photons:,} photons in {meas.duration_min * 60:.1f} s "
        f"({n_photons / (meas.duration_min * 60) * 1e-3:.1f} kHz)"
    )

    if SHOULD_CORRELATE:
        try:
            out["cf"] = meas.correlate_and_average(cf_name=label)
        except Exception as exc:  # e.g. the correlator library did not load
            print(f"{label}: could not correlate [{exc}]")
    return out


def fit_lifetimes(results: dict) -> None:
    """Fit the excitation-only histogram, then the depleted one with tau0 fixed."""

    exc = results.get("exc")
    if exc is None:
        print("No excitation-only measurement - nothing to fit.")
        return

    # t = 0 at the peak of the excitation-only histogram, for all measurements
    t_zero_ns = exc["t_ns"][np.argmax(exc["hist"])]
    for res in results.values():
        res["t_zero_ns"] = t_zero_ns

    # --- excitation only: single exponential + background
    exc["lifetime_fit"] = fit_tools.curve_fit_lims(
        fit_tools.exponent_with_background_fit,
        [exc["hist"].max(), 4, exc["hist"].min()],
        xs=exc["t_ns"] - t_zero_ns,
        ys=exc["hist"],
        ys_errors=exc["hist_err"].copy(),
        x_limits=fit_tools.Limits(EXC_FIT_RANGE_NS),
    )
    tau0_ns = exc["lifetime_fit"].beta["tau"]
    print(
        f"\nexc: tau0 = {tau0_ns:.3f} +/- {exc['lifetime_fit'].beta_error['tau']:.3f} ns "
        "(single exponential)"
    )

    # --- excitation + depletion: the Gaussian-depletion model, tau0 fixed
    sted = results.get("sted")
    if sted is None:
        return
    sted["lifetime_fit"] = fit_tools.fit_gaussian_depletion_hist(
        sted["t_ns"] - t_zero_ns,
        sted["hist"],
        a=GEOMETRY_A,
        tau_ns=tau0_ns,
        fit_range=STED_FIT_RANGE_NS,
        ys_errors=sted["hist_err"].copy(),
    )
    fit = sted["lifetime_fit"]
    xi, xi_err = fit.beta["xi"], fit.beta_error["xi"]
    print(
        f"sted: xi = I0/I_sat = {xi:.2f} +/- {xi_err:.2f}  "
        f"(Gaussian-depletion model, a = {GEOMETRY_A}, tau0 fixed)"
    )
    # the same thing without the model: the decay time right after the pulse
    try:
        initial = fit_tools.initial_decay_time(
            sted["t_ns"] - t_zero_ns,
            sted["hist"],
            tau0_ns=tau0_ns,
            bg=fit.beta["bg"],
        )
        xi_initial = fit_tools.xi_from_initial_rate(tau0_ns, initial["tau_init_ns"], GEOMETRY_A)
        print(
            f"sted: initial decay time {initial['tau_init_ns']:.3f} ns -> "
            f"xi = {xi_initial:.2f} (from the initial rate alone)"
        )
        sted["initial_decay"] = initial
    except ValueError as exc:
        print(f"sted: initial decay time not available [{exc}]")


def plot_lifetimes(results: dict):
    """All lifetime histograms and their fits, on one axes."""

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = dict(exc="tab:blue", sted="tab:red", dep="tab:gray")
    for label, res in results.items():
        t = res["t_ns"] - res.get("t_zero_ns", 0)
        color = colors.get(label)
        ax.plot(t, res["hist"], ".", ms=3, color=color, label=f"{label} - data")
        fit = res.get("lifetime_fit")
        if fit is None:
            continue
        t_fit = np.linspace(fit.x.min(), fit.x.max(), 1000)
        if label == "exc":
            fit_label = f"exc - exponential, $\\tau_0$ = {fit.beta['tau']:.2f} ns"
        else:
            fit_label = (
                f"{label} - Gaussian depletion, $\\xi$ = {fit.beta['xi']:.2f} "
                f"(a = {GEOMETRY_A}, $\\tau_0$ fixed)"
            )
        ax.plot(t_fit, fit.fit_func(t_fit, *fit.beta.values()), "-", color="k", lw=1.2)
        ax.plot([], [], "-", color="k", lw=1.2, label=fit_label)
    ax.set_yscale("log")
    ax.set_xlim(-3, 40)
    ax.set_xlabel("time after the excitation pulse (ns)")
    ax.set_ylabel("photons per laser pulse per ns (normalised histogram)")
    ax.set_title(f"Lifetime histograms and fits - {DAY.name}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_acfs(results: dict):
    """The averaged ACFs (count-rate normalised), on one axes."""

    fig, ax = plt.subplots(figsize=(9, 6))
    g0_max = 0.0
    for label, res in results.items():
        cf = res.get("cf")
        if cf is None or label == "dep":  # the depletion beam alone is background only
            continue
        in_range = cf.lag > ACF_MIN_LAG_MS  # shorter lags are dominated by afterpulsing
        ax.errorbar(
            cf.lag[in_range],
            cf.avg_cf_cr[in_range],
            cf.error_cf_cr[in_range],
            fmt=".",
            ms=3,
            elinewidth=0.5,
            label=f"{label}: G0 = {cf.g0:.0f}",
        )
        g0_max = max(g0_max, cf.g0)
    if g0_max > 0:
        ax.set_ylim(-0.1 * g0_max, 1.5 * g0_max)
    ax.set_xscale("log")
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("G(lag) x count rate")
    ax.set_title(f"Correlation functions - {DAY.name}")
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    results = {
        label: process(label, name, meas_type)
        for label, (name, meas_type) in MEASUREMENTS.items()
        if (DAY / f"{name}_0.pkl").exists()
    }
    fit_lifetimes(results)
    lifetime_fig = plot_lifetimes(results)
    acf_fig = plot_acfs(results)
    plt.show()
