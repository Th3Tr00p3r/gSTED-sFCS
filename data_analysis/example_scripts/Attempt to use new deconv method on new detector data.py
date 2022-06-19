# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# - Import core and 3rd party modules
# - Move current working directory to project root (if needed) and import project modules
# - Set data paths and other constants

# %%
######################################
# importing core and 3rd-party modules
######################################

import os
import pickle
import re
import scipy
from pathlib import Path
from winsound import Beep
from contextlib import suppress
from copy import deepcopy

import matplotlib as mpl

mpl.use("nbAgg")
import numpy as np
from IPython.core.debugger import set_trace
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft

###############################################
# Move to project root to easily import modules
###############################################

try:  # avoid changes if already set
    print("Working from: ", PROJECT_ROOT)
except NameError:
    try:  # running from Spyder
        PROJECT_ROOT = Path(__file__).resolve()
    except NameError:  # running as Jupyter Notebook
        PROJECT_ROOT = Path(os.getcwd()).resolve().parent.parent
    os.chdir(PROJECT_ROOT)
    print("Working from: ", PROJECT_ROOT)

from data_analysis.correlation_function import CorrFunc, SFCSExperiment, SolutionSFCSMeasurement
from data_analysis.software_correlator import CorrelatorType, SoftwareCorrelator
from utilities.display import Plotter, get_gradient_colormap
from utilities.file_utilities import (
    default_system_info,
    load_mat,
    load_object,
    save_object,
    save_processed_solution_meas,
)
from utilities.fit_tools import multi_exponent_fit
from utilities.helper import Limits, fourier_transform_1d, extrapolate_over_noise, unify_length

#################################################
# Setting up data path and other global constants
#################################################

DATA_ROOT = Path("D:\OneDrive - post.bgu.ac.il\gSTED_sFCS_Data")  # Laptop/Lab PC (same path)
# DATA_ROOT = Path("D:\???")  # Oleg's
DATA_TYPE = "solution"

# FORCE_PROCESSING = False
FORCE_PROCESSING = True

SHOULD_PLOT = True

from types import SimpleNamespace
from utilities.helper import largest_n


def deconvolve_afterpulse(
    lag,
    ap_signal_cf_cr,
    ap_cf_cr,
    n_bins=2 ** 17,
    bin_size=1e-7,  # 100 ns (lag delta of first correlator)
    n_robust=0,
    ifft_x_lims=None,
    should_plot=False,
):
    """Doc."""

    lag_s = lag * 1e-3  # ms to seconds

    ####################
    # interp/extrap both
    ####################

    x_lims = Limits(5e-7, 1e-3)  # (1 us to 1 ms)
    #     y_lims = Limits(10, np.inf)
    y_lims = Limits(0.1, np.inf)
    interp_type = "gaussian"
    extrap_x_lims = Limits(-1, 1e-3)  # up to 1 ms

    gauss_interp_ap_signal = extrapolate_over_noise(
        lag_s,
        ap_signal_cf_cr,
        n_bins=n_bins,
        x_lims=x_lims,
        y_lims=y_lims,
        n_robust=n_robust,
        interp_type=interp_type,
        extrap_x_lims=extrap_x_lims,
        should_plot=should_plot,
    )

    gauss_interp_ap = extrapolate_over_noise(
        lag_s,
        ap_cf_cr,
        n_bins=n_bins,
        x_lims=x_lims,
        y_lims=y_lims,
        n_robust=n_robust,
        interp_type=interp_type,
        extrap_x_lims=extrap_x_lims,
        should_plot=should_plot,
    )

    ap_signal_cf_cr_interp = gauss_interp_ap_signal.y_interp
    ap_cf_cr_interp = gauss_interp_ap.y_interp
    lag_s_interp = gauss_interp_ap.x_interp

    ###########################
    # Fourier-transforming both
    ###########################

    ap_signal_t, ap_signal_ft_interp, ap_signal_w, ap_signal_fw = fourier_transform_1d(
        lag_s_interp,
        ap_signal_cf_cr_interp,
        bin_size=bin_size,
        should_normalize=True,
        should_plot=should_plot,
    )

    ap_t, ap_ft_interp, ap_w, ap_fw = fourier_transform_1d(
        lag_s_interp,
        ap_cf_cr_interp,
        bin_size=bin_size,
        should_normalize=True,
        should_plot=should_plot,
    )

    ap_signal_fw_baseline = np.mean(np.real(ap_signal_fw[abs(ap_signal_w) > 1e7]))
    ap_fw_baseline = np.mean(np.real(ap_fw[abs(ap_w) > 1e7]))

    ap_signal_fw += 1 - ap_signal_fw_baseline
    ap_fw += 1 - ap_fw_baseline

    if should_plot:
        with Plotter(super_title="Comparing the transforms") as ax:
            ax.plot(ap_signal_w, np.real(ap_signal_fw), label="ap signal")
            ax.plot(ap_w, np.real(ap_fw), label="ap")

    ######################
    # Getting the quotient
    ######################

    quotient = ap_signal_fw / ap_fw

    if should_plot:
        with Plotter(super_title="Quotient of the transforms") as ax:
            ax.plot(ap_w, np.real(quotient), label="real part")
            ax.plot(ap_w, np.imag(quotient), label="imaginary part")
            ax.legend()

    ############################################
    # Smoothing before inverse Fourier transform
    ############################################

    ap_w_positive = ap_w[ap_w.size // 2 :]
    quotient_positive = np.real(quotient)[quotient.size // 2 :]

    if ifft_x_lims is None:
        factor = 3
        ifft_x_lims = Limits(ap_w_positive[np.argmax(quotient_positive)] * factor, np.inf)
        print("ifft_x_lims.lower: ", ifft_x_lims.lower)  # TESTESTEST
    ifft_y_lims = Limits(np.median(np.real(quotient)), np.inf)

    gauss_interp_quotient = extrapolate_over_noise(
        ap_w_positive,
        quotient_positive,
        n_bins=n_bins,
        x_lims=ifft_x_lims,
        y_lims=ifft_y_lims,
        n_robust=n_robust,
        interp_type="gaussian",
        extrap_x_lims=Limits(np.NINF, np.inf),
        should_plot=should_plot,
    )

    ###########################################
    # Finally, performing the inverse transform
    ###########################################

    ap_w_interp = gauss_interp_quotient.x_interp
    quotient_interp = gauss_interp_quotient.y_interp

    # Normalization to original bin_size is needed since the division above essentially normalizes to 1
    quotient_interp /= bin_size

    quotient_w, quotient_fw_interp, quotient_t, quotient_ft = fourier_transform_1d(
        ap_w_interp,
        quotient_interp,
        should_inverse=True,
        is_input_symmetric=False,
        bin_size=np.diff(ap_w_interp)[0],
        should_plot=should_plot,
    )

    if should_plot or True:  # TESTESTEST
        real_quotient_ft = np.real(quotient_ft)
        with Plotter(
            super_title="$G_{quotient}(t)$",
            ylim=(
                -abs(np.median(real_quotient_ft)) * 1.3,
                np.median(largest_n(real_quotient_ft, 5)) * 1.3,
            ),
            x_scale="log",
        ) as ax:
            ax.plot(quotient_t, real_quotient_ft)

    return SimpleNamespace(
        t=quotient_t,
        ft=quotient_ft,
        w=quotient_w,
        fw=quotient_fw_interp,
    )


# %% [markdown]
# We begin by loading the measurement and calibrating the TDC:

# %%
DATA_DATE = "24_03_2022"
confocal_template = "conc_1uW_low_delay_angular_exc_151408_*.pkl"
label = "Concentrated Plasmids"

DATA_PATH = DATA_ROOT / DATA_DATE / DATA_TYPE

# load experiment
exp1 = SFCSExperiment(name=label)
exp1.load_experiment(
    confocal_template=DATA_PATH / confocal_template,
    should_plot=True,
    #     should_use_preprocessed=True,  # TODO: load anew if not found
    should_re_correlate=True,  # True
    should_subtract_afterpulse=False,
    file_selection="Use 1-20",
)

# save processed data (to avoid re-processing)
exp1.save_processed_measurements()

# Show countrate
print(f"Count-Rate: {exp1.confocal.avg_cnt_rate_khz:.2f} kHz")

# calibrate TDC
exp1.calibrate_tdc(should_plot=True)

# %% [markdown]
# Now, let's get the afterpulsing from cross-corralting gates:

# %%
meas1 = deepcopy(exp1.confocal)
meas1.xcf = {}  # "halogen_afterpulsing": meas.cf["confocal"].afterpulse}

gate1_ns = Limits(3, 15)  # chosen according to TDC calibration
gate2_ns = Limits(35, 42.5)  # chosen according to TDC calibration

corr_names = ("AB", "BA")
XCF_AB, XCF_BA = meas1.cross_correlate_data(
    cf_name="fl_vs_wn",
    corr_names=corr_names,
    gate1_ns=gate1_ns,
    gate2_ns=gate2_ns,
    should_subtract_bg_corr=True,
    should_subtract_afterpulse=False,
)

# %% [markdown]
# Now, we need to perform Fourier transforms. To prepare the functions for the transform we would do well to trim the noisy tail end and to symmetrize them.
#
# Let's define them and take a look first:

# %%
# various xcorrs:
gate_A = (3, 8)
gate_B_list = [(3, 8), (6, 11), (9, 14), (12, 17), (20, 42.5), (42.5, 45.5)]
XCF_dict = {}
for gate_B in gate_B_list:
    XCF_dict[gate_B] = meas1.cross_correlate_data(
        cf_name="fl_vs_wn",
        gate1_ns=gate_A,
        gate2_ns=gate_B,
        should_subtract_bg_corr=True,
        should_subtract_afterpulse=False,
        should_dump_data=False,
    )

# %%
XCF_dict

# %%
# Normalizing and defining the ACF of the afterpulsing (from xcorr) and the afterpulsed signal
pulse_period_ns = 100
sbtrct_AB_BA_arr = np.empty(XCF_AB.corrfunc.shape)
norm_AB_arr = np.empty(XCF_AB.corrfunc.shape)
norm_BA_arr = np.empty(XCF_BA.corrfunc.shape)
for idx, (corrfunc_AB, corrfunc_BA, countrate_pair) in enumerate(
    zip(XCF_AB.corrfunc, XCF_BA.corrfunc, XCF_AB.countrate_list)
):
    norm_factor = pulse_period_ns / (
        gate2_ns.interval() / countrate_pair.b - gate1_ns.interval() / countrate_pair.a
    )
    #     print("gate2_ns.interval() / countrate_pair.b: ", gate2_ns.interval() / countrate_pair.b)
    #     print("gate1_ns.interval() / countrate_pair.a: ", gate1_ns.interval() / countrate_pair.a)
    norm_AB_arr[idx] = norm_factor * corrfunc_AB
    norm_BA_arr[idx] = norm_factor * corrfunc_BA

norm_AB = norm_AB_arr.mean(axis=0)
norm_BA = norm_BA_arr.mean(axis=0)

ap_signal_t_old = np.copy(exp1.confocal.cf["confocal"].avg_cf_cr)
lag_signal_old = np.copy(exp1.confocal.cf["confocal"].lag)
lag_ap_old = np.copy(XCF_AB.lag)

# TESTESTEST
gate_pulse_period_ratio = 1
calibrated_afterpulse = gate_pulse_period_ratio * multi_exponent_fit(
    lag_ap_old, *default_system_info["afterpulse_params"][1]
)

# TESTESTEST

# plotting
with Plotter(
    super_title="Comparison",
    xlim=(1e-3, 1e1),
    ylim=(-500, exp1.confocal.cf["confocal"].g0 * 1.3),
    x_scale="log",
    xlabel="Lag time (ms)",
    ylabel="avg_cf_cr",
) as ax:

    ax.plot(lag_signal_old, ap_signal_t_old, label="Afterpulsed Signal")
    #     ax.plot(
    #         lag_signal_old,
    #         ap_signal_t_old - unify_length(ap_t_old, len(ap_signal_t_old)),
    #         label="Afterpulse-Subtracted Signal",
    #     )
    ax.plot(lag_ap_old, norm_AB - norm_BA, label="X-Corr Afterpulsing")
    ax.plot(lag_ap_old, norm_AB, label="norm_AB")
    ax.plot(lag_ap_old, norm_BA, label="norm_BA")
    ax.plot(
        lag_ap_old, calibrated_afterpulse * 43 / 100, label="Calibrated Afterpulse (5 kHz Halogen)"
    )
    ax.legend()

with Plotter(
    super_title="Various Cross-Correlation 'B' gates",
    xlim=(1e-3, 1e1),
    #     ylim=(-500, exp1.confocal.cf["confocal"].g0 * 1.3),
    ylim=(0, 0.1),
    x_scale="log",
    xlabel="Lag time (ms)",
    ylabel="mean corrfunc",
) as ax:
    for gate_ns, (CF_AB, CF_BA) in XCF_dict.items():
        #         if gate_ns in {(3, 8), (43.5, 45.5)}:
        ax.plot(
            CF_AB.lag,
            CF_AB.corrfunc.mean(axis=0),
            label=f"AP Gated Signal {(3, 8)} vs. {gate_ns} AB",
        )
        ax.plot(
            CF_BA.lag,
            CF_BA.corrfunc.mean(axis=0),
            label=f"AP Gated Signal {(3, 8)} vs. {gate_ns} BA",
        )
    ax.legend()

# gate_list = [(3, 8), (6, 11), (9, 14), (12, 17), (20, 42.5), (42.5, 95)]

# %%
old_detector_quotient_FT = deconvolve_afterpulse(
    lag_signal_old, ap_signal_t_old, ap_t_old, should_plot=SHOULD_PLOT
)

# %% [markdown]
# Generating the Fourier transform of the afterpulse-subtracted (in the usual/old way) signal, to compare with the quotient in Fourier space:

# %%
# load experiment
exp2 = SFCSExperiment(name=label)
exp2.load_experiment(
    confocal_template=DATA_PATH / confocal_template,
    should_plot=True,
    should_use_preprocessed=False,  # TODO: load anew if not found
    should_re_correlate=True,
    should_subtract_afterpulse=True,
    should_use_inherent_afterpulsing=True,
    file_selection="Use 1-20",
)

# %% [markdown]
# Extrapolating over noise to facillitate the Fourier transform:

# %%
clean_signal_old_t = np.copy(exp2.confocal.cf["confocal"].avg_cf_cr)

n_bins = 2 ** 17
n_robust = 8
x_lims = Limits(1e-6, 1e-3)  # (1 us to 1 ms)
y_lims = Limits(50, np.inf)
interp_type = "gaussian"
extrap_x_lims = Limits(-1, 5e-3)

gauss_interp_signal = extrapolate_over_noise(
    lag_signal_old * 1e-3,
    clean_signal_old_t,
    n_bins=n_bins,
    x_lims=x_lims,
    y_lims=y_lims,
    n_robust=n_robust,
    interp_type=interp_type,
    extrap_x_lims=extrap_x_lims,
    should_plot=True,
)

# %% [markdown]
# Fourier-transforming the afterpulsing-subtracted signal:

# %%
lag_s_interp = gauss_interp_signal.x_interp
clean_signal_old_t_interp = gauss_interp_signal.y_interp

t_std_signal, ft_interp_std_signal, w_std_signal, fw_std_signal = fourier_transform_1d(
    lag_s_interp,
    clean_signal_old_t_interp,
    bin_size=1e-7,  # meaning 100 ns
    should_normalize=True,
    should_plot=True,
)

# %% [markdown]
# Plotting the transformed ap-subtracted signal together with the quotient:

# %%
factor = 1e-7

with Plotter(
    subplots=(1, 2),
    super_title="Fourier Transform of $G_{signal}$ vs. Quotient",
    xlabel="$\omega$ ($2\pi\cdot Hz$)",
) as axes:
    axes[0].plot(w_std_signal, np.real(fw_std_signal), label="$G_{signal}$ (real part)")
    axes[0].plot(
        old_detector_quotient_FT.w,
        factor * np.real(old_detector_quotient_FT.fw),
        label="$G(\omega)_{ap\ signal}/G(\omega)_{ap}$ (real part)",
    )
    axes[0].legend()
    axes[0].set_xscale("log")

    axes[1].plot(w_std_signal, np.real(fw_std_signal), label="$G_{signal}$ (real part)")
    axes[1].plot(
        old_detector_quotient_FT.w,
        factor * np.real(old_detector_quotient_FT.fw),
        label="$G(\omega)_{ap\ signal}/G(\omega)_{ap}$ (real part)",
    )
    axes[1].legend()
    axes[1].set_xlim(-1e6, 1e6)

# %% [markdown]
# Directly comparing the clean signal to the inverse transform of the quotient:

# %%
with Plotter(
    super_title="Comparing the transforms",
    #     xlim=(-1e6, 1e6),
    ylim=(-100, max(clean_signal_old_t_interp) * 1.1),
    x_scale="log",
) as ax:

    ax.plot(lag_s_interp, clean_signal_old_t_interp, label="$G(t)_{signal}$")
    ax.plot(
        old_detector_quotient_FT.t,
        np.real(old_detector_quotient_FT.ft),
        label="$G(t)_{quotient}$",
    )
    ax.legend()

# %% [markdown]
# Play sound when done:

# %%
Beep(4000, 300)
