# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
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

# %% [markdown]
# ## Afterpulsing removal through de-convolution

# %% [markdown]
# ### Derivation
# Let us define $F(t)$ as the signal at time $t$, and $p(t)$ as the probability density per unit time of afterpulsing at time $t$. We have, therefore:
# $$
# \delta I(t) = \int \delta F(t-t')p(t')dt'
# $$
#
# Presenting $F$ and $p$ as reverse Fourier transforms
# $$
# \delta F(t) = \frac{1}{\sqrt{2\pi}}\int d\omega\delta F(\omega)e^{i\omega t}
# $$
# $$
# \delta p(t) = \frac{1}{\sqrt{2\pi}}\int d\omega'\delta p(\omega')e^{i\omega't}
# $$
#
# and plugging back in:
# $$
# \delta I(t) = \frac{1}{2\pi}\int d\omega d\omega'\delta F(\omega)p(\omega') \int_{-\infty}^\infty e^{i[\omega(t-t')+\omega't']}dt'
# $$
#
# we notice that the last integral yields a delta function $e^{i\omega t}\delta(\omega'-\omega)$:
# $$
# = \int\delta\omega F(\omega)p(\omega)e^{i\omega t}
# $$
#
# Now, for the ACF. We conjugate $\delta I(t')$ for convenience (it is real):
# $$
# G(t)=\langle\delta I(0)\delta I(t)\rangle = \frac{1}{2T}\int dt' \langle\delta I^*(t')\delta I(t'+t)\rangle
# $$
# $$
# = \frac{1}{2T}\frac{1}{(2\pi)^2}\int d\omega d\omega'\delta F^*(\omega)p^*(\omega)\delta F(\omega')p(\omega')\int dt' e^{-i(\omega'-\omega) t'}e^{i\omega' t}
# $$
#
# and after noticing yet another delta function:
# $$
# =\frac{1}{2T}\frac{1}{(2\pi)^2}\int d\omega |F(\omega)|^2|p(\omega)|^2e^{i\omega t}
# $$
#
# Noticing that this is a reverse Fourier transform, we can perform a Fourier transform to extract the product of $F$ and $p$:
# $$
# FT\left[G(t)\right]=G(\omega)=\frac{1}{2T}\frac{1}{(2\pi)^{3/2}}|F(\omega)|^2|p(\omega)|^2
# $$
#
# So, the Fourier transform of the afterpulsed ACF is proportional to the Fourier transform of the afterpulsing-clean ACF times the Fourier transform of the afterpulsing ACF. In other words, the afterpulsing seperates from the signal in Fourier  space.
#
# We already saw (see above, for old detector data) that cross-correlating signal at different gates (signal vs. white noise tail) yields the afterpulsing. Until now we subtracted it from the ACF on grounds that the afterpulsing was strong at lag-times where the clean ACF was unintersting. We can attempt now to get rid of afterpulsing in a more general way:
# $$
# \frac{G_{ap\ signal}(\omega)}{G_{ap}(\omega)} = \frac{|F(\omega)|^2|p(\omega)|^2}{|p(\omega)|^2}=|F(\omega)|^2 = G_{signal}(\omega)
# $$
#
# Where $G_{ap\ signal}(\omega)$ is Fourier-transformed afterpulsed signal, $G_{ap}(\omega)$ is the Fourier-transformed cross-correlated signal afterpulsing and $G_{signal}(\omega)$ is the afterpulse-free Fourier-transformed ACF.
#
# We may need to reverse-transform $G_{signal}(\omega)$, since we need the regular-time ACF to perform the Hankel transform on (spatial 2D Fourier transform) for deconvoluting the PSF from the structure factor (in exactly the same process!).
#
# NOTE: is there a way to do it without reverse-transforming?

# %% [markdown]
# ### Testing

# %% [markdown]
# Here we would like to test the above method on old detector measurements (since for them we can compare this method to the well-tested subtraction method). Let us define a prototype function which accepts a 'avg_cf_cr' afterpulsed correlation function and an afterpulsing, and returns their inverse-transformed quotient (see derivation above):

# %%
from types import SimpleNamespace
from utilities.helper import largest_n


def deconvolve_afterpulse(
    lag,
    ap_signal_cf_cr,
    ap_cf_cr,
    n_bins=2**17,
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
        extrap_x_lims=Limits(-np.inf, np.inf),
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
DATA_DATE = "10_05_2018"
confocal_template = "bp300_angular_exc_*.mat"
label = "300 bp ATTO (old detector)"

DATA_PATH = DATA_ROOT / DATA_DATE / DATA_TYPE

# load experiment
exp1 = SFCSExperiment(name=label)
exp1.load_experiment(
    confocal_path_template=DATA_PATH / confocal_template,
    should_plot=True,
    should_use_preprocessed=False,  # TODO: load anew if not found
    should_re_correlate=True,  # True
    should_subtract_afterpulse=False,
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

gate1_ns = Limits(2, 15)
gate2_ns = Limits(35, 95)

corr_names = ("AB", "BA")
XCF_AB, XCF_BA = meas1.cross_correlate_data(
    cf_name="fl_vs_wn",
    corr_names=corr_names,
    gate1_ns=gate1_ns,
    gate2_ns=gate2_ns,
    subtract_spatial_bg_corr=True,
    should_subtract_afterpulse=False,
)

# %% [markdown]
# Now, we need to perform Fourier transforms. To prepare the functions for the transform we would do well to trim the noisy tail end and to symmetrize them.
#
# Let's define them and take a look first:

# %%
# Normalizing and defining the ACF of the afterpulsing (from xcorr) and the afterpulsed signal
pulse_period_ns = 100
sbtrct_AB_BA_arr = np.empty(XCF_AB.corrfunc.shape)
for idx, (corrfunc_AB, corrfunc_BA, countrate_pair) in enumerate(
    zip(XCF_AB.corrfunc, XCF_BA.corrfunc, XCF_AB.countrate_list)
):
    norm_factor = pulse_period_ns / (
        gate2_ns.interval() / countrate_pair.b - gate1_ns.interval() / countrate_pair.a
    )
    sbtrct_AB_BA_arr[idx] = norm_factor * (corrfunc_AB - corrfunc_BA)
sbtrct_AB_BA = sbtrct_AB_BA_arr.mean(axis=0)

ap_t_old = sbtrct_AB_BA
ap_signal_t_old = np.copy(exp1.confocal.cf["confocal"].avg_cf_cr)
lag_signal_old = np.copy(exp1.confocal.cf["confocal"].lag)
lag_ap_old = np.copy(XCF_AB.lag)

# plotting
with Plotter(
    super_title="Logarithmic Scale",
    xlim=(1e-3, 1e1),
    ylim=(-500, exp1.confocal.cf["confocal"].g0 * 1.3),
    x_scale="log",
) as ax:

    ax.plot(lag_signal_old, ap_signal_t_old, label="Afterpulsed Signal")
    ax.plot(lag_ap_old, ap_t_old, label="X-Corr Afterpulsing")
    ax.legend()

with Plotter(
    super_title="Linear Scale", xlim=(-1e-2, 1e-1), ylim=(-1e4, 1e5), x_scale="linear"
) as ax:

    ax.plot(lag_signal_old, ap_signal_t_old, label="Afterpulsed Signal")
    ax.plot(lag_ap_old, ap_t_old, label="X-Corr Afterpulsing")
    ax.legend()

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
    confocal_path_template=DATA_PATH / confocal_template,
    should_plot=True,
    should_use_preprocessed=False,  # TODO: load anew if not found
    should_re_correlate=True,
    should_subtract_afterpulse=True,
    should_use_inherent_afterpulsing=True,
)

# %% [markdown]
# Extrapolating over noise to facillitate the Fourier transform:

# %%
clean_signal_old_t = np.copy(exp2.confocal.cf["confocal"].avg_cf_cr)

n_bins = 2**17
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
# ## Testing on new detector data...

# %% [markdown]
# importing the data:

# %%
# DATA_DATE = "06_06_2022"
# confocal_template = "atto300bp_angular_exc_190337_*.pkl"
# label = "300 bp ATTO (new detector)"

DATA_DATE = "29_03_2022"
confocal_template = "bp300_20uW_angular_exc_172325_*.pkl"
label = "300 bp YOYO (new detector)"

# DATA_DATE = "13_03_2022"
# confocal_template = "atto_12uW_FR_static_exc_182414_*.pkl"
# label = "ATTO static new detector"

DATA_PATH = DATA_ROOT / DATA_DATE / DATA_TYPE

# load experiment
exp3 = SFCSExperiment(name=label)
exp3.load_experiment(
    confocal_path_template=DATA_PATH / confocal_template,
    should_plot=True,
    should_use_preprocessed=False,  # TODO: load anew if not found
    should_re_correlate=True,  # True
    should_subtract_afterpulse=False,
    #     file_selection="Use 1-5, 8-10",  # TESTESTEST
)

# save processed data (to avoid re-processing)
exp3.save_processed_measurements()

# Show countrate
print(f"Count-Rate: {exp3.confocal.avg_cnt_rate_khz} kHz")

# calibrate TDC
exp3.calibrate_tdc(should_plot=True)

# %% [markdown]
# Get the inverse quotient:

# %%
meas3 = deepcopy(exp3.confocal)
meas3.xcf = {}

gate1_ns = Limits(2, 15)
gate2_ns = Limits(35, 95)

corr_names = ("AB", "BA")
XCF_AB, XCF_BA = meas3.cross_correlate_data(
    cf_name="fl_vs_wn",
    corr_names=corr_names,
    gate1_ns=gate1_ns,
    gate2_ns=gate2_ns,
    subtract_spatial_bg_corr=True,
    #     subtract_spatial_bg_corr=False,
    should_subtract_afterpulse=False,
)

XCF_AB.average_correlation()
XCF_BA.average_correlation()

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
    norm_AB_arr[idx] = norm_factor * corrfunc_AB
    norm_BA_arr[idx] = norm_factor * corrfunc_BA
    sbtrct_AB_BA_arr[idx] = norm_AB_arr[idx] - norm_BA_arr[idx]
sbtrct_AB_BA = sbtrct_AB_BA_arr.mean(axis=0)
norm_AB = norm_AB_arr.mean(axis=0)
norm_BA = norm_BA_arr.mean(axis=0)

ap_new_t = sbtrct_AB_BA
ap_signal_new_t = np.copy(exp1.confocal.cf["confocal"].avg_cf_cr)
lag_ap_signal_new_t = np.copy(exp1.confocal.cf["confocal"].lag)
lag_ap_new = np.copy(XCF_AB.lag)

# plotting
with Plotter(
    super_title="Logarithmic Scale",
    xlim=(1e-3, 1e1),
    ylim=(-500, exp1.confocal.cf["confocal"].g0 * 1.3),
    x_scale="log",
) as ax:

    ax.plot(lag_signal_old, ap_signal_t_old, label="Afterpulsed Signal")
    ax.plot(
        lag_signal_old,
        ap_signal_t_old - unify_length(ap_t_old, len(ap_signal_t_old)),
        label="Afterpulse-Subtracted Signal",
    )
    ax.plot(lag_ap_old, ap_t_old, label="X-Corr Afterpulsing")
    ax.plot(lag_ap_old, norm_AB, label="norm_AB")
    ax.plot(lag_ap_old, norm_BA, label="norm_BA")
    ax.legend()

# %%
new_detector_quotient_inherent_FT = deconvolve_afterpulse(
    lag_ap_new, ap_signal_new_t, ap_new_t, should_plot=SHOULD_PLOT
)

# %% [markdown]
# Comparing to regularly subtracted afterpulsing (calibrated and inherent):

# %%
# load experiment
exp4 = SFCSExperiment(name=label)
exp4.load_experiment(
    confocal_path_template=DATA_PATH / confocal_template,
    should_plot=True,
    should_use_preprocessed=True,  # TODO: load anew if not found
    should_re_correlate=True,
    should_subtract_afterpulse=True,
    should_use_inherent_afterpulsing=True,
    inherent_afterpulsing_gates=(gate1_ns, gate2_ns),
    #     file_selection="Use 1-5, 8-10",
)

# save processed data (to avoid re-processing)
exp4.save_processed_measurements()

# Show countrate
print(f"Count-Rate: {exp4.confocal.avg_cnt_rate_khz} kHz")

# load experiment
exp5 = SFCSExperiment(name=label)
exp5.load_experiment(
    confocal_path_template=DATA_PATH / confocal_template,
    should_plot=True,
    should_use_preprocessed=True,  # TODO: load anew if not found
    should_re_correlate=True,
    should_subtract_afterpulse=True,
    should_use_inherent_afterpulsing=False,
    #     file_selection="Use 1-5, 8-10",
)

# save processed data (to avoid re-processing)
exp5.save_processed_measurements()

# Show countrate
print(f"Count-Rate: {exp5.confocal.avg_cnt_rate_khz} kHz")

# %% [markdown]
# Getting the inverse-quotient for the calibrated afterpulse (just to test):

# %%
ap_new_t_calibrated = exp5.confocal.cf["confocal"].afterpulse
new_detector_quotient_calibrated_FT = deconvolve_afterpulse(
    lag_ap_signal_new_t, ap_signal_new_t, ap_new_t_calibrated, should_plot=SHOULD_PLOT
)

# %%
clean_signal_inherent = exp4.confocal.cf["confocal"].avg_cf_cr
lag_s = exp4.confocal.cf["confocal"].lag * 1e-3

clean_signal_calibrated = exp5.confocal.cf["confocal"].avg_cf_cr

with Plotter(
    super_title="Comparing the old 'clean' signal\nwith various methods of (new detector)",
    #     xlim=(-1e6, 1e6),
    ylim=(-1000, max(clean_signal_old_t_interp) * 2),
    x_scale="log",
) as ax:

    ax.plot(lag_s_interp, clean_signal_old_t_interp, label="$G(t)_{signal} (old)$")
    ax.plot(lag_s, clean_signal_inherent * 2.7, label="$G(t)_{subtracted,\ inherent\ ap}$")
    ax.plot(lag_s, clean_signal_calibrated * 2.7, label="$G(t)_{subtracted,\ calibrated\ ap}$")
    ax.plot(
        new_detector_quotient_inherent_FT.t,
        np.real(new_detector_quotient_inherent_FT.ft) * 3.3,
        label="$G(t)_{deconvolved,\ inherent\ ap}$",
    )
    ax.plot(
        new_detector_quotient_calibrated_FT.t,
        np.real(new_detector_quotient_calibrated_FT.ft) * 3.7,
        label="$G(t)_{deconvolved,\ calibrated\ ap}$",
    )
    ax.legend()

# %% [markdown]
# Comparing the afterpulsings:

# %%
clean_signal_inherent = exp4.confocal.cf["confocal"]
clean_signal_calibrated = exp5.confocal.cf["confocal"]
lag = exp4.confocal.cf["confocal"].lag

with Plotter(
    super_title="Comparing the old 'clean' signal\nwith various methods of (new detector)",
    #     xlim=(-1e6, 1e6),
    ylim=(-1000, max(clean_signal_old_t_interp) * 2),
    x_scale="log",
) as ax:

    ax.plot(
        lag,
        unify_length(clean_signal_calibrated.avg_cf_cr / 3.5, len(lag)),
        label="$new - cal. ap$",
    )
    ax.plot(lag, unify_length(clean_signal_calibrated.afterpulse, len(lag)), label="new - cal. ap")
    ax.plot(lag, unify_length(clean_signal_inherent.afterpulse, len(lag)), label="new - xcorr ap")
    ax.plot(lag, unify_length(clean_signal_calibrated.afterpulse, len(lag)), label="new - cal. ap")
    ax.plot(lag, unify_length(clean_signal_inherent.afterpulse, len(lag)), label="new - xcorr ap")
    ax.legend()

# %%
gate1_ns = Limits(2, 15)
gate2_ns = Limits(35, 95)

# meas = deepcopy(exp1.confocal) # OLD detector
meas = deepcopy(exp3.confocal)  # NEW detector

# XCF_AB, XCF_BA, XCF_BB = old_meas.cross_correlate_data(
XCF_AB, XCF_BA, XCF_BB = meas.cross_correlate_data(
    cf_name="test",
    corr_names=("AB", "BA", "BB"),
    gate1_ns=gate1_ns,
    gate2_ns=gate2_ns,
    subtract_spatial_bg_corr=True,  # TEST ME!
    should_subtract_afterpulse=False,
)

XCF_AB.average_correlation()
XCF_BA.average_correlation()
XCF_BB.average_correlation()

# %%
pulse_period_ns = 100

norm_ab = XCF_AB.countrate_b * (pulse_period_ns / gate2_ns.interval()) / XCF_AB.countrate_a
norm_ba = XCF_BA.countrate_a * (pulse_period_ns / gate1_ns.interval()) / XCF_BA.countrate_b
norm_bb = pulse_period_ns / gate2_ns.interval()
# norm_ab, norm_ba, norm_bb = 1, 1, 1

pulse_period_ns = 100
sbtrct_AB_BA_arr = np.empty(XCF_AB.corrfunc.shape)
for idx, (corrfunc_AB, corrfunc_BA, countrate_pair) in enumerate(
    zip(XCF_AB.corrfunc, XCF_BA.corrfunc, XCF_AB.countrate_list)
):
    norm_factor = pulse_period_ns / (
        gate2_ns.interval() / countrate_pair.b - gate1_ns.interval() / countrate_pair.a
    )
    sbtrct_AB_BA_arr[idx] = norm_factor * (corrfunc_AB - corrfunc_BA)
sbtrct_AB_BA = sbtrct_AB_BA_arr.mean(axis=0)

with Plotter(
    #         super_title="Comparing inherent afterpulsings\nfrom different xcorr gates",
    #     ylim=(-1, 5),
    x_scale="log",
) as ax:
    ax.plot(XCF_AB.lag, XCF_AB.avg_cf_cr * norm_ab, label=XCF_AB.name)
    ax.plot(XCF_BA.lag, XCF_BA.avg_cf_cr * norm_ba, label=XCF_BA.name)
    ax.plot(XCF_BB.lag, XCF_BB.avg_cf_cr * norm_bb, label=XCF_BB.name)
    ax.plot(meas.cf["confocal"].lag, meas.cf["confocal"].avg_cf_cr, label=meas.name)
    ax.plot(XCF_BB.lag, sbtrct_AB_BA, label="AB - BA")
    #     ax.plot(
    #         lag,
    #         unify_length(clean_signal_calibrated.afterpulse, len(lag)),
    #         label="calibrated afterpulsing",
    #     )
    ax.legend()

meas_afterpulse_subtracted = deepcopy(exp5.confocal)  # NEW detector

with Plotter(
    #         super_title="Comparing inherent afterpulsings\nfrom different xcorr gates",
    #     ylim=(-1000, meas.cf["confocal"].g0 * 2),
    x_scale="log",
) as ax:
    ax.plot(XCF_BB.lag, sbtrct_AB_BA, label="AB - BA")
    ax.plot(
        meas_afterpulse_subtracted.cf["confocal"].lag,
        meas_afterpulse_subtracted.cf["confocal"].afterpulse,
    )
    ax.legend()

# %% [markdown]
# Play sound when done:

# %%
Beep(4000, 300)
