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
from utilities.helper import Limits, fourier_transform_1d, extrapolate_over_noise

#################################################
# Setting up data path and other global constants
#################################################

DATA_ROOT = Path("D:\OneDrive - post.bgu.ac.il\gSTED_sFCS_Data")  # Laptop/Lab PC (same path)
# DATA_ROOT = Path("D:\???")  # Oleg's
DATA_TYPE = "solution"

FORCE_PROCESSING = False
FORCE_PROCESSING = True

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


def deconvolve_afterpulse(lag, ap_signal_cf_cr, ap_cf_cr, should_plot=False):
    """Doc."""

    lag_s = lag * 1e-3  # ms to seconds

    ####################
    # interp/extrap both
    ####################

    n_bins = 2 ** 17
    n_robust = 0
    x_lims = Limits(1e-6, 1e-3)  # (1 us to 1 ms)
    y_lims = Limits(50, np.inf)
    interp_type = "gaussian"
    extrap_x_lims = Limits(-1, 5e-3)  # up to 5 ms

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
        bin_size=1e-7,  # meaning 100 ns
        should_normalize=True,
        should_plot=should_plot,
    )

    ap_t, ap_ft_interp, ap_w, ap_fw = fourier_transform_1d(
        lag_s_interp,
        ap_cf_cr_interp,
        bin_size=1e-7,  # meaning 100 ns
        should_normalize=True,
        should_plot=should_plot,
    )

    ap_signal_fw_baseline = np.mean(np.real(ap_signal_fw[abs(ap_signal_w) > 1e7]))
    ap_fw_baseline = np.mean(np.real(ap_fw[abs(ap_w) > 1e7]))

    print("fw_signal_baseline: ", ap_signal_fw_baseline)
    print("fw_ap_baseline: ", ap_fw_baseline)

    ap_signal_fw += 1 - ap_signal_fw_baseline
    ap_fw += 1 - ap_fw_baseline

    if should_plot:
        with Plotter(
            super_title="Comparing the transforms",
            #     xlim=(-1e6, 1e6),
            #     ylim=(0.99, 1.04),
        ) as ax:
            ax.plot(ap_signal_w, np.real(ap_signal_fw), label="ap signal")
            ax.plot(ap_w, np.real(ap_fw), label="ap")

    ######################
    # Getting the quotient
    ######################

    quotient = ap_signal_fw / ap_fw

    if should_plot:
        with Plotter() as ax:
            ax.plot(ap_w, np.real(quotient), label="quotient (real part)")
            ax.plot(ap_w, np.imag(quotient), label="quotient (imaginary part)")
            ax.legend()

    ############################################
    # Smoothing before inverse Fourier transform
    ############################################

    n_bins = 2 ** 17
    n_robust = 0
    x_lims = Limits(np.NINF, np.inf)
    x_lims = Limits(3e3, np.inf)
    y_lims = Limits(np.NINF, np.inf)
    interp_type = "gaussian"
    extrap_x_lims = Limits(np.NINF, np.inf)

    gauss_interp_quotient = extrapolate_over_noise(
        ap_w[ap_w.size // 2 :],
        np.real(quotient)[quotient.size // 2 :],
        n_bins=n_bins,
        x_lims=x_lims,
        y_lims=y_lims,
        n_robust=n_robust,
        interp_type=interp_type,
        extrap_x_lims=extrap_x_lims,
        should_plot=should_plot,
    )

    ###########################################
    # Finally, performing the inverse transform
    ###########################################

    ap_w_interp = gauss_interp_quotient.x_interp
    quotient_interp = gauss_interp_quotient.y_interp

    # w_ap_interp = w_ap
    # quotient_interp = quotient

    quotient_w, quotient_fw_interp, quotient_t, quotient_ft = fourier_transform_1d(
        ap_w_interp,
        quotient_interp,
        should_inverse=True,
        is_input_symmetric=False,
        bin_size=np.diff(ap_w_interp)[0],
        #     should_normalize=True,
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
# DATA_DATE = "10_05_2018"
# confocal_template = "bp300_angular_sted_*.mat"
# label = "MATLAB Free-Running 300 bp STED"

DATA_DATE = "05_10_2021"
confocal_template = "sol_static_exc_181325_*.pkl"
label = "ATTO static old detector"

DATA_PATH = DATA_ROOT / DATA_DATE / DATA_TYPE

# load experiment
exp1 = SFCSExperiment(name=label)
exp1.load_experiment(
    confocal_template=DATA_PATH / confocal_template,
    should_plot=True,
    should_use_preprocessed=True,  # TODO: load anew if not found
    should_re_correlate=True,  # True
    should_subtract_afterpulse=False,
)

# save processed data (to avoid re-processing)
exp1.save_processed_measurements()

# Show countrate
print(f"Count-Rate: {exp1.confocal.avg_cnt_rate_khz:.2f} kHz")

# calibrate TDC
exp1.calibrate_tdc(should_plot=False)

# %% [markdown]
# Now, let's get the afterpulsing from cross-corralting gates:

# %%
meas1 = exp1.confocal
meas1.xcf = {}  # "halogen_afterpulsing": meas.cf["confocal"].afterpulse}

gate1_ns = Limits(2, 10)
gate2_ns = Limits(35, 85)

corr_names = ("AB",)
XCF = meas1.cross_correlate_data(
    cf_name="fl_vs_wn",
    corr_names=corr_names,
    gate1_ns=gate1_ns,
    gate2_ns=gate2_ns,
    should_subtract_bg_corr=True,
    should_subtract_afterpulse=False,
)[0]

XCF.average_correlation()

# %% [markdown]
# Now, we need to perform Fourier transforms. To prepare the functions for the transform we would do well to trim the noisy tail end and to symmetrize them.
#
# Let's define them and take a look first:

# %%
# Normalizing and defining the ACF of the afterpulsing (from xcorr) and the afterpulsed signal
gate_width_ns = 100
countrate_a = np.mean([countrate_pair.a for countrate_pair in XCF.countrate_list])
countrate_b = np.mean([countrate_pair.b for countrate_pair in XCF.countrate_list])
norm_factor = countrate_b * (gate_width_ns / gate2_ns.interval()) / countrate_a
# norm_factor = 1
print("afterpulse norm_factor: ", norm_factor)

ap_t_old = XCF.avg_cf_cr * norm_factor
ap_signal_t_old = np.copy(exp1.confocal.cf["confocal"].avg_cf_cr)
lag_signal_old = np.copy(exp1.confocal.cf["confocal"].lag)
lag_ap_old = np.copy(XCF.lag)

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
old_detector_quotient_FT = deconvolve_afterpulse(lag_signal_old, ap_signal_t_old, ap_t_old)

# %% [markdown]
# Generating the Fourier transform of the afterpulse-subtracted (in the usual/old way) signal, to compare with the quotient in Fourier space:

# %%
# load experiment
exp2 = SFCSExperiment(name=label)
exp2.load_experiment(
    confocal_template=DATA_PATH / confocal_template,
    should_plot=True,
    should_use_preprocessed=True,  # TODO: load anew if not found
    should_re_correlate=True,
    should_subtract_afterpulse=True,
    should_use_inherent_afterpulsing=True,
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
factor = 1
# factor = 1.53

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
g0_factor = 1.055e7

with Plotter(
    super_title="Comparing the transforms",
    #     xlim=(-1e6, 1e6),
    ylim=(-100, max(clean_signal_old_t_interp) * 1.1),
    x_scale="log",
) as ax:

    ax.plot(lag_s_interp, clean_signal_old_t_interp, label="$G(t)_{signal}$")
    ax.plot(
        old_detector_quotient_FT.t,
        np.real(old_detector_quotient_FT.ft) * g0_factor,
        label="$G(t)_{quotient}$",
    )
    ax.legend()

# %% [markdown]
# ## Testing on new detector data...

# %% [markdown]
# importing the data:

# %%
# DATA_DATE = "29_03_2022"
# confocal_template = "bp300_20uW_angular_exc_172325_*.pkl"
# label = "300 bp new detector"

DATA_DATE = "13_03_2022"
confocal_template = "atto_12uW_FR_static_exc_182414_*.pkl"
label = "ATTO static new detector"

DATA_PATH = DATA_ROOT / DATA_DATE / DATA_TYPE

# load experiment
exp3 = SFCSExperiment(name=label)
exp3.load_experiment(
    confocal_template=DATA_PATH / confocal_template,
    should_plot=True,
    should_use_preprocessed=True,  # TODO: load anew if not found
    should_re_correlate=True,  # True
    should_subtract_afterpulse=False,
)

# save processed data (to avoid re-processing)
exp3.save_processed_measurements()

# Show countrate
print(f"Count-Rate: {exp3.confocal.avg_cnt_rate_khz} kHz")

# calibrate TDC
exp3.calibrate_tdc(should_plot=False)

# %% [markdown]
# Get the inverse quotient:

# %%
meas3 = exp3.confocal
meas3.xcf = {}

gate1_ns = Limits(2, 10)
gate2_ns = Limits(35, 85)

corr_names = ("AB",)
XCF = meas3.cross_correlate_data(
    cf_name="fl_vs_wn",
    corr_names=corr_names,
    gate1_ns=gate1_ns,
    gate2_ns=gate2_ns,
    should_subtract_bg_corr=True,
    should_subtract_afterpulse=False,
)[0]

XCF.average_correlation()

# %%
# Normalizing and defining the ACF of the afterpulsing (from xcorr) and the afterpulsed signal
gate_width_ns = 98
countrate_a = np.mean([countrate_pair.a for countrate_pair in XCF.countrate_list])
countrate_b = np.mean([countrate_pair.b for countrate_pair in XCF.countrate_list])
norm_factor = countrate_b * (gate_width_ns / gate2_ns.interval()) / countrate_a
# norm_factor = 1
print("afterpulse norm_factor: ", norm_factor)

ap_new_t = XCF.avg_cf_cr * norm_factor
ap_signal_new_t = np.copy(exp3.confocal.cf["confocal"].avg_cf_cr)
lag_ap_signal_new_t = np.copy(exp3.confocal.cf["confocal"].lag)
lag_ap_new = np.copy(XCF.lag)

# plotting
with Plotter(
    super_title="Logarithmic Scale",
    xlim=(1e-3, 1e1),
    ylim=(-500, exp3.confocal.cf["confocal"].g0 * 1.3),
    x_scale="log",
) as ax:

    ax.plot(lag_ap_signal_new_t, ap_signal_new_t, label="Afterpulsed Signal")
    ax.plot(lag_ap_new, ap_new_t, label="X-Corr Afterpulsing")
    ax.legend()

with Plotter(
    super_title="Linear Scale", xlim=(-1e-2, 1e-1), ylim=(-1e4, 1e5), x_scale="linear"
) as ax:

    ax.plot(lag_ap_signal_new_t, ap_signal_new_t, label="Afterpulsed Signal")
    ax.plot(lag_ap_new, ap_new_t, label="X-Corr Afterpulsing")
    ax.legend()

# %%
new_detector_quotient_inherent_FT = deconvolve_afterpulse(lag_new, ap_signal_new_t, ap_new_t)

# %% [markdown]
# Comparing to regularly subtracted afterpulsing (calibrated and inherent):

# %%
# DATA_DATE = "29_03_2022"
# confocal_template = "bp300_20uW_angular_exc_172325_*.pkl"
# label = "300 bp new detector"

DATA_DATE = "13_03_2022"
confocal_template = "atto_12uW_FR_static_exc_182414_*.pkl"
label = "ATTO static new detector"

DATA_PATH = DATA_ROOT / DATA_DATE / DATA_TYPE

# load experiment
exp4 = SFCSExperiment(name=label)
exp4.load_experiment(
    confocal_template=DATA_PATH / confocal_template,
    should_plot=True,
    should_use_preprocessed=True,  # TODO: load anew if not found
    should_re_correlate=True,
    should_subtract_afterpulse=True,
    should_use_inherent_afterpulsing=True,
)

# save processed data (to avoid re-processing)
exp4.save_processed_measurements()

# Show countrate
print(f"Count-Rate: {exp4.confocal.avg_cnt_rate_khz} kHz")

# load experiment
exp5 = SFCSExperiment(name=label)
exp5.load_experiment(
    confocal_template=DATA_PATH / confocal_template,
    should_plot=True,
    should_use_preprocessed=True,  # TODO: load anew if not found
    should_re_correlate=True,
    should_subtract_afterpulse=True,
    should_use_inherent_afterpulsing=False,
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
    lag_new, ap_signal_new_t, ap_new_t_calibrated
)

# %%
clean_signal_inherent = exp4.confocal.cf["confocal"].avg_cf_cr
lag_s = exp4.confocal.cf["confocal"].lag * 1e-3

clean_signal_calibrated = exp5.confocal.cf["confocal"].avg_cf_cr

g0_factor = 1.335e7
# g0_factor = 1

with Plotter(
    super_title="Comparing the 'clean' signal\nwith the inverse transform (new detector)",
    #     xlim=(-1e6, 1e6),
    ylim=(-1000, np.median(clean_signal_inherent) * 100),
    x_scale="log",
) as ax:

    ax.plot(lag_s_interp, clean_signal_old_t_interp * 1.3, label="$G(t)_{signal}$")
    ax.plot(lag_s, clean_signal_inherent, label="$G(t)_{subtracted,\ inherent\ ap}$")
    ax.plot(lag_s, clean_signal_calibrated, label="$G(t)_{subtracted,\ calibrated\ ap}$")
    ax.plot(
        new_detector_quotient_inherent_FT.t,
        np.real(new_detector_quotient_inherent_FT.ft) * g0_factor,
        label="$G(t)_{deconvolved,\ inherent\ ap}$ x " + f"{g0_factor:.2e}",
    )
    ax.plot(
        new_detector_quotient_calibrated_FT.t,
        np.real(new_detector_quotient_calibrated_FT.ft) * g0_factor,
        label="$G(t)_{deconvolved,\ calibrated\ ap}$ x " + f"{g0_factor:.2e}",
    )
    ax.legend()

# %% [markdown]
# Play sound when done:

# %%
Beep(4000, 300)
