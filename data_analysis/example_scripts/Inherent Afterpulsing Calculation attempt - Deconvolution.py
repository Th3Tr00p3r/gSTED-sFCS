# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
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
# Here we would like to test the above method on old detector measurements (since for them we can compare this method to the well-tested subtraction method).

# %% [markdown]
# We begin by loading the measurement and calibrating the TDC:

# %%
DATA_DATE = "10_05_2018"
confocal_template = "bp300_angular_sted_*.mat"
label = "MATLAB Free-Running 300 bp STED"

DATA_PATH = DATA_ROOT / DATA_DATE / DATA_TYPE

# load experiment
exp = SFCSExperiment(name=label)
exp.load_experiment(
    confocal_template=DATA_PATH / confocal_template,
    should_plot=True,
    should_use_preprocessed=True,  # TODO: load anew if not found
    should_re_correlate=False,  # True
    should_subtract_afterpulse=False,
)

# save processed data (to avoid re-processing)
exp.save_processed_measurements()

# Show countrate
print(f"Count-Rate: {exp.confocal.avg_cnt_rate_khz} kHz")

# calibrate TDC
exp.calibrate_tdc(should_plot=False)

# %% [markdown]
# Now, let's get the afterpulsing from cross-corralting gates:

# %%
meas = exp.confocal
meas.xcf = {}  # "halogen_afterpulsing": meas.cf["confocal"].afterpulse}

gate1_ns = Limits(2, 10)
gate2_ns = Limits(35, 85)

corr_names = ("AB",)
XCF = meas.cross_correlate_data(
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
# defining the ACF of the afterpulsing (from xcorr) and the afterpulsed signal
G_ap_t = np.copy(XCF.avg_cf_cr)
G_ap_signal_t = np.copy(exp.confocal.cf["confocal"].avg_cf_cr)
lag = np.copy(XCF.lag)

# plotting
with Plotter(
    super_title="Logarithmic Scale",
    xlim=(1e-3, 1e1),
    ylim=(-500, exp.confocal.cf["confocal"].g0 * 1.3),
    x_scale="log",
) as ax:

    ax.plot(lag, G_ap_signal_t, label="Afterpulsed Signal")
    ax.plot(lag, G_ap_t, label="X-Corr Afterpulsing")
    ax.legend()

with Plotter(
    super_title="Linear Scale", xlim=(-1e-2, 1e-1), ylim=(-1e4, 1e5), x_scale="linear"
) as ax:

    ax.plot(lag, G_ap_signal_t, label="Afterpulsed Signal")
    ax.plot(lag, G_ap_t, label="X-Corr Afterpulsing")
    ax.legend()

# %% [markdown]
# Checking out the probability density at bin 0:

# %%
lag[G_ap_t == max(G_ap_t)]

# %%
n = 0
G_ap_signal_t[n]
G_ap_signal_t[n] * np.diff(lag)[n] * 1e-3

# %%
G_ap_t[0] * np.diff(lag)[n] * 1e-3

# %% [markdown]
# Testing our Fourier transform and its inverse by performing a round-trip transform on a Gaussian function:

# %%
# # %debug

# define Gaussian on positive axis:
a = 7
sigma = 137  # say, in seconds
t = np.arange(int(1e4))  # say, in seconds
t = np.linspace(0, 1e4, 2**10)  # say, in seconds
ft = a * np.exp(-((t / sigma) ** 2))

# show it
with Plotter(
    super_title="Gaussian",
    xlabel="$t$ ($s$)",
    ylabel="$f(t)$ (Normalized)",
    xlim=(-1e-2, sigma * 4),
) as ax:
    ax.plot(t, ft, "o")

###################
# Fourier transform
###################

r, fr_interp, w, fw = fourier_transform_1d(
    t,
    ft,
)

# show Fourier transform interpolation
with Plotter(
    super_title="Interpolation",
    xlabel="$t$ ($s$)",
    xlim=(-sigma * 4, sigma * 4),
) as ax:
    ax.plot(t, ft, "o", label="before interpolation")
    ax.plot(r, fr_interp, "x", label="after interpolation")
    ax.legend()

# show our Fourier transform
with Plotter(
    super_title="Our Fourier Transform",
    xlabel="$\omega$ ($2\pi\cdot Hz$)",
    #     ylabel="f($\omega$) (UNITS?)",
    xlim=(-1 / sigma * 5, 1 / sigma * 5),
) as ax:
    ax.plot(w, np.real(fw), label="real part")
    ax.plot(w, np.imag(fw), label="imaginary part")
    ax.legend()

###################
# Inverse transform
###################

w_post, fw_interp, t_post, ft_post = fourier_transform_1d(
    w, fw, should_inverse=True, is_input_symmetric=True
)

# show Fourier transform interpolation
with Plotter(
    super_title="Interpolation",
    xlabel="$t$ ($s$)",
    xlim=(-1 / sigma * 5, 1 / sigma * 5),
) as ax:
    ax.plot(w, abs(fw), "o", label="before interpolation")
    ax.plot(w_post, np.real(fw_interp), "x", label="after interpolation (real)")
    ax.plot(w_post, np.imag(fw_interp), "x", label="after interpolation (imaginary)")
    #     ax.plot(w_post, abs(fw_interp), 'x', label="after interpolation (absolute)")
    ax.legend()

# show our inverse Fourier transform
with Plotter(
    super_title="Our Inverse Fourier Transform",
    xlabel="$\omega$ ($2\pi\cdot Hz$)",
    xlim=(-sigma * 5, sigma * 5),
    x_scale="linear",
) as ax:
    ax.plot(t_post, np.real(ft_post), label="real part")
    ax.plot(t_post, np.imag(ft_post), label="imaginary part")
    #     ax.plot(r, abs(fr), label="absolute")
    ax.legend()

# Comparing round-trip
with Plotter(
    super_title="Round-Trip Fourier Transform\n Compared to Original",
    xlabel="$t$ ($s$)",
    ylabel="$f(t)$ (Normalized)",
    xlim=(-sigma * 5, sigma * 5),
    x_scale="linear",
) as ax:
    ax.plot(t, ft, label="Original Gaussian")
    ax.plot(t_post, np.real(ft_post), label="Round-Trip FT Gaussian (real)")
    ax.plot(t_post, np.imag(ft_post), label="Round-Trip FT Gaussian (imaginary)")
    ax.legend()

# %% [markdown]
# Testing Gaussian interpolation/extrapolation on Gaussian:

# %%
x_lims = Limits(20, 600)
y_lims = Limits(1, 4)

gauss_interp = extrapolate_over_noise(
    t,
    ft,
    #     x_interp=np.array(range(1000)),
    n_bins=2**17,
    x_lims=x_lims,
    y_lims=y_lims,
    n_robust=2,
    interp_type="gaussian",
)

# Comparing interp/extrap with original
with Plotter(
    super_title="Gaussian interp/extrap testing",
    xlabel="$t$ ($s$)",
    ylabel="$f(t)$ (Normalized)",
    xlim=(-sigma * 5, sigma * 5),
) as ax:
    ax.plot(t, ft, "o", label="Original Gaussian")
    ax.plot(
        gauss_interp.x_interp, gauss_interp.y_interp, ".", markersize="4", label="interp/extrap"
    )
    ax.legend()

# %% [markdown]
# Extrapolating over noisy parts of signal and afterpulsing:

# %%
lag_s = lag * 1e-3  # ms to seconds

n_bins = 2**17
n_robust = 10
x_lims = Limits(1e-7, 1e-3)  # (100 ns to 1 ms)
y_lims = Limits(1e1, np.inf)
interp_type = "gaussian"
extrap_x_lims = Limits(1e-6, 5e-3)

gauss_interp_ap_signal = extrapolate_over_noise(
    lag_s,
    G_ap_signal_t[1:],
    n_bins=n_bins,
    x_lims=x_lims,
    y_lims=y_lims,
    n_robust=n_robust,
    interp_type=interp_type,
    extrap_x_lims=extrap_x_lims,
)

gauss_interp_ap = extrapolate_over_noise(
    lag_s,
    G_ap_t[1:],
    n_bins=n_bins,
    x_lims=x_lims,
    y_lims=y_lims,
    n_robust=n_robust,
    interp_type=interp_type,
    extrap_x_lims=extrap_x_lims,
)

# Comparing interp/extrap with original
with Plotter(
    super_title="Gaussian interp/extrap testing",
    xlabel="$t$ ($s$)",
    ylabel="$f(t)$ (Normalized)",
    #     xlim=(-sigma * 5, sigma * 5),
) as ax:
    ax.plot(lag_s, G_ap_signal_t, "o", label="original signal")
    ax.plot(
        gauss_interp_ap_signal.x_interp,
        gauss_interp_ap_signal.y_interp,
        ".",
        markersize="4",
        label="interp/extrap",
    )
    ax.legend()

with Plotter(
    super_title="Gaussian interp/extrap testing",
    xlabel="$t$ ($s$)",
    ylabel="$f(t)$ (Normalized)",
    #     xlim=(-sigma * 5, sigma * 5),
) as ax:
    ax.plot(lag_s, G_ap_t, "o", label="original afterpulsing")
    ax.plot(
        gauss_interp_ap.x_interp,
        gauss_interp_ap.y_interp,
        ".",
        markersize="4",
        label="interp/extrap",
    )
    ax.legend()

# %% [markdown]
# Using the interpolated tails instead:

# %%
G_ap_signal_t
gauss_interp_ap_signal

# %% [markdown]
# Fourier transform of afterpulsed signal:

# %%
t_signal, ft_interp_signal, w_signal, fw_signal = fourier_transform_1d(
    lag_s,
    G_ap_signal_t,
    bin_size=1e-7,  # meaning 100 ns
    should_normalize=True,
)

with Plotter(
    super_title="Interpolation of $G_{signal}$",
    xlabel="$t$ ($\mu s$)",
    ylabel="p",
    xlim=(-1, 1),
    x_scale="linear",
) as ax:
    ax.plot(lag_s * 1e6, G_ap_signal_t * 1e-7, "o", label="before interpolation")
    ax.plot(t_signal * 1e6, ft_interp_signal, "x", label="after interpolation")
    ax.legend()

with Plotter(
    super_title="Fourier Transform of $G_{ap\ signal}$",
    xlabel="$\omega$ ($2\pi\cdot MHz$)",
    xlim=(-0.25, 0.25),
) as ax:

    ax.plot(w_signal * 1e-6, np.real(fw_signal), label="real part")
    ax.plot(w_signal * 1e-6, np.imag(fw_signal), label="imaginary part")
    ax.plot(w_signal * 1e-6, abs(fw_signal), label="absolute")
    ax.legend()

# %% [markdown]
# Fourier transform of afterpulsing:

# %%
# # %debug
t_ap, ft_interp_ap, w_ap, fw_ap = fourier_transform_1d(
    lag_s,
    G_ap_t,
    bin_size=1e-7,  # meaning 100 ns
    should_normalize=True,
)

with Plotter(
    super_title="Interpolation of $G_{ap}$",
    xlabel="$t$ ($\mu s$)",
    ylabel="p",
    xlim=(-1, 1),
    x_scale="linear",
) as ax:
    ax.plot(lag_s * 1e6, G_ap_t * 1e-7, "o", label="before interpolation")
    ax.plot(t_ap * 1e6, ft_interp_ap, "x", label="after interpolation")
    ax.legend()

with Plotter(
    super_title="Fourier Transform of $G_{ap}$",
    xlabel="$\omega$ ($2\pi\cdot MHz$)",
    xlim=(-0.3, 0.3),
) as ax:

    ax.plot(w_ap * 1e-6, np.real(fw_ap), label="real part")
    ax.plot(w_ap * 1e-6, np.imag(fw_ap), label="imaginary part")
    ax.plot(w_ap * 1e-6, abs(fw_ap), label="absolute")
    ax.legend()

# %%
with Plotter() as ax:
    ax.plot(w_signal * 1e-6, np.real(fw_signal), label="signal")
    ax.plot(w_ap * 1e-6, np.real(fw_ap), label="ap")

# %% [markdown]
# Dividing the Fourier transform of the afterpulsed signal by that of the afterpulse (from cross-correlation)

# %%
quotient = abs(fw_signal) / abs(fw_ap)

with Plotter() as ax:
    ax.plot(w_ap, quotient)

# %% [markdown]
# Attempting to inverse-transform the quotient to get the afterpulse-free ACF:

# %%
# # %debug

w_min = 0.00015  # 0.015*1e6*2*np.pi

w_quotient, fw_interp_quotient, t_quotient, ft_quotient = fourier_transform_1d(
    #     w_ap[w_ap < w_min],
    #     quotient[w_ap < w_min],
    w_ap,
    quotient,
    should_inverse=True,
    is_input_symmetric=True,
    bin_size=np.diff(w_ap)[0]
    #     should_normalize=True,
)

with Plotter(
    super_title="Interpolation",
    xlabel="$\omega$ ($2\pi\cdot MHz$)",
    xlim=(-1, 1),
) as ax:
    ax.plot(w_ap * 1e-6, quotient, "o", label="before interpolation")
    ax.plot(w_quotient * 1e-6, abs(fw_interp_quotient), "x", label="after interpolation")
    ax.legend()

with Plotter(
    super_title="Inverse Fourier Transform\n of $G(\omega)_{signal}\ /\ G(\omega)_{ap}$",
    xlabel="$t$ ($\mu s$)",
    xlim=(-1, 1),
) as ax:

    ax.plot(t_quotient * 1e6, np.real(ft_quotient), label="real part")
    ax.plot(t_quotient * 1e6, np.imag(ft_quotient), label="imaginary part")
    ax.plot(t_quotient * 1e6, abs(ft_quotient), label="absolute")
    ax.legend()

# %% [markdown]
# Generating the Fourier transform of the afterpulse-subtracted (in the usual/old way) signal, to compare with the quotient in Fourier space:

# %%
# load experiment
exp = SFCSExperiment(name=label)
exp.load_experiment(
    confocal_template=DATA_PATH / confocal_template,
    should_plot=True,
    should_use_preprocessed=True,  # TODO: load anew if not found
    should_re_correlate=True,
    should_subtract_afterpulse=True,
)

# %%
G_signal_t = np.copy(exp.confocal.cf["confocal"].avg_cf_cr)

t_std_signal, ft_interp_std_signal, w_std_signal, fw_std_signal = fourier_transform_1d(
    lag_s[1:],
    G_signal_t[1:],
    bin_size=1e-7,  # meaning 100 ns
    should_normalize=True,
)

with Plotter(
    super_title="Interpolation of $G_{signal}$",
    xlabel="$t$ ($\mu s$)",
    ylabel="p",
    xlim=(-1, 1),
    x_scale="linear",
) as ax:
    ax.plot(lag_s * 1e6, G_signal_t * 1e-7, "o", label="before interpolation")
    ax.plot(t_std_signal * 1e6, ft_interp_std_signal, "x", label="after interpolation")
    ax.legend()

with Plotter(
    super_title="Fourier Transform of $G_{signal}$",
    xlabel="$\omega$ ($2\pi\cdot MHz$)",
    xlim=(-0.3, 0.3),
) as ax:

    ax.plot(w_std_signal * 1e-6, np.real(fw_std_signal), label="real part")
    ax.plot(w_std_signal * 1e-6, np.imag(fw_std_signal), label="imaginary part")
    ax.plot(w_std_signal * 1e-6, abs(fw_std_signal), label="absolute")
    ax.legend()

# %% [markdown]
# Plotting the transformed ap-subtracted signal together with the quotient:

# %%
with Plotter(
    super_title="Fourier Transform of $G_{signal}$",
    xlabel="$\omega$ ($2\pi\cdot MHz$)",
    xlim=(-0.3, 0.3),
) as ax:
    ax.plot(w_std_signal * 1e-6, abs(fw_std_signal), label="absolute $G_{signal}")

    ax.plot(w_ap * 1e-6, quotient, label="absolute $G_{signal}")
    ax.legend()

# %% [markdown]
# Play sound when done:

# %%
Beep(4000, 300)
