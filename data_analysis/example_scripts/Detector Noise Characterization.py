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
# # Attempting to find characteristics of noise oscillations

# %% [markdown]
# - Import core and 3rd party modules
# - Move current working directory to project root (if needed) and import project modules
# - Set data paths and other constants

# %%
######################################
# importing core and 3rd-party modules
######################################

import os
import sys
import pickle
import re
import scipy
from pathlib import Path
from winsound import Beep
from contextlib import suppress
from copy import deepcopy
from types import SimpleNamespace

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

from data_analysis.correlation_function import (
    CorrFunc,
    SFCSExperiment,
    SolutionSFCSMeasurement,
    calculate_afterpulse,
)
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
from utilities.fit_tools import curve_fit_lims, FitParams

#################################################
# Setting up data path and other global constants
#################################################

EPS = sys.float_info.epsilon

DATA_ROOT = Path("D:\OneDrive - post.bgu.ac.il\gSTED_sFCS_Data")  # Laptop/Lab PC (same path)
# DATA_ROOT = Path("D:\???")  # Oleg's
DATA_TYPE = "solution"

SHOULD_PLOT = True


# %% [markdown]
# Defining a fitting function for the oscillating temporal noise:

# %%
def decaying_cosine_fit(t, a, omega, phi, tau, c):
    return a * np.cos(omega * t + phi) * np.exp(-t / tau) + c


def fit_decaying_cosine(
    x,
    y,
    fit_range,
    y_errors=None,
    y_limits=(np.NINF, np.inf),
    fit_param_estimate=None,
    should_plot=False,
    bounds=None,
    **kwargs,
) -> FitParams:
    """Doc."""

    if fit_param_estimate is None or bounds is None:
        # Guess initial params
        in_lims_idxs = Limits(fit_range).valid_indices(x)
        y_limited = y[in_lims_idxs]
        x_limited = x[in_lims_idxs]
        omega = (
            2
            * np.pi
            / abs(x_limited[np.argmax(y_limited)] - x_limited[np.argmin(y_limited)])
            / np.pi
        )
        #         omega = 2*np.pi / (0.025)
        phi = 0
        tau = (max(x_limited) - min(x_limited)) / 2
        y_shift = np.mean(y_limited)
        amplitude = abs(y_limited[np.argmax(abs(y_limited))] - y_shift)

        # set initial parameters
        if fit_param_estimate is None:
            fit_param_estimate = [
                amplitude,
                omega,
                phi,
                tau,
                y_shift,
            ]

        # set bounds for parameters
        if bounds is None:
            bounds = (
                [amplitude / 3, omega / 3, -2 * np.pi, 0, y_shift - np.std(y) * 2],
                [amplitude * 3, omega * 3, +2 * np.pi, tau * 10, y_shift + np.std(y) * 2],
            )

    return curve_fit_lims(
        decaying_cosine_fit,
        fit_param_estimate,
        xs=x,
        ys=y,
        ys_errors=y_errors,
        x_limits=Limits(fit_range),
        y_limits=Limits(y_limits),
        should_plot=should_plot,
        bounds=bounds,
        **kwargs,
    )


# %% [markdown]
# Loading the white-noise (Halogen) data:

# %%
data_label_dict = {
    "White Noise 4 kHz": SimpleNamespace(
        date="28_06_2022",
        template="halogen4khz_static_exc_162446_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 5 kHz": SimpleNamespace(
        date="21_06_2022",
        template="halogen5khz_static_exc_152832_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 6 kHz": SimpleNamespace(
        date="28_06_2022",
        template="halogen6khz_static_exc_164636_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 6.5 kHz": SimpleNamespace(
        date="28_06_2022",
        template="halogen6_5khz_static_exc_171231_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 7 kHz": SimpleNamespace(
        date="28_06_2022",
        template="halogen7khz_static_exc_153022_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 10 kHz": SimpleNamespace(
        date="28_06_2022",
        template="halogen10khz_static_exc_144456_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 20 kHz": SimpleNamespace(
        date="28_06_2022",
        template="halogen20khz_static_exc_135859_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 30 kHz": SimpleNamespace(
        date="27_06_2022",
        template="halogen30khz_static_exc_153856_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 60 kHz": SimpleNamespace(
        date="27_06_2022",
        template="halogen60khz_static_exc_132729_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 90 kHz": SimpleNamespace(
        date="27_06_2022",
        template="halogen90khz_static_exc_123654_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 120 kHz": SimpleNamespace(
        date="27_06_2022",
        template="halogen120khz_static_exc_115748_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 150 kHz": SimpleNamespace(
        date="21_06_2022",
        template="halogen150khz_static_exc_133749_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 180 kHz": SimpleNamespace(
        date="21_06_2022",
        template="halogen180khz_static_exc_140956_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 210 kHz": SimpleNamespace(
        date="21_06_2022",
        template="halogen210khz_static_exc_144154_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 305 kHz": SimpleNamespace(
        date="21_06_2022",
        template="halogen300khz_static_exc_150947_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
}

data_labels = list(data_label_dict.keys())

n_meas = len(data_labels)

template_paths = [
    DATA_ROOT / data.date / DATA_TYPE / data.template for data in data_label_dict.values()
]

# initialize the experiment dictionary, if it doens't already exist
if "halogen_exp_dict" not in locals():
    print("resetting 'halogen_exp_dict'")
    halogen_exp_dict = {}

for label in data_labels:
    if label not in halogen_exp_dict:
        halogen_exp_dict[label] = SFCSExperiment(name=label)

label_load_kwargs_dict = {
    label: dict(confocal_template=tmplt_path, file_selection=data.file_selection)
    for label, tmplt_path, data in zip(data_labels, template_paths, data_label_dict.values())
}

# TEST - use pprint?
print(template_paths)

# %%
FORCE_PROCESSING = False
# FORCE_PROCESSING = True

# used_labels = data_labels
used_labels = [
    "White Noise 5 kHz",
    "White Noise 30 kHz",
    "White Noise 150 kHz",
    "White Noise 305 kHz",
]

# load experiment
for label in used_labels:
    # skip already loaded experiments, unless forced
    if not hasattr(halogen_exp_dict[label], "confocal") or data_label_dict[label].force_processing:
        halogen_exp_dict[label].load_experiment(
            should_plot=False,
            force_processing=data_label_dict[label].force_processing,
            should_re_correlate=FORCE_PROCESSING,
            should_subtract_afterpulse=False,
            should_unite_start_times=True,  # for uniting the two 5 kHz measurements
            **label_load_kwargs_dict[label],
        )

        #     # calibrate TDC
        #     exp.calibrate_tdc(should_plot=True, force_processing=FORCE_PROCESSING)

        # save processed data (to avoid re-processing)
        halogen_exp_dict[label].save_processed_measurements(
            should_force=data_label_dict[label].force_processing
        )

# Present count-rates
for label in used_labels:
    meas = halogen_exp_dict[label].confocal
    print(f"'{label}' countrate: {meas.avg_cnt_rate_khz:.2f} +/- {meas.std_cnt_rate_khz:.2f}")

# %% [markdown]
# Plot all:

# %%
with Plotter(x_scale="log", xlim=(1e-4, 1e0), ylim=(-1, 3e4)) as ax:
    for label in used_labels:
        cf = halogen_exp_dict[label].confocal.cf["confocal"]
        ax.plot(cf.lag, cf.avg_cf_cr, label=label)
        ax.legend()

# %% [markdown]
# Leave only the oscillating noise by dividing (cf_cr + CR) by the calibrated afterpusing + 1 (TODO: Add the derivation).

# %%
# c = 1
# labels_factors_dict = {
#     "White Noise 5 kHz": 1,
#     "White Noise 30 kHz": 1,
#     "White Noise 60 kHz": 1,
#     "White Noise 90 kHz": 1,
#     "White Noise 120 kHz": 1,
#     "White Noise 150 kHz": 0.9187 * c,
#     "White Noise 180 kHz": 0.9035 * c,
#     "White Noise 210 kHz": 0.892 * c,
#     "White Noise 305 kHz": 0.86 * c,
# }

labels_factors_dict = {
    #     "White Noise 4 kHz": 1,
    "White Noise 5 kHz": 1,
    #     "White Noise 6 kHz": 1,
    #     "White Noise 6.5 kHz": 1,
    #     "White Noise 7 kHz": 1,
    #     "White Noise 10 kHz": 1,
    #     "White Noise 20 kHz": 1,
    "White Noise 30 kHz": 1,
    #     "White Noise 60 kHz": 1,
    #     "White Noise 90 kHz": 1,
    #     "White Noise 120 kHz": 1,
    "White Noise 150 kHz": 1,
    #     "White Noise 180 kHz": 1,
    #     "White Noise 210 kHz": 1,
    "White Noise 305 kHz": 1,
}

label_G_noise_dict = {}
label_CF_CR_noise_dict = {}
for label, f in labels_factors_dict.items():
    cf = halogen_exp_dict[label].confocal.cf["confocal"]
    wn_lag = cf.lag
    G = cf.avg_corrfunc
    CF_CR = cf.avg_cf_cr
    G_ap = calculate_afterpulse(wn_lag) / cf.countrate * f
    label_G_noise_dict[label] = SimpleNamespace(
        lag=wn_lag, noise=(G + 1) / (G_ap + 1) - 1, G=G, countrate=cf.countrate
    )
    label_CF_CR_noise_dict[label] = SimpleNamespace(
        lag=wn_lag,
        noise=(CF_CR + cf.countrate) / (G_ap + 1) - cf.countrate,
        CF_CR=CF_CR,
        countrate=cf.countrate,
    )

# Plotting
with Plotter(
    super_title="Oscillations CF_CR", x_scale="log", xlim=(1e-3, 1e0), ylim=(-4e3, 4e3)
) as ax:
    for label in labels_factors_dict.keys():
        #         ax.plot(
        #             label_CF_CR_noise_dict[label].lag,
        #             label_CF_CR_noise_dict[label].CF_CR,
        #             label=f"G_AA ({label})",
        #         )
        #             ax.plot(wn_lag, G_ap, label=f"G_ap  ({label})")
        ax.plot(
            label_CF_CR_noise_dict[label].lag,
            label_CF_CR_noise_dict[label].noise,
            label=f"quotient ({label})",
        )
        ax.legend()

with Plotter(
    super_title="Oscillations G (corrfunc)", x_scale="log", xlim=(1e-3, 1e0), ylim=(-0.01, 0.03)
) as ax:
    for label in labels_factors_dict.keys():
        #         ax.plot(label_G_noise_dict[label].lag, label_G_noise_dict[label].G, label=f"G_AA ({label})")
        #             ax.plot(wn_lag, G_ap, label=f"G_ap  ({label})")
        ax.plot(
            label_G_noise_dict[label].lag,
            label_G_noise_dict[label].noise,
            label=f"quotient ({label})",
        )
        ax.legend()

# %% [markdown]
# Attempt to smooth:

# %%
# def smooth(x,window_len=11,window='hanning'):
#     """smooth the data using a window with requested size.

#     This method is based on the convolution of a scaled window with the signal.
#     The signal is prepared by introducing reflected copies of the signal
#     (with the window size) in both ends so that transient parts are minimized
#     in the begining and end part of the output signal.

#     input:
#         x: the input signal
#         window_len: the dimension of the smoothing window; should be an odd integer
#         window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
#             flat window will produce a moving average smoothing.

#     output:
#         the smoothed signal

#     example:

#     t=linspace(-2,2,0.1)
#     x=sin(t)+randn(len(t))*0.1
#     y=smooth(x)

#     see also:

#     numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
#     scipy.signal.lfilter

#     TODO: the window parameter could be the window itself if an array instead of a string
#     NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
#     """

#     if x.ndim != 1:
#         raise ValueError, "smooth only accepts 1 dimension arrays."

#     if x.size < window_len:
#         raise ValueError, "Input vector needs to be bigger than window size."


#     if window_len<3:
#         return x


#     if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#         raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


#     s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
#     #print(len(s))
#     if window == 'flat': #moving average
#         w=numpy.ones(window_len,'d')
#     else:
#         w=eval('numpy.'+window+'(window_len)')

#     y=numpy.convolve(w/w.sum(),s,mode='valid')
#     return y

# %% [markdown]
# Attempt to fit a decaying oscillating function to the detector noise:

# %%
fitted_labels = data_labels
# [
#     "White Noise 150 kHz",
#     "White Noise 180 kHz",
#     "White Noise 210 kHz",
#     "White Noise 305 kHz",
# ]

label_amplitude_dict = {}
for label, vals in label_G_noise_dict.items():
    if label in fitted_labels:
        fit_params = fit_decaying_cosine(
            vals.lag,
            vals.noise,
            fit_range=(2e-2, 2e-1),
            #             y_limits=(0, np.inf),
            should_plot=True,
            plot_kwargs=dict(x_scale="linear"),
            maxfev=1000000,
        )

        param_names = ["Amplitude", "Angular Frequency", "Phase", "Decay Time", "Y-shift"]
        fit_params_dict = {
            param_name: (param_val, param_error)
            for param_name, param_val, param_error in zip(
                param_names, fit_params.beta, fit_params.beta_error
            )
        }

        print(f"Amplitude: {fit_params.beta[0]:.2e}")
        print(f"Phase: {fit_params.beta[2]:.2f}")
        print(f"Frequency: {fit_params.beta[1] / 2*np.pi * 1e-1:.2f} kHz")
        print(f"Period: {2*np.pi / fit_params.beta[1] * 1e3:.2f} us")
        print(f"Decay time: {fit_params.beta[3] * 1e3:.2f} us")
        print("")

        label_amplitude_dict[label] = fit_params.beta[0]

amp_to_countrate_ratios = [
    label_amplitude_dict[label] / label_CF_CR_noise_dict[label].countrate for label in fitted_labels
]
coeffs = [labels_factors_dict[label] for label in fitted_labels]

with Plotter(super_title="Amplitude/Countrate vs. AP factor") as ax:
    ax.plot(coeffs, amp_to_countrate_ratios, "o-")

# %% [markdown]
# Play sound when done:

# %%
delta_t = 175
freq_range = (3000, 7000)
n_beeps = 5
[Beep(int(f), delta_t) for f in np.linspace(*freq_range, n_beeps)]
