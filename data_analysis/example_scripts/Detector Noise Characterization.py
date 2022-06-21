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
def fit_decaying_cosine(
    x,
    y,
    y_errors=None,
    fit_param_estimate=[1] * 5,
    fit_range=(0, np.inf),
    x_scale="log",
    y_scale="linear",
    should_plot=False,
    **kwargs,
) -> FitParams:
    """Doc."""

    return curve_fit_lims(
        "decaying_cosine_fit",
        fit_param_estimate,
        xs=x,
        ys=y,
        ys_errors=y_errors,
        x_limits=Limits(fit_range),
        should_plot=should_plot,
        x_scale=x_scale,
        y_scale=y_scale,
        **kwargs,
    )


# %% [markdown]
# Loading the white-noise (Halogen) data:

# %%
data_labels = [
    "White Noise 5 kHz",
    "White Noise 150 kHz",
    "White Noise 180 kHz",
    "White Noise 210 kHz",
    "White Noise 305 kHz",
]

n_meas = len(data_labels)

data_templates = [
    "halogen5khz_static_exc_152832_*.pkl",
    "halogen150khz_static_exc_133749_*.pkl",
    "halogen180khz_static_exc_140956_*.pkl",
    "halogen210khz_static_exc_144154_*.pkl",
    "halogen300khz_static_exc_150947_*.pkl",
]

file_selections = ["Use All"] * n_meas

force_processing = {
    "White Noise 5 kHz": True,
    "White Noise 150 kHz": False,
    "White Noise 180 kHz": False,
    "White Noise 210 kHz": False,
    "White Noise 305 kHz": False,
}

DATA_DATE = "21_06_2022"
DATA_PATH = DATA_ROOT / DATA_DATE / DATA_TYPE

template_paths = [DATA_PATH / tmplt for tmplt in data_templates]

halogen_exp_dict = {label: SFCSExperiment(name=label) for label in data_labels}

label_load_kwargs_dict = {
    label: dict(confocal_template=tmplt_path, file_selection=selection)
    for label, tmplt_path, selection in zip(data_labels, template_paths, file_selections)
}

# %%
FORCE_PROCESSING = False
# FORCE_PROCESSING = True

used_labels = [
    "White Noise 5 kHz",
    "White Noise 150 kHz",
    "White Noise 180 kHz",
    "White Noise 210 kHz",
    "White Noise 305 kHz",
]

# load experiment
for label in used_labels:
    halogen_exp_dict[label].load_experiment(
        should_plot=True,
        force_processing=force_processing[label],
        should_re_correlate=FORCE_PROCESSING,
        should_subtract_afterpulse=False,
        should_unite_start_times=True,  # TESTESTEST
        **label_load_kwargs_dict[label],
    )

    #     # calibrate TDC
    #     exp.calibrate_tdc(should_plot=True, force_processing=FORCE_PROCESSING)

    # save processed data (to avoid re-processing)
    if FORCE_PROCESSING:
        halogen_exp_dict[label].save_processed_measurements()

for label in used_labels:
    meas = halogen_exp_dict[label].confocal
    print(f"'{label}' countrate: {meas.avg_cnt_rate_khz:.2f} +/- {meas.std_cnt_rate_khz:.2f}")

# %%
# halogen_exp_dict['White Noise 150 kHz'].save_processed_measurements()

# %% [markdown]
# Plot all:

# %%
with Plotter(x_scale="log", xlim=(1e-4, 1e0), ylim=(-1, 3e4)) as ax:
    for label in used_labels:
        cf = halogen_exp_dict[label].confocal.cf["confocal"]
        ax.plot(cf.lag, cf.avg_cf_cr, label=label)
        ax.legend()

# %% [markdown]
# Leave only the oscillating noise by dividing (corrfunc + 1) by the calibrated afterpusing + 1 (TODO: Add the derivation).
#
# The factor 'f' is needed (persumebly) because the afterpulsing was actually calibrated for different detector parameters (hold-off, voltage bias...)

# %%
plotted_labels = [
    "White Noise 5 kHz",
    "White Noise 150 kHz",
    "White Noise 180 kHz",
    "White Noise 210 kHz",
    "White Noise 305 kHz",
]

# f = 0.85
f = 1

with Plotter(super_title="Oscillations", x_scale="log", ylim=(-1e-2, 1e-2)) as ax:
    for label in plotted_labels:
        cf = halogen_exp_dict[label].confocal.cf["confocal"]
        wn_lag = cf.lag
        G_AA = cf.avg_corrfunc
        G_ap = calculate_afterpulse(wn_lag) / cf.countrate * f
        noise = (G_AA + 1) / (G_ap + 1) - 1

        ax.plot(wn_lag, G_AA, label=f"G_AA ({label})")
        ax.plot(wn_lag, G_ap, label=f"G_ap  ({label})")
        ax.plot(wn_lag, noise, label=f"quotient ({label})")
        ax.legend()

# %% [markdown]
# Same as above, using newly calibrated afterpulse parameters (fit to the 5 kHz measurement):

# %%
with open("beta_dict.pkl", "rb") as f:
    beta_dict = pickle.load(f)
afterpulse_params = ("multi_exponent_fit", list(beta_dict.values())[0])

plotted_labels = [
    "White Noise 5 kHz",
    "White Noise 150 kHz",
    "White Noise 180 kHz",
    "White Noise 210 kHz",
    "White Noise 305 kHz",
]

f = 0.85
# f = 1

with Plotter(super_title="Oscillations", x_scale="log", ylim=(-1e-2, 1e-2)) as ax:
    for label, exp in halogen_exp_dict.items():
        if label in plotted_labels:
            cf = exp.confocal.cf["confocal"]
            wn_lag = cf.lag
            G_AA = cf.avg_corrfunc
            G_ap = calculate_afterpulse(wn_lag, afterpulse_params) / cf.countrate * f
            noise = (G_AA + 1) / (G_ap + 1) - 1

            ax.plot(wn_lag, G_AA, label=f"G_AA ({label})")
            ax.plot(wn_lag, G_ap, label=f"G_ap  ({label})")
            ax.plot(wn_lag, noise, label=f"quotient ({label})")
            ax.legend()

# %% [markdown]
# Same with cf_cr (attempt):

# %%
plotted_labels = [
    "White Noise 5 kHz",
    "White Noise 150 kHz",
    "White Noise 180 kHz",
    "White Noise 210 kHz",
    "White Noise 305 kHz",
]

# f = 0.85
f = 1

with Plotter(super_title="Oscillations", x_scale="log", xlim=(1e-3, 1e1), ylim=(-4e3, 4e3)) as ax:
    for label, exp in halogen_exp_dict.items():
        if label in plotted_labels:
            cf = exp.confocal.cf["confocal"]
            wn_lag = cf.lag
            CF_CR = cf.avg_cf_cr
            G_ap = calculate_afterpulse(wn_lag, afterpulse_params) / cf.countrate * f
            noise = (CF_CR + cf.countrate) / (G_ap + 1) - cf.countrate

            ax.plot(wn_lag, CF_CR, label=f"CF_CR ({label})")
            #             ax.plot(wn_lag, G_ap, label=f"G_ap * CR  ({label})")
            ax.plot(wn_lag, noise, label=f"noise ({label})")
            ax.legend()

# %% [markdown]
# Attempt to fit a decaying oscillating function to the detector noise:

# %%
fit_params = fit_decaying_cosine(
    wn_lag,
    noise,
    fit_param_estimate=[3e-3, 1 / (3e-2), 0, 1e-1, 0],
    #     fit_param_estimate=[1]*4,
    fit_range=(22e-3, 2.5e-1),
    should_plot=True,
    bounds=([-1, -1e3, -2 * np.pi, 0, -1], [1, 1e3, 2 * np.pi, 1e3, 1]),
    maxfev=1000000,
)

param_names = ["Amplitude", "Angular Frequency", "Phase", "Decay Time", "Y-shift"]
fit_params_dict = {
    param_name: param_val for param_name, param_val in zip(param_names, fit_params.beta)
}
for name, val in fit_params_dict.items():
    print(f"{name}: {val:e}")

print(f"\nFrequency: {fit_params.beta[1] / 2*np.pi * 1e-1:.2f} kHz")
print(f"Period: {2*np.pi / fit_params.beta[1] * 1e3:.2f} us")
print(f"Decay time: {fit_params.beta[3] * 1e3:.2f} us")

# %% [markdown]
# Play sound when done:

# %%
delta_t = 175
freq_range = (3000, 7000)
n_beeps = 5
[Beep(int(f), delta_t) for f in np.linspace(*freq_range, n_beeps)]
