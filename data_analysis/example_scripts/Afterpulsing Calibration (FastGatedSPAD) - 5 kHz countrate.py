# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# We begin by moving to the project **directory**, loading neccessary **packages and modules**, and **defining constants**:

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
# Choosing the data templates:

# %%
data_labels = [
    "White Noise 5 kHz",
]

data_templates = [
    "halogen5khz_static_exc_152832_*.pkl",
]

file_selections = [
    "Use All",
]

DATA_DATE = "21_06_2022"
DATA_PATH = DATA_ROOT / DATA_DATE / DATA_TYPE

template_paths = [DATA_PATH / tmplt for tmplt in data_templates]

halogen_exp_dict = {label: SFCSExperiment(name=label) for label in data_labels}

label_load_kwargs_dict = {
    label: dict(confocal_template=tmplt_path, file_selection=selection)
    for label, tmplt_path, selection in zip(data_labels, template_paths, file_selections)
}

# %% [markdown]
# Importing all needed data. Processing, correlating and averaging if no pre-processed measurement exist.

# %%
FORCE_PROCESSING = False
# FORCE_PROCESSING = True

used_labels = [
    "White Noise 5 kHz",
]

# load experiment
for label in used_labels:
    halogen_exp_dict[label].load_experiment(
        should_plot=True,
        force_processing=FORCE_PROCESSING,
        should_re_correlate=FORCE_PROCESSING,
        should_subtract_afterpulse=False,
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


# %% [markdown]
# Multiexponent Fitting

# %%
from utilities.fit_tools import FitError

beta_dict = dict()

for label in used_labels:
    meas = halogen_exp_dict[label].confocal

    print(f"Fitting {label}...")

    old_chi_sq_norm = 1e5
    new_chi_sq_norm = 1e5 - 1
    max_nfev = 10000
    eps = 0.5
    p0 = [1, 1]  # start from single decaying exponent (fit is sure to succeed)
    while True:
        old_chi_sq_norm = new_chi_sq_norm
        try:
            meas.cf["confocal"].fit_correlation_function(
                x_field="lag",
                y_field="avg_cf_cr",
                y_error_field="error_cf_cr",
                fit_name="multi_exponent_fit",
                fit_param_estimate=p0,
                fit_range=(1.1e-4, np.inf),
                x_scale="log",
                y_scale="linear",
                should_plot=True,
                bounds=(0, np.inf),
                max_nfev=max_nfev,
            )
        except FitError as exc:  # fit failed
            break
        else:
            new_chi_sq_norm = meas.cf["confocal"].fit_params["multi_exponent_fit"].chi_sq_norm
            if new_chi_sq_norm > old_chi_sq_norm + eps:
                print(f"chi_sq_norm increased {new_chi_sq_norm}, using previous fit as final.")
                break
            if new_chi_sq_norm < 1 - eps:
                print(f"chi_sq_norm dropped below 1, using previous fit as final.")
                new_chi_sq_norm = old_chi_sq_norm
                break
            beta = meas.cf["confocal"].fit_params["multi_exponent_fit"].beta
            print(f"number of parameters: {beta.size}")
            print(f"new_chi_sq_norm: {new_chi_sq_norm}\n")
            p0 += (
                beta.tolist()
            )  # use the found parameters, and add another exponent (TODO: this is not what this code does! Why does it work better this way?)
            if new_chi_sq_norm < 1 + eps:
                print("close enough")
                break

    #     break # TESTESTEST - only fit first measurement

    print(f"number of parameters: {beta.size}")
    print(f"beta: {beta.tolist()}")
    print(f"final_chi_sq_norm: {new_chi_sq_norm}\n")
    beta_dict["confocal"] = beta

    # BEEP WHEN DONE!
    Beep(4000, 1000)

# %% [markdown]
# Save the parameters:

# %%
import pickle

are_you_sure = True
# are_you_sure = False

if are_you_sure:
    with open("beta_dict.pkl", "wb") as f:
        pickle.dump(beta_dict, f)
else:
    print("Are you sure you wish to over-write?")

# %% [markdown]
# Load saved parameters:

# %%
import pickle

with open("beta_dict.pkl", "rb") as f:
    beta_dict = pickle.load(f)

afterpulse_params_list = [("multi_exponent_fit", beta) for beta in beta_dict]
print("Loaded 'beta_dict'.")

# %%
beta_dict

# %% [markdown]
# Load the 300 bp measurements with the correct afterpulsing subtraction:

# %%
data_type = "solution"

data_templates = [
    "atto_EB7V_HO81ns_static_exc_122404_*.pkl",
    "atto_EB5V_HO81ns_static_exc_120745_*.pkl",
    "atto_EB7V_HO51ns_static_exc_125000_*.pkl",
    "atto_EB5V_HO51ns_static_exc_123729_*.pkl",
]

n_measurements = len(data_templates)

data_dates = ["10_03_2022"] * n_measurements

data_labels = [str_.split("_static")[0] for str_ in data_templates]

data_file_selections = [""] * n_measurements

# %%
meas_dict = {label: SolutionSFCSMeasurement() for label in data_labels}
dir_paths = [DATA_ROOT / data_date / data_type for data_date in data_dates]

# Set to 'False' to force processing
should_use_processed = False

# Try to load pre-processed measurements. If no pre-processed measurement exist, process the data
for dir_path, file_template, label, file_selection, ap_params in zip(
    dir_paths, data_templates, data_labels, data_file_selections, afterpulse_params_list
):
    try:
        if should_use_processed:
            # load pre-processed
            file_path = dir_path / "processed" / re.sub("_[*]", "", file_template)
            meas_dict[label] = load_processed_solution_measurement(file_path)
            print(f"Loaded pre-processed '{label}'")
        else:
            raise FileNotFoundError

    except (FileNotFoundError, OSError):
        # or fully process data
        file_template = dir_path / file_template
        meas_dict[label].read_fpga_data(
            file_template, file_selection=file_selection, should_parallel_process=False
        )
        meas_dict[label].correlate_and_average(
            cf_name=label,
            afterpulse_params=ap_params,
            should_subtract_afterpulse=True,
            is_verbose=True,
            min_time_frac=0.1,
            norm_range=(2e-3, 4e-3),
        )

        # save processed data (to avoid processing) - raw data is saved in temporary folder and might get deleted!
        save_processed_solution_meas(meas_dict[label], dir_path)


# %% [markdown]
# Plot mean ACFs of all uncorrelated measurements together with current afterpulse and the old one:

# %%
from utilities import display
from cycler import cycler

meas = list(meas_dict.values())[0]
lag = list(meas.cf.values())[0].lag

with Plotter(xlim=(1.5e-4, 1e1), ylim=(-0.25, 4e4), super_title="Mean ACFs Comparison") as ax:
    ax.set_xlabel("Lag ($ms$)")
    ax.set_ylabel("G0")
    ax.set_prop_cycle(cycler("color", ["tab:blue", "tab:orange", "tab:green", "tab:red"]))
    [
        ax.plot(meas_dict[label].cf[label].lag, meas_dict[label].cf[label].avg_cf_cr)
        for label in data_labels
    ]
    [
        ax.plot(meas_dict[label].cf[label].lag, meas_dict[label].cf[label].afterpulse, "--")
        for label in data_labels
    ]
    ax.legend(
        data_labels + [label.replace("atto", "halogen") for label in data_labels], fontsize=14
    )
    ax.set_xscale("log")

# Show stats
[
    print(
        f"{label} - G0: {meas_dict[label].cf[label].g0/1e3:.2f}k, Count Rate: {meas_dict[label].avg_cnt_rate_khz:.1f} kHz\n"
    )
    for label in data_labels
]

# %% [markdown]
# Normalized:

# %%
from utilities import display
from cycler import cycler

meas = list(meas_dict.values())[0]
lag = list(meas.cf.values())[0].lag

with Plotter(
    xlim=(1.5e-4, 1e1), ylim=(-0.1, 2), super_title="Mean ACFs Comparison (Normalized)"
) as ax:
    ax.set_xlabel("Lag ($ms$)")
    ax.set_ylabel("G0")
    ax.set_prop_cycle(cycler("color", ["tab:blue", "tab:orange", "tab:green", "tab:red"]))
    [
        ax.plot(meas_dict[label].cf[label].lag, meas_dict[label].cf[label].normalized)
        for label in data_labels
    ]
    ax.legend(
        data_labels + [label.replace("atto", "halogen") for label in data_labels], fontsize=14
    )
    ax.set_xscale("log")

# Show stats
[
    print(
        f"{label} - G0: {meas_dict[label].cf[label].g0/1e3:.2f}k, Count Rate: {meas_dict[label].avg_cnt_rate_khz:.1f} kHz\n"
    )
    for label in data_labels
]
