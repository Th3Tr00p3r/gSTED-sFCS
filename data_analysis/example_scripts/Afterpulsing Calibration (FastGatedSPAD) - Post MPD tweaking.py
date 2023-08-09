# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
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
from pathlib import Path
from winsound import Beep
from copy import deepcopy
from types import SimpleNamespace

import matplotlib as mpl

mpl.use("nbAgg")
import numpy as np
from IPython.core.debugger import set_trace

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
    SolutionSFCSExperiment,
    calculate_calibrated_afterpulse,
)
from utilities.display import Plotter

#################################################
# Setting up data path and other global constants
#################################################

DATA_ROOT = Path("D:\OneDrive - post.bgu.ac.il\gSTED_sFCS_Data")  # Laptop/Lab PC (same path)
# DATA_ROOT = Path("D:\???")  # Oleg's
DATA_TYPE = "solution"

# SHOULD_SAVE = True
SHOULD_SAVE = False


def save_figure(fig, fig_path):
    if SHOULD_SAVE:
        fig.savefig(fig_path)
        print(f"Figure saved at: '{fig_path}'")
    else:
        print(f"Figure was NOT saved! 'SHOULD_SAVE' is False!")


# %% [markdown]
# Choosing the data templates:

# %%
data_label_dict = {
    "White Noise 5 kHz (new, after fix)": SimpleNamespace(
        date="17_08_2022",
        template="halogen5khz_static_exc_132750_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 60 kHz  (new, after fix)": SimpleNamespace(
        date="17_08_2022",
        template="halogen60khz_static_exc_145502_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 300 kHz  (new, after fix)": SimpleNamespace(
        date="17_08_2022",
        template="halogen300khz_static_exc_152832_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    #     "White Noise 5 kHz (before fix)": SimpleNamespace(
    #         date="21_06_2022",
    #         template="halogen5khz_static_exc_152832_*.pkl",
    #         file_selection="Use All",
    #         force_processing=True,
    #     ),
    #     "White Noise 60 kHz (before fix)": SimpleNamespace(
    #         date="27_06_2022",
    #         template="halogen60khz_static_exc_132729_*.pkl",
    #         file_selection="Use All",
    #         force_processing=True,
    #     ),
    #     "White Noise 305 kHz (before fix)": SimpleNamespace(
    #         date="21_06_2022",
    #         template="halogen300khz_static_exc_150947_*.pkl",
    #         file_selection="Use All",
    #         force_processing=True,
    #     ),
    "White Noise 4 kHz (old detector)": SimpleNamespace(
        date="08_11_2021",
        template="halogen_4kHz_static_exc_112742_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 40 kHz  (old detector)": SimpleNamespace(
        date="08_11_2021",
        template="halogen_40kHz_static_exc_121330_*.pkl",
        file_selection="Use All",
        force_processing=False,
    ),
    "White Noise 300 kHz  (old detector)": SimpleNamespace(
        date="08_11_2021",
        template="halogen_300kHz_static_exc_125927_*.pkl",
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
        halogen_exp_dict[label] = SolutionSFCSExperiment(name=label)

label_load_kwargs_dict = {
    label: dict(confocal_template=tmplt_path, file_selection=data.file_selection)
    for label, tmplt_path, data in zip(data_labels, template_paths, data_label_dict.values())
}

# TEST
print(template_paths)

# %% [markdown]
# Importing all needed data. Processing, correlating and averaging if no pre-processed measurement exist.

# %%
# FORCE_ALL = True
FORCE_ALL = False

used_labels = data_labels
# used_labels = [
#     "White Noise 5 kHz",
#     "White Noise 60 kHz",
#     "White Noise 300 kHz",
# ]

# load experiment
for label in used_labels:
    # skip already loaded experiments, unless forced
    if not hasattr(halogen_exp_dict[label], "confocal") or data_label_dict[label].force_processing:
        halogen_exp_dict[label].load_experiment(
            should_plot=False,
            force_processing=data_label_dict[label].force_processing or FORCE_ALL,
            should_re_correlate=FORCE_ALL,
            afterpulsing_method="none",
            should_unite_start_times=True,  # for uniting the two 5 kHz measurements
            **label_load_kwargs_dict[label],
        )

        #     # calibrate TDC
        #     exp.calibrate_tdc(should_plot=True, force_processing=FORCE_PROCESSING)

        # save processed data (to avoid re-processing)
        halogen_exp_dict[label].save_processed_measurements(
            should_force=data_label_dict[label].force_processing or FORCE_ALL
        )

# Present count-rates
for label in used_labels:
    meas = halogen_exp_dict[label].confocal
    print(f"'{label}' countrate: {meas.avg_cnt_rate_khz:.2f} +/- {meas.std_cnt_rate_khz:.2f}")


# %% [markdown]
# Display the 'after fix' afterpulsings together

# %%
with Plotter(
    super_title="Afterpulsing Curves",
    xlabel="$t\ (ms)$",
    ylabel="Mean ACF times\nCountrate",
    x_scale="log",
    xlim=(5e-4, 1e0),
    ylim=(-1, 8.5e4),
    figsize=(5, 5),
) as ax:
    for label, exp in halogen_exp_dict.items():
        cf = exp.confocal.cf["confocal"]
        ax.plot(cf.lag, cf.avg_cf_cr, "--" if "old" in label else "-", label=label)
    ax.legend()

# %% [markdown]
# Save the above figure

# %%
save_figure(ax.figure, "OldVsNewAP.eps")

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
    eps = 0.005
    MAX_N_PARAMS = 64
    p0 = [1, 1]  # start from single decaying exponent (fit is sure to succeed)
    chi_sq_beta_list = []
    while abs(new_chi_sq_norm - 1) > eps:
        old_chi_sq_norm = new_chi_sq_norm
        try:
            meas.cf["confocal"].fit_correlation_function(
                x_field="lag",
                y_field="avg_cf_cr",
                y_error_field="error_cf_cr",
                fit_name="multi_exponent_fit",
                fit_param_estimate=p0,
                fit_range=(1.1e-4, 1e1),
                x_scale="log",
                y_scale="linear",
                should_plot=False,
                bounds=(0, np.inf),
                max_nfev=max_nfev,
            )
        except FitError as exc:  # fit failed
            print(exc)
            break
        else:
            new_chi_sq_norm = meas.cf["confocal"].fit_params["multi_exponent_fit"].chi_sq_norm
            new_beta = meas.cf["confocal"].fit_params["multi_exponent_fit"].beta
            chi_sq_beta_list.append((new_chi_sq_norm, new_beta))

            print(f"number of parameters: {new_beta.size}")
            print(f"new_chi_sq_norm: {new_chi_sq_norm}\n")

            p0 = new_beta.tolist() + [1, 1]
            if len(p0) > MAX_N_PARAMS:
                print(f"Maximal number of parameters {MAX_N_PARAMS} reached.")
                break

    #     break # TESTESTEST - only fit first measurement

    chi_sq_vals = [pair[0] for pair in chi_sq_beta_list]
    best_fit_idx = np.argmin(abs(np.array(chi_sq_vals) - 1))
    best_chi_sq, best_beta = chi_sq_beta_list[best_fit_idx]
    print(f"number of parameters: {best_beta.size}")
    print(f"final beta: {best_beta.tolist()}")
    print(f"final_chi_sq_norm: {best_chi_sq}\n")
    beta_dict[label] = best_beta

    # BEEP WHEN DONE!
    Beep(4000, 1000)

# %% [markdown]
# Compare the fits:

# %%
lag = np.linspace(1e-5, 1e-1, num=int(1e5))
with Plotter(
    x_scale="log",
    xlim=(1e-4, 1e-1),
    ylim=(0, 2e6),
    xlabel="Lag (ms)",
    ylabel="Mean CF_CR",
    super_title="White Noise Auto-Correlation\n(Afterpulsing)",
) as ax:

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
    ]  #'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    for (label, beta), color in zip(beta_dict.items(), colors):
        # plot data
        cf = halogen_exp_dict[label].confocal.cf["confocal"]
        ax.plot(cf.lag, cf.avg_cf_cr, "x", label=label + " (data)", color=color)
        # calculate and plot afterpulsing
        ap = calculate_calibrated_afterpulse(lag, ("multi_exponent_fit", beta))
        ax.plot(lag, ap, label=label + " (fit)", color=color)
    ax.legend()


# %% [markdown]
# Comparing to the same curves before MPD fix:

# %%
lag = np.linspace(1e-5, 1e-1, num=int(1e5))
with Plotter(
    x_scale="log",
    xlim=(1e-4, 1e-1),
    ylim=(0, 2e6),
    xlabel="Lag (ms)",
    ylabel="Mean CF_CR",
    super_title="White Noise Auto-Correlation\n(Afterpulsing)",
) as ax:

    colors = ["tab:red", "tab:purple", "tab:brown"]
    for label, beta, color in zip(
        [
            "White Noise 5 kHz (before fix)",
            "White Noise 60 kHz (before fix)",
            "White Noise 305 kHz (before fix)",
        ],
        beta_dict.values(),
        colors,
    ):
        # plot data
        cf = halogen_exp_dict[label].confocal.cf["confocal"]
        ax.plot(cf.lag, cf.avg_cf_cr, "x", label=label + " (data)", color=color)
        # calculate and plot afterpulsing
        ap = calculate_calibrated_afterpulse(lag, ("multi_exponent_fit", beta))
        ax.plot(lag, ap, label="post-fix fit", color=color)
    ax.legend()

# %% [markdown]
# Save the parameters:

# %%
import pickle

# are_you_sure = True
are_you_sure = False

if are_you_sure:
    with open("calibrated_afterpulsing_parameters.pkl", "wb") as f:
        pickle.dump(beta_dict["confocal"], f)
else:
    print("Are you sure you wish to over-write?")

# %% [markdown]
# Load saved parameters:

# %%
import pickle

with open("calibrated_afterpulsing_parameters.pkl", "rb") as f:
    beta = pickle.load(f)

afterpulse_params_list = [("multi_exponent_fit", beta) for beta in beta_dict]
print("Loaded 'beta':")
beta
