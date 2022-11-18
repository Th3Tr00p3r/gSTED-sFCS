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
from contextlib import suppress

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
        #         PROJECT_ROOT = "D:\people\Idomic\gSTED-sFCS"
        PROJECT_ROOT = "D:\MEGA\BioPhysics_Lab\Optical_System\gSTEDsFCS"
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

# %% [markdown]
# Choosing the data templates:

# %%
FORCE_ALL = True
# FORCE_ALL = False

# general options
AP_METHOD = "filter"
# AP_METHOD = "none"
# AP_METHOD = "subtract calibrated"

# FILES = "Use 1"
# FILES = "Use 1-5"
FILES = "Use All"

data_label_kwargs = {
    #     "300 bp ATTO PDM - no ap rmvl": dict(
    #         date="10_05_2018",
    #         confocal_template="bp300_angular_exc_*.mat",
    #         sted_template="bp300_angular_sted_*.mat",
    #         file_selection=FILES,
    #         force_processing=False or FORCE_ALL,
    #         afterpulsing_method="none",
    #     ),
    #     "300 bp ATTO PDM - cal. subt.": dict(
    #         date="10_05_2018",
    #         confocal_template="bp300_angular_exc_*.mat",
    #         sted_template="bp300_angular_sted_*.mat",
    #         file_selection=FILES,
    #         force_processing=False or FORCE_ALL,
    #         afterpulsing_method="subtract calibrated",
    #     ),
    "300 bp ATTO PDM - filtering": dict(
        date="10_05_2018",
        confocal_template="bp300_angular_exc_*.mat",
        sted_template="bp300_angular_sted_*.mat",
        file_selection=FILES,
        force_processing=False or FORCE_ALL,
        afterpulsing_method="filter",
    ),
    #     "300 bp ATTO FastGatedSPAD - no ap rmvl": dict(
    #         date="16_11_2022",
    #         confocal_template="ATTO300bp_circle_exc_162743_*.pkl",
    #         sted_template="ATTO300bp_circle_sted_165440_*.pkl",
    #         file_selection=FILES,
    #         force_processing=False or FORCE_ALL,
    #         afterpulsing_method="none",
    #     ),
    #     "300 bp ATTO FastGatedSPAD - cal. subt.": dict(
    #         date="16_11_2022",
    #         confocal_template="ATTO300bp_circle_exc_162743_*.pkl",
    #         sted_template="ATTO300bp_circle_sted_165440_*.pkl",
    #         file_selection=FILES,
    #         force_processing=False or FORCE_ALL,
    #         afterpulsing_method="subtract calibrated",
    #     ),
    "300 bp ATTO FastGatedSPAD - filtering": dict(
        date="16_11_2022",
        confocal_template="ATTO300bp_circle_exc_162743_*.pkl",
        sted_template="ATTO300bp_circle_sted_165440_*.pkl",
        file_selection=FILES,
        force_processing=False or FORCE_ALL,
        afterpulsing_method="filter",
    ),
}

# build proper paths
for data in data_label_kwargs.values():
    data["confocal_template"] = DATA_ROOT / data["date"] / DATA_TYPE / data["confocal_template"]
    data["sted_template"] = (
        DATA_ROOT / data["date"] / DATA_TYPE / data["sted_template"]
        if data.get("sted_template")
        else None
    )

# initialize the experiment dictionary, if it doens't already exist
if "exp_dict" not in locals():
    print("resetting 'exp_dict'")
    exp_dict = {}

for label in list(data_label_kwargs.keys()):
    if label not in exp_dict:
        exp_dict[label] = SolutionSFCSExperiment(name=label)

# TEST
print(data_label_kwargs)

# %% [markdown]
# Importing all needed data. Processing, correlating and averaging if no pre-processed measurement exist.

# %%
# load experiment
for label, exp in exp_dict.items():

    # skip already loaded experiments, unless forced
    if not hasattr(exp_dict[label], "confocal") or data_label_kwargs[label]["force_processing"]:
        exp.load_experiment(
            should_plot=False,
            should_plot_meas=False,
            should_re_correlate=FORCE_ALL,
            **data_label_kwargs[label],
        )

#         # save processed data (to avoid re-processing)
#         exp.save_processed_measurements(
#             should_force=FORCE_ALL,
#         )

# Present count-rates
for label, exp in exp_dict.items():
    print(f"'{label}' countrates:")
    print(
        f"Confocal: {exp.confocal.avg_cnt_rate_khz:.2f} +/- {exp.confocal.std_cnt_rate_khz:.2f} kHz"
    )
    with suppress(AttributeError):
        print(f"STED: {exp.sted.avg_cnt_rate_khz:.2f} +/- {exp.sted.std_cnt_rate_khz:.2f} kHz")
    print()


# %% [markdown]
# ## Comparing afterpulsing removal by filtering

# %% [markdown]
# TDC Gating:

# %%
from itertools import product

# CHOOSE GATES
# lower_gate_list = [3.3] # PDM
lower_gate_list = [5]  # FastGatedSPAD
upper_gate_list = [20]

gate_list = [gate_ns for gate_ns in product(lower_gate_list, upper_gate_list)]

# GATE AND CALCULATE GATED APs
for label, exp in exp_dict.items():
    if FORCE_ALL:
        # gate STED
        exp.add_gates(gate_list, should_plot=False)
        # gate confocal
    #         exp.add_gates(gate_list, meas_type="confocal", should_plot=False)
    else:
        print("Using pre-processed...")

# %% [markdown]
# Adjusting normalization range (if needed) and reploting:

# %%
# SHOULD_CHANGE_NORM_RANGE = False
SHOULD_CHANGE_NORM_RANGE = True

# NORM_RANGE_ = (1e-3, 2e-3)
NORM_RANGE_ = (2e-3, 3e-3)
# NORM_RANGE_ = (7e-3, 9e-3)

for label, exp in exp_dict.items():

    if SHOULD_CHANGE_NORM_RANGE:
        print("Renormalizing...", end=" ")
        exp.renormalize_all(NORM_RANGE_)
        print("Done.")

    print("Re-plotting...")
    with Plotter(subplots=(2, 2), super_title=label) as axes:
        exp.plot_correlation_functions(
            x_field="lag", x_scale="log", xlim=(1e-4, 1), parent_ax=axes[0][0]
        )  # lag
        exp.plot_correlation_functions(parent_ax=axes[0][1])  # vt_um
        exp.plot_correlation_functions(
            x_field="vt_um_sq", y_scale="log", xlim=(1e-4, 1), ylim=(5e-3, 1), parent_ax=axes[1][1]
        )  # semilogy vt_um^2

# %% [markdown]
# ## Comparing afterpulsing removal methods for TDC-gated measurements

# %%
with Plotter(
    subplots=(len(gate_list), 1),
    x_scale="log",
    xlim=(1e-4, 1),
    #     ylim=(1e-4, 1.2),
    xlabel="lag (ms)",
    #     ylabel="Normalized",
    ylabel="Avg. CFxCR",
) as axes:
    if len(gate_list) == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        linewidth = 20
        for label, exp in exp_dict.items():
            linewidth /= 2
            #         cf = list(exp.confocal.cf.values())[0] # confocal
            cf = list(exp.confocal.cf.values())[idx]  # gated confocal
            #         cf = list(exp.sted.cf.values())[0] # sted
            #         cf = list(exp.sted.cf.values())[-1] # gated sted
            #         ax.plot(cf.lag, cf.avg_cf_cr, label=label, linewidth=linewidth)
            #             ax.plot(cf.lag, cf.normalized, label=label, linewidth=linewidth) # NORMALIZED
            ax.plot(cf.lag, cf.avg_cf_cr, label=label, linewidth=linewidth)  # CF_CR
            if "filtering" in label:
                ax.set_ylim(0, cf.g0 * 1.3)

        ax.set_title(cf.name)
        ax.legend()

# %% [markdown]
# ## Plotting all confocal gates together for each afterpulsing method (should look the same)

# %%
with Plotter(
    subplots=(len(exp_dict), 1),
    x_scale="log",
    xlim=(1e-4, 1),
    ylim=(1e-4, 1.2),
    xlabel="lag (ms)",
    ylabel="Normalized",
    #     ylabel="Avg. CFxCR",
) as axes:
    if len(exp_dict) == 1:
        axes = [axes]
    for idx, (label, exp) in enumerate(exp_dict.items()):
        for cf in exp.confocal.cf.values():
            axes[idx].plot(cf.lag, cf.normalized, label=cf.name)  # CF_CR

        axes[idx].set_title(label)
        axes[idx].legend()

# %% [markdown]
# ## Showing filters for gated measurements

# %%
for label, exp in exp_dict.items():
    if "filter" in label:
        print(f"===== {label} =====")
        with Plotter(subplots=(len(gate_list), 2)) as axes:
            for idx, cf in enumerate(exp.confocal.cf.values()):
                cf.afterpulsing_filter.plot(parent_ax=axes)
                axes[0].set_ylabel(cf.name)

# %% [markdown]
# Beep when done

# %%
Beep(4000, 1000)
