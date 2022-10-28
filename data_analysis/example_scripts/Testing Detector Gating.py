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

# %% [markdown]
# Choosing the data templates:

# %%
FORCE_ALL = True
# FORCE_ALL = False

GATE_NS = (7.5, 20)

data_label_kwargs = {
    "200 nW": dict(
        date="18_10_2022",
        confocal_template="25XDilutedConcentratedSample_200nW_exc_noGating_angular_exc_152653_*.pkl",
        sted_template=None,
        file_selection="Use 1-2",
        force_processing=False or FORCE_ALL,
        afterpulsing_method="filter (lifetime)",
        #         afterpulsing_method="subtract calibrated",
        #         baseline_method="fit",
        baseline_method="external",
        external_baseline=3.5e-4,
        #         gate_ns=GATE_NS # TESTESTEST
    ),
    "gated 12 uW": dict(
        date="18_10_2022",
        confocal_template="gated_15ns_angular_exc_161655_*.pkl",
        sted_template=None,
        file_selection="Use 1-5",
        force_processing=False or FORCE_ALL,
        afterpulsing_method="filter (lifetime)",
        #         afterpulsing_method="subtract calibrated",
        #         baseline_method="fit",
        baseline_method="external",
        external_baseline=2.7e-4,
        tdc_gate_ns=GATE_NS,  # TESTESTEST
    ),
}

data_labels = list(data_label_kwargs.keys())

n_meas = len(data_labels)

# build proper paths
for data in data_label_kwargs.values():
    data["confocal_template"] = DATA_ROOT / data["date"] / DATA_TYPE / data["confocal_template"]
    data["sted_template"] = (
        DATA_ROOT / data["date"] / DATA_TYPE / data["sted_template"]
        if data["sted_template"]
        else None
    )

# initialize the experiment dictionary, if it doens't already exist
if "exp_dict" not in locals():
    print("resetting 'exp_dict'")
    exp_dict = {}

for label in data_labels:
    if label not in exp_dict:
        exp_dict[label] = SolutionSFCSExperiment(name=label)

# TEST
print(data_label_kwargs)

# %% [markdown]
# Keeping the afterpulsing filter of the non-gated measurement to use with the gated one:

# %% [markdown]
# Importing all needed data. Processing, correlating and averaging if no pre-processed measurement exist.

# %%
used_labels = data_labels
# used_labels = [
#    "sample"
# ]

# load experiment
for label in used_labels:
    # skip already loaded experiments, unless forced
    if not hasattr(exp_dict[label], "confocal") or data_label_kwargs[label]["force_processing"]:
        exp_dict[label].load_experiment(
            should_plot=False,
            should_re_correlate=FORCE_ALL,
            **data_label_kwargs[label],
        )

        # plot TDC calibration
        exp_dict[label].confocal.tdc_calib.plot()

        # save processed data (to avoid re-processing)
        exp_dict[label].save_processed_measurements(
            should_force=FORCE_ALL,
        )

# Present count-rates
for label in used_labels:
    exp = exp_dict[label]
    print(f"'{label}' countrates:")
    print(f"Confocal: {exp.confocal.avg_cnt_rate_khz:.2f} +/- {exp.confocal.std_cnt_rate_khz:.2f}")
    with suppress(AttributeError):
        print(f"STED: {exp.sted.avg_cnt_rate_khz:.2f} +/- {exp.sted.std_cnt_rate_khz:.2f}")
    print()


# %% [markdown]
# ## Plotting both CF_CRs together

# %%
with Plotter() as ax:
    for label, exp in exp_dict.items():
        exp.plot_correlation_functions(
            parent_ax=ax,
            y_field="avg_cf_cr",
            x_field="lag",
            x_scale="log",
            xlim=None,  # autoscale x axis
        )

# %% [markdown]
# ## Checking out the afterpulsings

# %% [markdown]
# #### Calculate the filtered afterpulsings

# %%
if FORCE_ALL:
    for label, exp in exp_dict.items():
        exp.confocal.calculate_filtered_afterpulsing(tdc_gate_ns=GATE_NS)

# %%
with Plotter(
    xlim=(5e-4, 1),
    ylim=(0, 1.1e4),
    x_scale="log",
    xlabel="lag (ms)",
    ylabel="Avg. CF x CR",
) as ax:
    for label, exp in exp_dict.items():
        meas = exp.confocal
        # plot the filters
        print()
        meas.tdc_calib.calculate_afterpulsing_filter(
            meas.detector_settings["gate_ns"],
            should_plot=True,
            **data_label_kwargs[label],
            #             baseline_method="fit",
            #             baseline_method="external",
            #             external_baseline=1 # 2.7e-4,
            #             baseline_method="range",
            #             baseline_range=(30, 100),
        )
        cf_dict = meas.cf
        for cf_label, cf in cf_dict.items():
            ax.plot(
                cf.lag,
                cf.avg_cf_cr,
                label=f"{label}: {cf_label}",
            )
    ax.legend()
