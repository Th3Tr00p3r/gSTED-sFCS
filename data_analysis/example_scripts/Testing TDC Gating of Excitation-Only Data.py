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

# general options
# AP_METHOD = "none"
# AP_METHOD = "subtract calibrated"
AP_METHOD = "filter (lifetime)"
FILES = "Use 1-5"
# FILES = "Use All"

data_label_kwargs = {
    "Old Det. 300 bp ATTO": dict(
        date="10_05_2018",
        confocal_template="bp300_angular_exc_*.mat",
        sted_template=None,
        file_selection=FILES,
        force_processing=False or FORCE_ALL,
        afterpulsing_method=AP_METHOD,
    ),
    "New Det. 300 bp ATTO": dict(
        date="03_07_2022",
        confocal_template="bp300ATTO_20uW_angular_exc_153213_*.pkl",
        sted_template=None,
        file_selection=FILES,
        force_processing=False or FORCE_ALL,
        afterpulsing_method=AP_METHOD,
    ),
    "New Det. 300 bp YOYO": dict(
        date="05_07_2022",
        confocal_template="bp300YOYO_TEST_diluted_12uW_angular_exc_145527_*.pkl",
        sted_template=None,
        file_selection=FILES,
        force_processing=False or FORCE_ALL,
        afterpulsing_method=AP_METHOD,
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
for label, exp in exp_dict.items():
    # skip already loaded experiments, unless forced
    if not hasattr(exp_dict[label], "confocal") or data_label_kwargs[label]["force_processing"]:
        exp.load_experiment(
            should_plot=False,
            should_re_correlate=FORCE_ALL,
            **data_label_kwargs[label],
        )

        #         # calibrate TDC # TESTESTEST
        #         exp.confocal.calibrate_tdc(**data_label_kwargs[label])

        # plot TDC calibration
        try:
            exp.confocal.tdc_calib.plot()
        except AttributeError:
            print("NO TDC CALIBRATION TO PLOT!")

        # save processed data (to avoid re-processing)
        exp.save_processed_measurements(
            should_force=FORCE_ALL,
        )

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
# Notice how the "bump" which always comes right before the flurescence peak is more pronounced for "new detector" measurements - this might imply that it is related to afterpulsing

# %% [markdown]
# ## Comparing afterpulsing removal by filtering

# %% [markdown]
# Calculate the filtered afterpulsings

# %%
if FORCE_ALL:
    for label, exp in exp_dict.items():
        try:
            exp.confocal.calculate_filtered_afterpulsing()
        except AttributeError:
            print("NO TDC CALIBRATION!")
        else:
            # save processed data (to avoid re-processing)
            exp.save_processed_measurements(
                should_force=True,
            )
else:
    print("Using pre-processed...")

# %% [markdown]
# Plotting all ACFs and afterpulsings

# %%
with Plotter(
    xlim=(1e-4, 1),
    x_scale="log",
    xlabel="lag (ms)",
    ylabel="Avg. CF x CR",
) as ax:
    max_g0 = 0
    for label, exp in exp_dict.items():
        meas = exp.confocal
        # plot the filters
        print()
        try:
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
        except AttributeError:
            print("NO TDC CALIBRATION!")
        cf_dict = meas.cf
        for cf_label, cf in cf_dict.items():
            ax.plot(
                cf.lag,
                cf.avg_cf_cr if label == "gated 12 uW" and cf_label == "confocal" else cf.avg_cf_cr,
                label=f"{label}: {cf_label}",
            )
            if cf_label == "confocal":
                max_g0 = max(max_g0, cf.g0 * 1.5)

    ax.set_ylim(0, max_g0)
    ax.legend()

# %% [markdown]
# ## Testing the effect of TDC gating on confocal measurements

# %% [markdown]
# Gating

# %%
tdc_gates_ns = [
    (7.5, 30),
    (7.5, np.inf),
]

for label, exp in exp_dict.items():
    for tdc_gate_ns in tdc_gates_ns:
        exp.add_gate(
            tdc_gate_ns, meas_type="confocal", should_plot=False, **data_label_kwargs[label]
        )

# %% [markdown]
# Plotting together:

# %%
# TODOף TOO MANY CURVES IN ONE AXES - SEPARATE INTO 3 GRAPHS ׂ (ATTO OLD, ATTO NEW,
# EACH SHOWING EACH GATE AND ITS AFTERPULSING (SOLID FOR SIGNAL, DASHED FOR AFTERPULSING)

with Plotter(
    x_scale="log",
    xlim=(1e-4, 1),
    ylim=(0, 8.5e4),
    xlabel="lagׂ (ms)",
    ylabel="Avg. CF_CR",
) as ax:
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for (label, exp), color in zip(exp_dict.items(), colors):
        for cf_label, cf in exp.confocal.cf.items():
            if cf_label == "confocal":
                ax.plot(cf.lag, cf.avg_cf_cr, label=f"{label}: {cf_label}", color=color)
                with suppress(AttributeError):
                    ax.plot(
                        cf.lag,
                        cf.subtracted_afterpulsing,
                        linestyle="dotted",
                        label=f"{label}: afterpulsing",
                        color=color,
                    )
            elif "(7.5, inf)" in cf_label:
                ax.plot(
                    cf.lag,
                    cf.avg_cf_cr,
                    linestyle="dashed",
                    label=f"{label}: {cf_label}",
                    color=color,
                )
            elif "(7.5, 30)" in cf_label:
                ax.plot(
                    cf.lag,
                    cf.avg_cf_cr,
                    linestyle="dashdot",
                    label=f"{label}: {cf_label}",
                    color=color,
                )
            elif cf_label == "afterpulsing":
                ax.plot(
                    cf.lag,
                    cf.avg_cf_cr,
                    linestyle="dotted",
                    label=f"{label}: {cf_label}",
                    color=color,
                )

    ax.legend()

# %% [markdown]
# ## Checking TDC calibrations

# %%
for label, exp in exp_dict.items():
    meas = exp.confocal
    print(label)
    for cf_label, cf in meas.cf.items():
        print(f"{cf_label}: {cf.countrate}")
    meas.tdc_calib.plot()

# %% [markdown]
# Beep when done

# %%
Beep(4000, 1000)
