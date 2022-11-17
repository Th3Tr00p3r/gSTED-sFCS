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
AP_METHOD = "filter"

NORM_RANGE = (7e-3, 9e-3)

FILES = "Use 1"
# FILES = "Use 1-5"
# FILES = "Use All"

data_label_kwargs = {
    "Old Det. 300 bp ATTO": dict(
        date="10_05_2018",
        confocal_template="bp300_angular_exc_*.mat",
        file_selection=FILES,
        force_processing=False or FORCE_ALL,
        afterpulsing_method=AP_METHOD,
        norm_range=NORM_RANGE,
    ),
    #      "New Det. 300 bp ATTO": dict(
    #          date="03_07_2022",
    #          confocal_template="bp300ATTO_20uW_angular_exc_153213_*.pkl",
    #          file_selection=FILES,
    #          force_processing=False or FORCE_ALL,
    #          afterpulsing_method=AP_METHOD,
    #          norm_range=NORM_RANGE,
    #      ),
    #      "New Det. 300 bp YOYO": dict(
    #          date="05_07_2022",
    #          confocal_template="bp300YOYO_TEST_diluted_12uW_angular_exc_145527_*.pkl",
    #          file_selection=FILES,
    #          force_processing=False or FORCE_ALL,
    #          afterpulsing_method=AP_METHOD,
    #          norm_range=NORM_RANGE,
    #      ),
    #     "15 ns Gated New Det. 25X Diluted Conc. Sample YOYO": dict(
    #         date="18_10_2022",
    #         confocal_template="gated_15ns_angular_exc_161655_*.pkl",
    #         file_selection=FILES,
    #         force_processing=False or FORCE_ALL,
    #         afterpulsing_method=AP_METHOD,
    #         norm_range=NORM_RANGE,
    #     ),
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
# Keeping the afterpulsing filter of the non-gated measurement to use with the gated one:

# %% [markdown]
# Importing all needed data. Processing, correlating and averaging if no pre-processed measurement exist.

# %%
# load experiment
for label, exp in exp_dict.items():
    # skip already loaded experiments, unless forced
    if not hasattr(exp_dict[label], "confocal") or data_label_kwargs[label]["force_processing"]:
        exp.load_experiment(
            should_plot=False,
            should_re_correlate=FORCE_ALL,
            **data_label_kwargs[label],
        )

        # plot TDC calibration
        try:
            exp.confocal.tdc_calib.plot()
        except AttributeError:
            print("NO TDC CALIBRATION TO PLOT!")

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
# Notice how the "bump" which always comes right before the flurescence peak is more pronounced for "new detector" measurements - this might imply that it is related to afterpulsing

# %% [markdown]
# ## Comparing afterpulsing removal by filtering

# %% [markdown]
# ## Testing the effect of TDC gating on confocal measurements

# %% [markdown]
# Gating and calculating the gated filtered afterpulsings

# %%
# CHOOSE GATES
# UPPER_GATES = [30, np.inf]
UPPER_GATES = [20]

SHOULD_CALCULATE_AP = False
# SHOULD_CALCULATE_AP = True

# GATE AND CALCULATE GATED APs
tdc_gates_ns = [(5, upper_gate) for upper_gate in UPPER_GATES]
for label, exp in exp_dict.items():
    if FORCE_ALL:
        try:
            # CALCULATE UNGATED AP
            if SHOULD_CALCULATE_AP:
                exp.confocal.calculate_filtered_afterpulsing()

            # GATE & CALCULATE GATED APs
            for tdc_gate_ns in tdc_gates_ns:
                exp.add_gate(
                    tdc_gate_ns,
                    meas_type="confocal",
                    should_plot=False,
                )
                if SHOULD_CALCULATE_AP:
                    exp.confocal.calculate_filtered_afterpulsing(tdc_gate_ns=tdc_gate_ns)

        except AttributeError:
            print("NO TDC CALIBRATION!")

    #         else:
    #             # save processed data (to avoid re-processing)
    #             exp.save_processed_measurements(
    #                 should_force=True,
    #             )
    else:
        print("Using pre-processed...")

# %% [markdown]
# Displaying filters

# %%
for label, exp in exp_dict.items():
    print(f"{label}:")
    exp.plot_afterpulsing_filters()

# %% [markdown]
# Adjusting normalization range if needed:

# %%
SHOULD_CHANGE_NORM_RANGE = False
# SHOULD_CHANGE_NORM_RANGE = True

if SHOULD_CHANGE_NORM_RANGE:

    print("Renormalizing...", end=" ")

    NORM_RANGE_ = (7e-3, 9e-3)
    #     NORM_RANGE_ = (1e-3, 2e-3)

    for label, exp in exp_dict.items():
        for cf in exp.confocal.cf.values():
            cf.average_correlation(norm_range=NORM_RANGE_)

    print("Done.")

# %% [markdown]
# Plotting together:

# %%
from utilities.display import get_gradient_colormap
from collections import deque

# colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

SUP_TITLE = (
    "AP Removal by Calibration Subtraction"
    if AP_METHOD == "subtract calibrated"
    else "AP Removal by Filtering"
)

X_FIELD = "lag"
Y_FIELD = "avg_cf_cr"
Y_FIELD = "normalized"

with Plotter(
    super_title=SUP_TITLE,
    subplots=(len(exp_dict), 1),
    x_scale="log",
    xlim=(1e-4, 1),
    xlabel=X_FIELD,
    ylabel=Y_FIELD,
) as axes:
    if len(exp_dict) == 1:
        axes = [axes]
    for (label, exp), ax in zip(exp_dict.items(), axes):
        # create pairs of signal and afterpulsing and plot them together with matching colors
        signal_ap_pairs = deque([])
        n_lines = 0
        for cf in exp.confocal.cf.values():
            if not "afterpulsing" in cf.name:
                n_lines += 1
                if SHOULD_CALCULATE_AP:
                    for other_cf in exp.confocal.cf.values():
                        if "afterpulsing" in other_cf.name:
                            if other_cf.name.replace("afterpulsing", "confocal") == cf.name or (
                                cf.name == "confocal" and other_cf.name == "afterpulsing"
                            ):
                                signal_ap_pairs.append((cf, other_cf))
                else:
                    signal_ap_pairs.append((cf, None))

        # plotting
        colors = get_gradient_colormap(n_lines + 1)
        signal_ap_pairs.rotate(-1)  # TESTESTEST roll list
        for (signal_cf, ap_cf), color in zip(signal_ap_pairs, colors):
            ax.plot(
                getattr(signal_cf, X_FIELD),
                getattr(signal_cf, Y_FIELD),
                label=f"{signal_cf.name}",
                color=color,
            )
            if Y_FIELD == "avg_cf_cr" and not SHOULD_CALCULATE_AP:
                ax.plot(
                    getattr(ap_cf, X_FIELD),
                    getattr(ap_cf, Y_FIELD),
                    label=f"{ap_cf.name}",
                    color=color,
                    linestyle="dashed",
                )

        # adjusting ylims
        if Y_FIELD == "avg_cf_cr":
            # choose the first "signal" cf's g0
            ax.set_ylim(0, signal_ap_pairs[0][0].g0 * 1.5)
        elif Y_FIELD == "normalized":
            ax.set_ylim(-0.05, 1.3)

        # title and legend
        ax.set_title(label)
        ax.legend()

# %% [markdown]
# Beep when done

# %%
Beep(4000, 1000)
