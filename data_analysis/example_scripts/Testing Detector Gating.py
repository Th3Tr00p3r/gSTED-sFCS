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
data_label_dict = {
    "gated 12 uW": SimpleNamespace(
        date="18_10_2022",
        confocal_template="gated_15ns_angular_exc_161655_*.pkl",
        sted_template=None,
        file_selection="Use 1-5",
        #         force_processing=True,
        force_processing=False,
        afterpulsing_method="filter (lifetime)",
        #         hist_norm_factor=1.4117,
    ),
    "200 nW": SimpleNamespace(
        date="18_10_2022",
        confocal_template="25XDilutedConcentratedSample_200nW_exc_noGating_angular_exc_152653_*.pkl",
        sted_template=None,
        file_selection="Use 1-5",
        #         force_processing=True,
        force_processing=False,
        afterpulsing_method="filter (lifetime)",
        #         hist_norm_factor=1.4117,
    ),
}

data_labels = list(data_label_dict.keys())

n_meas = len(data_labels)

template_paths = [
    SimpleNamespace(
        confocal_template=DATA_ROOT / data.date / DATA_TYPE / data.confocal_template,
        sted_template=DATA_ROOT / data.date / DATA_TYPE / data.sted_template
        if data.sted_template
        else None,
    )
    for data in data_label_dict.values()
]

# initialize the experiment dictionary, if it doens't already exist
if "exp_dict" not in locals():
    print("resetting 'exp_dict'")
    exp_dict = {}

for label in data_labels:
    if label not in exp_dict:
        exp_dict[label] = SolutionSFCSExperiment(name=label)

label_load_kwargs_dict = {
    label: dict(
        confocal_template=tmplt_path.confocal_template,
        sted_template=tmplt_path.sted_template,
        file_selection=data.file_selection,
        afterpulsing_method=data.afterpulsing_method,
        #         hist_norm_factor=data.hist_norm_factor,
    )
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
#    "sample"
# ]

# load experiment
for label in used_labels:
    # skip already loaded experiments, unless forced
    if (
        not hasattr(exp_dict[label], "confocal")
        or data_label_dict[label].force_processing
        or FORCE_ALL
    ):
        exp_dict[label].load_experiment(
            should_plot=False,
            force_processing=data_label_dict[label].force_processing or FORCE_ALL,
            should_re_correlate=FORCE_ALL,
            baseline_method="range",
            **label_load_kwargs_dict[label],
        )

        # calibrate TDC
        exp_dict[label].calibrate_tdc(
            should_plot=True,
            force_processing=data_label_dict[label].force_processing or FORCE_ALL,
        )

        # save processed data (to avoid re-processing)
        exp_dict[label].save_processed_measurements(
            should_force=data_label_dict[label].force_processing or FORCE_ALL
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
# #### Caculate the filtered afterpulsings

# %%
for label, exp in exp_dict.items():
    exp.confocal.calculate_filtered_afterpulse()

# %%
with Plotter(
    xlim=(5e-4, 1),
    ylim=(0, 1.1e4),
) as ax:
    for label, exp in exp_dict.items():
        meas = exp.confocal
        # plot the filters
        print()
        meas.tdc_calib.calculate_afterpulsing_filter(should_plot=True)
        cf_dict = meas.cf
        for idx, (cf_name, CF) in enumerate(cf_dict.items()):
            CF.plot_correlation_function(
                parent_ax=ax,
                y_field="avg_cf_cr",
                x_field="lag",
                x_scale="log",
                xlim=None,  # autoscale x axis
                plot_kwargs={"label": f"{label}: {cf_name}"},
            )
    ax.legend()
