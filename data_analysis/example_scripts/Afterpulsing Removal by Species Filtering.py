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
    SFCSExperiment,
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
    #     "300 bp": SimpleNamespace(
    #         date="03_07_2022",
    #         confocal_template="bp300ATTO_20uW_angular_exc_153213_*.pkl",
    #         sted_template=None,
    #         file_selection="Use All",
    #         force_processing=False,
    #         should_subtract_afterpulse=False,
    #         #         force_processing=True,
    #     ),
    "free ATTO": SimpleNamespace(
        date="05_10_2021",
        confocal_template="sol_static_exc_181325_*.pkl",
        sted_template="sol_static_sted_183413_*.pkl",
        file_selection="Use 1",
        force_processing=False,
        #         gating_mechanism="binary filter",
        gating_mechanism="removal",
        #         afterpulsing_method="filter (lifetime)",
        #         afterpulsing_method="subtract calibrated",
        afterpulsing_method="none",
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
        exp_dict[label] = SFCSExperiment(name=label)

label_load_kwargs_dict = {
    label: dict(
        confocal_template=tmplt_path.confocal_template,
        sted_template=tmplt_path.sted_template,
        file_selection=data.file_selection,
        gating_mechanism=data.gating_mechanism,
        afterpulsing_method=data.afterpulsing_method,
    )
    for label, tmplt_path, data in zip(data_labels, template_paths, data_label_dict.values())
}

# TEST
print(template_paths)

# %% [markdown]
# Importing all needed data. Processing, correlating and averaging if no pre-processed measurement exist.

# %%
FORCE_ALL = True
# FORCE_ALL = False

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
# Constructing the filter (matrix) for afterpulsing, as described in [this paper](https://doi.org/10.1063/1.1863399).

# %%
from utilities.helper import Limits
from numpy.linalg import inv
from copy import copy


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


# label = "300 bp"
label = "free ATTO"

tdc_calib = exp_dict[label].confocal.tdc_calib

t_hist = tdc_calib.t_hist
all_hist = copy(tdc_calib.all_hist.astype(np.float64))
nonzero_idxs = tdc_calib.hist_weight > 0
all_hist[nonzero_idxs] /= tdc_calib.hist_weight[nonzero_idxs]  # weight the histogram

# interpolate over NaNs
all_hist[~nonzero_idxs] = np.nan  # NaN the zeros (for interpolation)
nans, x = nan_helper(all_hist)  # get nans and a way to interpolate over them later
all_hist[nans] = np.interp(x(nans), x(~nans), all_hist[~nans])

# normalize
norm_factor = all_hist.sum()  # / 1.4117
# TODO: why is this 1.4 factor needed?? not needed for static?? or is it a new detector thing??
all_hist_norm = all_hist / norm_factor

# calculate the baseline (which is assumed to be approximately the afterpulsing histogram)
baseline_limits = Limits(60, 78)
baseline_idxs = baseline_limits.valid_indices(t_hist)
baseline = np.mean(all_hist_norm[baseline_idxs])

# idc - ideal decay curve
M_j1 = all_hist_norm - baseline  # p1
M_j2 = 1 / len(t_hist) * np.ones(t_hist.shape)  # p2
I_j = all_hist_norm

M = np.vstack((M_j1, M_j2)).T
inv_I = np.diag(1 / I_j)

F = inv(M.T @ inv_I @ M) @ M.T @ inv_I

with Plotter(subplots=(1, 2)) as axes:
    axes[0].set_yscale("log")
    axes[0].plot(t_hist, I_j, label="I_j (raw histogram)")
    axes[0].plot(t_hist, baseline * np.ones(t_hist.shape), label="baseline")
    axes[0].plot(t_hist, M_j1, label="M_j1 (ideal fluorescence decay curve)")
    axes[0].plot(t_hist, M_j2, label="M_j2 (ideal afterpulsing 'decay' curve)")
    axes[0].legend()

    axes[1].set_ylim(-1, 2)
    axes[1].plot(t_hist, F.T)
    axes[1].plot(t_hist, F.sum(axis=0))
    axes[1].legend(["F_1j", "F_2j", "F.sum(axis=0)"])

print(f"F.shape: {F.shape}")
print(f"M.shape: {M.shape}")


# %% [markdown]
# Verifying:
#
# $\sum^L_{j=1}F_{ij}M_{jk} = \delta_{ik},$
#
# where $M_{jk} = p^{(k)}_j$, thus $F\cdot M$ should yield the unit matrix (2x2)

# %%
F @ M

# %% [markdown]
# and summing the filters should yield one for each channel $j$

# %%
total_prob_j = F.sum(axis=0)
print(f"{total_prob_j.mean()} +/- {total_prob_j.std()}")

# %% [markdown]
# ## Applying to correlation

# %% [markdown]
# We have designed a new correlator which accepts for each photon timestamp a filter value. The filter value is assigned according to which bin the photon timestamp belongs to and the value for that bin (given by the matrix $F$, built from the histogram).
#
# First, let's try assigning to each photon timestamp the correct bin

# %% [markdown]
# In order to test the lifetime correlation, let's attempt to use the filter as a gating mechanism - 0 for gated photon, 1 for included photon. We will need to use a circular-scan/static measurement with STED to notice the effects of gating, since there isn't a line correlator ready for lifetime correlation yet.
#
# Using a static atto STED measurement, let's try comparing gating in the regular way (by not correlating the gated photons at all) and gating by setting the filter values to 0/1:

# %%
exp = exp_dict["free ATTO"]

exp.add_gate((3.61, 20), **label_load_kwargs_dict[label])
