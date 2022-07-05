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
# Here I take a look at the difference between gating the same amount using post-measurement TDC gating and using real-time gating with the FastGatedSPAD

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
# Loading the data:

# %%
data_label_dict = {
    "300 bp ATTO exc FR": SimpleNamespace(
        date="03_07_2022",
        template="bp300ATTO_20uW_angular_exc_153213_*.pkl",
        file_selection="Use All",
        force_processing=True,
        should_use_inherent_afterpulsing=False,
    ),
    "300 bp ATTO STED FR": SimpleNamespace(
        date="03_07_2022",
        template="bp300ATTO_20uW_angular_sted_161127_*.pkl",
        file_selection="Use All",
        force_processing=True,
        should_use_inherent_afterpulsing=False,
    ),
    "300 bp ATTO STED 3 ns Detector Gating": SimpleNamespace(
        date="03_07_2022",
        template="bp300ATTO_20uW_Gated3ns_angular_sted_171540_*.pkl",
        file_selection="Use All",
        force_processing=True,
        should_use_inherent_afterpulsing=True,
    ),
}

confocal_label = "300 bp ATTO exc FR"

sted_labels = ["300 bp ATTO STED FR", "300 bp ATTO STED 3 ns Detector Gating"]

data_labels = list(data_label_dict.keys())

n_meas = len(data_labels)

label_template_paths_dict = {
    label: DATA_ROOT / data.date / DATA_TYPE / data.template
    for label, data in zip(data_labels, data_label_dict.values())
}

halogen_exp_dict = {label: SFCSExperiment(name=label) for label in sted_labels}

label_load_kwargs_dict = {
    label: dict(
        file_selection=data.file_selection,
        should_use_inherent_afterpulsing=data.should_use_inherent_afterpulsing,
    )
    for label, data in zip(data_labels, data_label_dict.values())
}

# TEST - print paths
print(list(label_template_paths_dict.values()))

# %%
FORCE_PROCESSING = False
# FORCE_PROCESSING = True

# load experiment
for label in sted_labels:
    halogen_exp_dict[label].load_experiment(
        confocal_template=label_template_paths_dict[confocal_label],
        sted_template=label_template_paths_dict[label],
        force_processing=data_label_dict[label].force_processing,
        should_re_correlate=FORCE_PROCESSING,
        should_subtract_afterpulse=True,
        should_unite_start_times=True,  # for uniting the two 5 kHz measurements
        inherent_afterpulsing_gates=(Limits(3, 10), Limits(30, 90)),
        **label_load_kwargs_dict[label],
    )

    # save processed data (to avoid re-processing)
    halogen_exp_dict[label].save_processed_measurements(
        should_force=data_label_dict[label].force_processing
    )

    # calibrate TDC
    halogen_exp_dict[label].calibrate_tdc(should_plot=True)

# Present count-rates
for label in sted_labels:
    print(f"\n{label}:")
    conf_meas = halogen_exp_dict[label].confocal
    sted_meas = halogen_exp_dict[label].sted
    print(
        f"Confocal countrate: {conf_meas.avg_cnt_rate_khz:.2f} +/- {conf_meas.std_cnt_rate_khz:.2f}"
    )
    print(f"STED countrate: {sted_meas.avg_cnt_rate_khz:.2f} +/- {sted_meas.std_cnt_rate_khz:.2f}")

# %% [markdown]
# Plot the STED CF_CRs:

# %%
with Plotter(
    xlabel="Lag (ms)", ylabel="Mean CF_CR", x_scale="log", xlim=(1e-4, 1e0), ylim=(-1, 6e4)
) as ax:
    for label in sted_labels:
        cf = halogen_exp_dict[label].sted.cf["sted"]
        ax.plot(cf.lag, cf.avg_cf_cr, label=label)
        ax.legend()

# %% [markdown]
# Now, lets gate the FR (Free Running) measurement for 3 ns as well. To make sure were normalizing the same way, let's TDC gate both of the measurements for the same gate.
#
# tdc/detector gating should overlap correctly - see line 1108 ["Unite TDC gate and detector gate"] in correlation_function.py

# %%
detector_label = "300 bp ATTO STED 3 ns Detector Gating"
tdc_label = "300 bp ATTO STED FR"

gate_list = [(3, 90), (4, 90), (5, 90), (6, 90)]

FORCE_CORR = True
# FORCE_CORR = False

halogen_exp_dict[tdc_label].add_gates(
    gate_list,
    #     should_use_inherent_afterpulsing=True,
    inherent_afterpulsing_gates=(Limits(3, 10), Limits(30, 90)),
    should_re_correlate=FORCE_CORR,
)
halogen_exp_dict[detector_label].add_gates(
    gate_list,
    #     norm_range=(5e-3, 6e-3),
    should_use_inherent_afterpulsing=True,
    inherent_afterpulsing_gates=(Limits(3, 10), Limits(30, 90)),
    should_re_correlate=FORCE_CORR,
)

# %% [markdown]
# and plot the 3 ns gate together with the detector gates one:

# %%
with Plotter(
    xlabel="Lag (ms)", ylabel="Normalized", x_scale="log", xlim=(1e-4, 1e0), ylim=(-0.1, 2)
) as ax:

    meas_det_gated = halogen_exp_dict[detector_label].sted
    meas_tdc_gated = halogen_exp_dict[tdc_label].sted

    ax.plot(
        meas_det_gated.cf["sted"].lag,
        meas_det_gated.cf["sted"].normalized,
        "--",
        label="hard-gated",
    )
    for gate in gate_list:
        ax.plot(
            meas_det_gated.cf[f"gSTED {gate}"].lag,
            meas_det_gated.cf[f"gSTED {gate}"].normalized,
            "--",
            label=f"hard-gated {gate}",
        )

    ax.set_prop_cycle(None)

    ax.plot(
        meas_tdc_gated.cf["sted"].lag, meas_tdc_gated.cf["sted"].normalized, label="free-running"
    )
    for gate in gate_list:
        ax.plot(
            meas_tdc_gated.cf[f"gSTED {gate}"].lag,
            meas_tdc_gated.cf[f"gSTED {gate}"].normalized,
            label=f"free-runing {gate}",
        )

    ax.legend()

# %% [markdown]
# Play sound when done:

# %%
delta_t = 175
freq_range = (3000, 7000)
n_beeps = 5
[Beep(int(f), delta_t) for f in np.linspace(*freq_range, n_beeps)]
