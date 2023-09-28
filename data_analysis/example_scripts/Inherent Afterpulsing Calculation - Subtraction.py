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
# - Import core and 3rd party modules
# - Move current working directory to project root (if needed) and import project modules
# - Set data paths and other constants

# %%
######################################
# importing core and 3rd-party modules
######################################

import os
import pickle
import re
import scipy
from pathlib import Path
from winsound import Beep
from contextlib import suppress

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

from data_analysis.correlation_function import CorrFunc, SFCSExperiment, SolutionSFCSMeasurement
from data_analysis.software_correlator import CorrelatorType, SoftwareCorrelator
from utilities.display import Plotter, get_gradient_colormap
from utilities.file_utilities import (
    default_system_info,
    load_mat,
    load_object,
    save_object,
    save_processed_solution_meas,
)
from utilities.helper import Limits, fourier_transform_1d

#################################################
# Setting up data path and other global constants
#################################################

DATA_ROOT = Path("D:\OneDrive - post.bgu.ac.il\gSTED_sFCS_Data")  # Laptop/Lab PC (same path)
# DATA_ROOT = Path("D:\???")  # Oleg's
DATA_TYPE = "solution"

FORCE_PROCESSING = False
FORCE_PROCESSING = True

# %% [markdown]
# ## Comparing afterpulsing subtration: whitenoise autocorrelation (Halogen lamp calibration) vs. cross-correlating gates

# %% [markdown]
# Here I attempt to compare the afterpulsing gotten from auto-correlating white noise, to this gotten from cross-correlating valid photons from "white-noise" photons in fluorescent sample measurements.
#
# First, let's quickly get the default system afterpulsing currently used (matching the currently used detector settings).
# For this we can load some measurement to get some lag:

# %%
# DATA_DATE = "29_03_2022"; confocal_template = "bp300_20uW_angular_exc_172325_*.pkl"; label = "Free-Running 300 bp"
# DATA_DATE = "29_03_2022"; confocal_template = "bp300_20uW_200mW_angular_sted_174126_*.pkl"; label = "Free-Running 300 bp STED"
# DATA_DATE = "30_01_2022"; confocal_template = "yoyo300bp500nW_angular_sted_125650_*.pkl"; label = "Old Detector Free-Running 300 bp STED"

# DATA_DATE = "10_05_2018"; confocal_template = "bp300_angular_exc_*.mat"; label = "MATLAB Free-Running 300 bp"
DATA_DATE = "10_05_2018"
confocal_template = "bp300_angular_sted_*.mat"
label = "MATLAB Free-Running 300 bp STED"
# DATA_DATE = "20_08_2018"; confocal_template = "EdU300bp_angular_sted_*.mat"; label = "MATLAB Free-Running EdU 300 bp STED"

# DATA_DATE = "06_04_2022"; confocal_template = "atto_FR_angular_exc_141224_*.pkl"; label = "Free-Running ATTO"
# DATA_DATE = "13_03_2022"; confocal_template = "atto_12uW_FR_static_exc_182414_*.pkl"; label = "Free-Running static ATTO"

DATA_PATH = DATA_ROOT / DATA_DATE / DATA_TYPE

# load experiment
exp = SFCSExperiment(name=label)
exp.load_experiment(
    confocal_template=DATA_PATH / confocal_template,
    should_plot=True,
    should_use_preprocessed=not FORCE_PROCESSING,
    should_re_correlate=True,
)

# save processed data (to avoid re-processing)
exp.save_processed_measurements()

# Show countrate
print(f"Count-Rate: {exp.confocal.avg_cnt_rate_khz} kHz")

# %% [markdown]
# Now, using the same measurement, we'll try cross-correlating the 'fluorescence' and 'calibration' photons.
#
# First, we need to calibrate the TDC, in order to know which photons belong in each group:

# %%
exp.calibrate_tdc()

# %% [markdown]
# Next, let's cross-correlate the fluorescence and white-noise photons. This can be done by choosing gating once from 0 to 40 ns, then from 40 to 100 (or np.inf) ns:

# %%
meas = exp.confocal
meas.xcf = {}  # "halogen_afterpulsing": meas.cf["confocal"].afterpulse}

gate1_ns = Limits(2, 10)
gate2_ns = Limits(35, 85)

corr_names = ("AB",)
CF_list = meas.cross_correlate_data(
    cf_name="fl_vs_wn",
    corr_names=corr_names,
    gate1_ns=gate1_ns,
    gate2_ns=gate2_ns,
    subtract_spatial_bg_corr=True,
    should_subtract_afterpulse=False,
)

CF_dict = {xx: CF_xx for xx, CF_xx in zip(corr_names, CF_list)}

for CF in CF_dict.values():
    CF.average_correlation()

# plotting all corrfuncs (from the experiment):
exp.plot_correlation_functions(y_field="avg_cf_cr", y_scale="log", ylim=Limits(5e1, 5e4))
exp.plot_correlation_functions(y_field="avg_cf_cr", ylim=Limits(5e1, 5e4))

# %% [markdown]
# Now let's try to use the cross-correlation as the afterpulsing:

# %%
ap_factor = 1.05  # 1

# calculate afterpulsing from cross-correlation
countrate_a = np.mean([countrate_pair.a for countrate_pair in CF_dict["AB"].countrate_list])
countrate_b = np.mean([countrate_pair.b for countrate_pair in CF_dict["AB"].countrate_list])
inherent_afterpulsing = (
    countrate_b
    * (exp.confocal.gate_width_ns / gate2_ns.interval())
    * CF_dict["AB"].avg_cf_cr
    / countrate_a
)

# load experiment
exp_xcorr_as_ap = SFCSExperiment(
    name="Free-Running 300 bp, X-Correlation of Fluorescent vs. White-Noise as Afterpulsing"
)
exp_xcorr_as_ap.load_experiment(
    confocal_template=DATA_PATH / confocal_template,
    should_plot=False,
    should_plot_meas=False,
    should_use_preprocessed=True,
    should_re_correlate=True,  # Need to re-process with ext. afterpulsing
    external_afterpulsing=inherent_afterpulsing * ap_factor,
)

# %% [markdown]
# Let's look at them together:

# %%
with Plotter(super_title=label, ylim=(-300, exp.confocal.cf["confocal"].g0 * 1.5)) as ax:
    exp.confocal.cf["confocal"].plot_correlation_function(
        parent_ax=ax, x_field="vt_um", plot_kwargs=dict(label="Regular")
    )
    exp_xcorr_as_ap.confocal.cf["confocal"].plot_correlation_function(
        parent_ax=ax, x_field="vt_um", plot_kwargs=dict(label="BA XCorr as Afterpulsing")
    )
    ax.legend()

with Plotter(
    super_title="Afterpulsing", x_scale="log", y_scale="linear", xlim=(1e-4, 1e0), ylim=(1e-2, 1e4)
) as ax:
    lag = exp.confocal.cf["confocal"].lag
    ax.plot(lag, exp.confocal.cf["confocal"].afterpulse, label="Halogen AutoCorr")
    ax.plot(CF_dict["AB"].lag, inherent_afterpulsing, label="X Corr")
    ax.legend()

# %% [markdown]
# Play sound when done:

# %%
Beep(4000, 300)
