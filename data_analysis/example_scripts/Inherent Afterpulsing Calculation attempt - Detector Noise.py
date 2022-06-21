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

#################################################
# Setting up data path and other global constants
#################################################

EPS = sys.float_info.epsilon

DATA_ROOT = Path("D:\OneDrive - post.bgu.ac.il\gSTED_sFCS_Data")  # Laptop/Lab PC (same path)
# DATA_ROOT = Path("D:\???")  # Oleg's
DATA_TYPE = "solution"

SHOULD_PLOT = True

# %% [markdown]
# ## Testing on new detector data...

# %% [markdown]
# Choosing the data

# %%
label = "Concentrated Plasmids"
DATA_DATE = "24_03_2022"
confocal_template = "conc_1uW_low_delay_angular_exc_151408_*.pkl"
file_selection = "Use 1-10"
# file_selection = "Use All"
gate1_ns, gate2_ns = Limits(3, 15), Limits(30, 42.5)
detector_gate_width_ns = 40

# label = "25X Diluted Plasmids"
# DATA_DATE = "03_04_2022"
# confocal_template = "conc_25xDiluted_20uW_59psGate_angular_exc_155134_*.pkl"
# file_selection = "Use All"
# gate1_ns, gate2_ns = Limits(3, 15), Limits(30, 42.2)
# detector_gate_width_ns = 40

# label = "250X Diluted Plasmids"
# DATA_DATE = "29_03_2022"
# confocal_template = "conc_250xDiluted_angular_exc_191600_*.pkl"
# file_selection = "Use All"
# gate1_ns, gate2_ns = Limits(1.25, 15), Limits(30, 79)
# detector_gate_width_ns = 100

# label = "300 bp ATTO (new detector)"
# DATA_DATE = "06_06_2022"
# confocal_template = "atto300bp_angular_exc_190337_*.pkl"
# file_selection = "Use All"
# gate1_ns, gate2_ns = Limits(3, 15), Limits(30, 90)
# detector_gate_width_ns = 100

# label = "300 bp YOYO (new detector)"
# DATA_DATE = "29_03_2022"
# confocal_template = "bp300_20uW_angular_exc_172325_*.pkl"
# file_selection = "Use All"
# gate1_ns, gate2_ns = Limits(3, 15), Limits(30, 90)
# detector_gate_width_ns = 100

# label = "ATTO static new detector"
# DATA_DATE = "13_03_2022"
# confocal_template = "atto_12uW_FR_static_exc_182414_*.pkl"
# file_selection = "Use All"
# gate1_ns, gate2_ns = Limits(3, 15), Limits(30, 90)
# detector_gate_width_ns = 100

DATA_PATH = DATA_ROOT / DATA_DATE / DATA_TYPE

# %% [markdown]
# importing the data:

# %%
# FORCE_PROCESSING = False
FORCE_PROCESSING = True

# load experiment
new_det_atto300bp_exp = SFCSExperiment(name=label)
new_det_atto300bp_exp.load_experiment(
    confocal_template=DATA_PATH / confocal_template,
    should_plot=True,
    force_processing=FORCE_PROCESSING,
    #     should_re_correlate=True,
    should_re_correlate=False,
    should_subtract_afterpulse=False,
    file_selection=file_selection,
)

# Show countrate
print(f"Count-Rate: {new_det_atto300bp_exp.confocal.avg_cnt_rate_khz} kHz")

# calibrate TDC
new_det_atto300bp_exp.calibrate_tdc(should_plot=True, force_processing=FORCE_PROCESSING)

# save processed data (to avoid re-processing)
if FORCE_PROCESSING:
    new_det_atto300bp_exp.save_processed_measurements()

# %% [markdown]
# # Testing a new type of afterpulsing calculation attempt (Oleg)

# %% [markdown]
# Get the white-noise calibration afterpulsing:

# %%
meas = new_det_atto300bp_exp.confocal

lag = meas.cf["confocal"].lag
pulse_period_ns = 100
p_cal_ap = calculate_afterpulse(lag) * detector_gate_width_ns / pulse_period_ns

# %% [markdown]
# Get the current standard of inherent afterpulsing, and the XCF_AB/BA objects for testing a new method of calculating afterpulsing:

# %%
p_inherent_ap, (XCF_AB, XCF_BA) = meas.calculate_inherent_afterpulsing(gate1_ns, gate2_ns)

print("rate A / rate B: ", XCF_AB.countrate_a / XCF_AB.countrate_b)

XCF_AB.average_correlation()
XCF_BA.average_correlation()

# %% [markdown]
# Plotting

# %%
G_AB = XCF_AB.avg_corrfunc
G_BA = XCF_BA.avg_corrfunc  # + EPS
G_new = (G_AB + 1) / (G_BA + 1) - 1

p_new_ap = G_new * pulse_period_ns * XCF_AB.countrate_b / gate2_ns.interval()

# plotting
with Plotter(
    super_title="Afterpulsing Comparison",
    xlim=(1e-4, 1e1),
    ylim=(0, 1e5),
    x_scale="log",
) as ax:

    ax.plot(lag, p_cal_ap, label="Calibrated Afterpulsing (white noise)")
    ax.plot(lag, unify_length(p_inherent_ap, len(lag)), label="Inherent Afterpulsing (AB-BA)")
    ax.plot(lag, unify_length(p_new_ap * 1, len(lag)), label="Test Afterpulsing")
    ax.legend()

# %% [markdown]
# Comparing the mean corrfuncs of G_AB+1, G_BA+1 and G_AA+1:

# %%
meas_cf = meas.cf["confocal"]

# plotting
with Plotter(
    super_title="XCorr Comparison",
    xlim=(1e-4, 1e1),
    ylim=(1, 1.1),
    x_scale="log",
) as ax:

    ax.plot(lag, G_AB + 1, label="G_AB + 1")
    ax.plot(lag, G_BA + 1, label="G_BA + 1")
    ax.plot(lag, meas_cf.avg_corrfunc + 1, label="G_AA + 1")
    ax.legend()


# %% [markdown]
# Preparing the signal in the same way:

# %%
cf_cr = meas_cf.avg_cf_cr
cf_cr_new = (cf_cr + XCF_AB.countrate_a) / (G_BA + 1) - XCF_AB.countrate_a

# %% [markdown]
# Plotting:

# %%
with Plotter(
    super_title="Signal Comparison",
    xlim=(1e-4, 1e1),
    ylim=(-500, meas_cf.g0 * 1.3),
    x_scale="log",
) as ax:

    ax.plot(lag, cf_cr, label="CF_CR")
    ax.plot(lag, cf_cr_new, label="New CF_CR")
    ax.legend()

# %% [markdown]
# Subtracting the afterpulsing:

# %%
with Plotter(
    super_title="Signal Comparison",
    xlim=(1e-4, 1e1),
    ylim=(-1000, 1000),
    x_scale="log",
) as ax:

    ax.plot(lag, cf_cr - p_cal_ap, label="Old CF_CR - calibrated afterpulsing")
    ax.plot(lag, cf_cr_new - p_new_ap, label="New CF_CR - new Afterpulsing")
    ax.legend()

# %% [markdown]
# Play sound when done:

# %%
delta_t = 250
freq_range = (3000, 8000)
n_beeps = 5
[Beep(int(f), delta_t) for f in np.linspace(*freq_range, n_beeps)]
