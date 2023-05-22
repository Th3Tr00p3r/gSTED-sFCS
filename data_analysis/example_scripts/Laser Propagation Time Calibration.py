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
# Here I derive a system-dependant calibration value for the delay time needed (in nano-seconds) in order to syncronize a detector-gated fluorescense pulse with the FPGA/laser clock.
#
# This calibrated value is used to know in advance the lower detector-gate.

# %%
# import native/external packages
import pickle
import numpy as np
import matplotlib as mpl

mpl.use("nbAgg")
from matplotlib import pyplot as plt
from pathlib import Path
import os
import re
from winsound import Beep
from IPython.core.debugger import set_trace

# Move to project root to easily import modules
PROJECT_ROOT = Path("D:\people\Idomic\gSTED-sFCS")  # Lab PC
# PROJECT_ROOT = Path("D:\MEGA\BioPhysics_Lab\Optical_System\gSTEDsFCS") # Laptop
os.chdir(PROJECT_ROOT)

# Import project modules
from data_analysis.correlation_function import SolutionSFCSMeasurement, SolutionSFCSExperiment
from data_analysis.data_processing import TDCCalibration
from utilities.display import Plotter
from utilities.file_utilities import default_system_info, load_object, save_object
from utilities.helper import Gate

# Define other global constants
DATA_ROOT = Path("D:\OneDrive - post.bgu.ac.il\gSTED_sFCS_Data")  # Laptop/Lab PC (same path)
DATA_TYPE = "solution"

FORCE_PROCESSING = False
# FORCE_PROCESSING = True

# %% [markdown]
# Load Data

# %%
DATA_DATE = "20_03_2022"
DATA_PATH = DATA_ROOT / DATA_DATE / DATA_TYPE

confocal_template = "mirror_100uW_wfilter_static_exc_143208_*.pkl"

# load experiment
laser_exp = SolutionSFCSExperiment(name="Laser Propagation Time")
laser_exp.load_experiment(
    confocal_template=DATA_PATH / confocal_template,
    should_plot=True,
    should_use_preprocessed=not FORCE_PROCESSING,
)

# %% [markdown]
# First we calibrate the laser sample TDC:

# %%
laser_exp.calibrate_tdc()
laser_tdc_calib = laser_exp.confocal.tdc_calib


# %% [markdown]
# Let us define a simple function for getting the pulse time derived from a TDC calibration:

# %%
def get_pulse_time_ns(tdc_calib: TDCCalibration) -> float:
    """Doc."""

    max_bin_idx = np.nanargmax(tdc_calib.all_hist_norm)
    return tdc_calib.t_hist[max_bin_idx]


print(f"laser pulse time: {get_pulse_time_ns(laser_tdc_calib):.2f} ns")

# %% [markdown]
# Judging from the upper-right graph (coarse bins), the fluoresence pulse arrives approximately $16\cdot2.5~ns=40~ns$ after the laser pulse.
# I need to think of a clean way of extracting the fluorescence pulse time from the histogram (or from some predecessor) and this way I could calculate the effective delay automatically from the TDC calibration, given the value I found here (subtracting the laser pulse time from the gated pulse time).
# I need to make a few measurements to test.

# %% [markdown]
# Now let's load up some free-running and gated data for testing:

# %%
# load experiments

DATA_DATE = "06_04_2022"
DATA_PATH = DATA_ROOT / DATA_DATE / DATA_TYPE

label, confocal_template = ("ATTO FR", "atto_FR_angular_exc_141224_*.pkl")
fr_exp = SolutionSFCSExperiment(name=label)
fr_exp.load_experiment(
    confocal_template=DATA_PATH / confocal_template,
    should_fix_shift=True,
    should_plot=True,
    should_use_preprocessed=not FORCE_PROCESSING,
)

DATA_DATE = "13_04_2022"
DATA_PATH = DATA_ROOT / DATA_DATE / DATA_TYPE

label, confocal_template = ("ATTO Gated (56 ns delay)", "ATTO_Gated7ns_angular_exc_110755_*.pkl")
gt_exp = SolutionSFCSExperiment(name=label)
gt_exp.load_experiment(
    confocal_template=DATA_PATH / confocal_template,
    should_fix_shift=True,
    should_plot=True,
    should_use_preprocessed=not FORCE_PROCESSING,
)

# %% [markdown]
# Now let's try to calibrate by syncing the coarse time to the laser pulse:

# %%
fr_exp.calibrate_tdc(sync_coarse_time_to=laser_tdc_calib)
fr_tdc_calib = fr_exp.confocal.tdc_calib
print(f"Free-running fluorescent pulse time: {get_pulse_time_ns(fr_tdc_calib):.2f} ns")

gt_exp.calibrate_tdc(sync_coarse_time_to=laser_tdc_calib)
gt_tdc_calib = gt_exp.confocal.tdc_calib
print(f"Free-running fluorescent pulse time: {get_pulse_time_ns(gt_tdc_calib):.2f} ns")

# %%
fr_lower_gate = get_pulse_time_ns(fr_tdc_calib) - get_pulse_time_ns(laser_tdc_calib)
gt_lower_gate = get_pulse_time_ns(gt_tdc_calib) - get_pulse_time_ns(laser_tdc_calib)

# %% [markdown]
# In order to get the actual lower gate, we need to remove the laser pulse time.
#
# Now we attempt to get the effective detector gate:

# %%
synced_meas = gt_exp.confocal
synced_tdc_calib = synced_meas.tdc_calib

lower_gate_ns = get_pulse_time_ns(synced_tdc_calib) - get_pulse_time_ns(laser_tdc_calib)
gate_width_ns = synced_meas.detector_settings["gate_width_ns"]
detector_gate_ns = Gate(lower_gate_ns, lower_gate_ns + gate_width_ns)
print(f"detector gate: {detector_gate_ns}")

# %% [markdown]
# Assuming this calculation is correct, I can also use it to calibrate the delayer -
# Since the set delay is known, and the actual lower gate is known, we can simply subtract the it from the delay and this will give the delay exactly syncs the laser and fluorescence pulse:

# %%
delayer_settings = synced_meas.delayer_settings
total_delay = delayer_settings["pulsewidth_ns"] + delayer_settings["delay_ps"] * 1e-3

sync_delay = total_delay - detector_gate_ns.lower
print(f"sync_delay: {sync_delay} ns")

# %%
synced_meas.detector_settings

# %% [markdown]
# This can used to know in advance the approximate gate given the set delay.
# A more robust way would be to use the laser tdc calibration, but this would force TDC calibration before doing correlations (which may not be so bad).
# I'll try using the 'sync_delay' as a pre-measurement calibrated estimation of the detector gating, and calibrate the TDC before correlating to know the "actual" detector gating.
