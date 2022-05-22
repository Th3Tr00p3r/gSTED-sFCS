# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
# import native/external packages
import pickle
import re
from pathlib import Path
from pprint import pprint as pp
from winsound import Beep

import matplotlib as mpl
import numpy as np
from IPython.core.debugger import set_trace
from matplotlib import pyplot as plt

# Import project modules
from data_analysis.correlation_function import (SFCSExperiment,
                                                SolutionSFCSMeasurement)
from utilities.display import Plotter
from utilities.file_utilities import (default_system_info,
                                      save_processed_solution_meas)
from utilities.helper import Limits

mpl.use("nbAgg")

# Move to project root to easily import modules
# PROJECT_ROOT = Path("D:\people\Idomic\gSTED-sFCS") # Lab PC
PROJECT_ROOT = Path("D:\MEGA\BioPhysics_Lab\Optical_System\gSTEDsFCS")  # Laptop
os.chdir(PROJECT_ROOT)


# Define other global constants
DATA_ROOT = Path("D:\OneDrive - post.bgu.ac.il\gSTED_sFCS_Data")  # Laptop/Lab PC (same path)
DATA_TYPE = "solution"
DATA_PATH = DATA_ROOT / "29_03_2022" / DATA_TYPE

# SHOULD_FORCE_PROCESSING = False
SHOULD_FORCE_PROCESSING = True

# %% [markdown]
# Load Calibration Data

# %%
confocal_template = DATA_PATH / "bp300_20uW_angular_exc_172325_*.pkl"
sted_template = DATA_PATH / "bp300_20uW_200mW_angular_sted_174126_*.pkl"

# load experiment
cal_exp = SFCSExperiment(name="300 bp")
cal_exp.load_experiment(
    confocal_template=confocal_template,
    sted_template=sted_template,
    should_plot=True,
    should_use_preprocessed=not SHOULD_FORCE_PROCESSING,
    file_selection="Use All",
)

# save processed data (to avoid re-processing)
if SHOULD_FORCE_PROCESSING:
    cal_exp.save_processed_measurements()

Beep(4000, 1000)

# %% [markdown]
# Load Sample Data

# %%
confocal_template = DATA_PATH / "conc_250xDiluted_angular_exc_191600_*.pkl"
sted_template = DATA_PATH / "conc_250xDiluted_angular_sted_193229_*.pkl"

# load experiment
sample_exp = SFCSExperiment(name="250x Diluted Sample")
sample_exp.load_experiment(
    confocal_template=confocal_template,
    sted_template=sted_template,
    should_plot=True,
    should_use_preprocessed=not SHOULD_FORCE_PROCESSING,
    roi_selection="auto",
    file_selection="Use All",
)

# save processed data (to avoid re-processing)
if SHOULD_FORCE_PROCESSING:
    cal_exp.save_processed_measurements()

Beep(5000, 1000)

# %%
sample_exp.confocal.__dict__

# %% [markdown]
# Print the count-rates (should add to 'load_experiment()' method)

# %%
sample_exp.confocal.__dict__.keys()
sample_exp.confocal.duration_min

# %%
print("300 bp YOYO Calibration Experiment:")
print(f"Average confocal count-rate: {cal_exp.confocal.avg_cnt_rate_khz:.2f} kHz")
print(f"Average STED count-rate: {cal_exp.sted.avg_cnt_rate_khz:.2f} kHz")

print("\n9.3 kbp YOYO Sample Experiment:")
print(f"Average confocal count-rate: {sample_exp.confocal.avg_cnt_rate_khz:.2f} kHz")
print(f"Average STED count-rate: {sample_exp.sted.avg_cnt_rate_khz:.2f} kHz")

# %% [markdown]
# Calibrate TDC and compare lifetimes:

# %%
if not hasattr(cal_exp.confocal, "tdc_calib"):

    # TDC calibration
    cal_exp.calibrate_tdc()
    sample_exp.calibrate_tdc()

    # Saving
    cal_exp.save_processed_measurements()
    sample_exp.save_processed_measurements()

else:
    print("TDC already calibrated. Skipping.")

# Compare lifetimes
cal_exp.compare_lifetimes()
sample_exp.compare_lifetimes()

Beep(6000, 1000)

# %% [markdown]
# Get lifetime parameters:

# %%
mpl.use("Qt5Agg")
cal_exp.get_lifetime_parameters()
sample_exp.get_lifetime_parameters()
mpl.use("nbAgg")

print("Lifetime Parameters:\n")
print("Calibration:")
pp(cal_exp.lifetime_params.__dict__)
print("\nSample:")
pp(sample_exp.lifetime_params.__dict__)

Beep(7000, 1000)

# %% [markdown]
# Gating

# %%
gate_list = [(6, 20), (8, 20)]
cal_exp.add_gates(gate_list)
sample_exp.add_gates(gate_list)

Beep(8000, 1000)

# %% [markdown]
# Structure Factor Testing:

# %%
n_robust = 3
g_min = 0.1

cal_exp.calculate_structure_factors(g_min=g_min, interp_pnts=n_interp_points)
sample_exp.calculate_structure_factors(g_min=g_min, n_robust=n_robust)

Beep(9000, 1000)

# %% [markdown]
# Now divide the sample $S(q)$ by the calibration's?
