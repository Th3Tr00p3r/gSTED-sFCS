# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
from pathlib import Path
from winsound import Beep
from contextlib import suppress

import numpy as np
import matplotlib as mpl
from IPython.core.debugger import set_trace

mpl.use("nbAgg")

###############################################
# Move to project root to easily import modules
###############################################

try:  # avoid changes if already set
    print("Working from: ", PROJECT_ROOT)
except NameError:
    try:  # running from Spyder
        PROJECT_ROOT = Path(__file__).resolve()
    except NameError:  # running as Jupyter Notebook
        #         PROJECT_ROOT = "D:\people\Idomic\gSTED-sFCS"
        PROJECT_ROOT = "D:\MEGA\BioPhysics_Lab\Optical_System\gSTEDsFCS"
    os.chdir(PROJECT_ROOT)
    print("Working from: ", PROJECT_ROOT)

from data_analysis.correlation_function import (
    SolutionSFCSExperiment,
    calculate_calibrated_afterpulse,
)
from utilities.display import Plotter, default_colors

#################################################
# Setting up data path and other global constants
#################################################

DATA_ROOT = Path("D:\OneDrive - post.bgu.ac.il\gSTED_sFCS_Data")  # Laptop/Lab PC (same path)
# DATA_ROOT = Path("D:\???")  # Oleg's

# %% [markdown]
# Choosing the data templates:

# %%
DATA_TYPE = "solution"
DATE = "30_06_2022"

# Use pre-processed measrements if available
# Warning - setting to False can cause errors, test on one file before using on many files
FORCE_ALL = True
# FORCE_ALL = False

# setting SHOULD_SAVE to True with 'FORCE_ALL = False' will save only new processed data
SHOULD_SAVE = True

FILES = "Use 1"
# FILES = "Use 1-10"
# FILES = "Use All"

data_label_kwargs = {
    "TEST": dict(
        date=DATE,
        confocal_template="bp300YOYOFresh_angular_exc_124351_*.pkl",
        file_selection=FILES,
        force_processing=False or FORCE_ALL,
    ),
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
# For each measurement file, seperate the byte data from the full_data part of the file_dict and recombine all to a single 1D ndarray:

# %%
import utilities.file_utilities as fu

file_path_template = data_label_kwargs["TEST"]["confocal_template"]

file_paths = fu.prepare_file_paths(Path(file_path_template), "Use All")
n_files = len(file_paths)
print(f"{file_path_template.name}: {n_files} files.")

byte_data_list = []
for file_path in file_paths:
    byte_data = fu.load_file_dict(file_path)["full_data"]["byte_data"]
    print(f"adding {byte_data.nbytes / 1e6:.2f} Mb to the list...")
    byte_data_list.append(byte_data)

total_byte_data = np.hstack(tuple(byte_data_list))
print(f"Total byte data: {total_byte_data.nbytes / 1e6:.2f} Mb.")

# create a new file (without overwriting any of the existing files) where the byte data is the total byte data:
file_dict = fu.load_file_dict(file_path)
file_dict["full_data"]["byte_data"] = total_byte_data
# and save it as a new file (adding 999 to index to be safe)
new_file_path = file_path.with_name(f"bp300YOYOFresh_angular_exc_124351_{n_files + 10}.pkl")
fu.save_object(file_dict, new_file_path)

# %% [markdown]
# now try loading just the one file containing all of the byte_data:

# %%
DATA_TYPE = "solution"
DATE = "30_06_2022"

FILES = f"Use {n_files + 10}"

data_label_kwargs = {
    "TEST2": dict(
        date=DATE,
        confocal_template="bp300YOYOFresh_angular_exc_124351_*.pkl",
        file_selection=FILES,
        force_processing=False or FORCE_ALL,
    ),
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

# %%
file_paths = fu.prepare_file_paths(Path(file_path_template), "Use All")
file_paths

# %% [markdown]
# Importing all needed data. Processing, correlating and averaging if no pre-processed measurement exist.

# %%
label = "TEST2"
exp = exp_dict[label]

exp.load_experiment(
    should_plot=True,
    should_re_correlate=FORCE_ALL,
    **data_label_kwargs[label],
)


# %%
data_label_kwargs

# %%
data_label_kwargs

# %% [markdown]
# Beep when done

# %%
Beep(4000, 1000)
