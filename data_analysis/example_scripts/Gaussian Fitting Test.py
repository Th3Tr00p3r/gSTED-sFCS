# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
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
# Loading excitation beam image

# %%
from utilities.file_utilities import load_object

IMG_DATA_ROOT = DATA_ROOT / "camera photos"

# img = load_object(IMG_DATA_ROOT / "cam1_190922_223540.pkl")
img = load_object(IMG_DATA_ROOT / "cam2_190922_231314.pkl")

# %% [markdown]
# Convert to grayscale

# %%
gs_img = img.mean(axis=2)

# %% [markdown]
# checkout with Pillow

# %%
import PIL

img_test = PIL.Image.fromarray(img, mode="RGB")
gs_img = img_test.convert("L")
display(gs_img)

# %%
np.array(img_test).max()

# %% [markdown]
# Attempt to use the maximum channel:

# %%
PIL.Image.fromarray(img.max(axis=2))

# %% [markdown]
# attempt to fit
