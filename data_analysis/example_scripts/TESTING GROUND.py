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
# ## General tests

# %%
import numpy as np
from typing import Tuple


def unify_length(arr: np.ndarray, out_len: Tuple[int, ...]) -> np.ndarray:
    """Either trims or zero-pads the right edge of an ndarray to match 'out_len' (2nd axis)"""

    # assume >=2D array
    try:
        if (arr_len := arr.shape[1]) >= out_len:
            return arr[:, :out_len]
        else:
            pad_width = tuple(
                [(0, 0), (0, out_len - arr_len)] + [(0, 0) for _ in range(len(arr.shape) - 2)]
            )
            return np.pad(arr, pad_width)

    # 1D array
    except IndexError:
        if (arr_len := arr.size) >= out_len:
            return arr[:out_len]
        else:
            pad_width = (0, out_len - arr_len)
            return np.pad(arr, pad_width)


test_1d_arr = np.arange(16)
test_2d_arr = np.arange(16).reshape(4, 4)

print("test_1d_arr:\n", test_1d_arr, "\n\ntest_2d_arr:\n", test_2d_arr)
print()

# %%
unify_length(test_1d_arr, 20)

# %%
unify_length(test_2d_arr, 15)

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
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import PIL

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

from data_analysis.correlation_function import ImageSFCSMeasurement
from utilities.display import Plotter
from utilities.file_utilities import load_object
from utilities.fit_tools import fit_2d_gaussian_to_image

#################################################
# Setting up data path and other global constants
#################################################

DATA_ROOT = Path("D:\OneDrive - post.bgu.ac.il\gSTED_sFCS_Data")  # Laptop/Lab PC (same path)
# DATA_ROOT = Path("D:\???")  # Oleg's
DATA_TYPE = "solution"


# %% [markdown]
# Attempting to get argument names from function:

# %%
def _get_args_dict(fn, args, kwargs):
    args_names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    return {**dict(zip(args_names, args)), **kwargs}


# %%
fn = load_object
fn.__code__.co_varnames[: fn.__code__.co_argcount]

# %%
a = dict(dog=1, cat=2)

# %%
[print(el) for el in a.values()]
