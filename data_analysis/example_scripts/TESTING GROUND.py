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
# ## General tests

# %%
import numpy as np
from typing import Tuple


def unify_length(arr: np.ndarray, req_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Returns a either a zero-padded array or a trimmed one, according to the requested shape.
    passing None as one of the shape axes ignores that axis.
    """

    if (dim_arr := len(arr.shape)) != len(req_shape):
        raise ValueError(f"Dimensionallities do not match {len(arr.shape)}, {len(req_shape)}")

    out_arr = np.copy(arr)

    # assume >=2D array
    try:
        for ax, req_length in enumerate(req_shape):
            if req_length is None:
                # no change required
                continue
            if (arr_len := arr.shape[ax]) >= req_shape[ax]:
                out_arr = out_arr[:, :req_length]
            else:
                pad_width = tuple(
                    [(0, 0)] * ax + [(0, req_length - arr_len)] + [(0, 0)] * (dim_arr - (ax + 1))
                )
                out_arr = np.pad(out_arr, pad_width)

        return out_arr

    # 1D array
    except IndexError:
        out_length = req_shape[0]
        if (arr_len := arr.size) >= out_length:
            return arr[:out_length]
        else:
            return np.pad(arr, (0, out_length - arr_len))


test_1d_arr = np.arange(16)
test_2d_arr = np.arange(16).reshape(4, 4)

print("test_1d_arr:\n", test_1d_arr, "\n\ntest_2d_arr:\n", test_2d_arr)
print()

# %%
unify_length(test_1d_arr, (50,))

# %%
unify_length(test_2d_arr, (None, None))

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
