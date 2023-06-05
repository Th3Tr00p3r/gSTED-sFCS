# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# ## Locating and Copying Old Fluorescent Bead (FB) Measurements to a Single Folder for Testing

# %%
from pathlib import Path
import shutil
import numpy as np
import os

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

from utilities.display import Plotter
from utilities.file_utilities import (
    load_object,
    save_object,
)

# define main data directory
DATA_ROOT = Path("D:\OneDrive - post.bgu.ac.il\gSTED_sFCS_Data/")

# %% [markdown]
# List all paths for files in directories named "FB" in the main folder (`DATA_ROOT`)

# %%
OLD_DATA_ROOT = Path("D:\old\Google Drive\Dotan's old drive\Dotan TDC Measurments")

fb_paths = [path for path in OLD_DATA_ROOT.rglob("*.mat") if "fb" in str(path).lower()]

# TEST
print(f"Found {len(fb_paths)} files:\n")
print("\n".join([str(file_path) for file_path in fb_paths]))

# %% [markdown]
# Copy those files to a 'regular' date directory

# %%
# create the folder if necessary
destination_dir_path = DATA_ROOT / "24_05_2023" / "image"
destination_dir_path.mkdir(parents=True, exist_ok=True)

# # copy the files to new directory
for file_path in fb_paths:
    shutil.copy(file_path, destination_dir_path)
