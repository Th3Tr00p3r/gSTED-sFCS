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
# Try simple searching in `DATA_ROOT`

# %%
from datetime import datetime

# TODO: upgrade this to find templates containing multiple strings (list), use any/all etc., as needed.
SEARCH_STR = "YOYO"

# get all unique templates by using their .log files from all directories in DATA_ROOT. filter using the SEARCH_STR.
template_date_dict = {
    path.stem[:-2]: datetime.strptime(path.parent.parent.name, "%d_%m_%Y").date()
    for path in DATA_ROOT.rglob("*_1.pkl")
    if SEARCH_STR.lower() in str(path).lower()
}

# sort by date in reverse order (newest first)
template_date_dict = dict(
    sorted(template_date_dict.items(), key=lambda item: item[1], reverse=True)
)

# print the findings (date first, though the key is the template)
if template_date_dict:
    print(f"Found {len(template_date_dict)} matching templates:\n")
    print("\n".join([f"{date}: {template}" for template, date in template_date_dict.items()]))
else:
    print("No matches found!")
