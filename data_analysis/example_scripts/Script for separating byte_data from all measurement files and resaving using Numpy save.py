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
# # WARNINGS:
# * DO NOT 'Run All'!
# * Please run cell-by-cell and read carefully the output of each cell.
# * Must be run from a PC which has all data on board (not in cloud), i.e. the Lab PC
# * THIS IS A ONE-TIME SCRIPT - but can safely be run on already converted data (KeyError will be caught)
# * Does not currently seem to work with old .mat files. A fallback was created for them in correlation_function.py

# %% [markdown]
# # FOR SOME REASON STATIC MEASUREMENT'S FILE_DICT FILE SIZE STAYS LARGE - CHECK IT OUT! (CAN BE DONE WITHIN GUI WITH DEBUGGER)

# %%
from pathlib import Path
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
DATA_ROOT = Path(
    "D:\OneDrive - post.bgu.ac.il\gSTED_sFCS_Data/"
)  # NOTE - WARNING - actual data root!!!
# DATA_ROOT = Path("D:/OneDrive - post.bgu.ac.il/gSTED_sFCS_Data/30_06_2022") # NOTE - warning, actual data!
# DATA_ROOT = Path("C:/Users/Idom/Desktop/TEST_DATA/") # TEST DATA, FOR TESTING!

print(
    "\n\n===============WARNING===============\n\n          The next cell will\n             IRREVERSIBLY\n  change measurement data in 'DATA_ROOT':\n\n",
    DATA_ROOT,
    "\n\n          and ANY SUB-PATHS!\n\n===============WARNING===============\n",
)

# %% [markdown]
# Get all paths to file_dict files and print them for testing:

# %%
# create an iterator over all 'solution' data folders

# for working from within actual data root (parent of date folders)
solution_dir_paths = [
    item / "solution" for item in DATA_ROOT.iterdir() if (item / "solution").is_dir()
]

# # from that, create an iterator over all measurement files
# file_paths = [
#     file_path for solution_dir_path in solution_dir_paths
#     for file_path in solution_dir_path.iterdir()
#     if file_path.suffix == ".pkl"
# ]

print(f"{len(solution_dir_paths)} 'date' folders:")
for dir_path in solution_dir_paths:
    print(dir_path)

# %%
print(f"Seperating byte_data from file_dicts in {len(solution_dir_paths)} folders")
for idx, solution_dir_path in enumerate(solution_dir_paths):

    print(f"{idx} - {solution_dir_path}... ", end="")

    # get all .pkl files in a 'solution' directory
    file_paths = [
        file_path for file_path in solution_dir_path.iterdir() if file_path.suffix == ".pkl"
    ]

    # loop over all paths, loading each file_dict using load_object,
    # Numpy-saving byte_data separately with the same name + byte_data.npy
    # removing byte_data key from file_dict, then resaving using save_object (compressed with blosc).
    for file_path in file_paths:
        converted_file_path = file_path.with_name(file_path.name.replace(".pkl", "_byte_data.npy"))
        if not converted_file_path.is_file():
            try:
                file_dict = load_object(file_path)
            except Exception as exc:
                print(
                    f" Exception [{exc}] raised for file path {file_path}. Skipping, but should be checked out! ",
                    end="",
                )
                continue
            try:
                byte_data = file_dict["full_data"].pop("byte_data")
            except TypeError as exc:
                # file dict is actually a list??
                file_dict = file_dict[0]
                byte_data = file_dict["full_data"].pop("byte_data")
                np.save(
                    file_path.with_name(file_path.name.replace(".pkl", "_byte_data.npy")),
                    byte_data,
                    allow_pickle=False,
                    fix_imports=False,
                )
                save_object(file_dict, file_path, "blosc")
                print("V", end="")
            except KeyError:
                # data key is "data" and not "byte_data"
                byte_data = file_dict["full_data"].pop("data")
                np.save(
                    file_path.with_name(file_path.name.replace(".pkl", "_byte_data.npy")),
                    byte_data,
                    allow_pickle=False,
                    fix_imports=False,
                )
                save_object(file_dict, file_path, "blosc")
                print("V", end="")
            else:
                np.save(
                    file_path.with_name(file_path.name.replace(".pkl", "_byte_data.npy")),
                    byte_data,
                    allow_pickle=False,
                    fix_imports=False,
                )
                save_object(file_dict, file_path, "blosc")
                print("V", end="")

    print(" Done.")

print("\n\nOperation finished!")
