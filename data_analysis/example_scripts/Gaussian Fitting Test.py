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

FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))
RESCALE_FACTOR = 10

# %% [markdown]
# Loading excitation beam image

# %%
IMG_DATA_ROOT = DATA_ROOT / "camera photos"

img = load_object(IMG_DATA_ROOT / "cam1_190922_223540.pkl")
# img = load_object(IMG_DATA_ROOT / "cam2_190922_231314.pkl")

# %% [markdown]
# Convert to grayscale

# %%
gs_img = img.mean(axis=2)

# %% [markdown]
# checkout with Pillow

# %%
img_test = PIL.Image.fromarray(img, mode="RGB")
gs_img = img_test.convert("L")
display(gs_img)

# %% [markdown]
# Attempt 2D Gaussian fitting:

# %%
gs_img_arr = np.asarray(gs_img)

plt.imshow(gs_img_arr)
plt.show()

fp = fit_2d_gaussian_to_image(gs_img_arr)
x0, y0, sigma_x, sigma_y = fp.beta["x0"], fp.beta["y0"], fp.beta["sigma_x"], fp.beta["sigma_y"]

# %%
x0

# %%
PIXEL_SIZE_UM = 3.6
sigma_mm = np.mean([sigma_x, sigma_y]) * FWHM_FACTOR * PIXEL_SIZE_UM * 1e-3
sigma_mm_err = np.std([sigma_x, sigma_y]) * FWHM_FACTOR * PIXEL_SIZE_UM * 1e-3

print(f"FWHM width determined to be {sigma_mm:.3f} +/- {sigma_mm_err:.3f} nm")

# %% [markdown]
# Attempt the same fitting with a gold beads image (much smaller in resolutions)

# %%
image_meas = ImageSFCSMeasurement()
image_meas.generate_ci_image_stack_data(file_path=IMG_DATA_ROOT / "gb_exc_XY_165607.pkl")

# get the center plane image, in "forward"
gb_image = image_meas.ci_image_data.get_image("forward")

# %%
gb_image.shape

# %%
fp = fit_2d_gaussian_to_image(gb_image)
x0, y0, sigma_x, sigma_y = fp.beta["x0"], fp.beta["y0"], fp.beta["sigma_x"], fp.beta["sigma_y"]

# %%
PIXEL_SIZE_UM = 1 / 80  # 1 um scan (?) 80 pixels
sigma_nm = np.mean([sigma_x, sigma_y]) * FWHM_FACTOR * PIXEL_SIZE_UM * 1e3
sigma_nm_err = np.std([sigma_x, sigma_y]) * FWHM_FACTOR * PIXEL_SIZE_UM * 1e3

print(f"FWHM width determined to be {sigma_nm:.3f} +/- {sigma_nm_err:.3f} nm")

# %%
ellipse = Ellipse(
    xy=(x0 * RESCALE_FACTOR + crop_delta_x, y0 * RESCALE_FACTOR + crop_delta_y),
    width=sigma_y * FWHM_FACTOR * RESCALE_FACTOR,
    height=sigma_x * FWHM_FACTOR * RESCALE_FACTOR,
    angle=phi,
)
ellipse.set_facecolor((0, 0, 0, 0))
ellipse.set_edgecolor("red")

fig, ax = plt.subplots()
ax.imshow(gb_image)
ax.add_artist(ellipse)
plt.show()

# %% [markdown]
# So TDC GB image works, beam CCD image doesn't... Why?
#
# Let's try lowering the resolution of the CCD image:

# %%
width, height = gs_img.size
RESCALE_FACTOR = 0.1

resized_gs_img = gs_img.resize(
    (round(width * RESCALE_FACTOR), round(height * RESCALE_FACTOR)), resample=PIL.Image.LANCZOS
)

plt.imshow(resized_gs_img)
plt.show()

# %% [markdown]
# now try fitting the resized image:

# %%
# converting to Numpy array
resized_gs_img_arr = np.asarray(resized_gs_img)

# fitting
fp = fit_2d_gaussian_to_image(resized_gs_img_arr)
x0, y0, sigma_x, sigma_y = fp.beta["x0"], fp.beta["y0"], fp.beta["sigma_x"], fp.beta["sigma_y"]

# printing
PIXEL_SIZE_UM = 3.6
sigma_mm = np.mean([sigma_x, sigma_y]) * FWHM_FACTOR * PIXEL_SIZE_UM * 1e-3
sigma_mm_err = np.std([sigma_x, sigma_y]) * FWHM_FACTOR * PIXEL_SIZE_UM * 1e-3

print(f"center is located at {(x0, y0)} pxls")
print(f"FWHM width determined to be {sigma_mm:.3f} +/- {sigma_mm_err:.3f} mm")

# plotting
ellipse = Ellipse(
    xy=(x0 * RESCALE_FACTOR + crop_delta_x, y0 * RESCALE_FACTOR + crop_delta_y),
    width=sigma_y * FWHM_FACTOR * RESCALE_FACTOR,
    height=sigma_x * FWHM_FACTOR * RESCALE_FACTOR,
    angle=phi,
)
ellipse.set_facecolor((0, 0, 0, 0))
ellipse.set_edgecolor("red")

fig, ax = plt.subplots()
ax.imshow(resized_gs_img_arr)
ax.add_artist(ellipse)
plt.show()

# %% [markdown]
# Seems like it might be an issue with the image not being rectangular? Let's try cropping before resizing:

# %%
width, height = gs_img.size

# cropping to square - assuming beam is centered, takes the same amount from both sides of the longer dimension
dim_diff = abs(width - height)
if width > height:
    crop_dims = (dim_diff / 2, 0, width - dim_diff / 2, height)
    width = height
else:
    crop_dims = (0, dim_diff / 2, width, height - dim_diff / 2)
    height = width
cropped_gs_img = gs_img.crop(crop_dims)

# resizing
resized_cropped_gs_img = cropped_gs_img.resize(
    (round(width * RESCALE_FACTOR), round(height * RESCALE_FACTOR)), resample=PIL.Image.LANCZOS
)

# plotting
plt.imshow(resized_cropped_gs_img)
plt.show()

# %% [markdown]
# Fitting one more time:

# %%
# converting to Numpy array
resized_cropped_gs_img_arr = np.asarray(resized_cropped_gs_img)

# fitting
fp = fit_2d_gaussian_to_image(resized_cropped_gs_img_arr)
x0, y0, sigma_x, sigma_y = fp.beta["x0"], fp.beta["y0"], fp.beta["sigma_x"], fp.beta["sigma_y"]

# printing
PIXEL_SIZE_UM = 3.6 / RESCALE_FACTOR
sigma_mm = np.mean([sigma_x, sigma_y]) * FWHM_FACTOR * PIXEL_SIZE_UM * 1e-3
sigma_mm_err = np.std([sigma_x, sigma_y]) * FWHM_FACTOR * PIXEL_SIZE_UM * 1e-3

print(f"center is located at {(x0, y0)} pxls")
print(f"FWHM width determined to be {sigma_mm:.3f} +/- {sigma_mm_err:.3f} mm")

# plotting
ellipse = Ellipse(
    xy=(x0 * RESCALE_FACTOR + crop_delta_x, y0 * RESCALE_FACTOR + crop_delta_y),
    width=sigma_y * FWHM_FACTOR * RESCALE_FACTOR,
    height=sigma_x * FWHM_FACTOR * RESCALE_FACTOR,
    angle=phi,
)
ellipse.set_facecolor((0, 0, 0, 0))
ellipse.set_edgecolor("red")

fig, ax = plt.subplots()
ax.imshow(resized_cropped_gs_img_arr)
ax.add_artist(ellipse)
plt.show()

# %% [markdown]
# So the non-square image was the culprit... Let's see if we can get a better estimate without resizing:

# %%
# converting to Numpy array
cropped_gs_img_arr = np.asarray(cropped_gs_img)

# fitting
fp = fit_2d_gaussian_to_image(cropped_gs_img_arr)
x0, y0, sigma_x, sigma_y = fp.beta["x0"], fp.beta["y0"], fp.beta["sigma_x"], fp.beta["sigma_y"]

# printing
PIXEL_SIZE_UM = 3.6
sigma_mm = np.mean([sigma_x, sigma_y]) * FWHM_FACTOR * PIXEL_SIZE_UM * 1e-3
sigma_mm_err = np.std([sigma_x, sigma_y]) * FWHM_FACTOR * PIXEL_SIZE_UM * 1e-3

print(f"center is located at {(x0, y0)} pxls")
print(f"FWHM width determined to be {sigma_mm:.3f} +/- {sigma_mm_err:.3f} mm")

# plotting
ellipse = Ellipse(
    xy=(x0 * RESCALE_FACTOR + crop_delta_x, y0 * RESCALE_FACTOR + crop_delta_y),
    width=sigma_y * FWHM_FACTOR * RESCALE_FACTOR,
    height=sigma_x * FWHM_FACTOR * RESCALE_FACTOR,
    angle=phi,
)
ellipse.set_facecolor((0, 0, 0, 0))
ellipse.set_edgecolor("red")

fig, ax = plt.subplots()
ax.imshow(cropped_gs_img_arr)
ax.add_artist(ellipse)
plt.show()

# %%
dir(ellipse)
# ellipse.get_center()
ellipse.get_width()

# %% [markdown]
# So, pretty much the same estimate in a much shorter calculation! Let's use resizing!
#
# Lastly, let's take a look at how the fit "sits" with the data itself (smoothed, perhaps?) and calculate chi_squared:

# %%
