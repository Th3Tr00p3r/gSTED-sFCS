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
# Imports

# %%
######################################
# importing core and 3rd-party modules
######################################

import os
from pathlib import Path

mpl.use("nbAgg")
import numpy as np

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
from utilities.helper import Limits, fourier_transform_1d, extrapolate_over_noise

# %% [markdown]
# Testing our Fourier transform and its inverse by performing a round-trip transform on a Gaussian function:

# %%
# define Gaussian on positive axis:
a = 7
sigma = 137  # say, in seconds
t = np.arange(int(1e4))  # say, in seconds
t = np.linspace(0, 1e4, 2 ** 10)  # say, in seconds
ft = a * np.exp(-((t / sigma) ** 2))

# show it
with Plotter(
    super_title="Gaussian",
    xlabel="$t$ ($s$)",
    ylabel="$f(t)$ (Normalized)",
    xlim=(-1e-2, sigma * 4),
) as ax:
    ax.plot(t, ft, "o")

###################
# Fourier transform
###################

r, fr_interp, w, fw = fourier_transform_1d(
    t,
    ft,
    should_plot=True,
)

###################
# Inverse transform
###################

w_post, fw_interp, t_post, ft_post = fourier_transform_1d(
    w,
    fw,
    should_inverse=True,
    is_input_symmetric=True,
    should_plot=True,
)

######################
# Comparing round-trip
######################

with Plotter(
    super_title="Round-Trip Fourier Transform\n Compared to Original",
    xlabel="$t$ ($s$)",
    ylabel="$f(t)$ (Normalized)",
    xlim=(-sigma * 5, sigma * 5),
    x_scale="linear",
) as ax:
    ax.plot(t, ft, label="Original Gaussian")
    ax.plot(t_post, np.real(ft_post), label="Round-Trip FT Gaussian (real)")
    ax.plot(t_post, np.imag(ft_post), label="Round-Trip FT Gaussian (imaginary)")
    ax.legend()

# %% [markdown]
# Testing Gaussian interpolation/extrapolation on Gaussian:

# %%
x_lims = Limits(20, 600)
y_lims = Limits(1, 4)

gauss_interp = extrapolate_over_noise(
    t,
    ft,
    n_bins=2 ** 17,
    x_lims=x_lims,
    y_lims=y_lims,
    n_robust=2,
    interp_type="gaussian",
    should_plot=True,
)
