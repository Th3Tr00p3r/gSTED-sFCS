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
# Here I take a look at the difference between gating the same amount using post-measurement TDC gating and using real-time gating with the FastGatedSPAD

# %% [markdown]
# - Import core and 3rd party modules
# - Move current working directory to project root (if needed) and import project modules
# - Set data paths and other constants

# %%
######################################
# importing core and 3rd-party modules
######################################

import os
import sys
import pickle
import re
import scipy
from pathlib import Path
from winsound import Beep
from contextlib import suppress
from copy import deepcopy
from types import SimpleNamespace

import matplotlib as mpl

mpl.use("nbAgg")
import numpy as np
from IPython.core.debugger import set_trace
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft

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
    CorrFunc,
    SFCSExperiment,
    SolutionSFCSMeasurement,
    calculate_afterpulse,
)
from data_analysis.software_correlator import CorrelatorType, SoftwareCorrelator
from utilities.display import Plotter, get_gradient_colormap
from utilities.file_utilities import (
    default_system_info,
    load_mat,
    load_object,
    save_object,
    save_processed_solution_meas,
)
from utilities.helper import Limits, fourier_transform_1d, extrapolate_over_noise, unify_length
from utilities.fit_tools import curve_fit_lims, FitParams

#################################################
# Setting up data path and other global constants
#################################################

EPS = sys.float_info.epsilon

DATA_ROOT = Path("D:\OneDrive - post.bgu.ac.il\gSTED_sFCS_Data")  # Laptop/Lab PC (same path)
# DATA_ROOT = Path("D:\???")  # Oleg's
DATA_TYPE = "solution"

SHOULD_PLOT = True

# %% [markdown]
# Loading the data:

# %%
data_label_dict = {
    "300 bp ATTO exc FR": SimpleNamespace(
        date="03_07_2022",
        template="bp300ATTO_20uW_angular_exc_153213_*.pkl",
        file_selection="Use All",
        force_processing=False,
        should_use_inherent_afterpulsing=False,
    ),
    "300 bp ATTO STED FR": SimpleNamespace(
        date="03_07_2022",
        template="bp300ATTO_20uW_angular_sted_161127_*.pkl",
        file_selection="Use All",
        force_processing=False,
        should_use_inherent_afterpulsing=False,
    ),
    "300 bp ATTO STED 3 ns Detector Gating": SimpleNamespace(
        date="03_07_2022",
        template="bp300ATTO_20uW_Gated3ns_angular_sted_171540_*.pkl",
        file_selection="Use All",
        force_processing=False,
        should_use_inherent_afterpulsing=True,
    ),
}

confocal_label = "300 bp ATTO exc FR"

sted_labels = ["300 bp ATTO STED FR", "300 bp ATTO STED 3 ns Detector Gating"]

data_labels = list(data_label_dict.keys())

n_meas = len(data_labels)

label_template_paths_dict = {
    label: DATA_ROOT / data.date / DATA_TYPE / data.template
    for label, data in zip(data_labels, data_label_dict.values())
}

halogen_exp_dict = {label: SFCSExperiment(name=label) for label in sted_labels}

label_load_kwargs_dict = {
    label: dict(
        file_selection=data.file_selection,
        should_use_inherent_afterpulsing=data.should_use_inherent_afterpulsing,
    )
    for label, data in zip(data_labels, data_label_dict.values())
}

# TEST - print paths
print(list(label_template_paths_dict.values()))

# %%
FORCE_PROCESSING = False
# FORCE_PROCESSING = True

# load experiment
for label in sted_labels:
    halogen_exp_dict[label].load_experiment(
        confocal_template=label_template_paths_dict[confocal_label],
        sted_template=label_template_paths_dict[label],
        force_processing=data_label_dict[label].force_processing,
        should_re_correlate=FORCE_PROCESSING,
        should_subtract_afterpulse=True,
        should_unite_start_times=True,  # for uniting the two 5 kHz measurements
        inherent_afterpulsing_gates=(Limits(3, 10), Limits(30, 90)),
        **label_load_kwargs_dict[label],
    )

    # save processed data (to avoid re-processing)
    halogen_exp_dict[label].save_processed_measurements(
        should_force=data_label_dict[label].force_processing
    )

    # calibrate TDC
    halogen_exp_dict[label].calibrate_tdc(should_plot=True)

# Present count-rates
for label in sted_labels:
    print(f"\n{label}:")
    conf_meas = halogen_exp_dict[label].confocal
    sted_meas = halogen_exp_dict[label].sted
    print(
        f"Confocal countrate: {conf_meas.avg_cnt_rate_khz:.2f} +/- {conf_meas.std_cnt_rate_khz:.2f}"
    )
    print(f"STED countrate: {sted_meas.avg_cnt_rate_khz:.2f} +/- {sted_meas.std_cnt_rate_khz:.2f}")

# %% [markdown]
# Plot the STED CF_CRs:

# %%
with Plotter(
    xlabel="Lag (ms)", ylabel="Mean CF_CR", x_scale="log", xlim=(1e-4, 1e0), ylim=(-1, 6e4)
) as ax:
    for label in sted_labels:
        cf = halogen_exp_dict[label].sted.cf["sted"]
        ax.plot(cf.lag, cf.avg_cf_cr, label=label)
        ax.legend()

# %% [markdown]
# Now, lets gate the FR (Free Running) measurement for 3 ns as well. To make sure were normalizing the same way, let's TDC gate both of the measurements for the same gate.
#
# tdc/detector gating should overlap correctly - see line 1108 ["Unite TDC gate and detector gate"] in correlation_function.py

# %% [markdown]
# ### Gates

# %%
upper_gate = 20
lower_gates = [0, 3, 4, 5, 6]
gate_list = [(lower_gate, upper_gate) for lower_gate in lower_gates]


# %%
def plot_tdc_vs_detector_gating():
    """Doc."""

    with Plotter(
        super_title="Free-Running vs. $3~ns$ Detector Gating",
        xlabel="Lag (ms)",
        ylabel="Normalized",
        x_scale="log",
        xlim=(1e-4, 1e0),
        ylim=(-0.1, 2),
    ) as ax:

        meas_det_gated = halogen_exp_dict[detector_label].sted
        meas_tdc_gated = halogen_exp_dict[tdc_label].sted

        ax.plot(
            meas_det_gated.cf["sted"].lag,
            meas_det_gated.cf["sted"].normalized,
            "--",
            label="hard-gated",
        )
        for gate in gate_list:

            if gate == (0, 20):
                ax.plot(
                    meas_det_gated.cf[f"gSTED {gate}"].lag,
                    meas_det_gated.cf[f"gSTED {gate}"].normalized,
                    "x",
                    label=f"hard-gated {gate}",
                )
            else:
                ax.plot(
                    meas_det_gated.cf[f"gSTED {gate}"].lag,
                    meas_det_gated.cf[f"gSTED {gate}"].normalized,
                    "--",
                    label=f"hard-gated {gate}",
                )

        ax.set_prop_cycle(None)

        ax.plot(
            meas_tdc_gated.cf["sted"].lag,
            meas_tdc_gated.cf["sted"].normalized,
            label="free-running",
        )
        for gate in gate_list:
            ax.plot(
                meas_tdc_gated.cf[f"gSTED {gate}"].lag,
                meas_tdc_gated.cf[f"gSTED {gate}"].normalized,
                label=f"free-runing {gate}",
            )

        ax.legend()


# %% [markdown]
# ### Using inherent afterpulsing

# %%
detector_label = "300 bp ATTO STED 3 ns Detector Gating"
tdc_label = "300 bp ATTO STED FR"

FORCE_CORR = True
# FORCE_CORR = False

halogen_exp_dict[tdc_label].add_gates(
    gate_list,
    should_use_inherent_afterpulsing=True,
    inherent_afterpulsing_gates=(Limits(3, 10), Limits(30, 90)),
    should_re_correlate=FORCE_CORR,
    is_verbose=False,
)
halogen_exp_dict[detector_label].add_gates(
    gate_list,
    #     norm_range=(5e-3, 6e-3),
    should_use_inherent_afterpulsing=True,
    inherent_afterpulsing_gates=(Limits(3, 10), Limits(30, 90)),
    should_re_correlate=FORCE_CORR,
    is_verbose=False,
)

# PLOT
plot_tdc_vs_detector_gating()

# %% [markdown]
# ### Using calibrated afterpulsing

# %%
detector_label = "300 bp ATTO STED 3 ns Detector Gating"
tdc_label = "300 bp ATTO STED FR"

FORCE_CORR = True
# FORCE_CORR = False

halogen_exp_dict[tdc_label].add_gates(
    gate_list,
    should_re_correlate=FORCE_CORR,
    is_verbose=False,
)
halogen_exp_dict[detector_label].add_gates(
    gate_list,
    norm_range=(5e-3, 6e-3),
    should_re_correlate=FORCE_CORR,
    is_verbose=False,
)

# %% [markdown]
# Plotting

# %%
# PLOT
plot_tdc_vs_detector_gating()

# %% [markdown]
# ### Checking out the countrates

# %%
print("TDC")
[print(f"{label}: {cf.countrate}") for label, cf in halogen_exp_dict[tdc_label].sted.cf.items()]

print("")

print("detector")
[
    print(f"{label}: {cf.countrate}")
    for label, cf in halogen_exp_dict[detector_label].sted.cf.items()
]

print("")

print("RATIOS")
[
    print(f"{label}: {cf_tdc.countrate / cf_det.countrate}")
    for (label, cf_tdc), (_, cf_det) in zip(
        halogen_exp_dict[tdc_label].sted.cf.items(),
        halogen_exp_dict[detector_label].sted.cf.items(),
    )
]

print("")

print("Sub")
[
    print(f"{label}: {cf_tdc.countrate - cf_det.countrate}")
    for (label, cf_tdc), (_, cf_det) in zip(
        halogen_exp_dict[tdc_label].sted.cf.items(),
        halogen_exp_dict[detector_label].sted.cf.items(),
    )
]

# %%
# TESTING

tdc_calib = halogen_exp_dict[tdc_label].sted.tdc_calib

t_hist = tdc_calib.t_hist
all_hist_norm = tdc_calib.all_hist_norm

# set nans to 0
all_hist_norm[np.isnan(all_hist_norm)] = 0

total_prob = np.diff(t_hist) @ all_hist_norm[1:]

half_period = int(len(t_hist) / 2)

tail_prob = np.diff(t_hist)[half_period:] @ all_hist_norm[half_period + 1 :]

print("Free-Running")
print("2 * tail_prob / (total_prob - 2*tail_prob): ", 2 * tail_prob / (total_prob - 2 * tail_prob))


# Same for detector gated

tdc_calib = halogen_exp_dict[detector_label].sted.tdc_calib

t_hist = tdc_calib.t_hist
all_hist_norm = tdc_calib.all_hist_norm

# set nans to 0
all_hist_norm[np.isnan(all_hist_norm)] = 0

total_prob = np.diff(t_hist) @ all_hist_norm[1:]

half_period = int(len(t_hist) / 2)

tail_prob = np.diff(t_hist)[half_period:] @ all_hist_norm[half_period + 1 :]

print("Detector-Gated")
print("2 * tail_prob / (total_prob - 2*tail_prob): ", 2 * tail_prob / (total_prob - 2 * tail_prob))

# %% [markdown]
# ### Comparison of lifetime histogram of FR vs Detector gated STED

# %%
fr_tdc_calib = halogen_exp_dict[tdc_label].sted.tdc_calib
det_gated_tdc_calib = halogen_exp_dict[detector_label].sted.tdc_calib

with Plotter(super_title="FR vs Detector gated STED") as ax:
    ax.plot(fr_tdc_calib.t_hist, fr_tdc_calib.all_hist_norm)
    ax.plot(det_gated_tdc_calib.t_hist, det_gated_tdc_calib.all_hist_norm)

# %% [markdown]
# ### Total Afterpulsing Probability

# %%
print("TDC")
[
    print(f"{label}: {(np.diff(cf.lag) * 1e-3) @ cf.afterpulse[1:]}")
    for label, cf in halogen_exp_dict[tdc_label].sted.cf.items()
]

print("")

print("detector gated")
[
    print(f"{label}: {(np.diff(cf.lag) * 1e-3) @ cf.afterpulse[1:]}")
    for label, cf in halogen_exp_dict[detector_label].sted.cf.items()
]

# %% [markdown]
# ### Afterpulsing Accumulated Sum

# %%
print("TDC")
tdc_cumsum_dict = {
    label: (cf.lag, np.cumsum((np.diff(cf.lag) * 1e-3) * cf.afterpulse[1:]))
    for label, cf in halogen_exp_dict[tdc_label].sted.cf.items()
}

print("")

print("detector gated")
det_cumsum_dict = {
    label: (cf.lag, np.cumsum((np.diff(cf.lag) * 1e-3) * cf.afterpulse[1:]))
    for label, cf in halogen_exp_dict[detector_label].sted.cf.items()
}

from contextlib import suppress

with Plotter(x_scale="log") as ax:

    for (label, pair_tdc) in tdc_cumsum_dict.items():
        lag_tdc, cumsum_tdc = pair_tdc
        ax.plot(lag_tdc[1:], cumsum_tdc, label=f"{label} (TDC)")

    ax.set_prop_cycle(None)

    for (label, pair_det) in det_cumsum_dict.items():
        lag_det, cumsum_det = pair_det
        ax.plot(lag_det[1:], cumsum_det, "--", label=f"{label} (detector)")

    ax.legend()

# %% [markdown]
# Notice two things:
# * the detector-gated STED measurement (blue dotted line) is much different than the rest, increasing towards longer times
# * The above mentioned measurement is expected to behave similarly to the orange dotted curve, which should have the same lower gate
#
# The solution is given by the inclusion of an additional gate (0, 20), which gives the same result as (3, 20) for the detector-gated measurement

# %% [markdown]
# ### Comparing signal to noise and cross-correlations:

# %%
corr_names = ("AA", "AB", "BA", "BB")
gate1_ns = Limits(3, 10)
gate2_ns = Limits(30, 95)

# %%
cf_list = halogen_exp_dict[tdc_label].sted.cross_correlate_data(
    corr_names=corr_names,
    gate1_ns=gate1_ns,
    gate2_ns=gate2_ns,
    should_add_to_xcf_dict=False,
    is_verbose=True,
    should_subtract_afterpulse=False,
)

for cf in cf_list:
    cf.average_correlation()

# %%
with Plotter() as ax:
    for cf in cf_list:
        #         if "AB" in cf.name or "BA" in cf.name:
        #             cf.normalized_test = cf.avg_cf_cr * (100/gate2_ns.interval() *cf.countrate_b)
        #             cf.plot_correlation_function(y_field="normalized_test", parent_ax=ax, plot_kwargs={"label": cf.name})
        #         else:
        cf.plot_correlation_function(
            y_field="avg_corrfunc", parent_ax=ax, plot_kwargs={"label": cf.name}
        )
        pass
    ax.legend()

# %%
gate2_ns.interval()

# %% [markdown]
# ### Comparing to total AP probability of some 'old detector'  300 bp measurement

# %%
# load experiment
old_exp = SFCSExperiment("old detector 300 bp ATTO")
old_exp.load_experiment(
    confocal_template=DATA_ROOT / "10_08_2021" / "solution" / "bp300_20uW_angular_exc_141639_*.pkl",
    force_processing=False,
    should_re_correlate=False,
    should_subtract_afterpulse=True,
    #         should_unite_start_times=True,  # for uniting measurements of same name
    #         inherent_afterpulsing_gates=(Limits(3, 10), Limits(30, 90)),
    file_selection="Use All",
    should_plot=False,
    should_plot_meas=False,
)

# save processed data (to avoid re-processing)
old_exp.save_processed_measurements(should_force=False)

# calibrate TDC
old_exp.calibrate_tdc(should_plot=True)

# Present count-rates
print(f"\n{old_exp.name}:")
conf_meas = old_exp.confocal
print(f"Confocal countrate: {conf_meas.avg_cnt_rate_khz:.2f} +/- {conf_meas.std_cnt_rate_khz:.2f}")

# %% [markdown]
# Calculate afterpulsing probability:

# %%
cf = old_exp.confocal.cf["confocal"]
(np.diff(cf.lag) * 1e-3) @ cf.afterpulse[1:]

# %%
cf.countrate

# %%
tdc_calib = old_exp.confocal.tdc_calib

t_hist = tdc_calib.t_hist
all_hist_norm = tdc_calib.all_hist_norm

# set nans to 0
all_hist_norm[np.isnan(all_hist_norm)] = 0

total_prob = np.diff(t_hist) @ all_hist_norm[1:]

half_period = int(len(t_hist) / 2)

tail_prob = np.diff(t_hist)[half_period:] @ all_hist_norm[half_period + 1 :]

print("Detector-Gated")
print("2 * tail_prob / (total_prob - 2*tail_prob): ", 2 * tail_prob / (total_prob - 2 * tail_prob))

# %%
cf_list = old_exp.confocal.cross_correlate_data(
    corr_names=corr_names,
    gate1_ns=gate1_ns,
    gate2_ns=gate2_ns,
    should_add_to_xcf_dict=False,
    is_verbose=True,
    should_subtract_afterpulse=False,
)

for cf in cf_list:
    cf.average_correlation()

# %%
with Plotter() as ax:
    for cf in cf_list:
        #         if "AB" in cf.name or "BA" in cf.name:
        #             cf.normalized_test = cf.avg_cf_cr * (100/gate2_ns.interval() *cf.countrate_b)
        #             cf.plot_correlation_function(y_field="normalized_test", parent_ax=ax, plot_kwargs={"label": cf.name})
        #         else:
        cf.plot_correlation_function(
            y_field="avg_corrfunc", parent_ax=ax, plot_kwargs={"label": cf.name}
        )
        pass
    ax.legend()

# %% [markdown]
# ### Comparing to a 'Laser only' measurement

# %%
# load experiment
laser_exp = SFCSExperiment("old detector 300 bp ATTO")
laser_exp.load_experiment(
    confocal_template=DATA_ROOT
    / "20_03_2022"
    / "solution"
    / "mirror_100uW_wfilter_static_exc_143208_*.pkl",
    force_processing=False,
    should_re_correlate=False,
    should_subtract_afterpulse=True,
    #         should_unite_start_times=True,  # for uniting measurements of same name
    #         inherent_afterpulsing_gates=(Limits(3, 10), Limits(30, 90)),
    file_selection="Use All",
    should_plot=False,
    should_plot_meas=False,
)

# save processed data (to avoid re-processing)
laser_exp.save_processed_measurements(should_force=False)

# calibrate TDC
laser_exp.calibrate_tdc(should_plot=True)

# Present count-rates
print(f"\n{laser_exp.name}:")
conf_meas = laser_exp.confocal
print(f"Confocal countrate: {conf_meas.avg_cnt_rate_khz:.2f} +/- {conf_meas.std_cnt_rate_khz:.2f}")

# %% [markdown]
# Calculate afterpulsing:

# %%
cf = laser_exp.confocal.cf["confocal"]
(np.diff(cf.lag) * 1e-3) @ cf.afterpulse[1:]

# %%
cf.countrate

# %%
tdc_calib = laser_exp.confocal.tdc_calib

t_hist = tdc_calib.t_hist
all_hist_norm = tdc_calib.all_hist_norm

# set nans to 0
all_hist_norm[np.isnan(all_hist_norm)] = 0

total_prob = np.diff(t_hist) @ all_hist_norm[1:]

half_period = int(len(t_hist) / 2)

tail_prob = np.diff(t_hist)[half_period:] @ all_hist_norm[half_period + 1 :]

print("Detector-Gated")
print("2 * tail_prob / (total_prob - 2*tail_prob): ", 2 * tail_prob / (total_prob - 2 * tail_prob))

# %% [markdown]
# Getting and showing AA, AB, BA, BB auto/cross correlations:

# %%
gate1_ns = (3.2, 4.5)
gate2_ns = (10, 95)

cf_list = laser_exp.confocal.cross_correlate_data(
    corr_names=corr_names,
    gate1_ns=gate1_ns,
    gate2_ns=gate2_ns,
    should_add_to_xcf_dict=False,
    is_verbose=True,
    should_subtract_afterpulse=False,
)

for cf in cf_list:
    cf.average_correlation()

# %%
with Plotter() as ax:
    for cf in cf_list:
        #         if "AB" in cf.name or "BA" in cf.name:
        #             cf.normalized_test = cf.avg_cf_cr * (100/gate2_ns.interval() *cf.countrate_b)
        #             cf.plot_correlation_function(y_field="normalized_test", parent_ax=ax, plot_kwargs={"label": cf.name})
        #         else:
        cf.plot_correlation_function(
            y_field="avg_corrfunc", parent_ax=ax, plot_kwargs={"label": cf.name}
        )
        pass
    ax.legend()

# %% [markdown]
# Play sound when done:

# %%
delta_t = 175
freq_range = (3000, 7000)
n_beeps = 5
[Beep(int(f), delta_t) for f in np.linspace(*freq_range, n_beeps)]
