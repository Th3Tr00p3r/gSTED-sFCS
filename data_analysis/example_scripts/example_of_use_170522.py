#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Fri Apr 23 14:10:22 2021

@author: oleg
"""

import os

# import h5py
import sys

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

# from data_analysis.correlation_function import CorrFuncTDC
from data_analysis.correlation_function import SFCSExperiment
from data_analysis.software_correlator import CorrelatorType, SoftwareCorrelator
from utilities.helper import Limits

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
print(os.path.dirname(__file__))
# sys.path.append('//Users/oleg/Documents/Python programming/FCS Python/')


mpl.rc("figure", max_open_warning=0)

# %%
SC = SoftwareCorrelator()


# %%

# SC.correlate(np.array([ 5, 5, 5, 5, 5]), CorrelatorType.PH_DELAY_CORRELATOR, 1)

# %%

# SC.correlate(np.array([ 5, 5, 5, 5, 5]), CorrelatorType.PH_DELAY_CORRELATOR, 1)

# %% load all of the measurements and compare

Sang = SFCSExperiment("T100518", SC=SC)

# %%
# Sang.load_experiment(confocal_template = '/Users/oleg/Google Drive/10.5.18/AngularScan/120/AngularScan300bpConf*.mat',
#                     sted_template = '/Users/oleg/Google Drive/10.5.18/AngularScan/120/AngularScan300bpSted*.mat',
#                     file_selection='Use 0')

Sang.load_experiment(
    confocal_path_template="/Users/oleg/Google Drive/10.5.18/AngularScan/120/AngularScan300bpConf*.mat",
    sted_path_template="/Users/oleg/Google Drive/10.5.18/AngularScan/120/AngularScan300bpSted*.mat",
)


# %%
Sang.plot_correlation_functions()

# %% calibrate TDC
Sang.calibrate_tdc()

# %%
Sang.compare_lifetimes()


# %%
Sang.add_gate(gate_ns=(4, 6), verbose=True)

# %% try cross correlation

CF_AB_4_6_40_80, *_ = Sang.sted.cross_correlate_data(
    corr_names=("AB",),
    cf_name="Afterpulsing",
    gate1_ns=Limits(4, 6),
    gate2_ns=Limits(40, 80),
    should_subtract_afterpulse=False,
    # subtract_spatial_bg_corr=False,
    # should_rotate_data=False,  # abort data rotation decorator
)
CF_AB_4_6_40_80.average_correlation()
CF_AB_4_6_40_80.plot_correlation_function()

# %%
CF_AB, *_ = Sang.sted.cross_correlate_data(
    corr_names=("AB",),
    cf_name="Afterpulsing",
    gate1_ns=Limits(2, 10),
    gate2_ns=Limits(40, 80),
    should_subtract_afterpulse=False,
    # subtract_spatial_bg_corr=False,
    # should_rotate_data=False,  # abort data rotation decorator
)
CF_AB.average_correlation()
CF_AB.plot_correlation_function()

CF_AB_2_10_40_80 = CF_AB

# %%
CF_AB, *_ = Sang.sted.cross_correlate_data(
    corr_names=("AB",),
    cf_name="Afterpulsing",
    gate1_ns=Limits(2, 10),
    gate2_ns=Limits(60, 80),
    should_subtract_afterpulse=False,
    # subtract_spatial_bg_corr=False,
    # should_rotate_data=False,  # abort data rotation decorator
)
CF_AB.average_correlation()
CF_AB.plot_correlation_function()

CF_AB_2_10_60_80 = CF_AB

# %%
CF_AB, *_ = Sang.confocal.cross_correlate_data(
    corr_names=("AB",),
    cf_name="Afterpulsing",
    gate1_ns=Limits(2, 10),
    gate2_ns=Limits(60, 80),
    should_subtract_afterpulse=False,
    # subtract_spatial_bg_corr=False,
    # should_rotate_data=False,  # abort data rotation decorator
)
CF_AB.average_correlation()
CF_AB.plot_correlation_function()

CF_ABconf_2_10_60_80 = CF_AB

# %% Add gate in the background part
CF_AB, *_ = Sang.confocal.cross_correlate_data(
    corr_names=("AA",),
    cf_name="Afterpulsing",
    gate1_ns=Limits(40, 90),
    gate2_ns=Limits(40, 90),
    should_subtract_afterpulse=False,
    # subtract_spatial_bg_corr=False,
    # should_rotate_data=False,  # abort data rotation decorator
)

#
CF_AB.average_correlation()
CF_AB.plot_correlation_function()
CF_conf_40_90 = CF_AB

# %% Add gate in the background part
CF_AB, *_ = Sang.confocal.cross_correlate_data(
    corr_names=("AA",),
    cf_name="Afterpulsing",
    gate1_ns=Limits(40, 75),
    gate2_ns=Limits(40, 75),
    should_subtract_afterpulse=False,
    # subtract_spatial_bg_corr=False,
    # should_rotate_data=False,  # abort data rotation decorator
)

#
CF_AB.average_correlation()
CF_AB.plot_correlation_function()
CF_conf_40_75 = CF_AB

# %%
plt.cla()
# plt.semilogx(CF_AB_4_6_40_80.lag[1:], CF_AB_4_6_40_80.avg_cf_cr[1:]/CF_AB_4_6_40_80.countrate_list[0].a)
plt.semilogx(
    CF_ABconf_2_10_60_80.lag[1:],
    CF_ABconf_2_10_60_80.avg_cf_cr[1:] / CF_ABconf_2_10_60_80.countrate_list[0].a,
)
# plt.semilogx(CF_AB_2_10_60_80.lag[1:], CF_AB_2_10_60_80.avg_cf_cr[1:]/CF_AB_2_10_60_80.countrate_list[0].a)
plt.semilogx(Sang.confocal.cf["confocal"].lag, Sang.confocal.cf["confocal"].afterpulse)
plt.semilogx(CF_conf_40_90.lag[1:], CF_conf_40_90.avg_cf_cr[1:] / CF_conf_40_90.countrate_list[0])
plt.semilogx(CF_conf_40_75.lag[1:], CF_conf_40_75.avg_cf_cr[1:] / CF_conf_40_75.countrate_list[0])
# plt.semilogx(sted.lag[1:], sted.afterpulse[1:]/500)
plt.show()


# %%
plt.cla()
# plt.semilogx(CF_AB_4_6_40_80.lag[1:], CF_AB_4_6_40_80.avg_cf_cr[1:]/CF_AB_4_6_40_80.countrate_list[0].a)
plt.semilogx(
    CF_ABconf_2_10_60_80.lag[1:],
    CF_ABconf_2_10_60_80.avg_cf_cr[1:]
    / CF_ABconf_2_10_60_80.countrate_list[0].a
    * CF_ABconf_2_10_60_80.countrate_list[0].b
    * 98
    / 20,
)
# plt.semilogx(CF_AB_2_10_60_80.lag[1:], CF_AB_2_10_60_80.avg_cf_cr[1:]/CF_AB_2_10_60_80.countrate_list[0].a)
plt.semilogx(Sang.confocal.cf["confocal"].lag, Sang.confocal.cf["confocal"].afterpulse)
plt.semilogx(CF_conf_40_90.lag[1:], CF_conf_40_90.avg_cf_cr[1:])
plt.semilogx(CF_conf_40_75.lag[1:], CF_conf_40_75.avg_cf_cr[1:])
# plt.semilogx(sted.lag[1:], sted.afterpulse[1:]/500)
plt.show()


# %% evaluate average intensity
CF = CF_ABconf_2_10_60_80
cr = np.array([x.b for x in CF.countrate_list])


# %% try cross correlation

CFlist = Sang.sted.cross_correlate_data(
    cf_name="Afterpulsing",
    gate1_ns=Limits(4, 6),
    gate2_ns=Limits(4, 6),
    should_subtract_afterpulse=False,
    # subtract_spatial_bg_corr=False,
    # should_rotate_data=False,  # abort data rotation decorator
)
for CF in CFlist:
    CF.average_correlation()

# %%
CF_4_6 = CFlist[2]

# %%
plt.cla()
for CF in CFlist:
    CF.plot_correlation_function()
plt.show()


# %%
plt.cla()
for CF in CFlist:
    plt.semilogx(CF.lag, CF.average_all_cf_cr)

plt.show()

# %%
CFlist = Sang.sted.cross_correlate_data(
    cf_name="Afterpulsing",
    gate1_ns=Limits(4, 6),
    gate2_ns=Limits(40, 80),
    should_subtract_afterpulse=False,
    # subtract_spatial_bg_corr=False,
    # should_rotate_data=False,  # abort data rotation decorator
)
for CF in CFlist:
    CF.average_correlation()


CF_4_6_40_80 = CFlist[2]
# %%
plt.cla()
for CF in CFlist[:4]:
    plt.semilogx(CF.lag, CF.average_all_cf_cr)

plt.show()


# %%

plt.cla()
plt.semilogx(CFlist[3].lag[1:], CFlist[3].avg_cf_cr[1:])
plt.semilogx(CFlist[2].lag[1:], CFlist[2].avg_cf_cr[1:])

plt.show()

# %%
CFlist = Sang.sted.cross_correlate_data(
    cf_name="Afterpulsing",
    gate1_ns=Limits(4, 6),
    gate2_ns=Limits(40, 80),
    should_subtract_afterpulse=False,
    # subtract_spatial_bg_corr=False,
    # should_rotate_data=False,  # abort data rotation decorator
)
for CF in CFlist:
    CF.average_correlation()

CF_4_6 = CFlist[2]


# %%
CFlist = Sang.sted.cross_correlate_data(
    cf_name="Afterpulsing",
    gate1_ns=Limits(2, 4),
    gate2_ns=Limits(75, 80),
    should_subtract_afterpulse=False,
    # subtract_spatial_bg_corr=False,
    # should_rotate_data=False,  # abort data rotation decorator
)
for CF in CFlist:
    CF.average_correlation()

CF_4_6_40_60 = CFlist[2]

# %%

sted = Sang.sted.cf["sted"]
plt.cla()
plt.semilogx(CF_2_10.lag[1:], CF_2_10.avg_cf_cr[1:] / CF_2_10.countrate_list[0][0])
plt.semilogx(CF_4_6.lag[1:], CF_4_6.avg_cf_cr[1:] / CF_4_6.countrate_list[0][0])
plt.semilogx(CF_4_6_40_60.lag[1:], CF_4_6_40_60.avg_cf_cr[1:] / CF_4_6_40_60.countrate_list[0][0])
# plt.semilogx(sted.lag[1:], sted.afterpulse[1:]/500)
plt.show()

# %%
Sang.add_gate(gate_ns=(6, 20), verbose=True)


# %% load confocal
Sang = SFCSExperiment()
Sang.load_experiment(
    "confocal", "/Users/oleg/Google Drive/10.5.18/AngularScan/120/AngularScan300bpConf*.mat"
)

# %% load sted
Sang.load_experiment(
    "sted", "/Users/oleg/Google Drive/10.5.18/AngularScan/120/AngularScan300bpSted*.mat"
)

# %%
plt.cla()
Sang.plot_correlation_functions()


# %% calibrate TDC


# %% old angular scan data
Sang = CorrFuncTDC()
Sang.read_fpga_data(
    "/Users/oleg/Google Drive/10.5.18/AngularScan/120/AngularScan300bpSted*.mat",
    should_fix_shift=True,
)  # fixing shift since these are old data


# %% Atto of 100821

Sang = CorrFuncTDC()
Sang.read_fpga_data(
    "/Users/oleg/Documents/Students/Ido Michaelovich/Experiments/100821/Atto488_20uW_angular_exc_121936_*.pkl"
)


# #%%  ****** test angular scan *********
# Sang = CorrFuncTDC()
# Sang.read_fpga_data('/Users/oleg/Google Drive/10.5.18/AngularScan/120/AngularScan300bpSted*.mat',
#                     should_fix_shift = True) #fixing shift since these are old data

# %% Correlate
Sang.correlate_and_average()

# %% if need just to plot
plt.cla()
Sang.plot_correlation_functions()

# %%
Sang.calibrate_tdc()

# %%
Sang.correlate_and_average(gate_ns=(4, 20), verbose=True)

# %%
plt.cla()
Sang.plot_correlation_functions()


# %%
Sang.correlate_and_average(gate_ns=(6, 20), verbose=True)

# %%
plt.cla()
Sang.plot_correlation_functions()


# %% Atto of 100821

S = CorrFuncTDC()
S.read_fpga_data(
    "/Users/oleg/Documents/Students/Ido Michaelovich/Experiments/051021/sol_static_sted_183413_*.pkl"
)

# %% Correlate
S.correlate_and_average(cf_name="CW STED", verbose=True)

# %% if need just to plot
plt.cla()
S.plot_correlation_functions()

# %%
S.calibrate_tdc()

# %%
plt.subplot(111)
S.fit_lifetime_hist(fit_range=(2, 40))
for key, item in S.tdc_calib["fit_param"].items():
    print(key + ":")
    print("fit parameters:  " + str(item["beta"]))

# %% Correlate
S.correlate_and_average(gate_ns=(4, 20), verbose=True)

# %%
plt.cla()
S.plot_correlation_functions()


# %% ****** test angular scan *********
Sang = CorrFuncTDC()
Sang.read_fpga_data(
    "/Users/oleg/Google Drive/10.5.18/AngularScan/120/AngularScan300bpSted*.mat",
    should_fix_shift=True,
)  # fixing shift since these are old data

# Sang.after_pulse_param = S.after_pulse_param # a patch: in these data a different from multi_exponent_fit model is used for afterpulsing

# %% Correlate
Sang.correlate_and_average()


# %%
Sang.calibrate_tdc()

# %%
plt.subplot(111)
plt.cla()
Sang.fit_lifetime_hist()

# %%
plt.subplot(111)
S.fit_lifetime_hist(fit_range=(3, 40))
for key, item in S.tdc_calib["fit_param"].items():
    print(key + ":")
    print("fit parameters:  " + str(item["beta"]))

# %% if need just to plot
plt.cla()
Sang.plot_correlation_functions()
# %%
plt.cla()
Sang.fit_correlation_function()

# %%
Sang.calibrate_tdc()


# %% Load static Data
S = CorrFuncTDC()
S.read_fpga_data(
    "/Users/oleg/Documents/Experiments/STED/For testing Python/solScan_exc_5_mins_2204/solScan_exc_*.mat"
)

# %% Correlate
S.correlate_and_average()

# %% if need just to plot
plt.cla()
S.plot_correlation_function()
# %%
plt.cla()
S.fit_correlation_function(should_plot=True)

# %%
S.calibrate_tdc()

# %%
S.fit_lifetime_hist()

# %%
plt.subplot(111)
S.fit_lifetime_hist(fit_range=(3, 40))
for key, item in S.tdc_calib["fit_param"].items():
    print(key + ":")
    print("fit parameters:  " + str(item["beta"]))
# %%
plt.cla()
S.compare_lifetimes(legend_label="staticTest")

# %% check of compare_lifetimes would work with two objects
plt.subplot(111)
plt.cla()
S.compare_lifetimes(legend_label="staticTest", angularScan=Sang)

# %% old angular scan data
Sang = CorrFuncTDC()
Sang.read_fpga_data("/Users/oleg/Google Drive/10.5.18/AngularScan/120/AngularScan300bpConf*.mat")

# %%
Sang.calibrate_tdc()

# %%
plt.cla()

Sang.fit_lifetime_hist(fit_range=(3, 40))
for key, item in Sang.tdc_calib["fit_param"].items():
    print(key + ":")
    print("fit parameters:  " + str(item["beta"]))


# %% Atto of 100821

Sang = CorrFuncTDC()
Sang.read_fpga_data(
    "/Users/oleg/Documents/Students/Ido Michaelovich/Experiments/100821/Atto488_20uW_angular_exc_121936_*.pkl"
)

# %% Correlate
Sang.correlate_and_average()


# %%
Sang.calibrate_tdc()

# %%
plt.subplot(111)
plt.cla()
Sang.fit_lifetime_hist()

# %%
plt.subplot(111)
S.fit_lifetime_hist(fit_range=(3, 40))
for key, item in S.tdc_calib["fit_param"].items():
    print(key + ":")
    print("fit parameters:  " + str(item["beta"]))
