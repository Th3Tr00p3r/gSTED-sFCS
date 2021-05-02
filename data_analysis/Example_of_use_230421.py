#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:10:22 2021

@author: oleg
"""

#import h5py 
import sys
import os
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
#sys.path.append('//Users/oleg/Documents/Python programming/FCS Python/')

from CorrFuncTDCclass import CorrFuncTDCclass

#%% Load Data
S = CorrFuncTDCclass()   
S.DoReadFPGAdata( '/Users/oleg/Documents/Experiments/STED/For testing Python/solScan_exc_5_mins_2204/solScan_exc_*.mat')           

#%% Correlate
S.DoCorrelateRegularData()#%% Test how correlation works
S.DoAverageCorr()
#%% if need just to plot
plt.cla()
S.DoPlotCorrFunc()
#%%
plt.cla()
S.DoFit()

