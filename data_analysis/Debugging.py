#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:09:52 2021

@author: oleg
"""
import numpy as np 
#import h5py 
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
#sys.path.append('//Users/oleg/Documents/Python programming/FCS Python/')

import scipy.io

from PhotonDataClass import PhotonDataClass
from MatlabUtilities import loadmat #loads matfiles with structures (scipy.io does not do structures)
from CorrFuncTDCclass import CorrFuncTDCclass
import glob
import re #regular expressions

S = CorrFuncTDCclass()   
S.DoReadFPGAdata( '/Users/oleg/Documents/Experiments/STED/For testing Python/solScan_exc_5_mins_2204/solScan_exc_1.mat')           
S.DoCorrelateRegularData()#%% Test how correlation works
S.DoAverageCorr()
S.DoPlotCorrFunc()