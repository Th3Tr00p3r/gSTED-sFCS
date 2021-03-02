#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 17:50:37 2021

@author: oleg
"""

import numpy as np 
#import h5py 
import sys
sys.path.append('//Users/oleg/Documents/Python programming/FCS Python/')

import scipy.io

from PhotonDataClass import PhotonDataClass

#%%

FPGAdata = np.fromfile('/Users/oleg/Documents/Python programming/FCS Python/testData.bin', dtype=np.uint8)
#PathToData = '//Users/oleg/Documents/Python programming/FCS Python/testData.bin'
#FullData = scipy.io.loadmat(PathToData)['FullData']

#f = h5py.File(PathToData,'r') 

# for name, data in f.items():
#     print("Name ", name)  # Name
#     if type(data) is h5py.Dataset:
#         # If DataSet pull the associated Data
#         # If not a dataset, you may need to access the element sub-items
#         value = data.value
#         print("Value", value)  # NumPy Array / Value
        
#data = f.get('data/variable1') # Get a certain dataset
#data = np.array(data)

#%%
PD = PhotonDataClass();
PD.DoConvertFPGAdataToPhotons(FPGAdata)

#%% save to Matlab

scipy.io.savemat('/Users/oleg/Documents/Python programming/FCS Python/procData.mat', vars(PD))

#%%
#from ctypes import c_double, c_int, CDLL, cdll, c_long, byref

#%% compile from MacOS terminal window

#clang++ -o SoftCorrelatorDynamicLib.so -shared -fPIC -O2 SoftCorrelatorDynamicLib.cpp Correlator.cpp CountCorrelator.cpp CPhDelayCrossCorrelator.cpp

# or the same with g++



#%%
import sys
from ctypes import c_double, c_int, CDLL, cdll, c_long, byref

lib_path = '/Users/oleg/Documents/C programming/SoftCorrelatorDynamicLib/SoftCorrelatorDynamicLib/SoftCorrelatorDynamicLib.so'# % (sys.platform)
try:
    SoftCorrelatorDynamicLib = CDLL(lib_path)
   #lib = ctypes.cdll.LoadLibrary(lib_path)
   #lib = cdll.LoadLibrary(lib_path)
except:
    print('OS %s not recognized' % (sys.platform))


csquare = SoftCorrelatorDynamicLib.c_square

SoftCorr = SoftCorrelatorDynamicLib.softwareCorrelator


#%%
maxChanLen = 1000
maxChanLen3 = maxChanLen*3
corrPy =  np.zeros(maxChanLen3, dtype=float)
corrC = (c_double * maxChanLen3)(*corrPy)
HistLen = 2000
phHistPy = 5*np.ones(HistLen, dtype=int)
phHistC = (c_long * HistLen)(*phHistPy)
NoCorrChannels = (c_long*1)()

#%%
SoftCorr(c_int(1), c_long(HistLen), phHistC, NoCorrChannels, corrC)

#%%


#%%
list_in = np.arange(5)
n = len(list_in)
c_arr_in = (c_double * n)(*list_in)
c_arr_out = (c_double * n)()

csquare(c_int(10), c_arr_in, c_arr_out)


#%%
from contextlib import contextmanager
import io
import os

@contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout

import ctypes
libc = ctypes.CDLL(None)

f = io.StringIO()
#%%
with stdout_redirector(f):
    print('foobar')
    print(12)
    libc.puts(b'this comes from C')
    os.system('echo and this is from echo')
print('Got stdout: "{0}"'.format(f.getvalue()))

#%% Trying to use numpy arrays directly
import sys
from ctypes import c_double, c_int, CDLL, c_long
from numpy.ctypeslib import ndpointer

lib_path = '/Users/oleg/Documents/C programming/SoftCorrelatorDynamicLib/SoftCorrelatorDynamicLib/SoftCorrelatorDynamicLib.so'# % (sys.platform)
try:
    SoftCorrelatorDynamicLib = CDLL(lib_path)
   #lib = ctypes.cdll.LoadLibrary(lib_path)
   #lib = cdll.LoadLibrary(lib_path)
except:
    print('OS %s not recognized' % (sys.platform))


csquare = SoftCorrelatorDynamicLib.c_square
csquare.restype = None
csquare.argtypes = [c_int, ndpointer(c_double), ndpointer(c_double)]

SoftCorr = SoftCorrelatorDynamicLib.softwareCorrelator
SoftCorr.restype = None
#SoftCorr.argtypes = [c_int, c_long, ndpointer(c_long), ndpointer(c_int), ndpointer(c_double)]

SoftCorr.argtypes = [c_int, c_long, ndpointer(c_long, ndim=1, flags='C_CONTIGUOUS'), ndpointer(c_int), ndpointer(c_double, ndim=2, flags='C_CONTIGUOUS')]

getCorrParams = SoftCorrelatorDynamicLib.getCorrelatorParams
getCorrParams.restype = None
getCorrParams.argtypes = [ndpointer(c_int), ndpointer(c_int)]

#%%
DoubSize = np.zeros(1, dtype=np.int32)
NumCorr = np.zeros(1, dtype=np.int32)
getCorrParams(DoubSize, NumCorr)

#%%
TotalChanLen = DoubSize[0]*(NumCorr[0]+1)+1
#maxChanLen3 = maxChanLen*3
corrPy =  np.zeros((3, TotalChanLen), dtype= float)
#corrC = (c_double * maxChanLen3)(*corrPy)
HistLen = 2000
phHistPy = 5*np.ones(HistLen, dtype= np.int64)
#phHistC = (c_long * HistLen)(*phHistPy)
NoCorrChannels = np.zeros(1, dtype = np.int32)

#%%
SoftCorr(1, phHistPy.size, phHistPy, NoCorrChannels, corrPy)

#%%
list_in = np.arange(5, dtype = float)
n = len(list_in)
#c_arr_in = (c_double * n)(*list_in)
c_arr_out = np.zeros(5, dtype=float)

csquare(5, list_in, c_arr_out)


#%%
from SoftwareCorrelatorModule import SoftwareCorrelatorClass

SC = SoftwareCorrelatorClass()

#%%
DS, NC = SC.getCorrelatorParams()

#%%
from SoftwareCorrelatorModule import SoftwareCorrelatorClass, CorrelatorType

SC = SoftwareCorrelatorClass()
SC.SoftCrossCorrelator(phHistPy, CorrelatorType.PhDelayCorrelator, 1)

                         

