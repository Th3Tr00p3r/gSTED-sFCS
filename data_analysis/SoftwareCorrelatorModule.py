#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 14:08:54 2021

@author: oleg
"""
import numpy as np
from ctypes import c_double, c_int, CDLL, c_long
from numpy.ctypeslib import ndpointer
import enum
from warnings import warn
 

class CorrelatorType(enum.Enum):
    PhDelayCorrelator = 1
    PhCountCorrelator = 2
    PhDelayCrossCorrelator = 3
    PhCountCrossCorrelator = 4  #does not seem to be implemented check! 1st column is photon arrival times, 2nd column boolean vector with 1s for photons arriving on the A channel and  0s otherwise, and the 3rd column is 1s for photons arriving on B channel and 0s otherwise
    PhDelayCorrelatorLines = 5  # additional column with 1s for photons that have to be counted, on 0s the delay chain is reset; used in image correlation and in angular scan
    PhDelayCrossCorrelatorLines = 6 # as PhCountCrossCorrelator with an additional column with 1s for photons that have to be counted, on 0s the delay chain is reset; used in image correlation and in angular scan

   
class SoftwareCorrelatorClass:
    def __init__(self):
        lib_path = '/Users/oleg/Documents/Python programming/Scanning setups Lab/gSTED-sFCS/SoftCorrelatorDynamicLib/SoftCorrelatorDynamicLib/SoftCorrelatorDynamicLib.so'
        self.libPath = lib_path
        SoftCorrelatorDynamicLib = CDLL(lib_path)
        getCorrParams = SoftCorrelatorDynamicLib.getCorrelatorParams
        getCorrParams.restype = None
        getCorrParams.argtypes = [ndpointer(c_int), ndpointer(c_int)]
        DoubSize = np.zeros(1, dtype=np.int32)
        NumCorr = np.zeros(1, dtype=np.int32)
        getCorrParams(DoubSize, NumCorr)
        self.DoublingSize = DoubSize[0]
        self.NumOfCorrelators = NumCorr[0]
        self.getCorrParams = getCorrParams
        self.TotalCorrChanLen = DoubSize[0]*(NumCorr[0]+1)+1
        self.corrPy =  np.zeros((3, self.TotalCorrChanLen), dtype= float)
        
        SoftCorr = SoftCorrelatorDynamicLib.softwareCorrelator
        SoftCorr.restype = None
        SoftCorr.argtypes = [c_int, c_long, ndpointer(c_long, ndim=1, flags='C_CONTIGUOUS'), ndpointer(c_int), ndpointer(c_double, ndim=2, flags='C_CONTIGUOUS')]
        self.SoftCorr = SoftCorr
        
        self.corrPy =  np.zeros((3, self.TotalCorrChanLen), dtype= float)
    
    def getCorrelatorParams(self):
        DoubSize = np.zeros(1, dtype=np.int32)
        NumCorr = np.zeros(1, dtype=np.int32)
        self.getCorrParams(DoubSize, NumCorr)
        self.DoublingSize = DoubSize[0]
        self.NumOfCorrelators = NumCorr[0]
        return DoubSize[0], NumCorr[0]
    
    def SoftCrossCorrelator(self, photonArray, CType = CorrelatorType.PhDelayCorrelator, timebase_ms = 1):
        if len(photonArray.shape) == 1:
            Nentries = photonArray.size
        else:
            Nentries = photonArray.shape[1]
        phHist = photonArray.reshape(-1) #convert to 1D array
        NoCorrChannels = np.zeros(1, dtype=np.int32)
        
        
        if CType == CorrelatorType.PhDelayCorrelator:
            if len(photonArray.shape) != 1:
                raise Exception("Photon Array should be 1D for this correlator option!")
            self.countrate = Nentries/photonArray.sum()/timebase_ms*1000
            
        elif CType == CorrelatorType.PhCountCorrelator:
            if len(photonArray.shape) != 1:
                raise Exception("Photon Array should be 1D for this correlator option!")
            self.countrate = photonArray.sum()/Nentries/timebase_ms*1000
            
        elif CType == CorrelatorType.PhDelayCrossCorrelator:
            if (len(photonArray.shape) == 1) or (photonArray.shape[0] != 3):
                raise Exception("""Photon Array should have 3 rows for this correlator option! 
                                0st row with photon delay times, 1st (2nd)  row contains 1s 
                                for photons in channel A (B) and 0s for photons in channel B(A)""")
            Duration_s = photonArray[0, :].sum()*timebase_ms/1000
            self.countrateA = photonArray[1, :].sum()/Duration_s
            self.countrateB = photonArray[2, :].sum()/Duration_s
            
        elif CType == CorrelatorType.PhDelayCorrelatorLines:
            if (len(photonArray.shape) == 1) or (photonArray.shape[0] != 2):
                raise Exception("""Photon Array should have 2 rows for this correlator option! 
                                0st row with photon delay times, 1st row is 1 for valid lines""")
            Valid = (photonArray[1, :] == 1) or (photonArray[1, :] == -2)
            Duration_s = photonArray[0, Valid].sum()*timebase_ms/1000
            self.countrate = np.sum(photonArray[1, :] == 1)/Duration_s
            
        elif CType == CorrelatorType.PhDelayCrossCorrelatorLines:
            if (len(photonArray.shape) == 1) or (photonArray.shape[0] != 4):
                raise Exception("""Photon Array should have 3 rows for this correlator option! 
                                0st row with photon delay times, 1st (2nd)  row contains 1s 
                                for photons in channel A (B) and 0s for photons in channel B(A),
                                and 3rd column is 1s for valid lines""")
            Valid = (photonArray[3, :] == 1) or (photonArray[3, :] == -2)
            Duration_s = photonArray[0, Valid].sum()*timebase_ms/1000
            self.countrateA = np.sum(photonArray[1, :] == 1)/Duration_s
            self.countrateB = np.sum(photonArray[2, :] == 1)/Duration_s
        else :
            raise Exception("Invalid correlator type!")
            
        
        self.SoftCorr(CType.value, Nentries, phHist, NoCorrChannels, self.corrPy)
        if NoCorrChannels[0] != self.TotalCorrChanLen:
            warn("Number of correlator channels inconsistent!")
            
  

        self.lag = self.corrPy[1, :]*timebase_ms
        #corr.lag(corr.lag < 0) = 0; % fix zero channel time
        self.corrfunc = self.corrPy[0, :]
        self.weights = self.corrPy[2, :]
        validCorr = self.weights > 0
        self.lag = self.lag[validCorr]
        self.corrfunc = self.corrfunc[validCorr]
        self.weights = self.weights[validCorr]


    
    
        

        