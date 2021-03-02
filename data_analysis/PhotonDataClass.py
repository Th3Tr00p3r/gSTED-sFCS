#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 22:54:26 2020

@author: oleg
"""

import numpy as np



class PhotonDataClass:
    
   # def __init__(self):
    
    def DoConvertFPGAdataToPhotons(self, FPGAdata, Version = '7bytes noreset'):
        if Version == '7bytes noreset':
           GroupLen = 7
           maxval = 256**3
            
        
        # find brackets    
        I_248 = np.where(FPGAdata == 248)[0]
        I_254 = np.where(FPGAdata == 254)[0]
        II = np.intersect1d(I_248, I_254-GroupLen+1, assume_unique = True) #find those in registry
        
        #find different sections
        modII = II % GroupLen #remainder
        modII_true = (modII[:-1] == modII[1:]) # remove singular events that are naturally occuring
        modII_true = np.append(modII_true, [False])
        modII = modII[modII_true]
        I = II[modII_true]

        # split into sections with constant modII
        k = np.where(np.diff(modII) !=0)[0]
        print('Found ' + str(len(k)+1) + ' sections of lengths:')
        print(np.diff(np.concatenate(([0], k, [len(modII)]))))

                            
        counter = FPGAdata[I+1]*256**2 + FPGAdata[I+2]*256 + FPGAdata[I+3]
        time_stamps = np.diff(counter.astype(int))
        #J = find(time_stamps < 0);
        #find simple "inversions": the data with a missing byte
        # decrease in counter on data j+1, yet the next counter data (j+2) is
        # higher than j.
        J = np.where( np.logical_and((time_stamps[:-1] < 0), (time_stamps[:-1]+time_stamps[1:] > 0)))[0]
        J1 = np.where(time_stamps[J+1] > 256)[0]
        if J1.size != 0:
            print('Found ' + str(len(J1)) +' of missing byte data: fixing...')
            time_stamps[J[J1]] = time_stamps[J[J1]] + 256
            time_stamps[J[J1]+1] = time_stamps[J[J1]+1] - 256
            counter[J[J1]+1] = counter[J[J1]+1] + 256


        # checking: perhaps missing byte is turning the next byte do more corrections
        J2 = np.where((time_stamps[J] < 0) and (time_stamps[J+1] > 256))[0]
        time_stamps[J[J2]] = time_stamps[J[J2]] + 256**2 - 256 # remove back what we added at the previous step
        time_stamps[J[J2]+1] = time_stamps[J[J2]+1] - 256**2 + 256
        counter[J[J2]+1] = counter[J[J2]+1] + 256**2

        #last resort: if nothing helps: place the photon in the middle
        J3 = np.where(time_stamps[J] < 0)[0] 
        if len(J3) > 0:
            print(str(len(J3)) + ' ad hoc data fixing.')
            temp = (time_stamps[J[J3]] + time_stamps[J[J3]+1])/2
            time_stamps[J[J3]] = np.floor(temp)
            time_stamps[J[J3]+1] = np.ceil(temp)
            counter[J[J3]+1] = counter[J[J3]] + time_stamps[J[J3]]


        # some rare cases of a small negative number preceded by a relatively large
        # positive number: not treated above
        J4 = np.where( np.logical_and((time_stamps[1:] < 0), (time_stamps[:-1]+time_stamps[1:] > 0)))[0]
        if len(J4) >0:
            print(str(len(J4)) + ' more ad hoc data fixing.')
            temp = (time_stamps[J4] + time_stamps[J4+1])/2
            time_stamps[J4] = np.floor(temp)
            time_stamps[J4+1] = np.ceil(temp)
            counter[J4] = counter[J4+1] - time_stamps[J4]


        NoOfNotfixed = np.sum(np.logical_and((time_stamps < 0), (time_stamps > -256**2)))  # check what this means
        if NoOfNotfixed > 0:
            print(str(NoOfNotfixed) + ' data cannot be fixed.')

        J = np.where(time_stamps < 0)[0]
        J = np.setdiff1d(J, k); # remove section edges, i.e. treat each section separately 
        time_stamps[J] = time_stamps[J]+maxval

        coarse = FPGAdata[I+4]
        fine =  FPGAdata[I+5]
        

        self.counter = counter
 #       self.time_stamps = time_stamps
        self.fname = ''
        self.coarse = coarse
        self.fine = fine  
        self.runtime = np.cumsum(np.concatenate(([0], time_stamps)))
 #       if self.runtime(end) > flintmax:
 #           error('Runtime overrun intmax!')

        self.sectionEdges = np.array([np.append([1], [k]),  np.append([k], [len(self.coarse)])]);
        self.AllSectionEdges = np.array([]);
        self.counterEnds = np.array([counter[0], counter[-1]]);
#P.NoOfInversionProblems = NoOfInversionProblems;       

       
        