//
//  SoftCorrelatorDynamicLib.hpp
//  SoftCorrelatorDynamicLib
//
//  Created by Oleg on 11/01/2021.
//  Copyright © 2021 Oleg. All rights reserved.
//

#ifndef SoftCorrelatorDynamicLib_
#define SoftCorrelatorDynamicLib_



/* The classes below are exported */

#pragma GCC visibility push(default)

#include <typeinfo>
#include <cstdio>
//#include <stdio.h>
#include "Correlator.h"
#include "CountCorrelator.h"
#include "CPhDelayCrossCorrelator.h"
#include "CorrArray.h"
//#include <afxtempl.h>  //support for array template
//#include <afxwin.h>
#define PhDelayCorrelator 1
#define PhCountCorrelator 2
#define PhDelayCrossCorrelator 3
#define PhCountCrossCorrelator 4 // 1st column is photon arrival times, 2nd column boolean vector with 1s for photons arriving on the A channel and  0s otherwise, and the 3rd column is 1s for photons arriving on B channel and 0s otherwise
#define PhDelayCorrelatorLines 5 // additional column with 1s for photons that have to be counted, on 0s the delay chain is reset; used in image correlation and in angular scan
#define PhDelayCrossCorrelatorLines 6 // as PhCountCrossCorrelator with an additional column with 1s for photons that have to be counted, on 0s the delay chain is reset; used in image correlation and in angular scan

#define ARRAY_MAXSIZE  50

#define DoublingSize 16
#define NumOfCorrelators 24
#define FirstSamplingTime 1
//#define phHistType int32_t


 
extern "C" void softwareCorrelator(
    int CorrelatorType,
    long Nentries,  // number of supplied photons (rows in the array)
    const EntryType *phHist,
    long *NoCorrChannels,           // number of correlator channels
    double* corr
);

extern "C" void getCorrelatorParams(int* DoubSize, int* NumCorr);

extern "C" void c_square(int n, double *array_in, double *array_out);

/*
class SoftCorrelatorDynamicLib
{
    public:
    void HelloWorld(const char *);
};

*/
#pragma GCC visibility pop
#endif
