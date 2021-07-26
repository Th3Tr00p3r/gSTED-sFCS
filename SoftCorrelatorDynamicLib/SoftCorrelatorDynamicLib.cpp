//
//  SoftCorrelatorDynamicLib.cpp
//  SoftCorrelatorDynamicLib
//
//  Created by Oleg on 11/01/2021.
//  Copyright © 2021 Oleg. All rights reserved.
//

//#include <iostream>
#include "SoftCorrelatorDynamicLib.hpp"
//#include "SoftCorrelatorDynamicLibPriv.hpp"

/*
void SoftCorrelatorDynamicLib::HelloWorld(const char * s)
{
    SoftCorrelatorDynamicLibPriv *theObj = new SoftCorrelatorDynamicLibPriv;
    theObj->HelloWorldPriv(s);
    delete theObj;
};

void SoftCorrelatorDynamicLibPriv::HelloWorldPriv(const char * s) 
{
    std::cout << s << std::endl;
};

 */
/*
#include <typeinfo>
#include <cstdio>
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

*/
extern "C" void softwareCorrelator(
    int CorrelatorType,
    long Nentries,  // number of supplied photons (rows in the array)
    const EntryType *phHist,
    long* NoCorrChannels,           // number of correlator channels
    double* corr  // Array of right hand side arguments
)
{

    if (CorrelatorType == PhDelayCorrelator)
    {
        CCorrArray<CCorrelator, NumOfCorrelators> CorrelatorArray(NumOfCorrelators, DoublingSize);
        
        const EntryType* HistEnd = phHist + Nentries;
        for (; phHist < HistEnd; )
            CorrelatorArray.ProcessEntry(*phHist++);
    
        CorrelatorArray.GetAccumulators(corr);
        *NoCorrChannels = CorrelatorArray.TotalLength;
            
    }
    else if (CorrelatorType == PhCountCorrelator)
    {
        CCorrArray<CCountCorrelator, NumOfCorrelators> CorrelatorArray(NumOfCorrelators, DoublingSize);
/*
        corr = (double_t *)calloc(CorrelatorArray.TotalLength*3, sizeof(double_t));
        if ( corr == NULL ) {
        fprintf(stderr, "Failed to allocate space for A!\n");
        exit(1);
        }
 */
        const EntryType* HistEnd = phHist + Nentries;


//        double* HistEnd = phHist + Nentries;
        for (; phHist < HistEnd; )
            CorrelatorArray.ProcessEntry(*phHist++);
    
        CorrelatorArray.GetAccumulators(corr);
        *NoCorrChannels = CorrelatorArray.TotalLength;

    }
    
    else if (CorrelatorType == PhDelayCrossCorrelator)
    {
        CCorrArray<CPhDelayCrossCorrelator, NumOfCorrelators> CorrelatorArray(NumOfCorrelators, DoublingSize);
                

        const EntryType* HistEnd = phHist + Nentries;
        const EntryType* belongToAch = phHist + Nentries;
        const EntryType* belongToBch = phHist + 2*Nentries;
        
        
        for (; phHist < HistEnd; )
            CorrelatorArray.ProcessEntry(*phHist++, (bool)*belongToAch++, (bool)*belongToBch++);
        
        CorrelatorArray.GetAccumulators(corr);
        *NoCorrChannels = CorrelatorArray.TotalLength;
        
    }
    
    else if (CorrelatorType == PhDelayCorrelatorLines)
    {
        CCorrArray<CCorrelator, NumOfCorrelators> CorrelatorArray(NumOfCorrelators, DoublingSize);
        
        
        const EntryType* HistEnd = phHist + Nentries;
        const EntryType* valid = phHist + Nentries; // pointer to the validity column

        
        for (; phHist < HistEnd; )
            CorrelatorArray.ProcessEntry(*phHist++, (long)*valid++);
        
        CorrelatorArray.GetAccumulators(corr);
        *NoCorrChannels = CorrelatorArray.TotalLength;
        
    }
    
    else if (CorrelatorType == PhDelayCrossCorrelatorLines)
    {
        CCorrArray<CPhDelayCrossCorrelator, NumOfCorrelators> CorrelatorArray(NumOfCorrelators, DoublingSize);
        
        const EntryType* HistEnd = phHist + Nentries;
        const EntryType* belongToAch = phHist + Nentries;
        const EntryType* belongToBch = phHist + 2*Nentries;
        const EntryType* valid = phHist + 3*Nentries; // pointer to the validity column

        
        for (; phHist < HistEnd; )
            CorrelatorArray.ProcessEntry(*phHist++, (bool)*belongToAch++, (bool)*belongToBch++, (long)*valid++);
        
        CorrelatorArray.GetAccumulators(corr);
        *NoCorrChannels = CorrelatorArray.TotalLength;
        
    }
    
    else
    {
        fprintf(stderr, "Nonexistent correlator type!\n");
              exit(1);
   //     std::cout << "Size: " << size << "\n";
    }

}

extern "C" void getCorrelatorParams(int* DoubSize, int* NumCorr)
{
    *DoubSize = DoublingSize;
    *NumCorr = NumOfCorrelators;
}

extern "C" void c_square(int n, double *array_in, double *array_out)
{ //return the square of array_in of length n in array_out
//    fprintf(stdout, "square\n");
    int i;

    for (i = 0; i < n; i++)
    {
        array_out[i] = array_in[i] * array_in[i];
    }
}


