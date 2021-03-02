// Correlator.h: interface for the CCorrelator class.
//
// by Oleg Krichevsky, Ben-Gurion University, Dec. 2003
// okrichev@bgu.ac.il
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_CORRELATOR_H__F7ABBA25_2D5A_4868_960F_578E8A38E015__INCLUDED_)
#define AFX_CORRELATOR_H__F7ABBA25_2D5A_4868_960F_578E8A38E015__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <stdlib.h>
//#include <malloc.h>
//#include <iostream>
#include <math.h>
//#include "mex.h"

// typedef double EntryType;
typedef long EntryType;

// #include "winnt.h" for ulonglong



class CCorrelator  
{
public:
	void GetWeights(double *WeightsOut);
	void GetLags(double* lagOut);
    double GetLastLag();
	int GetNumOfAccumulators();
	void GetAccumulators(double* accout);
	virtual void ProcessEntry(EntryType Entry);
    virtual void ProcessEntry(EntryType Entry, long Valid);
	CCorrelator(long Stime, int Nchannels, double startTime);
	CCorrelator();
	virtual ~CCorrelator();

protected:
	double* lag;
	long Delay;
	int NumOfAccumulators;
	double* delayedCursor;
	void ShiftChannelsByN(double Nshifts);
	double* LastDelayChannel;
	void GetCounterIn();
	double* cursor;
	long smplTime;
	int NumOfChannels;
	long counter;
	long countDown;
	double maxdelay;
	double* delayChannel;
	double* accumulator;
	double TotalSamplingTimes;
	double TotalCounts;
	double FirstDelayTime;
    bool DelayChainCleared;
    int NoChainClears;

	double* SumZeroTimeCounts;  // <I(0)>
	double* SumDelayedTimeCounts; // <I(t)>
    
    void ClearDelayChain();
};

#endif // !defined(AFX_CORRELATOR_H__F7ABBA25_2D5A_4868_960F_578E8A38E015__INCLUDED_)
