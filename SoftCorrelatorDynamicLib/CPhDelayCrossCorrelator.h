// CPhDelayCrossCorrelator.h: interface for the CPhDelayCrossCorrelator class.
//
//////////////////////////////////////////////////////////////////////

#ifndef CPhDelayCrossCorrelator_
#define CPhDelayCrossCorrelator_


#include "Correlator.h"

class CPhDelayCrossCorrelator : public CCorrelator
{
public:
	CPhDelayCrossCorrelator();
	CPhDelayCrossCorrelator(long Stime, int Nchannels, double startTime);
	virtual ~CPhDelayCrossCorrelator();
	void ProcessEntry(EntryType Entry, bool belongToAch, bool belongToBch);
    void ProcessEntry(EntryType Entry);
    void ProcessEntry(EntryType Entry, bool Valid);
    void ProcessEntry(EntryType Entry, bool belongToAch, bool belongToBch, long Valid);
    void GetAccumulators(double* accout);
    
protected:
    long counterA;
    long counterB;
    double TotalCountsA;
    double TotalCountsB;
    void GetCounterIn();
    void ClearDelayChain();

};

#endif

