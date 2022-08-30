// CPhDelayLifeTimeCrossCorrelator.h: interface for the CPhDelayLifeTimeCrossCorrelator class.
//
//////////////////////////////////////////////////////////////////////

#ifndef CPhDelayLifeTimeCrossCorrelator_
#define CPhDelayLifeTimeCrossCorrelator_


#include "Correlator.h"

class CPhDelayLifeTimeCrossCorrelator : public CCorrelator
{
public:
    CPhDelayLifeTimeCrossCorrelator();
    CPhDelayLifeTimeCrossCorrelator(long Stime, int Nchannels, double startTime);
    virtual ~CPhDelayLifeTimeCrossCorrelator();
    void ProcessEntry(EntryType Entry, double belongToAch, double belongToBch);
    void ProcessEntry(EntryType Entry);
    void ProcessEntry(EntryType Entry, bool Valid);
    void ProcessEntry(EntryType Entry, double belongToAch, double belongToBch, long Valid);
    void GetAccumulators(double* accout);
    
protected:
    double counterA;
    double counterB;
    double TotalCountsA;
    double TotalCountsB;
    void GetCounterIn();
    void ClearDelayChain();

};

#endif

