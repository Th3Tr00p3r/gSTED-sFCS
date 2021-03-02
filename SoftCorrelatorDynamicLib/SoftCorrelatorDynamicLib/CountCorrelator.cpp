// CountCorrelator.cpp: implementation of the CCountCorrelator class.
//
//////////////////////////////////////////////////////////////////////

#include "CountCorrelator.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CCountCorrelator::CCountCorrelator()
{

}

CCountCorrelator::CCountCorrelator(long Stime, int Nchannels, double startTime) : CCorrelator(Stime, Nchannels, startTime)
{
}

CCountCorrelator::~CCountCorrelator()
{

}

void CCountCorrelator::ProcessEntry(EntryType Entry)
{
    if (DelayChainCleared)
    {
        NoChainClears++;
        DelayChainCleared = false;
    }
    
    if (countDown > 1)
	{
		countDown --;
		counter += Entry;
	}
	else //(countDown == 1)
	{
		counter += Entry;
		GetCounterIn();
	}

}   
