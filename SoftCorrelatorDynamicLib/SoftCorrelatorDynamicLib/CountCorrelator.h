// CountCorrelator.h: interface for the CCountCorrelator class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_COUNTCORRELATOR_H__7C58BAA2_ACA0_4E9F_884C_18191362A8C2__INCLUDED_)
#define AFX_COUNTCORRELATOR_H__7C58BAA2_ACA0_4E9F_884C_18191362A8C2__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Correlator.h"

class CCountCorrelator : public CCorrelator  
{
public:
	CCountCorrelator();
	CCountCorrelator(long Stime, int Nchannels, double startTime);
	virtual ~CCountCorrelator();
	void ProcessEntry(EntryType Entry);
	//void ProcessEntry(EntryType Entry, bool isA, bool isB);
};

#endif // !defined(AFX_COUNTCORRELATOR_H__7C58BAA2_ACA0_4E9F_884C_18191362A8C2__INCLUDED_)
