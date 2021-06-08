// CorrArray.h: interface for the CCorrArray class.
//
// by Oleg Krichevsky, Ben-Gurion University, Dec. 2003
// okrichev@bgu.ac.il
//////////////////////////////////////////////////////////////////////

#ifndef _CORRARRAY_H_
#define _CORRARRAY_H_

#if !defined(AFX_ANARRAY_H__4E8A1080_5F80_11D3_AD32_00008633F9F7__INCLUDED_)
#define AFX_ANARRAY_H__4E8A1080_5F80_11D3_AD32_00008633F9F7__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000


#include "Correlator.h"
//#include "afxwin.h" never use in Matlab mexfiles


#define ARRAY_MAXSIZE  50

template <class CorrType, int MAXSIZE>
class CCorrArray  
{
private:
  int      mSize;
  CorrType  x[MAXSIZE];
public:
	long TotalLength;
	void GetAccumulators(double* corr);
	void ProcessEntry(EntryType Entry);
    void ProcessEntry(EntryType Entry, long Valid); // for use in image or angular scan correlation
	void ProcessEntry(EntryType Entry, bool isA, bool isB);
    void ProcessEntry(EntryType Entry, bool isA, bool isB, long Valid); // for use in image or angular scan cross correlation
	CCorrArray();
	CCorrArray(int NumOfCorrelators, int DoublingSize);
	virtual ~CCorrArray();

	int     GetSize  () const;
	CorrType GetAt    (int idx);

	//BOOL    RemoveAt (int idx);
	//BOOL    InsertAt (const CCorrelator& c, int idx);
	//BOOL    Add      (const CCorrelator& c);

	int    RemoveAt (int idx);
	int    InsertAt (const CorrType& c, int idx);
	int    Add      (const CorrType& c);
};


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

template <class CorrType, int MAXSIZE>
CCorrArray<CorrType, MAXSIZE>::CCorrArray()
:mSize(0),
TotalLength(0)
{   }

template <class CorrType, int MAXSIZE>
CCorrArray<CorrType, MAXSIZE>::CCorrArray(int NumOfCorrelators, int DoublingSize)
:mSize(0),
TotalLength(0)
{

	long SamplingTime = 1;

	//First correlator is twice the length
	Add(CorrType(SamplingTime, 2 * DoublingSize +1, 0));

	//double LastTime = (2*DoublingSize - 1) * SamplingTime;
    
    double LastTime = x[0].GetLastLag();
    
	for (int i = 1; i < NumOfCorrelators; i++)
	{
		SamplingTime = SamplingTime * 2;
		Add(CorrType(SamplingTime, DoublingSize, LastTime + SamplingTime)); // open THIS
        // Add(CorrType(SamplingTime, DoublingSize, 0)); // close THIS
		//LastTime += (DoublingSize - 1) * SamplingTime;
		//LastTime += DoublingSize  * SamplingTime;
        LastTime = x[i].GetLastLag();
	}
}

template <class CorrType, int MAXSIZE>
CCorrArray<CorrType, MAXSIZE>::~CCorrArray()
{   }

//BOOL
template <class CorrType, int MAXSIZE>
int CCorrArray<CorrType, MAXSIZE>::Add(const CorrType& c)
{
  x[mSize] = c;
  TotalLength += x[mSize].GetNumOfAccumulators();
  mSize++;
  return 0; //TRUE;
}

//BOOL
template <class CorrType, int MAXSIZE>
int CCorrArray<CorrType, MAXSIZE>::InsertAt(const CorrType& c, int idx)
{ long int NoErr = 0;
  if (mSize >= ARRAY_MAXSIZE)  
    return 1; //FALSE;          // full
  if (idx < 0 || idx > mSize) 
    return 1; //FALSE;          // illegal idx
  for (int i = mSize-1; i >= idx; i--) 
    x[i+1] = x[i];         // shift up

  x[idx] = c;              // insert
  mSize++;
  return NoErr; //TRUE;
}

//BOOL
template <class CorrType, int MAXSIZE>
int CCorrArray<CorrType, MAXSIZE>::RemoveAt(int idx)
{
  if (idx < 0 || idx >= mSize)
    return 1; //FALSE;    // illegal idx
  for (int i = idx; i < mSize-1; i++)
    x[i] = x[i+1];
  mSize--;
  return 0; //TRUE;
}

template <class CorrType, int MAXSIZE>
CorrType CCorrArray<CorrType, MAXSIZE>::GetAt(int idx)
{
   return x[idx];
}

template <class CorrType, int MAXSIZE>
int CCorrArray<CorrType, MAXSIZE>::GetSize() const
{
  return mSize;
}

template <class CorrType, int MAXSIZE>
void CCorrArray<CorrType, MAXSIZE>::ProcessEntry(EntryType Entry)
{
	for (int i = 0; i < mSize; i++)
		x[i].ProcessEntry(Entry);

}

template <class CorrType, int MAXSIZE>
void CCorrArray<CorrType, MAXSIZE>::ProcessEntry(EntryType Entry, bool isA, bool isB)
{
	for (int i = 0; i < mSize; i++)
		x[i].ProcessEntry(Entry, isA, isB);

}

template <class CorrType, int MAXSIZE>
void CCorrArray<CorrType, MAXSIZE>::ProcessEntry(EntryType Entry, long Valid)
{
    for (int i = 0; i < mSize; i++)
        x[i].ProcessEntry(Entry, Valid);
    
}

template <class CorrType, int MAXSIZE>
void CCorrArray<CorrType, MAXSIZE>::ProcessEntry(EntryType Entry, bool isA, bool isB, long Valid)
{
    for (int i = 0; i < mSize; i++)
        x[i].ProcessEntry(Entry, isA, isB, Valid);
    
}

template <class CorrType, int MAXSIZE>
void CCorrArray<CorrType, MAXSIZE>::GetAccumulators(double *corr)
{
	int CumLen = 0; 
	for (int i = 0; i < mSize; i++)
	{
        //mexPrintf("Correlator number: %d \n",  (long) i);
		x[i].GetAccumulators(corr + CumLen);
		x[i].GetLags(corr+CumLen+TotalLength);
		x[i].GetWeights(corr+CumLen+2*TotalLength);
		CumLen += x[i].GetNumOfAccumulators();
	}
}


#endif // !defined(AFX_ANARRAY_H__4E8A1080_5F80_11D3_AD32_00008633F9F7__INCLUDED_)

#endif //ifndef _CORRARRAY_H_

