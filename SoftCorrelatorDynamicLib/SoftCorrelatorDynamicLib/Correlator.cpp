// Correlator.cpp: implementation of the CCorrelator class.
//lag
// by Oleg Krichevsky, Ben-Gurion University, Dec. 2003
// okrichev@bgu.ac.il
//////////////////////////////////////////////////////////////////////

#include "Correlator.h"
#include <math.h>
#include <cstdio>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CCorrelator::CCorrelator()
{
}


CCorrelator::CCorrelator(long Stime, int Nchannels, double startTime)
{
	smplTime = Stime;
	NumOfAccumulators = Nchannels;
	FirstDelayTime = startTime;
	Delay = FirstDelayTime/smplTime;
	NumOfChannels = NumOfAccumulators + Delay;
	maxdelay = FirstDelayTime + NumOfChannels * smplTime;
	counter = 0;
	countDown = smplTime;
	TotalCounts = 0;
	TotalSamplingTimes = 0;
    DelayChainCleared = true;
    NoChainClears = 1;
 
	delayChannel = (double *)calloc( NumOfChannels, sizeof( double ) );
	accumulator = (double *)calloc( NumOfAccumulators, sizeof( double ) );
	lag = (double *)calloc( NumOfAccumulators, sizeof( double ) );

	SumZeroTimeCounts = (double *)calloc( NumOfAccumulators, sizeof( double ) );
	SumDelayedTimeCounts = (double *)calloc( NumOfAccumulators, sizeof( double ) );


	cursor = delayChannel;
	delayedCursor = delayChannel + Delay;

	LastDelayChannel = delayChannel + NumOfChannels -1;
	
	if ((delayChannel == NULL) || (accumulator == NULL) || (lag == NULL) || (SumZeroTimeCounts == NULL) || (SumDelayedTimeCounts == NULL))
	{
		//#ifdef _WIN32
//		mexErrMsgTxt("Low memory! Cannot allocate arrays!");
	//	#else
			// cout << "Low memory! Cannot allocate arrays!";
	//	#endif
        fprintf(stderr, "Nonexistent correlator type!\n");
              exit(1);

		
		free(delayChannel);
		free(accumulator);
		free(lag);
		free(SumZeroTimeCounts);
		free(SumDelayedTimeCounts);
		return;
	};

    // mexPrintf("Allocating correlators\n");
	
	for (double* tempDelayCh = delayChannel; tempDelayCh <= LastDelayChannel;)
		*tempDelayCh++ = 0;


	double* tempSZTC = SumZeroTimeCounts;
	double* tempSDTC = SumDelayedTimeCounts;
	for (double* tempAcc = accumulator; tempAcc < accumulator + NumOfAccumulators; )
	{
		*tempAcc++ =0;
		*tempSZTC++ = 0;
		*tempSDTC++ = 0;
	};


    double time = FirstDelayTime;// - smplTime/2; // middle of the range
	for (double* temp = lag; temp < lag + NumOfAccumulators; )
	{
		*temp++ = time;
		time += smplTime;
	}

}

CCorrelator::~CCorrelator()
{
//		free(delayChannel);
//		free(accumulator);
//		mexPrintf("Destroying correlators\n");
}

void CCorrelator::ClearDelayChain()
{
    counter = 0;
    countDown = smplTime;
    DelayChainCleared = true;
    NoChainClears++;
    
    cursor = delayChannel;
    delayedCursor = delayChannel + Delay;
    LastDelayChannel = delayChannel + NumOfChannels -1;
    
    for (double* tempDelayCh = delayChannel; tempDelayCh <= LastDelayChannel;)
        *tempDelayCh++ = 0;
    
}

void CCorrelator::GetCounterIn()
{
    if (cursor > delayChannel)
		cursor--;
	else cursor = LastDelayChannel;

	if (delayedCursor > delayChannel)
		delayedCursor--;
	else delayedCursor = LastDelayChannel;

	*cursor = counter;
	double* accumcursor = accumulator;
	double* tempSZTC = SumZeroTimeCounts;
	double* tempSDTC = SumDelayedTimeCounts;

	double* tempcursor;

	if (delayedCursor >= cursor)
	{
		for ( tempcursor = delayedCursor; tempcursor <= LastDelayChannel; tempcursor ++ )
		{
			*accumcursor += (*tempcursor) * counter;
			accumcursor ++;

			*tempSZTC += counter;
			tempSZTC ++;

			*tempSDTC += (*tempcursor);
			tempSDTC ++;

		};

		for (tempcursor = delayChannel; tempcursor < cursor; tempcursor ++ )
		{
			*accumcursor += (*tempcursor) * counter;
			accumcursor ++;

			*tempSZTC += counter;
			tempSZTC ++;

			*tempSDTC += (*tempcursor);
			tempSDTC ++;

		};
	} 
	else
	{
		for (double* tempcursor = delayedCursor; tempcursor < cursor; tempcursor ++ )
		{
			*accumcursor += (*tempcursor) * counter;
			accumcursor ++;

			*tempSZTC += counter;
			tempSZTC ++;

			*tempSDTC += (*tempcursor);
			tempSDTC ++;

		};
	}


	countDown = smplTime;
	TotalSamplingTimes ++;
	TotalCounts += counter;
	counter = 0;
}

void CCorrelator::ShiftChannelsByN(double Nshifts)
{
	for (int i=0; i < Nshifts; i++)
	{
		if (cursor  > delayChannel)
			cursor--;
		else
			cursor = LastDelayChannel;
		*cursor = 0;
		
		if (delayedCursor  > delayChannel)
			delayedCursor--;
		else
			delayedCursor = LastDelayChannel;
	};

	TotalSamplingTimes += Nshifts;
}



void CCorrelator::ProcessEntry(EntryType Entry)
{
    if (DelayChainCleared)
    {
   //     NoChainClears++;
        DelayChainCleared = false;
    }
    
	if (Entry < countDown)
	{
		countDown -= Entry;
		counter++;
	}
	else if (Entry == countDown)
	{
		counter++;
		GetCounterIn();
	}
	else   //Entry > countDown
	{
		Entry -= countDown;
		GetCounterIn();

		double Nshifts = floorf((Entry-1)/smplTime);
		if (Nshifts <= NumOfChannels)
		{
			ShiftChannelsByN(Nshifts);
			Entry -= Nshifts * smplTime;
			counter = 1;
			if (Entry == smplTime)
				GetCounterIn();
			else 
				countDown -= Entry;
		}
		else  // Nshifts > NumOfChannels
		{
			for (cursor = delayChannel; cursor <= LastDelayChannel;)
				*cursor++ = 0;
			
			cursor = delayChannel;
			delayedCursor = cursor + Delay;
			TotalSamplingTimes += Nshifts;
			Entry -= Nshifts * smplTime;
			
			if (Entry == smplTime)
			{
				*cursor = 1;
				if (FirstDelayTime < 1)
				{
					(*accumulator)++; //zeroth channel
					(*SumZeroTimeCounts)++;
					(*SumDelayedTimeCounts)++;
				};
				TotalSamplingTimes++;
				TotalCounts++;
			}
			else
			{
				counter = 1;
				countDown -= Entry;
			}
			
		}
	}
}   

void CCorrelator::ProcessEntry(EntryType Entry, long Valid)
{
    //mexPrintf("%d \n",  Valid);
    
    switch (Valid)
    {
        case 1: ProcessEntry(Entry);
                break;
        case -1: // new line beginning
                if (!DelayChainCleared) // the chain is already cleared before the first line
                    ClearDelayChain();
                break;
        case -2: //line ending
                if (Entry >= countDown)
                {
                    Entry -= countDown;
                    GetCounterIn();
                    double Nshifts = floorf(Entry/smplTime);
                    TotalSamplingTimes += Nshifts;
                    DelayChainCleared = false;
                }
                break;
        default : //mexErrMsgIdAndTxt("CCorrelator::ProcessEntry(Entry, Valid)",
                   //                 "Valid can only be 1, -1 or -2.");
//            mexPrintf("CCorrelator::ProcessEntry(Entry, Valid) error: Valid can only be 1, -1 or -2.");
        fprintf(stderr, "CCorrelator::ProcessEntry(Entry, Valid) error: Valid can only be 1, -1 or -2!\n");
              exit(1);
        
    }
    
/*
    if (Valid == 1)
    {
        ProcessEntry(Entry);
    }
    else if (Valid == -1) // new line beginning
            if (!DelayChainCleared) // the chain is already cleared before the first line
                ClearDelayChain();
    else //must be -2
 */
}

void CCorrelator::GetAccumulators(double *accout)
{
	
	double accumTime = TotalSamplingTimes - NoChainClears*Delay;
	//long sqTime = smplTime * smplTime;
	double avCountsSq = TotalCounts / TotalSamplingTimes;
	avCountsSq = avCountsSq * avCountsSq;

//	double* tempSZTC = SumZeroTimeCounts;
//	double* tempSDTC = SumDelayedTimeCounts;

    /*
    mexPrintf("TotalCounts: %d \n",  (long) TotalCounts);
    mexPrintf("AccumTime: %d \n",  (long) accumTime);
    mexPrintf("TotalSamplingTimes: %d \n",   (long) TotalSamplingTimes);
    mexPrintf("NoChainClears: %d \n",  NoChainClears);
    mexPrintf("avCountsSq: %g \n",  avCountsSq);
   */
    
	for (double* temp = accumulator; (temp < (accumulator + NumOfAccumulators)); )
	{
        //*accout++ = *temp++/(avCountsSq*accumTime--) - 1;
		*accout++ = *temp++/(avCountsSq*accumTime) - 1; // open THIS
        //*accout++ = *temp++; // close THIS
        accumTime -= NoChainClears;
	//	*accout++ = (*temp++)*(accumTime--)/((*tempSZTC++)*(*tempSDTC++)) - 1;
	//	*accout++ = (*temp++)/((*tempSZTC++)*(*tempSDTC++)) - 1;

	}	
	
}

int CCorrelator::GetNumOfAccumulators()
{
	return NumOfAccumulators;
}

void CCorrelator::GetLags(double *lagOut)
{
	for (double* temp = lag; (temp < (lag + NumOfAccumulators)); )
        *lagOut++ = *temp++ ; // (double) smplTime/2.0;
}

double CCorrelator::GetLastLag()
{
    double* temp = lag  + NumOfAccumulators - 1;// + smplTime/2 //right boundary of the last channel
    return *temp;
}

void CCorrelator::GetWeights(double *WeightsOut)
{
	double StartingWeight = TotalSamplingTimes - NoChainClears*Delay;
	for (int i=0; i < NumOfAccumulators; i++)
    {
		//*WeightsOut++ = StartingWeight--;
        *WeightsOut++ = StartingWeight;
        StartingWeight -= NoChainClears;
    }
}
