// CPhDelayLifeTimeCrossCorrelator.cpp: implementation of the CCorrelator class.
//
// by Oleg Krichevsky, Ben-Gurion University, Sept. 2017
// okrichev@bgu.ac.il
//////////////////////////////////////////////////////////////////////

#include "CPhDelayLifeTimeCrossCorrelator.h"
//#include <math.h>
#include <cmath>
#include <cstdio>
#include <typeinfo>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CPhDelayLifeTimeCrossCorrelator::CPhDelayLifeTimeCrossCorrelator()
{
}


CPhDelayLifeTimeCrossCorrelator::CPhDelayLifeTimeCrossCorrelator(long Stime, int Nchannels, double startTime)
{
	smplTime = Stime;
	NumOfAccumulators = Nchannels;
	FirstDelayTime = startTime;
	Delay = FirstDelayTime/smplTime;
	NumOfChannels = NumOfAccumulators + Delay;
	maxdelay = FirstDelayTime + NumOfChannels * smplTime;
	counterA = 0;
    counterB = 0;
	countDown = smplTime;
	TotalCountsA = 0;
    TotalCountsB = 0;
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


	double time = FirstDelayTime;
	for (double* temp = lag; temp < lag + NumOfAccumulators; )
	{
		*temp++ = time;
		time += smplTime;
	}

}

CPhDelayLifeTimeCrossCorrelator::~CPhDelayLifeTimeCrossCorrelator()
{
//		free(delayChannel);
//		free(accumulator);
//		mexPrintf("Destroying correlators\n");
}

void CPhDelayLifeTimeCrossCorrelator::ClearDelayChain()
{
    counterA = 0;
    counterB = 0;
    countDown = smplTime;
    DelayChainCleared = true;
    NoChainClears++;
    
    cursor = delayChannel;
    delayedCursor = delayChannel + Delay;
    LastDelayChannel = delayChannel + NumOfChannels -1;
    
    for (double* tempDelayCh = delayChannel; tempDelayCh <= LastDelayChannel;)
        *tempDelayCh++ = 0;
}

void CPhDelayLifeTimeCrossCorrelator::GetCounterIn() // <A(0)*B(t)>
{
	if (cursor > delayChannel)
		cursor--;
	else cursor = LastDelayChannel;

	if (delayedCursor > delayChannel)
		delayedCursor--;
	else delayedCursor = LastDelayChannel;

	*cursor = counterB;
	double* accumcursor = accumulator;
	double* tempSZTC = SumZeroTimeCounts;
	double* tempSDTC = SumDelayedTimeCounts;

	double* tempcursor;

	if (delayedCursor >= cursor)
	{
		for ( tempcursor = delayedCursor; tempcursor <= LastDelayChannel; tempcursor ++ )
		{
			*accumcursor += (*tempcursor) * counterA;
			accumcursor ++;

			*tempSZTC += counterA;
			tempSZTC ++;

			*tempSDTC += (*tempcursor);
			tempSDTC ++;

		};

		for (tempcursor = delayChannel; tempcursor < cursor; tempcursor ++ )
		{
			*accumcursor += (*tempcursor) * counterA;
			accumcursor ++;

			*tempSZTC += counterA;
			tempSZTC ++;

			*tempSDTC += (*tempcursor);
			tempSDTC ++;

		};
	} 
	else
	{
		for (double* tempcursor = delayedCursor; tempcursor < cursor; tempcursor ++ )
		{
			*accumcursor += (*tempcursor) * counterA;
			accumcursor ++;

			*tempSZTC += counterA;
			tempSZTC ++;

			*tempSDTC += (*tempcursor);
			tempSDTC ++;

		};
	}


	countDown = smplTime;
	TotalSamplingTimes ++;
	TotalCountsA += counterA;
    TotalCountsB += counterB;
	counterA = 0;
    counterB = 0;
}



void CPhDelayLifeTimeCrossCorrelator::ProcessEntry(EntryType Entry, double belongToAch, double belongToBch)
{
//    if (smplTime == 1){
//        fprintf(stderr, "Process entry\n");
//        fprintf(stderr,"%fL\n", belongToAch);
//        fprintf(stderr,"%s\n", typeid(belongToAch).name());
//    };
    
    if (DelayChainCleared)
    {
        //NoChainClears++;
        DelayChainCleared = false;
    }
    
	if (Entry < countDown)
	{
		countDown -= Entry;
		counterA +=  belongToAch;
        counterB +=  belongToBch;
	}
	else if (Entry == countDown)
	{
        counterA +=  belongToAch;
        counterB +=  belongToBch;
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
            counterA =  belongToAch;
            counterB =  belongToBch;
;
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
				*cursor = belongToBch;
				counterA = 0;// belongToAch; changes Oleg 16.05.22
				counterB = 0;// changes Oleg 16.05.22
				if (FirstDelayTime < 1)
				{
					(*accumulator)+= belongToBch*belongToAch; //zeroth channel
					(*SumZeroTimeCounts)++;
					(*SumDelayedTimeCounts)++;
				};
				TotalSamplingTimes++;
				TotalCountsA += belongToAch;
                TotalCountsB += belongToBch;
			}
			else
			{
                //counterA +=  belongToAch;
                //counterB +=  belongToBch;
                counterA =  belongToAch;
                counterB =  belongToBch;
				countDown -= Entry;
			}
			
		}
	}
}

void CPhDelayLifeTimeCrossCorrelator::ProcessEntry(EntryType Entry)
{
    ProcessEntry(Entry, true, true);
}

void CPhDelayLifeTimeCrossCorrelator::ProcessEntry(EntryType Entry, bool Valid)
{
    if (Valid)
    {
        ProcessEntry(Entry, true, true);
    }
    else if (!DelayChainCleared)
        ClearDelayChain();

}

void CPhDelayLifeTimeCrossCorrelator::ProcessEntry(EntryType Entry, double belongToAch, double belongToBch, long Valid)
{
    //mexPrintf("%d \n",  Valid);
    
    switch (Valid)
    {
        case 1: ProcessEntry(Entry, belongToAch, belongToBch);
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
//            mexPrintf("CPhDelayLifeTimeCrossCorrelator::ProcessEntry(Entry, belongToAch, belongToBch, Valid) error: Valid can only be 1, -1 or -2.");
            fprintf(stderr, "CPhDelayLifeTimeCrossCorrelator::ProcessEntry(Entry, belongToAch, belongToBch, Valid) error: Valid can only be 1, -1 or -2\n");
              exit(1);

            
    }
}

    
void CPhDelayLifeTimeCrossCorrelator::GetAccumulators(double *accout)
{
    
    double accumTime = TotalSamplingTimes - NoChainClears*Delay;
    //long sqTime = smplTime * smplTime;
    double avCountsSqA = TotalCountsA / TotalSamplingTimes;
    double avCountsSqB = TotalCountsB / TotalSamplingTimes;
    double avCountsSq = avCountsSqA * avCountsSqB;
    
//    double* tempSZTC = SumZeroTimeCounts;
//    double* tempSDTC = SumDelayedTimeCounts;
    
    //mexPrintf("%f\n", TotalCountsA);
   // mexPrintf("%f   ", TotalCountsB);
   // mexPrintf("%f\n", TotalSamplingTimes);
    
//    if (smplTime == 1){         fprintf(stderr, "%ld\n", smplTime); //long
//         fprintf(stderr,"%fL\n", TotalCountsA);
//        fprintf(stderr,"%fL\n", TotalCountsB);
//         fprintf(stderr,"%fL\n", TotalSamplingTimes);
//     };

    
    
    for (double* temp = accumulator; (temp < (accumulator + NumOfAccumulators)); )
    {
        //*accout++ = *temp++/(avCountsSq*accumTime--) - 1;
        *accout++ = *temp++/(avCountsSq*accumTime) - 1;
        accumTime -= NoChainClears;
        //	*accout++ = (*temp++)*(accumTime--)/((*tempSZTC++)*(*tempSDTC++)) - 1;
        //	*accout++ = (*temp++)/((*tempSZTC++)*(*tempSDTC++)) - 1;
    }	
    
}

