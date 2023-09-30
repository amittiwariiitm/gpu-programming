#include <stdio.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <stdlib.h>

//Input arrays


/*Host-Side Arrays*/

int *trainClassCount;		//No of classes in a train
int *trainSourceId;			//train source station number
int *trainDestinationId;	//train destination station number
bool *trainMovesForward;	//src>des or not
int *batchRequestCount;		//No of requests in a batch	
int **requestTrainId;		//requested train to be booked in the given <batch,request>
int **requestClassId;		//requested class number
int **requestSourceId;		//requested source station number
int **requestDestinationId; //requested destination station number
int **requestSeats;	//No of requested seats
int trainCount;		//No of trains
int batchCount;		//No of batches
int ***maxSeatsArray;		//max no of seats for the <train,class,station>   
int *startingSeatIndexForTrain;	// starting index for a train's information in the 1D seatsAvailableArray	

/*Device-Side Arrays*/

int *dtrainClassCount;
int *dtrainSourceId;
int *dtrainDestinationId;
bool *dtrainMovesForward;
int *dstartingSeatIndexForTrain;
int *dbatchRequestCount;
int *drequestTrainId;
int *drequestClassId;
int *drequestSourceId;
int *drequestDestinationId;
int *drequestSeats;
int *dseatsAvailableArray;

/*Output Arrays*/

bool *drequestResult;
int *dallottedSeats;

int getStationCount(int);
bool printDebugging = false; 
							 
void printEverything()
{	
	if(!printDebugging)
		return;

	fprintf(stderr, "%d\n" , trainCount);

	for (int train = 0; train < trainCount; train++) 
	{
		fprintf(stderr, "%d %d %d %d\n",train,trainClassCount[train], trainSourceId[train], trainDestinationId[train]);

		for(int classI = 0; classI < trainClassCount[train]; classI++ )
		{
			fprintf(stderr, "%d %d\n",classI,maxSeatsArray[train][classI][0]);
		}

	}

	fprintf(stderr,"%d\n",batchCount);

	for(int batch=0; batch<batchCount; batch++)
	{
		fprintf(stderr,"%d\n",batchRequestCount[batch]);

		for(int req=0; req<batchRequestCount[batch]; req++)
		{
			fprintf(stderr, "%d %d %d %d %d %d\n",req,requestTrainId[batch][req],requestClassId[batch][req],
			requestSourceId[batch][req],requestDestinationId[batch][req],requestSeats[batch][req]);
		}
	}
}


__device__ int dgetStationCount(int train, bool *dtrainMovesForward, int *dtrainSourceId, int *dtrainDestinationId) {
		int stationCount;

		if(dtrainMovesForward[train]) 
		{
			stationCount = dtrainDestinationId[train]-dtrainSourceId[train];
		}

		else
		{
			stationCount = dtrainSourceId[train]-dtrainDestinationId[train];
		}

		return stationCount;
}

/*
 * The main kernel which confirms(success or failure) of a booked ticket
 */
__global__ void bookingApplication(int batchId, int trainCount,int *dtrainClassCount,int *dtrainSourceId,int *dtrainDestinationId,bool *dtrainMovesForward,
		int brc,int *drequestTrainId,int *drequestClassId,int *drequestSourceId,int *drequestDestinationId,int *drequestSeats,
		int *dseatsAvailableArray,int* dstartingSeatIndexForTrain,bool *drequestResult,int *dallottedSeats, bool printDebugging)
{
	//Step 1: Obtain the trainId & classId
	int trainId = blockIdx.x * 32 + (threadIdx.x/32);  
	int classId = threadIdx.x % 32;

	int stationCount = dgetStationCount(trainId,dtrainMovesForward,dtrainSourceId,dtrainDestinationId);
	int seats = dtrainClassCount[trainId] * stationCount;

	/*
	 * Step 2: Use shared memory for most-frequently used arrays in each block.
	 * (Ones that are shared with the classes of a train.)
	 * Copy info from global to shared.
	 */
	const int sizer = 1250;
	__shared__ int mySeatAvailabiltyArray[sizer]; //per train
	
	if(classId==0 && trainId%32 == 0)
	{
		int j = 0;
		for(int i = dstartingSeatIndexForTrain[trainId]; i<seats; i++)
		{
			mySeatAvailabiltyArray[j++] = dseatsAvailableArray[i];
		}
	}
	__syncthreads(); 

	
	//Step 3: Iterate over each request in the current batch
	
	for(int req=0; req<brc; req++)
	{
		// 3a: Continue if the train id doesnt match 
		int reqTrainId = drequestTrainId[req];
		
		if(reqTrainId!=trainId)
		{
			continue;
		}

		// 3b: Continue if the class id doesnt match
		int reqClassId = drequestClassId[req];

		if(reqClassId!=classId)
		{
			continue;
		}

		// 3c: Mark request as failed if any station has less seats than requested

		int normalisedSource;
		int normalisedDestination;

		if(dtrainMovesForward[trainId])
		{
			normalisedSource = drequestSourceId[req] - dtrainSourceId[trainId];
			normalisedDestination = drequestDestinationId[req] - dtrainSourceId[trainId]; 
		}
		else 
		{
			normalisedSource = drequestDestinationId[req] - dtrainDestinationId[trainId];
			normalisedDestination = drequestSourceId[req] - dtrainDestinationId[trainId];
		}

		normalisedSource += (reqClassId) * stationCount;
		normalisedDestination += (reqClassId) * stationCount;

		bool failure = false;
		int reqCapacity = drequestSeats[req];
		
		for(int ns = normalisedSource; ns<normalisedDestination; ns++)
		{
			if(dseatsAvailableArray[dstartingSeatIndexForTrain[trainId] + ns]<reqCapacity)
			{
				failure = true;
				break;
			}
		}

		if(failure)
		{
			drequestResult[req] = false;
			dallottedSeats[req] = 0;
		}
		else
		{
			// 3d: Else mark request as succeeded 
			drequestResult[req] = true;
			dallottedSeats[req] = (normalisedDestination - normalisedSource) * reqCapacity;
			for(int ns = normalisedSource; ns<normalisedDestination; ns++)
			{
				dseatsAvailableArray[dstartingSeatIndexForTrain[trainId] + ns] -= reqCapacity;
			}
		} 

	}
}


/*
*The utility kernel that provides starting index of a train's maxSeatsArray 
*/
__global__ void indexCounter(int trainCount,int* dtrainClassCount,int* dtrainSourceId, int* dtrainDestinationId,
							bool* dtrainMovesForward,int* dstartingSeatIndexForTrain)
{
	int train = blockIdx.x*1024+threadIdx.x;

	if(train<trainCount)
	{
		int classCount = dtrainClassCount[train];
		int stationCount = dgetStationCount(train, dtrainMovesForward, dtrainSourceId, dtrainDestinationId);

		int trainMaxCapacity = stationCount*classCount;
		dstartingSeatIndexForTrain[train] = trainMaxCapacity;

	}
}

int main() {
	scanf("%d", &trainCount);
	/*
	 *  Step 1: Allocate memory for cpu & gpu for TRAIN INFO
	 */

	trainMovesForward = (bool*) malloc(sizeof(bool) * trainCount);
	cudaMalloc(&dtrainMovesForward,sizeof(bool) * trainCount);
	
	trainDestinationId = (int*) malloc(sizeof(int) * trainCount);
	cudaMalloc(&dtrainDestinationId,sizeof(int) * trainCount);
	
	trainSourceId = (int*) malloc(sizeof(int) * trainCount);
	cudaMalloc(&dtrainSourceId,sizeof(int) * trainCount);

	trainClassCount = (int*) malloc(sizeof(int) * trainCount);
	cudaMalloc(&dtrainClassCount,sizeof(int) * trainCount);

	/*
	 * END of Step 1.
	 * Step 2: Read the Input for TRAIN INFO, and initialize maxSeatsArray.
	 */

	maxSeatsArray = (int***) malloc(sizeof(int **) * trainCount);

	/* Note: This loop scans the data in CPU -- hence not parallelized in GPU. */
	for(int train=0;train<trainCount;train++)
	{
		int tn;
		scanf("%d%d%d%d", &tn, &trainClassCount[train], &trainSourceId[train], &trainDestinationId[train]);
		trainMovesForward[train] = trainSourceId[train]<trainDestinationId[train];

		maxSeatsArray[train] = (int**) malloc(sizeof(int*) * trainClassCount[train]);
		
		for (int classI=0; classI<trainClassCount[train]; classI++)
		{
			maxSeatsArray[train][classI] = (int*) malloc(sizeof(int) * getStationCount(train));
			int cn;
			int maxCapacity;
			scanf("%d%d", &cn, &maxCapacity);

			for(int station=0; station<getStationCount(train); station++)
			{
				maxSeatsArray[train][classI][station] = maxCapacity;
			} // stations
		
		} // classes

	} // trains

	/*
	 * END of Step 2
	 * Step 3: Read the Input for BATCH and REQUEST INFOs
	 */

	scanf("%d",&batchCount);
	batchRequestCount = (int*) malloc(sizeof(int) * batchCount);
	cudaMalloc(&dbatchRequestCount, sizeof(int) * batchCount);

	// Create space for first dimension (batch) of request arrays

	requestTrainId = (int**) malloc(sizeof(int*) * batchCount);
	requestClassId = (int**) malloc(sizeof(int*) * batchCount);
	requestSourceId = (int**) malloc(sizeof(int*) * batchCount);
	requestDestinationId = (int**) malloc(sizeof(int*) * batchCount);
	requestSeats = (int**) malloc(sizeof(int*) * batchCount);

	/* Note: Scans information -- not parallelized. */
	for(int batch=0 ; batch<batchCount ; batch++)
	{
		int brc;
		scanf("%d",&brc);
		batchRequestCount[batch] = brc;
		// Create space for second dimension (request) of request arrays

		requestTrainId[batch] = (int*) malloc(sizeof(int) * brc);
		requestClassId[batch] = (int*) malloc(sizeof(int) * brc);
		requestSourceId[batch] = (int*) malloc(sizeof(int) * brc);
		requestDestinationId[batch] = (int*) malloc(sizeof(int) * brc);
		requestSeats[batch] = (int*) malloc(sizeof(int) * brc);

		for(int req=0; req<brc; req++)
		{
			int rid;
			scanf("%d%d%d%d%d%d", &rid,&requestTrainId[batch][req], &requestClassId[batch][req], 
			&requestSourceId[batch][req], &requestDestinationId[batch][req], &requestSeats[batch][req]);

		}

	}	
	
	/*
	 * END of Step 3
	 * Step 4 : Data transfer from Cpu->Gpu for TRAIN INFO 
	 */

	cudaMemcpy(dtrainMovesForward, trainMovesForward, sizeof(bool) * trainCount, cudaMemcpyHostToDevice);
	cudaMemcpy(dtrainDestinationId, trainDestinationId , sizeof(int) * trainCount, cudaMemcpyHostToDevice);
	cudaMemcpy(dtrainSourceId, trainSourceId, sizeof(int) * trainCount, cudaMemcpyHostToDevice);
	cudaMemcpy(dtrainClassCount, trainClassCount, sizeof(int) * trainCount, cudaMemcpyHostToDevice);
	
	/*
	 * END of Step 4
	 * Step 5 : Data transfer from Cpu->Gpu for BATCH INFO 
	 */

	cudaMemcpy(dbatchRequestCount, batchRequestCount, sizeof(int) * batchCount, cudaMemcpyHostToDevice);

	/*
	 * END of Step 5
	 * Step 6 : Calculate 1D Array of size for the given train named d/v-startingSeatIndexForTrain 
	 */

	thrust::device_vector<int> vstartingSeatIndexForTrain(trainCount);
	dstartingSeatIndexForTrain = thrust::raw_pointer_cast(vstartingSeatIndexForTrain.data());

	int blockSize = ceil(trainCount*1.0/1024);
	int threadSize = 1024;

	indexCounter<<<blockSize,threadSize>>>(trainCount,dtrainClassCount,dtrainSourceId,
		dtrainDestinationId,dtrainMovesForward,dstartingSeatIndexForTrain);
	
	cudaDeviceSynchronize();
	
	int allButLast = vstartingSeatIndexForTrain[trainCount-1];

	thrust::exclusive_scan(vstartingSeatIndexForTrain.begin(), vstartingSeatIndexForTrain.end(), vstartingSeatIndexForTrain.begin());

	int last = vstartingSeatIndexForTrain[trainCount-1];

	/*
	 * END of Step 6
	 * Step 7 : Allocate & initialize 1D array corresponding to available seat capacity for <train,class,station>
	 * 			Initial value for every <train,class,station> is the maxSeatsArray<train,class,station>
	 * 			Note that the indexing in this array is done via d/vstartingSeatIndexForTrain 
	 */


	thrust::device_vector<int> vseatsAvailableArray(last+allButLast);
	
	//dseatsAvailableArray : to be used in the kernel for checking any seats avl info
	dseatsAvailableArray = thrust::raw_pointer_cast(vseatsAvailableArray.data());


	int indexer=0;

	/*
	 * Initialize the value for vseatsAvailableArray with Max Seats.
	 * Note: This loop isn't parallelized since maxSeatsArray is an unsymmetrical 3D array--
	 * 		 Indexing in this array would require constructing another prefix sum array; inefficient.
	 * 		 Note that maxSeatsArray is kept unsymmetrical to save space.
	 */

	for(int train=0; train<trainCount; train++)
	{
		for(int classId=0; classId<trainClassCount[train]; classId++)
		{
			for(int station=0; station<getStationCount(train);station++)
			{
				vseatsAvailableArray[indexer++] = maxSeatsArray[train][classId][station];
			}
		}
	}

	if(printDebugging)
	{
		fprintf(stderr,"1D form of the seats available for <train,class,station>\n");

		thrust::copy(vseatsAvailableArray.begin(), vseatsAvailableArray.end(), std::ostream_iterator<int>(std::cerr, " "));
		fprintf(stderr,"\n");
		
		fprintf(stderr,"Prefix sum for all the trains. \n");
		
		thrust::copy(vstartingSeatIndexForTrain.begin(), vstartingSeatIndexForTrain.end(), std::ostream_iterator<int>(std::cerr, " "));
		fprintf(stderr,"\n");
	}
	
	/*
	* END of Step 7
	*/


	for(int batcher=0; batcher<batchCount; batcher++)
	{
		int brc = batchRequestCount[batcher];

		/*
		 * Step 8 : Data transfer from Cpu->Gpu for REQUEST INFO 
		 */

		cudaMalloc(&drequestTrainId, sizeof(int) * brc);
		cudaMalloc(&drequestClassId, sizeof(int) * brc);
		cudaMalloc(&drequestSourceId, sizeof(int) * brc);
		cudaMalloc(&drequestDestinationId, sizeof(int) * brc);
		cudaMalloc(&drequestSeats, sizeof(int) * brc);

		cudaMemcpy(drequestTrainId, requestTrainId[batcher], sizeof(int) * brc, cudaMemcpyHostToDevice);
		cudaMemcpy(drequestClassId, requestClassId[batcher], sizeof(int) * brc, cudaMemcpyHostToDevice);
		cudaMemcpy(drequestSourceId, requestSourceId[batcher], sizeof(int) * brc, cudaMemcpyHostToDevice);
		cudaMemcpy(drequestDestinationId, requestDestinationId[batcher], sizeof(int) * brc, cudaMemcpyHostToDevice);
		cudaMemcpy(drequestSeats, requestSeats[batcher], sizeof(int) * brc, cudaMemcpyHostToDevice);

		/*
		 * END of Step 8 
		 * Step 9 : Allocate Output Arrays : d/v-requestResult & d/v-allottedSeats 
		 */

		thrust::device_vector<bool> vrequestResult(brc);
		thrust::device_vector<int> vallottedSeats(brc);

		drequestResult = thrust::raw_pointer_cast(vrequestResult.data());
		dallottedSeats = thrust::raw_pointer_cast(vallottedSeats.data());


		int blockSize = ceil(trainCount*1.0/32);
		dim3 block_a(blockSize,1,1);
		dim3 thread_a(1024, 1, 1);


		bookingApplication<<<block_a,thread_a>>>(batcher,trainCount,dtrainClassCount,dtrainSourceId,dtrainDestinationId,dtrainMovesForward,
		brc,drequestTrainId,drequestClassId,drequestSourceId,drequestDestinationId,drequestSeats,
		dseatsAvailableArray,dstartingSeatIndexForTrain,drequestResult,dallottedSeats, printDebugging);

		cudaDeviceSynchronize();

		if (printDebugging) 
		{
			fprintf(stderr,"Results for batch %d: ", batcher);
			thrust::copy(vrequestResult.begin(), vrequestResult.end(), std::ostream_iterator<int>(std::cerr, " "));
			fprintf(stderr, "\n");
			thrust::copy(vallottedSeats.begin(), vallottedSeats.end(), std::ostream_iterator<int>(std::cerr, " "));
			fprintf(stderr,"\n");
		}

		/*
		 * END of Step 8
		 * Step 9: Print the output for this batch
		 */
		int totalSuccesses = thrust::count(vrequestResult.begin(), vrequestResult.end(), 1);
		int seatsBooked = thrust::reduce(vallottedSeats.begin(), vallottedSeats.end());
		for (int i = 0; i < brc; i++) 
		{
			if (vrequestResult[i]) 
			{
				printf("success\n");
			}
			else
			{
				printf("failure\n");
			}
		}
		printf("%d %d\n", totalSuccesses, brc - totalSuccesses);
		printf("%d\n", seatsBooked);

	}

	//printEverything();

	return 0;
}


int getStationCount(int train)
{
	
	
	if(trainMovesForward[train]) 
	{
		return trainDestinationId[train]-trainSourceId[train];
	}

	else
	{
		return trainSourceId[train]-trainDestinationId[train];
	}

}
/* End of Program. */