#include <stdio.h>
#include <cuda.h>

using namespace std;

//You can make this program more efficeint by using shared memory (For fast execution)

//You can also find min using atomicMin or using reduction
__device__ int findMin(int m, int* gpuCounter)
{   
    int min = gpuCounter[0];
    int index = 0;
    for(int i = 1; i < m; i++)
    {
        if(gpuCounter[i] < min)
        {  
            min = gpuCounter[i];
            index = i;
        }
    }
    return index;
}

__global__ void dkernel (int m, int n, int* gpuExecutionTime, int* gpuPriority, int* gpuFirst, int* gpuCounter, int* globalCounter, int* gpuResult)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    
     while (*globalCounter < n)
        {  
            if(id < m)
            {
        
            int globalCounterTemp = *globalCounter;
            int taskCoreNo;

            if (gpuPriority[globalCounterTemp] != gpuPriority[globalCounterTemp-1])
            {
                if (globalCounterTemp != 0)
                    {
                        int prev = gpuFirst[gpuPriority[globalCounterTemp - 1]]; 
                        if (gpuCounter[id] < gpuCounter[prev] - gpuExecutionTime[globalCounterTemp - 1])
                        {
                            gpuCounter[id] = gpuCounter[prev] - gpuExecutionTime[globalCounterTemp - 1];
                        }
                    }
            }

            //Checking if it is the first task of particular priority
            if (gpuFirst[gpuPriority[globalCounterTemp]] == -1)
            {
                //If not: then find free core
                              
                taskCoreNo = findMin(m, gpuCounter);
              
                gpuFirst[gpuPriority[globalCounterTemp]] = taskCoreNo;

            }
            else
            {
                //Aleady core number is known
                taskCoreNo =  gpuFirst[gpuPriority[globalCounterTemp]];
            }

            //If core on which task should run is same as threadID
            if(taskCoreNo == id)
            {

                if (globalCounterTemp == 0)
                {
                    //If it is a very very first task OR Task 0
                    gpuCounter[id] += gpuExecutionTime[globalCounterTemp]; 

                    gpuResult[globalCounterTemp] = gpuCounter[id];
                }
                else
                {
                    //Check if its priority is same as previous task
                    if (gpuPriority[globalCounterTemp] == gpuPriority[globalCounterTemp-1])
                    {
                        gpuCounter[id] += gpuExecutionTime[globalCounterTemp]; 
                        gpuResult[globalCounterTemp] = gpuCounter[id];
                    }
                    else
                    {
                            gpuCounter[id] += gpuExecutionTime[globalCounterTemp];
                            gpuResult[globalCounterTemp] = gpuCounter[id];                            
                        
                    }

                }

                //Go for next task in ready queue
                *globalCounter = *globalCounter + 1;
            }
            //Else do nothing


        }
            //Add a barrier here for synchronization
        __syncthreads();

    }

    // printf ("\nReached here %d", id);
}


//Complete the following function
void operations ( int m, int n, int *executionTime, int *priority, int *result )  {
	// Allocate Cuda memory, copy from host into cuda memory 
	
	// call the kernels for doing required computations
	
	// copy the result back
	
	//Initialization : This can be parallelized easily
	 int *first = (int *) malloc ( (m) * sizeof (int) );
    for ( int i=0; i<m; i++ )  {
        first[i] = -1;
    }
    
    int *counter = (int *) malloc ( (m) * sizeof (int) );
    for ( int i=0; i<m; i++ )  {
        counter[i] = 0;
    }
	
	 int *gpuResult;
    cudaMalloc( &gpuResult, sizeof(int) * (n) );
	cudaMemcpy(gpuResult, result, sizeof(int) * (n), cudaMemcpyHostToDevice);
	
    
    int *gpuFirst;
    cudaMalloc( &gpuFirst, sizeof(int) * (m) );
	cudaMemcpy(gpuFirst, first, sizeof(int) * (m), cudaMemcpyHostToDevice);

    
    int *gpuCounter;
    cudaMalloc( &gpuCounter, sizeof(int) * (m) );
	cudaMemcpy(gpuCounter, counter, sizeof(int) * (m), cudaMemcpyHostToDevice);

    int* globalCounter;
    int *initCounter = 0;
    cudaMalloc(&globalCounter, sizeof(int));
    cudaMemcpy(globalCounter, initCounter, sizeof(int), cudaMemcpyHostToDevice);


    int *gpuExecutionTime;
    cudaMalloc( &gpuExecutionTime, sizeof(int) * (n) );
	cudaMemcpy(gpuExecutionTime, executionTime, sizeof(int) * (n), cudaMemcpyHostToDevice);

    int *gpuPriority;
    cudaMalloc( &gpuPriority, sizeof(int) * (n) );
	cudaMemcpy(gpuPriority, priority, sizeof(int) * (n), cudaMemcpyHostToDevice);

	
    dkernel <<<1, m>>> (m, n, gpuExecutionTime, gpuPriority, gpuFirst, gpuCounter, globalCounter, gpuResult);
    cudaDeviceSynchronize();

	cudaMemcpy(result, gpuResult, n * sizeof(int), cudaMemcpyDeviceToHost);  
	

}

int main(int argc,char **argv)
{
    int m,n;
    //Input file pointer declaration
    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");
    
    //Checking if file ptr is NULL
    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &m );      //scaning for number of cores
    fscanf( inputfilepointer, "%d", &n );      //scaning for number of tasks
   
   //Taking execution time and priorities as input	
    int *executionTime = (int *) malloc ( n * sizeof (int) );
    int *priority = (int *) malloc ( n * sizeof (int) );
    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &executionTime[i] );
    }

    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &priority[i] );
    }

    //Allocate memory for final result output 
    int *result = (int *) malloc ( (n) * sizeof (int) );
    for ( int i=0; i<n; i++ )  {
        result[i] = 0;
    }
    
     cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);

    //==========================================================================================================
	
	operations ( m, n, executionTime, priority, result ); 
	
    //===========================================================================================================
    
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by function to execute is: %.6f ms\n", milliseconds);
    
    // Output file pointer declaration
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    //Total time of each task: Final Result
    for ( int i=0; i<n; i++ )  {
        fprintf( outputfilepointer, "%d ", result[i]);
    }

    fclose( outputfilepointer );
    fclose( inputfilepointer );
    
    free(executionTime);
    free(priority);
    free(result);
    
    
    
}
