#include <stdio.h>
#include <cuda.h>

using namespace std;

__global__ void ultimateTasker(int m ,int n,int *g_executionTime,int *g_priority,
int *g_allocationTime,int *g_allocatedCore, int *g_smallestFree,int *g_priorityAllocated,int *g_coreForPriority)

{
    /*
    initialize allocation time & core array
    */


    int tid = threadIdx.x;

    if(tid==0)
    {

        for(int i=0;i<n;i++)
        {
            g_allocationTime[i] = -1;
            g_allocatedCore[i] = -1;
        }

        for(int i=0;i<m;i++)
        {
            g_priorityAllocated[i] = -1;
            g_coreForPriority[i] = -1;
        }

    }

    int core_id = tid;
    int previous_task = -1;

    __syncthreads();


    /*
    loop over all the tasks
    */

    for(int i=0;i<n;i++)
    {
        /*
        STEP 0 : Initialize the smallest free array
        */
        int taskid = i;
        g_smallestFree[core_id] = 0;

        
        /*
        STEP 1 : check if taskid is allocated to this core(thread)
        */
        bool allocate = false;

        /*
        STEP 1a : compare my priorities with the current task

        */

        for(int j=0;j<m;j++)
        {
            if(g_priorityAllocated[j]==g_priority[taskid])
            {
                if(g_coreForPriority[j]==core_id)
                {
                    allocate = true;
                    g_allocatedCore[taskid] = core_id;
                }

                break;
            }
            
        }

        __syncthreads();

       
        /*
        STEP 1b : tasks which are not allocated any core are given a core from the smallestFree Array
        */

        if(g_allocatedCore[taskid]== -1)
        {
            // 1b1. Find whether i'm(core) free or not
            
            if(previous_task == -1)
            {
                g_smallestFree[core_id] = 0;
            } else
            {
              int freetime = g_allocationTime[previous_task]+ g_executionTime[previous_task];
              int eligibletime;

              if(taskid==0)
              {
                  eligibletime = 0;
              }else
              {
                  eligibletime = g_allocationTime[taskid-1];
              }  

              if(eligibletime>=freetime)
              {
                  g_smallestFree[core_id] = 0;
              }else
              {
                  g_smallestFree[core_id] = 1;
              }

              
            }

            __syncthreads();

            
            // 1b2. If free which is the smallest among us(cores)
            
            int k;

            for(k=0;k<m;k++)
            {
                if(g_smallestFree[k]==0)
                {
                    if(k==core_id)
                    {
                        g_allocatedCore[taskid] = core_id;
                        allocate = true;
                        
                        for(int j=0;j<m;j++)
                        {
                            if(g_priorityAllocated[j]== -1)
                            {
                                g_priorityAllocated[j] = g_priority[taskid];
                                g_coreForPriority[j] = core_id;
                                break;
                            }
                            
                        }
                    } 
                    break;
                }
                
            }
        

        }

        __syncthreads();

        /*
        STEP 2 : if allocated to this core, calc allocation time for taskid
        */

        if(allocate==true)
        {
            int x,y;

            if(i==0)
            {
                y=0;
            } else
            {
                y = g_allocationTime[i-1];
            }
            if(previous_task==-1)
            {
                x=0;
            } else
            {
                x= g_executionTime[previous_task]+g_allocationTime[previous_task];
            }

            g_allocationTime[i] = (x<y) ? y:x;
            previous_task = i;

        }


        __syncthreads();
    }

}

//Complete the following function
void operations ( int m, int n, int *executionTime, int *priority, int *result )  {

/*
initialising allocation time & allocated cores to each task = -1
*/

    int *g_allocationTime; //time at which ith task is allocated
    int *g_allocatedCore; //core on which ith task is allocated
    int *g_executionTime; //exe time for ith task
    int *g_priority;    //priority for ith task
    int *g_smallestFree;    //array to keep track of which are free
    int *g_priorityAllocated; // priorities that have been allocated 
    int *g_coreForPriority;     //core on which priority at ith index of priorityAllocated has been allocated

    cudaMalloc(&g_allocationTime,n*sizeof(int));
    cudaMalloc(&g_allocatedCore,n*sizeof(int));
    cudaMalloc(&g_executionTime,n*sizeof(int));
    cudaMalloc(&g_priority,n*sizeof(int));
    cudaMalloc(&g_smallestFree,m*sizeof(int));
    cudaMalloc(&g_priorityAllocated,m*sizeof(int));
    cudaMalloc(&g_coreForPriority,m*sizeof(int));

    cudaMemcpy(g_executionTime,executionTime,n*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(g_priority,priority,n*sizeof(int),cudaMemcpyHostToDevice);

    ultimateTasker<<<1,m>>>(m,n,g_executionTime,g_priority,g_allocationTime,
    g_allocatedCore,g_smallestFree,g_priorityAllocated,g_coreForPriority);
    
    cudaDeviceSynchronize();
    cudaMemcpy(result,g_allocationTime,n*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i=0;i<n;i++)
    {
        result[i] = result[i] + executionTime[i];
    }

    return;
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
