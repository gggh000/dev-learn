
#include <stdio.h>
#include <stdbool.h>
#include "hip/hip_runtime.h"

#define ALLOC_NORMAL 1
#define ALLOC_PAGE_LOCKED 2
#define SIZE (10*1024*1024)

float hip_mem_alloc_test(int size, bool up, int allocType) {
    int debug = 0;
    hipEvent_t start, stop;
    int *a, *dev_a;
    float elapsedTime;

    if (debug) {
        printf("hip_mem_alloc_test: entered.\n");
        printf("creating events...");
    }

    hipEventCreate( &start);
    hipEventCreate( &stop);

    if (debug) 
        printf("allocating on host...");

    if (allocType == ALLOC_PAGE_LOCKED) {
        hipHostMalloc((void**)&a, size * sizeof(*a), hipHostRegisterDefault);
    } else if  (allocType == ALLOC_NORMAL ) {
        a = (int*) malloc(size*sizeof(a));
    }

    if (debug) 
        printf("allocating on gpu...");

    hipMalloc((void**)&dev_a, size * sizeof(&dev_a));

    hipEventRecord( start, 0);
    for (int i = 0; i < 100 ; i ++ ) {
        if (up) {  
            hipMemcpy(dev_a, a, size * sizeof(a), hipMemcpyHostToDevice);
        } else { 
            hipMemcpy(a, dev_a, size * sizeof(a), hipMemcpyDeviceToHost);
        }
               
    }
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime( &elapsedTime, start, stop);
    return elapsedTime;
}
int main()
{
    float elapsedTime;
    printf("main: entered.\n");

    float MB  = (float)100 * SIZE * sizeof(int) / 1024 / 1024;
    elapsedTime = hip_mem_alloc_test(SIZE, true, ALLOC_NORMAL);
    printf("Time using hipMalloc: %3.1f ms.\n", elapsedTime);
    printf("\tMB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));

    elapsedTime = hip_mem_alloc_test(SIZE, false, ALLOC_NORMAL);
    printf("Time using hipMalloc: %3.1f ms.\n", elapsedTime);
    printf("\tMB/s during copy down: %3.1f.\n", MB / (elapsedTime / 1000));

    elapsedTime = hip_mem_alloc_test(SIZE, true, ALLOC_PAGE_LOCKED);
    printf("Time using hipHostMalloc: %3.1f ms.\n", elapsedTime);
    printf("\tMB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));

    elapsedTime = hip_mem_alloc_test(SIZE, false, ALLOC_PAGE_LOCKED);
    printf("Time using hipHostMalloc: %3.1f ms.\n", elapsedTime);
    printf("\tMB/s during copy down: %3.1f.\n", MB / (elapsedTime / 1000));
    
}
