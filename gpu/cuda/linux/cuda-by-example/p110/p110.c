/*
This is not working due to cpubitmap initiation causing error.
*/
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "cpu_bitmap.h"n

int main(void) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(&start, 0);

    CPUBitmap bitmap(DIM, DIM);

    return 0;
}


