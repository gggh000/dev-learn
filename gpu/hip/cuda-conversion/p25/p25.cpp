/**
 */

/**
 */

// System includes
#include <stdio.h>
#include "hip/hip_runtime.h"


__global__ void add( int a, int b, int * c) {
	*c  = a + b;
    *c = 100;
}

int main ( void ) {
	int c = 200;
	int * dev_c;
    int status = 1000;
    int device;

    hipSetDevice(device);
	hipMalloc(& dev_c, sizeof(int));
    hipLaunchKernelGGL(add, 1, 1, 0, 0, 20, 70, dev_c);
    status = hipMemcpy(dev_c, &c, sizeof(int), hipMemcpyHostToDevice);
    printf("cudaMemcpy status: %d\n.", status); 
	printf( " 20 + 70 = %d\n", c);
}
