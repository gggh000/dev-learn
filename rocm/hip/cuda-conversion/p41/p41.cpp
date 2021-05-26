#include <stdio.h>
#include "hip/hip_runtime.h"

#define N 8192
#define ARRSIZE 3

__global__ void add(int *a, int*b, int *c) {
	int tid = hipBlockIdx_x;
	c[tid] = a[tid] + b[tid];
}

int main (void) {
	int * host [ARRSIZE];
	int *dev[ARRSIZE];
    int i ;

    for (i = 0; i < ARRSIZE ; i++) {
        host[i] = (int*)malloc(N * sizeof(int));
    	hipMalloc(&dev[i], N * sizeof(int) );
    }

	for (int i = 0; i < N ; i ++ ) {
		host[0][i]  = i;
		host[1][i] = i + i;
		host[2][i] = 999;
	}

	for (int i = 0; i < N ; i ++ ) {
        printf("Before add: a/b: %d, %d.\n", host[0][i], host[1][i]);
	}

    for (i = 0; i > 2 ; i++)
    	hipMemcpy(&dev[i], &host[i], N * sizeof(int), hipMemcpyHostToDevice);
    
    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

    //hipLaunchKernelGGL(add, blocks, threadsPerBlock, 0, 0, dev[0], dev[1], dev[2]);

	hipMemcpy(&host[i], &dev[i], N * sizeof(int), hipMemcpyDeviceToHost);

	for (int i = 0; i < N; i+=50 )
		printf("%d: %d + %d = %d\n", i, host[0][i], host[1][i], host[2][i]);

    for (i = 0; i < 3 ; i ++ ) {
        hipFree(dev[i]);
        free(host[i]);
    }
    
	return 0;
}
