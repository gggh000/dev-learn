#include <stdio.h>
#include "hip/hip_runtime.h"

#define N 256
#define LOOPSTRIDE 4
__global__ void add(int *a, int*b, int *c, int *d) {
    //int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x ;
    int tid = hipBlockIdx_x;
    //int tid = hipThreadIdx_x;
    if (tid < N) {
    	c[tid] = a[tid] + b[tid];
    }
    d[tid] = tid;
}

int main (void) {
    int *a, *b, *c, *d;
    int *dev_a, *dev_b, *dev_c, *dev_d;
    int i ;
    

    a = (int*)malloc(N * sizeof(int));
    b = (int*)malloc(N * sizeof(int));
    c = (int*)malloc(N * sizeof(int));
    d = (int*)malloc(N * sizeof(int));
 	hipMalloc(&dev_a, N * sizeof(int) );
 	hipMalloc(&dev_b, N * sizeof(int) );
 	hipMalloc(&dev_c, N * sizeof(int) );
 	hipMalloc(&dev_d, N * sizeof(int) );

	for (int i = 0; i < N ; i ++ ) {
		a[i]  = i + 1;
		b[i] = i + i + 2;
		c[i] = 999;
		c[i] = 888;
	}

	for (int i = 0; i < N ; i+=LOOPSTRIDE ) {
        printf("Before add: a/b: %04d, %04d.\n", a[i], b[i]);
	}

   	hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
   	hipMemcpy(dev_b, b, N * sizeof(int), hipMemcpyHostToDevice);
   	hipMemcpy(dev_c, c, N * sizeof(int), hipMemcpyHostToDevice);
    
    // No. of workgroups.
    // threads/block, workgroup size.

    const unsigned blocks = 256;
    const unsigned threadsPerBlock = 1;

    hipLaunchKernelGGL(add, blocks, threadsPerBlock, 0, 0, dev_a, dev_b, dev_c, dev_d);

    hipMemcpy(a, dev_a, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(b, dev_b, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(c, dev_c, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(d, dev_d, N * sizeof(int), hipMemcpyDeviceToHost);

	for (int i = 0; i < N; i+=LOOPSTRIDE )
		printf("After add: N: %02d/TID: 0x%08x: %04u + %04u = %04u\n", i, d[i], a[i], b[i], c[i]);

    hipFree(dev_a);
    hipFree(dev_b);
    hipFree(dev_c);
    free(a);
    free(b);
    free(c);
    
	return 0;
}
