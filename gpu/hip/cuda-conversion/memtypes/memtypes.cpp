#include <stdio.h>
#include "hip/hip_runtime.h"

#define N 128
#define LOOPSTRIDE 4
__global__ void add(int *a, int*b, int *c) {
    //int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x ;
    //int tid = hipBlockIdx_x;
    int tid = hipThreadIdx_x;
    if (tid < N) {
    	c[tid] = a[tid] + b[tid];
    }
}

int main (void) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int i ;
    

    a = (int*)malloc(N * sizeof(int));
    b = (int*)malloc(N * sizeof(int));
    c = (int*)malloc(N * sizeof(int));
 	hipMalloc(&dev_a, N * sizeof(int) );
 	hipMalloc(&dev_b, N * sizeof(int) );
 	hipMalloc(&dev_c, N * sizeof(int) );

	for (int i = 0; i < N ; i ++ ) {
		a[i] = i;
		b[i] = i + 2;
		c[i] = 999;
	}

	for (int i = 0; i < N ; i+=LOOPSTRIDE ) {
        printf("Before add: a/b/c: %04u, %04u %04u.\n", a[i], b[i], c[i]);
	}

   	hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
   	hipMemcpy(dev_b, b, N * sizeof(int), hipMemcpyHostToDevice);
   	hipMemcpy(dev_c, c, N * sizeof(int), hipMemcpyHostToDevice);
    
    // No. of workgroups.
    // threads/block, workgroup size.

    const unsigned threadsPerBlock = 128;
    //const unsigned blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    const unsigned blocks = N / threadsPerBlock;
    printf("Threads/Blocks: %u, %u\n", threadsPerBlock, blocks);
    hipLaunchKernelGGL(add, blocks, threadsPerBlock, 0, 0, dev_a, dev_b, dev_c);
    hipMemcpy(c, dev_c, N * sizeof(int), hipMemcpyDeviceToHost);

	for (int i = 0; i < N; i+=LOOPSTRIDE )
		printf("After add: N %02u: %04u + %04u = %04u\n", i, a[i], b[i], c[i]);

    hipFree(dev_a);
    hipFree(dev_b);
    hipFree(dev_c);
    free(a);
    free(b);
    free(c);
    
	return 0;
}
