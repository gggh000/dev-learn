#include <stdio.h>
#include "hip/hip_runtime.h"

// 1. if N is set to up to 1024, then sum is OK.
// 2. Set N past the 1024 which is past No. of threads per blocks, and then all iterations of sum results in 
// even the ones within the block.

// 3. To circumvent the problem described in 2. above, since if N goes past No. of threads per block, we need multiple block launch.
// The trick is describe in p65 to use formula (N+127) / 128 for blocknumbers so that when block number starts from 1, it is 
// (1+127) / 128.

#define N 2048
#define MAX_THREAD_PER_BLOCK 1024

__global__ void add( int * a, int * b, int * c ) {
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x ;
    if (tid < N) 
        c[tid] = a[tid] + b[tid];
}    

int main (void) {
    int *a, *b, *c;
	int *dev_a, *dev_b, *dev_c;

	// allocate dev memory for N size for pointers declared earlier.

    printf("\nAllocating memory...");
    a = (int*)malloc(N * sizeof(int));
    b = (int*)malloc(N * sizeof(int));
    c = (int*)malloc(N * sizeof(int));
	hipMalloc( (void**)&dev_a, N * sizeof(int));
	hipMalloc( (void**)&dev_b, N * sizeof(int));
	hipMalloc( (void**)&dev_c, N * sizeof(int));

	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i+2;
		c[i] = 999;
	}

	// copy the initialized local memory values to device memory. 

    printf("\nCopy host to device...");
	hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
	hipMemcpy(dev_b, b, N * sizeof(int), hipMemcpyHostToDevice);
	hipMemcpy(dev_c, c, N * sizeof(int), hipMemcpyHostToDevice);

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

	// invoke the kernel: 
	// block count: (N+127)/128
	// thread count: 128
    
    hipLaunchKernelGGL(add, blocks, threadsPerBlock, 0, 0, dev_a, dev_b, dev_c);
    hipMemcpy(a, dev_a, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(b, dev_b, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(c, dev_c, N * sizeof(int), hipMemcpyDeviceToHost);

	for (int i = 0; i < N; i+=50) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	hipFree(dev_a);
	hipFree(dev_b);
	hipFree(dev_c);
    free(a);
    free(b);
    free(c);
}
