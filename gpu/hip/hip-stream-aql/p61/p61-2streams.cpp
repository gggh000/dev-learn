#include <stdio.h>
#include "hip/hip_runtime.h"

// 1. if N is set to up to 1024, then sum is OK.
// 2. Set N past the 1024 which is past No. of threads per blocks, and then all iterations of sum results in 
// even the ones within the block.

// 3. To circumvent the problem described in 2. above, since if N goes past No. of threads per block, we need multiple block launch.
// The trick is describe in p65 to use formula (N+127) / 128 for blocknumbers so that when block number starts from 1, it is 
// (1+127) / 128.

#define N 2048
#define N 536870912 
#define MAX_THREAD_PER_BLOCK 1024

__global__ void add( int * a, int * b, int * c ) {
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x ;
    for (int i  = 0; i < 10000 ; i ++) {
        if (tid < N) 
            c[tid] = a[tid] + b[tid];
    }
}    
__global__ void add2( int * a, int * b, int * c ) {
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x ;
    for (int i  = 0; i < 10000 ; i ++) {
        if (tid < N) 
            c[tid] = a[tid] + b[tid];
    }
}    

int main (void) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int stepSize;

    int count = 0;
    hipStream_t stream_1;
    hipStream_t stream_2;
    hipGetDeviceCount(&count);

    printf("\nDevice count: %d.", count);

	// allocate dev memory for N size for pointers declared earlier.

    printf("\nAllocating memory...(size %u array size of INT).\n", N );

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

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

	// invoke the kernel: 
	// block count: (N+127)/128
	// thread count: 128
    
    hipStreamCreate(&stream_1);
    hipStreamCreate(&stream_2);
    hipLaunchKernelGGL(add, blocks, threadsPerBlock, 0, stream_1, dev_a, dev_b, dev_c);
    hipLaunchKernelGGL(add2, blocks, threadsPerBlock, 0, stream_2, dev_a, dev_b, dev_c);
    hipDeviceSynchronize();
    
	hipFree(dev_a);
	hipFree(dev_b);
	hipFree(dev_c);
}
