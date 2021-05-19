#include <stdio.h>

// 1. if N is set to up to 1024, then sum is OK.
// 2. Set N past the 1024 which is past No. of threads per blocks, and then all iterations of sum results in 
// even the ones within the block.

// 3. To circumvent the problem described in 2. above, since if N goes past No. of threads per block, we need multiple block launch.
// The trick is describe in p65 to use formula (N+127) / 128 for blocknumbers so that when block number starts from 1, it is 
// (1+127) / 128.

#define N 1024
#define MAX_THREAD_PER_BLOCK 1024
__global__ void add( int * a, int * b, int * c ) {
//	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;

//	while (tid < N)
//	tid += blockDim.x * gridDim.x;

    if (tid < N)
       	c[tid] = a[tid] + b[tid];
}

int main (void) {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	// allocate dev memory for N size for pointers declared earlier.

    printf("\nAllocating memory...");
	cudaMalloc( (void**)&dev_a, N * sizeof(int));
	cudaMalloc( (void**)&dev_b, N * sizeof(int));
	cudaMalloc( (void**)&dev_c, N * sizeof(int));

	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i+2;
	}

	// copy the initialized local memory values to device memory. 

    printf("\nCopy host to device...");
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	// invoke the kernel: 
	// block count: (N+127)/128
	// thread count: 128

    
    printf("\nLaunch cuda kernel...");
	add<<<(N+MAX_THREAD_PER_BLOCK-1)/MAX_THREAD_PER_BLOCK, MAX_THREAD_PER_BLOCK>>> (dev_a, dev_b, dev_c);
	//add<<<1, N>>> (dev_a, dev_b, dev_c);

    printf("\nCopy back from GPU to host...");
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i+=50) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}
