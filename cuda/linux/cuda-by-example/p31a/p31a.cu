/**
This example demonstrates what happens if the vector sum addition exceeds the threads per block.
The N is assigned the threads / block size right from prop. As long as the vector size is withing this
size, the sum will succeed. However, if it exceeds, the sum will fail and c[] array will return with 
garbage. It will not return the partial data, the part of the vector that fits within the threads per block.
Rather, whole vector sum return data will be invalid. To prove this, increase the N such that N > threads/block.

 */

/**
 */

// System includes

#include <stdio.h>
#include <assert.h>

// CUDA runtime

#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA

#include <helper_functions.h>
#include <helper_cuda.h>

#define N 65536
#define DEBUG 0

__global__ void add( long int * a, long int * b, long int * c) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if ( tid < N ) {
		c[tid] = a[tid] + b[tid];
		//c[tid] = (long int)&c[tid];

		if (tid < N && DEBUG == 1) {
			printf("\ntid: %d, blockDim.x: 0x%0x", tid,  blockDim.x);
			printf("\ntid: %d, blockDim.y: 0x%0x", tid,  blockDim.y);
			printf("\ntid: %d, blockDim.z: 0x%0x", tid,  blockDim.z);
			printf("\ntid: %d, gridDim.x:  0x%0x", tid,  gridDim.x);
			printf("\ntid: %d, gridDim.y:  0x%0x", tid,  gridDim.y);
	}
		tid += blockDim.x * gridDim.x;
	}

}
int main ( void ) {
	long int *dev_a, *dev_b, *dev_c;
	int errors;
	int blockDim;

        cudaDeviceProp prop;
        int count, i;
        cudaGetDeviceCount ( &count);

        for (i = 0 ; i < count ; i ++ )
                cudaGetDeviceProperties ( &prop, i);

	blockDim = prop.maxThreadsPerBlock;
	printf("Max threads per block for device 0: %d", blockDim);

	int a[N], b[N], c[N];

	cudaMalloc( (void**) &dev_a, N * sizeof(int));
	cudaMalloc( (void**) &dev_b, N * sizeof(int));
	cudaMalloc( (void**) &dev_c, N * sizeof(int));

	errors = 0;

	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = 1;
	}

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	add <<<(N + blockDim - 1) / blockDim, blockDim>>>(dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		if (c[i] != 6) {
			//printf("\n0x%x did not add correctly: %d", i, c[i]);
			errors ++;
			//continue;
		}
		printf("\n%d. GPU address: 0x%08x, host addr: 0x%0x", i, c[i], &c[i]);
	}

	printf("\nsize of int, long int: %d, %d", sizeof(int), sizeof(long int));
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	printf("\nNo. of errors in vector sum: %d.", errors);
	printf("\n");
}
