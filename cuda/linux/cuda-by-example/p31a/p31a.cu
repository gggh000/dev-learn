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

//#define N 1025

__global__ void add( long int * a, long int * b, long int * c, int N) {
	int tid = threadIdx.x;
	if ( tid < N )
		c[tid] = a[tid] + b[tid];
		c[tid] = (long int)&c[tid];
}
int main ( void ) {
	long int *dev_a, *dev_b, *dev_c;
	int errors, N;

        cudaDeviceProp prop;
        int count, i;
        cudaGetDeviceCount ( &count);

        for (i = 0 ; i < count ; i ++ )
                cudaGetDeviceProperties ( &prop, i);

	N = prop.maxThreadsPerBlock;
	N = 10;
	printf("Max threads per block for device 0: %d", N);
	

	int a[N], b[N], c[N];

	cudaMalloc( (void**) &dev_a, N * sizeof(int));
	cudaMalloc( (void**) &dev_b, N * sizeof(int));
	cudaMalloc( (void**) &dev_c, N * sizeof(int));

	errors = 0;

	for (int i = 0; i < N; i++) {
		a[i] = 2;
		b[i] = 4;
	}

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	add <<<1, N>>>(dev_a, dev_b, dev_c, N);

	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		if (c[i] != 6) {
			//printf("\n0x%x did not add correctly: %d", i, c[i]);
			errors ++;
			//continue;
		}
		printf("\n%d. GPU address: 0x%0x, host addr: 0x%0x", i, c[i], &c[i]);
	}

	printf("\nsize of int, long int: %d, %d", sizeof(int), sizeof(long int));
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	printf("\nNo. of errors in vector sum: %d.", errors);
	printf("\n");
}
