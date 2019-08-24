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

#define N 1024*1024*1024
#define DEBUG 0

__global__ void memtrf(char * a) {
	//int tid = threadIdx.x + blockIdx.x * blockDim.x;
}
int main ( void ) {
	char *dev_a;
	int blockDim;
	cudaEvent_t start, stop;
	//int errors;
	char * a;

        cudaDeviceProp prop;
        int count, i;
        cudaGetDeviceCount ( &count);
	float elapsedTime;

        for (i = 0 ; i < count ; i ++ )
                cudaGetDeviceProperties ( &prop, i);

	blockDim = prop.maxThreadsPerBlock;
	printf("Max threads per block for device 0: %d", blockDim);

	//int a[N];
	a = (char*) malloc(sizeof(char) * N);
	printf("\nAllocated memory on the host ok: 0x%08x", N);
	cudaMalloc( (void**) &dev_a, N * sizeof(char));
	printf("\nAllocated memory on the GPU ok: 0x%08x", N);


	cudaEventCreate( &start);
	cudaEventCreate( &stop);
	cudaEventRecord( start, 0);
	cudaMemcpy(dev_a, a, N * sizeof(char), cudaMemcpyHostToDevice);
	cudaEventRecord( stop, 0);
	cudaEventElapsedTime( &elapsedTime, start, stop);
	printf("\nTime taken: %3.1f ms\n", elapsedTime);
	
	//memtrf <<<(N + blockDim - 1) / blockDim, blockDim>>>(dev_a);
	//cudaMemcpy(c, dev_c, N * sizeof(char), cudaMemcpyDeviceToHost);

	/*
	for (int i = 0; i < N; i++) {
		if (c[i] != 6) {
			//printf("\n0x%x did not add correctly: %d", i, c[i]);
			errors ++;
			//continue;
		}
		//printf("\n%d. GPU address: a/b/c: 0x%08x, 0x%08x, 0x%08x, host addr: 0x%0x", i, a[i], b[i], c[i], &c[i]);
	}
	*/

	printf("\nsize of int, long int: %d, %d", sizeof(char), sizeof(long int));

	printf("\nPress a key to release the cuda memory...");
	getchar();
	cudaFree(dev_a);

	//printf("\nNo. of errors in vector sum: %d.", errors);
	printf("\n");
}
