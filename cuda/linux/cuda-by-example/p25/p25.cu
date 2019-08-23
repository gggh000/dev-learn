/**
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

__global__ void add( int a, int b, int * c) {
	*c  = a + b;
}

int main ( void ) {
	int c;
	int * dev_c;
	cudaMalloc( ( void ** ) & dev_c, sizeof(int) );
	add <<<1,1>>> (2,7, dev_c);
	cudaMemcpy( &c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf( " 2 + 7 = %d\n", c);
}
