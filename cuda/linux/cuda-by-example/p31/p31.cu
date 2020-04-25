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
//#include <helper_functions.h>
//#include <helper_cuda.h>

int main ( void ) {
	cudaDeviceProp prop;
	int count, i;
	cudaGetDeviceCount ( &count);

	for (i = 0 ; i < count ; i ++ ) {	
		cudaGetDeviceProperties ( &prop, i);
		printf("\n\n====DEVICE %d=====", i);
		printf("\nDevice name: 		%s", prop.name);
		printf("\nTotal global mem: 	0x%0x", prop.totalGlobalMem);
		printf("\nwarpSize: 		0x%0x", prop.warpSize);
		printf("\nmaxThreads/Block: 	0x%0x", prop.maxThreadsPerBlock);
		printf("\nmaxThreads/DIM: 	0x%0x", prop.maxThreadsDim);
		printf("\nmaxGridSize: 		0x%0x", prop.maxGridSize);
		printf("\ntotal const mem: 	0x%0x", prop.totalConstMem);
		printf("\ncompute cap: 		%d.%d", prop.major, prop.minor);
		printf("\nmultiprocessors: 	0x%0x", prop.multiProcessorCount);
		printf("\ncanMapHostMem: 	0x%0x", prop.canMapHostMemory);
		printf("\ncomputeMode: 		0x%0x", prop.computeMode);
		printf("\nconcurrentKernels: 	0x%0x", prop.concurrentKernels);
	}

	printf("\n");
}
