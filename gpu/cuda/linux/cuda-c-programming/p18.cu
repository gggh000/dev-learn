// print from kernel not working.

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void helloFromGpu() {
	printf("Hello from GPU.\n");
}
int main(void) { 
	// hello from
    printf("Hello from cpu.\n");
	helloFromGpu<<<1, 10>>>();
	return 0;
}
