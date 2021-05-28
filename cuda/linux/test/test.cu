#include <stdio.h>

#define N 2048

__global__ void add(int *a, int*b, int *c) {
	int tid = blockIdx.x;
//	if (tid < N) 
	c[tid] = a[tid] + b[tid];
	c[tid] = blockIdx.x;
}

int main (void) {

    dim3 b1d( 5       );
    dim3 b2d( 5, 5       );
    dim3 b3d( 5, 5, 5       );
    printf("%d, %d, %d\n", b1d.x, b1d.y, b1d.z);
    printf("%d, %d, %d\n", b2d.x, b2d.y, b2d.z);
    printf("%d, %d, %d\n", b3d.x, b3d.y, b3d.z);
    /*
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	cudaMalloc( (void**)&dev_a, N * sizeof(int) );
	cudaMalloc( (void**)&dev_b, N * sizeof(int) );
	cudaMalloc( (void**)&dev_c, N * sizeof(int) );

	for (int i = 0; i < N ; i ++ ) {
		a[i]  = -i;
		b[i] = i * i;
	}
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	add<<<N,1>>> (dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i+=10 ) {
		printf("%d: %d + %d = %d\n", i, a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
    */
	return 0;
}
