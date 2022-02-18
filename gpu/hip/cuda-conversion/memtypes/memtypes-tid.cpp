#include <stdio.h>
#include "hip/hip_runtime.h"

#define N 256
#define LOOPSTRIDE 4
__global__ void add(int *a, int*b, int *c, int *ofs, int *bl_id, int *th_id) {
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x ;
    //int tid = hipBlockIdx_x;
    //int tid = hipThreadIdx_x;

    if (tid < N) {
    	c[tid] = a[tid] + b[tid];
    }

    ofs[tid] = tid;
    bl_id[tid] = hipBlockIdx_x;
    th_id[tid] = hipThreadIdx_x;
}

int main (void) {
    int *a, *b, *c, *ofs, *bl_id, *th_id;
    int *dev_a, *dev_b, *dev_c, *dev_ofs, *dev_bl_id, *dev_th_id;
    int i ;
    

    a = (int*)malloc(N * sizeof(int));
    b = (int*)malloc(N * sizeof(int));
    c = (int*)malloc(N * sizeof(int));
    ofs = (int*)malloc(N * sizeof(int));
    bl_id = (int*)malloc(N * sizeof(int));
    th_id = (int*)malloc(N * sizeof(int));

 	hipMalloc(&dev_a, N * sizeof(int) );
 	hipMalloc(&dev_b, N * sizeof(int) );
 	hipMalloc(&dev_c, N * sizeof(int) );
 	hipMalloc(&dev_ofs, N * sizeof(int) );
 	hipMalloc(&dev_bl_id, N * sizeof(int) );
 	hipMalloc(&dev_th_id, N * sizeof(int) );

	for (int i = 0; i < N ; i ++ ) {
		a[i]  = i + 1;
		b[i] = i + i + 2;
		c[i] = 999;
		c[i] = 888;
	}

	for (int i = 0; i < N ; i+=LOOPSTRIDE ) {
        printf("Before add: a/b: %04d, %04d.\n", a[i], b[i]);
	}

   	hipMemcpy(dev_a, a, N * sizeof(int), hipMemcpyHostToDevice);
   	hipMemcpy(dev_b, b, N * sizeof(int), hipMemcpyHostToDevice);
   	hipMemcpy(dev_c, c, N * sizeof(int), hipMemcpyHostToDevice);
    
    // No. of workgroups.
    // threads/block, workgroup size.

    const unsigned blocks = 4;
    const unsigned threadsPerBlock = 256/4;

    hipLaunchKernelGGL(add, blocks, threadsPerBlock, 0, 0, dev_a, dev_b, dev_c, dev_ofs, dev_bl_id, dev_th_id);

    hipMemcpy(a, dev_a, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(b, dev_b, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(c, dev_c, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(ofs, dev_ofs, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(bl_id, dev_bl_id, N * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(th_id, dev_th_id, N * sizeof(int), hipMemcpyDeviceToHost);

	for (int i = 0; i < N; i+=LOOPSTRIDE )
		printf("After add: N: %02d/TID/BLID/THID: 0x%04x/0x%04x/0x%04x: %04u + %04u = %04u\n", i, ofs[i], bl_id[i], th_id[i],  a[i], b[i], c[i]);

    hipFree(dev_a);
    hipFree(dev_b);
    hipFree(dev_c);
    hipFree(dev_ofs);
    hipFree(dev_bl_id);
    hipFree(dev_th_id);
    free(a);
    free(b);
    free(c);
    free(ofs);
    free(bl_id);
    free(th_id);
    
	return 0;
}
