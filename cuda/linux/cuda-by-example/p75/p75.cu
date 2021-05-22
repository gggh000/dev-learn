// shared memory concept. 

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;

__global__ void dot(float * a, float * b, float * c) {

    // this variable will be repeated and will have same value on every blocks.

    __shared__ float cache[threadsPerBlock];

    int tid = threadIdx.x * blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N)  {

        // temp will accumulate the product of a and b at every at every grid.

        temp += a[tid] * b[tid] ;
        tid += blockDim.x * gridDim.x;
    }

    // holds product of a and b at every grid. 

    cache[cacheIndex] = temp; 

    // sync threads in this block.

    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2 because of following code.
    // couple of examples (still can not understand the divide in loop.
    // blockDim.x = 256.
    //                              i         cache[tid] + cache[tid+i]
    // threadIdx/cacheindex 128:    i0 = 128, cache[128] + cache[128+128] -> t128, t256.
    // threadIdx/cacheindex 127:    i0 = 128, cache[127] + cache[128+127] -> t127, t255.
    // threadIdx/cacheindex 126:    i0 = 128, cache[126] + cache[128+126] -> t126, t254.
    // threadIdx/cacheindex 125:    i0 = 128, cache[125] + cache[128+125] -> t125, t253.
    // ...
    // threadIdx/cacheindex 64:     i0 = 128, cache[64] + cache[128+64] -> t64, t192.
    // ...
    // threadIdx/cacheindex 32:     i0 = 128, cache[32] + cache[128+32] -> t32, t160.
    // ...
    // threadIdx/cacheindex 2:      i0 = 128, cache[2] + cache[128+2] -> t2, t130.
    // threadIdx/cacheindex 1:      i0 = 128, cache[1] + cache[128+1] -> t1, t129.

    int i = blockDim.x/2; 

    while (i != 0 ) {
        if (cacheIndex < i) 
            cache[cacheIndex] += cache[cacheIndex + i];

        __syncthreads();
        i /= 2 ;
    }

    if (cacheIndex == 0) {
        c[blockIdx.x] = cache[0];
    }
}

int main( void ) {
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1 ) / threadsPerBlock) ;

    // allocate memory on the cpu side.

    a = (float*)malloc( N * sizeof(float));
    b = (float*)malloc( N * sizeof(float));
    partial_c = (float*)malloc( blocksPerGrid * sizeof(float));

    // allocate memory on the gpu side.

    cudaMalloc((void**) &dev_a, N * sizeof(float));
    cudaMalloc((void**) &dev_b, N * sizeof(float));
    cudaMalloc((void**) &dev_partial_c, blocksPerGrid * sizeof(float));

    // file in the host memory with data.
    
    for (int i = 0; i < N; i++ ) {
        a[i] = i;
        b[i] = i * 2;
    }

    // copy the arrays a and b to the gpu.

    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    dot<<< blocksPerGrid, threadsPerBlock>>> ( dev_a, dev_b, dev_partial_c);

    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    c = 0;    
    for (int i = 0; i < blocksPerGrid; i++)
        c += partial_c[i];

    #define sum_squares(x) (x*(x+1)*(2*x+1)/6)

    printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float) (N-1)));

    // free memory on gpu side.

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    // free on cpu side.

    free(a);
    free(b);
    free(partial_c);
}
