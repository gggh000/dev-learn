#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda_runtime.h>

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void reduceNeighbored(int * g_idata, int * g_odata, unsigned int n) {
    // Set thread iD.

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert global data pointer to the local pointer of this block.
    // Offset into current block from the beginning of data stream.

    int * idata = g_idata + blockIdx.x + blockDim.x;

    // Boundary check.

    if (idx >= n ) return ;

    // In-place reduction in global memory.
    // Stride is multiple for every loop until blockDim is reached.
    // if even number of threads, add current even value plus value at stride away from current even.

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0 ) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within block.

        __syncthreads();
    }

    // Write result for this block to global mem.

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void warmup(int * g_idata, int * g_odata, unsigned int n) {
    // Set thread iD.

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert global data pointer to the local pointer of this block.
    // Offset into current block from the beginning of data stream.

    int * idata = g_idata + blockIdx.x + blockDim.x;

    // Boundary check.

    if (idx >= n ) return ;

    // In-place reduction in global memory.
    // Stride is multiple for every loop until blockDim is reached.
    // if even number of threads, add current even value plus value at stride away from current even.

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0 ) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within block.

        __syncthreads();
    }

    // Write result for this block to global mem.

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


int main(int argc, char ** argv) {
    // Setup device.

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev); 
    printf("%s starting reduction at ", argv[0]);
    printf("Device %d: %s.", dev, deviceProp.name);
    cudaSetDevice(dev);

    bool bResult = false;
    
    // Initialization.

    int size = 1 << 24; // total No. of elements.
    printf(" with array size %d.", size);

    // Execution configuration.

    int blocksize = 512; // init. block size.
    if (argc > 1 ) {
        blocksize = atoi(argv[1]);
    }

    dim3 block (blocksize, 1);
    dim3 grid ((size+block.x-1)/block.x, 1);
    
    printf("grid %d block %d\n", grid.x, block.x);

    // Allocate host memory.

    size_t bytes = size * sizeof(int);
    int *h_idata = (int*) malloc(bytes);

    // h_odata will hold the sum of reduction in every block therefore this is 
    // a grid size or total No. of block in grid. 

    int *h_odata = (int*) malloc(grid.x * sizeof(int));
    int *tmp = (int*) malloc(bytes);

    // Initialize the array.

    for (int i = 0; i < size; i ++ ) {
        // mask off high 2 bytes to  force max number to 255.

        h_idata[i] = (int)(rand() & 0xff);
    }    
    memcpy(tmp, h_idata, bytes);

    size_t iStart, iElaps;
    int gpu_sum = 0;
    
    // Allocate device memory.

    int * d_idata = NULL;
    int * d_odata = NULL;
    cudaMalloc((void **) &d_idata, bytes);
    cudaMalloc((void **) &d_odata, grid.x * sizeof(int));

    // CPU reduction.  I am going to skip this one.
    /*
    iStart = seconds();
    int cpu_sum = re
    */

    // kernel 1: reduceNeighbored.

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    warmup<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) 
        gpu_sum += h_odata[i];

    printf("GPU warmup elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 1: reduceNeighbored.

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) 
        gpu_sum += h_odata[i];

    printf("GPU Neighbored elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x/8; i++) 
        gpu_sum += h_odata[i];

    printf("GPU Cmptnrollelapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

    // Free host memory.

    free(h_idata);
    free(h_odata);

    // Free device memory.

    cudaFree(d_idata);
    cudaFree(d_odata);

    // Reset device.

    cudaDeviceReset();

}
