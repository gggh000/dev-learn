// Reducing with unrolling.

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#define DEBUG 0
inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Recursive Implementation of Interleaved Pair Approach

int cpuRecursiveReduce(int *data, int const size)
{
    if (DEBUG == 1) 
        printf("cpuRecursiveReduce: size: %u.\n", size);

    // Stop condition.

    if (size == 1) return data[0];

    // Renew the stride.

    int const stride = size / 2;

    // In-place reduction.

    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    // call recursively

    return cpuRecursiveReduce(data, stride);
}

__global__ void reduceInterleavedLess(int * g_idata, int * g_odata, unsigned int n) {
    // Set thread iD.

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert global data pointer to the local pointer of this block.
    // Offset into current block from the beginning of data stream.

    int * idata = g_idata + blockIdx.x * blockDim.x;

    // Boundary check.

    if (idx >= n ) return ;

    // In-place reduction in global memory.
    // Stride is multiple for every loop until blockDim is reached.
    // if even number of threads, add current even value plus value at stride away from current even.

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if ( index < blockDim.x ) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within block.
        __syncthreads();
    }

    // Write result for this block to global mem.

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2(int * g_idata, int * g_odata, unsigned int n) {
    // Set thread iD.

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2  + threadIdx.x;

    // Convert global data pointer to the local pointer of this block.
    // Offset into current block from the beginning of data stream.

    int * idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unroll two data blocks.

    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();

    // In-place reduction in global memory.
    // Stride is multiple for every loop until blockDim is reached.
    // if even number of threads, add current even value plus value at stride away from current even.

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within block.

        __syncthreads();
    }

    // Write result for this block to global mem.

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling4 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    // unrolling 4
    if (idx + 3 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling8 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
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
    
    iStart = seconds();
    int cpu_sum = cpuRecursiveReduce(tmp, size);
    iElaps = seconds() - iStart;

    printf("cpu reduce elapsed %d ms cpu_sum: %d\n", iElaps, cpu_sum);

    // kernel 1: reduceInterleaved.

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

    //printf("GPU warmup elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);
    printf("GPU warmup elapsed %d.\n", iElaps);

    // reduceUnrolling2

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrolling2<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) 
        gpu_sum += h_odata[i];

    printf("GPU gpu unrollng2  elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // reduceUnrolling4

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrolling4<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) 
        gpu_sum += h_odata[i];

    printf("GPU gpu unrollng4  elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // reduceUnrolling8

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrolling8<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) 
        gpu_sum += h_odata[i];

    printf("GPU gpu unrollng8  elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);


    /*
    // kernel 2: reduceInterleavedLess.

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceInterleavedLess<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) 
        gpu_sum += h_odata[i];

    printf("GPU NeighboredLess elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);
    */

    /*
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x/8; i++) 
        gpu_sum += h_odata[i];

    printf("GPU Cmptnrollelapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);
    */

    // Free host memory.

    free(h_idata);
    free(h_odata);

    // Free device memory.

    cudaFree(d_idata);
    cudaFree(d_odata);

    // Reset device.

    cudaDeviceReset();

}