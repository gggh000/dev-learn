#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void warmingUp(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void mathKernel1(float * c) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if (tid % 2 == 0) {
        a = 100.0f;        
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel2(float * c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f;        
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

int main(int argc, char **argv) {
    // setup device.

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s using Device %d: %s.\n", argv[0], dev, deviceProp.name);
    
    // setup data size.

    int size = 64;
    int blocksize = 64;
    if (argc > 1) blocksize = atoi(argv[1]);
    if (argc > 2) size = atoi(argv[2]);

    printf("Data size %d.\n", size);

    // setup execution congiguration.

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x-1 ) / block.x, 1);
    
    printf("Execution configure (block %d grid %d).\n", block.x, grid.x);
    
    // allocate gpu memory.

    float *d_C;
    size_t nBytes = size * sizeof(float);
    cudaMalloc((float**)&d_C, nBytes);
    
    // run a warmup kernel to remove overhead.

    size_t iStart, iElaps;
    cudaDeviceSynchronize();
    iStart = seconds();
    warmingUp<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("warmup <<< %d %d >>> elapsed %06d sec.\n", grid.x, block.x, iElaps);

    // run kernel 1.

    iStart = seconds();
    mathKernel1<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel1 <<< %4d %4d >>> elapsed %06d sec.\n", grid.x, block.x, iElaps);

    // run kernel 2.

    iStart = seconds();
    mathKernel2<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel2 <<< %4d %4d >>> elapsed %06d sec.\n", grid.x, block.x, iElaps);

    // run kernel 3.
    /*

    iStart = seconds();
    mathKernel3<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel3 <<< %4d %4d >>> elapsed %06d sec.\n", grid.x, block.x, iElaps);

    // run kernel 4.

    iStart = seconds();
    mathKernel4<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel4 <<< %4d %4d >>> elapsed %06d sec.\n", grid.x, block.x, iElaps);
    */

    cudaFree(d_C);
    cudaDeviceReset();
    return 0;
}
