#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#define ENABLE_34 1

void initialData(float *ip, int size ) {
    // generate different seed for random number.

    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i ++ ) {
        ip[i] = (float) (rand() & 0xff ) / 10.0f;
    }
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void warmup(float *A, float * B, float * C, const int n, int offset) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;
    if ( k < n ) C[i] = A[k] + B[k];
}

__global__ void readOffset(float * A, float * B, float *C, const int n, int offset) { 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;
    if ( k < n ) C[i] = A[k] + B[k];
}

void sumArrayOnHost(float * A, float * B, float *C, const int n, int offset ) {
    for (int idx = offset, k = 0; idx < n; idx++, k++ ) {
        C[k] = B[idx] + A[idx];
    }
}

int main(int argc, char **argv) {

    // Setup device.

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s using Device %d: %s.\n", argv[0], dev, deviceProp.name);
    cudaSetDevice(dev);

    // Setup data size.

    int nElem = 1<< 20; // total No. of elements to reduce.
    printf(" with array size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // Set up offset for memory.

    int blocksize = 512;
    int size = 64;
    int offset = 0;
    double iStart, iElaps;

    if (argc > 1) offset    = atoi(argv[1]);
    if (argc > 2) blocksize = atoi(argv[2]);

    printf("blocksize / size %d, %d.\n", blocksize, size);

    // setup execution congiguration.

    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x-1 ) / block.x, 1);
    
    //printf("Execution configure (block %d grid %d).\n", block.x, grid.x);
    //sleep(3);
    
    // allocate host memory.

    float *h_A = (float*)malloc(nBytes);
    float *h_B = (float*)malloc(nBytes);
    float *hostRef = (float*)malloc(nBytes);
    float *gpuRef = (float*)malloc(nBytes);

    // Initialize host arrya.
    
    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    // Summary at host side.

    sumArrayOnHost(h_A, h_B, hostRef, nElem, offset);

    // Allocate device memory.
    
    float * d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);
    
    // Copy data from host to device.

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice);

    // run warmup.

    iStart = seconds();
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("warmup <<< %4d %4d >>> offset %4d elapsed %f sec .\n", grid.x, block.x, offset, iElaps);

    // run kernel 1.

    iStart = seconds();
    readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("readOffset <<< %4d %4d >>> offset %4d elapsed %f sec .\n", grid.x, block.x, offset, iElaps);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // checkResult(hostRef, gpuRef, nElem-offset);

    // Free host and device memory.

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    cudaDeviceReset();
    return 0;
}
