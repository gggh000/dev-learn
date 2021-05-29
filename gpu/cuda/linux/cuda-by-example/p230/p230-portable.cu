// https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-/blob/master/chapter11/multidevice.cu

#include <stdio.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <pthread.h>

#define ALLOC_NORMAL 1
#define ALLOC_PAGE_HOST_PORTABLE 4
#define imin(a,b) (a<b?a:b)


const int N = 33 * 1024 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1 ) / threadsPerBlock) ;

__global__ void dot(int size, float * a, float * b, float * c) {

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

struct DataStruct {
    int     deviceID;
    int     size;
    int     offset;
    float   *a;
    float   *b;
    float   returnValue;
};

void * routine (void * pvoidData)  {
    DataStruct * data = (DataStruct *)pvoidData;

    if (data->deviceID != 0 ) {
        cudaSetDevice(data->deviceID);
        cudaSetDeviceFlags(cudaDeviceMapHost);
    }

    int size = data->size;
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    a = data->a;
    b = data->b;
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    cudaHostGetDevicePointer(&dev_a, a, 0);
    cudaHostGetDevicePointer(&dev_b, b, 0);
    cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float));

    dev_a += data->offset;
    dev_b += data->offset;

    dot <<< blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    c = 0;
    for (int i = 0; i < blocksPerGrid; i++ ) {
        c += partial_c[i];
    }

    free(dev_partial_c);
    data->returnValue = c;
    return 0;
}

float alloc_test(int size, int allocType) {
    int debug = 0;
    cudaEvent_t start, stop;
    float *a, *b;
    float *dev_a, *dev_b, *dev_partial_c;
    float elapsedTime;

    cudaHostAlloc((void**)&a, size * sizeof(float), cudaHostAllocWriteCombined|cudaHostAllocMapped|cudaHostAllocPortable);
    cudaHostAlloc((void**)&b, size * sizeof(float), cudaHostAllocWriteCombined|cudaHostAllocMapped|cudaHostAllocPortable);

    for (int i = 0; i < size ; i++ ) {
        a[i] = i;
        b[i] = i * 2;
    }

    // prepare for multithread.

    DataStruct data[2];
    data[0].deviceID = 0;
    data[0].offset = 0;
    data[0].size = size / 2;
    data[0].a = a;
    data[0].b = b;

    data[1].deviceID = 1;
    data[1].offset = size / 2;
    data[1].size = size / 2;
    data[1].a = a;
    data[1].b = b;

    CUTThread thread = start_thread(routine, &(data[1]));
    routine(&data[0]);
    end_thread(thread);

    cudaFreeHost(a);
    cudaFreeHost(b);

    printf("Value calculated: %04.02f.\n", data[0].returnValue + data[1].returnValue);

    return 0;
}

int main()
{
    float elapsedTime;
    cudaDeviceProp prop;
    int whichDevice;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 2) {
        printf("we need at least 2 compute 1.0 or greater devices, but only found %d.\n", deviceCount);
        return 0;
    } 

    for (int i = 0 ; i < 2 ; i++ ) {
        cudaGetDeviceProperties(&prop, whichDevice);

        if (prop.canMapHostMemory != 1 ) {
            printf("Device can not map host memory.\n");
            return 0;
        } else {
            printf("Device can map host memory ok...\n");
        }
    }

    printf("set flag to map host memory...\n");
    cudaSetDevice( 0 );
    cudaSetDeviceFlags( cudaDeviceMapHost);

    //float MB  = (float)100 * SIZE * sizeof(int) / 1024 / 1024;
    elapsedTime = alloc_test(N, ALLOC_NORMAL);
    printf("Time using cudaMalloc: %3.1f ms.\n", elapsedTime);
    //printf("\tMB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));

    printf("Calling alloc_test with ALLOC_PAGE_HOST_PORTABLE...\n");
    elapsedTime = alloc_test(N, ALLOC_PAGE_HOST_PORTABLE);
    printf("Time using cudaHostAlloc: %3.1f ms.\n", elapsedTime);
    //printf("\tMB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));
    return 0;
}
