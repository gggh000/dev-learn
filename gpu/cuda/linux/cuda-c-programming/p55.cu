// Summing matrices with 2d grid and 2d blocks.

#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

void checkResult(float * hostRef, float * gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;
    
    for (int i = 0; i < N ; i ++ ) {
        if (abs(hostRef[i] - gpuRef[i] > epsilon)) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f current %d.\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) printf("Arrays match OK.\n");
}
void sumMatrixOnHost ( float * A, float * B, float *C, const int nx, const int ny) {
    float *ia = A;
    float *ib = B;
    float *ic = C;
    
    for (int iy = 0; iy < ny; iy++ ) {
        for (int ix = 0; ix < nx; ix++ ) {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    } 

}
__global__ void sumArraysOnGPU2D(float *MatA, float *MatB, float *MatC, const int nx, const int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = ix + iy * nx;

    if (ix < nx && iy < ny) 
        MatC[idx] = MatA[idx] + MatB[idx];
}

void initialData(float *ip, int size ) {
    // generate different seed for random number.

    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i ++ ) {
        ip[i] = (float) (rand() & 0xff ) / 10.0f;
    }
}

double cpuSecond() {
    struct timeval tp; 
    gettimeofday(&tp, NULL);
    return (( double )tp.tv_sec + (double) tp.tv_usec * 1.e-6);

}
int main(int argc, char ** argv) {
    printf("%s Starting ...\n", argv[0]);

    // setup device.

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using device %d. %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // setup date size of vectors.

    int nx = 1 << 14;
    int ny = 1 << 14;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    double iStart, iElaps;

    printf("Matrix size %d, %d.\n", nx, ny); 
    
    // malloc host memory

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*) malloc(nBytes);
    h_B = (float*) malloc(nBytes);
    hostRef = (float*) malloc(nBytes);
    gpuRef = (float*) malloc(nBytes);
    
    // initialize data at host side.

    iStart = cpuSecond();
    initialData (h_A, nxy);
    initialData (h_B, nxy);
    iElaps = cpuSecond() - iStart;
    printf("initialData: Time Elapsed %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector at host side for result checks.

    iStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = cpuSecond() - iStart;

    printf("sumArrayOncPU2D: Time Elapsed %f sec\n", iElaps);
    
    // malloc device global memory.
    
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((float**)&d_MatA, nBytes);
    cudaMalloc((float**)&d_MatB, nBytes);
    cudaMalloc((float**)&d_MatC, nBytes);

    // transfer data from host to device

    iStart = cpuSecond();
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);
    iElaps = cpuSecond() - iStart;
    printf("transfer data to gpu: Time Elapsed %f sec\n", iElaps);

    // invoke kernel at host side.

    int dimx = 32;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid ((nx + block.x-1) / block.x, (ny + block.y-1) / block.y);
    
    iStart = cpuSecond();
 
    sumArraysOnGPU2D <<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumArrayOnGPU2D: <<<(%d,%d), (%d, %d)>>> Time Elapsed %f sec\n", \
        grid.x, \
        grid.y, \
        block.x, block.y, iElaps);

    // copy kernel result back to host side.

    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    // check result.

    checkResult(hostRef, gpuRef, nxy);

    // free device global memory.
    

    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
