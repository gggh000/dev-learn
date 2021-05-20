//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

// A minimalist OpenCL program.

#include <CL/cl.h>
#include <stdio.h>

#define printDeviceInfo(X)   printf("\n%s: %s",  (X));
#define declareDeviceInfo(X) char str(X)[] = "(X)";

#define NWITEMS 2048
#define DEBUG 0
#define SIZE (10*1024*1024)
#define ALLOC_NORMAL 1
#define ALLOC_PAGE_LOCKED 2

// A simple kernelfcn kernel
const char *source =

"kernel void kernelfcn(     global uint *dev_c, global uint *dev_a, global uint *dev_b)  \n"
"{                                                                      \n"
" uint tid = get_global_id(0);                                          \n"
" dev_c[tid] = dev_a[tid] + dev_b[tid];                                 \n"
"}                                                                      \n";

float opencl_mem_alloc_test(int size, int up, int allocType) {

    cl_event e1;
    cl_ulong start, end, duration;
    int *a;
    cl_mem * dev_a;
    int ret;

    // 1. Get a platform.

    cl_uint CONFIG_MAX_PLATFORMS=20;
    cl_platform_id platforms[CONFIG_MAX_PLATFORMS];
    cl_uint platforms_available;

    clGetPlatformIDs(CONFIG_MAX_PLATFORMS, platforms, &platforms_available );
    printf("\nNo. of platforms available: %d.\n", platforms_available);

    for (int i = 0 ; i < platforms_available; i ++ ) {
        printf("Platform %d: %d.\n", i, platforms[i]);
    }

    // 2. Find a gpu/cpu device.

    cl_uint CONFIG_MAX_DEVICES=20;
    cl_uint devices_available;
    cl_device_id device[CONFIG_MAX_DEVICES];

    ret = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_ALL, CONFIG_MAX_DEVICES, device, &devices_available);

    printf("No. of devices available: %d.\n", devices_available);

    // 3. Create a context and command queue on that device.

    cl_context context = clCreateContext( NULL, 1,  &device[0], NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue( context, device[0], 0, NULL );

    if (allocType == ALLOC_PAGE_LOCKED) {
        //!!!!cudaHostAlloc((void**)&a, size * sizeof(*a), cudaHostAllocDefault);
        printf("\nNot implemented yet.");
    } else if  (allocType == ALLOC_NORMAL ) {
        a = (int*) malloc(size*sizeof(a));
    }

    dev_a = clCreateBuffer( context, CL_MEM_READ_WRITE, size * sizeof(cl_uint), NULL, NULL);

    if (DEBUG==1)
        printf("Copying data to GPU...");

    for (int i = 0; i < 100 ; i ++ ) {
        if (up)
            ret = clEnqueueWriteBuffer(queue, dev_a, CL_TRUE, 0, size * sizeof(cl_uint), a, NULL, NULL, &e1);
        else
        clEnqueueReadBuffer(queue, dev_a, CL_TRUE, 0, NWITEMS * sizeof(cl_uint), a, NULL, NULL, &e1);
    }    

    clGetEventProfilingInfo(e1, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(e1, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    duration = end - start;
    return (duration / 1024 * 1024);
}

int main(int argc, char ** argv)
{
    int stat;
    char str1[100];
    ushort ushort1;
    uint uint1;
    ulong ulong1;
    size_t strLen;
    cl_int ret;
    uint a[NWITEMS], b[NWITEMS], c[NWITEMS];
    int i;
    float elapsedTime;

    float MB  = (float)100 * SIZE * sizeof(int) / 1024 / 1024;

    elapsedTime = opencl_mem_alloc_test(SIZE, 1, ALLOC_NORMAL);
    printf("Time using normal allocation: %3.1f ms.\n", elapsedTime);
    printf("\tMB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));

    elapsedTime = opencl_mem_alloc_test(SIZE, 0, ALLOC_NORMAL);
    printf("Time using normal allocation: %3.1f ms.\n", elapsedTime);
    printf("\tMB/s during copy down: %3.1f.\n", MB / (elapsedTime / 1000));

    elapsedTime = opencl_mem_alloc_test(SIZE, 1, ALLOC_PAGE_LOCKED);
    printf("Time using host allocation: %3.1f ms.\n", elapsedTime);
    printf("\tMB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));

    elapsedTime = opencl_mem_alloc_test(SIZE, 0, ALLOC_PAGE_LOCKED);
    printf("Time using host allocation: %3.1f ms.\n", elapsedTime);
    printf("\tMB/s during copy down: %3.1f.\n", MB / (elapsedTime / 1000));

    /*
    
    // 4. Perform runtime source compilation, and obtain kernel entry point.

   cl_program program = clCreateProgramWithSource( context, 1, &source,  NULL, NULL );

    if (ret) {
        printf("Error: clCreateProgramWithSource returned non-zero: %d.\n", ret);
        return 1;
    } else  {
        printf("clCreateProgramWithSource return OK.... %d.\n", ret);
    }


    clBuildProgram( program, 1, device, NULL, NULL, NULL );

    if (ret) {
        printf("Error: clBuildProgram returned non-zero: %d.\n", ret);
        return 1;
    } else  {
        printf("clBuildProgram return OK.... %d.\n", ret);
    }


    cl_kernel kernel = clCreateKernel( program, "kernelfcn", &ret);

    if (ret) {
        printf("Error: clCreateKernel returned non-zero: %d.\n", ret);
//        return 1;
    } else  {
        printf("clCreateKernel return OK.... %d.\n", ret);
    }

    if (ret) {
        printf("Error: clCreateKernel returned non-zero: %d.\n", ret);
        return 1;
    } else  {
        printf("clCreateKernel return OK.... %d.\n", ret);
    }

    // 5. Create a data buffer.

    for (int i = 0; i < NWITEMS; i ++ ) {
        a[i]  = i;
        b[i] = i * i;
    }

    for(i=0; i < NWITEMS; i+=100)
    {
        printf("globalID: 0x%02u. value: 0x%08u.\n", i, a[i]);
        
    }

    printf("Creating mem on GPU.....");

    cl_mem dev_a = clCreateBuffer( context, CL_MEM_READ_WRITE, NWITEMS * sizeof(cl_uint), NULL, NULL );
    cl_mem dev_b = clCreateBuffer( context, CL_MEM_READ_ONLY, NWITEMS * sizeof(cl_uint), NULL, NULL );
    cl_mem dev_c = clCreateBuffer( context, CL_MEM_WRITE_ONLY, NWITEMS * sizeof(cl_uint), NULL, NULL );

    if (DEBUG==1)
        printf("Copying data to GPU...");

    ret = clEnqueueWriteBuffer(queue, dev_a, CL_TRUE, 0, NWITEMS * sizeof(cl_uint), a, NULL, NULL, NULL);
    printf("ret: %d\n", ret); 
    ret = clEnqueueWriteBuffer(queue, dev_b, CL_TRUE, 0, NWITEMS * sizeof(cl_uint), b, NULL, NULL, NULL);
    printf("ret: %d\n", ret); 

    //Erasing a[] for test.

    for (int i = 0; i < NWITEMS; i ++ ) {
        a[i] = 0;
        b[i] = 0;
    }

    for(i=0; i < NWITEMS; i+=100)
    {
        printf("globalID: 0x%02u. value: 0x%08u.\n", i, a[i]);
        
    }
    

    // 6. Launch the kernel. Let OpenCL pick the local work size.

    if (DEBUG==1) {
        printf("Launch kernel\n");
        //getchar();
    }

    // start measuring event here.

    cl_event start, stop;
    cl_int clGetEventProfilingInfo (

    size_t global_work_size = NWITEMS;
    size_t local_work_size = ulong1;
    clGetDeviceInfo(device[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(ulong), &ulong1, &strLen);
    printf("CL_DEVICE_MAX_WORK_GROUP_SIZE: %u.\n", ulong1);

    printf("set kernel args...\n");
    clSetKernelArg(kernel, 0, sizeof(dev_c), (void*) &dev_c);
    printf("ret: %d\n", ret); 
    clSetKernelArg(kernel, 1, sizeof(dev_a), (void*) &dev_a);
    printf("ret: %d\n", ret); 
    clSetKernelArg(kernel, 2, sizeof(dev_b), (void*) &dev_b);
    printf("ret: %d\n", ret); 
    clEnqueueNDRangeKernel( queue, kernel,  1, NULL, &global_work_size, &local_work_size, 0,  NULL, NULL);
    printf("clEnqueueNDRangeKernel OK...\n");
    //getchar();

    clFinish( queue );

    if (DEBUG==1) {
        printf("clFinish OK...\n");
        //getchar();
    }

    // 7. Look at the results via synchronous buffer map.

    printf("Reading back from GPU the sum...\n");

    if (DEBUG==1)  {
        ret = clEnqueueReadBuffer(queue, dev_a, CL_TRUE, 0, NWITEMS * sizeof(cl_uint), a, NULL, NULL, NULL);
        printf("ret: %d\n", ret); 
    }

    ret = clEnqueueReadBuffer(queue, dev_c, CL_TRUE, 0, NWITEMS * sizeof(cl_uint), c, NULL, NULL, NULL);
    printf("ret: %d\n", ret); 

    printf("Printing sums now...\n");

    for(i=0; i < NWITEMS; i+=100)
    {
        if (DEBUG==1) 
            printf("globalID: 0x%02u. value (a/c): 0x%08u/0x%08u.\n", i, a[i], c[i]);
        printf("globalID: 0x%02u. value (a/c): 0x%08u.\n", i, c[i]);
    }

    */ 
    printf("\n");
    return 0;
}
