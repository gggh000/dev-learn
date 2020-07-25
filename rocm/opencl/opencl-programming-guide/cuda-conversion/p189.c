//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

// A minimalist OpenCL program.

#include <CL/cl.h>
#include <stdio.h>

#define printDeviceInfo(X)   printf("\n%s: %s",  (X));
#define declareDeviceInfo(X) char str(X)[] = "(X)";

#define SIZE 2048
#define LOCAL_WORK_SIZE 256
#define DEBUG 0
//#define SIZE (10*1024*1024)
#define SIZE (512*1024)

// A simple kernelfcn kernel
const char *source =

"kernel void kernelfcn(     global uint *dev_c, global uint *dev_a, global uint *dev_b)  \n"
"{                                                                      \n"
" uint tid = get_global_id(0);                                          \n"
" dev_c[tid] = dev_a[tid] + dev_b[tid];                                 \n"
"}                                                                      \n";


float opencl_malloc_test(int size, int up, int hostAlloc, cl_context * context, cl_command_queue * queue) {
    int *a;
    cl_mem dev_a;
    float elapsedTime = 0;
    int ret;
    cl_event evtReadWrite;
    cl_ulong start, end;

    if (hostAlloc) {
        //a = (int*)cudaHostAlloc((void**)&a, size * sizeof(*a), cudaHostAllocDefault);
        a = (int*)malloc(size * sizeof(*a));
    } else {
        a = (int*)malloc(size * sizeof(*a));

        if (a == NULL) {
            printf("malloc fail with size %u.\n", size); 
            return 1;
        }
    }

    dev_a = clCreateBuffer( *context, CL_MEM_READ_WRITE, SIZE * sizeof(cl_uint), NULL, &ret);

    if (ret) {
        printf("clCreateBuffer fail code %d.\n", ret);
        return 1;
    }

    if (up)
        //cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice);
        ret = clEnqueueWriteBuffer(*queue, dev_a, CL_TRUE, 0, SIZE * sizeof(cl_uint), a, NULL, NULL, &evtWrite);
    else
        //cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost);
       //ret = clEnqueueReadBuffer(*queue, dev_a, CL_TRUE, 0, SIZE * sizeof(cl_uint), a, NULL, NULL, &evtWrite);
        ret = clEnqueueReadBuffer(*queue, dev_a, CL_TRUE, 0, SIZE * sizeof(cl_uint), a, NULL, NULL, &evtWrite);

    if (ret) {
        printf("clEnqueueWrite/ReadBuffer fail code %d.\n", ret);
        return 1;
    }

    ret = clWaitForEvents(1, &evtReadWrite);
    printf("clWaitForEvent return code %d.\n", ret);

    ret = clGetEventProfilingInfo(evtReadWrite,CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    
    if (ret != 0) {
        printf("clGetEventProfilingInfo (END) failed with code %d.\n", ret);
        return 1;
    }

    ret = clGetEventProfilingInfo(evtReadWrite,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);

    if (ret != 0) {
        printf("clGetEventProfilingInfo (END) failed with code %d.\n", ret);
        return 1;
    }

    printf("Elapsed time in ns: %d\n", elapsedTime);
    elapsedTime  = (end - start) * 1.0e-9f;

    /*
    if (hostAlloc) {
        cudaFreeHost(dev_a);
    } else {
        cudaFree(dev_a);
    }
    */
    //cudaFree(a);
    //cudaEventDestroy(start);
    //cudaEventDestroy(stop);
    return elapsedTime;
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
    uint a[SIZE];
    int i;

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

    cl_uint CONFIG_MAX_DEVICES = 20;
    cl_uint devices_available;

    enum enum_device_info_types {DEVINFO_STRING=1, DEVINFO_USHORT=2, DEVINFO_UINT=3, DEVINFO_ULONG=4, DEVINFO_SIZE_T=5};

    enum enum_device_info_types device_info_types[] = {
        DEVINFO_STRING, \
        DEVINFO_STRING, \    
        DEVINFO_STRING, \    
        DEVINFO_STRING, \    
        DEVINFO_ULONG, \    
        DEVINFO_ULONG, \    
        DEVINFO_USHORT, \    
        DEVINFO_UINT, \    
        DEVINFO_UINT, \    
        DEVINFO_SIZE_T, \    
        DEVINFO_UINT, \    
        DEVINFO_SIZE_T, \    
        DEVINFO_USHORT, \    
        DEVINFO_STRING, \    
	DEVINFO_SIZE_T \
    };
    char *str_device_info[]={\
        "CL_DEVICE_NAME", \
        "CL_DEVICE_VENDOR", \
        "CL_DEVICE_VERSION", \
        "CL_DRIVER_VERSION", \
        "CL_DEVICE_GLOBAL_MEM_SIZE", \
        "CL_DEVICE_LOCAL_MEM_SIZE", \
        "CL_DEVICE_LOCAL_MEM_TYPE", \
        "CL_DEVICE_MAX_CLOCK_FREQUENCY", \
        "CL_DEVICE_MAX_COMPUTE_UNITS", \
        "CL_DEVICE_MAX_WORK_GROUP_SIZE", \
        "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS", \
        "CL_DEVICE_MAX_WORK_ITEM_SIZES", \
        "CL_DEVICE_TYPE", \
        "CL_DEVICE_EXTENSIONS", \
	"CL_DEVICE_MAX_PARAMETER_SIZE" \

    };
    cl_device_id device[CONFIG_MAX_DEVICES];
    cl_device_info deviceInfos[]={\
        CL_DEVICE_NAME, \
        CL_DEVICE_VENDOR, \
        CL_DEVICE_VERSION, \
        CL_DRIVER_VERSION, \
        CL_DEVICE_GLOBAL_MEM_SIZE, \
        CL_DEVICE_LOCAL_MEM_SIZE, \
        CL_DEVICE_LOCAL_MEM_TYPE, \
        CL_DEVICE_MAX_CLOCK_FREQUENCY, \
        CL_DEVICE_MAX_COMPUTE_UNITS, \
        CL_DEVICE_MAX_WORK_GROUP_SIZE, \
        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, \
        CL_DEVICE_MAX_WORK_ITEM_SIZES, \
        CL_DEVICE_TYPE, \
        CL_DEVICE_EXTENSIONS, \
	CL_DEVICE_MAX_PARAMETER_SIZE \
};
    stat = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_ALL, CONFIG_MAX_DEVICES, device, &devices_available);

    printf("No. of devices available: %d.\n", devices_available);

    // 3. Create a context and command queue on that device.

    cl_context context = clCreateContext( NULL, 1,  &device[0], NULL, NULL, NULL);

    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
    cl_command_queue queue = clCreateCommandQueue( context, device[0], properties, &ret );
    //cl_command_queue queue = clCreateCommandQueue( context, device[0], CL_QUEUE_PROFILING_ENABLE, NULL );

    if (ret) {
        printf("Error: clCreateCommandQueue returned non-zero: %d.\n", ret);
        return 1;
    } else  {
        printf("clCreateCommandQueue return OK.... %d.\n", ret);
    }

    // 4. Perform runtime source compilation, and obtain kernel entry point.

   cl_program program = clCreateProgramWithSource( context, 1, &source,  NULL, NULL );

    if (ret) {
        printf("Error: clCreateProgramWithSource returned non-zero: %d.\n", ret);
        return 1;
    } else  {
        printf("clCreateProgramWithSource return OK.... %d.\n", ret);
    }

    float elapsedTime;

    printf("clCreateBuffer test:\n");
    float MB = (float)100 * SIZE * sizeof(int) / 1024 / 1024;
    elapsedTime = opencl_malloc_test(SIZE, CL_TRUE, 0, &context, &queue);
    printf("Time using clCreateBuffer(up): %3.1f ms.\n", elapsedTime);
    printf("MB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));
    elapsedTime = opencl_malloc_test(SIZE, CL_FALSE, 0, &context, &queue);
    printf("Time using clCreateBuffer(down): %3.1f ms.\n", elapsedTime);
    printf("MB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));

    printf("Hostalloc test:\n");
    elapsedTime = opencl_malloc_test(SIZE, CL_TRUE, 1, &context, &queue);
    printf("Time using clCreateBuffer(up): %3.1f ms.\n", elapsedTime);
    printf("MB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));
    elapsedTime = opencl_malloc_test(SIZE, CL_FALSE, 1, &context, &queue);
    printf("Time using clCreateBuffer(down): %3.1f ms.\n", elapsedTime);
    printf("MB/s during copy up: %3.1f.\n", MB / (elapsedTime / 1000));

    printf("\n");
    return 0;
}
