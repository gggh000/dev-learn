    //
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

// A minimalist OpenCL program.

#include <CL/cl.h>
#include <stdio.h>

#define printDeviceInfo(X)   printf("\n%s: %s",  (X));
#define declareDeviceInfo(X) char str(X)[] = "(X)";
#define DEBUG 1

// A simple kernelfcn kernel
const char *source =

"kernel void kernelfcn(     global uint *dev_c,  global uint * dev_a, global uint * dev_b)      \n"
"{                                           \n"
" uint tid = get_global_id(0);               \n"
" *dev_c = *dev_a + *dev_b;                  \n"
"}                                           \n";

/*
*/

int main(int argc, char ** argv) {
 
    int stat;
    char str1[100];
    size_t strLen;
    cl_int ret; 
    uint a[1], b[1], c[1];
    int i;
    const void* ptr;

    // 1. Get a platform.

    cl_platform_id platform;
    clGetPlatformIDs( 1, &platform, NULL );

    // 2. Find a gpu device.

    cl_device_id device;
    cl_device_info deviceInfos[]={CL_DEVICE_NAME, CL_DEVICE_VENDOR, CL_DEVICE_VERSION, CL_DRIVER_VERSION, CL_DEVICE_EXTENSIONS};

    stat = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    for (int i = 0 ; i < sizeof(deviceInfos)/sizeof(cl_device_info); i ++ ) {
        clGetDeviceInfo(device, deviceInfos[i], sizeof(str1), str1, &strLen);

        if (stat == 0)  {
            printf("\n%s.", str1);
        } else {
            printf("\nclGetDevicesIDs FAIL.");
        return 1;
        }
    }    

    printf("\n");

    // 3. Create a context and command queue on that device.

    cl_context context = clCreateContext( NULL, 1,  &device, NULL, NULL, &ret);
    cl_command_queue queue = clCreateCommandQueue( context, device, 0,  &ret);

    // 4. Perform runtime source compilation, and obtain kernel entry point.

    cl_program program = clCreateProgramWithSource( context, 1, &source,  NULL, &ret);

    if (ret) {
        printf("Error: clCreateProgramWithSource returned non-zero: %d.\n", ret);
        return 1;
    } else  {
        printf("clCreateProgramWithSource return OK.... %d.\n", ret);
    }

    clBuildProgram( program, 1, &device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel( program, "kernelfcn", &ret);

    if (ret) {
        printf("Error: clCreateKernel returned non-zero: %d.\n", ret);
//        return 1;
    } else  {
        printf("clCreateKernel return OK.... %d.\n", ret);
    }


    // 5. Create a data buffer.

    *a = 2; 
    *b = 7;

    printf("Creating mem on GPU.....");
    cl_mem dev_a = clCreateBuffer( context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, NULL );
    cl_mem dev_b = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(cl_uint), NULL, NULL );
    cl_mem dev_c = clCreateBuffer( context, CL_MEM_WRITE_ONLY, sizeof(cl_uint), NULL, NULL );

    if (DEBUG == 1)
        printf("Copying data to GPU...");

    ret = clEnqueueWriteBuffer(queue, dev_a, CL_TRUE, 0, sizeof(cl_uint), a, NULL, NULL, NULL);
    printf("ret: %d\n", ret); 
    ret = clEnqueueWriteBuffer(queue, dev_b, CL_TRUE, 0, sizeof(cl_uint), b, NULL, NULL, NULL);
    printf("ret: %d\n", ret); 

    // 6. Launch the kernel. Let OpenCL pick the local work size.

    if (DEBUG == 1) {
        printf("Launch kernel\n");
        //getchar();
    }

    size_t global_work_size = 1;
    size_t local_work_size = 1;

    printf("set kernel args...\n");
    clSetKernelArg(kernel, 0, sizeof(dev_c), (void*) &dev_c);
    printf("ret: %d\n", ret); 
    clSetKernelArg(kernel, 1, sizeof(dev_a), (void*) &dev_a);
    printf("ret: %d\n", ret); 
    clSetKernelArg(kernel, 2, sizeof(dev_b), (void*) &dev_b);
    printf("ret: %d\n", ret); 
    clEnqueueNDRangeKernel( queue, kernel,  1, NULL, &global_work_size, &local_work_size, 0,  NULL, NULL);
    printf("clEnqueueNDRangeKernel OK...\n");

    clFinish( queue );

    if (ret) {
        printf("Error: clFinish returned non-zero: %u", ret);
        return 1;
    }

    // 7. Look at the results via synchronous buffer map.

    printf("Reading back from GPU the sum...\n");

    ret = clEnqueueReadBuffer(queue, dev_c, CL_TRUE, 0, sizeof(cl_uint), c, NULL, NULL, NULL);
    printf("ret: %d\n", ret); 
    printf("output is: (%d + %d) = %d\n", *a, *b, *c);
    return 0;
}

