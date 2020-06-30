    //
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

// A minimalist OpenCL program.

#include <CL/cl.h>
#include <stdio.h>

#define printDeviceInfo(X)   printf("\n%s: %s",  (X));
#define declareDeviceInfo(X) char str(X)[] = "(X)";

#define NWITEMS 512
// A simple simple_add kernel
const char *source =

"kernel void simple_add(     global uint *c,  uint a, uint b)      \n"
"{                                           \n"
" *c = a + b;                               \n"
" *c = 68;                                  \n"
"}                                           \n";

int main(int argc, char ** argv) {
    int c;
    int * dev_c;

    int stat;
    char str1[100];
    size_t strLen;
    int i;
    cl_int ret; 

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

    if (ret) {
    printf("Error: clCreateContext returned non-zero: %d.\n", ret);
    return 1;
    }

    cl_command_queue queue = clCreateCommandQueue( context, device, 0, &ret );

    if (ret) {
    printf("Error: clCreateCommandQueue returned non-zero: %d.\n", ret);
    return 1;
    }
    // 4. Perform runtime source compilation, and obtain kernel entry point.

    cl_program program = clCreateProgramWithSource( context, 1, &source,  NULL, &ret);

    if (ret) {
	printf("Error: clCreateProgramWithSource returned non-zero: %d.\n", ret);
	return 1;
    }

    clBuildProgram( program, 1, &device, NULL, NULL, NULL );
    cl_kernel kernel = clCreateKernel( program, "simple_add", &ret);

    if (ret) {
	printf("Error: clCreateKernel returned non-zero: %d.\n", ret);
	return 1;
    }

    // 5. Create a data buffer.

    cl_mem buffer = clCreateBuffer( context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, NULL );

    // 6. Launch the kernel. Let OpenCL pick the local work size.

    size_t global_work_size = 1;  
    clSetKernelArg(kernel, 0, sizeof(buffer), (void*) &buffer);
    cl_uint a = 2;
    cl_uint b = 7;
    //clSetKernelArg(kernel, 1, sizeof(a), (void*)a);
    //clSetKernelArg(kernel, 2, sizeof(b), (void*)b);
    clEnqueueNDRangeKernel( queue, kernel,  1,  NULL, &global_work_size, NULL, 0,  NULL, NULL);
    ret = clFinish( queue );

    if (ret) {
	printf("Error: clFinish returned non-zero: %u", ret);
	return 1;
    }

    // 7. Look at the results via synchronous buffer map.

    cl_uint *ptr;
    ptr = (cl_uint *) clEnqueueMapBuffer( queue, buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(cl_uint), 0, NULL, NULL, &ret);

    if (ret == 0) {
        for (int i = 0; i < 32; i++)
            printf("output is: idx: %d, %d\n", i, ptr[i]);
    } else {
        printf("ERROR: clEnqueueMapBuffer returned error, error code: %d.\n", ret);
        printf("output is: %d\n", ptr[0]);
    } 
    	        
        return 0;
}
