//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

// A minimalist OpenCL program.

#include <CL/cl.h>
#include <stdio.h>

#define printDeviceInfo(X)   printf("\n%s: %s",  (X));
#define declareDeviceInfo(X) char str(X)[] = "(X)";

#define NWITEMS 512
// A simple memset kernel
const char *source =
"kernel void memset(     global uint *dst )                         \n"
"{                                                                                                        \n"
"        dst[get_global_id(0)] = get_global_id(0);                \n"
"}                                                                                                        \n";

int main(int argc, char ** argv)
{
    int stat;
    char str1[100];
    size_t strLen;

    // 1. Get a platform.

    cl_uint CONFIG_MAX_PLATFORMS=20;
    cl_platform_id platforms[CONFIG_MAX_PLATFORMS];
    cl_uint platforms_available;

    clGetPlatformIDs(CONFIG_MAX_PLATFORMS, platforms, &platforms_available );
    printf("\nNo. of platforms available: %d", platforms_available);

    for (int i = 0 ; i < platforms_available; i ++ ) {
        printf("\nPlatform %d: %d.", i, platforms[i]);
    }

    // 2. Find a gpu/cpu device.

    cl_uint CONFIG_MAX_DEVICES=20;
    cl_uint devices_available;

    cl_device_id device[CONFIG_MAX_DEVICES];
    cl_device_info deviceInfos[]={CL_DEVICE_NAME, CL_DEVICE_VENDOR, CL_DEVICE_VERSION, CL_DRIVER_VERSION, CL_DEVICE_EXTENSIONS};
    stat = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_ALL, CONFIG_MAX_DEVICES, device, &devices_available);

    printf("\nNo. of devices available: %d", devices_available);

    for (int j = 0 ; j <  devices_available; j++) {
        for (int i = 0 ; i < sizeof(deviceInfos)/sizeof(cl_device_info); i ++ ) {
            clGetDeviceInfo(device[0], deviceInfos[i], sizeof(str1), str1, &strLen);

            if (stat == 0)  {
                printf("\n%s.", str1);
            } else {
                printf("\nclGetDevicesIDs FAIL.");
            return 1;
            }
        }
    }    

    /*

    // 3. Create a context and command queue on that device.
    cl_context context = clCreateContext( NULL, 1,  &device, NULL, NULL, NULL);

    cl_command_queue queue = clCreateCommandQueue( context, device, 0, NULL );

    // 4. Perform runtime source compilation, and obtain kernel entry point.
    cl_program program = clCreateProgramWithSource( context, 1, &source,  NULL, NULL );

    clBuildProgram( program, 1, &device, NULL, NULL, NULL );

    cl_kernel kernel = clCreateKernel( program, "memset", NULL );

    // 5. Create a data buffer.
    cl_mem buffer = clCreateBuffer( context, CL_MEM_WRITE_ONLY, NWITEMS * sizeof(cl_uint), NULL, NULL );

    // 6. Launch the kernel. Let OpenCL pick the local work size.
    size_t global_work_size = NWITEMS;
    clSetKernelArg(kernel, 0, sizeof(buffer), (void*) &buffer);

    clEnqueueNDRangeKernel( queue, kernel,  1,  NULL, &global_work_size, NULL, 0,  NULL, NULL);

    clFinish( queue );

    // 7. Look at the results via synchronous buffer map.
    cl_uint *ptr;
    ptr = (cl_uint *) clEnqueueMapBuffer( queue, buffer, CL_TRUE, CL_MAP_READ, 0, NWITEMS * sizeof(cl_uint), 0, NULL, NULL, NULL );

    int i;

    for(i=0; i < NWITEMS; i++)
    {
        if (i % 16 == 0) 
            printf("\n");

        printf("%03d: %04d. ", i, ptr[i]);
        
    }
    */
    printf("\n");
    return 0;
}
