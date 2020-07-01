//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

// A minimalist OpenCL program.

#include <CL/cl.h>
#include <stdio.h>

#define printDeviceInfo(X)   printf("\n%s: %s",  (X));
#define declareDeviceInfo(X) char str(X)[] = "(X)";

#define NWITEMS 2048
#define LOCAL_WORK_SIZE 256
#define DEBUG 0
// A simple kernelfcn kernel
const char *source =

"kernel void kernelfcn(     global uint *dev_c, global uint *dev_a, global uint *dev_b)  \n"
"{                                                                      \n"
" uint tid = get_global_id(0);                                          \n"
" dev_c[tid] = dev_a[tid] + dev_b[tid];                                 \n"
"}                                                                      \n";

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

    /*

    for (int j = 0 ; j <  devices_available; j++) {
        for (int i = 0 ; i < sizeof(deviceInfos)/sizeof(cl_device_info); i ++ ) {

            if (stat == 0)  {
                switch (device_info_types[i]) {
                    case  DEVINFO_STRING:
		        clGetDeviceInfo(device[0], deviceInfos[i], sizeof(str1), str1, &strLen);
                        printf("\n%40s: %30s.", str_device_info[i], str1);
                        break;
                    case  DEVINFO_USHORT:
		        clGetDeviceInfo(device[0], deviceInfos[i], sizeof(ushort), (void*)&ushort1, &strLen);
                        printf("\n%40s: %02u (%02x).", str_device_info[i], ushort1, ushort1);
			break;
                    case  DEVINFO_UINT:
		        clGetDeviceInfo(device[0], deviceInfos[i], sizeof(uint), (void*)&uint1, &strLen);
                        printf("\n%40s: %04u (%04x).", str_device_info[i], uint1, uint1);
			break;
                    case  DEVINFO_ULONG:
		        clGetDeviceInfo(device[0], deviceInfos[i], sizeof(ulong), (void*)&ulong1, &strLen);
                        printf("\n%40s: %08u (%08x).", str_device_info[i], ulong1, ulong1);
                        break;
		    case DEVINFO_SIZE_T:
		        clGetDeviceInfo(device[0], deviceInfos[i], sizeof(ulong), (void*)&ulong1, &strLen);
                        printf("\n%40s: %08u (%08x).", str_device_info[i], ulong1, ulong1);
                        break;
                }
                //enum device_info_types={DEVINFO_STRING=1, DEVINFO_USHORT=2, DEVINFO_UINT=3, DEVINFO_ULONG=4};
            } else {
                printf("\nclGetDevicesIDs FAIL.");
            return 1;
            }
        }
    } 
    */   

    // 3. Create a context and command queue on that device.

    cl_context context = clCreateContext( NULL, 1,  &device[0], NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue( context, device[0], 0, NULL );

    // 4. Perform runtime source compilation, and obtain kernel entry point.
    cl_program program = clCreateProgramWithSource( context, 1, &source,  NULL, NULL );
    clBuildProgram( program, 1, device, NULL, NULL, NULL );
    cl_kernel kernel = clCreateKernel( program, "kernelfcn", &ret);

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

    /*
    cl_int clEnqueueWriteBuffer(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    size_t offset,
    size_t size,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
    */

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

    size_t global_work_size = NWITEMS;
    size_t local_work_size = LOCAL_WORK_SIZE;

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

    printf("\n");
    return 0;
}
