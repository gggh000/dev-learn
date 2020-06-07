//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

// A minimalist OpenCL program.

#include <CL/cl.h>
#include <stdio.h>

#define printDeviceInfo(X)   printf("\n%s: %s",  (X));
#define declareDeviceInfo(X) char str(X)[] = "(X)";

#define NWITEMS 4096
#define GET_LOCAL_ID 0
#define GET_GLOBAL_ID 1
#define GET_ID_TYPE GET_LOCAL_ID
// A simple memset kernel
const char *source =

#if GET_ID_TYPE == GET_GLOBAL_ID

"kernel void memset(     global uint *l_global_id, global uint *l_global_size)      \n"
"{                                                                      \n"
"       l_global_id[get_global_id(0)] = get_global_id(0);               \n"
"       l_global_size[get_global_id(0)] = get_global_id(0);             \n"
"}                                                                      \n";

#else

"kernel void memset(     global uint *l_global_id, global uint *l_global_size)      \n"
"{                                                                      \n"
"       l_global_id[get_global_id(0)] = get_local_id(0);               \n"
"       l_global_size[get_global_id(0)] = get_local_size(0);           \n"
"} \n";
#endif


int main(int argc, char ** argv)
{
    int stat;
    char str1[100];
    ushort ushort1;
    uint uint1;
    ulong ulong1;
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
        DEVINFO_UINT, \    
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

    printf("\nNo. of devices available: %d", devices_available);

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

    // 3. Create a context and command queue on that device.

    cl_context context = clCreateContext( NULL, 1,  &device[0], NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue( context, device[0], 0, NULL );

    // 4. Perform runtime source compilation, and obtain kernel entry point.
    cl_program program = clCreateProgramWithSource( context, 1, &source,  NULL, NULL );

    clBuildProgram( program, 1, device, NULL, NULL, NULL );

    cl_kernel kernel = clCreateKernel( program, "memset", NULL );

    // 5. Create a data buffer.
    cl_mem global_id_buffer    = clCreateBuffer( context, CL_MEM_WRITE_ONLY, NWITEMS * sizeof(cl_uint), NULL, NULL );
    cl_mem global_size_buffer  = clCreateBuffer( context, CL_MEM_WRITE_ONLY, NWITEMS * sizeof(cl_uint), NULL, NULL );

    // 6. Launch the kernel. Let OpenCL pick the local work size.

    size_t global_work_size = NWITEMS;
    clSetKernelArg(kernel, 0, sizeof(global_id_buffer), (void*) &global_id_buffer);
    clSetKernelArg(kernel, 1, sizeof(global_size_buffer), (void*) &global_size_buffer);
    clEnqueueNDRangeKernel( queue, kernel,  1,  NULL, &global_work_size, NULL, 0,  NULL, NULL);
    clFinish( queue );

    // 7. Look at the results via synchronous buffer map.

    cl_uint *int_global_id, *int_global_size;
    int_global_id  = (cl_uint *) clEnqueueMapBuffer( queue, global_id_buffer, CL_TRUE, CL_MAP_READ, 0, NWITEMS * sizeof(cl_uint), 0, NULL, NULL, NULL );
    int_global_size  = (cl_uint *) clEnqueueMapBuffer( queue, global_size_buffer, CL_TRUE, CL_MAP_READ, 0, NWITEMS * sizeof(cl_uint), 0, NULL, NULL, NULL );

    int i;

    for(i=0; i < NWITEMS; i+=100)
    {

#if GET_ID_TYPE == GET_GLOBAL_ID
    printf("\n%2d: global_id: 0x%08u. global_size: 0x%08u", i, int_global_id[i], int_global_size[i]);
#else
    printf("\n%2d: local_id: 0x%08u. local_size: 0x%08u", i, int_global_id[i], int_global_size[i]);
#endif
        
    }

    printf("\n");
    return 0;
}
