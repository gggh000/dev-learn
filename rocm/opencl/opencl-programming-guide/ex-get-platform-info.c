//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

// A minimalist OpenCL program.

#include <CL/cl.h>
#include <stdio.h>

#define printDeviceInfo(X)   printf("\n%s: %s",  (X));
#define declareDeviceInfo(X) char str(X)[] = "(X)";

#define G_NWITEMS 2048
#define L_NWITEMS 128
// A simple k_get_global_id kernel
const char *src_get_global_id =
"kernel void k_get_global_id(     global uint *l_global_id, global uint *l_global_size)      \n"
"{                                                                      \n"
"       l_global_id[get_global_id(0)] = get_global_id(0);               \n"
"       l_global_size[get_global_id(0)] = get_global_size(0);           \n"
"}                                                                      \n";

const char *src_get_local_id =
"kernel void k_get_local_id(     global uint *l_local_id, global uint *l_local_size)      \n"
"{                                                                      \n"
"       l_local_id[get_local_id(0)] = get_local_id(0);                  \n"
"       l_local_size[get_local_id(0)] = get_local_size(0);              \n"
"}                                                                      \n";

int main(int argc, char ** argv)
{
    int stat;
    char str1[100];
    ushort ushort1;
    uint uint1;
    ulong ulong1;
    size_t sizet1;
    size_t * sizetl1;
    size_t strLen;
    int i;
    int inc;
    int result;
    inc =32;

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

    enum enum_device_info_types {DEVINFO_STRING=1, DEVINFO_USHORT=2, DEVINFO_UINT=3, DEVINFO_ULONG=4, DEVINFO_SIZET=5, DEVINFO_SIZETL=6};

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
        DEVINFO_SIZET, \    
        DEVINFO_UINT, \    
        DEVINFO_SIZETL, \    
        DEVINFO_USHORT, \    
        DEVINFO_STRING\    
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
        "CL_DEVICE_EXTENSIONS" \
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
        CL_DEVICE_EXTENSIONS};
    stat = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_ALL, CONFIG_MAX_DEVICES, device, &devices_available);

    printf("\nNo. of devices available: %d", devices_available);

    for (int j = 0 ; j <  devices_available; j++) {
        for (int i = 0 ; i < sizeof(deviceInfos)/sizeof(cl_device_info); i ++ ) {

            if (stat == 0)  {
                switch (device_info_types[i]) {
                    case  DEVINFO_STRING:
		                result = clGetDeviceInfo(device[0], deviceInfos[i], sizeof(str1), str1, &strLen);
                        printf("\n[%04d] %40s: %30s.", result, str_device_info[i], str1);
                        break;
                    case  DEVINFO_USHORT:
		                result = clGetDeviceInfo(device[0], deviceInfos[i], sizeof(ushort), (void*)&ushort1, &strLen);
                        printf("\n[%04d]%40s: %02u (%02x).", result, str_device_info[i], ushort1, ushort1);
		               	break;
                    case  DEVINFO_UINT:
		                result = clGetDeviceInfo(device[0], deviceInfos[i], sizeof(uint), (void*)&uint1, &strLen);
                        printf("\n[%04d]%40s: %04u (%04x).", result, str_device_info[i], uint1, uint1);
			            break;
                    case  DEVINFO_SIZET:
		                result = clGetDeviceInfo(device[0], deviceInfos[i], sizeof(size_t), (void*)&sizet1, &strLen);
                        printf("\n[%04d]%40s: %04u (%04x).", result, str_device_info[i], sizet1, sizet1);
			            break;
                    case  DEVINFO_SIZETL:
		                result = clGetDeviceInfo(device[0], deviceInfos[i], sizeof(size_t), (void*)&sizetl1, &strLen);
                        printf("\n[%04d]%40s: %04u (%04x).", result, str_device_info[i], sizetl1, sizetl1);
			            break;
                    case  DEVINFO_ULONG:
		                result = clGetDeviceInfo(device[0], deviceInfos[i], sizeof(ulong), (void*)&ulong1, &strLen);
                        printf("\n[%04d]%40s: %08u (%08x).", result, str_device_info[i], ulong1, ulong1);
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
    cl_program program1 = clCreateProgramWithSource( context, 1, &src_get_global_id,  NULL, NULL );
    cl_program program2 = clCreateProgramWithSource( context, 1, &src_get_local_id,  NULL, NULL );

    clBuildProgram( program1, 1, device, NULL, NULL, NULL );
    clBuildProgram( program2, 1, device, NULL, NULL, NULL );

    cl_kernel kernel1 = clCreateKernel( program1, "k_get_global_id", NULL );
    cl_kernel kernel2 = clCreateKernel( program2, "k_get_local_id", NULL );

    // 5. Create a data buffer.
    cl_mem global_id_buffer    = clCreateBuffer( context, CL_MEM_WRITE_ONLY, G_NWITEMS * sizeof(cl_uint), NULL, NULL );
    cl_mem global_size_buffer  = clCreateBuffer( context, CL_MEM_WRITE_ONLY, G_NWITEMS * sizeof(cl_uint), NULL, NULL );
    cl_mem local_id_buffer    = clCreateBuffer( context, CL_MEM_WRITE_ONLY, G_NWITEMS * sizeof(cl_uint), NULL, NULL );
    cl_mem local_size_buffer  = clCreateBuffer( context, CL_MEM_WRITE_ONLY, G_NWITEMS * sizeof(cl_uint), NULL, NULL );

    // 6. Launch the kernel. Let OpenCL pick the local work size.

    size_t global_work_size = G_NWITEMS;
    clSetKernelArg(kernel1, 0, sizeof(global_id_buffer), (void*) &global_id_buffer);
    clSetKernelArg(kernel1, 1, sizeof(global_size_buffer), (void*) &global_size_buffer);
    clEnqueueNDRangeKernel( queue, kernel1,  1,  NULL, &global_work_size, NULL, 0,  NULL, NULL);
    clFinish( queue );

    // 7. Look at the results via synchronous buffer map.

    cl_uint *int_global_id, *int_global_size;
    int_global_id  = (cl_uint *) clEnqueueMapBuffer( queue, global_id_buffer, CL_TRUE, CL_MAP_READ, 0, G_NWITEMS * sizeof(cl_uint), 0, NULL, NULL, NULL );
    int_global_size  = (cl_uint *) clEnqueueMapBuffer( queue, global_size_buffer, CL_TRUE, CL_MAP_READ, 0, G_NWITEMS * sizeof(cl_uint), 0, NULL, NULL, NULL );

    for(i=0; i < G_NWITEMS; i+=inc)
    {

        printf("\n%2d: global_id: 0x%08x. global_size: 0x%08x", i, int_global_id[i], int_global_size[i]);
        
    }

    printf("\n");

    
    size_t local_work_size = L_NWITEMS;
    clSetKernelArg(kernel2, 0, sizeof(local_id_buffer), (void*) &local_id_buffer);
    clSetKernelArg(kernel2, 1, sizeof(local_size_buffer), (void*) &local_size_buffer);
    clEnqueueNDRangeKernel( queue, kernel2,  1,  NULL, &global_work_size, &local_work_size, 0,  NULL, NULL);
    clFinish( queue );

    cl_uint *int_local_id, *int_local_size;
    int_local_id  = (cl_uint *) clEnqueueMapBuffer( queue, local_id_buffer, CL_TRUE, CL_MAP_READ, 0, G_NWITEMS * sizeof(cl_uint), 0, NULL, NULL, NULL );
    int_local_size  = (cl_uint *) clEnqueueMapBuffer( queue, local_size_buffer, CL_TRUE, CL_MAP_READ, 0, G_NWITEMS * sizeof(cl_uint), 0, NULL, NULL, NULL );

    for(i=0; i < G_NWITEMS; i+=inc)
    {
        printf("\n%2d: local_id: 0x%08x. local_size: 0x%08x", i, int_local_id[i], int_local_size[i]);
    }

    printf("\n");
    return 0;
}
