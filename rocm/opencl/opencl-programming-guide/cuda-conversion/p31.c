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

    return 0;
}
