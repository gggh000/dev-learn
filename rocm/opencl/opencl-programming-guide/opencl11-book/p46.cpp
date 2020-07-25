#include <CL/cl.h>
#include <stdio.h>
#include <thread>
#include <iostream>
#include <chrono>
#include <fstream>
#define ARRAY_SIZE 1024 
#define BUILD_FROM_FILE 0
using namespace std;

// A simple kernelfcn kernel
const char *source =

"kernel void kernelfcn(     global uint *dev_c, global uint *dev_a, global uint *dev_b)  \n"
"{                                                                      \n"
" uint tid = get_global_id(0);                                          \n"
" dev_c[tid] = dev_a[tid] + dev_b[tid];                                 \n"
"}                                                                      \n";

int main(int argc, char ** argv) {
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem memObjects[3] = {0,0,0};
    cl_int errNum;
    cl_int ret;

    cl_uint CONFIG_MAX_PLATFORMS=20;
    cl_platform_id platforms[CONFIG_MAX_PLATFORMS];
    cl_uint platforms_available;

    clGetPlatformIDs(CONFIG_MAX_PLATFORMS, platforms, &platforms_available );
    printf("\nNo. of platforms available: %d.\n", platforms_available);

    for (int i = 0 ; i < platforms_available; i ++ ) {
        printf("Platform %d: %d.\n", i, platforms[i]);
    }

    // 2. Find a gpu/cpu device.

    cl_uint CONFIG_MAX_DEVICES=10;
    cl_uint devices_available;
    cl_device_id devices[CONFIG_MAX_DEVICES];

    ret  = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_ALL, CONFIG_MAX_DEVICES, devices, &devices_available);
    printf("No. of devices available: %d.\n", devices_available);
    // device = devices[0];

    cout << "device ID 1st device: " << device << endl;

    // Create an opencl context on first available platform

    context = clCreateContext(NULL, 1, (const cl_device_id* )&devices[0], NULL, NULL, &ret);

    if (context == NULL ) {
        cerr << "Failed to create opencl context: errCode: " << ret << endl;
        return 1;
    }

    // Create a command queue on the first device available

    commandQueue = clCreateCommandQueue(context, devices[0], 0, &ret);
    
    if (commandQueue == NULL ) { 
       //Cleanup(context, commandQueue, program, kernel, memObjects);
       return 1;
    }

    if (BUILD_FROM_FILE==1) {
        // Create opencl program from helloworld.cl kernel source

        ifstream myFile;
        myFile.open ("HelloWorld.cl");
        string str;
        string src = "";
        
        while (getline(myFile, str))
            src +=  str;

        program = clCreateProgramWithSource(context, 1, (const char**)&src, (const size_t*)src.length(), &ret);

        if (program == NULL) { 
           //Cleanup(context, commandQueue, program, kernel, memObjects);
        }

        // Create opencl kernel.

        cout << "Building kernel from file...";
        kernel = clCreateKernel(program, "hello_kernel", &ret);
    } else { 
        cout << "Building kernel from source string...";
        kernel = clCreateKernel(program, "kernel_fcn", &ret);
    }

    if (kernel == NULL) {
        cerr << "Failed to create a kernel: errCode: " << ret << endl;
        //Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;    
    }

    // Create memory objects that will be used as arguments to kernel. Create host memory arrays
    // that will be used to store the arguments to the kernel.

    float result[ARRAY_SIZE];
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];

    for (int i = 0 ; i < ARRAY_SIZE;  i ++ ) {
        a[i]  = i;
        b[i] = i * 2;
    }

    cl_mem dev_memObjects = clCreateBuffer( context, CL_MEM_READ_WRITE, ARRAY_SIZE * sizeof(cl_uint), NULL, NULL );
    cl_mem dev_a = clCreateBuffer( context, CL_MEM_READ_WRITE, ARRAY_SIZE * sizeof(cl_uint), NULL, NULL );
    cl_mem dev_b  = clCreateBuffer( context, CL_MEM_READ_WRITE, ARRAY_SIZE * sizeof(cl_uint), NULL, NULL );

    ret = clEnqueueWriteBuffer(commandQueue, dev_a, CL_TRUE, 0, ARRAY_SIZE * sizeof(cl_uint), a, NULL, NULL, NULL);
    ret = clEnqueueWriteBuffer(commandQueue, dev_b, CL_TRUE, 0, ARRAY_SIZE * sizeof(cl_uint), b, NULL, NULL, NULL);
    //ret = clEnqueueWriteBuffer(commandQueue, dev_memObjects, CL_TRUE, 0, ARRAY_SIZE * sizeof(cl_uint), memObjects, NULL, NULL, NULL);

    // copy from a/b/memObjects to dev_a/dev_b/dev_memObjets.!!!!

    /*
    if (!CreateMemObjects(context, memObjects, a, b))
    {
        //Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }
    */

    // Set the kernel arguments (result, a, b)

    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);

    if (errNum != CL_SUCCESS)  {
        cerr << "Error setting kernel arguments" << endl;
        //Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    size_t globalWorkSize[1] = {ARRAY_SIZE} ; 
    size_t localWorkSize[1] = {1} ; 

    // Queue the kernel up for execution across the array

    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    
    if (errNum != CL_SUCCESS) { 
        cerr << "Error queueing kernel for execution." << endl;
        //Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Read the output buffer back to the Host.

    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0, ARRAY_SIZE * sizeof(float), result, 0, NULL, NULL);

    if (errNum != CL_SUCCESS) { 
        cerr << "Error reading results buffer." << endl;
        //Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Output the result buffer.

    for (int i = 0; i < ARRAY_SIZE; i ++ ) {
        cout << result[i] << " ";
    }

    cout << endl;    
    cout << "Executed program successfully." << endl;
    //Cleanup(context, commandQueue, program, kernel, memObjects);
    return 0;
    
}




