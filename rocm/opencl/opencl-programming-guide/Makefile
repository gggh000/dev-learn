FILE1=ex-code-1
FILE2=ex-code-2
FILE3=ex-code-3
FILE_GET_PLATFORM_INFO=ex-get-platform-info

ROCMOPENCL=/opt/rocm/opencl/

ex-code-1: $(FILE1).c
	g++ -o $(FILE1).o -c $(FILE1).c -I$(ROCMOPENCL)/include
	g++ -o $(FILE1) $(FILE1).o -lOpenCL -L$(ROCMOPENCL)/lib/x86_64

ex-code-2: $(FILE2).c
	g++ -o $(FILE2).o -c $(FILE2).c -I$(ROCMOPENCL)/include
	g++ -o $(FILE2) $(FILE2).o -lOpenCL -L$(ROCMOPENCL)/lib/x86_64

ex-code-3: $(FILE3).c
	g++ -o $(FILE3).o -c $(FILE3).c -I$(ROCMOPENCL)/include
	g++ -o $(FILE3) $(FILE3).o -lOpenCL -L$(ROCMOPENCL)/lib/x86_64

clean:
	rm -rf *.o $(FILE1) $(FILE2) $(FILE3)

ex-get-platform-info: $(FILE_GET_PLATFORM_INFO).c
	g++ -o $(FILE_GET_PLATFORM_INFO).o -c $(FILE_GET_PLATFORM_INFO).c -I$(ROCMOPENCL)/include
	g++ -o $(FILE_GET_PLATFORM_INFO) $(FILE_GET_PLATFORM_INFO).o -lOpenCL -L$(ROCMOPENCL)/lib/x86_64

