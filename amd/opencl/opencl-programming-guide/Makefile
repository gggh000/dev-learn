FILE1=ex-code-1
FILE2=ex-code-2

ROCMOPENCL=/opt/rocm/opencl/

ex-code-1: $(FILE1).c
	g++ -o $(FILE1).o -c $(FILE1).c -I$(ROCMOPENCL)/include
	g++ -o $(FILE1) $(FILE1).o -lOpenCL -L$(ROCMOPENCL)/lib/x86_64

ex-code-2: $(FILE2).c
	g++ -o $(FILE2).o -c $(FILE2).c -I$(ROCMOPENCL)/include
	g++ -o $(FILE2) $(FILE2).o -lOpenCL -L$(ROCMOPENCL)/lib/x86_64
