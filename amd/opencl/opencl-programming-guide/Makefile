FILE=ex-code-1
ROCMOPENCL=/opt/rocm/opencl/

ex-code-1: $(FILE).c
	g++ -o $(FILE).o -c $(FILE).c -I$(ROCMOPENCL)/include
	g++ -o $(FILE) $(FILE).o -lOpenCL -L$(ROCMOPENCL)/lib/x86_64
