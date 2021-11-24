FILE1=p46
FILE2=p242
FILE3=p284
FILE_GET_PLATFORM_INFO=ex-get-platform-info
ROCMOPENCL=/opt/rocm/opencl/
ROCMOPENCL_LIB=$(ROCMOPENCL)/lib/
FREE_IMAGE_INCLUDE=/git.co/freeimage/Source
p46: $(FILE1).cpp
	g++ -o $(FILE1).o -c $(FILE1).cpp -I$(ROCMOPENCL)/include
	g++ -o $(FILE1) $(FILE1).o -lOpenCL -L$(ROCMOPENCL_LIB)

p242: $(FILE2).cpp
	g++ -o $(FILE2).o -c $(FILE2).cpp -I$(ROCMOPENCL)/include
	g++ -o $(FILE2) $(FILE2).o -lOpenCL -L$(ROCMOPENCL_LIB)

p284: $(FILE3).cpp
	g++ -o $(FILE3).o -c $(FILE3).cpp -I$(ROCMOPENCL)/include -I$(FREE_IMAGE_INCLUDE)
	g++ -o $(FILE3) $(FILE3).o -lOpenCL -L$(ROCMOPENCL_LIB) -L$(FREE_IMAGE_INCLUDE)

clean:
	rm -rf *.o $(FILE1) $(FILE2)  $(FILE3)

ex-get-platform-info: $(FILE_GET_PLATFORM_INFO).c
	g++ -o $(FILE_GET_PLATFORM_INFO).o -c $(FILE_GET_PLATFORM_INFO).c -I$(ROCMOPENCL)/include
	g++ -o $(FILE_GET_PLATFORM_INFO) $(FILE_GET_PLATFORM_INFO).o -lOpenCL -L$(ROCMOPENCL_LIB)
