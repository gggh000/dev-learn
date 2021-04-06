cmake_minimum_required(VERSION 3.10)
set(ROCMOPENCL /opt/rocm/opencl/)
project(p25-cuda)
message( "ROCM library path:" ${ROCMOPENCL} )

#add_subdirectory(${ROCMOPENCL}/lib/ build)
add_executable(p25-cuda p25-cuda.c)
target_link_libraries(p25-cuda PUBLIC ${ROCMOPENCL}/lib/libOpenCL.so)
target_include_directories(p25-cuda PUBLIC ${ROCMOPENCL}/include)
