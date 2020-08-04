cmake_minimum_required(VERSION 3.10)
set(ROCMOPENCL /opt/rocm/opencl/)
project(p46)
message( "ROCM library path:" ${ROCMOPENCL} )

add_subdirectory(${ROCMOPENCL}/lib/ build)
add_executable(p46 p46.cpp)
target_link_libraries(p46 PUBLIC ${ROCMOPENCL}/lib/libOpenCL.so)
target_include_directories(p46 PUBLIC ${ROCMOPENCL}/include)
