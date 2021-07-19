cmake_minimum_required(VERSION 3.5)
set(CUDA_NVCC_EXECUTABLE "/usr/local/cuda-9.2/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-9.2/bin/")
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
set(CUDA_CUDART_LIBRARY "/usr/local/cuda/lib64")
set(CUDA_VERSION 9.2)
#set(ROCMOPENCL /opt/rocm/opencl/)
#message( "ROCM library path:" ${ROCMOPENCL} )
#find_package(CUDA_CUDART_LIBRARY REQUIRED)
find_package(CUDA QUIET REQUIRED)
#add_subdirectory(${ROCMOPENCL}/lib/ build)

if(FALSE)
foreach(var p18 p33 p25 p45 p51 p55 p58 p84 p125 p115 p165)
    message(STATUS "Generating makefile for ${var}")
    project(${var})
    cuda_add_executable(${var} ${var}.c)
    #project(${var})
    #add_executable(${var} ${var}.c)
    #target_link_libraries(${var} PUBLIC ${ROCMOPENCL}/lib/libOpenCL.so)
    #target_include_directories(${var} PUBLIC ${ROCMOPENCL}/include)
endforeach()
endif()
#exclused FOR TIME BEING because of customer shell build script:
#p84
#p100
#p125

#a.out         CMakeLists.txt  p100.cu   p107.cu  p125.cu   p18.cu  p51.cu  p84.cu     run-p100.sh
#build-125.sh  cuda.prop.out   p100.log  p112.cu  p125.out  p33.cu  p55.cu  p84.out    run-p125.sh
#build.sh      ncu.sh          p100.out  p115.cu  p165.cu   p45.cu  p58.cu  readme.md  run-p84.sh

