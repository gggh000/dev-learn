# For llvm compilation, used resource at https://llvm.org/docs/CMake.html
# Embedding LLVM in your project section.

cmake_minimum_required(VERSION 3.10)
set(ROCMOPENCL /opt/rocm/opencl/)
message( "ROCM library path:" ${ROCMOPENCL} )
#add_subdirectory(${ROCMOPENCL}/lib/ build)
find_package(ZLIB REQUIRED)
find_package(LLVM REQUIRED CONFIG)
#CMAKE_FIND_LIBRARY_PREFIXES=lib
#CMAKE_FIND_LIBRARY_SUFFIXES=.so
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message("prefix: ${CMAKE_FIND_LIBRARY_PREFIXES}")

message("LLVM_DEFINITIONS: ${LLVM_DEFINITIONS}")
message("LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")	

SET(P25_CUDA p25-cuda)
SET(P189 p189)
SET(P41 p41) 
SET(EXCODE_CODE_1 ex-code-1)

foreach(var ${P25_CUDA} ${P189} ${P41} ${EXCODE_CODE_1})

	include_directories(${LLVM_INCLUDE_DIRS})
	include_directories(${var} PUBLIC ${ROCMOPENCL}/include)
	add_definitions(${LLVM_DEFINITIONS})

	message(STATUS "Generating makefile for ${var}")
	project(${var})
	add_executable(${var} ${var}.c)
	project(${var})
	add_executable(${var}.out ${var}.c)

	target_link_libraries(${var} PUBLIC ${ROCMOPENCL}/lib/libOpenCL.so)
	llvm_map_components_to_libnames(llvm_libs support core irreader)

	target_link_libraries(${var}.out ${llvm_libs})
endforeach()
