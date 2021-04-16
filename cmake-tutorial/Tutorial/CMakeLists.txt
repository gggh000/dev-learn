option(USE_MYMATH "Use tutorial provided math implementation" ON)

cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(Tutorial VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file(TutorialConfig.h.in TutorialConfig.h)
message("PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")

# add the executable
add_executable(Tutorial tutorial.cxx)
add_subdirectory(MathFunctions)
target_link_libraries(Tutorial PUBLIC MathFunctions)

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
target_include_directories(Tutorial PUBLIC
                           	"${PROJECT_BINARY_DIR}" 
							"${PROJECT_SOURCE_DIR}/MathFunctions"
                           )
