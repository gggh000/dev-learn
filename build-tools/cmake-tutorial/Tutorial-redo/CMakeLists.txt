cmake_minimum_required(VERSION 3.10)
# set the project name
project(Tutorial VERSION 1.0)
configure_file(TutorialConfig.h.in TutorialConfig.h)

cmake_minimum_required(VERSION 3.10)

# add the MathFunctions library
add_subdirectory(MathFunctions)

# add the executable
add_executable(Tutorial tutorial.cxx)

target_link_libraries(Tutorial PUBLIC MathFunctions)

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
target_include_directories(Tutorial PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/MathFunctions"
                          )
