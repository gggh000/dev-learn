cmake_minimum_required (VERSION 2.6)
project (test)

# The version number.

set (test_VERSION_MAJOR 1)
set (test_VERSION_MINOR 0)

# should we use our own math functions?

option(DYN_BUILD "Dynamic build" OFF)

# configure a header file to pass some of the CMake settings
# to the source code

configure_file (
  "${PROJECT_SOURCE_DIR}/testConfig.h.in"
  "${PROJECT_BINARY_DIR}/testConfig.h"
  )
 
# add the binary tree to the search path for include files
# so that we will find testConfig.h

include_directories("${PROJECT_BINARY_DIR}")
 
# add the MathFunctions library?
#
#if (USE_MYMATH)
#  include_directories ("${PROJECT_SOURCE_DIR}/MathFunctions")
#  add_subdirectory (MathFunctions)
#  set (EXTRA_LIBS ${EXTRA_LIBS} MathFunctions)
#endif (USE_MYMATH)

#add the executable

add_executable(test test.c)
target_link_libraries(test PUBLIC lib1)
target_link_libraries(test PUBLIC lib2)
target_link_libraries(test PUBLIC lib3)

if(DYN_BUILD)
  message("Building shared library...")
  add_library(lib1 SHARED libfile1.c)
  add_library(lib2 SHARED libfile2.c)
  add_library(lib3 SHARED libfile3.c)
else (DYN_BUILD)
  message("Building static library...")
  add_library(lib1 STATIC libfile1.c)
  add_library(lib2 STATIC libfile2.c)
  add_library(lib3 STATIC libfile3.c)
endif (DYN_BUILD)

add_custom_command(OUTPUT output1 COMMAND gcc *.o -shared -o liball.so)


#install (TARGETS MathFunctions DESTINATION bin)
#install (FILES MathFunctions.h DESTINATION include)

#include(CTest)

# does the application run
#add_test (testRuns test 25)
# does it sqrt of 25
#add_test (testComp25 test 25)
##set_tests_properties (testComp25 PROPERTIES PASS_REGULAR_EXPRESSION "25 is 5")
# does it handle negative numbers
#add_test (testNegative test -25)
#set_tests_properties (testNegative PROPERTIES PASS_REGULAR_EXPRESSION "-25 is 0")
# does it handle small numbers
#add_test (testSmall test 0.0001)
#set_tests_properties (testSmall PROPERTIES PASS_REGULAR_EXPRESSION "0.0001 is 0.01")
# does the usage message work?
#add_test (testUsage test)
#set_tests_properties (testUsage PROPERTIES PASS_REGULAR_EXPRESSION "Usage:.*number")

#define a macro to simplify adding tests, then use it
#macro (do_test arg result)
  #add_test (testComp${arg} test ${arg})
  #set_tests_properties (testComp${arg}
#    PROPERTIES PASS_REGULAR_EXPRESSION ${result})
#endmacro (do_test)
 
# do a bunch of result based tests
#do_test (25 "25 is 5")
#do_test (-25 "-25 is 0")
