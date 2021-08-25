set(CMAKE_CPP_COMPILER g++)
cmake_minimum_required(VERSION 3.10)
# set the project name
project(Tutorial VERSION 1.0)
option(USE_MYMATH "Use tutorial provided math implementation" ON)
configure_file(TutorialConfig.h.in TutorialConfig.h)

cmake_minimum_required(VERSION 3.10)

# add the MathFunctions library

if(USE_MYMATH)
    add_subdirectory(MathFunctions)
    list(APPEND EXTRA_LIBS MathFunctions)
endif()

message("USE_MYMATH: ${USE_MYMATH}")

if(USE_MYMATH)
  list(APPEND EXTRA_LIBS MathFunctions)
  list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/MathFunctions")
endif()

# add the executable
add_executable(Tutorial tutorial.cxx)

if(USE_MYMATH)
    target_link_libraries(Tutorial PUBLIC ${EXTRA_LIBS})

    # add the binary tree to the search path for include files
    # so that we will find TutorialConfig.h
    target_include_directories(Tutorial PUBLIC
                               "${PROJECT_BINARY_DIR}"
                               )
else()
    target_link_libraries(Tutorial PUBLIC)

    # add the binary tree to the search path for include files
    # so that we will find TutorialConfig.h
    target_include_directories(Tutorial PUBLIC
                               "${PROJECT_BINARY_DIR}"
                               )
endif()

install(TARGETS Tutorial DESTINATION bin)
install(FILES "${PROJECT_BINARY_DIR}/TutorialConfig.h"
  DESTINATION include
  )

