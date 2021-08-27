set(CMAKE_CPP_COMPILER g++)
cmake_minimum_required(VERSION 3.10)
# set the project name
project(Tutorial VERSION 1.0)
configure_file(TutorialConfig.h.in TutorialConfig.h)

cmake_minimum_required(VERSION 3.10)

# control where the static and shared libraries are built so that on windows
# we don't need to tinker with the path to run the executable

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# option section.

#option(USE_MYMATH "Use tutorial provided math implementation" ON)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

add_executable(Tutorial tutorial.cxx)

install(TARGETS Tutorial DESTINATION bin)
install(FILES "${PROJECT_BINARY_DIR}/TutorialConfig.h"
  DESTINATION include
  )

include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "${Tutorial_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${Tutorial_VERSION_MINOR}")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER, "Guyen G")
set(CPACK_PACKAGE_CONTACT, "Guyen G")
include(CPack)

# enable testing
enable_testing()

# enable dashboard scripting
include(CTest)

set(CTEST_PROJECT_NAME "CMakeTutorial")
set(CTEST_NIGHTLY_START_TIME "00:00:00 EST")

set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "my.cdash.org")
set(CTEST_DROP_LOCATION "/submit.php?project=CMakeTutorial")
set(CTEST_DROP_SITE_CDASH TRUE)


