cmake_minimum_required(VERSION 3.5)
project(example)

add_executable(myapp
    "${CMAKE_CURRENT_SOURCE_DIR}/myapp.c"
)

target_compile_definitions(myapp
    PRIVATE
        AWESOME
)

add_library(mylib mylibrary.c)
target_link_libraries(myapp PUBLIC mylib)
