cmake_minimum_required(VERSION 3.18)
project(CheckErrorApp LANGUAGES CXX CUDA)

# Ensure language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Use debug flags if needed
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "*** Using debug flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -Wpedantic")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O0")
endif()

# Create a library that uses the CUDA code
add_library(check_error STATIC
    check_error.cu
)

# Ensure that "check_error.h" will be found
target_include_directories(check_error PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)       

# Create the main executable
add_executable(check_error_main.out
    check_error_main.cc
)

# Link the CUDA library
target_link_libraries(check_error_main.out PRIVATE check_error)