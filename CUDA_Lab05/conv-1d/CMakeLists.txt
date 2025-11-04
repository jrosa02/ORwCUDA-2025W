cmake_minimum_required(VERSION 3.18)
project(Conv1D LANGUAGES CXX CUDA)

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
add_library(conv_1d STATIC
    conv_1d.cu
)      

# Create the main executable
add_executable(conv_1d_main.out
    conv_1d_main.cc
)
target_include_directories(conv_1d_main.out PUBLIC ../utils)
target_link_libraries(conv_1d_main.out PRIVATE conv_1d)

# Create random convolution input generator
add_executable(testcase_generator.out
    testcase_generator.cc
)

# Create convolution output verificator
add_executable(testcase_verificator.out
    testcase_verificator.cc
)
target_include_directories(testcase_verificator.out PUBLIC ../utils)