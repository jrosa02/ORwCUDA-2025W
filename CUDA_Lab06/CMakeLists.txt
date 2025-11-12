cmake_minimum_required(VERSION 3.18)
project(Reduction LANGUAGES CXX CUDA)

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
add_library(reduction STATIC
    reduction.cu
)

# Create the main executable
add_executable(reduction_main.out
    reduction_main.cc
)
target_link_libraries(reduction_main.out PRIVATE reduction)