cmake_minimum_required(VERSION 3.18)
project(Conv2D LANGUAGES CXX CUDA)

# Ensure language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Use debug flags if needed
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "*** Using debug flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -Wpedantic")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O0")
endif()

# Build matrix library
add_library(matrix STATIC
    ../utils/matrix.cc
)
target_include_directories(matrix PUBLIC ../utils)

# Build image handling library
add_library(image STATIC
    ../utils/image.cc
)
target_include_directories(image PUBLIC ../utils)
target_link_libraries(image PRIVATE matrix)

# Create a library that uses the CUDA code
add_library(conv_2d STATIC
    conv_2d.cu
)      
target_include_directories(conv_2d PUBLIC ../utils)

# Create the main executable
add_executable(conv_2d_main.out
    conv_2d_main.cc
)
target_include_directories(conv_2d_main.out PUBLIC ../utils)
target_link_libraries(conv_2d_main.out PRIVATE conv_2d image)

# Create convolution output verificator
add_executable(image_diff.out
    ../utils/image_diff.cc
)
target_include_directories(image_diff.out PUBLIC ../utils)
target_link_libraries(image_diff.out PRIVATE image)