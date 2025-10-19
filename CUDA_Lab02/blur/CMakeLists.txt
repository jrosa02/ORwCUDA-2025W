cmake_minimum_required(VERSION 3.18)
project(BlurApp LANGUAGES CXX CUDA)

# Ensure language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Use debug flags if needed
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "*** Using debug flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -Wpedantic")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O0")
endif()

# Add the utils directory (for image operations)
add_library(image STATIC
    ../utils/image.cc
    ../utils/matrix.cc
)

# Create a library that uses the CUDA code
add_library(blur STATIC
    blur.cu
)      
target_include_directories(blur PUBLIC ../utils)

# Create the main executable
add_executable(blur_main.out
    blur_main.cc
)
target_link_libraries(blur_main.out PRIVATE blur image)

# Create image_diff executable
add_executable(image_diff.out
    ../utils/image_diff.cc
)
target_link_libraries(image_diff.out PRIVATE image)