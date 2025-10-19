cmake_minimum_required(VERSION 3.18)
project(MatrixVectorMultApp LANGUAGES CXX CUDA)

# Ensure language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Use debug flags if needed
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "*** Using debug flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -Wpedantic")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O0")
endif()

# Add the utils directory (for matrix operations)
add_library(matrix STATIC
    ../utils/matrix.cc
)

# Create a library that uses the CUDA code
add_library(mult_mv STATIC
    mult_mv.cu
)      
target_include_directories(mult_mv PUBLIC ../utils)

# Create the main executable
add_executable(mult_mv_main.out
    mult_mv_main.cc
)
target_link_libraries(mult_mv_main.out PRIVATE mult_mv matrix)

# Create random matrix generator
add_executable(testcase_generator.out
    testcase_generator.cc
)
target_include_directories(testcase_generator.out PUBLIC ../utils)
target_link_libraries(testcase_generator.out PRIVATE matrix)