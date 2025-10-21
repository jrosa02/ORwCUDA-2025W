cmake_minimum_required(VERSION 3.18)
project(MatrixMatrixMultApp LANGUAGES CXX CUDA)

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
add_library(mult_mm STATIC
    mult_mm.cu
)      
target_include_directories(mult_mm PUBLIC ../utils)

# Create the main executable
add_executable(mult_mm_main.out
    mult_mm_main.cc
)
target_link_libraries(mult_mm_main.out PRIVATE mult_mm matrix)

# Create the main granular executable
add_executable(mult_mm_granular_main.out
    mult_mm_granular_main.cc
)
target_link_libraries(mult_mm_granular_main.out PRIVATE mult_mm matrix)

# Create random matrix generator
add_executable(testcase_generator.out
    testcase_generator.cc
)
target_include_directories(testcase_generator.out PUBLIC ../utils)
target_link_libraries(testcase_generator.out PRIVATE matrix)