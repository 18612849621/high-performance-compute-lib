cmake_minimum_required(VERSION 3.16.3)
project(high-performance-compute-lib LANGUAGES CXX CUDA)

# env
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/src"
                "/usr/local/cuda-11.8/include")

# compile dirs
add_subdirectory(src)
add_subdirectory(tests)
