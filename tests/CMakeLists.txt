cmake_minimum_required(VERSION 3.16.3)  # 确保使用支持CUDA的版本
project(tests LANGUAGES CXX CUDA)

# env
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 指定include的dir
include_directories(${INCLUDE_DIR})

# bin
add_executable(test kernels/test.cc)

# link
target_link_libraries(test PRIVATE kernels)