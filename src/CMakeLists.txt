cmake_minimum_required(VERSION 3.16.3)  # 确保使用支持CUDA的版本
project(kernels LANGUAGES CXX CUDA)

# env
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 指定include的dir
include_directories(${INCLUDE_DIR})

# print
add_library(kernels STATIC kernels/print.cu kernels/simple_matmul.cu)
