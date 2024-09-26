#include "print.h"
#include <iostream>

__global__ void PrintDevice();

__global__ void PrintDevice()
{
    printf("Hello from CUDA kernel!, threadid:%d\n", threadIdx.x);
}

void Print()
{
    int32_t block_num = 1;
    int32_t thread_num = 4;
    PrintDevice<<<block_num, thread_num>>>();
    cudaDeviceSynchronize(); // 等待内核完成
}
