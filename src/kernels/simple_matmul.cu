#include "simple_matmul.h"
#include <cuda_runtime.h>
#include <iostream>

__device__ void DisplayThreadInfo(int eid, int bid, int tid, int col_num, float *A, float *B)
{
    printf("[execute_%d, block_%d, thread_%d]: %f %f %f %f\n", eid, bid, tid, A[eid * col_num], B[0], A[eid * col_num + 1], B[1]);
}

__global__ void SimpleMatmul(float *A, float *B, float *C)
{
    // 用两个线程去并行计算两行的结果 所以线程号就等于是行号
    int execute_id = blockIdx.x * blockDim.x + threadIdx.x;
    int col_num = 2;
    DisplayThreadInfo(execute_id, blockIdx.x, threadIdx.x, col_num, A, B);
    C[execute_id] = A[col_num * execute_id] * B[0] + A[col_num * execute_id + 1] * B[1];
}

void SimpleMatmulTest()
{
    printf("SIMPLEMATMUL\n");
    // 测试行存数据的二维矩阵相乘
    // 定义矩阵 A B 结果为 C
    // [1 2]   [5]    [17]
    // [3 4]   [6]    [39]
    float h_A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_B[2] = {5.0f, 6.0f};
    float h_C[2];
    // device ptr
    float *d_A, *d_B, *d_C;
    // device mem apply
    cudaMalloc((void **)&d_A, 4 * sizeof(float));
    cudaMalloc((void **)&d_B, 2 * sizeof(float));
    cudaMalloc((void **)&d_C, 2 * sizeof(float));
    // host -> device
    cudaMemcpy(d_A, h_A, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 2 * sizeof(float), cudaMemcpyHostToDevice);
    // kernel launch
    dim3 block_num(1, 1, 1);
    dim3 thread_num(2, 1, 1);
    SimpleMatmul<<<block_num, thread_num>>>(d_A, d_B, d_C);
    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    // 所有kernel强制同步
    cudaDeviceSynchronize();
    // device -> host
    cudaMemcpy(h_C, d_C, 2 * sizeof(float), cudaMemcpyDeviceToHost); // display result
    for (int i = 0; i < 2; ++i)
    {
        printf("%f \n", h_C[i]);
    }
    // free
    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(h_C);
}