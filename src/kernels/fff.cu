#include <cstdio>
#include <cuda_runtime.h>

__global__ void matrixMultiply(float *A, float *B, float *C)
{
    int row = blockIdx.x; // 确定当前行
    if (row < 2)
    {
        C[row] = A[row * 2] * B[0] + A[row * 2 + 1] * B[1]; // 计算矩阵乘法
    }
}

int main()
{
    // 定义输入矩阵 A 和 B
    float h_A[4] = {1.0f, 2.0f, 3.0f, 4.0f}; // 2x2 矩阵
    float h_B[2] = {5.0f, 6.0f};             // 2x1 矩阵
    float h_C[2];                            // 2x1 结果矩阵

    float *d_A, *d_B, *d_C;

    // 分配设备内存
    cudaMalloc((void **)&d_A, 4 * sizeof(float));
    cudaMalloc((void **)&d_B, 2 * sizeof(float));
    cudaMalloc((void **)&d_C, 2 * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 2 * sizeof(float), cudaMemcpyHostToDevice);

    // 启动内核，处理 2 行
    matrixMultiply<<<2, 1>>>(d_A, d_B, d_C);

    // 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < 2; i++)
    {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
