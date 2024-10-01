#include "stdio.h"

#define TILE_DIM 32
#define BLOCK_SIZE 8
// NOTE(panyuchen) : 这里每个 kernel 处理 TILE_DIM x TILE_DIM 的数据大小
// 一个 kernel 中使用 BLOCK_SIZE 个线程处理数据
__global__ void Transpose(float *odata, float *idata) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
}

void TransposeTest() {}