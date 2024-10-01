#include "stdio.h"

#define TILE_DIM 32
#define BLOCK_SIZE 8
#define mx 1024
#define my 1024

// NOTE(panyuchen) : 这里每个 kernel 处理 TILE_DIM x TILE_DIM 的数据大小
// 一个 kernel 中使用 BLOCK_SIZE 个线程处理数据
__global__ void Transpose(float *odata, float *idata) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
}

void TransposeTest() {
  printf("Expect handle matrix shape is [%d, %d]\n", mx, my);
  // Kernel的最小处理单元定义为 TILE_DIM x TILE_DIM (逻辑层定义).
  // 真正对于kernel的处理是使用 TILE_DIM x BLOCK_SIZE 个 threads 处理上面定一个逻辑矩阵.
  // 所以真正的BLOCK_NUM的计算是需要分别按照x/y轴对于 max num / TILE_NUM 进行任务切分.
  dim3 threads(TILE_DIM, BLOCK_SIZE, 1);
  dim3 blocks((mx + TILE_DIM - 1) / TILE_DIM, (my + TILE_DIM - 1) / TILE_DIM, 1);
}