#include "stdio.h"
#include <iostream>
#include <string>

#define TILE_DIM 32
#define BLOCK_SIZE 8
#define MX 1024
#define MY 1024

inline void printDim3(const dim3 &d) {
  std::cout << "dim3: (" << d.x << ", " << d.y << ", " << d.z << ")"
            << std::endl;
}
inline void PrintMatrix(int *matrix, const std::string &name) {
  printf("\n%s:\n", name.c_str());
  for (int r = 0; r < MY; ++r) {
    for (int c = 0; c < MX; ++c) {
      printf("%d, ", *(matrix + r * MX + c));
    }
    printf("\n");
  }
  printf("\n");
}

inline void PrintMatrixT(int *matrix, const std::string &name) {
  printf("\n%s:\n", name.c_str());
  for (int r = 0; r < MX; ++r) {
    for (int c = 0; c < MY; ++c) {
      printf("%d, ", *(matrix + r * MX + c));
    }
    printf("\n");
  }
  printf("\n");
}

// NOTE(panyuchen) : 这里每个 kernel 处理 TILE_DIM x TILE_DIM 的数据大小.
// 一个 kernel 中使用 TILE_DIM x BLOCK_SIZE 个线程处理数据.
__global__ void Transpose(int *odata, int *idata) {
  // 这里不是 blockIdx * blockDim + threadIdx 是因为每一个block内的线程资源
  // 不会按照真实位置进行一一映射，而是按照逻辑层的 TILE_DIM x TILE_DIM
  // 去分配资源

  int c = blockIdx.x * TILE_DIM + threadIdx.x;
  int r = blockIdx.y * TILE_DIM + threadIdx.y;
  // printf("block.y:%d, block.x:%d\n", (int)blockIdx.y, (int)blockIdx.x);
  if (c >= MX || r >= MY) {
    return;
  }
  // 实际矩阵的一大行
  int w = gridDim.x * TILE_DIM;

  // 本 kernel 按照行进行跨步循环.
  // Transpose formula : o[x * w + y + i] = i[(y + i) * w + x].
  // x/y 表示处理的列和行索引
  // i 表示跨步区域的索引
  // w 表示
  // kernel 去处理 TILE_DIM ^ 2 的数据.
  // 因为只有 TILE_DIM x BLOCK_SIZE 的 thread 的资源.
  // Q: 为什么不直接计算出 stride 按照 stride 进行[跨步循环].
  // A: int stride = TILE_DIM / BLOCK_SIZE; 这样写反而麻烦.
  // 因为还需要重新计算每一个 stride 的实际 offset.
  for (int i = 0; i < TILE_DIM; i += BLOCK_SIZE) {
    // y为行 x为列.
    // printf("(%d, %d), w:%d: \n", r + i + 1, c + 1, w);
    odata[c * w + r + i] = idata[(r + i) * w + c];
  }
}

bool CheckTranspose(int *dst, int *src) {
  // PrintMatrixT(dst, "dst");
  // PrintMatrix(src, "src");
  // check
  for (int r = 0; r < MY; ++r) {
    for (int c = 0; c < MX; ++c) {
      if (dst[r * MX + c] != src[c * MY + r]) {
        return false;
      }
    }
  }
  return true;
}

void TransposeTest() {
  printf("Transpose kernel handle matrix shape is [%d, %d]\n", MX, MY);
  size_t data_size = MX * MY * sizeof(int);
  int *d_odata, *d_idata, *h_idata, *h_odata, *check_data;
  h_idata = (int *)malloc(data_size);
  h_odata = (int *)malloc(data_size);
  check_data = (int *)malloc(data_size);
  cudaMalloc(&d_odata, data_size);
  cudaMalloc(&d_idata, data_size);

  for (int r = 0; r < MY; ++r) {
    for (int c = 0; c < MX; ++c) {
      h_idata[r * MX + c] = r * MX + c + 1;
      check_data[c * MY + r] = r * MX + c + 1;
    }
  }

  // PrintMatrix(h_idata, "src_data");
  // PrintMatrixT(check_data, "src_data.T");

  cudaMemcpy(d_idata, h_idata, data_size, cudaMemcpyHostToDevice);
  // Kernel的最小处理单元定义为 TILE_DIM x TILE_DIM (逻辑层定义).
  // 真正对于kernel的处理是使用 TILE_DIM x BLOCK_SIZE 个 threads
  // 处理上面定一个逻辑矩阵. 所以真正的BLOCK_NUM的计算是需要分别按照x/y轴对于
  // max num / TILE_NUM 进行任务切分.
  dim3 threads(TILE_DIM, BLOCK_SIZE, 1);
  dim3 blocks((MX + TILE_DIM - 1) / TILE_DIM, (MY + TILE_DIM - 1) / TILE_DIM,
              1);
  // printDim3(threads);
  // printDim3(blocks);
  Transpose<<<blocks, threads>>>(d_odata, d_idata);

  cudaMemcpy(h_odata, d_odata, data_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  CheckTranspose(h_odata, h_idata) ? printf("Transpose success.\n")
                                   : printf("Transpose error.\n");
  free(h_idata);
  free(h_odata);
  cudaFree(d_idata);
  cudaFree(d_odata);
}