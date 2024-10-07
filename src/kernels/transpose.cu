#include "stdio.h"
#include <iostream>
#include <string>

#define TILE_DIM 32
#define BLOCK_SIZE 8
#define MX 1024
#define MY 1024

inline void PrintDim3(const dim3 &d) {
  std::cout << "dim3: (" << d.x << ", " << d.y << ", " << d.z << ")"
            << std::endl;
}

inline void PrintComputeInfo(const float &ms) {
  printf("%25s%25s\n", "Routine", "Bandwidth(GB/s)");
  printf("%25s", "native transpose");
  printf("%20.2f\n", 2 * MX * MY * sizeof(int) * 1e-6 * 100 / ms);
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

__global__ void TransposeV2(int *odata, int *idata) {
  // 通过使用共享内存提升kernel的带宽能力.
  // + 1 解决存储块冲突
  __shared__ float shared_memory[TILE_DIM][TILE_DIM + 1];
  int c = blockIdx.x * TILE_DIM + threadIdx.x;
  int r = blockIdx.y * TILE_DIM + threadIdx.y;
  if (c >= MX || r >= MY) {
    return;
  }
  int w = gridDim.x * TILE_DIM;
  for (int i = 0; i < TILE_DIM; i += BLOCK_SIZE) {
    // 每个bank可以同时被一个线程访问。因此，多个线程可以并行访问不同的银行，
    // 但如果多个线程同时访问同一个银行，就会发生bank
    // 冲突，导致访问延迟
    // link: https://blog.csdn.net/dataowner/article/details/123537966
    // 这里说白了不是对应的访问 32一组 就无法 1 clock 的性能模式完成处理
    // 所以一定注意行访问的方式才可以高并发，同时列访问没用
    shared_memory[threadIdx.y + i][threadIdx.x] = idata[(r + i) * w + c];
  }
  // 同步操作符号保证数据是最新需要的
  __syncthreads();
  // 这里仅仅是行列交换，所使用的shard mem的行列是没有变的
  c = blockIdx.y * TILE_DIM + threadIdx.x;
  r = blockIdx.x * TILE_DIM + threadIdx.y;
  for (int i = 0; i < TILE_DIM; i += BLOCK_SIZE) {
    // NOTE(panyuchen) : 在shared mem中完成了转置操作，可以大幅度增加kernels吞吐
    odata[(r + i) * w + c] = shared_memory[threadIdx.x][threadIdx.y + i];
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
  // PrintDim3(threads);
  // PrintDim3(blocks);

  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  cudaEventRecord(startEvent, 0);
  TransposeV2<<<blocks, threads>>>(d_odata, d_idata);
  // Transpose<<<blocks, threads>>>(d_odata, d_idata);
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  float ms;
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  PrintComputeInfo(ms);

  cudaMemcpy(h_odata, d_odata, data_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  CheckTranspose(h_odata, h_idata) ? printf("Transpose success.\n")
                                   : printf("Transpose error.\n");
  free(h_idata);
  free(h_odata);
  cudaFree(d_idata);
  cudaFree(d_odata);
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
}