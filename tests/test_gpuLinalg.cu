#include "common/gpuLinalg.cuh"
#include "common/gpuUtils.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

// Thin wrapper to expose _blockReduce (a __device__ fn) to a test kernel.
__global__ void testBlockReduce(float* data, float* result, uint32_t N) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  float val = (idx < N) ? data[idx] : 0.0f;
  val = common::gpu::_blockReduce<float>(val);
  if (threadIdx.x == 0)
    atomicAdd(result, val);
}

TEST(GpuLinalgTest, BlockReduce) {
  const uint32_t N = 1024;
  float h_data[N];
  for (uint32_t i = 0; i < N; i++)
    h_data[i] = static_cast<float>(i + 1);

  float* d_data;
  float* d_result;
  CUDACHECK(cudaMalloc(&d_data, N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_result, sizeof(float)));
  CUDACHECK(cudaMemset(d_result, 0, sizeof(float)));
  CUDACHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

  testBlockReduce<<<1, 1024>>>(d_data, d_result, N);
  CUDACHECK(cudaGetLastError());

  float h_result;
  CUDACHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

  float expected = (N * (N + 1)) / 2.0f;
  EXPECT_FLOAT_EQ(h_result, expected);

  cudaFree(d_data);
  cudaFree(d_result);
}

TEST(GpuLinalgTest, DotProduct) {
  const uint32_t N = 1024;
  float h_a[N], h_b[N];
  for (uint32_t i = 0; i < N; i++)
    h_a[i] = h_b[i] = static_cast<float>(i + 1);

  float* d_a;
  float* d_b;
  float* d_result;
  CUDACHECK(cudaMalloc(&d_a, N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_b, N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_result, sizeof(float)));
  CUDACHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

  common::gpu::dotProduct(d_a, d_b, d_result, N);
  CUDACHECK(cudaGetLastError());

  float h_result;
  CUDACHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

  float expected = (N * (N + 1) * (2 * N + 1)) / 6.0f;
  EXPECT_FLOAT_EQ(h_result, expected);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
}

TEST(GpuLinalgTest, Project) {
  const uint32_t N = 1024;
  float h_a[N], h_b[N];
  for (uint32_t i = 0; i < N; i++)
    h_a[i] = h_b[i] = static_cast<float>(i + 1);

  float* d_a;
  float* d_b;
  float* d_result;
  float* d_dot_ab;
  float* d_dot_bb;
  CUDACHECK(cudaMalloc(&d_a, N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_b, N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_result, N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_dot_ab, sizeof(float)));
  CUDACHECK(cudaMalloc(&d_dot_bb, sizeof(float)));
  CUDACHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

  // Normalize b as project does not normalize b internally.
  common::gpu::normalize(d_b, d_b, N, d_dot_bb);
  common::gpu::project(d_a, d_b, d_result, N, d_dot_ab);
  CUDACHECK(cudaGetLastError());

  float h_result[N];
  CUDACHECK(cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost));

  // a == b so dot_ab/dot_bb == 1, projection == b
  for (uint32_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(h_result[i], h_b[i]);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
  cudaFree(d_dot_ab);
  cudaFree(d_dot_bb);
}
