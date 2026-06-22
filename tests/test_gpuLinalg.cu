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

TEST(GpuLinalgTest, MatMul) {
  const uint32_t M = 64;
  const uint32_t K = 64;
  const uint32_t N = 16;
  float h_A[M * K], h_B[K * N];

  for (uint32_t i = 0; i < M * K; i++)
    h_A[i] = 0.0f;

  for (uint32_t i = 0; i < M; i++)
    h_A[i + i * K] = 1.0f;

  for (uint32_t i = 0; i < K * N; i++)
    h_B[i] = static_cast<float>(i + 1);

  float* d_A;
  float* d_B;
  float* d_C;
  CUDACHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
  CUDACHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  common::gpu::matMul(d_A, d_B, d_C, M, K, N);
  CUDACHECK(cudaGetLastError());

  float h_C[M * N];
  CUDACHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  for (uint32_t i = 0; i < M; i++) {
    for (uint32_t j = 0; j < N; j++) {
      float expected = h_B[i + j * K];
      EXPECT_FLOAT_EQ(h_C[i + j * M], expected);
    }
  }
}
