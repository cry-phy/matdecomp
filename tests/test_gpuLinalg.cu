#include "common/cpuLinalg.hpp"
#include "common/cpuUtils.hpp"
#include "common/gpuLinalg.cuh"
#include "common/gpuUtils.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

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
  const uint32_t M = 2024;
  const uint32_t K = 2024;
  const uint32_t N = 128;
  // Heap, not stack: h_A alone is ~16 MB and would overflow the 8 MB stack.
  std::vector<float> h_A(static_cast<size_t>(M) * K, 0.0f);
  std::vector<float> h_B(static_cast<size_t>(K) * N);

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
  CUDACHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

  common::gpu::matMul(d_A, d_B, d_C, M, K, N, M, K, M);
  CUDACHECK(cudaGetLastError());

  std::vector<float> h_C(static_cast<size_t>(M) * N);
  CUDACHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  for (uint32_t i = 0; i < M; i++) {
    for (uint32_t j = 0; j < N; j++) {
      float expected = h_B[i + j * K];
      EXPECT_FLOAT_EQ(h_C[i + j * M], expected);
    }
  }
}

TEST(GpuLinalgTest, MatMulTransposedMan) {
  const uint32_t M = 1024;
  const uint32_t K = 1024;
  const uint32_t N = 1024;

  // Heap, not stack: seven 1024x1024 float buffers are ~28 MB total, far past
  // the 8 MB stack limit.
  std::vector<float> h_A(static_cast<size_t>(M) * K, 0.0f);
  std::vector<float> h_B(static_cast<size_t>(K) * N, 0.0f);
  std::vector<float> h_C(static_cast<size_t>(M) * N, 0.0f);
  std::vector<float> h_C_refnn(static_cast<size_t>(M) * N, 0.0f);
  std::vector<float> h_C_reftt(static_cast<size_t>(M) * N, 0.0f);
  std::vector<float> h_C_reftn(static_cast<size_t>(M) * N, 0.0f);
  std::vector<float> h_C_refnt(static_cast<size_t>(M) * N, 0.0f);

  for (uint32_t i = 0; i < K; i++) {
    h_A[i * M + i] = 1.0f;
    h_B[i * K + i] = 1.0f;
  }

  for (uint32_t i = 1; i < K - 1; i++) {
    h_A[(i - 1) * K + i] = -1.0f;
    h_B[i * M + i - 1] = -1.0f;
  }
  h_A[(K - 2) * K + K - 1] = -1.0f;
  h_B[M * (K - 1) + K - 2] = -1.0f;

  float *d_A, *d_B, *d_C;
  CUDACHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

  CUDACHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

  common::gpu::matMul(d_A, d_B, d_C, M, K, N, M, K, M);
  CUDACHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  h_C_refnn[0] = 1.0f;
  for (uint32_t i = 1; i < K; i++)
    h_C_refnn[M * i + i] = 2.0f;

  for (uint32_t i = 1; i < K - 1; i++) {
    h_C_refnn[i * M + i - 1] = -1.0f;
    h_C_refnn[(i - 1) * M + i] = -1.0f;
  }
  h_C_refnn[M * (K - 2) + K - 1] = -1.0f;
  h_C_refnn[M * (K - 1) + K - 2] = -1.0f;

  for (uint32_t i = 0; i < M; i++) {
    for (uint32_t j = 0; j < N; j++) {
      EXPECT_FLOAT_EQ(h_C_refnn[i + j * M], h_C[i + j * M]);
    }
  }

  common::gpu::matMul<float, true, true>(d_A, d_B, d_C, M, K, N, M, K, M);
  CUDACHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  for (uint32_t i = 0; i < K - 1; i++)
    h_C_reftt[M * i + i] = 2.0f;
  h_C_reftt[M * (K - 1) + K - 1] = 1.0f;

  for (uint32_t i = 1; i < K - 1; i++) {
    h_C_reftt[i * M + i - 1] = -1.0f;
    h_C_reftt[(i - 1) * M + i] = -1.0f;
  }
  h_C_reftt[M * (K - 2) + K - 1] = -1.0f;
  h_C_reftt[M * (K - 1) + K - 2] = -1.0f;

  for (uint32_t i = 0; i < M; i++) {
    for (uint32_t j = 0; j < N; j++) {
      EXPECT_FLOAT_EQ(h_C_reftt[i + j * M], h_C[i + j * M]);
    }
  }

  common::gpu::matMul<float, true, false>(d_A, d_B, d_C, M, K, N, M, K, M);
  CUDACHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  for (uint32_t i = 0; i < K; i++)
    h_C_reftn[M * i + i] = 1.0f;

  for (uint32_t i = 1; i < K; i++)
    h_C_reftn[i * M + i - 1] = -2.0f;

  for (uint32_t i = 2; i < K; i++)
    h_C_reftn[i * M + i - 2] = 1.0f;

  for (uint32_t i = 0; i < M; i++) {
    for (uint32_t j = 0; j < N; j++) {
      EXPECT_FLOAT_EQ(h_C_reftn[i + j * M], h_C[i + j * M]);
    }
  }

  common::gpu::matMul<float, false, true>(d_A, d_B, d_C, M, K, N, M, K, M);
  CUDACHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  for (uint32_t i = 0; i < K; i++)
    h_C_refnt[M * i + i] = 1.0f;

  for (uint32_t i = 1; i < K; i++)
    h_C_refnt[(i - 1) * M + i] = -2.0f;

  for (uint32_t i = 2; i < K; i++)
    h_C_refnt[(i - 2) * M + i] = 1.0f;

  for (uint32_t i = 0; i < M; i++) {
    for (uint32_t j = 0; j < N; j++) {
      EXPECT_FLOAT_EQ(h_C_refnt[i + j * M], h_C[i + j * M]);
    }
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

TEST(GpuLinalgTest, MatMulTransposed) {
  const uint32_t M = 40;
  const uint32_t K = 24;
  const uint32_t N = 16;

  float opA[M * K], opB[K * N];
  common::cpu::fillRandom(opA, M * K);
  common::cpu::fillRandom(opB, K * N);

  float opAT[K * M], opBT[N * K];
  common::cpu::transpose(opA, opAT, M, K); // K x M (ld = K)
  common::cpu::transpose(opB, opBT, K, N); // N x K (ld = N)

  float ref[M * N];
  common::cpu::matMul(opA, M, K, opB, K, N, ref);

  float *d_A, *d_AT, *d_B, *d_BT, *d_C;
  CUDACHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_AT, K * M * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_BT, N * K * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
  CUDACHECK(cudaMemcpy(d_A, opA, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_AT, opAT, K * M * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_B, opB, K * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_BT, opBT, N * K * sizeof(float), cudaMemcpyHostToDevice));

  float h_C[M * N];
  auto expectRef = [&](const char* combo) {
    CUDACHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < M * N; ++i)
      EXPECT_NEAR(h_C[i], ref[i], 1e-3f) << combo << " mismatch at " << i;
  };

  common::gpu::matMul<float, false, false>(d_A, d_B, d_C, M, K, N, M, K, M);
  expectRef("nn");
  common::gpu::matMul<float, true, false>(d_AT, d_B, d_C, M, K, N, K, K, M);
  expectRef("tn");
  common::gpu::matMul<float, false, true>(d_A, d_BT, d_C, M, K, N, M, N, M);
  expectRef("nt");
  common::gpu::matMul<float, true, true>(d_AT, d_BT, d_C, M, K, N, K, N, M);
  expectRef("tt");

  cudaFree(d_A);
  cudaFree(d_AT);
  cudaFree(d_B);
  cudaFree(d_BT);
  cudaFree(d_C);
}

TEST(GpuLinalgTest, matVecMult) {
  const uint32_t M = 2024;
  const uint32_t N = 128;

  float h_A[M * N], h_x[N], h_y[M], h_y_ref[M];
  common::cpu::fillRandom(h_A, M * N);
  common::cpu::fillRandom(h_x, N);

  float *d_A, *d_x, *d_y;
  CUDACHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_x, N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_y, M * sizeof(float)));
  CUDACHECK(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_y, 0, M * sizeof(float)));

  common::gpu::matVecMult(d_A, d_x, d_y, M, N, M);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));

  common::cpu::matVecMul(h_A, h_x, h_y_ref, M, N);

  for (uint32_t i = 0; i < M; ++i)
    EXPECT_NEAR(h_y[i], h_y_ref[i], 1e-3f) << " mismatch at index " << i;
}
