#include "common/gpuLinalg.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

__global__ void testReduceSum(float* data, float* result, uint32_t N) {
  __shared__ float sdata[1024];
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[threadIdx.x] = (idx < N) ? data[idx] : 0.0f;
  __syncthreads();

  common::gpu::reduceSum<float, 1024>(sdata, &result[blockIdx.x], N);
}

TEST(GpuLinalgTest, ReduceSum) {
  const uint32_t N = 1024;
  float* d_data;
  float* d_result;
  float h_result;

  cudaMalloc(&d_data, N * sizeof(float));
  cudaMalloc(&d_result, sizeof(float));

  // Initialize data on the device
  float h_data[N];
  for (uint32_t i = 0; i < N; i++) {
    h_data[i] = static_cast<float>(i + 1); // Fill with values from 1 to N
  }
  cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel to test reduceSum
  testReduceSum<<<1, 1024>>>(d_data, d_result, N);
  cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

  // Calculate expected result on the host
  float expected = (N * (N + 1)) / 2.0f; // Sum of first N natural numbers

  EXPECT_FLOAT_EQ(h_result, expected);

  cudaFree(d_data);
  cudaFree(d_result);
}

__global__ void testDotProduct(float* a, float* b, float* result, uint32_t N) {
  common::gpu::dotProduct<float, 1024>(a, b, result, N);
}

TEST(GpuLinalgTest, DotProduct) {
  const uint32_t N = 1024;
  float* d_a;
  float* d_b;
  float* d_result;
  float h_result;

  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_b, N * sizeof(float));
  cudaMalloc(&d_result, sizeof(float));

  // Initialize data on the device
  float h_a[N];
  float h_b[N];
  for (uint32_t i = 0; i < N; i++) {
    h_a[i] = static_cast<float>(i + 1); // Fill with values from 1 to N
    h_b[i] = static_cast<float>(i + 1); // Fill with values from 1 to N
  }
  cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel to test dotProduct
  testDotProduct<<<1, 1024>>>(d_a, d_b, d_result, N);
  cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

  // Calculate expected result on the host
  float expected = (N * (N + 1) * (2 * N + 1)) / 6.0f; // Sum of squares of first N natural numbers

  EXPECT_FLOAT_EQ(h_result, expected);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
}

__global__ void testProject(float* a, float* b, float* result, uint32_t N) {
  common::gpu::project<float, 1024>(a, b, result, N);
}

TEST(GpuLinalgTest, Project) {
  const uint32_t N = 1024;
  float* d_a;
  float* d_b;
  float* d_result;
  float h_result[N];

  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_b, N * sizeof(float));
  cudaMalloc(&d_result, N * sizeof(float));

  // Initialize data on the device
  float h_a[N];
  float h_b[N];
  for (uint32_t i = 0; i < N; i++) {
    h_a[i] = static_cast<float>(i + 1); // Fill with values from 1 to N
    h_b[i] = static_cast<float>(i + 1); // Fill with values from 1 to N
  }
  cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel to test project
  testProject<<<1, 1024>>>(d_a, d_b, d_result, N);
  cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Calculate expected result on the host
  float dot_ab = (N * (N + 1) * (2 * N + 1)) / 6.0f; // Sum of squares of first N natural numbers
  float dot_bb = dot_ab;                             // Since a and b are the same
  float factor = dot_ab / dot_bb;

  for (uint32_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(h_result[i], h_b[i] * factor);
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
}
