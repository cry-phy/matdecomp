#include "common/gpuUtils.cuh"
#include "qr/gpuGramSchmidt.cuh"
#include <benchmark/benchmark.h>

template <typename T>
__global__ __launch_bounds__(MAX_BLOCK_SIZE) void executeGramSchmidt(T* A, T* Q, T* R, int M,
                                                                      int N, T* workspace) {
  qr::gpuGramSchmidt(A, Q, R, M, N, workspace);
}

template <typename T> static void benchmarkGpuGramSchmidt(benchmark::State& state) {
  int M = state.range(0);
  int N = state.range(1);

  // Allocate memory for A, Q, R, and workspace
  float* d_A;
  float* d_Q;
  float* d_R;
  float* d_workspace;
  cudaMalloc(&d_A, M * N * sizeof(float));
  cudaMalloc(&d_Q, M * N * sizeof(float));
  cudaMalloc(&d_R, N * N * sizeof(float));
  cudaMalloc(&d_workspace, M * sizeof(float));

  // Initialize A with random values (for benchmarking purposes)
  common::gpu::randomFill<<<(M * N + 255) / 256, 256>>>(d_A, M * N);
  cudaDeviceSynchronize();

  for (auto _ : state) {
    executeGramSchmidt<<<1, MAX_BLOCK_SIZE>>>(d_A, d_Q, d_R, M, N, d_workspace);
    cudaDeviceSynchronize();
  }
}

/**
 * @brief Benchmark square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkGpuGramSchmidt, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 128; M <= 4096; M *= 2) {
    b->Args({M, M});
  }
});

BENCHMARK_MAIN();
