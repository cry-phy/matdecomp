#include "common/gpuUtils.cuh"
#include <benchmark/benchmark.h>
#include <chrono>
#include <cusolverDn.h>
#include <iostream>

static void benchmarkCusolver(benchmark::State& state) {
  int M = state.range(0);
  int N = state.range(1);

  float *d_A, *d_Q, *d_R, *d_work, *d_tau;
  int* d_info;
  CUDACHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_Q, M * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_R, N * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_tau, N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_info, sizeof(int)));

  cusolverDnHandle_t cusolverHandle;
  if (cusolverDnCreate(&cusolverHandle) != CUSOLVER_STATUS_SUCCESS) {
    state.SkipWithError("cusolverDnCreate failed");
    return;
  }

  int workspaceSize = 0;
  if (cusolverDnSgeqrf_bufferSize(cusolverHandle, M, N, d_A, M, &workspaceSize) !=
      CUSOLVER_STATUS_SUCCESS) {
    state.SkipWithError("cusolverDnSgeqrf_bufferSize failed");
    cusolverDnDestroy(cusolverHandle);
    return;
  }
  CUDACHECK(cudaMalloc(&d_work, workspaceSize * sizeof(float)));

  for (auto _ : state) {
    // Re-fill A each iteration: cusolverDnSgeqrf overwrites it in-place.
    common::gpu::randomFill<<<(M * N + 255) / 256, 256>>>(d_A, M * N);
    CUDACHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    cusolverDnSgeqrf(cusolverHandle, M, N, d_A, M, d_tau, d_work, workspaceSize, d_info);
    CUDACHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());

    int h_info = 0;
    CUDACHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
      state.SkipWithError("cusolverDnSgeqrf reported failure");
      break;
    }
  }

  cusolverDnDestroy(cusolverHandle);
  cudaFree(d_A);
  cudaFree(d_Q);
  cudaFree(d_R);
  cudaFree(d_work);
  cudaFree(d_tau);
  cudaFree(d_info);
}

BENCHMARK(benchmarkCusolver)
    ->Apply([](benchmark::Benchmark* b) {
      for (long M = 512; M <= (1 << 16); M *= 2)
        for (long N = 2; N <= M / 8; N *= 2)
          b->Args({M, N});
    })
    ->UseManualTime();

BENCHMARK(benchmarkCusolver)
    ->Apply([](benchmark::Benchmark* b) {
      for (long M = 128; M <= 4096; M *= 2)
        b->Args({M, M});
    })
    ->UseManualTime();

BENCHMARK_MAIN();
