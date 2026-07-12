#include "common/gpuUtils.cuh"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <chrono>
#include <cusolverDn.h>
#include <iostream>

// Extract the N x N upper-triangular R from the M x N factored matrix (column-major).
// Must run before orgqr overwrites the factored matrix with Q.
__global__ void extractUpperR(const float* A, float* R, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * N) {
    int i = idx % N;
    int j = idx / N;
    R[i + static_cast<int64_t>(j) * N] =
        (i <= j) ? A[i + static_cast<int64_t>(j) * M] : 0.0f;
  }
}

// Full QR to match our implementations: factorize (geqrf), extract R, and form the
// explicit Q (orgqr). cuSOLVER's geqrf alone only leaves R + packed reflectors, so
// timing geqrf on its own would understate the work our algorithms actually do.
static void benchmarkCusolver(benchmark::State& state) {
  int M = state.range(0);
  int N = state.range(1);

  float *d_A, *d_R, *d_work, *d_tau;
  int* d_info;
  CUDACHECK(cudaMalloc(&d_A, static_cast<size_t>(M) * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_R, static_cast<size_t>(N) * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_tau, static_cast<size_t>(N) * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_info, sizeof(int)));

  cusolverDnHandle_t cusolverHandle;
  if (cusolverDnCreate(&cusolverHandle) != CUSOLVER_STATUS_SUCCESS) {
    state.SkipWithError("cusolverDnCreate failed");
    return;
  }

  // Shared workspace sized for the larger of the two factorization steps.
  int lworkGeqrf = 0;
  int lworkOrgqr = 0;
  if (cusolverDnSgeqrf_bufferSize(cusolverHandle, M, N, d_A, M, &lworkGeqrf) !=
          CUSOLVER_STATUS_SUCCESS ||
      cusolverDnSorgqr_bufferSize(cusolverHandle, M, N, N, d_A, M, d_tau, &lworkOrgqr) !=
          CUSOLVER_STATUS_SUCCESS) {
    state.SkipWithError("cusolver bufferSize failed");
    cusolverDnDestroy(cusolverHandle);
    return;
  }
  int lwork = std::max(lworkGeqrf, lworkOrgqr);
  CUDACHECK(cudaMalloc(&d_work, static_cast<size_t>(lwork) * sizeof(float)));

  uint32_t blocksNN = (static_cast<uint32_t>(N) * N + 255) / 256;

  for (auto _ : state) {
    // Re-fill A each iteration: geqrf/orgqr overwrite it in-place.
    common::gpu::randomFill<<<(M * N + 255) / 256, 256>>>(d_A, M * N);
    CUDACHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    // 1. Factorize: A <- R (upper triangle) + Householder reflectors (below), tau.
    cusolverDnSgeqrf(cusolverHandle, M, N, d_A, M, d_tau, d_work, lwork, d_info);
    // 2. Extract R before orgqr overwrites A.
    extractUpperR<<<blocksNN, 256>>>(d_A, d_R, M, N);
    // 3. Form the explicit Q (M x N) in place in A.
    cusolverDnSorgqr(cusolverHandle, M, N, N, d_A, M, d_tau, d_work, lwork, d_info);
    CUDACHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());

    int h_info = 0;
    CUDACHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
      state.SkipWithError("cusolver geqrf/orgqr reported failure");
      break;
    }
  }

  cusolverDnDestroy(cusolverHandle);
  cudaFree(d_A);
  cudaFree(d_R);
  cudaFree(d_work);
  cudaFree(d_tau);
  cudaFree(d_info);
}

BENCHMARK(benchmarkCusolver)
    ->Apply([](benchmark::internal::Benchmark* b) {
      for (long M = 512; M <= (1 << 16); M *= 2)
        for (long N = 2; N <= M / 8; N *= 2)
          b->Args({M, N});
    })
    ->UseManualTime();

BENCHMARK(benchmarkCusolver)
    ->Apply([](benchmark::internal::Benchmark* b) {
      for (long M = 128; M <= 4096; M *= 2)
        b->Args({M, M});
    })
    ->UseManualTime();

BENCHMARK_MAIN();
