#include "common/gpuUtils.cuh"
#include "qr/gpuBlockedHouseholder.cuh"
#include "qr/gpuGramSchmidt.cuh"
#include "qr/gpuModifiedGramSchmidt.cuh"
#include "qr/gpuTSQR.cuh"
#include "qr/gpuhouseholder.cuh"
#include <benchmark/benchmark.h>

template <typename T> static void benchmarkGpuTSQR(benchmark::State& state) {
  int M = state.range(0);
  int N = state.range(1);

  // Allocate memory only for A and R (TSQR focuses on R computation)
  T* d_A;
  T* d_R;
  cudaMalloc(&d_A, M * N * sizeof(T));
  cudaMalloc(&d_R, N * N * sizeof(T));

  // Initialize A with random values
  common::gpu::randomFill<<<(M * N + 255) / 256, 256>>>(d_A, M * N);
  cudaDeviceSynchronize();

  // Benchmark loop
  for (auto _ : state) {
    qr::gpu::TSQR(d_A, d_R, M, N);
    cudaDeviceSynchronize();
  }

  cudaFree(d_A);
  cudaFree(d_R);
}

/**
 * @brief Benchmark tall skinny to square matrix performance for TSQR
 */
BENCHMARK_TEMPLATE(benchmarkGpuTSQR, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 512; M <= (1 << 16); M *= 2) {
    for (long N = 2; N <= M / 2; N *= 2) {
      b->Args({M, N});
    }
  }
});

/**
 * @brief Benchmark square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkGpuTSQR, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 128; M <= 4096; M *= 2) {
    b->Args({M, M});
  }
});

template <typename T> static void benchmarkGpuBlockedHouseholder(benchmark::State& state) {
  int M = state.range(0);
  int N = state.range(1);

  // Allocate memory for A, Q, R, and workspace. Check the results: on OOM (large
  // M*N) skip this case gracefully instead of dereferencing a failed pointer,
  // which would trigger a sticky illegal-memory-access that poisons later cases.
  float* d_A = nullptr;
  float* d_Q = nullptr;
  float* d_R = nullptr;
  const size_t szMN = static_cast<size_t>(M) * N * sizeof(float);
  if (cudaMalloc(&d_A, szMN) != cudaSuccess || cudaMalloc(&d_Q, szMN) != cudaSuccess ||
      cudaMalloc(&d_R, static_cast<size_t>(N) * N * sizeof(float)) != cudaSuccess) {
    cudaGetLastError(); // clear the sticky OOM error before returning
    cudaFree(d_A);
    cudaFree(d_Q);
    cudaFree(d_R);
    state.SkipWithError("cudaMalloc failed (out of memory for this size)");
    return;
  }

  // Initialize A with random values (for benchmarking purposes)
  common::gpu::randomFill<<<(M * N + 255) / 256, 256>>>(d_A, M * N);
  cudaDeviceSynchronize();
  qr::gpu::BlockedHouseholderWorkspace<float> d_workspace;
  if (qr::gpu::mallocWorkspace(d_workspace, M, N, 32) != cudaSuccess) {
    cudaGetLastError(); // clear the sticky OOM error before returning
    cudaFree(d_A);
    cudaFree(d_Q);
    cudaFree(d_R);
    state.SkipWithError("workspace allocation failed (out of memory for this size)");
    return;
  }

  for (auto _ : state) {
    qr::gpu::blockedHouseHolder(d_A, d_Q, d_R, M, N, d_workspace);
    cudaDeviceSynchronize();
  }

  qr::gpu::freeWorkspace(d_workspace);
  cudaFree(d_A);
  cudaFree(d_Q);
  cudaFree(d_R);
}

/**
 * @brief Benchmark tall skinny to square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkGpuBlockedHouseholder, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 512; M <= (1 << 16); M *= 2) {
    for (long N = 2; N <= M / 4; N *= 2) {
      b->Args({M, N});
    }
  }
});

/**
 * @brief Benchmark square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkGpuBlockedHouseholder, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 128; M <= 4096; M *= 2) {
    b->Args({M, M});
  }
});

template <typename T> static void benchmarkGpuHouseholder(benchmark::State& state) {
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
    qr::gpu::Householder(d_A, d_Q, d_R, M, N, d_workspace);
    cudaDeviceSynchronize();
  }
}

/**
 * @brief Benchmark tall skinny to square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkGpuHouseholder, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 512; M <= (1 << 16); M *= 2) {
    for (long N = 2; N <= M / 4; N *= 2) {
      b->Args({M, N});
    }
  }
});

/**
 * @brief Benchmark square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkGpuHouseholder, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 128; M <= 4096; M *= 2) {
    b->Args({M, M});
  }
});

template <typename T> static void benchmarkGpuModifiedGramSchmidt(benchmark::State& state) {
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
    qr::gpu::modifiedGramSchmidt(d_A, d_Q, d_R, M, N, d_workspace);
    cudaDeviceSynchronize();
  }
}

/**
 * @brief Benchmark tall skinny to square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkGpuModifiedGramSchmidt, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 512; M <= (1 << 16); M *= 2) {
    for (long N = 2; N <= M / 4; N *= 2) {
      b->Args({M, N});
    }
  }
});

/**
 * @brief Benchmark square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkGpuModifiedGramSchmidt, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 128; M <= 4096; M *= 2) {
    b->Args({M, M});
  }
});

template <typename T> static void benchmarkGpuModifiedGramSchmidtNaive(benchmark::State& state) {
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
    qr::gpu::modifiedGramSchmidtNaive(d_A, d_Q, d_R, M, N, d_workspace);
    cudaDeviceSynchronize();
  }
}

/**
 * @brief Benchmark tall skinny to square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkGpuModifiedGramSchmidtNaive, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 512; M <= (1 << 16); M *= 2) {
    for (long N = 2; N <= M / 4; N *= 2) {
      b->Args({M, N});
    }
  }
});

/**
 * @brief Benchmark square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkGpuModifiedGramSchmidtNaive, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 128; M <= 4096; M *= 2) {
    b->Args({M, M});
  }
});

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
    qr::gpu::GramSchmidt(d_A, d_Q, d_R, M, N, d_workspace);
    cudaDeviceSynchronize();
  }
}

/**
 * @brief Benchmark tall skinny to square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkGpuGramSchmidt, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 512; M <= (1 << 16); M *= 2) {
    for (long N = 2; N <= M / 4; N *= 2) {
      b->Args({M, N});
    }
  }
});

/**
 * @brief Benchmark square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkGpuGramSchmidt, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 128; M <= 4096; M *= 2) {
    b->Args({M, M});
  }
});

template <typename T> static void benchmarkGpuGramSchmidtNaive(benchmark::State& state) {
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
    qr::gpu::GramSchmidtNaive(d_A, d_Q, d_R, M, N, d_workspace);
    cudaDeviceSynchronize();
  }
}

/**
 * @brief Benchmark tall skinny to square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkGpuGramSchmidtNaive, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 512; M <= (1 << 16); M *= 2) {
    for (long N = 2; N <= M / 4; N *= 2) {
      b->Args({M, N});
    }
  }
});

/**
 * @brief Benchmark square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkGpuGramSchmidtNaive, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 128; M <= 4096; M *= 2) {
    b->Args({M, M});
  }
});

BENCHMARK_MAIN();
