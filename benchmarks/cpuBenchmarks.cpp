#include "common/cpuBuffer.hpp"
#include "common/cpuUtils.hpp"
#include "qr/cpuGramSchmidt.hpp"
#include <benchmark/benchmark.h>

template <typename T> static void benchmarkCpuGramSchmidt(benchmark::State& state) {
  size_t M = state.range(0);
  size_t N = state.range(1);
  T* A = common::cpu::allocBuffer<T>(M * N);
  T* Q = common::cpu::allocBuffer<T>(M * N);
  T* R = common::cpu::allocBuffer<T>(N * N);

  common::cpu::fillRandom(A, M * N);

  for (auto _ : state) {
    qr::cpu::gramSchmidt(A, Q, R, M, N);
    benchmark::DoNotOptimize(Q);
    benchmark::DoNotOptimize(R);
  }

  common::cpu::freeBuffer(A);
  common::cpu::freeBuffer(Q);
  common::cpu::freeBuffer(R);
}

/**
 * @brief Benchmark tall skinny to square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkCpuGramSchmidt, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 512; M <= 4096; M *= 2) {
    for (long N = 2; N <= M / 2; N *= 2) {
      b->Args({M, N});
    }
  }
});

/**
 * @brief Benchmark square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkCpuGramSchmidt, float)->Apply([](benchmark::Benchmark* b) {
  for (long M = 128; M <= 4096; M *= 2) {
    b->Args({M, M});
  }
});

BENCHMARK_MAIN();
