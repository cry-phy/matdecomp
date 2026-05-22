#include "common/cpuBuffer.hpp"
#include "common/utils.hpp"
#include "qr/cpuGramSchmidt.hpp"
#include <benchmark/benchmark.h>

template <typename T>
static void benchmarkCpuGramSchmidt(benchmark::State &state) {
  size_t M = state.range(0);
  size_t N = state.range(1);
  T *A = common::allocBuffer<T>(M * N);
  T *Q = common::allocBuffer<T>(M * N);
  T *R = common::allocBuffer<T>(N * N);

  common::fillRandom(A, M * N);

  for (auto _ : state) {
    qr::gramSchmidt(A, Q, R, M, N);
    benchmark::DoNotOptimize(Q);
    benchmark::DoNotOptimize(R);
  }

  common::freeBuffer(A);
  common::freeBuffer(Q);
  common::freeBuffer(R);
}

/**
 * @brief Benchmark square matrix performance
 */
BENCHMARK_TEMPLATE(benchmarkCpuGramSchmidt, float)
    ->Apply([](benchmark::Benchmark *b) {
      for (long M = 128; M <= 4096; M *= 2) {
        b->Args({M, M});
      }
    });

BENCHMARK_MAIN();
