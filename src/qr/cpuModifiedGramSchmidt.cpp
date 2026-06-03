#include "qr/cpuModifiedGramSchmidt.hpp"
#include "common/cpuBuffer.hpp"
#include "common/cpuLinalg.hpp"
#include <cstddef>

namespace qr::cpu {

template <typename T> void modifiedGramSchmidt(const T* A, T* Q, T* R, size_t M, size_t N) {
  assert(M >= N);
  gsComputeQ(A, Q, M, N);
  gsComputeR(A, Q, R, M, N);
}

template void modifiedGramSchmidt<float>(const float*, float*, float*, size_t, size_t);
template void modifiedGramSchmidt<double>(const double*, double*, double*, size_t, size_t);

} // namespace qr::cpu

namespace {

template <typename T> void gsComputeQ(const T* A, T* Q, size_t M, size_t N) {
  T* tmp = common::cpu::allocBuffer<T>(M);
  common::cpu::copyBuffer(Q, A, M * N);
  for (size_t j = 0; j < N; ++j) {
    T* qj = Q + j * M;
    common::cpu::normalize(qj, qj, M);
    for (size_t i = j + 1; i < N; ++i) {
      T* qi = Q + i * M;
      common::cpu::project(qi, qj, tmp, M);
      common::cpu::sub(qi, tmp, qi, M);
    }
  }
  common::cpu::freeBuffer(tmp);
}

template <typename T> void gsComputeR(const T* A, const T* Q, T* R, size_t M, size_t N) {
  common::cpu::zeroBuffer(R, N * N);
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i <= j; ++i) {
      R[i + j * N] = common::cpu::dot(Q + i * M, A + j * M, M);
    }
  }
}

} // anonymous namespace
