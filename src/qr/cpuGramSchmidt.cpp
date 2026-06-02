#include "qr/cpuGramSchmidt.hpp"
#include "common/cpuBuffer.hpp"
#include "common/cpuLinalg.hpp"
#include <cstddef>

namespace qr::cpu {

template <typename T> void gramSchmidt(const T* A, T* Q, T* R, size_t M, size_t N) {
  assert(M >= N);
  gsComputeQ(A, Q, M, N);
  gsComputeR(A, Q, R, M, N);
}

template void gramSchmidt<float>(const float*, float*, float*, size_t, size_t);
template void gramSchmidt<double>(const double*, double*, double*, size_t, size_t);

} // namespace qr::cpu

namespace {

template <typename T> void gsComputeQ(const T* A, T* Q, size_t M, size_t N) {
  T* tmp = common::cpu::allocBuffer<T>(M);
  common::cpu::copyBuffer(Q, A, M * N);
  common::cpu::normalize(Q, Q, M);
  for (size_t j = 1; j < N; ++j) {
    T* qj = Q + j * M;
    for (size_t i = 0; i < j; ++i) {
      const T* qi = Q + i * M;
      common::cpu::project(qj, qi, tmp, M);
      common::cpu::sub(qj, tmp, qj, M);
    }
    common::cpu::normalize(qj, qj, M);
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
