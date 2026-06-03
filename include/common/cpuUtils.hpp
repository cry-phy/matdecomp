#pragma once
#include "common/cpuBuffer.hpp"
#include "common/cpuLinalg.hpp"
#include <cmath>
#include <cstddef>
#include <cstdlib>

namespace common::cpu {
template <typename T> inline void fillRandom(T* buffer, size_t size) {
  for (size_t i = 0; i < size; i++) {
    buffer[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
  }
}

// Frobenius norm of (QᵀQ - I). Computes all N² entries to avoid reading
// uninitialised memory from a partial fill.
template <typename T> inline T orthogonalityError(const T* Q, size_t M, size_t N) {
  T error = T(0);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      T d = dot(Q + i * M, Q + j * M, M) - (i == j ? T(1) : T(0));
      error += d * d;
    }
  }
  return std::sqrt(error);
}

} // namespace common::cpu
