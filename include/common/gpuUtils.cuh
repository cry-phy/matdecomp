#pragma once
#include <cuda_runtime.h>

namespace common {

namespace gpu {
template <typename T> __device__ void fillBuffer(T* buffer, size_t N, T value) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N) {
    buffer[idx] = value;
  }
}

template <typename T> __device__ void zeroBuffer(T* buffer, size_t N) {
  fillBuffer(buffer, N, static_cast<T>(0.0f));
}

} // namespace gpu

} // namespace common
