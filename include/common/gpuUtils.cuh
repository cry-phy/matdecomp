#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

namespace common {

namespace gpu {

#define CUDACHECK(err)                                                                             \
  do {                                                                                             \
    common::gpu::cuda_check((err), __FILE__, __LINE__);                                            \
  } while (false)
inline void cuda_check(cudaError_t error_code, const char* file, int line) {
  if (error_code != cudaSuccess) {
    fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code,
            cudaGetErrorString(error_code), file, line);
    fflush(stderr);
    exit(error_code);
  }
}
template <typename T> __device__ void fillBuffer(T* buffer, size_t N, T value) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N) {
    buffer[idx] = value;
  }
}

template <typename T> __device__ void zeroBuffer(T* buffer, size_t N) {
  fillBuffer(buffer, N, static_cast<T>(0.0f));
}

template <typename T> __global__ void randomFill(T* buffer, size_t N) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N) {
    // Simple LCG for random number generation
    uint32_t state = idx + 1;                                         // Avoid zero state
    state = (1103515245 * state + 12345) & 0x7fffffff;                // LCG parameters
    buffer[idx] = static_cast<T>(state) / static_cast<T>(0x7fffffff); // Normalize to [0, 1)
  }
}

} // namespace gpu

} // namespace common
