#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace utils {

#define CUDACHECK(err)                                                         \
  do {                                                                         \
    cuda_check((err), __FILE__, __LINE__);                                     \
  } while (false)
inline void cuda_check(cudaError_t error_code, const char *file, int line) {
  if (error_code != cudaSuccess) {
    fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code,
            cudaGetErrorString(error_code), file, line);
    fflush(stderr);
    exit(error_code);
  }
}

template <typename T> inline T *allocateCpuBuffer(size_t size) {
  T *buffer;
  CUDACHECK(cudaMallocHost(&buffer, size * sizeof(T)));
  return buffer;
}

template <typename T> inline void freeCpuBuffer(T *buffer) {
  CUDACHECK(cudaFreeHost(buffer));
}

} // namespace utils
