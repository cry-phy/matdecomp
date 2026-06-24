#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

namespace common {

namespace gpu {

/**
 * @brief Checks the return value of a CUDA API call and prints an error message if it indicates a
 * failure, including the file and line number where the error occurred. If an error is detected,
 * the program will exit with the error code.
 *
 * @param error_code The return value from a CUDA API call to be checked for errors
 * @param file The name of the source file where the error occurred (automatically provided by
 * __FILE__)
 * @param line The line number in the source file where the error occurred (automatically provided
 * by __LINE__)
 * @return void
 */
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
template <typename T> __global__ void fillBufferKernel(T* buffer, uint64_t N, T value) {
  uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N) {
    buffer[idx] = value;
  }
}

template <typename T> void fillBuffer(T* buffer, uint64_t N, T value) {
  const uint32_t threadsPerBlock = std::min(static_cast<uint32_t>(256), static_cast<uint32_t>(N));
  uint32_t blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  std::cout << "Filling buffer with " << N << " elements using " << blocks << " blocks of "
            << threadsPerBlock << " threads each." << std::endl;
  fillBufferKernel<<<blocks, threadsPerBlock>>>(buffer, N, value);
  CUDACHECK(cudaGetLastError());
}

/**
 * @brief Fills the given buffer with zeros. Each thread will set one element to zero until the
 * entire
 *
 * @param buffer Pointer to the buffer to be filled with zeros
 * @param N Number of elements in the buffer
 * @return void
 */
template <typename T> void zeroBuffer(T* buffer, uint64_t N) {
  fillBuffer(buffer, N, static_cast<T>(0.0f));
}

/**
 * @brief Fills the given buffer with random values in the range [0, 1).
 *
 * @param buffer Pointer to the buffer to be filled with random values
 * @param N Number of elements in the buffer
 * @return void
 */
template <typename T> __global__ void randomFill(T* buffer, uint64_t N) {
  uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N) {
    // Simple LCG for random number generation
    uint32_t state = idx + 1;                                         // Avoid zero state
    state = (1103515245 * state + 12345) & 0x7fffffff;                // LCG parameters
    buffer[idx] = static_cast<T>(state) / static_cast<T>(0x7fffffff); // Normalize to [0, 1)
  }
}

/**
 * @brief Copies data from the source buffer to the destination buffer.
 *
 * @param src pointer to the source buffer
 * @param dst pointer to the destination buffer
 * @param N number of elements to copy
 * @return void
 */
template <typename T> __global__ void copyBuffer(const T* src, T* dst, uint64_t N) {
  uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N) {
    dst[idx] = src[idx];
  }
}

} // namespace gpu

} // namespace common
