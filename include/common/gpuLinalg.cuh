#pragma once
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#include <cuda_runtime.h>

namespace common {

namespace gpu {

#define INIT_REDUCTION_COUNT 16
#define MAX_BLOCK_SIZE 1024
#define WARP_SIZE 32

// ── device building blocks ────────────────────────────────────────────────────

/**
 * @brief Performs a warp-level reduction to sum val across the warp using shuffle down operations.
 *
 * @param val Value to be reduced across the warp
 * @return The sum of val across the warp
 */
template <typename T> __device__ T _warpReduceSum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

/**
 * @brief Reduces val across the block using warp-level primitives in shared memory.
 *
 * @param val Value to be reduced across the block
 * @return The sum of val across the block
 */
template <typename T> __device__ T _blockReduce(T val) {
  __shared__ T warpSums[MAX_BLOCK_SIZE / WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int warpId = threadIdx.x / WARP_SIZE;

  val = _warpReduceSum<T>(val);
  if (lane == 0)
    warpSums[warpId] = val;
  __syncthreads();

  if (warpId == 0) {
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpSums[lane] : static_cast<T>(0);
    val = _warpReduceSum<T>(val);
  }
  return val;
}

/**
 * @brief Scales vector b by factor and stores in res.
 *
 * @param a Vector to be scaled
 * @param res Output vector to hold the scaled result
 * @param N Number of elements in a and res
 * @param factor Scaling factor
 * @return void
 */
template <typename T> __device__ void _scale(const T* a, T* res, size_t N, T factor) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    res[idx] = a[idx] * factor;
}

// ── kernels ───────────────────────────────────────────────────────────────────

/**
 * @brief Subtracts two vectors a and b, storing the result in res.
 *
 * @param a Input vector a
 * @param b Input vector b
 * @param res Output vector to hold the result of a - b
 * @param N Number of elements in a, b, and res
 * @return void
 */
template <typename T> __global__ void subtractKernel(const T* a, const T* b, T* res, size_t N) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    res[idx] = a[idx] - b[idx];
}

/**
 * @brief Performs the dot product of two vectors a and b, storing the result in res.
 *
 * @param a Input vector a
 * @param b Input vector b
 * @param res Device pointer to store the result. Must be initialized to zero before launch.
 * @param N Number of elements in a and b
 * @return void
 */
template <typename T> __global__ void dotProductKernel(const T* a, const T* b, T* res, uint32_t N) {
  uint32_t base = blockIdx.x * blockDim.x * INIT_REDUCTION_COUNT + threadIdx.x;
  T val = static_cast<T>(0);
  for (uint32_t i = 0; i < INIT_REDUCTION_COUNT; i++) {
    uint32_t idx = base + i * blockDim.x;
    if (idx < N)
      val += a[idx] * b[idx];
  }
  val = _blockReduce<T>(val);
  if (threadIdx.x == 0)
    atomicAdd(res, val);
}

/**
 * @brief Scales vector a by factor.
 *
 * @param a Input vector to be scaled
 * @param res Output vector to hold the scaled result
 * @param N Number of elements in a and res
 * @param factor Scaling factor
 * @return void
 */
template <typename T> __global__ void scaleKernel(const T* a, T* res, size_t N, T factor) {
  _scale(a, res, N, factor);
}

/**
 * @brief Normalizes vector a and stores the result in res.
 *
 * @param a Vector to be normalized
 * @param res Output vector to hold the normalized result
 * @param normSq Device pointer to the squared norm of a (one T on device)
 * @param N Number of elements in a and res
 * @return void
 */
template <typename T>
__global__ void normalizeKernel(const T* a, T* res, const T* normSq, uint32_t N) {
  T invNorm = rsqrtf(*normSq);
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    res[idx] = a[idx] * invNorm;
}

/**
 * @brief scale b by factor and store in res. Factor is read from device memory to enable queuing
 * both the dot product and scaling.
 *
 * @param b Vector to be scaled
 * @param res Output vector to hold the scaled result
 * @param N Number of elements in b and res
 * @param factor Scaling factor read from device memory (one T on device)
 * @return void
 */
template <typename T> __global__ void scaleKernel(const T* b, T* res, size_t N, const T* factor) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    res[idx] = b[idx] * *factor;
}

// ── host launchers ────────────────────────────────────────────────────────────

/**
 * @brief Subtracts two vectors a and b, storing the result in res.
 *
 * @param a Input vector a
 * @param b Input vector b
 * @param res Output vector to hold the result of a - b
 * @param N Number of elements in a, b, and res
 * @return void
 */
template <typename T> void subtract(const T* a, const T* b, T* res, size_t N) {
  uint32_t blockSize = MAX_BLOCK_SIZE;
  uint32_t numBlocks = (N + blockSize - 1) / blockSize;
  subtractKernel<T><<<numBlocks, blockSize>>>(a, b, res, N);
}

/**
 * @brief Performs the dot product of two vectors a and b, storing the result in res.
 *
 * @param a Input vector a
 * @param b Input vector b
 * @param res Device pointer to store the result. Must be initialized to zero.
 * @param N Number of elements in a and b
 * @return void
 */
template <typename T> void dotProduct(const T* a, const T* b, T* res, uint32_t N) {
  uint32_t blockSize = MAX_BLOCK_SIZE;
  uint32_t numBlocks =
      (N + blockSize * INIT_REDUCTION_COUNT - 1) / (blockSize * INIT_REDUCTION_COUNT);
  cudaMemset(res, 0, sizeof(T));
  dotProductKernel<T><<<numBlocks, blockSize>>>(a, b, res, N);
}

/**
 * @brief Scales a vector a by a factor and stores the result in res.
 *
 * @param a Input vector to be scaled
 * @param res Output vector to hold the scaled result
 * @param N Number of elements in a and res
 * @param factor Scaling factor
 * @return void
 */
template <typename T> void scale(const T* a, T* res, size_t N, T factor) {
  uint32_t blockSize = MAX_BLOCK_SIZE;
  uint32_t numBlocks = (N + blockSize - 1) / blockSize;
  scaleKernel<T><<<numBlocks, blockSize>>>(a, res, N, factor);
}

/**
 * @brief Normalizes a vector a and stores the result.
 *
 * @param a Vector to be normalized
 * @param res Output vector to hold the normalized result
 * @param N Number of elements in a and res
 * @param d_normSq Device pointer to store the squared norm of a (one T on device)
 * @return void
 */
template <typename T> void normalize(const T* a, T* res, uint32_t N, T* d_normSq) {
  uint32_t blockSize = MAX_BLOCK_SIZE;
  cudaMemset(d_normSq, 0, sizeof(T));
  uint32_t numBlocks =
      (N + blockSize * INIT_REDUCTION_COUNT - 1) / (blockSize * INIT_REDUCTION_COUNT);
  dotProductKernel<T><<<numBlocks, blockSize>>>(a, a, d_normSq, N);
  numBlocks = (N + blockSize - 1) / blockSize;
  normalizeKernel<T><<<numBlocks, blockSize>>>(a, res, d_normSq, N);
}

/**
 * @brief Projects a onto b by computing the dot product of a and b. Does not normalize b, so the
 * result is not a unit vector.
 *
 * @param a Vector to be projected
 * @param b Vector to project onto
 * @param res Output vector
 * @param N Number of elements in a, b, and res
 * @param d_dot_ab Device pointer to store the dot product of a and b (one T on device)
 * @return void
 */
template <typename T> void project(const T* a, const T* b, T* res, size_t N, T* d_dot_ab) {
  uint32_t blockSize = MAX_BLOCK_SIZE;
  uint32_t numBlocks =
      (N + blockSize * INIT_REDUCTION_COUNT - 1) / (blockSize * INIT_REDUCTION_COUNT);
  cudaMemset(d_dot_ab, 0, sizeof(T));
  dotProductKernel<T><<<numBlocks, blockSize>>>(a, b, d_dot_ab, N);
  numBlocks = (N + blockSize - 1) / blockSize;
  scaleKernel<T><<<numBlocks, blockSize>>>(b, res, N, d_dot_ab);
}

} // namespace gpu

} // namespace common

#endif // __CUDACC__
