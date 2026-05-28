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

template <typename T> __device__ T _warpReduceSum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

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

template <typename T> __device__ void _scale(const T* a, T* res, size_t N, T factor) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    res[idx] = a[idx] * factor;
}

// ── kernels ───────────────────────────────────────────────────────────────────

template <typename T> __global__ void subtractKernel(const T* a, const T* b, T* res, size_t N) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    res[idx] = a[idx] - b[idx];
}

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

template <typename T> __global__ void scaleKernel(const T* a, T* res, size_t N, T factor) {
  _scale(a, res, N, factor);
}

// Reads normSq from device memory — used as the second step of normalize so
// that both launches can be enqueued on the same stream without a host sync.
template <typename T>
__global__ void normalizeKernel(const T* a, T* res, const T* normSq, uint32_t N) {
  T invNorm = rsqrtf(*normSq);
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    res[idx] = a[idx] * invNorm;
}

// Reads numerator and denominator from device memory so both dot products and
// the scale can be enqueued on the same stream without a host sync.
template <typename T>
__global__ void projectScaleKernel(const T* b, T* res, size_t N, const T* num, const T* den) {
  T factor = *num / *den;
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    res[idx] = b[idx] * factor;
}

// ── host launchers ────────────────────────────────────────────────────────────

template <typename T> void subtract(const T* a, const T* b, T* res, size_t N) {
  uint32_t blockSize = MAX_BLOCK_SIZE;
  uint32_t numBlocks = (N + blockSize - 1) / blockSize;
  subtractKernel<T><<<numBlocks, blockSize>>>(a, b, res, N);
}

template <typename T> void dotProduct(const T* a, const T* b, T* res, uint32_t N) {
  uint32_t blockSize = MAX_BLOCK_SIZE;
  uint32_t numBlocks =
      (N + blockSize * INIT_REDUCTION_COUNT - 1) / (blockSize * INIT_REDUCTION_COUNT);
  cudaMemset(res, 0, sizeof(T));
  dotProductKernel<T><<<numBlocks, blockSize>>>(a, b, res, N);
}

template <typename T> void scale(const T* a, T* res, size_t N, T factor) {
  uint32_t blockSize = MAX_BLOCK_SIZE;
  uint32_t numBlocks = (N + blockSize - 1) / blockSize;
  scaleKernel<T><<<numBlocks, blockSize>>>(a, res, N, factor);
}

// Enqueues dotProductKernel(a,a) then normalizeKernel on the same stream —
// no host sync needed. Caller owns d_normSq (one T on device).
template <typename T> void normalize(const T* a, T* res, uint32_t N, T* d_normSq) {
  uint32_t blockSize = MAX_BLOCK_SIZE;
  cudaMemset(d_normSq, 0, sizeof(T));
  uint32_t numBlocks =
      (N + blockSize * INIT_REDUCTION_COUNT - 1) / (blockSize * INIT_REDUCTION_COUNT);
  dotProductKernel<T><<<numBlocks, blockSize>>>(a, a, d_normSq, N);
  numBlocks = (N + blockSize - 1) / blockSize;
  normalizeKernel<T><<<numBlocks, blockSize>>>(a, res, d_normSq, N);
}

// Enqueues two dot products then projectScaleKernel on the same stream.
// Caller owns d_dot_ab and d_dot_bb (one T each on device).
template <typename T>
void project(const T* a, const T* b, T* res, size_t N, T* d_dot_ab, T* d_dot_bb) {
  uint32_t blockSize = MAX_BLOCK_SIZE;
  uint32_t numBlocks =
      (N + blockSize * INIT_REDUCTION_COUNT - 1) / (blockSize * INIT_REDUCTION_COUNT);
  cudaMemset(d_dot_ab, 0, sizeof(T));
  cudaMemset(d_dot_bb, 0, sizeof(T));
  dotProductKernel<T><<<numBlocks, blockSize>>>(a, b, d_dot_ab, N);
  dotProductKernel<T><<<numBlocks, blockSize>>>(b, b, d_dot_bb, N);
  numBlocks = (N + blockSize - 1) / blockSize;
  projectScaleKernel<T><<<numBlocks, blockSize>>>(b, res, N, d_dot_ab, d_dot_bb);
}

} // namespace gpu

} // namespace common

#endif // __CUDACC__
