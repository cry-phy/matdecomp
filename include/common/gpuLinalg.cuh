#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_types.h>
#include <sys/types.h>

namespace common {

namespace gpu {

// Parameter for atomic bound reduction. Number of elements reduced by each thread at start (reduces
// inter block reduction overhead). Must be tuned to the hardware!
#define INIT_REDUCTION_COUNT 16
#define MAX_BLOCK_SIZE 1024

/**
 * @brief Reduce a block of threads to a single value using addition.
 *
 * @param sdata Data to be reduced, stored in shared memory (size equal to blockSize).
 * @param tid Thread ID within the block (0 to blockSize-1).
 * @return The reduced sum of the values in sdata for the block.
 */
template <typename T, int blockSize>
__device__ void warpReduceSum(volatile T* sdata, uint32_t tid) {
  // Fold-based reduction within a warp, with no shared memory bank conflicts,
  // no divergence, no loop overhead, no manual synchronization.
  //
  // The volatile qualifier is needed to ensure the compiler does not use
  // registers, breaking the coalesced access pattern.
  if (blockSize >= 64)
    sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32)
    sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16)
    sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8)
    sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4)
    sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2)
    sdata[tid] += sdata[tid + 1];
}

/**
 * @brief Reduce a block of threads to a single value using addition.
 *
 * @param sdata Data to be reduced, stored in shared memory (size equal to blockSize).
 * @param output Pointer to the output variable where the reduced sum will be stored (in global
 * memory).
 * @param N Total number of elements being reduced
 * @return void
 */
template <typename T, uint32_t blockSize>
__device__ void reduceSum(volatile T* sdata, T* output, uint32_t N) {
  // Fully unrolled reduction, with warp-level reduction for the last warp
  uint32_t tid = threadIdx.x;

  if (blockSize >= 1024) {
    if (tid < 512) {
      sdata[tid] += sdata[tid + 512];
    }
    __syncthreads();
  }

  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
  }

  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
  }

  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
  }

  // Since we can fold directly into the warp size array we can avoid an extra
  // synchronization step.
  if (tid < 32) {
    warpReduceSum<T, blockSize>(sdata, tid);
  }
  if (tid == 0)
    atomicAdd(output, sdata[0]);
}

/**
 * @brief Compute the dot product of two vectors a and b, storing the result in res.
 *
 * @param a Vector a (size N, input)
 * @param b Vector b (size N, input)
 * @param res Pointer to the output variable where the dot product will be stored (in global
 * memory).
 * @param N Size of the input vectors a and b
 * @return void
 */
template <typename T, uint32_t blockSize>
__device__ void dotProduct(const T* a, const T* b, T* res, uint32_t N) {
  // Fully unrolled reduction, with warp-level reduction for the last warp
  __shared__ T sdata[blockSize];
  uint32_t xBlock = blockSize * INIT_REDUCTION_COUNT;
  uint32_t idx = blockIdx.x * xBlock + threadIdx.x;
  uint32_t tid = threadIdx.x;

  sdata[tid] = static_cast<T>(0.0f);
  T aval;
  T bval;
  for (uint32_t i = 0; i < INIT_REDUCTION_COUNT; i++) {
    if (idx + i * blockSize < N) {
      aval = a[idx + i * blockSize];
      bval = b[idx + i * blockSize];
    } else {
      aval = static_cast<T>(0.0f);
      bval = static_cast<T>(0.0f);
    }
    sdata[tid] += aval * bval;
  }
  __syncthreads();

  reduceSum<T, blockSize>(sdata, res, N);
}

/**
 * @brief Compute the L2 norm of a vector a, storing the result in res.
 *
 * @param a Vector a (size N, input)
 * @param res Pointer to the output variable where the L2 norm will be stored (in global memory).
 * @param N Size of the input vector a
 * @return void
 */
template <typename T, uint32_t blockSize> __device__ void l2Norm(const T* a, T* res, uint32_t N) {
  dotProduct<T, blockSize>(a, a, res, N);
}

template <typename T, uint32_t blockSize> __device__ void normalize(T* a, T* res, uint32_t N) {
  __shared__ T norm;
  if (threadIdx.x == 0) {
    norm = 0.0f;
  }
  __syncthreads();

  l2Norm<T, blockSize>(a, &norm, N);
  __syncthreads();

  T invNorm = rsqrtf(norm);
  uint32_t idx = blockIdx.x * blockSize + threadIdx.x;
  if (idx < N) {
    res[idx] = a[idx] * invNorm;
  }
}

/**
 * @brief Scale a vector a by a factor, storing the result in res. The scaling is performed
 * element-wise.
 *
 * @param a Vector a (size N, input)
 * @param res Vector res (size N, output)
 * @param N Size of the input vector a and output vector res
 * @param factor Scaling factor (scalar, input)
 * @return void
 */
template <typename T> __device__ void scale(const T* a, T* res, size_t N, const T factor) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    res[idx] = a[idx] * factor;
}

/**
 * @brief Project vector a onto vector b, storing the result in res. The projection is computed as:
 *
 * @param a Vector a to be projected (size N, input)
 * @param b Vector b to project onto (size N, input)
 * @param res Vector res to store the projected result (size N, output)
 * @param N Size of the input vectors a and b, and output vector res
 * @return void
 */
template <typename T, uint32_t blockSize>
__device__ void project(const T* a, const T* b, T* res, size_t N) {
  __shared__ T dot_ab;
  __shared__ T dot_bb;
  if (threadIdx.x == 0) {
    dot_ab = static_cast<T>(0.0f);
    dot_bb = static_cast<T>(0.0f);
  }
  __syncthreads();
  dotProduct<T, blockSize>(a, b, &dot_ab, N);
  dotProduct<T, blockSize>(b, b, &dot_bb, N);
  __syncthreads();
  scale(b, res, N, dot_ab / dot_bb);
}

} // namespace gpu

} // namespace common

#endif // __CUDACC__
