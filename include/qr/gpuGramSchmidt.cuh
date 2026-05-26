#pragma once
#include "common/gpuLinalg.cuh"
#include <cstdint>
#include <cuda_runtime.h>

namespace qr {

template <typename T>
__device__ void gpuGramSchmidt(const T* A, T* Q, T* R, uint32_t M, uint32_t N, T* workspace) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (uint32_t i = 0; i < N; i++) {
    if (idx < M) {
      Q[i * M + idx] = A[i * M + idx];
    }
    if (idx < N) {
      R[i * N + idx] = 0;
    }
  }
  __syncthreads();

  common::gpu::normalize<T, MAX_BLOCK_SIZE>(&Q[0], &Q[0], M);
  __syncthreads();

  for (uint32_t j = 1; j < M; j++) {
    for (uint32_t i = 0; i < j; i++) {
      common::gpu::project<T, MAX_BLOCK_SIZE>(&Q[j * M], &Q[i * M], workspace, M);
      __syncthreads();
      if (idx < M) {
        Q[j * M + idx] = Q[j * M + idx] - workspace[idx];
      }
      __syncthreads();
    }
    common::gpu::normalize<T, MAX_BLOCK_SIZE>(&Q[j * M], &Q[j * M], M);
    __syncthreads();
  }

  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i <= j; ++i) {
      common::gpu::dotProduct<T, MAX_BLOCK_SIZE>(const_cast<T*>(&Q[i * M]), &A[j * M],
                                                 &R[i + j * N], M);
    }
  }
}

} // namespace qr
namespace {

template <typename T> __device__ void gsComputeQ(T* A, T* Q, uint32_t M, uint32_t N) {}

template <typename T> __device__ void gsComputeR(T* A, T* Q, T* R, uint32_t M, uint32_t N) {}

} // namespace
