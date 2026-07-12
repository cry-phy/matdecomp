#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace qr::gpu {
// Returns cudaSuccess, or the failing cudaMalloc's error (freeing any partial
// allocation) so callers can skip oversized cases instead of aborting.
template <typename T>
cudaError_t Householder(const T* A, T* Q, T* R, uint32_t M, uint32_t N, T* workspace);
} // namespace qr::gpu
