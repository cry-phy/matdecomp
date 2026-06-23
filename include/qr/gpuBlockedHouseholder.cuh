#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace qr::gpu::_detail {

template <typename T> void singleReflector(T* v_j, T* tau_j, int M);

template <typename T>
__global__ void applyReflectorKernel(T* C, const T* v_j, const T* tau_j, size_t stride, size_t N,
                                     size_t M_eff);

template <typename T> void unblockedPanel(T* V, T* tau, size_t M, size_t N);

} // namespace qr::gpu::_detail

namespace qr::gpu {

template <typename T>
void blockedHouseHolder(const T* A, T* Q, T* R, uint32_t m, uint32_t n, uint32_t blockSize);

} // namespace qr::gpu
