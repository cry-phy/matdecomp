#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace qr::gpu {

template <typename T> struct BlockedHouseholderWorkspace {
  T* VWorkspace;
  T* TWorkspace;
  T* WWorkspace;
  T* CWorkspace1;
  T* CWorkspace2;

  T* tauWorkspace;
  T* TwWorkspace;
};

template <typename T>
void blockedHouseHolder(const T* A, T* Q, T* R, uint32_t m, uint32_t n, uint32_t blockSize);

} // namespace qr::gpu
namespace qr::gpu::_detail {

template <typename T> void singleReflector(T* v_j, T* tau_j, uint32_t M_eff);

template <typename T>
__global__ void applyReflectorKernel(T* C, const T* v_j, const T* tau_j, uint32_t stride,
                                     uint32_t N, uint32_t M_eff);

template <typename T>
void unblockedPanel(T* V, T* tau, uint32_t stride, uint32_t N, uint32_t M_eff);

template <typename T>
void buildTMatrix(T* V, T* tau, T* Tmat, uint32_t stride, uint32_t panelWidth, T* workspace);

template <typename T>
void blockTrailingUpdate(T* C, const T* V, const T* Tmat, uint32_t M, uint32_t N,
                         uint32_t panelWidth, T* W_workspace, T* C_workspace);

template <typename T>
void blockedFinalMatrix(const T* A, T* Q, T* R, uint32_t M, uint32_t N,
                        BlockedHouseholderWorkspace<T>&& workspace);
} // namespace qr::gpu::_detail
