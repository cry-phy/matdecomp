#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace qr::gpu {

template <typename T> struct BlockedHouseholderWorkspace {
  T* res;     // working copy of A; receives packed V (below diag) + R (upper)
  T* tau;     // reflector scalars, length N
  T* Tmat;    // block T factor, (panelWidth x panelWidth)
  T* W1;      // trailing-update workspace (ld = M, holds up to M x N)
  T* W2;      // trailing-update workspace (ld = M, holds up to M x N)
  T* C_trail; // trailing-update result workspace (M x N)
  uint32_t panelWidth;
};

template <typename T>
void mallocWorkspace(BlockedHouseholderWorkspace<T>& workspace, uint32_t M, uint32_t N,
                     uint32_t panelWidth);

template <typename T> void freeWorkspace(BlockedHouseholderWorkspace<T>& workspace);

template <typename T>
void blockedHouseHolder(const T* A, T* Q, T* R, uint32_t M, uint32_t N,
                        BlockedHouseholderWorkspace<T>& workspace);

} // namespace qr::gpu

namespace qr::gpu::_detail {

template <typename T> void singleReflector(T* v_j, T* tau_j, uint32_t M_eff);

template <typename T>
__global__ void applyReflectorKernel(T* C, const T* v_j, const T* tau_j, uint32_t stride,
                                     uint32_t N, uint32_t M_eff);

template <typename T>
void unblockedPanel(T* V, T* tau, uint32_t stride, uint32_t N, uint32_t M_eff);

template <typename T>
void buildTMatrix(T* V, T* tau, T* Tmat, uint32_t M, uint32_t N, uint32_t ldv);

template <typename T>
void blockTrailingUpdate(T* C, const T* V, const T* Tmat, uint32_t M, uint32_t N,
                         uint32_t panelWidth, T* W_workspace1, T* W_workspace2, T* C_workspace);

template <typename T>
void blockedOrchestrator(const T* A, T* res, uint32_t M, uint32_t N,
                         BlockedHouseholderWorkspace<T>&& workspace);

template <typename T> void constructR(const T* finalMatrix, T* R, uint32_t M, uint32_t N);

template <typename T> void constructQ(const T* finalMatrix, T* tau, T* Q, uint32_t M, uint32_t N);

} // namespace qr::gpu::_detail
