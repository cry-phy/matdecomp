#include "common/gpuLinalg.cuh"
#include "common/gpuUtils.cuh"
#include "qr/gpuBlockedHouseholder.cuh"
#include "qr/gpuhouseholder.cuh"
#include <algorithm>
#include <cmath>
#define PANEL_WIDTH 32

namespace qr::gpu::_detail {

template <typename T> void singleReflector(T* v_j, T* tau_j, int M) {
  T* d_tmp;
  T norm, alpha;
  CUDACHECK(cudaMalloc(&d_tmp, sizeof(T)));
  common::gpu::dotProduct(v_j + 1, v_j + 1, d_tmp, M - 1);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaMemcpy(&norm, d_tmp, sizeof(T), cudaMemcpyDeviceToHost));
  norm = sqrt(norm);

  // Column already zero below the diagonal: identity reflector, diagonal is its
  // own R value. tau=0 makes applyReflector a no-op regardless of v.
  if (norm <= 1e-8) {
    const T zero = static_cast<T>(0);
    CUDACHECK(cudaMemcpy(tau_j, &zero, sizeof(T), cudaMemcpyHostToDevice));
    CUDACHECK(cudaFree(d_tmp));
    return;
  }

  CUDACHECK(cudaMemcpy(&alpha, v_j, sizeof(T), cudaMemcpyDeviceToHost));
  // Sign choice avoids cancellation in (alpha - beta).
  T beta = -copysign(sqrt(alpha * alpha + norm * norm), alpha);
  T tau = (beta - alpha) / beta;
  CUDACHECK(cudaMemcpy(tau_j, &tau, sizeof(T), cudaMemcpyHostToDevice));
  T scale = 1 / (alpha - beta);
  common::gpu::scale(v_j + 1, v_j + 1, M - 1, scale);
  // LAPACK packed convention: the reflector's first element is implicitly 1, so the
  // diagonal stores beta (the R diagonal). applyReflectorKernel uses v[0]==1 implicitly.
  CUDACHECK(cudaMemcpy(v_j, &beta, sizeof(T), cudaMemcpyHostToDevice));
  CUDACHECK(cudaFree(d_tmp));
  cudaDeviceSynchronize();
}

template <typename T>
__global__ void applyReflectorKernel(T* C, const T* v_j, const T* tau_j, size_t stride, size_t N,
                                     size_t M_eff) {
  // Each block processes one column: C[:,col] -= tau * (v^T C[:,col]) * v
  // The reflector's first element is implicitly 1 (LAPACK packed convention);
  // v_j[0] holds the R diagonal, not a reflector entry.
  size_t tid = threadIdx.x;
  size_t col = blockIdx.x;

  if (col >= N)
    return;

  T vv = static_cast<T>(0);
  T contrib = static_cast<T>(0);
  if (tid < M_eff) {
    vv = (tid == 0) ? static_cast<T>(1) : v_j[tid];
    contrib = C[col * stride + tid] * vv;
  }

  // All threads must enter _blockReduce (it syncs internally) and it returns the
  // full block sum to every thread.
  T w_c = common::gpu::_blockReduce(contrib);

  if (tid < M_eff)
    C[col * stride + tid] -= (*tau_j) * w_c * vv;
}

template <typename T> void unblockedPanel(T* V, T* tau, size_t M, size_t N) {
  for (size_t j = 0; j < N; ++j) {
    singleReflector(V + M * j + j, tau + j, M - j);
    if (j < N - 1) {
      applyReflectorKernel<<<N - j - 1, M - j>>>(V + M * (j + 1) + j, &V[M * j + j], tau + j, M,
                                                 N - j - 1, M - j);
      cudaDeviceSynchronize();
    }
  }
}

template void singleReflector<float>(float*, float*, int);
template void singleReflector<double>(double*, double*, int);
template void unblockedPanel<float>(float*, float*, size_t, size_t);
template void unblockedPanel<double>(double*, double*, size_t, size_t);
template __global__ void applyReflectorKernel<float>(float*, const float*, const float*, size_t,
                                                     size_t, size_t);
template __global__ void applyReflectorKernel<double>(double*, const double*, const double*, size_t,
                                                      size_t, size_t);

} // namespace qr::gpu::_detail

namespace qr::gpu {

template <typename T> struct BlockedHouseholderWorkspace {
  T* VWorkspace;
  T* TWorkspace;
  T* panelWorkspace;
  size_t panelWidth;
  size_t panelHeight;
};

template <typename T>
void blockedHouseHolder(const T* A, T* Q, T* R, uint32_t M, uint32_t N,
                        BlockedHouseholderWorkspace<T>&& workspace) {
  for (uint32_t k = 0; k < N; k += workspace.panelWidth) {
    uint32_t currentBlockSize = std::min(workspace.panelWidth, N - k);

    // Perform panel Householder transformation (sequential)
    qr::gpu::Householder(&A[k * M], workspace.VWorkspace, workspace.TWorkspace, M, currentBlockSize,
                         workspace.panelWorkspace);
  }
}

} // namespace qr::gpu
