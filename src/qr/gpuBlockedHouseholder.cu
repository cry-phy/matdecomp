#include "common/gpuLinalg.cuh"
#include "common/gpuUtils.cuh"
#include "qr/gpuBlockedHouseholder.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#define PANEL_WIDTH 32

namespace qr::gpu::_detail {

template <typename T> void singleReflector(T* v_j, T* tau_j, uint32_t M_eff) {
  T* d_tmp;
  T norm, alpha;
  CUDACHECK(cudaMalloc(&d_tmp, sizeof(T)));
  common::gpu::dotProduct(v_j + 1, v_j + 1, d_tmp, M_eff - 1);
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
  T beta = -copysign(sqrt(alpha * alpha + norm * norm), alpha);
  T tau = (beta - alpha) / beta;
  CUDACHECK(cudaMemcpy(tau_j, &tau, sizeof(T), cudaMemcpyHostToDevice));
  T scale = 1 / (alpha - beta);
  common::gpu::scale(v_j + 1, v_j + 1, M_eff - 1, scale);
  // LAPACK packed convention: the reflector's first element is implicitly 1, so the
  // diagonal stores beta (the R diagonal). applyReflectorKernel uses v[0]==1 implicitly.
  CUDACHECK(cudaMemcpy(v_j, &beta, sizeof(T), cudaMemcpyHostToDevice));
  CUDACHECK(cudaFree(d_tmp));
  cudaDeviceSynchronize();
}

template <typename T>
__global__ void applyReflectorKernel(T* C, const T* v_j, const T* tau_j, uint32_t stride,
                                     uint32_t N, uint32_t M_eff) {
  // Each block processes one column: C[:,col] -= tau * (v^T C[:,col]) * v
  // The reflector's first element is implicitly 1 (LAPACK packed convention);
  // v_j[0] holds the R diagonal, not a reflector entry.
  uint32_t tid = threadIdx.x;
  uint32_t col = blockIdx.x;

  if (col >= N)
    return;

  T vv = static_cast<T>(0);
  T contrib = static_cast<T>(0);
  if (tid < M_eff) {
    vv = (tid == 0) ? static_cast<T>(1) : v_j[tid];
    contrib = C[static_cast<uint64_t>(col) * stride + tid] * vv;
  }

  T w_c = common::gpu::_blockReduce(contrib);

  if (tid < M_eff)
    C[static_cast<uint64_t>(col) * stride + tid] -= (*tau_j) * w_c * vv;
}

template <typename T>
void unblockedPanel(T* V, T* tau, uint32_t stride, uint32_t N, uint32_t M_eff) {
  for (uint32_t j = 0; j < N; ++j) {
    singleReflector(V + static_cast<uint64_t>(stride) * j + j, tau + j, M_eff - j);
    if (j < N - 1) {
      applyReflectorKernel<<<N - j - 1, M_eff - j>>>(
          V + static_cast<uint64_t>(stride) * (j + 1) + j,
          &V[static_cast<uint64_t>(stride) * j + j], tau + j, stride, N - j - 1, M_eff - j);
      cudaDeviceSynchronize();
    }
  }
} // namespace qr::gpu::_detail

template <typename T>
void buildTMatrix(T* V, T* tau, T* Tmat, uint32_t stride, uint32_t panelWidth, T* workspace) {
  // TODO merge into single larft kernel to avoid host-device round trips.
  T* local_tau = new T[panelWidth];
  CUDACHECK(cudaMemcpy(local_tau, tau, panelWidth * sizeof(T), cudaMemcpyDeviceToHost));

  // Cache for V's diagonal since its needed for v_j implicit unit diagonal
  T* tmp_beta;
  CUDACHECK(cudaMalloc(&tmp_beta, sizeof(T)));

  // Lower triangle must be zero so the triangular matvec can sum over the full j x j block.
  CUDACHECK(cudaMemset(Tmat, 0, static_cast<uint64_t>(panelWidth) * panelWidth * sizeof(T)));

  for (uint32_t j = 0; j < panelWidth; ++j) {
    T* t_col = Tmat + static_cast<uint64_t>(j) * panelWidth; // column j of T
    if (std::abs(local_tau[j]) <= static_cast<T>(1e-8))
      continue;

    if (j > 0) {
      T* v_diag = V + static_cast<uint64_t>(j) * stride + j;
      CUDACHECK(cudaMemcpy(tmp_beta, v_diag, sizeof(T), cudaMemcpyDeviceToDevice));
      common::gpu::fillBuffer(v_diag, 1, static_cast<T>(1));
      for (uint32_t k = 0; k < j; ++k) {
        T* d_vk = V + static_cast<uint64_t>(k) * stride + j;
        common::gpu::dotProduct(d_vk, v_diag, t_col + k, stride - j);
      }
      CUDACHECK(cudaMemcpy(v_diag, tmp_beta, sizeof(T), cudaMemcpyDeviceToDevice));

      // w := -tau_j * w (single scalar), then T(0:j, j) := T(0:j, 0:j) * w.
      common::gpu::scale(t_col, t_col, j, -local_tau[j]);
      common::gpu::matVecMult(Tmat, t_col, workspace, j, j, panelWidth);
      CUDACHECK(cudaMemcpy(t_col, workspace, static_cast<uint64_t>(j) * sizeof(T),
                           cudaMemcpyDeviceToDevice));
    }

    // Diagonal: T(j, j) = tau_j.
    CUDACHECK(cudaMemcpy(t_col + j, tau + j, sizeof(T), cudaMemcpyDeviceToDevice));
  }

  CUDACHECK(cudaFree(tmp_beta));
  delete[] local_tau;
}

template <typename T>
void blockTrailingUpdate(T* C, const T* V, const T* Tmat, uint32_t M, uint32_t N,
                         uint32_t panelWidth, T* W_workspace1, T* W_workspace2, T* C_workspace) {
  T* W1 = W_workspace1;
  T* W2 = W_workspace2;
  // Step 1: W = V^T C
  common::gpu::matMul<T, true, false>(V, C, W1, panelWidth, M, N, M, M, M);
  // Step 2: W = T^T W
  common::gpu::matMul<T, true, false>(Tmat, W1, W2, panelWidth, panelWidth, N, panelWidth, M, M);
  // Step 3: C_workspace = V W
  common::gpu::matMul<T, false, false>(V, W2, C_workspace, M, panelWidth, N, M, M, M);
  // Step 4: C -= C_workspace
  common::gpu::subtract<T>(C, C_workspace, C, static_cast<uint64_t>(M) * N);
}

template <typename T>
void blockedFinalMatrix(const T* A, T* Q, T* R, uint32_t M, uint32_t N,
                        BlockedHouseholderWorkspace<T>&& workspace) {
  T* tau = workspace.tauWorkspace;
  T* Tmat = workspace.TWorkspace;
  T* W = workspace.WWorkspace;
  T* C = workspace.CWorkspace1;
  T* C_tmp = workspace.CWorkspace2;

  // Copy the current panel from A to V
  CUDACHECK(cudaMemcpy(C, A, sizeof(T) * M * N, cudaMemcpyDeviceToDevice));
  for (uint32_t k = 0; k < N; k += workspace.panelWidth) {
    uint32_t panelWidth = std::min(workspace.panelWidth, N - k);
    uint32_t panelHeight = M - k;

    T* V = C + static_cast<uint64_t>(k) * M + k; // current panel in C, starting at row k, column k.

    // Step 1: Factor the panel in place.
    _detail::unblockedPanel(V, tau, M, panelWidth, panelHeight);

    // Step 2: Build the T matrix
    _detail::buildTMatrix(V, tau, Tmat, panelHeight, panelWidth, workspace.TwWorkspace);

    // Step 3: Update the trailing matrix
    if (k + panelWidth < N) {
      T* C = const_cast<T*>(A) + static_cast<uint64_t>(k) * M + k + panelWidth; // Trailing matrix
      _detail::blockTrailingUpdate(C, V, Tmat, panelHeight, N - k - panelWidth, panelWidth, W,
                                   C_tmp);
    }
  }
}

template void singleReflector<float>(float*, float*, uint32_t);
template void singleReflector<double>(double*, double*, uint32_t);
template void unblockedPanel<float>(float*, float*, uint32_t, uint32_t, uint32_t);
template void unblockedPanel<double>(double*, double*, uint32_t, uint32_t, uint32_t);
template __global__ void applyReflectorKernel<float>(float*, const float*, const float*, uint32_t,
                                                     uint32_t, uint32_t);
template __global__ void applyReflectorKernel<double>(double*, const double*, const double*,
                                                      uint32_t, uint32_t, uint32_t);
template void buildTMatrix<float>(float*, float*, float*, uint32_t, uint32_t, float*);
template void buildTMatrix<double>(double*, double*, double*, uint32_t, uint32_t, double*);
template void blockTrailingUpdate<float>(float*, const float*, const float*, uint32_t, uint32_t,
                                         uint32_t, float*, float*, float*);
template void blockTrailingUpdate<double>(double*, const double*, const double*, uint32_t, uint32_t,
                                          uint32_t, double*, double*, double*);

} // namespace qr::gpu::_detail

namespace qr::gpu {

template <typename T>
void blockedHouseHolder(const T* A, T* Q, T* R, uint32_t M, uint32_t N,
                        BlockedHouseholderWorkspace<T>&& workspace) {}

} // namespace qr::gpu
