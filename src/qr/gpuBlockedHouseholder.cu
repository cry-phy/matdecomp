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
  // own R value.
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
  // Rows are strided across the block, so M_eff may exceed the 1024-thread cap.
  uint32_t col = blockIdx.x;

  if (col >= N)
    return;

  T contrib = static_cast<T>(0);
  for (uint32_t r = threadIdx.x; r < M_eff; r += blockDim.x) {
    T vv = (r == 0) ? static_cast<T>(1) : v_j[r];
    contrib += C[static_cast<uint64_t>(col) * stride + r] * vv;
  }

  T w_c = common::gpu::_blockReduce(contrib);

  const T scale = (*tau_j) * w_c;
  for (uint32_t r = threadIdx.x; r < M_eff; r += blockDim.x) {
    T vv = (r == 0) ? static_cast<T>(1) : v_j[r];
    C[static_cast<uint64_t>(col) * stride + r] -= scale * vv;
  }
}

/**
 * @brief Threads per block for applyReflectorKernel: enough to cover M_eff rows, warp-aligned so
 * _blockReduce never shuffles across absent lanes, and capped at the hardware limit. Rows beyond
 * the cap are handled by the kernel's strided loop.
 */
inline uint32_t reflectorBlockSize(uint32_t M_eff) {
  uint32_t warpAligned = ((M_eff + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  return std::min<uint32_t>(MAX_BLOCK_SIZE, warpAligned);
}

template <typename T>
void unblockedPanel(T* V, T* tau, uint32_t stride, uint32_t N, uint32_t M_eff) {
  T* local_tau = new T[N];
  for (uint32_t j = 0; j < N; ++j) {
    singleReflector(V + static_cast<uint64_t>(stride) * j + j, tau + j, M_eff - j);
    cudaMemcpy(local_tau, tau, N * sizeof(T), cudaMemcpyDeviceToHost);
    if (j < N - 1) {
      applyReflectorKernel<<<N - j - 1, reflectorBlockSize(M_eff - j)>>>(
          V + static_cast<uint64_t>(stride) * (j + 1) + j,
          &V[static_cast<uint64_t>(stride) * j + j], tau + j, stride, N - j - 1, M_eff - j);
      cudaDeviceSynchronize();
    }
  }
  delete[] local_tau;
}

template <typename T>
void buildTMatrix(T* V, T* tau, T* Tmat, uint32_t M, uint32_t N, uint32_t ldv) {
  // TODO merge into single larft kernel to avoid host-device round trips.
  T* workspace;
  CUDACHECK(cudaMalloc(&workspace, static_cast<uint64_t>(N) * sizeof(T)));
  T* local_tau = new T[N];
  CUDACHECK(cudaMemcpy(local_tau, tau, N * sizeof(T), cudaMemcpyDeviceToHost));

  // Cache for V's diagonal since its needed for v_j implicit unit diagonal
  T* tmp_beta;
  CUDACHECK(cudaMalloc(&tmp_beta, sizeof(T)));

  // Lower triangle must be zero so the triangular matvec can sum over the full j x j block.
  common::gpu::zeroBuffer(Tmat, static_cast<uint64_t>(N) * N);

  for (uint32_t j = 0; j < N; ++j) {
    T* t_col = Tmat + static_cast<uint64_t>(j) * N;
    if (std::abs(local_tau[j]) <= static_cast<T>(1e-8))
      continue;

    if (j > 0) {
      T* v_diag = V + static_cast<uint64_t>(j) * ldv + j;
      CUDACHECK(cudaMemcpy(tmp_beta, v_diag, sizeof(T), cudaMemcpyDeviceToDevice));
      common::gpu::fillBuffer(v_diag, 1, static_cast<T>(1));
      for (uint32_t k = 0; k < j; ++k) {
        T* d_vk = V + static_cast<uint64_t>(k) * ldv + j;
        common::gpu::dotProduct(d_vk, v_diag, t_col + k, M - j);
      }
      CUDACHECK(cudaMemcpy(v_diag, tmp_beta, sizeof(T), cudaMemcpyDeviceToDevice));

      common::gpu::scale(t_col, t_col, j, -local_tau[j]);
      common::gpu::matVecMult(Tmat, t_col, workspace, j, j, N);
      CUDACHECK(cudaMemcpy(t_col, workspace, static_cast<uint64_t>(j) * sizeof(T),
                           cudaMemcpyDeviceToDevice));
    }

    // Diagonal: T(j, j) = tau_j.
    CUDACHECK(cudaMemcpy(t_col + j, tau + j, sizeof(T), cudaMemcpyDeviceToDevice));
  }

  CUDACHECK(cudaFree(tmp_beta));
  CUDACHECK(cudaFree(workspace));
  delete[] local_tau;
}

template <typename T>
void blockTrailingUpdate(T* C, const T* V, const T* Tmat, uint32_t M, uint32_t N,
                         uint32_t panelWidth, T* W_workspace1, T* W_workspace2, T* C_workspace,
                         uint32_t ldp) {
  T* W1 = W_workspace1;
  T* W2 = W_workspace2;
  // Step 1: W = V^T C
  common::gpu::matMul<T, true, false>(V, C, W1, panelWidth, M, N, ldp, ldp, ldp);
  // Step 2: W = T^T W
  common::gpu::matMul<T, true, false>(Tmat, W1, W2, panelWidth, panelWidth, N, panelWidth, ldp,
                                      ldp);
  // Step 3: C_workspace = V W
  common::gpu::matMul<T, false, false>(V, W2, C_workspace, M, panelWidth, N, ldp, ldp, ldp);
  // Step 4: C -= C_workspace  (element-wise over the M x N trailing block, ld = ldp)
  common::gpu::subtractMatrix<T>(C, C_workspace, C, M, N, ldp, ldp, ldp);
}

template <typename T>
void blockedOrchestrator(const T* A, T* reflectors, T* R, uint32_t M, uint32_t N,
                         BlockedHouseholderWorkspace<T>&& workspace) {

  common::gpu::zeroBuffer(R, static_cast<uint64_t>(N) * N);
  T* C_trail = workspace.C_trail;
  T* W1 = workspace.W1;
  T* W2 = workspace.W2;
  T* tau = workspace.tau;
  T* Tmat = workspace.Tmat;
  // Copy the current panel from A to workspace.res for in-place updates.
  CUDACHECK(cudaMemcpy(reflectors, A, sizeof(T) * M * N, cudaMemcpyDeviceToDevice));
  for (uint32_t k = 0; k < N; k += workspace.panelWidth) {
    uint32_t panelWidth = std::min(workspace.panelWidth, N - k);

    T* V = reflectors + static_cast<uint64_t>(k) * M + k;

    unblockedPanel(V, tau + k, M, panelWidth, M - k);

    buildTMatrix(V, tau + k, Tmat, M - k, panelWidth, M);
    cudaDeviceSynchronize();

    // We will build R along the way, so it must not be stashed before the GEMM calls in trailing
    // update. Also prepare V to be only the reflectors below the diagonal, with diagonal set to
    // 1!
    for (uint32_t j = k; j < k + panelWidth; ++j) {
      CUDACHECK(cudaMemcpy(R + static_cast<uint64_t>(j) * N,
                           reflectors + static_cast<uint64_t>(j) * M, sizeof(T) * (j + 1),
                           cudaMemcpyDeviceToDevice));
    }

    for (uint32_t c = 1; c < panelWidth; ++c) {
      common::gpu::zeroBuffer(V + static_cast<uint64_t>(c) * M, c);
    }
    for (uint32_t c = 0; c < panelWidth; ++c) {
      common::gpu::fillBuffer(V + static_cast<uint64_t>(c) * M + c, 1, static_cast<T>(1));
    }
    cudaDeviceSynchronize();

    if (k + panelWidth < N) {
      T* remaining = reflectors + static_cast<uint64_t>(k + panelWidth) * M + k;
      blockTrailingUpdate(remaining, V, Tmat, M - k, N - k - panelWidth, panelWidth, W1, W2,
                          C_trail, M);
    }
  }
}

template <typename T> void constructR(const T* finalMatrix, T* R, uint32_t M, uint32_t N) {
  // Copy the upper triangular part of finalMatrix to R
  for (uint32_t j = 0; j < N; ++j) {
    CUDACHECK(cudaMemcpy(R + static_cast<uint64_t>(j) * N,
                         finalMatrix + static_cast<uint64_t>(j) * M, sizeof(T) * (j + 1),
                         cudaMemcpyDeviceToDevice));
    common::gpu::zeroBuffer(R + static_cast<uint64_t>(j) * N + j + 1, N - j - 1);
  }
}

template <typename T> void constructQ(const T* finalMatrix, T* tau, T* Q, uint32_t M, uint32_t N) {
  // Initialize Q as the identity matrix
  common::gpu::identityMatrixSlice(Q, M, N);

  // Apply the reflectors in reverse order to construct Q. One block per column of
  // Q — Q is M x N, so the grid and the column count are N, not M.
  for (int j = N - 1; j >= 0; --j) {
    qr::gpu::_detail::applyReflectorKernel<<<N, reflectorBlockSize(M - j)>>>(
        Q + j, finalMatrix + j * M + j, tau + j, M, N, M - j);
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
template void buildTMatrix<float>(float*, float*, float*, uint32_t, uint32_t, uint32_t);
template void buildTMatrix<double>(double*, double*, double*, uint32_t, uint32_t, uint32_t);
template void blockTrailingUpdate<float>(float*, const float*, const float*, uint32_t, uint32_t,
                                         uint32_t, float*, float*, float*, uint32_t);
template void blockTrailingUpdate<double>(double*, const double*, const double*, uint32_t, uint32_t,
                                          uint32_t, double*, double*, double*, uint32_t);
template void blockedOrchestrator<float>(const float*, float*, float*, uint32_t, uint32_t,
                                         BlockedHouseholderWorkspace<float>&&);
template void blockedOrchestrator<double>(const double*, double*, double*, uint32_t, uint32_t,
                                          BlockedHouseholderWorkspace<double>&&);
template void constructR<float>(const float*, float*, uint32_t, uint32_t);
template void constructR<double>(const double*, double*, uint32_t, uint32_t);
template void constructQ<float>(const float*, float*, float*, uint32_t, uint32_t);
template void constructQ<double>(const double*, double*, double*, uint32_t, uint32_t);

} // namespace qr::gpu::_detail

namespace qr::gpu {

template <typename T>
cudaError_t mallocWorkspace(BlockedHouseholderWorkspace<T>& workspace, uint32_t M, uint32_t N,
                            uint32_t panelWidth) {
  workspace.panelWidth = panelWidth;
  workspace.res = workspace.tau = workspace.Tmat = nullptr;
  workspace.W1 = workspace.W2 = workspace.C_trail = nullptr;

  const uint64_t mn = static_cast<uint64_t>(M) * N;
  cudaError_t err;
  // Don't hard-exit on OOM: free any partial allocations and report, so callers can
  // skip a too-large size and keep going.
  if ((err = cudaMalloc(&workspace.res, mn * sizeof(T))) != cudaSuccess ||
      (err = cudaMalloc(&workspace.tau, static_cast<uint64_t>(N) * sizeof(T))) != cudaSuccess ||
      (err = cudaMalloc(&workspace.Tmat,
                        static_cast<uint64_t>(panelWidth) * panelWidth * sizeof(T))) !=
          cudaSuccess ||
      (err = cudaMalloc(&workspace.W1, mn * sizeof(T))) != cudaSuccess ||
      (err = cudaMalloc(&workspace.W2, mn * sizeof(T))) != cudaSuccess ||
      (err = cudaMalloc(&workspace.C_trail, mn * sizeof(T))) != cudaSuccess) {
    freeWorkspace(workspace);
    return err;
  }
  return cudaSuccess;
}

template <typename T> void freeWorkspace(BlockedHouseholderWorkspace<T>& workspace) {
  CUDACHECK(cudaFree(workspace.res));
  CUDACHECK(cudaFree(workspace.tau));
  CUDACHECK(cudaFree(workspace.Tmat));
  CUDACHECK(cudaFree(workspace.W1));
  CUDACHECK(cudaFree(workspace.W2));
  CUDACHECK(cudaFree(workspace.C_trail));
}

template <typename T>
void blockedHouseHolder(const T* A, T* Q, T* R, uint32_t M, uint32_t N,
                        BlockedHouseholderWorkspace<T>& workspace) {
  _detail::blockedOrchestrator(A, workspace.res, R, M, N, std::move(workspace));
  _detail::constructQ(workspace.res, workspace.tau, Q, M, N);
}

template void blockedHouseHolder<float>(const float*, float*, float*, uint32_t, uint32_t,
                                        BlockedHouseholderWorkspace<float>&);
template void blockedHouseHolder<double>(const double*, double*, double*, uint32_t, uint32_t,
                                         BlockedHouseholderWorkspace<double>&);
template cudaError_t mallocWorkspace<float>(BlockedHouseholderWorkspace<float>&, uint32_t, uint32_t,
                                            uint32_t);
template cudaError_t mallocWorkspace<double>(BlockedHouseholderWorkspace<double>&, uint32_t,
                                             uint32_t, uint32_t);
template void freeWorkspace(BlockedHouseholderWorkspace<float>&);
template void freeWorkspace(BlockedHouseholderWorkspace<double>&);
} // namespace qr::gpu
