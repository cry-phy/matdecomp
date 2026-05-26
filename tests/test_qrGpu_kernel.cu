// GPU Gram-Schmidt kernel and host-callable launcher.
// Kept in a separate .cu (nvcc) from the GTest machinery (g++)
#include "common/gpuLinalg.cuh"
#include "common/gpuUtils.cuh"
#include "qr/gpuGramSchmidt.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

static constexpr uint32_t kBlockSize = MAX_BLOCK_SIZE;

__global__ __launch_bounds__(MAX_BLOCK_SIZE) static void gpuGramSchmidtKernel(
    const float* A, float* Q, float* R, size_t M, size_t N, float* workspace) {
  qr::gpuGramSchmidt(A, Q, R, M, N, workspace);
}

void launchGpuGramSchmidt(const float* d_A, float* d_Q, float* d_R, size_t M, size_t N,
                          float* d_workspace) {
  uint32_t grid = (static_cast<uint32_t>(N) + kBlockSize - 1) / kBlockSize;
  gpuGramSchmidtKernel<<<grid, kBlockSize>>>(d_A, d_Q, d_R, M, N, d_workspace);
  CUDACHECK(cudaGetLastError());
}
