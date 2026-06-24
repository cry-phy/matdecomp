#include "common/gpuLinalg.cuh"
#include "common/gpuUtils.cuh"
#include "qr/gpuGramSchmidt.cuh"

namespace {} // namespace

namespace qr::gpu {

template <typename T>
void GramSchmidt(const T* A, T* Q, T* R, uint32_t M, uint32_t N, T* workspace) {
  T* d_normSq;
  cudaMalloc(&d_normSq, sizeof(T));
  cudaMemcpy(Q, A, M * N * sizeof(T), cudaMemcpyDeviceToDevice);

  cudaMemset(R, 0, static_cast<size_t>(N) * N * sizeof(T));

  common::gpu::normalize<T>(Q, Q, M, d_normSq);
  common::gpu::sqrtKernel<T><<<1, 1>>>(d_normSq, &R[0], 1);

  for (uint32_t j = 1; j < N; j++) {
    T* r = &R[static_cast<size_t>(j) * N];

    common::gpu::matVecMult<T, true>(Q, &A[j * M], r, j, M, M);
    common::gpu::matVecMult<T, false>(Q, r, workspace, M, j, M);
    common::gpu::subtract<T>(&A[j * M], workspace, &Q[j * M], M);
    common::gpu::normalize<T>(&Q[j * M], &Q[j * M], M, d_normSq);
    common::gpu::sqrtKernel<T><<<1, 1>>>(d_normSq, &r[j], 1);
  }

  cudaFree(d_normSq);
}

template void GramSchmidt<float>(const float*, float*, float*, uint32_t, uint32_t, float*);

} // namespace qr::gpu
