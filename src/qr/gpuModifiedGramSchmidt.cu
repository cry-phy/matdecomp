#include "common/gpuLinalg.cuh"
#include "common/gpuUtils.cuh"
#include "qr/gpuModifiedGramSchmidt.cuh"

namespace qr::gpu {

template <typename T>
void modifiedGramSchmidt(const T* A, T* Q, T* R, uint32_t M, uint32_t N, T* workspace) {
  T* d_normSq;
  cudaMalloc(&d_normSq, sizeof(T));
  cudaMemcpy(Q, A, M * N * sizeof(T), cudaMemcpyDeviceToDevice);
  cudaMemset(R, 0, static_cast<size_t>(N) * N * sizeof(T));

  for (uint32_t j = 0; j < N; j++) {
    T* qj = &Q[static_cast<size_t>(j) * M];
    common::gpu::normalize<T>(qj, qj, M, d_normSq);                   // q_j /= ||q_j||
    common::gpu::sqrtKernel<T><<<1, 1>>>(d_normSq, &R[j + j * N], 1); // R[j, j] = ||q_j||

    uint32_t t = N - j - 1;
    if (t == 0)
      continue;
    T* trailing = &Q[static_cast<size_t>(j + 1) * M];

    common::gpu::matVecMult<T, true>(trailing, qj, workspace, t, M, M);
    common::gpu::scatterStrided<T>(workspace, &R[j + (j + 1) * N], t, N);
    common::gpu::ger<T>(qj, workspace, trailing, M, t, M);
  }

  cudaFree(d_normSq);
}

template void modifiedGramSchmidt<float>(const float*, float*, float*, uint32_t, uint32_t, float*);

} // namespace qr::gpu
