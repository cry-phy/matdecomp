#include "common/gpuLinalg.cuh"
#include "common/gpuUtils.cuh"
#include "qr/gpuModifiedGramSchmidt.cuh"

namespace {

template <typename T> void gsComputeQ(const T* A, T* Q, uint32_t M, uint32_t N, T* workspace) {
  T* d_normSq;
  T* d_dot_ab;
  T* d_dot_bb;
  cudaMalloc(&d_normSq, sizeof(T));
  cudaMalloc(&d_dot_ab, sizeof(T));
  cudaMalloc(&d_dot_bb, sizeof(T));
  cudaMemcpy(Q, A, M * N * sizeof(T), cudaMemcpyDeviceToDevice);

  for (uint32_t j = 0; j < N; j++) {
    common::gpu::normalize<T>(&Q[j * M], &Q[j * M], M, d_normSq);
    for (uint32_t i = j + 1; i < N; i++) {
      common::gpu::project<T>(&Q[i * M], &Q[j * M], workspace, M, d_dot_ab);
      common::gpu::subtract<T>(&Q[i * M], workspace, &Q[i * M], M);
    }
  }

  cudaFree(d_normSq);
  cudaFree(d_dot_ab);
  cudaFree(d_dot_bb);
}

template <typename T> void gsComputeR(const T* A, const T* Q, T* R, uint32_t M, uint32_t N) {
  cudaMemset(R, 0, N * N * sizeof(T));
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i <= j; ++i) {
      common::gpu::dotProduct<T>(&Q[i * M], &A[j * M], &R[i + j * N], M);
    }
  }
}

} // namespace

namespace qr::gpu {

template <typename T>
void modifiedGramSchmidtNaive(const T* A, T* Q, T* R, uint32_t M, uint32_t N, T* workspace) {
  gsComputeQ(A, Q, M, N, workspace);
  gsComputeR(A, Q, R, M, N);
}

template void modifiedGramSchmidtNaive<float>(const float*, float*, float*, uint32_t, uint32_t,
                                              float*);

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
