#include "common/gpuLinalg.cuh"
#include "common/gpuUtils.cuh"
#include "qr/gpuGramSchmidt.cuh"

namespace {

template <typename T> void gsComputeQ(const T* A, T* Q, uint32_t M, uint32_t N, T* workspace) {
  T* d_normSq;
  T* d_dot_ab;
  T* d_dot_bb;
  cudaMalloc(&d_normSq, sizeof(T));
  cudaMalloc(&d_dot_ab, sizeof(T));
  cudaMalloc(&d_dot_bb, sizeof(T));
  cudaMemcpy(Q, A, M * N * sizeof(T), cudaMemcpyDeviceToDevice);

  common::gpu::normalize<T>(Q, Q, M, d_normSq);

  for (uint32_t j = 1; j < N; j++) {
    for (uint32_t i = 0; i < j; i++) {
      common::gpu::project<T>(&A[j * M], &Q[i * M], workspace, M, d_dot_ab);
      common::gpu::subtract<T>(&Q[j * M], workspace, &Q[j * M], M);
    }
    common::gpu::normalize<T>(&Q[j * M], &Q[j * M], M, d_normSq);
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

template <typename T>
void GramSchmidtNaive(const T* A, T* Q, T* R, uint32_t M, uint32_t N, T* workspace) {
  gsComputeQ(A, Q, M, N, workspace);
  gsComputeR(A, Q, R, M, N);
}

template void GramSchmidtNaive<float>(const float*, float*, float*, uint32_t, uint32_t, float*);

} // namespace qr::gpu
