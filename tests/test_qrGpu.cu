#include "common/cpuLinalg.hpp"
#include "common/cpuUtils.hpp"
#include "common/gpuUtils.cuh"
#include "qr/gpuGramSchmidt.cuh"
#include "qrTestData.hpp"
#include <cstdlib>
#include <cuda_runtime.h>
#include <limits>

// ── manufactured tests (known Q and R) ───────────────────────────────────────

using QRTestFloat = QRTest<float>;

INSTANTIATE_TEST_SUITE_P(QRTests, QRTestFloat, qrFloatManTestCases(),
                         [](const ::testing::TestParamInfo<QRTestFloat::ParamType>& info) {
                           return info.param.name;
                         });

TEST_P(QRTestFloat, GpuGramSchmidt) {
  const auto& tc = GetParam();
  float *d_A, *d_Q, *d_R, *d_workspace;
  CUDACHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_Q, M * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_R, N * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_workspace, M * sizeof(float)));
  CUDACHECK(cudaMemcpy(d_A, tc.A, M * N * sizeof(float), cudaMemcpyHostToDevice));

  qr::gpu::GramSchmidt(d_A, d_Q, d_R, M, N, d_workspace);
  CUDACHECK(cudaDeviceSynchronize());

  float tol = static_cast<float>(tc.kappa) * static_cast<float>(N) *
              std::numeric_limits<float>::epsilon() * 100.0f;

  float* h_Q = common::cpu::allocBuffer<float>(M * N);
  float* h_R = common::cpu::allocBuffer<float>(N * N);
  CUDACHECK(cudaMemcpy(h_Q, d_Q, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(h_R, d_R, N * N * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < M * N; ++i)
    EXPECT_NEAR(h_Q[i], tc.expectedQ[i], tol) << "Q mismatch at index " << i;
  for (size_t i = 0; i < N * N; ++i)
    EXPECT_NEAR(h_R[i], tc.expectedR[i], tol) << "R mismatch at index " << i;

  common::cpu::freeBuffer(h_Q);
  common::cpu::freeBuffer(h_R);
  cudaFree(d_A);
  cudaFree(d_Q);
  cudaFree(d_R);
  cudaFree(d_workspace);
}

// ── property-based tests (random A, check QR=A, Q orthonormal, R upper tri) ──

class QRGpuReversalTest : public ::testing::TestWithParam<QRReversalCase> {
protected:
  void SetUp() override {
    const auto& tc = GetParam();
    M = tc.M;
    N = tc.N;
    h_A = new float[M * N];
    h_Q = new float[M * N];
    h_R = new float[N * N];

    std::srand(tc.seed);
    for (size_t i = 0; i < M * N; i++)
      h_A[i] = 2.0f * (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) - 1.0f;

    CUDACHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_Q, M * N * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_R, N * N * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_workspace, M * sizeof(float)));
    CUDACHECK(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
  }

  void TearDown() override {
    delete[] h_A;
    delete[] h_Q;
    delete[] h_R;
    cudaFree(d_A);
    cudaFree(d_Q);
    cudaFree(d_R);
    cudaFree(d_workspace);
  }

  void copyResultsToHost() {
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaMemcpy(h_Q, d_Q, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(h_R, d_R, N * N * sizeof(float), cudaMemcpyDeviceToHost));
  }

  float tol() const {
    return static_cast<float>(N) * static_cast<float>(N) *
           std::numeric_limits<float>::epsilon() * 1000.0f;
  }

  float *h_A, *h_Q, *h_R;
  float *d_A, *d_Q, *d_R, *d_workspace;
  size_t M, N;
};

INSTANTIATE_TEST_SUITE_P(QRGpuReversalTests, QRGpuReversalTest, qrFloatReversalTestCases(),
                         [](const ::testing::TestParamInfo<QRGpuReversalTest::ParamType>& info) {
                           return info.param.name;
                         });

TEST_P(QRGpuReversalTest, GpuGramSchmidt) {
  qr::gpu::GramSchmidt(d_A, d_Q, d_R, M, N, d_workspace);
  copyResultsToHost();
  float t = tol();

  EXPECT_NEAR(common::cpu::orthogonalityError(h_Q, M, N), 0.0f, t);

  for (size_t j = 0; j < N; j++)
    for (size_t i = j + 1; i < N; i++)
      EXPECT_NEAR(h_R[i + j * N], 0.0f, t) << "R not upper triangular at [" << i << "," << j << "]";

  float* QR = new float[M * N];
  common::cpu::matMul(h_Q, M, N, h_R, N, N, QR);
  for (size_t i = 0; i < M * N; i++)
    EXPECT_NEAR(QR[i], h_A[i], t) << "QR != A at index " << i;
  delete[] QR;
}
