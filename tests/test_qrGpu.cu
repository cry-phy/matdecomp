#include "common/cpuLinalg.hpp"
#include "common/cpuUtils.hpp"
#include "common/gpuUtils.cuh"
#include "qr/gpuBlockedHouseholder.cuh"
#include "qr/gpuGramSchmidt.cuh"
#include "qr/gpuModifiedGramSchmidt.cuh"
#include "qr/gpuTSQR.cuh"
#include "qr/gpuhouseholder.cuh"
#include "qrTestData.hpp"
#include "gtest/gtest.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
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

TEST_P(QRTestFloat, GpuModifiedGramSchmidt) {
  const auto& tc = GetParam();
  float *d_A, *d_Q, *d_R, *d_workspace;
  CUDACHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_Q, M * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_R, N * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_workspace, M * sizeof(float)));
  CUDACHECK(cudaMemcpy(d_A, tc.A, M * N * sizeof(float), cudaMemcpyHostToDevice));

  qr::gpu::modifiedGramSchmidt(d_A, d_Q, d_R, M, N, d_workspace);
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

TEST_P(QRTestFloat, GpuHouseholder) {
  const auto& tc = GetParam();
  float *d_A, *d_Q, *d_R, *d_workspace;
  CUDACHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_Q, M * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_R, N * N * sizeof(float)));

  // Note: Adjust workspace size here if Householder needs more than M elements
  CUDACHECK(cudaMalloc(&d_workspace, M * sizeof(float)));
  CUDACHECK(cudaMemcpy(d_A, tc.A, M * N * sizeof(float), cudaMemcpyHostToDevice));

  qr::gpu::Householder(d_A, d_Q, d_R, M, N, d_workspace);
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

TEST_P(QRTestFloat, GpuTSQR) {
  const auto& tc = GetParam();
  float *d_A, *d_R;
  CUDACHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_R, N * N * sizeof(float)));
  CUDACHECK(cudaMemcpy(d_A, tc.A, M * N * sizeof(float), cudaMemcpyHostToDevice));

  // Run TSQR (only computes R)
  qr::gpu::TSQR(d_A, d_R, M, N);
  CUDACHECK(cudaDeviceSynchronize());

  float tol = static_cast<float>(tc.kappa) * static_cast<float>(N) *
              std::numeric_limits<float>::epsilon() * 100.0f;

  float* h_R = common::cpu::allocBuffer<float>(N * N);
  CUDACHECK(cudaMemcpy(h_R, d_R, N * N * sizeof(float), cudaMemcpyDeviceToHost));

  // Verify R matches expected manufacturing data
  for (size_t i = 0; i < N * N; ++i) {
    EXPECT_NEAR(h_R[i], tc.expectedR[i], tol) << "R mismatch at index " << i;
  }

  common::cpu::freeBuffer(h_R);
  cudaFree(d_A);
  cudaFree(d_R);
}

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
    return static_cast<float>(N) * static_cast<float>(N) * std::numeric_limits<float>::epsilon() *
           1000.0f;
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

TEST_P(QRGpuReversalTest, GpuHouseholder) {
  qr::gpu::Householder(d_A, d_Q, d_R, M, N, d_workspace);
  copyResultsToHost();
  float t = tol();

  // Verify Q orthogonality
  EXPECT_NEAR(common::cpu::orthogonalityError(h_Q, M, N), 0.0f, t);

  // Verify R upper triangularity
  for (size_t j = 0; j < N; j++)
    for (size_t i = j + 1; i < N; i++)
      EXPECT_NEAR(h_R[i + j * N], 0.0f, t) << "R not upper triangular at [" << i << "," << j << "]";

  // Verify Q * R == A
  float* QR = new float[M * N];
  common::cpu::matMul(h_Q, M, N, h_R, N, N, QR);
  for (size_t i = 0; i < M * N; i++)
    EXPECT_NEAR(QR[i], h_A[i], t) << "QR != A at index " << i;
  delete[] QR;
}

TEST_P(QRGpuReversalTest, GpuModifiedGramSchmidt) {
  qr::gpu::modifiedGramSchmidt(d_A, d_Q, d_R, M, N, d_workspace);
  copyResultsToHost();
  float t = tol();

  // Verify Q orthogonality
  EXPECT_NEAR(common::cpu::orthogonalityError(h_Q, M, N), 0.0f, t);

  // Verify R upper triangularity
  for (size_t j = 0; j < N; j++)
    for (size_t i = j + 1; i < N; i++)
      EXPECT_NEAR(h_R[i + j * N], 0.0f, t) << "R not upper triangular at [" << i << "," << j << "]";

  // Verify Q * R == A
  float* QR = new float[M * N];
  common::cpu::matMul(h_Q, M, N, h_R, N, N, QR);
  for (size_t i = 0; i < M * N; i++)
    EXPECT_NEAR(QR[i], h_A[i], t) << "QR != A at index " << i;
  delete[] QR;
}

TEST_P(QRGpuReversalTest, GpuBlockedHouseholder) {
  qr::gpu::BlockedHouseholderWorkspace<float> d_workspace;
  qr::gpu::mallocWorkspace(d_workspace, M, N, 32);
  qr::gpu::blockedHouseHolder(d_A, d_Q, d_R, M, N, d_workspace);
  copyResultsToHost();
  float t = tol();

  // Verify Q orthogonality
  EXPECT_NEAR(common::cpu::orthogonalityError(h_Q, M, N), 0.0f, t) << "Q is not orthogonal";

  // Verify R upper triangularity
  for (size_t j = 0; j < N; j++)
    for (size_t i = j + 1; i < N; i++)
      EXPECT_NEAR(h_R[i + j * N], 0.0f, t) << "R not upper triangular at [" << i << "," << j << "]";

  // Verify Q * R == A
  float* QR = new float[M * N];
  common::cpu::matMul(h_Q, M, N, h_R, N, N, QR);
  for (size_t i = 0; i < M * N; i++)
    EXPECT_NEAR(QR[i], h_A[i], t) << "QR != A at index " << i;
  delete[] QR;
  qr::gpu::freeWorkspace(d_workspace);
}

TEST_P(QRGpuReversalTest, GpuTSQR) {
  // Run TSQR
  qr::gpu::TSQR(d_A, d_R, M, N);
  CUDACHECK(cudaDeviceSynchronize());

  // Bring R back to host
  CUDACHECK(cudaMemcpy(h_R, d_R, N * N * sizeof(float), cudaMemcpyDeviceToHost));
  float t = tol();

  // 1. Verify R upper triangularity
  for (size_t j = 0; j < N; j++) {
    for (size_t i = j + 1; i < N; i++) {
      EXPECT_NEAR(h_R[i + j * N], 0.0f, t) << "R not upper triangular at [" << i << "," << j << "]";
    }
  }

  // 2. Verify A^T * A == R^T * R
  // Compute A^T * A (Result size: N x N)
  float* AtA = new float[N * N];
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < N; ++i) {
      float sum = 0.0f;
      for (size_t k = 0; k < M; ++k) {
        sum += h_A[k + i * M] * h_A[k + j * M];
      }
      AtA[i + j * N] = sum;
    }
  }

  // Compute R^T * R (Result size: N x N)
  float* RtR = new float[N * N];
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < N; ++i) {
      float sum = 0.0f;
      for (size_t k = 0; k < N; ++k) {
        sum += h_R[k + i * N] * h_R[k + j * N];
      }
      RtR[i + j * N] = sum;
    }
  }

  // Check mathematical equality between A^T*A and R^T*R
  for (size_t i = 0; i < N * N; i++) {
    EXPECT_NEAR(AtA[i], RtR[i], t * 10.0f) << "A^T * A != R^T * R at linear index " << i;
  }

  delete[] AtA;
  delete[] RtR;
}
