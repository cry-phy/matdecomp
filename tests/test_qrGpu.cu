#include "common/cpuBuffer.hpp"
#include "qr/gpuGramSchmidt.cuh"
#include "qrTestData.hpp"
#include <cuda_runtime.h>
#include <limits>

using QRTestFloat = QRTest<float>;

INSTANTIATE_TEST_SUITE_P(QRTests, QRTestFloat, qrFloatTestCases(),
                         [](const ::testing::TestParamInfo<QRTestFloat::ParamType>& info) {
                           return info.param.name;
                         });

TEST_P(QRTestFloat, GpuGramSchmidt) {
  const auto& tc = GetParam();
  float* d_A;
  float* d_Q;
  float* d_R;
  float* d_workspace;

  cudaMalloc(&d_A, M * N * sizeof(float));
  cudaMalloc(&d_Q, M * N * sizeof(float));
  cudaMalloc(&d_R, N * N * sizeof(float));
  cudaMalloc(&d_workspace, M * sizeof(float));

  cudaMemcpy(d_A, tc.A, M * N * sizeof(float), cudaMemcpyHostToDevice);

  qr::gpu::GramSchmidt(d_A, d_Q, d_R, M, N, d_workspace);
  cudaDeviceSynchronize();

  float tol = static_cast<float>(tc.kappa) * static_cast<float>(N) *
              std::numeric_limits<float>::epsilon() * 100.0f;

  float* h_Q = common::cpu::allocBuffer<float>(M * N);
  float* h_R = common::cpu::allocBuffer<float>(N * N);

  cudaMemcpy(h_Q, d_Q, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_R, d_R, N * N * sizeof(float), cudaMemcpyDeviceToHost);

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
