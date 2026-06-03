#include "common/cpuUtils.hpp"
#include "qr/cpuGramSchmidt.hpp"
#include "qr/cpuModifiedGramSchmidt.hpp"
#include "qrTestData.hpp"
#include <cstdlib>
#include <limits>

// ── manufactured tests (known Q and R) ───────────────────────────────────────

using QRTestFloat = QRTest<float>;

INSTANTIATE_TEST_SUITE_P(QRTests, QRTestFloat, qrFloatManTestCases(),
                         [](const ::testing::TestParamInfo<QRTestFloat::ParamType>& info) {
                           return info.param.name;
                         });

TEST_P(QRTestFloat, CpuGramSchmidt) {
  const auto& tc = GetParam();
  qr::cpu::gramSchmidt(tc.A, Q, R, M, N);

  float tol = static_cast<float>(tc.kappa) * static_cast<float>(N) *
              std::numeric_limits<float>::epsilon() * 100.0f;

  for (size_t i = 0; i < M * N; ++i)
    EXPECT_NEAR(Q[i], tc.expectedQ[i], tol) << "Q mismatch at index " << i;
  for (size_t i = 0; i < N * N; ++i)
    EXPECT_NEAR(R[i], tc.expectedR[i], tol) << "R mismatch at index " << i;
}

TEST_P(QRTestFloat, CpuModifiedGramSchmidt) {
  const auto& tc = GetParam();
  qr::cpu::modifiedGramSchmidt(tc.A, Q, R, M, N);

  float tol = static_cast<float>(tc.kappa) * static_cast<float>(N) *
              std::numeric_limits<float>::epsilon() * 100.0f;

  for (size_t i = 0; i < M * N; ++i)
    EXPECT_NEAR(Q[i], tc.expectedQ[i], tol) << "Q mismatch at index " << i;
  for (size_t i = 0; i < N * N; ++i)
    EXPECT_NEAR(R[i], tc.expectedR[i], tol) << "R mismatch at index " << i;
}

// ── property-based tests (random A, check QR=A, Q orthonormal, R upper tri) ──

INSTANTIATE_TEST_SUITE_P(QRReversalTests, QRReversalTest, qrFloatReversalTestCases(),
                         [](const ::testing::TestParamInfo<QRReversalTest::ParamType>& info) {
                           return info.param.name;
                         });

TEST_P(QRReversalTest, CpuGramSchmidt) {
  qr::cpu::gramSchmidt(A, Q, R, M, N);
  float t = tol();

  EXPECT_NEAR(common::cpu::orthogonalityError(Q, M, N), 0.0f, t);

  for (size_t j = 0; j < N; j++)
    for (size_t i = j + 1; i < N; i++)
      EXPECT_NEAR(R[i + j * N], 0.0f, t) << "R not upper triangular at [" << i << "," << j << "]";

  float* QR = common::cpu::allocBuffer<float>(M * N);
  common::cpu::matMul(Q, M, N, R, N, N, QR);
  for (size_t i = 0; i < M * N; i++)
    EXPECT_NEAR(QR[i], A[i], t) << "QR != A at index " << i;
  common::cpu::freeBuffer(QR);
}

TEST_P(QRReversalTest, CpuModifiedGramSchmidt) {
  qr::cpu::modifiedGramSchmidt(A, Q, R, M, N);
  float t = tol();

  EXPECT_NEAR(common::cpu::orthogonalityError(Q, M, N), 0.0f, t);

  for (size_t j = 0; j < N; j++)
    for (size_t i = j + 1; i < N; i++)
      EXPECT_NEAR(R[i + j * N], 0.0f, t) << "R not upper triangular at [" << i << "," << j << "]";

  float* QR = common::cpu::allocBuffer<float>(M * N);
  common::cpu::matMul(Q, M, N, R, N, N, QR);
  for (size_t i = 0; i < M * N; i++)
    EXPECT_NEAR(QR[i], A[i], t) << "QR != A at index " << i;
  common::cpu::freeBuffer(QR);
}
