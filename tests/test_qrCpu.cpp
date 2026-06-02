#include "qrTestData.hpp"
#include "qr/cpuGramSchmidt.hpp"
#include <limits>

using QRTestFloat = QRTest<float>;

INSTANTIATE_TEST_SUITE_P(
    QRTests, QRTestFloat, qrFloatTestCases(),
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
