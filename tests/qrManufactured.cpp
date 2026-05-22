#include "common/cpuBuffer.hpp"
#include "common/cpuLinalg.hpp"
#include "qr/cpuGramSchmidt.hpp"
#include <gtest/gtest.h>
#include <limits>

template <typename T> struct QRTestCase {
  const char *name;
  const T *A;
  const T *expectedQ;
  const T *expectedR;
  size_t M;
  size_t N;
  double kappa;
};

template <typename T>
class QRTest : public ::testing::TestWithParam<QRTestCase<T>> {
protected:
  void SetUp() override {
    const auto &tc = this->GetParam();
    M = tc.M;
    N = tc.N;
    Q = common::allocBuffer<T>(M * N);
    R = common::allocBuffer<T>(N * N);
  }

  void TearDown() override {
    common::freeBuffer(Q);
    common::freeBuffer(R);
  }

  T *Q;
  T *R;
  size_t M;
  size_t N;
};

template <typename T> void PrintTo(const QRTestCase<T> &tc, std::ostream *os) {
  *os << tc.name;
}

// ------------------------------------------------------------------
// All matrices stored column-major (column j starts at offset j*M).
// ------------------------------------------------------------------
//
// Case 1: 2x2 identity — trivial, Q=I, R=I, kappa=1
// ------------------------------------------------------------------
//
// Case 2: 2x2 well-conditioned (kappa ~ 2)
//
//   Q = [[1/√2, -1/√2],    R = [[√2,   1/√2],
//        [1/√2,  1/√2]]         [ 0,   1/√2]]
//
//   A = Q·R = [[1, 0],
//              [1, 1]]
//
// ------------------------------------------------------------------
//
// Case 3: 3x3 well-conditioned (kappa ~ 2)
//
//   Q = (1/3)*[[ 2, -2,  1],   R = [[3, 1, 2],
//              [ 1,  2,  2],        [0, 6, 1],
//              [ 2,  1, -2]]        [0, 0, 4]]
//
//   A = Q·R = [[ 2,      -10/3,  2],
//              [ 1,       13/3,  4],
//              [ 2,        8/3, -1]]
//
// ------------------------------------------------------------------
//
// Case 4: 4x4 identity
//
// ------------------------------------------------------------------
//
// Case 5: 2x2 ill-conditioned (kappa=100) — stress test
//
//   Q = [[1/√2, -1/√2],    R = [[10, 5  ],
//        [1/√2,  1/√2]]         [ 0, 0.1]]
//
//   A = Q·R = [[10/√2,  4.9/√2],
//              [10/√2,  5.1/√2]]
//
// ------------------------------------------------------------------

// Column-major: identity col0=[1,0], col1=[0,1]
static float qr_2x2_identity_A[] = {1.0f, 0.0f, 0.0f, 1.0f};
static float qr_2x2_identity_Q[] = {1.0f, 0.0f, 0.0f, 1.0f};
static float qr_2x2_identity_R[] = {1.0f, 0.0f, 0.0f, 1.0f};

// A col-maj: col0=[1,1], col1=[0,1]
static float qr_2x2_A[] = {1.0f, 1.0f, 0.0f, 1.0f};
// Q col-maj: col0=[1/√2,1/√2], col1=[-1/√2,1/√2]
static float qr_2x2_Q[] = {0.7071068f, 0.7071068f, -0.7071068f, 0.7071068f};
// R col-maj: col0=[√2,0], col1=[1/√2,1/√2]
static float qr_2x2_R[] = {1.4142136f, 0.0f, 0.7071068f, 0.7071068f};

// A col-maj: col0=[2,1,2], col1=[-10/3,13/3,8/3], col2=[2,4,-1]
static float qr_3x3_A[] = {2.0f,       1.0f, 2.0f, -3.3333333f, 4.3333333f,
                           2.6666667f, 2.0f, 4.0f, -1.0f};
// Q col-maj: col0=[2/3,1/3,2/3], col1=[-2/3,2/3,1/3], col2=[1/3,2/3,-2/3]
static float qr_3x3_Q[] = {0.6666667f,  0.3333333f, 0.6666667f,
                           -0.6666667f, 0.6666667f, 0.3333333f,
                           0.3333333f,  0.6666667f, -0.6666667f};
// R col-maj: col0=[3,0,0], col1=[1,6,0], col2=[2,1,4]
static float qr_3x3_R[] = {3.0f, 0.0f, 0.0f, 1.0f, 6.0f,
                           0.0f, 2.0f, 1.0f, 4.0f};

static float qr_4x4_identity_A[] = {1, 0, 0, 0, 0, 1, 0, 0,
                                    0, 0, 1, 0, 0, 0, 0, 1};
static float qr_4x4_identity_Q[] = {1, 0, 0, 0, 0, 1, 0, 0,
                                    0, 0, 1, 0, 0, 0, 0, 1};
static float qr_4x4_identity_R[] = {1, 0, 0, 0, 0, 1, 0, 0,
                                    0, 0, 1, 0, 0, 0, 0, 1};

// A col-maj: col0=[10/√2,10/√2], col1=[4.9/√2,5.1/√2]
static float qr_2x2_stress_A[] = {7.0710678f, 7.0710678f, 3.4648232f,
                                  3.6062446f};
// Q col-maj: col0=[1/√2,1/√2], col1=[-1/√2,1/√2]
static float qr_2x2_stress_Q[] = {0.7071068f, 0.7071068f, -0.7071068f,
                                  0.7071068f};
// R col-maj: col0=[10,0], col1=[5,0.1]
static float qr_2x2_stress_R[] = {10.0f, 0.0f, 5.0f, 0.1f};

using QRTestFloat = QRTest<float>;

INSTANTIATE_TEST_SUITE_P(
    QRTests, QRTestFloat,
    ::testing::Values(
        QRTestCase<float>{"2x2_identity", qr_2x2_identity_A, qr_2x2_identity_Q,
                          qr_2x2_identity_R, 2, 2, 1.0},
        QRTestCase<float>{"2x2_well_conditioned", qr_2x2_A, qr_2x2_Q, qr_2x2_R,
                          2, 2, 2.0},
        QRTestCase<float>{"3x3_well_conditioned", qr_3x3_A, qr_3x3_Q, qr_3x3_R,
                          3, 3, 2.0},
        QRTestCase<float>{"4x4_identity", qr_4x4_identity_A, qr_4x4_identity_Q,
                          qr_4x4_identity_R, 4, 4, 1.0},
        QRTestCase<float>{"2x2_kappa100_stress", qr_2x2_stress_A,
                          qr_2x2_stress_Q, qr_2x2_stress_R, 2, 2, 100.0}),
    [](const ::testing::TestParamInfo<QRTestFloat::ParamType> &info) {
      return info.param.name;
    });

TEST_P(QRTestFloat, CpuGramSchmidt) {
  const auto &tc = GetParam();
  qr::gramSchmidt(tc.A, Q, R, M, N);

  float tol = static_cast<float>(tc.kappa) * static_cast<float>(N) *
              std::numeric_limits<float>::epsilon() * 100.0f;

  for (size_t i = 0; i < M * N; ++i)
    EXPECT_NEAR(Q[i], tc.expectedQ[i], tol) << "Q mismatch at index " << i;

  for (size_t i = 0; i < N * N; ++i)
    EXPECT_NEAR(R[i], tc.expectedR[i], tol) << "R mismatch at index " << i;
}
