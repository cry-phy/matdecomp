#pragma once
#include "common/cpuBuffer.hpp"
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>

// ---------------------------------------------------------------------------
// Shared test fixture and manufactured test cases for QR decomposition.
// Included by both test_qrCpu.cpp (g++) and test_qrGpu.cu (nvcc) — keep
// this header free of any CUDA-specific code.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Manufactured test cases for QR decomposition, with known Q and R factors.

template <typename T> struct ManQRTestCase {
  const char* name;
  const T* A;
  const T* expectedQ;
  const T* expectedR;
  size_t M;
  size_t N;
  double kappa;
};

template <typename T> void PrintTo(const ManQRTestCase<T>& tc, std::ostream* os) { *os << tc.name; }

// Fixture: allocates host Q and R buffers sized from the test case parameters.
template <typename T> class QRTest : public ::testing::TestWithParam<ManQRTestCase<T>> {
protected:
  void SetUp() override {
    const auto& tc = this->GetParam();
    M = tc.M;
    N = tc.N;
    Q = common::cpu::allocBuffer<T>(M * N);
    R = common::cpu::allocBuffer<T>(N * N);
  }

  void TearDown() override {
    common::cpu::freeBuffer(Q);
    common::cpu::freeBuffer(R);
  }

  T* Q;
  T* R;
  size_t M;
  size_t N;
};

// ---------------------------------------------------------------------------
// All matrices are stored column-major (column j starts at offset j*M).
//
// Case 1: 2x2 identity — trivial, Q=I, R=I, kappa=1
//
// Case 2: 2x2 well-conditioned (kappa ~ 2)
//   Q = [[1/√2, -1/√2],    R = [[√2,   1/√2],
//        [1/√2,  1/√2]]         [ 0,   1/√2]]
//   A = Q·R = [[1, 0],
//              [1, 1]]
//
// Case 3: 3x3 well-conditioned (kappa ~ 2)
//   Q = (1/3)*[[ 2, -2,  1],   R = [[3, 1, 2],
//              [ 1,  2,  2],        [0, 6, 1],
//              [ 2,  1, -2]]        [0, 0, 4]]
//   A = Q·R = [[ 2,      -10/3,  2],
//              [ 1,       13/3,  4],
//              [ 2,        8/3, -1]]
//
// Case 4: 4x4 identity
//
// Case 5: 2x2 ill-conditioned (kappa=100) — stress test
//   Q = [[1/√2, -1/√2],    R = [[10, 5  ],
//        [1/√2,  1/√2]]         [ 0, 0.1]]
//   A = Q·R = [[10/√2,  4.9/√2],
//              [10/√2,  5.1/√2]]
// ---------------------------------------------------------------------------

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
static float qr_3x3_Q[] = {0.6666667f, 0.3333333f, 0.6666667f, -0.6666667f, 0.6666667f,
                           0.3333333f, 0.3333333f, 0.6666667f, -0.6666667f};
// R col-maj: col0=[3,0,0], col1=[1,6,0], col2=[2,1,4]
static float qr_3x3_R[] = {3.0f, 0.0f, 0.0f, 1.0f, 6.0f, 0.0f, 2.0f, 1.0f, 4.0f};

static float qr_4x4_identity_A[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
static float qr_4x4_identity_Q[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
static float qr_4x4_identity_R[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

// A col-maj: col0=[10/√2,10/√2], col1=[4.9/√2,5.1/√2]
static float qr_2x2_stress_A[] = {7.0710678f, 7.0710678f, 3.4648232f, 3.6062446f};
// Q col-maj: col0=[1/√2,1/√2], col1=[-1/√2,1/√2]
static float qr_2x2_stress_Q[] = {0.7071068f, 0.7071068f, -0.7071068f, 0.7071068f};
// R col-maj: col0=[10,0], col1=[5,0.1]
static float qr_2x2_stress_R[] = {10.0f, 0.0f, 5.0f, 0.1f};

// Returns the full set of float test cases, used in INSTANTIATE_TEST_SUITE_P
// in both the CPU and GPU test files.
inline auto qrFloatManTestCases() {
  return ::testing::Values(
      ManQRTestCase<float>{"2x2_identity", qr_2x2_identity_A, qr_2x2_identity_Q, qr_2x2_identity_R,
                           2, 2, 1.0},
      ManQRTestCase<float>{"2x2_well_conditioned", qr_2x2_A, qr_2x2_Q, qr_2x2_R, 2, 2, 2.0},
      ManQRTestCase<float>{"3x3_well_conditioned", qr_3x3_A, qr_3x3_Q, qr_3x3_R, 3, 3, 2.0},
      ManQRTestCase<float>{"4x4_identity", qr_4x4_identity_A, qr_4x4_identity_Q, qr_4x4_identity_R,
                           4, 4, 1.0},
      ManQRTestCase<float>{"2x2_kappa100_stress", qr_2x2_stress_A, qr_2x2_stress_Q, qr_2x2_stress_R,
                           2, 2, 100.0});
}

// ---------------------------------------------------------------------------
// Property-based tests: random A, check Q orthonormality + R upper-triangular
// + QR == A reconstruction. No expected Q/R needed.

struct QRReversalCase {
  const char* name;
  size_t M, N;
  unsigned seed;
};

inline void PrintTo(const QRReversalCase& tc, std::ostream* os) { *os << tc.name; }

class QRReversalTest : public ::testing::TestWithParam<QRReversalCase> {
protected:
  void SetUp() override {
    const auto& tc = GetParam();
    M = tc.M;
    N = tc.N;
    A = common::cpu::allocBuffer<float>(M * N);
    Q = common::cpu::allocBuffer<float>(M * N);
    R = common::cpu::allocBuffer<float>(N * N);
    // Fill A with reproducible random values in [-1, 1].
    std::srand(tc.seed);
    for (size_t i = 0; i < M * N; i++)
      A[i] = 2.0f * (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) - 1.0f;
  }

  void TearDown() override {
    common::cpu::freeBuffer(A);
    common::cpu::freeBuffer(Q);
    common::cpu::freeBuffer(R);
  }

  // Absolute tolerance calibrated for float GS on random matrices in [-1,1].
  float tol() const {
    return static_cast<float>(N) * static_cast<float>(N) *
           std::numeric_limits<float>::epsilon() * 1000.0f;
  }

  float* A;
  float* Q;
  float* R;
  size_t M, N;
};

inline auto qrFloatReversalTestCases() {
  return ::testing::Values(QRReversalCase{"square_16x16",  16,  16,  1},
                           QRReversalCase{"square_64x64",  64,  64,  2},
                           QRReversalCase{"square_128x128",128, 128, 3},
                           QRReversalCase{"tall_256x16",   256, 16,  4},
                           QRReversalCase{"tall_512x8",    512, 8,   5});
}
