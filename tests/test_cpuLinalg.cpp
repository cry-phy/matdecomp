#include "common/cpuLinalg.hpp"
#include <gtest/gtest.h>
#include <limits>

TEST(CpuLinalgTest, DotProduct) {
  float a[] = {1.0f, 2.0f, 3.0f};
  float b[] = {4.0f, 5.0f, 6.0f};
  float result = common::cpu::dot(a, b, 3);
  EXPECT_NEAR(result, 32.0f, std::numeric_limits<float>::epsilon());
}

TEST(CpuLinalgTest, Normalize) {
  float a[] = {3.0f, 4.0f};
  float result[2];
  common::cpu::normalize(a, result, 2);
  EXPECT_NEAR(result[0], 0.6f, std::numeric_limits<float>::epsilon());
  EXPECT_NEAR(result[1], 0.8f, std::numeric_limits<float>::epsilon());
}

TEST(CpuLinalgTest, Project) {
  float a[] = {1.0f, 2.0f};
  float b[] = {3.0f, 4.0f};
  float result[2];
  common::cpu::project(a, b, result, 2);
  EXPECT_NEAR(result[0], 1.32f, std::numeric_limits<float>::epsilon() * 10);
  EXPECT_NEAR(result[1], 1.76f, std::numeric_limits<float>::epsilon() * 10);
}

TEST(CpuLinalgTest, Subtract) {
  float a[] = {5.0f, 7.0f};
  float b[] = {2.0f, 3.0f};
  float result[2];
  common::cpu::sub(a, b, result, 2);
  EXPECT_NEAR(result[0], 3.0f, std::numeric_limits<float>::epsilon());
  EXPECT_NEAR(result[1], 4.0f, std::numeric_limits<float>::epsilon());
}

TEST(CpuLinalgTest, Add) {
  float a[] = {1.0f, 2.0f};
  float b[] = {3.0f, 4.0f};
  float result[2];
  common::cpu::add(a, b, result, 2);
  EXPECT_NEAR(result[0], 4.0f, std::numeric_limits<float>::epsilon());
  EXPECT_NEAR(result[1], 6.0f, std::numeric_limits<float>::epsilon());
}

TEST(CpuLinalgTest, Scale) {
  float a[] = {1.0f, 2.0f};
  float result[2];
  common::cpu::scale(a, result, 2, 3.0f);
  EXPECT_NEAR(result[0], 3.0f, std::numeric_limits<float>::epsilon());
  EXPECT_NEAR(result[1], 6.0f, std::numeric_limits<float>::epsilon());
}

TEST(CpuLinalgTest, IsClose) {
  float a[] = {1.0f, 2.0f};
  float b[] = {1.0f + 1e-6f, 2.0f - 1e-6f};
  EXPECT_TRUE(common::cpu::isClose(a, b, 2, 1e-5f));
  EXPECT_FALSE(common::cpu::isClose(a, b, 2, 1e-7f));
}

TEST(CpuLinalgTest, matVecMul) {
  float A[] = {1.0f, 2.0f, 3.0f, 4.0f}; // 2x2 matrix
  float x[] = {1.0f, 2.0f};             // 2x1 vector
  float res[2];
  common::cpu::matVecMul(A, x, res, 2, 2);
  EXPECT_NEAR(res[0], 7.0f, std::numeric_limits<float>::epsilon());
  EXPECT_NEAR(res[1], 10.0f, std::numeric_limits<float>::epsilon());
}
