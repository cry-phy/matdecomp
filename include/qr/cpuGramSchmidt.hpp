#pragma once
#include <cstddef>

namespace qr::cpu {

/**
 * @brief Naive Gram-Schmidt QR decomposition (CPU).
 *
 * Orthogonalizes the columns of A and writes the result into Q.
 * A is M×N stored column-major with leading dimension M.
 *
 * @param A Input matrix, M×N column-major
 * @param Q Output orthogonal matrix, M×N column-major
 * @param R Output upper triangular matrix, N×N column-major
 * @param M Number of rows
 * @param N Number of columns
 */
template <typename T> void gramSchmidt(const T* A, T* Q, T* R, size_t M, size_t N);

} // namespace qr::cpu

namespace {

template <typename T> void gsComputeQ(const T* A, T* Q, size_t M, size_t N);

template <typename T> void gsComputeR(const T* A, const T* Q, T* R, size_t M, size_t N);

} // anonymous namespace
