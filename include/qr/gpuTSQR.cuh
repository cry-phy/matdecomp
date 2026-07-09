#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace qr::gpu {

/**
 * @brief Computes the Tall and Skinny QR (TSQR) decomposition of a matrix using
 * a parallel binary reduction tree of Householder transformations.
 * * @tparam T      Data type (supported: float, double)
 * @param d_A     Pointer to the input matrix A of size M x N in device memory (Column-Major).
 * @param d_R     Pointer to the output upper-triangular matrix R of size N x N in device memory (Column-Major).
 * @param M       Number of rows in matrix A (M >= N).
 * @param N       Number of columns in matrix A.
 */
template <typename T>
void TSQR(const T* d_A, T* d_R, uint32_t M, uint32_t N);

} // namespace qr::gpu
