#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace qr::gpu {

/**
 * @brief Computes the QR decomposition of a matrix A using the modified Gram-Schmidt process.
 * Leverages matrix vector operations for better performance.
 *
 * @param A Input matrix of size MxN
 * @param Q Output orthogonal matrix of size MxN
 * @param R Output upper triangular matrix of size NxN
 * @param M Dimension 1 (rows) of the input matrix A
 * @param N Dimension 2 (columns) of the input matrix A
 * @param workspace Pointer to workspace memory of size M, used for intermediate computations
 * @return void
 */
template <typename T>
void modifiedGramSchmidt(const T* A, T* Q, T* R, uint32_t M, uint32_t N, T* workspace);

/**
 * @brief Computes the QR decomposition of a matrix A using the modified Gram-Schmidt process in the
 * most naive way possible.
 *
 * @param A Input matrix of size MxN
 * @param Q Output orthogonal matrix of size MxN
 * @param R Output upper triangular matrix of size NxN
 * @param M Dimension 1 (rows) of the input matrix A
 * @param N Dimension 2 (columns) of the input matrix A
 * @param workspace Pointer to workspace memory of size M, used for intermediate computations
 * @return void
 */
template <typename T>
void modifiedGramSchmidtNaive(const T* A, T* Q, T* R, uint32_t M, uint32_t N, T* workspace);

} // namespace qr::gpu
