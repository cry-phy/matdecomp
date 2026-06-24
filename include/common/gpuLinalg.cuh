#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime.h>

namespace common {

namespace gpu {

#define INIT_REDUCTION_COUNT 16
#define MAX_BLOCK_SIZE 1024
#define WARP_SIZE 32

// ── device building blocks ────────────────────────────────────────────────────

/**
 * @brief Performs a warp-level reduction to sum val across the warp using shuffle down operations.
 *
 * @param val Value to be reduced across the warp
 * @return The sum of val across the warp
 */
template <typename T> __device__ T _warpReduceSum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

/**
 * @brief Reduces val across the block using warp-level primitives in shared memory.
 *
 * @param val Value to be reduced across the block
 * @return The sum of val across the block
 */
template <typename T> __device__ T _blockReduce(T val) {
  __shared__ T warpSums[MAX_BLOCK_SIZE / WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int warpId = threadIdx.x / WARP_SIZE;

  val = _warpReduceSum<T>(val);
  if (lane == 0)
    warpSums[warpId] = val;
  __syncthreads();

  if (warpId == 0) {
    uint32_t num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? warpSums[threadIdx.x] : static_cast<T>(0);
    warpSums[0] = _warpReduceSum<T>(val);
  }
  __syncthreads();
  return warpSums[0];
}

/**
 * @brief Scales vector b by factor and stores in res.
 *
 * @param a Vector to be scaled
 * @param res Output vector to hold the scaled result
 * @param N Number of elements in a and res
 * @param factor Scaling factor
 * @return void
 */
template <typename T> __device__ void _scale(const T* a, T* res, uint64_t N, T factor) {
  uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    res[idx] = a[idx] * factor;
}

// ── kernels ───────────────────────────────────────────────────────────────────

/**
 * @brief Subtracts two vectors a and b, storing the result in res.
 *
 * @param a Input vector a
 * @param b Input vector b
 * @param res Output vector to hold the result of a - b
 * @param N Number of elements in a, b, and res
 * @return void
 */
template <typename T> __global__ void subtractKernel(const T* a, const T* b, T* res, uint64_t N) {
  uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    res[idx] = a[idx] - b[idx];
}

/**
 * @brief Performs the dot product of two vectors a and b, storing the result in res.
 *
 * @param a Input vector a
 * @param b Input vector b
 * @param res Device pointer to store the result. Must be initialized to zero before launch.
 * @param N Number of elements in a and b
 * @return void
 */
template <typename T> __global__ void dotProductKernel(const T* a, const T* b, T* res, uint32_t N) {
  uint32_t base = blockIdx.x * blockDim.x * INIT_REDUCTION_COUNT + threadIdx.x;
  T val = static_cast<T>(0);
  for (uint32_t i = 0; i < INIT_REDUCTION_COUNT; i++) {
    uint32_t idx = base + i * blockDim.x;
    if (idx < N)
      val += a[idx] * b[idx];
  }
  val = _blockReduce<T>(val);
  if (threadIdx.x == 0)
    atomicAdd(res, val);
}

/**
 * @brief Scales vector a by factor.
 *
 * @param a Input vector to be scaled
 * @param res Output vector to hold the scaled result
 * @param N Number of elements in a and res
 * @param factor Scaling factor
 * @return void
 */
template <typename T> __global__ void scaleKernel(const T* a, T* res, uint64_t N, T factor) {
  _scale(a, res, N, factor);
}

/**
 * @brief Scales vector a by factor read from device memory.
 *
 * @param a Input vector to be scaled
 * @param res Output vector to hold the scaled result
 * @param N Number of elements in a and res
 * @param factor Device pointer to the scaling factor (one T on device)
 * @return void
 */
template <typename T>
__global__ void scaleKernelDevice(const T* a, T* res, uint64_t N, const T* factor) {
  _scale(a, res, N, *factor);
}

/**
 * @brief Normalizes vector a and stores the result in res.
 *
 * @param a Vector to be normalized
 * @param res Output vector to hold the normalized result
 * @param normSq Device pointer to the squared norm of a (one T on device)
 * @param N Number of elements in a and res
 * @return void
 */
template <typename T>
__global__ void normalizeKernel(const T* a, T* res, const T* normSq, uint32_t N) {
  T invNorm = rsqrtf(*normSq);
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    res[idx] = a[idx] * invNorm;
}

/**
 * @brief scale b by factor and store in res. Factor is read from device memory to enable queuing
 * both the dot product and scaling.
 *
 * @param b Vector to be scaled
 * @param res Output vector to hold the scaled result
 * @param N Number of elements in b and res
 * @param factor Scaling factor read from device memory (one T on device)
 * @return void
 */
template <typename T> __global__ void scaleKernel(const T* b, T* res, uint64_t N, const T* factor) {
  uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    res[idx] = b[idx] * *factor;
}

template <typename T> __global__ void sqrtKernel(const T* a, T* res, uint64_t N) {
  uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    res[idx] = sqrtf(a[idx]);
}

// ── host launchers ────────────────────────────────────────────────────────────

/**
 * @brief Subtracts two vectors a and b, storing the result in res.
 *
 * @param a Input vector a
 * @param b Input vector b
 * @param res Output vector to hold the result of a - b
 * @param N Number of elements in a, b, and res
 * @return void
 */
template <typename T> void subtract(const T* a, const T* b, T* res, uint64_t N) {
  uint32_t blockSize = MAX_BLOCK_SIZE;
  uint32_t numBlocks = (N + blockSize - 1) / blockSize;
  subtractKernel<T><<<numBlocks, blockSize>>>(a, b, res, N);
}

/**
 * @brief Performs element-wise subtraction of two matrices A and B. Allows for different leading
 * dimensions (ldA, ldB, ldRes) for each matrix.
 *
 * @param A Matrix A of size M x N, stored in column-major with ldA
 * @param B Matrix B of size M x N, stored in column-major with ldB
 * @param res Output matrix to hold the result of A - B, stored in column-major with ldRes
 * @param M dimension 1 of the matrices
 * @param N dimension 2 of the matrices
 * @param ldA leading dimension of A (number of rows)
 * @param ldB leading dimension of B (number of rows)
 * @param ldRes leading dimension of res (number of rows)
 * @return value
 */
template <typename T>
__global__ void subtractMatrixKernel(const T* A, const T* B, T* res, uint32_t M, uint32_t N,
                                     uint32_t ldA, uint32_t ldB, uint32_t ldRes) {
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    res[row + col * ldRes] = A[row + col * ldA] - B[row + col * ldB];
  }
}

/**
 * @brief Host launcher for element-wise subtraction of two matrices A and B.
 *
 * @param A Matrix A of size M x N, stored in column-major with ldA
 * @param B Matrix B of size M x N, stored in column-major with ldB
 * @param res Output matrix to hold the result of A - B, stored in column-major with ldRes
 * @param M dimension 1 of the matrices
 * @param N dimension 2 of the matrices
 * @param ldA leading dimension of A (number of rows)
 * @param ldB leading dimension of B (number of rows)
 * @param ldRes leading dimension of res (number of rows)
 * @return void
 */
template <typename T>
void subtractMatrix(const T* A, const T* B, T* res, uint32_t M, uint32_t N, uint32_t ldA,
                    uint32_t ldB, uint32_t ldRes) {
  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
  subtractMatrixKernel<T><<<gridDim, blockDim>>>(A, B, res, M, N, ldA, ldB, ldRes);
}

/**
 * @brief Performs the dot product of two vectors a and b, storing the result in res.
 *
 * @param a Input vector a
 * @param b Input vector b
 * @param res Device pointer to store the result. Must be initialized to zero.
 * @param N Number of elements in a and b
 * @return void
 */
template <typename T> void dotProduct(const T* a, const T* b, T* res, uint32_t N) {
  uint32_t blockSize = MAX_BLOCK_SIZE;
  uint32_t numBlocks =
      std::max(static_cast<uint32_t>(1),
               (N + blockSize * INIT_REDUCTION_COUNT - 1) / (blockSize * INIT_REDUCTION_COUNT));
  cudaMemset(res, 0, sizeof(T));
  dotProductKernel<T><<<numBlocks, blockSize>>>(a, b, res, N);
}

/**
 * @brief Scales a vector a by a factor and stores the result in res.
 *
 * @param a Input vector to be scaled
 * @param res Output vector to hold the scaled result
 * @param N Number of elements in a and res
 * @param factor Scaling factor
 * @return void
 */
template <typename T> void scale(const T* a, T* res, uint64_t N, T factor) {
  uint32_t blockSize = MAX_BLOCK_SIZE;
  uint32_t numBlocks = (N + blockSize - 1) / blockSize;
  scaleKernel<T><<<numBlocks, blockSize>>>(a, res, N, factor);
}

/**
 * @brief Normalizes a vector a and stores the result.
 *
 * @param a Vector to be normalized
 * @param res Output vector to hold the normalized result
 * @param N Number of elements in a and res
 * @param d_normSq Device pointer to store the squared norm of a (one T on device)
 * @return void
 */
template <typename T> void normalize(const T* a, T* res, uint32_t N, T* d_normSq) {
  uint32_t blockSize = MAX_BLOCK_SIZE;
  cudaMemset(d_normSq, 0, sizeof(T));
  uint32_t numBlocks =
      (N + blockSize * INIT_REDUCTION_COUNT - 1) / (blockSize * INIT_REDUCTION_COUNT);
  dotProductKernel<T><<<numBlocks, blockSize>>>(a, a, d_normSq, N);
  numBlocks = (N + blockSize - 1) / blockSize;
  normalizeKernel<T><<<numBlocks, blockSize>>>(a, res, d_normSq, N);
}

/**
 * @brief Projects a onto b by computing the dot product of a and b. Does not normalize b, so the
 * result is not a unit vector.
 *
 * @param a Vector to be projected
 * @param b Vector to project onto
 * @param res Output vector
 * @param N Number of elements in a, b, and res
 * @param d_dot_ab Device pointer to store the dot product of a and b (one T on device)
 * @return void
 */
template <typename T> void project(const T* a, const T* b, T* res, uint32_t N, T* d_dot_ab) {
  uint32_t blockSize = MAX_BLOCK_SIZE;
  uint32_t numBlocks =
      (N + blockSize * INIT_REDUCTION_COUNT - 1) / (blockSize * INIT_REDUCTION_COUNT);
  cudaMemset(d_dot_ab, 0, sizeof(T));
  dotProductKernel<T><<<numBlocks, blockSize>>>(a, b, d_dot_ab, N);
  numBlocks = (N + blockSize - 1) / blockSize;
  scaleKernel<T><<<numBlocks, blockSize>>>(b, res, N, d_dot_ab);
}

#define BLOCK_SIZE 32

/**
 * @brief Blocked matrix multiplication kernel for multiplying matrices A and B, storing the result
 * in C.
 *
 * @param A Left matrix of size M x K
 * @param B Right matrix of size K x N
 * @param C Output matrix of size M x N
 * @param M Size of the first dimension of A and C
 * @param N Size of the second dimension of B and C
 * @return void
 */
template <typename T, bool tA = false, bool tB = false>
__global__ void __launch_bounds__(BLOCK_SIZE* BLOCK_SIZE)
    matMulKernel(const T* A, const T* B, T* C, uint32_t M, uint32_t K, uint32_t N, uint32_t ldA,
                 uint32_t ldB, uint32_t ldC) {
  __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

  uint32_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  uint32_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  T Cvalue = static_cast<T>(0);
  for (uint32_t m = 0; m < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
    uint32_t aCol = m * BLOCK_SIZE + threadIdx.x;
    if (row < M && aCol < K) {
      if constexpr (tA)
        As[threadIdx.y][threadIdx.x] = A[static_cast<uint64_t>(row) * ldA + aCol];
      else
        As[threadIdx.y][threadIdx.x] = A[static_cast<uint64_t>(aCol) * ldA + row];
    } else {
      As[threadIdx.y][threadIdx.x] = static_cast<T>(0);
    }

    uint32_t bRow = m * BLOCK_SIZE + threadIdx.y;
    if (bRow < K && col < N) {
      if constexpr (tB)
        Bs[threadIdx.y][threadIdx.x] = B[static_cast<uint64_t>(bRow) * ldB + col];
      else
        Bs[threadIdx.y][threadIdx.x] = B[static_cast<uint64_t>(col) * ldB + bRow];
    } else {
      Bs[threadIdx.y][threadIdx.x] = static_cast<T>(0);
    }

    __syncthreads();
    for (uint32_t k = 0; k < BLOCK_SIZE; ++k)
      Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }

  if (row < M && col < N)
    C[static_cast<uint64_t>(col) * ldC + row] = Cvalue;
}

/**
 * @brief Host launcher for blocked matrix multiplication.
 *
 * @param A Left matrix of size M x K
 * @param B Right matrix of size K x N
 * @param C Output matrix of size M x N
 * @param M Size of the first dimension of A and C
 * @param N Size of the second dimension of B and C
 * @param K Size of the shared dimension of A and B
 * @return void
 */
template <typename T, bool tA = false, bool tB = false>
void matMul(const T* A, const T* B, T* C, uint32_t M, uint32_t K, uint32_t N, uint32_t ldA,
            uint32_t ldB, uint32_t ldC) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  matMulKernel<T, tA, tB><<<gridDim, blockDim>>>(A, B, C, M, K, N, ldA, ldB, ldC);
}

/**
 * @brief Performs matrix-vector multiplication of matrix A and vector x, storing the result in y.
 *
 * @param A Input matrix of size M x N
 * @param x Input vector of size N
 * @param y Output vector of size M
 * @param M Number of rows in A and size of y
 * @param N Number of columns in A and size of x
 * @param lda Leading dimension of A (number of rows)
 * @return void
 */
template <typename T, bool tA = false>
__global__ void matVecMultKernel(const T* A, const T* x, T* y, uint32_t M, uint32_t N,
                                 uint32_t lda) {
  uint32_t row = blockIdx.x;
  if (row >= M)
    return;

  T tsum = static_cast<T>(0);
  uint64_t idx = 0;
  for (uint32_t col = threadIdx.x; col < N; col += blockDim.x) {
    if constexpr (tA)
      idx = static_cast<uint64_t>(row) * lda + col;
    else
      idx = static_cast<uint64_t>(col) * lda + row;
    tsum += A[idx] * x[col];
  }

  T result = _blockReduce(tsum);
  if (threadIdx.x == 0)
    y[row] = result;
}

/**
 * @brief Host launcher for matrix-vector multiplication.
 *
 * @param A Input matrix of size M x N
 * @param x Input vector of size N
 * @param y Output vector of size M
 * @param M Number of rows in A and size of y
 * @param N Number of columns in A and size of x
 * @param lda Leading dimension of A (number of rows)
 * @return void
 */
template <typename T, bool tA = false>
void matVecMult(const T* A, const T* x, T* y, uint32_t M, uint32_t N, uint32_t lda) {
  // Round the block up to a full warp so _blockReduce never shuffles across absent lanes.
  uint32_t blockDim =
      std::min<uint32_t>(MAX_BLOCK_SIZE, ((N + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE);
  matVecMultKernel<T, tA><<<M, blockDim>>>(A, x, y, M, N, lda);
}

/**
 * @brief Rank-1 update
 *
 * @param x Column vector of length M
 * @param y Row vector of length N
 * @param A Matrix of size M x N, column-major with leading dimension lda
 * @param M Number of rows
 * @param N Number of columns
 * @param lda Leading dimension of A
 * @return void
 */
template <typename T>
__global__ void gerKernel(const T* x, const T* y, T* A, uint32_t M, uint32_t N, uint32_t lda) {
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N)
    A[static_cast<uint64_t>(col) * lda + row] -= x[row] * y[col];
}

/**
 * @brief Host launcher for the rank-1 update
 *
 * @param x Column vector of length M
 * @param y Row vector of length N
 * @param A Matrix of size M x N, column-major with leading dimension lda
 * @param M Number of rows
 * @param N Number of columns
 * @param lda Leading dimension of A
 * @return void
 */
template <typename T> void ger(const T* x, const T* y, T* A, uint32_t M, uint32_t N, uint32_t lda) {
  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
  gerKernel<T><<<gridDim, blockDim>>>(x, y, A, M, N, lda);
}

/**
 * @brief Scatters a contiguous vector into a strided destination: dst[i * stride] = src[i]. Used to
 * write an R row (strided in column-major) from a contiguous mat-vec result.
 *
 * @param src Contiguous source vector of length n
 * @param dst Destination base pointer; element i is written at dst[i * stride]
 * @param n Number of elements
 * @param stride Stride in elements between consecutive destination writes
 * @return void
 */
template <typename T>
__global__ void scatterStridedKernel(const T* src, T* dst, uint32_t n, uint32_t stride) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[static_cast<uint64_t>(i) * stride] = src[i];
}

/**
 * @brief Host launcher for scatterStrided.
 *
 * @param src Contiguous source vector of length n
 * @param dst Destination base pointer; element i is written at dst[i * stride]
 * @param n Number of elements
 * @param stride Stride in elements between consecutive destination writes
 * @return void
 */
template <typename T> void scatterStrided(const T* src, T* dst, uint32_t n, uint32_t stride) {
  uint32_t blockSize = MAX_BLOCK_SIZE;
  uint32_t numBlocks = (n + blockSize - 1) / blockSize;
  scatterStridedKernel<T><<<numBlocks, blockSize>>>(src, dst, n, stride);
}

/**
 * @brief Scales vector a by factors read from device memory, storing the result in res.
 *
 * @param a Input vector to be scaled
 * @param res Output vector to hold the scaled result
 * @param N Number of elements in a and res
 * @param factors Device pointer to the scaling factors (one T per element on device)
 * @return void
 */
template <typename T>
__global__ void scalePerElementKernel(const T* a, T* res, uint64_t N, const T* factors) {
  uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    res[idx] = a[idx] * factors[idx];
}

/**
 * @brief Scales vector a by factors read from device memory, storing the result in res.
 *
 * @param a Input vector to be scaled
 * @param res Output vector to hold the scaled result
 * @param N Number of elements in a and res
 * @param factors Device pointer to the scaling factors (one T per element on device)
 * @return voidOn exit: R = upper triangle of A; reflectors live in the lower part plus tau. For
 * explicit Q (org2r/orgqr): start from I and apply panels in reverse with larfb using Q (not Qᵀ).
 */
template <typename T> void scalePerElement(const T* a, T* res, uint64_t N, const T* factors) {
  uint32_t blockDim = MAX_BLOCK_SIZE;
  uint32_t gridDim = (N + blockDim - 1) / blockDim;
  scalePerElementKernel<T><<<gridDim, blockDim>>>(a, res, N, factors);
}

/**
 * @brief Kernel to identify a slice of an identity matrix of size M x M and store the first N
 * columns in the provided matrix.
 *
 * @param matrix Output matrix to hold the identity matrix slice (size M x N)
 * @param M Number of rows in the identity matrix
 * @param N Number of columns in the identity matrix slice
 * @return void
 */
template <typename T> __global__ void identifyMatrixSliceKernel(T* matrix, uint32_t M, uint32_t N) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < M) {
    matrix[row * M + col] = (row == col) ? static_cast<T>(1) : static_cast<T>(0);
  }
}

/**
 * @brief Generates an identity matrix of size M x M and stores the first N columns in the provided
 * matrix.
 *
 * @param matrix Output matrix to hold the identity matrix slice (size M x N)
 * @param M Number of rows in the identity matrix
 * @param N Number of columns in the identity matrix slice
 * @return void
 */
template <typename T> void identityMatrixSlice(T* matrix, uint32_t M, uint32_t N) {
  dim3 blockDim(16, 16);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
  identifyMatrixSliceKernel<T><<<gridDim, blockDim>>>(matrix, M, N);
}

} // namespace gpu

} // namespace common

#endif // __CUDACC__
