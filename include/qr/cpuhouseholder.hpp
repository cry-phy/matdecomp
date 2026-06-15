#pragma once
#include <cstddef>

namespace qr::cpu{
    /**
    * @brief Householder QR decomposition for CPU
    *
    * Computes the qr decomposition using the householder algorithm
    * @param A Input Matrix, MxN
    * @param Q Output orthogonal Matrix
    * @param R Output upper triangular Matrix
    * @param M Number of rows
    * @param N Number of columns
    */
template <typename T> void householder(const T* A,T* Q, T* R, size_t M, size_t N);
} //namespace qr::cpu
