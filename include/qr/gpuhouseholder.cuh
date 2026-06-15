#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace qr::gpu {
    template <typename T>
void Householder(const T* A, T* Q, T* R, uint32_t M, uint32_t N, T* workspace);
}//namespace qr::gpu
