#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace common::cpu {

template <typename T>
inline T *allocBuffer(size_t n) {
    return static_cast<T *>(malloc(n * sizeof(T)));
}

template <typename T>
inline void freeBuffer(T *buf) {
    free(buf);
}

template <typename T>
inline void copyBuffer(T *dst, const T *src, size_t n) {
    memcpy(dst, src, n * sizeof(T));
}

template <typename T>
inline void zeroBuffer(T *buf, size_t n) {
    memset(buf, 0, n * sizeof(T));
}

} // namespace common::cpu
