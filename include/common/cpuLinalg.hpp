#pragma once
#include <cassert>
#include <cmath>
#include <cstddef>

namespace common {

/**
 * @brief Computes the dot product of two vectors.
 *
 * @param a First input vector
 * @param b Second input vector
 * @param n Length of both vectors
 * @return Dot product a·b
 */
template <typename T> inline T dot(const T* a, const T* b, size_t n) {
  T result = 0;
  for (size_t i = 0; i < n; i++) {
    result += a[i] * b[i];
  }
  return result;
}

/**
 * @brief Computes the L2 norm of a vector.
 *
 * @param a Input vector
 * @param n Length of the vector
 * @return L2 norm ||a||
 */
template <typename T> inline T norm(const T* a, size_t n) { return std::sqrt(dot(a, a, n)); }

/**
 * @brief Scales a vector by a factor: res = factor * a, allows in-place
 * operation if res == a
 *
 * @param a Input vector
 * @param res Output vector
 * @param n Length of the vector
 * @param factor Scaling factor
 * @return Scaled vector in res
 */
template <typename T> inline void scale(const T* a, T* res, size_t n, T factor) {
  for (size_t i = 0; i < n; i++) {
    res[i] = factor * a[i];
  }
}

/**
 * @brief Normalizes a vector to have unit length: res = a / ||a||, allows
 * in-place operation if res == a
 *
 * @param a Input vector
 * @param res Output normalized vector
 * @param n Length of the vector
 * @return Normalized vector in res
 */
template <typename T> inline void normalize(const T* a, T* res, size_t n) {
  T a_norm = norm(a, n);
  assert(a_norm > 0 && "Cannot normalize zero-length vector");
  scale(a, res, n, 1 / a_norm);
}

/**
 * @brief Adds two vectors element-wise: res = a + b, allows in-place
 * operation if res == a or res == b
 *
 * @param a First input vector
 * @param b Second input vector
 * @param res Output vector
 * @param n Length of the vectors
 * @return Element-wise sum in res
 */
template <typename T> inline void add(const T* a, const T* b, T* res, size_t n) {
  for (size_t i = 0; i < n; i++) {
    res[i] = a[i] + b[i];
  }
}

/**
 * @brief Subtracts two vectors element-wise: res = a - b, allows in-place
 * operation if res == a or res == b
 *
 * @param a First input vector
 * @param b Second input vector
 * @param res Output vector
 * @param n Length of the vectors
 * @return Element-wise difference in res
 */
template <typename T> inline void sub(const T* a, const T* b, T* res, size_t n) {
  for (size_t i = 0; i < n; i++) {
    res[i] = a[i] - b[i];
  }
}

/**
 * @brief Projects vector a onto vector b: res = (a·b / b·b) * b
 *
 * @param a Vector to project
 * @param b Vector to project onto
 * @param res Output projection, must be pre-allocated of length n
 * @param n Length of all vectors
 */
template <typename T> inline void project(const T* a, const T* b, T* res, size_t n) {
  T s = dot(a, b, n) / dot(b, b, n);
  for (size_t i = 0; i < n; i++) {
    res[i] = s * b[i];
  }
}

/**
 * @brief Checks if two vectors are element-wise close within a given tolerance
 *
 * @param a First input vector
 * @param b Second input vector
 * @param n Length of the vectors
 * @return True if all elements of a and b are within tol, false otherwise
 */
template <typename T>
inline bool isClose(const T* a, const T* b, size_t n, T tol = std::numeric_limits<T>::epsilon()) {
  for (size_t i = 0; i < n; i++) {
    if (std::abs(a[i] - b[i]) > tol) {
      return false;
    }
  }
  return true;
}

} // namespace common
