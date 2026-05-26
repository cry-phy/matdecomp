#pragma once
#include <cstddef>
#include <cstdlib>

namespace common {
template <typename T> inline void fillRandom(T *buffer, size_t size) {
  for (size_t i = 0; i < size; i++) {
    buffer[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
  }
}
} // namespace common
