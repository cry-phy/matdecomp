cmake_minimum_required(VERSION 3.14)

project(MatrixDecomposition VERSION 1.0
    DESCRIPTION "QR and SVD decomposition CUDA implementations and benchmarks"
    LANGUAGES C CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Test dependencies
include(FetchContent)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.5
)
FetchContent_MakeAvailable(benchmark)

option(MATDECOMP_REFERENCE_BENCHMARKS "Build cuSPARSE/cuSOLVER reference benchmarks" OFF)

if(MATDECOMP_REFERENCE_BENCHMARKS)
  find_package(CUDAToolkit REQUIRED COMPONENTS cusparse cusolver)
endif()

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)
