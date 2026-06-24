cmake_minimum_required(VERSION 3.14)

project(MatrixDecomposition VERSION 1.0
    DESCRIPTION "QR and SVD decomposition CUDA implementations and benchmarks"
    LANGUAGES C CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA architecture — sm_86 = RTX 30xx / A-series
# Override at configure time: cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA GPU architecture (sm_XX without 'sm_')")

# Warnings — applied to all build types.
set(MATDECOMP_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
)

set(MATDECOMP_CUDA_WARNING_FLAGS
    -Wall
    -Wextra
    -Wno-unused-parameter
    --ptxas-options=-v
)

# Maximum Debug Flags
set(MATDECOMP_CXX_DEBUG_FLAGS
    -O0
    -g3
)

# CUDA debug flags — passed to nvcc.
set(MATDECOMP_CUDA_DEBUG_FLAGS
    -G
)

# Intentionally excluded from -ffast-math:
#   -ffinite-math-only     would assume no NaN/Inf — unsafe for general inputs
set(MATDECOMP_RELEASE_FLAGS
    -march=native
    -fno-math-errno       # skip errno after sqrt/log/etc.
    -fno-trapping-math    # assume no FP trap signals (default MXCSR on x86)
    -fno-signed-zeros     # -0.0 == 0.0; required by GCC to activate fassociative-math
    -fassociative-math    # allow FP reordering → vectorised reductions (dot product)
    -ffp-contract=fast    # fuse multiply-add into FMA instructions
)

set(MATDECOMP_EXTRA_FLAGS "" CACHE STRING
    "Additional compiler flags appended to all build types (semicolon-separated)")
separate_arguments(MATDECOMP_EXTRA_FLAGS_LIST UNIX_COMMAND "${MATDECOMP_EXTRA_FLAGS}")

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

add_custom_target(run-benchmark
    COMMAND matdecomp_cpu_benchmarks
    COMMAND matdecomp_gpu_benchmarks
    COMMENT "Running all benchmarks..."
)
    

option(MATDECOMP_REFERENCE_BENCHMARKS "Build cuSPARSE/cuSOLVER reference benchmarks" OFF)

find_package(CUDAToolkit REQUIRED)

if(MATDECOMP_REFERENCE_BENCHMARKS)
  find_package(CUDAToolkit REQUIRED COMPONENTS cusparse cusolver)
endif()

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)
