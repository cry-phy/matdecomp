include(GoogleTest)

# ── CPU tests ────────────────────────────────────────────────────────────────
add_executable(matdecomp_tests
    test_cpuLinalg.cpp
    test_qrCpu.cpp
)

target_include_directories(matdecomp_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(matdecomp_tests PRIVATE
    matdecomp::qr
    GTest::gtest_main
)

target_compile_options(matdecomp_tests PRIVATE
    ${MATDECOMP_WARNING_FLAGS}
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:${MATDECOMP_CXX_DEBUG_FLAGS}>
    $<$<CONFIG:Release>:${MATDECOMP_RELEASE_FLAGS}>
    ${MATDECOMP_EXTRA_FLAGS_LIST}
)

gtest_discover_tests(matdecomp_tests)

# ── GPU tests ────────────────────────────────────────────────────────────────
add_executable(matdecomp_gpu_tests
    test_gpuLinalg.cu
    test_qrGpu.cu
)

target_include_directories(matdecomp_gpu_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(matdecomp_gpu_tests PRIVATE
    matdecomp::qr
    GTest::gtest_main
    CUDA::cudart
)

target_compile_options(matdecomp_gpu_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${MATDECOMP_WARNING_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${MATDECOMP_CUDA_WARNING_FLAGS}>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:${MATDECOMP_CUDA_DEBUG_FLAGS}>
    $<$<CONFIG:Release>:${MATDECOMP_RELEASE_FLAGS}>
    ${MATDECOMP_EXTRA_FLAGS_LIST}
)

# PRE_TEST: defers test discovery to run time, not configure time.
# Safe when a GPU may not be present on the build machine.
gtest_discover_tests(matdecomp_gpu_tests
    DISCOVERY_MODE PRE_TEST
)
