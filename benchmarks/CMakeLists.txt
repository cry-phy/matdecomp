
# --- CPU benchmarks ──────────────────────────────────────────────────────────
add_executable(matdecomp_cpu_benchmarks
    cpuBenchmarks.cpp
)

target_include_directories(matdecomp_cpu_benchmarks PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(matdecomp_cpu_benchmarks PRIVATE
    matdecomp_qr
    benchmark::benchmark
)

target_compile_options(matdecomp_cpu_benchmarks PRIVATE
    ${MATDECOMP_WARNING_FLAGS}
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:${MATDECOMP_CXX_DEBUG_FLAGS}>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:${MATDECOMP_CUDA_DEBUG_FLAGS}>
    $<$<CONFIG:Release>:${MATDECOMP_RELEASE_FLAGS}>
    ${MATDECOMP_EXTRA_FLAGS_LIST}
)

# ── GPU tests ────────────────────────────────────────────────────────────────
add_executable(matdecomp_gpu_benchmarks
  gpuBenchmarks.cu
)

target_include_directories(matdecomp_gpu_benchmarks PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(matdecomp_gpu_benchmarks PRIVATE
    matdecomp::qr
    benchmark::benchmark
    CUDA::cudart
)

target_compile_options(matdecomp_gpu_benchmarks PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${MATDECOMP_WARNING_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${MATDECOMP_CUDA_WARNING_FLAGS}>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:${MATDECOMP_CUDA_DEBUG_FLAGS}>
    $<$<CONFIG:Release>:${MATDECOMP_RELEASE_FLAGS}>
    ${MATDECOMP_EXTRA_FLAGS_LIST}
)

if(MATDECOMP_REFERENCE_BENCHMARKS)
    add_executable(matdecomp_reference_benchmarks
        cpuBenchmarks
    )

    target_link_libraries(matdecomp_reference_benchmarks PRIVATE
        benchmark::benchmark
        CUDA::cusparse
        CUDA::cusolver
    )
endif()
