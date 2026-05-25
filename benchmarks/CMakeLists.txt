add_executable(matdecomp_benchmarks
    cpuBenchmarks.cpp
)

target_link_libraries(matdecomp_benchmarks PRIVATE
    matdecomp_qr
    benchmark::benchmark
)

target_compile_options(matdecomp_benchmarks PRIVATE
    ${MATDECOMP_WARNING_FLAGS}
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:${MATDECOMP_CXX_DEBUG_FLAGS}>
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
