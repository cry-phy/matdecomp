add_executable(matdecomp_benchmarks
    cpuBenchmarks.cpp
)

target_link_libraries(matdecomp_benchmarks PRIVATE
    matdecomp_qr
    benchmark::benchmark
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
