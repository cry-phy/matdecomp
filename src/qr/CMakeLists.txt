add_library(matdecomp_qr STATIC)

target_sources(matdecomp_qr PRIVATE
    cpuGramSchmidt.cpp
    cpuModifiedGramSchmidt.cpp
    cpuhouseholder.cpp
    gpuGramSchmidt.cu
    gpuModifiedGramSchmidt.cu
    gpuhouseholder.cu
)

add_library(matdecomp::qr ALIAS matdecomp_qr)

target_include_directories(matdecomp_qr PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(matdecomp_qr PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${MATDECOMP_WARNING_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${MATDECOMP_CUDA_WARNING_FLAGS}>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:${MATDECOMP_CXX_DEBUG_FLAGS}>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:${MATDECOMP_CUDA_DEBUG_FLAGS}>
    $<$<CONFIG:Release>:${MATDECOMP_RELEASE_FLAGS}>
    ${MATDECOMP_EXTRA_FLAGS_LIST}
)


set_target_properties(matdecomp_qr PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
