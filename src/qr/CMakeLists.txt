add_library(matdecomp_qr STATIC)

target_sources(matdecomp_qr PRIVATE
    cpuGramSchmidt.cpp
)

add_library(matdecomp::qr ALIAS matdecomp_qr)

target_include_directories(matdecomp_qr PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(matdecomp_qr PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
