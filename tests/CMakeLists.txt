include(GoogleTest)

add_executable(matdecomp_tests
    cpuLinalg.cpp
    qrManufactured.cpp
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
