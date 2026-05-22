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

gtest_discover_tests(matdecomp_tests)
