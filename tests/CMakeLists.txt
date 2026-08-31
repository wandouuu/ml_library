include(GoogleTest)

add_executable(testing
    test_value.cpp
)

target_link_libraries(testing PRIVATE
    GTest::gtest_main
    ml_core
)

gtest_discover_tests(testing)