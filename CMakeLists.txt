cmake_minimum_required(VERSION 3.15)
project(ml_library CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(ml_core
    src/value.cpp
)
target_include_directories(ml_core PUBLIC inc)

add_executable(ml_app
    src/main.cpp
)
target_link_libraries(ml_app PRIVATE ml_core)

enable_testing()
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/d94a7326e97c9d564950e36c48ac402ba4933985.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(tests)