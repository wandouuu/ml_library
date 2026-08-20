cmake_minimum_required(VERSION 3.15)
project(ml_library CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ml_app
    src/main.cpp
    src/value.cpp
)

target_include_directories(ml_app PRIVATE inc)