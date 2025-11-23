cmake_minimum_required(VERSION 3.16)
project(apple_silicon_vec_playground LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vecplay
    src/kernels.cpp
    src/bench.cpp
)

target_include_directories(vecplay
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(vec_bench benchmark/vec_bench.cpp)
target_link_libraries(vec_bench PRIVATE vecplay)