cmake_minimum_required(VERSION 3.20)
project(xtensor_nanobind_benchmark LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Locate nanobind via cmake variable or environment variable
if(NOT NANOBIND_DIR)
    if(DEFINED ENV{NANOBIND_DIR})
        set(NANOBIND_DIR "$ENV{NANOBIND_DIR}")
    else()
        message(FATAL_ERROR
            "NANOBIND_DIR is not set. Point it at the nanobind source tree:\n"
            "  cmake -DNANOBIND_DIR=/path/to/nanobind ..")
    endif()
endif()

add_subdirectory(${NANOBIND_DIR} nanobind)

find_package(xtl REQUIRED CONFIG)
find_package(xtensor REQUIRED CONFIG)

# xsimd is optional — when found, xtensor uses SIMD instructions automatically
find_package(xsimd CONFIG QUIET)

nanobind_add_module(benchmark_ext src/benchmark.cpp)
target_link_libraries(benchmark_ext PRIVATE xtensor xtl)

if(xsimd_FOUND)
    target_link_libraries(benchmark_ext PRIVATE xsimd)
    target_compile_definitions(benchmark_ext PRIVATE XTENSOR_USE_XSIMD)
    message(STATUS "xsimd found — SIMD-accelerated xtensor enabled")
else()
    message(STATUS "xsimd not found — building without SIMD acceleration")
endif()
target_compile_features(benchmark_ext PRIVATE cxx_std_20)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

target_compile_options(benchmark_ext PRIVATE -O3 -march=native)
