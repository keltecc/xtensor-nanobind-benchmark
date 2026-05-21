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

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# =============================================================================
# nanobind backend (always built)
# =============================================================================

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(NATIVE_ARCH_FLAG -march=native)
else()
    set(NATIVE_ARCH_FLAG)
endif()

nanobind_add_module(benchmark_ext src/benchmark.cpp)
target_link_libraries(benchmark_ext PRIVATE xtensor xtl)
target_compile_features(benchmark_ext PRIVATE cxx_std_20)
target_compile_options(benchmark_ext PRIVATE -O3 ${NATIVE_ARCH_FLAG})

if(xsimd_FOUND)
    target_link_libraries(benchmark_ext PRIVATE xsimd)
    target_compile_definitions(benchmark_ext PRIVATE XTENSOR_USE_XSIMD)
    message(STATUS "xsimd found — SIMD-accelerated xtensor enabled")
else()
    message(STATUS "xsimd not found — building without SIMD acceleration")
endif()

# =============================================================================
# pybind11 + xtensor-python backend (optional, built when XTENSOR_PYTHON_DIR set)
# =============================================================================

if(NOT XTENSOR_PYTHON_DIR AND DEFINED ENV{XTENSOR_PYTHON_DIR})
    set(XTENSOR_PYTHON_DIR "$ENV{XTENSOR_PYTHON_DIR}")
endif()

if(XTENSOR_PYTHON_DIR)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(benchmark_xtp_ext src/benchmark_xtensor_python.cpp)
    target_include_directories(benchmark_xtp_ext PRIVATE
        ${XTENSOR_PYTHON_DIR}/include)
    target_link_libraries(benchmark_xtp_ext PRIVATE xtensor xtl)
    target_compile_features(benchmark_xtp_ext PRIVATE cxx_std_20)
    target_compile_options(benchmark_xtp_ext PRIVATE -O3 ${NATIVE_ARCH_FLAG})

    # NumPy headers — xtensor-python requires them directly.
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    target_include_directories(benchmark_xtp_ext PRIVATE ${NUMPY_INCLUDE_DIR})

    if(xsimd_FOUND)
        target_link_libraries(benchmark_xtp_ext PRIVATE xsimd)
        target_compile_definitions(benchmark_xtp_ext PRIVATE XTENSOR_USE_XSIMD)
    endif()
    message(STATUS "xtensor-python backend enabled: ${XTENSOR_PYTHON_DIR}")
else()
    message(STATUS "XTENSOR_PYTHON_DIR not set — xtensor-python backend skipped")
endif()
