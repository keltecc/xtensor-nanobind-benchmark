# xtensor-nanobind benchmark (AI generated)

Benchmark suite comparing nanobind's xtensor binding types against numpy and native C++ baselines, with optional xtensor-python cross-framework comparison.

## What it measures

| Category | Description |
|---|---|
| **Call overhead** | Minimal-work function call — isolates type caster/conversion cost |
| **Sum reduction (1D/2D)** | Iterate all elements via `xt::sum` — measures iteration speed |
| **Element-wise compute** | `sin(a) * s + t` — tests xtensor expression template evaluation |
| **2D compute layouts** | `sin(a) * s + t` on 2D arrays with explicit `row_major` / `column_major` / `dynamic` layouts across all binding types |
| **Vectorization** | `abs()` on centered array — `nb::xvectorize` vs numpy |

### Binding types compared

- **`xt::xarray<T>`** — owning, dynamic ndim (copies data from numpy)
- **`xt::xtensor<T, N>`** — owning, fixed ndim (copies data from numpy)
- **`nb::xarray_view<T>`** — non-owning view, dynamic ndim (zero-copy)
- **`nb::xtensor_view<T, N>`** — non-owning view, fixed ndim (zero-copy)
- **`nb::ndarray`** — nanobind's native numpy wrapper (raw pointer access)
- **numpy** — pure Python numpy operations
- **native C++** — same operation timed entirely in C++ (no per-call Python overhead)

## Prerequisites

- Python 3.9+
- CMake 3.20+
- C++20 compiler (GCC 10+, Clang 13+)
- [nanobind](https://github.com/wjakob/nanobind) source with xtensor bindings (clone with `--recursive` to pull pybind11 and robin-map submodules)
- [xtensor](https://github.com/xtensor-stack/xtensor) >= 0.26.0
- [xtl](https://github.com/xtensor-stack/xtl)
- [xsimd](https://github.com/xtensor-stack/xsimd) (optional — enables SIMD-accelerated xtensor operations)
- [xtensor-python](https://github.com/xtensor-stack/xtensor-python) (optional — enables cross-framework comparison against pybind11-based bindings)

### Installing xtensor / xtl / xsimd / xtensor-python

All four are header-only libraries. Building them with CMake generates the config files needed by `find_package()`.

With apt (Debian/Ubuntu):
```bash
sudo apt install libxtensor-dev xtl-dev libxsimd-dev
```
(xtensor-python is not packaged in apt — build from source or use conda.)

With conda:
```bash
conda install xtensor xtl xsimd xtensor-python -c conda-forge
```

Or build from source:
```bash
git clone https://github.com/xtensor-stack/xtl       && cd xtl       && cmake -B build && cd ..
git clone https://github.com/xtensor-stack/xsimd     && cd xsimd     && cmake -B build && cd ..
git clone https://github.com/xtensor-stack/xtensor   && cd xtensor   && cmake -B build -Dxtl_DIR=../xtl/build && cd ..
git clone https://github.com/xtensor-stack/xtensor-python && cd xtensor-python && cmake -B build && cd ..
```

## Quick start

```bash
export NANOBIND_DIR=/path/to/nanobind

# Build with default settings (nanobind only)
make build

# Run benchmarks
make benchmark
```

The build step automatically creates a `.venv/` virtual environment, installs numpy into it, and builds the C++ extension against that Python.

### With xtensor-python (cross-framework comparison)

Pass paths to dependencies that are not installed system-wide via their `*_DIR` variables:

```bash
make clean
make build \
    NANOBIND_DIR=/path/to/nanobind \
    XTENSOR_DIR=/path/to/xtensor/build \
    XTL_DIR=/path/to/xtl/build \
    XSIMD_DIR=/path/to/xsimd/build \
    XTENSOR_PYTHON_DIR=/path/to/xtensor-python/build
make benchmark
```

The benchmark runner automatically detects which backends are available (nanobind, xtensor-python, or both) and prints a cross-framework comparison table when both are present.

### Custom compiler

```bash
make build CMAKE_CXX_COMPILER=g++-14
```

## Manual build

```bash
# Create venv
python3 -m venv .venv
.venv/bin/pip install numpy

# Build (nanobind only)
cmake -S . -B build \
    -DNANOBIND_DIR=/path/to/nanobind \
    -DPython_ROOT_DIR=.venv \
    -Dxtensor_DIR=/path/to/xtensor/build \
    -Dxtl_DIR=/path/to/xtl/build \
    -Dxsimd_DIR=/path/to/xsimd/build \
    -DCMAKE_BUILD_TYPE=Release

# With xtensor-python, add:
#   -DXTENSOR_PYTHON_DIR=/path/to/xtensor-python/build

cmake --build build -j

# Run
PYTHONPATH=build .venv/bin/python scripts/run_benchmarks.py
```

## Sample output

```
==============================================================
  xtensor bindings benchmark (nanobind vs xtensor-python)
==============================================================
  OS:         Linux 6.8.0-87-generic (x86_64)
  CPU:        AMD Ryzen 9 9950X 16-Core Processor
  1D array:     1,000,000 float64 elements
  2D array:  1000x 1000 float64 elements
  Iterations: 1000
  Python:     3.14.4
  NumPy:      2.4.6
  Backends:   nanobind, xtensor-python
    nanobind         xsimd=yes
    xtensor-python   xsimd=yes
==============================================================

##############################################################
#  Running nanobind backend
##############################################################

=== Call Overhead (return first element) ===
+---------------+--------------+
| Backend       |    Time/call |
+---------------+--------------+
| numpy         |        34 ns |
| ndarray       |        87 ns |
| view          |       115 ns |
| tensor_view   |        76 ns |
| owning array  |     116.0 us |
| owning tensor |     109.5 us |
+---------------+--------------+

=== Sum Reduction (1D, 1,000,000 float64) ===
+---------------+--------------+
| Backend       |    Time/call |
+---------------+--------------+
| numpy         |     118.0 us |
| ndarray       |     391.4 us |
| view          |     390.7 us |
| tensor_view   |     387.3 us |
| owning array  |     517.6 us |
| owning tensor |     527.7 us |
| native        |     387.2 us |
+---------------+--------------+

=== Sum Reduction (2D, 1000x1000 float64) ===
+---------------+--------------+
| Backend       |    Time/call |
+---------------+--------------+
| numpy         |     120.0 us |
| ndarray       |     395.9 us |
| view          |     369.0 us |
| tensor_view   |     368.9 us |
| owning array  |     493.3 us |
| owning tensor |     510.8 us |
| native        |     392.0 us |
+---------------+--------------+

=== Element-wise sin(a)*s+t (1D, 1,000,000 float64) ===
+---------------+--------------+
| Backend       |    Time/call |
+---------------+--------------+
| numpy         |      6.45 ms |
| view          |     822.2 us |
| tensor_view   |     828.9 us |
| owning array  |     983.3 us |
| owning tensor |      1.04 ms |
| native        |      6.19 ms |
+---------------+--------------+

=== 2D Compute sin(a)*s+t (1000x1000, layout variants) ===
+-----------------------+--------------+
| Backend               |    Time/call |
+-----------------------+--------------+
| xarray row_major      |      1.00 ms |
| xarray col_major      |      8.51 ms |
| xarray dynamic        |      1.02 ms |
| xtensor row_major     |      1.03 ms |
| xtensor col_major     |      8.37 ms |
| xtensor dynamic       |      1.01 ms |
| view row_major        |     820.5 us |
| view col_major        |      8.24 ms |
| view dynamic          |     813.5 us |
| tensor_view row_major |     819.6 us |
| tensor_view col_major |      8.92 ms |
| tensor_view dynamic   |     815.0 us |
+-----------------------+--------------+

=== Vectorization abs(x) (float64, 1,000,000) ===
+--------------+--------------+
| Backend      |    Time/call |
+--------------+--------------+
| np.abs       |     128.6 us |
| np.vectorize |     50.87 ms |
| xvectorize   |     142.8 us |
+--------------+--------------+

##############################################################
#  Running xtensor-python backend
##############################################################

=== Call Overhead (return first element) ===
+---------------+--------------+
| Backend       |    Time/call |
+---------------+--------------+
| numpy         |        48 ns |
| ndarray       |       282 ns |
| view          |        86 ns |
| tensor_view   |        84 ns |
| owning array  |     122.2 us |
| owning tensor |     130.6 us |
+---------------+--------------+

=== Sum Reduction (1D, 1,000,000 float64) ===
+---------------+--------------+
| Backend       |    Time/call |
+---------------+--------------+
| numpy         |     129.9 us |
| ndarray       |     389.9 us |
| view          |      2.33 ms |
| tensor_view   |     391.3 us |
| owning array  |     530.0 us |
| owning tensor |     529.9 us |
| native        |     390.3 us |
+---------------+--------------+

=== Sum Reduction (2D, 1000x1000 float64) ===
+---------------+--------------+
| Backend       |    Time/call |
+---------------+--------------+
| numpy         |     118.5 us |
| ndarray       |     390.1 us |
| view          |     389.9 us |
| tensor_view   |     369.7 us |
| owning array  |     499.9 us |
| owning tensor |     509.0 us |
| native        |     389.9 us |
+---------------+--------------+

=== Element-wise sin(a)*s+t (1D, 1,000,000 float64) ===
+---------------+--------------+
| Backend       |    Time/call |
+---------------+--------------+
| numpy         |      6.50 ms |
| view          |     829.4 us |
| tensor_view   |     801.0 us |
| owning array  |      3.85 ms |
| owning tensor |      3.81 ms |
| native        |      6.24 ms |
+---------------+--------------+

=== 2D Compute sin(a)*s+t (1000x1000, layout variants) ===
+-----------------------+--------------+
| Backend               |    Time/call |
+-----------------------+--------------+
| xarray row_major      |      3.81 ms |
| xarray col_major      |     11.93 ms |
| xarray dynamic        |      3.74 ms |
| xtensor row_major     |      3.80 ms |
| xtensor col_major     |     10.85 ms |
| xtensor dynamic       |      3.79 ms |
| view row_major        |     838.7 us |
| view col_major        |      8.63 ms |
| view dynamic          |     837.9 us |
| tensor_view row_major |     801.9 us |
| tensor_view col_major |      7.54 ms |
| tensor_view dynamic   |     806.8 us |
+-----------------------+--------------+

=== Vectorization abs(x) (float64, 1,000,000) ===
+--------------+--------------+
| Backend      |    Time/call |
+--------------+--------------+
| np.abs       |     144.5 us |
| np.vectorize |     50.20 ms |
| xvectorize   |     140.3 us |
+--------------+--------------+

##############################################################
#  Cross-backend summary
##############################################################

=== Nanobind vs xtensor-python ===
+-------------------------------------------+----------+----------------+
| Metric                                    | nanobind | xtensor-python |
+-------------------------------------------+----------+----------------+
| Call overhead  [ndarray]                  |    87 ns |         282 ns |
| Call overhead  [view]                     |   115 ns |          86 ns |
| Call overhead  [owning array]             | 116.0 us |       122.2 us |
| Sum 1D  [ndarray]                         | 391.4 us |       389.9 us |
| Sum 1D  [view]                            | 390.7 us |        2.33 ms |
| Sum 1D  [owning array]                    | 517.6 us |       530.0 us |
| Sum 2D  [ndarray]                         | 395.9 us |       390.1 us |
| Sum 2D  [view]                            | 369.0 us |       389.9 us |
| Sum 2D  [owning array]                    | 493.3 us |       499.9 us |
| Compute sin(a)*s+t  [view]                | 822.2 us |       829.4 us |
| Compute sin(a)*s+t  [owning array]        | 983.3 us |        3.85 ms |
| Compute2D xarray  [xarray row_major]      |  1.00 ms |        3.81 ms |
| Compute2D xarray  [xarray dynamic]        |  1.02 ms |        3.74 ms |
| Compute2D xtensor  [xtensor row_major]    |  1.03 ms |        3.80 ms |
| Compute2D xtensor  [xtensor dynamic]      |  1.01 ms |        3.79 ms |
| Compute2D view  [view row_major]          | 820.5 us |       838.7 us |
| Compute2D view  [view dynamic]            | 813.5 us |       837.9 us |
| Compute2D t.view  [tensor_view row_major] | 819.6 us |       801.9 us |
| Compute2D t.view  [tensor_view dynamic]   | 815.0 us |       806.8 us |
| Vectorize abs  [xvectorize]               | 142.8 us |       140.3 us |
+-------------------------------------------+----------+----------------+
```

(Actual results depend on hardware, compiler, and numpy version.)
