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

Example build from source:

```bash
git clone https://github.com/xtensor-stack/xtl /tmp/xtl && cd /tmp/xtl && cmake . && sudo make install
git clone https://github.com/xtensor-stack/xsimd /tmp/xsimd && cd /tmp/xsimd && cmake . && sudo make install
git clone https://github.com/xtensor-stack/xtensor /tmp/xtensor && cd /tmp/xtensor && cmake . && sudo make install
git clone https://github.com/xtensor-stack/xtensor-python /tmp/xtensor-python
git clone https://github.com/wjakob/nanobind --recursive /tmp/nanobind
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
    XTENSOR_PYTHON_DIR=/path/to/xtensor-python \
    XTENSOR_DIR=/path/to/xtensor/build \
    XTL_DIR=/path/to/xtl/build \
    XSIMD_DIR=/path/to/xsimd/build
make benchmark
```

The benchmark runner automatically detects which backends are available (nanobind, xtensor-python, or both) and prints a cross-framework comparison table when both are present.

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
| numpy         |        43 ns |
| ndarray       |        84 ns |
| view          |       109 ns |
| tensor_view   |        77 ns |
| owning array  |     110.4 us |
| owning tensor |     106.8 us |
+---------------+--------------+

=== Sum Reduction (1D, 1,000,000 float64) ===
+---------------+--------------+
| Backend       |    Time/call |
+---------------+--------------+
| numpy         |     118.0 us |
| ndarray       |     390.1 us |
| view          |     392.9 us |
| tensor_view   |     391.3 us |
| owning array  |     520.3 us |
| owning tensor |     526.6 us |
| native        |     389.6 us |
+---------------+--------------+

=== Sum Reduction (2D, 1000x1000 float64) ===
+---------------+--------------+
| Backend       |    Time/call |
+---------------+--------------+
| numpy         |     120.5 us |
| ndarray       |     391.3 us |
| view          |     365.8 us |
| tensor_view   |     364.0 us |
| owning array  |     543.8 us |
| owning tensor |     517.0 us |
| native        |     388.5 us |
+---------------+--------------+

=== Element-wise sin(a)*s+t (1D, 1,000,000 float64) ===
+---------------+--------------+
| Backend       |    Time/call |
+---------------+--------------+
| numpy         |      6.46 ms |
| view          |     811.2 us |
| tensor_view   |     825.2 us |
| owning array  |     997.5 us |
| owning tensor |      1.02 ms |
| native        |      6.17 ms |
+---------------+--------------+

=== 2D Compute sin(a)*s+t (1000x1000, layout variants) ===
+-----------------------+--------------+
| Backend               |    Time/call |
+-----------------------+--------------+
| xarray row_major      |      1.05 ms |
| xarray col_major      |      8.67 ms |
| xarray dynamic        |      1.05 ms |
| xtensor row_major     |      1.03 ms |
| xtensor col_major     |      8.69 ms |
| xtensor dynamic       |      1.05 ms |
| view row_major        |     813.4 us |
| view col_major        |      8.35 ms |
| view dynamic          |     813.0 us |
| tensor_view row_major |     817.3 us |
| tensor_view col_major |      8.30 ms |
| tensor_view dynamic   |     836.4 us |
+-----------------------+--------------+

=== Vectorization abs(x) (float64, 1,000,000) ===
+--------------+--------------+
| Backend      |    Time/call |
+--------------+--------------+
| np.abs       |     142.9 us |
| np.vectorize |     50.76 ms |
| xvectorize   |     138.4 us |
+--------------+--------------+

##############################################################
#  Running xtensor-python backend
##############################################################

=== Call Overhead (return first element) ===
+---------------+--------------+
| Backend       |    Time/call |
+---------------+--------------+
| numpy         |        36 ns |
| ndarray       |       308 ns |
| view          |        86 ns |
| tensor_view   |        98 ns |
| owning array  |     132.1 us |
| owning tensor |     113.6 us |
+---------------+--------------+

=== Sum Reduction (1D, 1,000,000 float64) ===
+---------------+--------------+
| Backend       |    Time/call |
+---------------+--------------+
| numpy         |     116.9 us |
| ndarray       |     394.2 us |
| view          |      2.33 ms |
| tensor_view   |     389.9 us |
| owning array  |     546.5 us |
| owning tensor |     558.4 us |
| native        |     391.1 us |
+---------------+--------------+

=== Sum Reduction (2D, 1000x1000 float64) ===
+---------------+--------------+
| Backend       |    Time/call |
+---------------+--------------+
| numpy         |     119.2 us |
| ndarray       |     389.2 us |
| view          |     391.7 us |
| tensor_view   |     367.5 us |
| owning array  |     522.9 us |
| owning tensor |     511.5 us |
| native        |     388.9 us |
+---------------+--------------+

=== Element-wise sin(a)*s+t (1D, 1,000,000 float64) ===
+---------------+--------------+
| Backend       |    Time/call |
+---------------+--------------+
| numpy         |      6.48 ms |
| view          |     843.4 us |
| tensor_view   |     791.7 us |
| owning array  |      3.01 ms |
| owning tensor |      2.96 ms |
| native        |      6.16 ms |
+---------------+--------------+

=== 2D Compute sin(a)*s+t (1000x1000, layout variants) ===
+-----------------------+--------------+
| Backend               |    Time/call |
+-----------------------+--------------+
| xarray row_major      |      1.13 ms |
| xarray col_major      |      8.93 ms |
| xarray dynamic        |      1.11 ms |
| xtensor row_major     |      1.11 ms |
| xtensor col_major     |      9.08 ms |
| xtensor dynamic       |      1.09 ms |
| view row_major        |     839.6 us |
| view col_major        |      8.81 ms |
| view dynamic          |     848.0 us |
| tensor_view row_major |     794.0 us |
| tensor_view col_major |      7.54 ms |
| tensor_view dynamic   |     805.8 us |
+-----------------------+--------------+

=== Vectorization abs(x) (float64, 1,000,000) ===
+--------------+--------------+
| Backend      |    Time/call |
+--------------+--------------+
| np.abs       |     124.9 us |
| np.vectorize |     50.87 ms |
| xvectorize   |     127.2 us |
+--------------+--------------+

##############################################################
#  Cross-backend summary
##############################################################

=== Nanobind vs xtensor-python ===
+-------------------------------------------+----------+----------------+
| Metric                                    | nanobind | xtensor-python |
+-------------------------------------------+----------+----------------+
| Call overhead  [ndarray]                  |    84 ns |         308 ns |
| Call overhead  [view]                     |   109 ns |          86 ns |
| Call overhead  [owning array]             | 110.4 us |       132.1 us |
| Sum 1D  [ndarray]                         | 390.1 us |       394.2 us |
| Sum 1D  [view]                            | 392.9 us |        2.33 ms |
| Sum 1D  [owning array]                    | 520.3 us |       546.5 us |
| Sum 2D  [ndarray]                         | 391.3 us |       389.2 us |
| Sum 2D  [view]                            | 365.8 us |       391.7 us |
| Sum 2D  [owning array]                    | 543.8 us |       522.9 us |
| Compute sin(a)*s+t  [view]                | 811.2 us |       843.4 us |
| Compute sin(a)*s+t  [owning array]        | 997.5 us |        3.01 ms |
| Compute2D xarray  [xarray row_major]      |  1.05 ms |        1.13 ms |
| Compute2D xarray  [xarray dynamic]        |  1.05 ms |        1.11 ms |
| Compute2D xtensor  [xtensor row_major]    |  1.03 ms |        1.11 ms |
| Compute2D xtensor  [xtensor dynamic]      |  1.05 ms |        1.09 ms |
| Compute2D view  [view row_major]          | 813.4 us |       839.6 us |
| Compute2D view  [view dynamic]            | 813.0 us |       848.0 us |
| Compute2D t.view  [tensor_view row_major] | 817.3 us |       794.0 us |
| Compute2D t.view  [tensor_view dynamic]   | 836.4 us |       805.8 us |
| Vectorize abs  [xvectorize]               | 138.4 us |       127.2 us |
+-------------------------------------------+----------+----------------+
```

(Actual results depend on hardware, compiler, and numpy version.)
