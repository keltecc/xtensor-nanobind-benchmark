# xtensor-nanobind benchmark (AI generated)

Benchmark suite comparing nanobind's xtensor binding types against numpy and native C++ baselines.

## What it measures

| Category | Description |
|---|---|
| **Call overhead** | Minimal-work function call — isolates type caster/conversion cost |
| **Sum reduction (1D/2D)** | Iterate all elements via `xt::sum` — measures iteration speed |
| **Element-wise compute** | `sin(a) * s + t` — tests xtensor expression template evaluation |
| **Vectorization** | `abs()` on complex array — `nb::xvectorize` vs numpy |
| **Layout comparison** | `row_major` (flat pointer) vs `dynamic` (stepper) view iteration |

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
- [nanobind](https://github.com/wjakob/nanobind) source with xtensor bindings
- [xtensor](https://github.com/xtensor-stack/xtensor) >= 0.26.0
- [xtl](https://github.com/xtensor-stack/xtl)
- [xsimd](https://github.com/xtensor-stack/xsimd) (optional — enables SIMD-accelerated xtensor operations)

### Installing xtensor/xtl/xsimd

With apt (Debian/Ubuntu):
```bash
sudo apt install libxtensor-dev xtl-dev libxsimd-dev
```

With conda:
```bash
conda install xtensor xtl xsimd -c conda-forge
```

Or build from source:
```bash
git clone https://github.com/xtensor-stack/xtl && cd xtl && cmake -B build && cd ..
git clone https://github.com/xtensor-stack/xsimd && cd xsimd && cmake -B build && cd ..
git clone https://github.com/xtensor-stack/xtensor && cd xtensor && cmake -B build -Dxtl_DIR=../xtl/build && cd ..
```

## Quick start

```bash
export NANOBIND_DIR=/path/to/nanobind

# Build (creates venv, installs numpy, compiles C++ module)
make build

# Run benchmarks
make benchmark
```

The build step automatically creates a `.venv/` virtual environment, installs numpy into it, and builds the C++ extension against that Python.

### Custom xtensor/xtl paths

If xtensor and xtl are not installed system-wide, pass their cmake config directories:

```bash
make build XTENSOR_DIR=/path/to/xtensor/build XTL_DIR=/path/to/xtl/build
```

### Enabling xsimd (SIMD acceleration)

xsimd is optional. When detected, xtensor operations (`xt::sum`, `xt::sin`, etc.) use SIMD instructions (NEON on ARM, SSE/AVX on x86), which can significantly speed up computation.

```bash
make build XSIMD_DIR=/path/to/xsimd/build
```

The benchmark output shows whether xsimd is enabled (`xsimd: yes/no`).

### Custom compiler

```bash
make build CMAKE_CXX_COMPILER=g++-14
```

## Manual build

```bash
# Create venv
python3 -m venv .venv
.venv/bin/pip install numpy

# Build
cmake -S . -B build \
    -DNANOBIND_DIR=/path/to/nanobind \
    -DPython_ROOT_DIR=.venv \
    -Dxtensor_DIR=/path/to/xtensor/build \
    -Dxtl_DIR=/path/to/xtl/build \
    -Dxsimd_DIR=/path/to/xsimd/build \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build -j

# Run
PYTHONPATH=build .venv/bin/python scripts/run_benchmarks.py
```

## Sample output

```
==============================================================
  xtensor-nanobind benchmark
==============================================================
  1D array:     1,000,000 float64 elements
  2D array:  1000x 1000 float64 elements
  Iterations: 1000
  Python:     3.14.4
  NumPy:      2.4.4
  xsimd:      yes
==============================================================

=== Call Overhead (return first element) ===
+----------------+--------------+----------+
| Backend        |    Time/call |  Speedup |
+----------------+--------------+----------+
| numpy          |        35 ns |  1.0000x |
| ndarray        |        92 ns |  0.3771x |
| xarray_view    |       118 ns |  0.2947x |
| xtensor_view   |        73 ns |  0.4790x |
| xarray (copy)  |     145.0 us |  0.0002x |
| xtensor (copy) |     133.7 us |  0.0003x |
+----------------+--------------+----------+

=== Sum Reduction (1D, 1,000,000 float64) ===
+----------------+--------------+----------+
| Backend        |    Time/call |  Speedup |
+----------------+--------------+----------+
| numpy          |     118.1 us |  1.0000x |
| ndarray        |     389.3 us |  0.3033x |
| xarray_view    |     390.7 us |  0.3021x |
| xtensor_view   |     391.6 us |  0.3015x |
| xarray (copy)  |     602.2 us |  0.1960x |
| xtensor (copy) |     596.6 us |  0.1979x |
| native C++     |     389.0 us |  0.3034x |
+----------------+--------------+----------+

=== Sum Reduction (2D, 1000x1000 float64) ===
+----------------+--------------+----------+
| Backend        |    Time/call |  Speedup |
+----------------+--------------+----------+
| numpy          |     118.1 us |  1.0000x |
| ndarray        |     393.4 us |  0.3001x |
| xarray_view    |     371.7 us |  0.3176x |
| xtensor_view   |     374.1 us |  0.3156x |
| xarray (copy)  |     545.4 us |  0.2165x |
| xtensor (copy) |     521.3 us |  0.2265x |
| native C++     |     390.5 us |  0.3024x |
+----------------+--------------+----------+

=== Element-wise sin(a)*s+t (1D, 1,000,000 float64) ===
+----------------+--------------+----------+
| Backend        |    Time/call |  Speedup |
+----------------+--------------+----------+
| numpy          |      6.44 ms |  1.0000x |
| xarray_view    |     886.4 us |  7.2633x |
| xtensor_view   |     899.2 us |  7.1605x |
| xarray (copy)  |      1.25 ms |  5.1583x |
| xtensor (copy) |      1.24 ms |  5.1966x |
| native C++     |      6.08 ms |  1.0582x |
+----------------+--------------+----------+

=== Vectorization abs(x) (float64, 1,000,000) ===
+----------------+--------------+----------+
| Backend        |    Time/call |  Speedup |
+----------------+--------------+----------+
| np.abs         |     121.9 us |  1.0000x |
| np.vectorize   |     49.04 ms |  0.0025x |
| nb::xvectorize |     111.3 us |  1.0949x |
+----------------+--------------+----------+

=== Layout: row_major vs dynamic (xarray_view) ===
+--------------+--------------+----------+
| Backend      |    Time/call |  Speedup |
+--------------+--------------+----------+
| 1D row_major |     392.0 us |  1.0000x |
| 1D dynamic   |     395.0 us |  0.9923x |
| 2D row_major |     368.7 us |  1.0631x |
| 2D dynamic   |     376.5 us |  1.0411x |
+--------------+--------------+----------+
```

(Actual results depend on hardware, compiler, and numpy version.)
