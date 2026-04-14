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

```

(Actual results depend on hardware, compiler, and numpy version.)
