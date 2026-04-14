#!/usr/bin/env python3
"""
xtensor-nanobind benchmark suite.

Measures conversion overhead, iteration speed, expression evaluation,
and vectorization across nanobind/xtensor binding strategies.
"""

import timeit
import sys
import numpy as np

import benchmark_ext as bm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_1D = 1_000_000
N_2D_ROWS = 1000
N_2D_COLS = 1000
ITERATIONS = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bench(stmt, number=ITERATIONS, globals_dict=None):
    """Run timeit, return seconds per call."""
    total = timeit.timeit(stmt, number=number, globals=globals_dict)
    return total / number


def format_time(seconds):
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def print_table(title, rows, baseline_key="numpy"):
    """Print a formatted benchmark table.

    rows: list of (name, time_per_call_seconds)
    """
    baseline = None
    for name, t in rows:
        if name == baseline_key:
            baseline = t
            break
    if baseline is None and rows:
        baseline = rows[0][1]

    name_w = max(len(r[0]) for r in rows)
    name_w = max(name_w, 7)  # min width for "Backend"

    print(f"\n=== {title} ===")
    print(f"+-{'-' * name_w}-+-{'-' * 12}-+-{'-' * 8}-+")
    print(f"| {'Backend':<{name_w}} | {'Time/call':>12} | {'Speedup':>8} |")
    print(f"+-{'-' * name_w}-+-{'-' * 12}-+-{'-' * 8}-+")
    for name, t in rows:
        ratio = baseline / t if t > 0 else float("inf")
        ratio_str = f"{ratio:.4f}x"
        print(f"| {name:<{name_w}} | {format_time(t):>12} | {ratio_str:>8} |")
    print(f"+-{'-' * name_w}-+-{'-' * 12}-+-{'-' * 8}-+")


# ---------------------------------------------------------------------------
# Benchmark categories
# ---------------------------------------------------------------------------

def run_call_overhead():
    a = np.random.rand(N_1D)
    g = {"a": a, "bm": bm, "np": np}

    rows = [
        ("numpy",          bench("a[0]", globals_dict=g)),
        ("ndarray",        bench("bm.noop_ndarray(a)", globals_dict=g)),
        ("xarray_view",    bench("bm.noop_xarray_view(a)", globals_dict=g)),
        ("xtensor_view",   bench("bm.noop_xtensor_view(a)", globals_dict=g)),
        ("xarray (copy)",  bench("bm.noop_xarray(a)", globals_dict=g)),
        ("xtensor (copy)", bench("bm.noop_xtensor(a)", globals_dict=g)),
    ]
    print_table("Call Overhead (return first element)", rows)


def run_sum_1d():
    a = np.random.rand(N_1D)
    g = {"a": a, "bm": bm, "np": np}

    _, native_ms = bm.native_sum(a, ITERATIONS)
    native_per_call = native_ms / 1000.0 / ITERATIONS

    rows = [
        ("numpy",          bench("np.sum(a)", globals_dict=g)),
        ("ndarray",        bench("bm.sum_ndarray(a)", globals_dict=g)),
        ("xarray_view",    bench("bm.sum_xarray_view(a)", globals_dict=g)),
        ("xtensor_view",   bench("bm.sum_xtensor_view(a)", globals_dict=g)),
        ("xarray (copy)",  bench("bm.sum_xarray(a)", globals_dict=g)),
        ("xtensor (copy)", bench("bm.sum_xtensor(a)", globals_dict=g)),
        ("native C++",     native_per_call),
    ]
    print_table(f"Sum Reduction (1D, {N_1D:,} float64)", rows)


def run_sum_2d():
    a = np.random.rand(N_2D_ROWS, N_2D_COLS)
    g = {"a": a, "bm": bm, "np": np}

    _, native_ms = bm.native_sum2d(a, ITERATIONS)
    native_per_call = native_ms / 1000.0 / ITERATIONS

    rows = [
        ("numpy",          bench("np.sum(a)", globals_dict=g)),
        ("ndarray",        bench("bm.sum2d_ndarray(a)", globals_dict=g)),
        ("xarray_view",    bench("bm.sum2d_xarray_view(a)", globals_dict=g)),
        ("xtensor_view",   bench("bm.sum2d_xtensor_view(a)", globals_dict=g)),
        ("xarray (copy)",  bench("bm.sum2d_xarray(a)", globals_dict=g)),
        ("xtensor (copy)", bench("bm.sum2d_xtensor(a)", globals_dict=g)),
        ("native C++",     native_per_call),
    ]
    print_table(f"Sum Reduction (2D, {N_2D_ROWS}x{N_2D_COLS} float64)", rows)


def run_compute():
    a = np.random.rand(N_1D)
    s, t = 2.5, 1.0
    g = {"a": a, "s": s, "t": t, "bm": bm, "np": np}

    _, native_ms = bm.native_compute(a, s, t, ITERATIONS)
    native_per_call = native_ms / 1000.0 / ITERATIONS

    rows = [
        ("numpy",          bench("np.sin(a) * s + t", globals_dict=g)),
        ("xarray_view",    bench("bm.compute_xarray_view(a, s, t)", globals_dict=g)),
        ("xtensor_view",   bench("bm.compute_xtensor_view(a, s, t)", globals_dict=g)),
        ("xarray (copy)",  bench("bm.compute_xarray(a, s, t)", globals_dict=g)),
        ("xtensor (copy)", bench("bm.compute_xtensor(a, s, t)", globals_dict=g)),
        ("native C++",     native_per_call),
    ]
    print_table(f"Element-wise sin(a)*s+t (1D, {N_1D:,} float64)", rows)


def run_vectorize():
    a = np.random.rand(N_1D) - 0.5  # centered so abs is non-trivial
    vf = np.vectorize(abs)
    g = {"a": a, "bm": bm, "np": np, "vf": vf}

    rows = [
        ("np.abs",           bench("np.abs(a)", globals_dict=g)),
        ("np.vectorize",     bench("vf(a)", globals_dict=g)),
        ("nb::xvectorize",   bench("bm.vectorize_abs(a)", globals_dict=g)),
    ]
    print_table(f"Vectorization abs(x) (float64, {N_1D:,})", rows,
                baseline_key="np.abs")


def run_layout_comparison():
    a1d = np.random.rand(N_1D)
    a2d = np.random.rand(N_2D_ROWS, N_2D_COLS)
    g = {"a1d": a1d, "a2d": a2d, "bm": bm}

    rows = [
        ("1D row_major",  bench("bm.sum_xarray_view(a1d)", globals_dict=g)),
        ("1D dynamic",    bench("bm.sum_xarray_view_dynamic(a1d)", globals_dict=g)),
        ("2D row_major",  bench("bm.sum2d_xarray_view(a2d)", globals_dict=g)),
        ("2D dynamic",    bench("bm.sum2d_xarray_view_dynamic(a2d)", globals_dict=g)),
    ]
    print_table("Layout: row_major vs dynamic (xarray_view)", rows,
                baseline_key="1D row_major")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("  xtensor-nanobind benchmark")
    print("=" * 62)
    print(f"  1D array:  {N_1D:>12,} float64 elements")
    print(f"  2D array:  {N_2D_ROWS}x{N_2D_COLS:>5} float64 elements")
    print(f"  Iterations: {ITERATIONS}")
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  NumPy:      {np.__version__}")
    print(f"  xsimd:      {'yes' if bm.has_xsimd else 'no'}")
    print("=" * 62)

    run_call_overhead()
    run_sum_1d()
    run_sum_2d()
    run_compute()
    run_vectorize()
    run_layout_comparison()

    print()


if __name__ == "__main__":
    main()
