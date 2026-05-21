#!/usr/bin/env python3
"""
xtensor-nanobind benchmark suite.

Runs the same set of benchmarks against two backends (when available):
  - nanobind  (benchmark_ext)
  - xtensor-python  (benchmark_xtp_ext)

Produces per-backend tables and a final side-by-side comparison.
"""

import argparse
import platform
import subprocess
import timeit
import sys
import numpy as np


def _cpu_brand():
    system = platform.system()
    try:
        if system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _os_name():
    system = platform.system()
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]} ({platform.machine()})"
    return f"{system} {platform.release()} ({platform.machine()})"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_1D = 1_000_000
N_2D_ROWS = 1000
N_2D_COLS = 1000
ITERATIONS = 1000


# ---------------------------------------------------------------------------
# Backends: map each generic slot name to the concrete function exported by
# each backend module. Missing keys mean the slot is not supported.
# ---------------------------------------------------------------------------

def _nanobind_backend(mod):
    return {
        "label": "nanobind",
        "mod": mod,
        "has_xsimd": mod.has_xsimd,
        "names": {
            # call overhead
            "noop_ndarray":        "noop_ndarray",
            "noop_view":           "noop_xarray_view",
            "noop_tensor_view":    "noop_xtensor_view",
            "noop_owning":         "noop_xarray",
            "noop_owning_tensor":  "noop_xtensor",
            # sum 1D
            "sum_ndarray":         "sum_ndarray",
            "sum_view":            "sum_xarray_view",
            "sum_tensor_view":     "sum_xtensor_view",
            "sum_owning":          "sum_xarray",
            "sum_owning_tensor":   "sum_xtensor",
            # sum 2D
            "sum2d_ndarray":       "sum2d_ndarray",
            "sum2d_view":          "sum2d_xarray_view",
            "sum2d_tensor_view":   "sum2d_xtensor_view",
            "sum2d_owning":        "sum2d_xarray",
            "sum2d_owning_tensor": "sum2d_xtensor",
            # compute
            "compute_view":        "compute_xarray_view",
            "compute_tensor_view": "compute_xtensor_view",
            "compute_owning":      "compute_xarray",
            "compute_owning_tensor": "compute_xtensor",
            # vectorize
            "vectorize_abs":       "vectorize_abs",
            # compute 2D layouts
            "compute2d_xarray_row_major":     "compute2d_xarray_row_major",
            "compute2d_xarray_column_major":  "compute2d_xarray_column_major",
            "compute2d_xarray_dynamic":       "compute2d_xarray_dynamic",
            "compute2d_xtensor_row_major":    "compute2d_xtensor_row_major",
            "compute2d_xtensor_column_major": "compute2d_xtensor_column_major",
            "compute2d_xtensor_dynamic":      "compute2d_xtensor_dynamic",
            "compute2d_view_row_major":       "compute2d_xarray_view_row_major",
            "compute2d_view_column_major":    "compute2d_xarray_view_column_major",
            "compute2d_view_dynamic":         "compute2d_xarray_view_dynamic",
            "compute2d_tensor_view_row_major":    "compute2d_xtensor_view_row_major",
            "compute2d_tensor_view_column_major": "compute2d_xtensor_view_column_major",
            "compute2d_tensor_view_dynamic":      "compute2d_xtensor_view_dynamic",
            # native
            "native_sum":          "native_sum",
            "native_compute":      "native_compute",
            "native_sum2d":        "native_sum2d",
        },
    }


def _xtensor_python_backend(mod):
    return {
        "label": "xtensor-python",
        "mod": mod,
        "has_xsimd": mod.has_xsimd,
        "names": {
            "noop_ndarray":        "noop_array_t",
            "noop_view":           "noop_pyarray",
            "noop_tensor_view":    "noop_pytensor",
            "noop_owning":         "noop_xarray",
            "noop_owning_tensor":  "noop_xtensor",
            "sum_ndarray":         "sum_array_t",
            "sum_view":            "sum_pyarray",
            "sum_tensor_view":     "sum_pytensor",
            "sum_owning":          "sum_xarray",
            "sum_owning_tensor":   "sum_xtensor",
            "sum2d_ndarray":       "sum2d_array_t",
            "sum2d_view":          "sum2d_pyarray",
            "sum2d_tensor_view":   "sum2d_pytensor",
            "sum2d_owning":        "sum2d_xarray",
            "sum2d_owning_tensor": "sum2d_xtensor",
            "compute_view":        "compute_pyarray",
            "compute_tensor_view": "compute_pytensor",
            "compute_owning":      "compute_xarray",
            "compute_owning_tensor": "compute_xtensor",
            "vectorize_abs":       "vectorize_abs",
            # compute 2D layouts
            "compute2d_xarray_row_major":     "compute2d_xarray_row_major",
            "compute2d_xarray_column_major":  "compute2d_xarray_column_major",
            "compute2d_xarray_dynamic":       "compute2d_xarray_dynamic",
            "compute2d_xtensor_row_major":    "compute2d_xtensor_row_major",
            "compute2d_xtensor_column_major": "compute2d_xtensor_column_major",
            "compute2d_xtensor_dynamic":      "compute2d_xtensor_dynamic",
            "compute2d_view_row_major":       "compute2d_pyarray_row_major",
            "compute2d_view_column_major":    "compute2d_pyarray_column_major",
            "compute2d_view_dynamic":         "compute2d_pyarray_dynamic",
            "compute2d_tensor_view_row_major":    "compute2d_pytensor_row_major",
            "compute2d_tensor_view_column_major": "compute2d_pytensor_column_major",
            "compute2d_tensor_view_dynamic":      "compute2d_pytensor_dynamic",
            "native_sum":          "native_sum",
            "native_compute":      "native_compute",
            "native_sum2d":        "native_sum2d",
        },
    }


def load_backends(requested):
    """Import whichever backends are available (honoring the request filter)."""
    backends = []

    if requested in ("both", "nanobind"):
        try:
            import benchmark_ext as bm
            backends.append(_nanobind_backend(bm))
        except ImportError as e:
            if requested == "nanobind":
                print(f"benchmark_ext not importable: {e}", file=sys.stderr)
                sys.exit(1)

    if requested in ("both", "xtensor-python"):
        try:
            import benchmark_xtp_ext as bm
            backends.append(_xtensor_python_backend(bm))
        except ImportError as e:
            if requested == "xtensor-python":
                print(f"benchmark_xtp_ext not importable: {e}", file=sys.stderr)
                sys.exit(1)

    if not backends:
        print("No benchmark extensions found. Build them first with `make build` "
              "(optionally with XTENSOR_PYTHON_DIR=...).", file=sys.stderr)
        sys.exit(1)

    return backends


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bench(stmt, number=ITERATIONS, globals_dict=None):
    total = timeit.timeit(stmt, number=number, globals=globals_dict)
    return total / number


def format_time(seconds):
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def print_table(title, rows, baseline_key="numpy"):
    name_w = max(len(r[0]) for r in rows)
    name_w = max(name_w, 7)

    print(f"\n=== {title} ===")
    print(f"+-{'-' * name_w}-+-{'-' * 12}-+")
    print(f"| {'Backend':<{name_w}} | {'Time/call':>12} |")
    print(f"+-{'-' * name_w}-+-{'-' * 12}-+")
    for name, t in rows:
        print(f"| {name:<{name_w}} | {format_time(t):>12} |")
    print(f"+-{'-' * name_w}-+-{'-' * 12}-+")


def print_comparison_table(title, columns, rows):
    """rows: list of (metric_name, dict[column -> time_seconds]).

    columns: list of column labels (e.g. ['nanobind', 'xtensor-python']).
    """
    header = ["Metric"] + columns
    widths = [max(len(h), 8) for h in header]
    for name, vals in rows:
        widths[0] = max(widths[0], len(name))
        for i, col in enumerate(columns, 1):
            v = vals.get(col)
            widths[i] = max(widths[i], len(format_time(v)) if v else 1)

    def hline():
        return "+-" + "-+-".join("-" * w for w in widths) + "-+"

    print(f"\n=== {title} ===")
    print(hline())
    print("| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |")
    print(hline())
    for name, vals in rows:
        cells = [name.ljust(widths[0])]
        for i, col in enumerate(columns, 1):
            v = vals.get(col)
            cells.append(format_time(v).rjust(widths[i]) if v is not None
                         else "-".rjust(widths[i]))
        print("| " + " | ".join(cells) + " |")
    print(hline())


# ---------------------------------------------------------------------------
# Single-backend benchmark runs
# ---------------------------------------------------------------------------

def run_for_backend(backend):
    """Run all benchmark categories against one backend. Returns dict mapping
    metric name -> time per call (seconds) for downstream comparison."""
    label = backend["label"]
    mod = backend["mod"]
    names = backend["names"]

    print()
    print("#" * 62)
    print(f"#  Running {label} backend")
    print("#" * 62)

    results = {}
    g = {"bm": mod, "np": np}

    def get(slot):
        """Look up a function on the module via the name map."""
        cpp_name = names.get(slot)
        if cpp_name is None:
            return None
        return getattr(mod, cpp_name, None)

    # ---- Call overhead ----
    a = np.random.rand(N_1D)
    g["a"] = a
    for slot, label_ in [
        ("noop_ndarray",       "ndarray"),
        ("noop_view",          "view"),
        ("noop_tensor_view",   "tensor_view"),
        ("noop_owning",        "owning array"),
        ("noop_owning_tensor", "owning tensor"),
    ]:
        fn = get(slot)
        if fn is None:
            continue
        g["fn"] = fn
        results[f"noop/{label_}"] = bench("fn(a)", globals_dict=g)

    results["noop/numpy"] = bench("a[0]", globals_dict=g)

    rows = [("numpy", results["noop/numpy"])]
    for key in ["noop/ndarray", "noop/view", "noop/tensor_view",
                "noop/owning array", "noop/owning tensor"]:
        if key in results:
            rows.append((key.split("/", 1)[1], results[key]))
    print_table("Call Overhead (return first element)", rows)

    # ---- Sum 1D ----
    results["sum1d/numpy"] = bench("np.sum(a)", globals_dict=g)
    for slot, label_ in [
        ("sum_ndarray",       "ndarray"),
        ("sum_view",          "view"),
        ("sum_tensor_view",   "tensor_view"),
        ("sum_owning",        "owning array"),
        ("sum_owning_tensor", "owning tensor"),
    ]:
        fn = get(slot)
        if fn is None:
            continue
        g["fn"] = fn
        results[f"sum1d/{label_}"] = bench("fn(a)", globals_dict=g)

    native_fn = get("native_sum")
    if native_fn is not None:
        _, native_ms = native_fn(a, ITERATIONS)
        results["sum1d/native"] = native_ms / 1000.0 / ITERATIONS

    rows = [("numpy", results["sum1d/numpy"])]
    for key in ["sum1d/ndarray", "sum1d/view", "sum1d/tensor_view",
                "sum1d/owning array", "sum1d/owning tensor", "sum1d/native"]:
        if key in results:
            rows.append((key.split("/", 1)[1], results[key]))
    print_table(f"Sum Reduction (1D, {N_1D:,} float64)", rows)

    # ---- Sum 2D ----
    a2d = np.random.rand(N_2D_ROWS, N_2D_COLS)
    g["a2d"] = a2d
    results["sum2d/numpy"] = bench("np.sum(a2d)", globals_dict=g)
    for slot, label_ in [
        ("sum2d_ndarray",       "ndarray"),
        ("sum2d_view",          "view"),
        ("sum2d_tensor_view",   "tensor_view"),
        ("sum2d_owning",        "owning array"),
        ("sum2d_owning_tensor", "owning tensor"),
    ]:
        fn = get(slot)
        if fn is None:
            continue
        g["fn"] = fn
        results[f"sum2d/{label_}"] = bench("fn(a2d)", globals_dict=g)

    native2d_fn = get("native_sum2d")
    if native2d_fn is not None:
        _, native_ms = native2d_fn(a2d, ITERATIONS)
        results["sum2d/native"] = native_ms / 1000.0 / ITERATIONS

    rows = [("numpy", results["sum2d/numpy"])]
    for key in ["sum2d/ndarray", "sum2d/view", "sum2d/tensor_view",
                "sum2d/owning array", "sum2d/owning tensor", "sum2d/native"]:
        if key in results:
            rows.append((key.split("/", 1)[1], results[key]))
    print_table(f"Sum Reduction (2D, {N_2D_ROWS}x{N_2D_COLS} float64)", rows)

    # ---- Element-wise compute ----
    s, t = 2.5, 1.0
    g["s"] = s
    g["t"] = t
    results["compute/numpy"] = bench("np.sin(a) * s + t", globals_dict=g)
    for slot, label_ in [
        ("compute_view",          "view"),
        ("compute_tensor_view",   "tensor_view"),
        ("compute_owning",        "owning array"),
        ("compute_owning_tensor", "owning tensor"),
    ]:
        fn = get(slot)
        if fn is None:
            continue
        g["fn"] = fn
        results[f"compute/{label_}"] = bench("fn(a, s, t)", globals_dict=g)

    native_compute = get("native_compute")
    if native_compute is not None:
        _, native_ms = native_compute(a, s, t, ITERATIONS)
        results["compute/native"] = native_ms / 1000.0 / ITERATIONS

    rows = [("numpy", results["compute/numpy"])]
    for key in ["compute/view", "compute/tensor_view",
                "compute/owning array", "compute/owning tensor",
                "compute/native"]:
        if key in results:
            rows.append((key.split("/", 1)[1], results[key]))
    print_table(f"Element-wise sin(a)*s+t (1D, {N_1D:,} float64)", rows)

    # ---- 2D compute with explicit layouts ----
    a2d_c = a2d
    a2d_f = np.asfortranarray(a2d_c.copy())
    g["a2d_c"] = a2d_c
    g["a2d_f"] = a2d_f
    rows = []
    for slot, label_, inp in [
        ("compute2d_xarray_row_major",         "xarray row_major",     "c"),
        ("compute2d_xarray_column_major",      "xarray col_major",    "f"),
        ("compute2d_xarray_dynamic",           "xarray dynamic",       "c"),
        ("compute2d_xtensor_row_major",        "xtensor row_major",    "c"),
        ("compute2d_xtensor_column_major",     "xtensor col_major",   "f"),
        ("compute2d_xtensor_dynamic",          "xtensor dynamic",      "c"),
        ("compute2d_view_row_major",           "view row_major",       "c"),
        ("compute2d_view_column_major",        "view col_major",      "f"),
        ("compute2d_view_dynamic",             "view dynamic",         "c"),
        ("compute2d_tensor_view_row_major",    "tensor_view row_major","c"),
        ("compute2d_tensor_view_column_major", "tensor_view col_major","f"),
        ("compute2d_tensor_view_dynamic",      "tensor_view dynamic",  "c"),
    ]:
        fn = get(slot)
        if fn is None:
            continue
        arr_name = "a2d_c" if inp == "c" else "a2d_f"
        g["fn"] = fn
        g["arr"] = g[arr_name]
        t_ = bench("fn(arr, s, t)", globals_dict=g)
        results[f"compute2d/{label_}"] = t_
        rows.append((label_, t_))
    if rows:
        print_table(f"2D Compute sin(a)*s+t ({N_2D_ROWS}x{N_2D_COLS}, layout variants)", rows)

    # ---- Vectorization ----
    a_centered = a - 0.5
    g["a_abs"] = a_centered
    results["vectorize/np.abs"] = bench("np.abs(a_abs)", globals_dict=g)
    vf = np.vectorize(abs)
    g["vf"] = vf
    results["vectorize/np.vectorize"] = bench("vf(a_abs)", globals_dict=g)
    vec_fn = get("vectorize_abs")
    if vec_fn is not None:
        g["fn"] = vec_fn
        results["vectorize/xvectorize"] = bench("fn(a_abs)", globals_dict=g)

    rows = [("np.abs", results["vectorize/np.abs"]),
            ("np.vectorize", results["vectorize/np.vectorize"])]
    if "vectorize/xvectorize" in results:
        rows.append(("xvectorize", results["vectorize/xvectorize"]))
    print_table(f"Vectorization abs(x) (float64, {N_1D:,})", rows,
                baseline_key="np.abs")

    return results


# ---------------------------------------------------------------------------
# Cross-backend summary
# ---------------------------------------------------------------------------

SUMMARY_ROWS = [
    ("Call overhead",        ["noop/ndarray", "noop/view", "noop/owning array"]),
    ("Sum 1D",               ["sum1d/ndarray", "sum1d/view", "sum1d/owning array"]),
    ("Sum 2D",               ["sum2d/ndarray", "sum2d/view", "sum2d/owning array"]),
    ("Compute sin(a)*s+t",   ["compute/view", "compute/owning array"]),
    ("Compute2D xarray",     ["compute2d/xarray row_major", "compute2d/xarray dynamic"]),
    ("Compute2D xtensor",    ["compute2d/xtensor row_major", "compute2d/xtensor dynamic"]),
    ("Compute2D view",       ["compute2d/view row_major", "compute2d/view dynamic"]),
    ("Compute2D t.view",     ["compute2d/tensor_view row_major", "compute2d/tensor_view dynamic"]),
    ("Vectorize abs",        ["vectorize/xvectorize"]),
]


def print_summary(backend_results):
    """backend_results: dict[label -> {metric -> time}]."""
    if len(backend_results) < 2:
        return

    print()
    print("#" * 62)
    print("#  Cross-backend summary")
    print("#" * 62)

    columns = list(backend_results.keys())
    rows = []
    for group_name, keys in SUMMARY_ROWS:
        for key in keys:
            label = f"{group_name}  [{key.split('/', 1)[1]}]"
            vals = {col: backend_results[col].get(key) for col in columns}
            if any(v is not None for v in vals.values()):
                rows.append((label, vals))
    print_comparison_table("Nanobind vs xtensor-python", columns, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["both", "nanobind", "xtensor-python"],
                        default="both")
    args = parser.parse_args()

    backends = load_backends(args.backend)

    print("=" * 62)
    print("  xtensor bindings benchmark (nanobind vs xtensor-python)")
    print("=" * 62)
    print(f"  OS:         {_os_name()}")
    print(f"  CPU:        {_cpu_brand()}")
    print(f"  1D array:  {N_1D:>12,} float64 elements")
    print(f"  2D array:  {N_2D_ROWS}x{N_2D_COLS:>5} float64 elements")
    print(f"  Iterations: {ITERATIONS}")
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  NumPy:      {np.__version__}")
    print(f"  Backends:   {', '.join(b['label'] for b in backends)}")
    for b in backends:
        print(f"    {b['label']:<16} xsimd={'yes' if b['has_xsimd'] else 'no'}")
    print("=" * 62)

    backend_results = {}
    for b in backends:
        backend_results[b["label"]] = run_for_backend(b)

    print_summary(backend_results)
    print()


if __name__ == "__main__":
    main()
