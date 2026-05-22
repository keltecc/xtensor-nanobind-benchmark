#!/usr/bin/env python3
"""
Generate comparison plots for 2D sin(a)*s+t benchmarks.
Compares nanobind vs xtensor-python across varying N×N array sizes.
"""

import sys
import timeit
import numpy as np
import matplotlib.pyplot as plt

N_VALS = [1, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200,
          250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]

S, T = 2.5, 1.0

COLORS = {"nanobind": "#2196F3", "xtensor-python": "#FF5722"}

COMPARISONS = [
    {
        "num": 1,
        "nb_func": "compute2d_xarray_dynamic",
        "xtp_func": "compute2d_xarray_dynamic",
        "title": "xt::xarray (owning, dynamic layout)",
    },
    {
        "num": 2,
        "nb_func": "compute2d_xtensor_dynamic",
        "xtp_func": "compute2d_xtensor_dynamic",
        "title": "xt::xtensor (owning, fixed 2D)",
    },
    {
        "num": 3,
        "nb_func": "compute2d_xarray_view_dynamic",
        "xtp_func": "compute2d_pyarray_dynamic",
        "title": "nb::xarray_view  vs  xt::pyarray",
    },
    {
        "num": 4,
        "nb_func": "compute2d_xtensor_view_dynamic",
        "xtp_func": "compute2d_pytensor_dynamic",
        "title": "nb::xtensor_view  vs  xt::pytensor",
    },
]


REPEAT = 10


def iters_for_n(n):
    if n <= 5:
        return 2000
    if n <= 50:
        return 500
    if n <= 200:
        return 100
    return 20


def bench(fn, arr, iters):
    g = {"fn": fn, "arr": arr, "s_val": S, "t_val": T}
    times = timeit.repeat("fn(arr, s_val, t_val)", number=iters,
                          repeat=REPEAT, globals=g)
    return min(times) / iters


def load_modules():
    import benchmark_ext as nb_mod

    try:
        import benchmark_xtp_ext as xtp_mod
    except ImportError:
        xtp_mod = None
    return nb_mod, xtp_mod


def main():
    nb_mod, xtp_mod = load_modules()
    have_xtp = xtp_mod is not None
    if not have_xtp:
        print("Warning: benchmark_xtp_ext not found — only nanobind will be plotted.\n")

    for cmp in COMPARISONS:
        nb_times = []
        xtp_times = []

        for n in N_VALS:
            arr = np.random.rand(n, n)
            iters = iters_for_n(n)

            fn_nb = getattr(nb_mod, cmp["nb_func"])
            nb_times.append(bench(fn_nb, arr, iters) * 1e6)

            if have_xtp:
                fn_xtp = getattr(xtp_mod, cmp["xtp_func"])
                xtp_times.append(bench(fn_xtp, arr, iters) * 1e6)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(N_VALS, nb_times, marker="o", color=COLORS["nanobind"],
                label="nanobind", linewidth=2, markersize=4)
        if have_xtp:
            ax.plot(N_VALS, xtp_times, marker="s", color=COLORS["xtensor-python"],
                    label="xtensor-python", linewidth=2, markersize=4)

        ax.set_xlabel("N  (N×N array)")
        ax.set_ylabel("Time (us)")
        ax.set_title(f"2D  sin(a)·s + t  —  {cmp['title']}")
        ax.legend()
        ax.grid(True, alpha=0.35)

        fname = f"{cmp['num']}.jpg"
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved {fname}")


if __name__ == "__main__":
    main()
