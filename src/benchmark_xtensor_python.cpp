#include <cmath>
#include <chrono>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>
#include <xtensor-python/pyvectorize.hpp>
#include <xtensor-python/pynative_casters.hpp>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xmath.hpp>

namespace py = pybind11;

PYBIND11_MODULE(benchmark_xtp_ext, m) {
    xt::import_numpy();

#ifdef XTENSOR_USE_XSIMD
    m.attr("has_xsimd") = true;
#else
    m.attr("has_xsimd") = false;
#endif

    // Minimum C++ numpy binding — closest equivalent to nb::ndarray.
    using array_t_c = py::array_t<double, py::array::c_style | py::array::forcecast>;

    // =========================================================================
    // Call overhead: return first element (isolates caster/conversion cost)
    // =========================================================================

    m.def("noop_xarray", [](const xt::xarray<double>& a) {
        return a(0);
    });

    m.def("noop_xtensor", [](const xt::xtensor<double, 1>& a) {
        return a(0);
    });

    m.def("noop_pyarray", [](const xt::pyarray<double>& a) {
        return a(0);
    });

    m.def("noop_pytensor", [](const xt::pytensor<double, 1>& a) {
        return a(0);
    });

    m.def("noop_array_t", [](array_t_c a) {
        return a.data()[0];
    });

    // =========================================================================
    // 1D sum reduction (iteration speed)
    // =========================================================================

    m.def("sum_xarray", [](const xt::xarray<double>& a) {
        return xt::sum(a)();
    });

    m.def("sum_xtensor", [](const xt::xtensor<double, 1>& a) {
        return xt::sum(a)();
    });

    m.def("sum_pyarray", [](const xt::pyarray<double>& a) {
        return xt::sum(a)();
    });

    m.def("sum_pytensor", [](const xt::pytensor<double, 1>& a) {
        return xt::sum(a)();
    });

    m.def("sum_array_t", [](array_t_c a) {
        double s = 0.0;
        const double* ptr = a.data();
        size_t n = static_cast<size_t>(a.shape(0));
        for (size_t i = 0; i < n; ++i)
            s += ptr[i];
        return s;
    });

    // =========================================================================
    // 2D sum reduction
    // =========================================================================

    m.def("sum2d_xarray", [](const xt::xarray<double>& a) {
        return xt::sum(a)();
    });

    m.def("sum2d_xtensor", [](const xt::xtensor<double, 2>& a) {
        return xt::sum(a)();
    });

    m.def("sum2d_pyarray", [](const xt::pyarray<double>& a) {
        return xt::sum(a)();
    });

    m.def("sum2d_pytensor", [](const xt::pytensor<double, 2>& a) {
        return xt::sum(a)();
    });

    m.def("sum2d_array_t", [](py::array_t<double, py::array::c_style | py::array::forcecast> a) {
        double s = 0.0;
        const double* ptr = a.data();
        size_t n = static_cast<size_t>(a.shape(0)) * static_cast<size_t>(a.shape(1));
        for (size_t i = 0; i < n; ++i)
            s += ptr[i];
        return s;
    });

    // =========================================================================
    // Element-wise computation: sin(a) * s + t
    // =========================================================================

    m.def("compute_xarray",
        [](const xt::xarray<double>& a, const double& s, const double& t) {
        return xt::xarray<double>(xt::sin(a) * s + t);
    });

    m.def("compute_xtensor",
        [](const xt::xtensor<double, 1>& a, const double& s, const double& t) {
        return xt::xtensor<double, 1>(xt::sin(a) * s + t);
    });

    m.def("compute_pyarray",
        [](const xt::pyarray<double>& a, const double& s, const double& t) {
        return xt::pyarray<double>(xt::sin(a) * s + t);
    });

    m.def("compute_pytensor",
        [](const xt::pytensor<double, 1>& a, const double& s, const double& t) {
        return xt::pytensor<double, 1>(xt::sin(a) * s + t);
    });

    // =========================================================================
    // 2D compute with explicit layouts: sin(a) * s + t (1000x1000 float64).
    // Column-major variants expect Fortran-contiguous input.
    // =========================================================================

    // --- xarray (owning) ---
    m.def("compute2d_xarray_row_major",
        [](const xt::xarray<double, xt::layout_type::row_major>& a,
            const double& s, const double& t) {
        return xt::xarray<double>(xt::sin(a) * s + t);
    });
    m.def("compute2d_xarray_column_major",
        [](const xt::xarray<double, xt::layout_type::column_major>& a,
            const double& s, const double& t) {
        return xt::xarray<double>(xt::sin(a) * s + t);
    });
    m.def("compute2d_xarray_dynamic",
        [](const xt::xarray<double>& a, const double& s, const double& t) {
        return xt::xarray<double>(xt::sin(a) * s + t);
    });

    // --- xtensor (owning, 2D) ---
    m.def("compute2d_xtensor_row_major",
        [](const xt::xtensor<double, 2, xt::layout_type::row_major>& a,
            const double& s, const double& t) {
        return xt::xtensor<double, 2>(xt::sin(a) * s + t);
    });
    m.def("compute2d_xtensor_column_major",
        [](const xt::xtensor<double, 2, xt::layout_type::column_major>& a,
            const double& s, const double& t) {
        return xt::xtensor<double, 2>(xt::sin(a) * s + t);
    });
    m.def("compute2d_xtensor_dynamic",
        [](const xt::xtensor<double, 2>& a, const double& s, const double& t) {
        return xt::xtensor<double, 2>(xt::sin(a) * s + t);
    });

    // --- pyarray (zero-copy) ---
    m.def("compute2d_pyarray_row_major",
        [](const xt::pyarray<double, xt::layout_type::row_major>& a,
            const double& s, const double& t) {
        return xt::pyarray<double>(xt::sin(a) * s + t);
    });
    m.def("compute2d_pyarray_column_major",
        [](const xt::pyarray<double, xt::layout_type::column_major>& a,
            const double& s, const double& t) {
        return xt::pyarray<double>(xt::sin(a) * s + t);
    });
    m.def("compute2d_pyarray_dynamic",
        [](const xt::pyarray<double>& a, const double& s, const double& t) {
        return xt::pyarray<double>(xt::sin(a) * s + t);
    });

    // --- pytensor (zero-copy, 2D) ---
    m.def("compute2d_pytensor_row_major",
        [](const xt::pytensor<double, 2, xt::layout_type::row_major>& a,
            const double& s, const double& t) {
        return xt::pytensor<double, 2>(xt::sin(a) * s + t);
    });
    m.def("compute2d_pytensor_column_major",
        [](const xt::pytensor<double, 2, xt::layout_type::column_major>& a,
            const double& s, const double& t) {
        return xt::pytensor<double, 2>(xt::sin(a) * s + t);
    });
    m.def("compute2d_pytensor_dynamic",
        [](const xt::pytensor<double, 2>& a, const double& s, const double& t) {
        return xt::pytensor<double, 2>(xt::sin(a) * s + t);
    });

    // =========================================================================
    // Vectorization: abs on float64
    // =========================================================================

    m.def("vectorize_abs", xt::pyvectorize([](double x) -> double {
        return std::abs(x);
    }));

    // =========================================================================
    // Native C++ baselines (identical to nanobind version).
    // =========================================================================

    m.def("native_sum", [](array_t_c a, int iters) {
        const double* ptr = a.data();
        size_t n = static_cast<size_t>(a.shape(0));
        auto start = std::chrono::high_resolution_clock::now();
        double result = 0.0;
        for (int it = 0; it < iters; ++it) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i)
                s += ptr[i];
            result = s;
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        return py::make_tuple(result, ms);
    });

    m.def("native_compute", [](array_t_c a, double s, double t, int iters) {
        const double* ptr = a.data();
        size_t n = static_cast<size_t>(a.shape(0));
        std::vector<double> out(n);
        auto start = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; ++it) {
            for (size_t i = 0; i < n; ++i)
                out[i] = std::sin(ptr[i]) * s + t;
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        return py::make_tuple(out[0], ms);
    });

    m.def("native_sum2d", [](py::array_t<double, py::array::c_style | py::array::forcecast> a,
                              int iters) {
        const double* ptr = a.data();
        size_t n = static_cast<size_t>(a.shape(0)) * static_cast<size_t>(a.shape(1));
        auto start = std::chrono::high_resolution_clock::now();
        double result = 0.0;
        for (int it = 0; it < iters; ++it) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i)
                s += ptr[i];
            result = s;
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        return py::make_tuple(result, ms);
    });
}
