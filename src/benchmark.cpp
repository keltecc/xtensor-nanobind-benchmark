#include <cmath>
#include <chrono>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/xtensor.h>
#include <xtensor/core/xmath.hpp>

namespace nb = nanobind;
NB_MODULE(benchmark_ext, m) {

#ifdef XTENSOR_USE_XSIMD
    m.attr("has_xsimd") = true;
#else
    m.attr("has_xsimd") = false;
#endif

    // =========================================================================
    // Call overhead: return first element (isolates caster/conversion cost)
    // =========================================================================

    m.def("noop_xarray", [](const xt::xarray<double>& a) {
        return a(0);
    });

    m.def("noop_xtensor", [](const xt::xtensor<double, 1>& a) {
        return a(0);
    });

    m.def("noop_xarray_view", [](const nb::xarray_view<double>& a) {
        return a(0);
    });

    m.def("noop_xtensor_view", [](const nb::xtensor_view<double, 1>& a) {
        return a(0);
    });

    m.def("noop_ndarray", [](nb::ndarray<double, nb::numpy, nb::ndim<1>> a) {
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

    m.def("sum_xarray_view", [](const nb::xarray_view<double>& a) {
        return xt::sum(a)();
    });

    m.def("sum_xtensor_view", [](const nb::xtensor_view<double, 1>& a) {
        return xt::sum(a)();
    });

    m.def("sum_ndarray", [](nb::ndarray<double, nb::numpy, nb::ndim<1>> a) {
        double s = 0.0;
        const double* ptr = a.data();
        size_t n = a.shape(0);
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

    m.def("sum2d_xarray_view", [](const nb::xarray_view<double>& a) {
        return xt::sum(a)();
    });

    m.def("sum2d_xtensor_view", [](const nb::xtensor_view<double, 2>& a) {
        return xt::sum(a)();
    });

    m.def("sum2d_ndarray", [](nb::ndarray<double, nb::numpy, nb::ndim<2>> a) {
        double s = 0.0;
        const double* ptr = a.data();
        size_t n = a.shape(0) * a.shape(1);
        for (size_t i = 0; i < n; ++i)
            s += ptr[i];
        return s;
    });

    // =========================================================================
    // Layout comparison: row_major (fast) vs dynamic (stepper-based, slow)
    // =========================================================================

    m.def("sum_xarray_view_dynamic",
        [](const nb::xarray_view<double, xt::layout_type::dynamic>& a) {
        return xt::sum(a)();
    });

    m.def("sum2d_xarray_view_dynamic",
        [](const nb::xarray_view<double, xt::layout_type::dynamic>& a) {
        return xt::sum(a)();
    });

    m.def("sum2d_xtensor_view_dynamic",
        [](const nb::xtensor_view<double, 2, xt::layout_type::dynamic>& a) {
        return xt::sum(a)();
    });

    // =========================================================================
    // Element-wise computation: sin(a) * s + t
    // Parameters must be const& because xtensor expressions capture references.
    // =========================================================================

    m.def("compute_xarray",
        [](const xt::xarray<double>& a, const double& s, const double& t) {
        return xt::sin(a) * s + t;
    });

    m.def("compute_xtensor",
        [](const xt::xtensor<double, 1>& a, const double& s, const double& t) {
        return xt::sin(a) * s + t;
    });

    m.def("compute_xarray_view",
        [](const nb::xarray_view<double>& a, const double& s, const double& t) {
        return xt::sin(a) * s + t;
    });

    m.def("compute_xtensor_view",
        [](const nb::xtensor_view<double, 1>& a, const double& s, const double& t) {
        return xt::sin(a) * s + t;
    });

    // =========================================================================
    // Vectorization: abs on float64
    // =========================================================================

    m.def("vectorize_abs", nb::xvectorize([](double x) -> double {
        return std::abs(x);
    }));

    // =========================================================================
    // Native C++ baselines: run N iterations internally with chrono timing.
    // Returns (result, elapsed_ms) to bypass Python per-call overhead.
    // =========================================================================

    m.def("native_sum", [](nb::ndarray<double, nb::numpy, nb::ndim<1>> a, int iters) {
        const double* ptr = a.data();
        size_t n = a.shape(0);
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
        return nb::make_tuple(result, ms);
    });

    m.def("native_compute", [](nb::ndarray<double, nb::numpy, nb::ndim<1>> a,
                                double s, double t, int iters) {
        const double* ptr = a.data();
        size_t n = a.shape(0);
        std::vector<double> out(n);
        auto start = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; ++it) {
            for (size_t i = 0; i < n; ++i)
                out[i] = std::sin(ptr[i]) * s + t;
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        return nb::make_tuple(out[0], ms);
    });

    m.def("native_sum2d", [](nb::ndarray<double, nb::numpy, nb::ndim<2>> a, int iters) {
        const double* ptr = a.data();
        size_t n = a.shape(0) * a.shape(1);
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
        return nb::make_tuple(result, ms);
    });
}
