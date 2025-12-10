#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#include "xtensor-python/pytensor.hpp"     // Numpy bindings

typedef xt::pytensor<double, 2, xt::layout_type::row_major> Array2;
using std::shared_ptr;
using std::vector;

namespace py = pybind11;
#include "regular_grid_interpolant_3d.h"

void init_interpolant(py::module_ &m){

    py::class_<InterpolationRule, shared_ptr<InterpolationRule>>(m, "InterpolationRule", "Abstract class for interpolation rules on an interval.")
        .def_readonly("degree", &InterpolationRule::degree, "The degree of the polynomial. The number of interpolation points in `degree+1`.")
        .def_readonly("nodes", &InterpolationRule::nodes, "The interpolation nodes (points) within each cell.")
        .def_readonly("scalings", &InterpolationRule::scalings, "The scaling factors for interpolation weights.");

    py::class_<UniformInterpolationRule, shared_ptr<UniformInterpolationRule>, InterpolationRule>(m, "UniformInterpolationRule", "Polynomial interpolation using equispaced points.")
        .def(py::init<int>())
        .def_readonly("degree", &UniformInterpolationRule::degree, "The degree of the polynomial. The number of interpolation points in `degree+1`.");
    py::class_<ChebyshevInterpolationRule, shared_ptr<ChebyshevInterpolationRule>, InterpolationRule>(m, "ChebyshevInterpolationRule", "Polynomial interpolation using chebychev points.")
        .def(py::init<int>())
        .def_readonly("degree", &ChebyshevInterpolationRule::degree, "The degree of the polynomial. The number of interpolation points in `degree+1`.");

    py::class_<RegularGridInterpolant3D<Array2>, shared_ptr<RegularGridInterpolant3D<Array2>>>(m, "RegularGridInterpolant3D",
            R"pbdoc(
            Interpolates a (vector valued) function on a uniform grid.
            This interpolant is optimized for fast function evaluation (at the cost of memory usage). The main purpose of this class is to be used to interpolate magnetic fields and then use the interpolant for tasks such as fieldline or particle tracing for which the field needs to be evaluated many many times.
            )pbdoc")
        .def(py::init<InterpolationRule, RangeTriplet, RangeTriplet, RangeTriplet, int, bool, std::function<std::vector<bool>(Vec, Vec, Vec)>>())
        .def(py::init<InterpolationRule, RangeTriplet, RangeTriplet, RangeTriplet, int, bool>())
        .def("interpolate_batch", &RegularGridInterpolant3D<Array2>::interpolate_batch, "Interpolate a function by evaluating the function on all interpolation nodes simultanuously.")
        .def("evaluate", &RegularGridInterpolant3D<Array2>::evaluate, "Evaluate the interpolant at a point.")
        .def("evaluate_batch", &RegularGridInterpolant3D<Array2>::evaluate_batch, "Evaluate the interpolant at multiple points (faster than `evaluate` as it uses prefetching).")
        // ========================================================================
        // SAVE/LOAD BINDINGS: Enable interpolant serialization to avoid recomputation
        // These methods are used internally by InterpolatedBoozerField.to_json()
        // and the JSON loading constructor
        // ========================================================================
        .def("get_interpolant_data", &RegularGridInterpolant3D<Array2>::get_interpolant_data,
             "Get interpolant data (vals array and grid parameters) for saving.")
        .def("set_interpolant_data", &RegularGridInterpolant3D<Array2>::set_interpolant_data,
             "Set interpolant data from saved data, reconstructing all_local_vals_map.")
        .def("is_computed", &RegularGridInterpolant3D<Array2>::is_computed,
             "Check if interpolant has computed data (vals array is not empty).")
        // Load mode control: prevents interpolate_batch() from running during loading
        .def_static("set_load_mode", &RegularGridInterpolant3D<Array2>::set_load_mode,
                    "Set load mode (true prevents expensive computation during loading).")
        .def_static("get_load_mode", &RegularGridInterpolant3D<Array2>::get_load_mode,
                    "Get current load mode status.");
}
