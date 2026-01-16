#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#include "xtensor-python/pytensor.hpp"     // Numpy bindings

typedef xt::pytensor<double, 2, xt::layout_type::row_major> Array2;
using std::shared_ptr;
using std::vector;

namespace py = pybind11;
#include "regular_grid_interpolant_3d.h"

#ifdef USE_MPI
#include <mpi.h>
#include <Python.h>
#include <cstring>

// Include mpi4py C API if available
#if defined(MPI4PY_VERSION) || defined(HAVE_MPI4PY)
#include <mpi4py/mpi4py.h>
#define HAVE_MPI4PY_C_API
#endif

// Helper function to extract MPI_Comm from mpi4py communicator
// Made static to avoid duplicate symbol errors
static MPI_Comm get_mpi_comm_from_py(py::object comm_obj) {
    PyObject* py_obj = comm_obj.ptr();
    
    // Method 1: Try mpi4py C API if available
#ifdef HAVE_MPI4PY_C_API
    if (import_mpi4py() >= 0) {
        MPI_Comm* comm_ptr = PyMPIComm_Get(py_obj);
        if (comm_ptr != nullptr) {
            return *comm_ptr;
        }
        // Clear any Python errors from PyMPIComm_Get
        PyErr_Clear();
    }
#endif
    
    // Method 2: Try accessing 'handle' attribute (mpi4py 3.0+)
    PyObject* handle_attr = PyObject_GetAttrString(py_obj, "handle");
    if (handle_attr != nullptr) {
        if (PyLong_Check(handle_attr)) {
            long long comm_val = PyLong_AsLongLong(handle_attr);
            Py_DECREF(handle_attr);
            if (PyErr_Occurred() == nullptr) {
                return (MPI_Comm)comm_val;
            }
            PyErr_Clear();
        }
        Py_DECREF(handle_attr);
    }
    
    // Method 3: Try accessing ob_mpi attribute (older mpi4py versions)
    PyObject* ob_mpi_attr = PyObject_GetAttrString(py_obj, "ob_mpi");
    if (ob_mpi_attr != nullptr) {
        // Check if it's a PyCapsule (mpi4py stores MPI_Comm* in a capsule)
        if (PyCapsule_CheckExact(ob_mpi_attr)) {
            const char* name = PyCapsule_GetName(ob_mpi_attr);
            void* ptr = PyCapsule_GetPointer(ob_mpi_attr, name);
            if (ptr != nullptr) {
                // MPI_Comm is typically a pointer, so we need to dereference
                MPI_Comm result = *((MPI_Comm*)ptr);
                Py_DECREF(ob_mpi_attr);
                return result;
            }
        }
        
        // Check if it's stored as an integer (some MPI implementations)
        if (PyLong_Check(ob_mpi_attr)) {
            long long comm_val = PyLong_AsLongLong(ob_mpi_attr);
            if (PyErr_Occurred() == nullptr) {
                Py_DECREF(ob_mpi_attr);
                return (MPI_Comm)comm_val;
            }
            PyErr_Clear();
        }
        
        Py_DECREF(ob_mpi_attr);
    }
    
    // Method 4: Try ctypes interface (some mpi4py versions expose this)
    PyObject* pyobj_type = PyObject_Type(py_obj);
    if (pyobj_type) {
        PyObject* type_name = PyObject_GetAttrString(pyobj_type, "__name__");
        Py_DECREF(pyobj_type);
        if (type_name) {
            const char* name_str = PyUnicode_AsUTF8(type_name);
            if (name_str && strstr(name_str, "Comm") != nullptr) {
                // This looks like an MPI communicator, try to get underlying value
                PyObject* value_attr = PyObject_GetAttrString(py_obj, "value");
                if (value_attr) {
                    if (PyLong_Check(value_attr)) {
                        long long comm_val = PyLong_AsLongLong(value_attr);
                        Py_DECREF(value_attr);
                        Py_DECREF(type_name);
                        if (PyErr_Occurred() == nullptr) {
                            return (MPI_Comm)comm_val;
                        }
                        PyErr_Clear();
                    }
                    Py_DECREF(value_attr);
                }
            }
            Py_DECREF(type_name);
        }
    }
    
    // Method 5: Last resort - try direct integer conversion
    try {
        long long comm_val = py::cast<long long>(comm_obj);
        return (MPI_Comm)comm_val;
    } catch (...) {
        // Continue to error
    }
    
    // If all methods fail, provide helpful error message
    PyObject* pyobj_repr = PyObject_Repr(py_obj);
    std::string obj_str = "<unknown>";
    if (pyobj_repr) {
        const char* repr_str = PyUnicode_AsUTF8(pyobj_repr);
        if (repr_str) {
            obj_str = repr_str;
        }
        Py_DECREF(pyobj_repr);
    }
    
    throw std::runtime_error(
        "Could not extract MPI_Comm from Python object: " + obj_str + ". "
        "Expected an mpi4py communicator object (e.g., MPI.COMM_WORLD). "
        "Make sure mpi4py is properly installed and MPI is initialized."
    );
}

// Wrapper function for MPI version of interpolate_batch
void interpolate_batch_mpi(RegularGridInterpolant3D<Array2>& self, 
                           std::function<Vec(Vec, Vec, Vec)>& f, 
                           py::object comm_obj) {
    MPI_Comm comm = get_mpi_comm_from_py(comm_obj);
    self.interpolate_batch(f, comm);
}
#endif

void init_interpolant(py::module_ &m){

    py::class_<InterpolationRule, shared_ptr<InterpolationRule>>(m, "InterpolationRule", "Abstract class for interpolation rules on an interval.")
        .def_readonly("degree", &InterpolationRule::degree, "The degree of the polynomial. The number of interpolation points in `degree+1`.");

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
        .def("interpolate_batch", 
             py::overload_cast<std::function<Vec(Vec, Vec, Vec)>&>(&RegularGridInterpolant3D<Array2>::interpolate_batch),
             "Interpolate a function by evaluating the function on all interpolation nodes simultanuously.")
#ifdef USE_MPI
        .def("interpolate_batch",
             [](RegularGridInterpolant3D<Array2>& self, std::function<Vec(Vec, Vec, Vec)>& f, py::object comm_obj) {
                 interpolate_batch_mpi(self, f, comm_obj);
             },
             py::arg("f"), py::arg("comm"),
             "Interpolate a function with MPI parallelization. 'comm' should be an mpi4py communicator (e.g., MPI.COMM_WORLD).")
#endif
        .def("evaluate", &RegularGridInterpolant3D<Array2>::evaluate, "Evaluate the interpolant at a point.")
        .def("evaluate_batch", &RegularGridInterpolant3D<Array2>::evaluate_batch, "Evaluate the interpolant at multiple points (faster than `evaluate` as it uses prefetching).");
}
