#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#include "boozermagneticfield.h"
#include "boozermagneticfield_interpolated.h"
#include "pyboozermagneticfield.h"
#include "regular_grid_interpolant_3d.h"
#include "shearalfvenwave.h"
#include "pyshearalfvenwave.h"
#include <string>
#ifdef USE_MPI
#include <mpi.h>
#include <Python.h>
#include <cstring>

#if defined(MPI4PY_VERSION) || defined(HAVE_MPI4PY)
#include <mpi4py/mpi4py.h>
#define HAVE_MPI4PY_C_API
#endif
#endif

using std::string;
using std::shared_ptr;
using std::vector;

namespace py = pybind11;

#ifdef USE_MPI
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
#endif

void init_boozermagneticfields(py::module_ &m){
  auto mf = py::class_<
      BoozerMagneticField,
      BoozerMagneticFieldTrampoline<BoozerMagneticField>,
      shared_ptr<BoozerMagneticField>
      >(m, "BoozerMagneticField", "")
    .def(py::init<double,string>());
    mf
    .def(
        "dKdtheta",
        &BoozerMagneticField::dKdtheta,
        "Returns a `(npoints, 1)` array containing the theta derivative of the"
        " Boozer radial covariant component, K, where B = G nabla zeta + I " "nabla theta + K nabla psi."
    )
    .def(
        "dKdzeta",
        &BoozerMagneticField::dKdzeta,
        "Returns a `(npoints, 1)` array containing the zeta derivative of the"
        " Boozer radial covariant component, K, where B = G nabla zeta + I "
        " nabla theta + K nabla psi."
    )
    .def(
        "K_derivs",
        &BoozerMagneticField::K_derivs,
        "Returns a `(npoints, 2)` array containing the dKdtheta and dKdzeta, " "where K is the Boozer radial covariant component, and B = G nabla " "zeta + I nabla theta + K nabla psi."
    )
    .def(
        "K",
        &BoozerMagneticField::K,
        "Returns a `(npoints, 1)` array containing the Boozer radial covariant"
        "component, K, where B = G nabla zeta + I nabla theta + K nabla psi."
    )
    .def(
        "nu",
        &BoozerMagneticField::nu,
        "Returns a `(npoints, 1)` array containing the difference between the" " Boozer and cylindrical toroidal angles, e.g. zeta_b = phi + nu."
    )
    .def(
        "dnudtheta",
        &BoozerMagneticField::dnudtheta,
        "Returns a `(npoints, 1)` array containing the derivative of nu wrt"
        " theta as a function of Boozer coordinates."
    )
    .def(
        "dnudzeta",
        &BoozerMagneticField::dnudzeta,
        "Returns a `(npoints, 1)` array containing the derivative of nu wrt " "zeta as a function of Boozer coordinates."
    )
    .def(
        "dnuds",
        &BoozerMagneticField::dnuds,
        "Returns a `(npoints, 1)` array containing the derivative of nu wrt s"
        " as a function of Boozer coordinates."
    )
    .def(
        "nu_derivs",
        &BoozerMagneticField::nu_derivs,
        "Returns a `(npoints, 3)` array containing (dnuds,dnudtheta,dnudzeta)."
    )
    .def(
        "R",
        &BoozerMagneticField::R,
        "Returns a `(npoints, 1)` array containing the major radius as a " "function of Boozer coordinates."
    )
    .def(
        "dRdtheta",
        &BoozerMagneticField::dRdtheta,
        "Returns a `(npoints, 1)` array containing the derivative of the major"
        " radius wrt theta as a function of Boozer coordinates."
    )
    .def(
        "dRdzeta",
        &BoozerMagneticField::dRdzeta,
        "Returns a `(npoints, 1)` array containing the derivative of the major"
        " radius wrt zeta as a function of Boozer coordinates."
    )
    .def(
        "dRds",
        &BoozerMagneticField::dRds,
        "Returns a `(npoints, 1)` array containing the derivative of the major"
        " radius wrt s as a function of Boozer coordinates."
    )
    .def(
        "R_derivs",
        &BoozerMagneticField::R_derivs,
        "Returns a `(npoints, 3)` array containing (dRds,dRdtheta,dRdzeta)."
    )
    .def(
        "Z",
        &BoozerMagneticField::Z,
        "Returns a `(npoints, 1)` array containing the height as a function of"
        " Boozer coordinates."
    )
    .def(
        "dZdtheta",
        &BoozerMagneticField::dZdtheta,
        "Returns a `(npoints, 1)` array containing the derivative of the "
        "height wrt theta as a function of Boozer coordinates."
    )
    .def(
        "dZdzeta",
        &BoozerMagneticField::dZdzeta,
        "Returns a `(npoints, 1)` array containing the derivative of the " "height wrt zeta as a function of Boozer coordinates."
    )
    .def(
        "dZds",
        &BoozerMagneticField::dZds,
        "Returns a `(npoints, 1)` array containing the derivative of the " "height wrt s as a function of Boozer coordinates."
    )
    .def(
        "Z_derivs",
        &BoozerMagneticField::Z_derivs,
        "Returns a `(npoints, 3)` array containing (dZds,dZdtheta,dZdzeta)."
    )
    .def(
        "modB",
        &BoozerMagneticField::modB,
        "Returns a `(npoints, 1)` array containing the magnetic field strength"
        " in Boozer coordinates."
    )
    .def(
        "dmodBdtheta",
        &BoozerMagneticField::dmodBdtheta,
        "Returns a `(npoints, 1)` array containing the derivative of the " "magnetic field strength wrt theta in Boozer coordinates."
    )
    .def(
        "dmodBdzeta",
        &BoozerMagneticField::dmodBdzeta,
        "Returns a `(npoints, 1)` array containing the derivative of the " "magnetic field strength wrt zeta in Boozer coordinates."
    )
    .def(
        "dmodBds",
        &BoozerMagneticField::dmodBds,
        "Returns a `(npoints, 1)` array containing the derivative of the "
        "magnetic field strength wrt s in Boozer coordinates."
    )
    .def(
        "modB_derivs",
        &BoozerMagneticField::modB_derivs,
        "Returns a `(npoints, 3)` array containing (dmodBds,dmodBdtheta,dmodBdzeta)."
    )
    .def(
        "G",
        &BoozerMagneticField::G,
        "Returns a `(npoints, 1)` array containing the magnetic field toroidal"
        " covariant component in Boozer coordinates."
    )
    .def(
        "I",
        &BoozerMagneticField::I,
        "Returns a `(npoints, 1)` array containing the magnetic field "
        "poloidal covariant component in Boozer coordinates."
    )
    .def(
        "psip",
        &BoozerMagneticField::psip,
        "Returns a `(npoints, 1)` array containing the (poloidal flux)/(2*pi) "
        "in Boozer coordinates."
    )
    .def(
        "iota",
        &BoozerMagneticField::iota,
        "Returns a `(npoints, 1)` array containing the rotational transform "
        "in Boozer coordinates."
    )
    .def(
        "dGds",
        &BoozerMagneticField::dGds,
        "Returns a `(npoints, 1)` array containing the derivative of the " "magnetic field toroidal covariant component wrt s in Boozer " "coordinates."
    )
    .def(
        "dIds",
        &BoozerMagneticField::dIds,
        "Returns a `(npoints, 1)` array containing the derivative of the " "magnetic field poloidal covariant component wrt s in Boozer " "coordinates."
    )
    .def(
        "diotads",
        &BoozerMagneticField::diotads,
        "Returns a `(npoints, 1)` array containing the derivative of the " "rotational transform wrt s in Boozer coordinates."
    )
    .def(
        "dKdtheta_ref",
        &BoozerMagneticField::dKdtheta_ref,
        "Same as `dKdtheta`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "dKdzeta_ref",
        &BoozerMagneticField::dKdzeta_ref,
        "Same as `dKdzeta`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "K_derivs_ref",
        &BoozerMagneticField::K_derivs_ref,
        "Same as `K_derivs`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "K_ref",
        &BoozerMagneticField::K_ref,
        "Same as `K`, but returns a reference to the array :"
        "(this array should be read only)."
    )
    .def(
        "nu_ref",
        &BoozerMagneticField::nu_ref,
        "Same as `nu`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "dnudtheta_ref",
        &BoozerMagneticField::dnudtheta_ref,
        "Same as `dnudtheta`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "dnudzeta_ref",
        &BoozerMagneticField::dnudzeta_ref,
        "Same as `dnudzeta`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "dnuds_ref",
        &BoozerMagneticField::dnuds_ref,
        "Same as `dnuds`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "nu_derivs_ref",
        &BoozerMagneticField::nu_derivs_ref,
        "Same as `nu_derivs`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "R_ref",
        &BoozerMagneticField::R_ref,
        "Same as `R`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "Z_ref",
        &BoozerMagneticField::Z_ref,
        "Same as `Z`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "dRdtheta_ref",
        &BoozerMagneticField::dRdtheta_ref,
        "Same as `dRdtheta`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "dRdzeta_ref",
        &BoozerMagneticField::dRdzeta_ref,
        "Same as `dRdzeta`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "dRds_ref",
        &BoozerMagneticField::dRds_ref,
        "Same as `dRds`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "R_derivs_ref",
        &BoozerMagneticField::R_derivs_ref,
        "Same as `R_derivs`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "dZdtheta_ref",
        &BoozerMagneticField::dZdtheta_ref,
        "Same as `dZdtheta`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "dZdzeta_ref",
        &BoozerMagneticField::dZdzeta_ref,
        "Same as `dZdzeta`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "dZds_ref",
        &BoozerMagneticField::dZds_ref,
        "Same as `dZds`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "Z_derivs_ref",
        &BoozerMagneticField::Z_derivs_ref,
        "Same as `Z_derivs`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "modB_ref",
        &BoozerMagneticField::modB_ref,
        "Same as `modB`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "dmodBdtheta_ref",
        &BoozerMagneticField::dmodBdtheta_ref,
        "Same as `dmodBdtheta`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "dmodBdzeta_ref",
        &BoozerMagneticField::dmodBdzeta_ref,
        "Same as `dmodBdzeta`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "dmodBds_ref",
        &BoozerMagneticField::dmodBds_ref,
        "Same as `dmodBds`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "modB_derivs_ref",
        &BoozerMagneticField::modB_derivs_ref,
        "Same as `modB_derivs`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "G_ref",
        &BoozerMagneticField::G_ref,
        "Same as `G`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "I_ref",
        &BoozerMagneticField::I_ref,
        "Same as `I`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "psip_ref",
        &BoozerMagneticField::psip_ref,
        "Same as `psip`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "iota_ref",
        &BoozerMagneticField::iota_ref,
        "Same as `iota`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "dGds_ref",
        &BoozerMagneticField::dGds_ref,
        "Same as `dGds`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "dIds_ref",
        &BoozerMagneticField::dIds_ref,
        "Same as `dIds`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "diotads_ref",
        &BoozerMagneticField::diotads_ref,
        "Same as `diotads`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "get_points",
        &BoozerMagneticField::get_points,
        "Get the point where the field should be evaluated in "
        "Boozer  coordinates."
    )
    .def(
        "get_points_ref",
        &BoozerMagneticField::get_points_ref,
        "As `get_points`, but returns a reference to the array "
        "(this array should be read only)."
    )
    .def(
        "set_points",
        &BoozerMagneticField::set_points,
        "Set the points where the field should be evaluated in "
        "Boozer coordinates `(s,theta,zeta)`."
    );

  auto ifield = py::class_<
      InterpolatedBoozerField,
      BoozerMagneticField,
      shared_ptr<InterpolatedBoozerField>
      >(m, "InterpolatedBoozerField")
      .def(
          py::init<shared_ptr<BoozerMagneticField>,
          InterpolationRule,
          RangeTriplet,
          RangeTriplet,
          RangeTriplet,
          bool,
          int,
          bool,
          string>()
      )
      .def(
          py::init<shared_ptr<BoozerMagneticField>,
          int,
          RangeTriplet,
          RangeTriplet,
          RangeTriplet,
          bool,
          int,
          bool,
          string>()
      )
#ifdef USE_MPI
      .def(
          "set_mpi_comm",
          [](InterpolatedBoozerField& self, py::object comm_obj) {
              MPI_Comm comm = get_mpi_comm_from_py(comm_obj);
              self.set_mpi_comm(comm);
          },
          py::arg("comm"),
          "Set the MPI communicator to use for interpolate_batch."
      )
#endif
      .def(
          "estimate_error_K",
          &InterpolatedBoozerField::estimate_error_K
      )
      .def(
          "estimate_error_modB",
          &InterpolatedBoozerField::estimate_error_modB
      )
      .def(
          "estimate_error_R",
          &InterpolatedBoozerField::estimate_error_R
      )
      .def(
          "estimate_error_Z",
          &InterpolatedBoozerField::estimate_error_Z
      )
      .def(
          "estimate_error_nu",
          &InterpolatedBoozerField::estimate_error_nu
      )
      .def(
          "estimate_error_G",
          &InterpolatedBoozerField::estimate_error_G
      )
      .def(
          "estimate_error_I",
          &InterpolatedBoozerField::estimate_error_I
      )
      .def(
          "estimate_error_iota",
          &InterpolatedBoozerField::estimate_error_iota
      )
      .def_readonly(
          "s_range",
          &InterpolatedBoozerField::s_range
      )
      .def_readonly(
          "theta_range",
          &InterpolatedBoozerField::theta_range
      )
      .def_readonly(
          "zeta_range",
          &InterpolatedBoozerField::zeta_range
      )
      .def_readonly(
          "rule",
          &InterpolatedBoozerField::rule
      )
      .def_readwrite("status_modB",&InterpolatedBoozerField::status_modB)
      .def_readwrite("status_dmodBdtheta",&InterpolatedBoozerField::status_dmodBdtheta)
      .def_readwrite("status_dmodBdzeta",&InterpolatedBoozerField::status_dmodBdzeta)
      .def_readwrite("status_dmodBds",&InterpolatedBoozerField::status_dmodBds)
      .def_readwrite("status_G",&InterpolatedBoozerField::status_G)
      .def_readwrite("status_I",&InterpolatedBoozerField::status_I)
      .def_readwrite("status_iota",&InterpolatedBoozerField::status_iota)
      .def_readwrite("status_dGds",&InterpolatedBoozerField::status_dGds)
      .def_readwrite("status_dIds",&InterpolatedBoozerField::status_dIds)
      .def_readwrite("status_diotads",&InterpolatedBoozerField::status_diotads)
      .def_readwrite("status_psip",&InterpolatedBoozerField::status_psip)
      .def_readwrite("status_R",&InterpolatedBoozerField::status_R)
      .def_readwrite("status_Z",&InterpolatedBoozerField::status_Z)
      .def_readwrite("status_nu",&InterpolatedBoozerField::status_nu)
      .def_readwrite("status_K",&InterpolatedBoozerField::status_K)
      .def_readwrite("status_dRdtheta",&InterpolatedBoozerField::status_dRdtheta)
      .def_readwrite("status_dRdzeta",&InterpolatedBoozerField::status_dRdzeta)
      .def_readwrite("status_dRds",&InterpolatedBoozerField::status_dRds)
      .def_readwrite("status_dZdtheta",&InterpolatedBoozerField::status_dZdtheta)
      .def_readwrite("status_dZdzeta",&InterpolatedBoozerField::status_dZdzeta)
      .def_readwrite("status_dZds",&InterpolatedBoozerField::status_dZds)
      .def_readwrite("status_dnudtheta",&InterpolatedBoozerField::status_dnudtheta)
      .def_readwrite("status_dnudzeta",&InterpolatedBoozerField::status_dnudzeta)
      .def_readwrite("status_dnuds",&InterpolatedBoozerField::status_dnuds)
      .def_readwrite("status_dKdtheta",&InterpolatedBoozerField::status_dKdtheta)
      .def_readwrite("status_dKdzeta",&InterpolatedBoozerField::status_dKdzeta)
      .def_readwrite("status_K_derivs",&InterpolatedBoozerField::status_K_derivs)
      .def_readwrite("status_R_derivs",&InterpolatedBoozerField::status_R_derivs)
      .def_readwrite("status_Z_derivs",&InterpolatedBoozerField::status_Z_derivs)
      .def_readwrite("status_nu_derivs",&InterpolatedBoozerField::status_nu_derivs)
      .def_readwrite("status_modB_derivs",&InterpolatedBoozerField::status_modB_derivs)
      ;

    // ShearAlfvenWave:
    auto saw = py::class_<
        ShearAlfvenWave,
        ShearAlfvenWaveTrampoline<ShearAlfvenWave>,
        shared_ptr<ShearAlfvenWave>
        >(m, "ShearAlfvenWave", "")
        .def(py::init<shared_ptr<BoozerMagneticField>>())
        .def(
            "Phi",
            &ShearAlfvenWave::Phi,
            "Returns scalar potential Phi"
        )
        .def(
            "dPhidpsi",
            &ShearAlfvenWave::dPhidpsi,
            "Returns psi derivative of Phi"
        )
        .def(
            "Phidot",
            &ShearAlfvenWave::Phidot,
            "Returns time derivative of Phi"
        )
        .def(
            "dPhidtheta",
            &ShearAlfvenWave::dPhidtheta,
            "Returns theta derivative of Phi"
        )
        .def(
            "dPhidzeta",
            &ShearAlfvenWave::dPhidzeta,
            "Returns zeta derivative of Phi"
        )
        .def(
            "alpha",
            &ShearAlfvenWave::alpha,
            "Returns alpha value"
        )
        .def(
            "alphadot",
            &ShearAlfvenWave::alphadot,
            "Returns time derivative of alpha"
        )
        .def(
            "dalphadtheta",
            &ShearAlfvenWave::dalphadtheta,
            "Returns theta derivative of alpha"
        )
        .def(
            "dalphadpsi",
            &ShearAlfvenWave::dalphadpsi,
            "Returns psi derivative of alpha"
        )
        .def(
            "dalphadzeta",
            &ShearAlfvenWave::dalphadzeta,
            "Returns zeta derivative of alpha"
        )

        .def("Phi_ref", &ShearAlfvenWave::Phi_ref)
        .def("dPhidpsi_ref", &ShearAlfvenWave::dPhidpsi_ref)
        .def("dPhidtheta_ref", &ShearAlfvenWave::dPhidtheta_ref)
        .def("dPhidzeta_ref", &ShearAlfvenWave::dPhidzeta_ref)
        .def("Phidot_ref", &ShearAlfvenWave::Phidot_ref)
        .def("alphadot_ref", &ShearAlfvenWave::alphadot_ref)
        .def("dalphadtheta_ref", &ShearAlfvenWave::dalphadtheta_ref)
        .def("dalphadpsi_ref", &ShearAlfvenWave::dalphadpsi_ref)
        .def("dalphadzeta_ref", &ShearAlfvenWave::dalphadzeta_ref)

        .def("set_points", &ShearAlfvenWave::set_points)
        .def("get_points", &ShearAlfvenWave::get_points);

    // Phihat:
    py::class_<Phihat>(m, "Phihat")
        .def(
            py::init<const std::vector<double>&,
            const std::vector<double>&>()
        )
        .def("__call__", &Phihat::operator())
        .def("derivative", &Phihat::derivative)
        .def("get_s_basis", &Phihat::get_s_basis);

    // ShearAlfvenHarmonic:
    py::class_<
        ShearAlfvenHarmonic,
        ShearAlfvenWave,
        shared_ptr<ShearAlfvenHarmonic>
        >(m, "ShearAlfvenHarmonic")
        .def(py::init<const Phihat&, int, int, double, double, shared_ptr<BoozerMagneticField>>())
        .def_readwrite("Phim", &ShearAlfvenHarmonic::Phim)
        .def_readwrite("Phin", &ShearAlfvenHarmonic::Phin)
        .def_readwrite("omega", &ShearAlfvenHarmonic::omega)
        .def_readwrite("phase", &ShearAlfvenHarmonic::phase)
        .def_property_readonly("B0", &ShearAlfvenHarmonic::get_B0)
        .def_property_readonly("phihat", &ShearAlfvenHarmonic::get_phihat);

    // ShearAlfvenWavesSuperposition:
    py::class_<
        ShearAlfvenWavesSuperposition,
        ShearAlfvenWave,
        shared_ptr<ShearAlfvenWavesSuperposition>
        >(m, "ShearAlfvenWavesSuperposition")
        .def(py::init<shared_ptr<ShearAlfvenWave>>())
        .def("add_wave", &ShearAlfvenWavesSuperposition::add_wave)
        .def("set_points", &ShearAlfvenWavesSuperposition::set_points)
        .def("get_wave", &ShearAlfvenWavesSuperposition::get_wave)
        .def("size", &ShearAlfvenWavesSuperposition::size)
        .def("__len__", &ShearAlfvenWavesSuperposition::size)
        .def("__getitem__", [](const ShearAlfvenWavesSuperposition& self, size_t index) {
            return self.get_wave(index);
        })
        .def_property_readonly("B0", &ShearAlfvenWavesSuperposition::get_B0);
}
