#ifndef FIRM3DPP_MPI_UTILS_H
#define FIRM3DPP_MPI_UTILS_H

#ifdef USE_MPI
#include <mpi.h>

namespace firm3dpp {
namespace mpi {

// Helper function to convert Fortran MPI communicator handle to C MPI_Comm
// The Fortran handle is an integer obtained from comm.py2f() in Python
inline MPI_Comm get_mpi_comm_from_fortran(long long fortran_handle) {
    // Convert Fortran MPI communicator handle to C MPI_Comm
    // MPI_Comm_f2c converts a Fortran MPI communicator to a C MPI_Comm
    MPI_Fint comm_fortran = static_cast<MPI_Fint>(fortran_handle);
    MPI_Comm comm = MPI_Comm_f2c(comm_fortran);
    return comm;
}

} // namespace mpi
} // namespace firm3dpp

#endif // USE_MPI

#endif // FIRM3DPP_MPI_UTILS_H
