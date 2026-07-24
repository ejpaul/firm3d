#!/bin/bash
# Install firm3d on NCSA Delta (CPU build). For a GPU build, see
# install_firm3d_delta.sh on the delta-install-script branch (requires
# an interactive GPU node allocation; also see the "GPU note" below).
#
# Usage: from the directory where firm3d has been cloned, e.g.:
#   cd /path/to/parent && bash firm3d/install/delta/install_firm3d_delta_cpu.sh
#
# Usage notes:
#   - Uses the SYSTEM cray-mpich (via PrgEnv-gnu) for mpicc/mpicxx, NOT
#     conda-forge's own openmpi package. conda-forge's openmpi does not
#     integrate with Slurm's srun launcher on Delta: `srun -n N ...`
#     silently starts N independent single-process ("singleton", rank=0
#     size=1) MPI jobs instead of one N-rank job -- confirmed via
#     mpi4py.MPI.COMM_WORLD.Get_rank()/Get_size() under both plain srun
#     and `srun --mpi=pmix` (the latter fails outright: PMIx's `psec`
#     framework can't find/load the `munge` security component in this
#     environment). This silently breaks firm3d's own particle
#     distribution across MPI ranks: every rank redundantly traces the
#     FULL particle set instead of its own 1/N share, causing ~Nx
#     redundant compute and ~Nx memory blowup (observed to cause both
#     multi-hour runtimes and OOM kills at N=128 on a 10000-particle
#     collisional trace). Building against cray-mpich instead fixes
#     this (verified: unique ranks 0..N-1, size=N).
#   - nlohmann_json MUST be the pip package, not conda-forge's. firm3d's
#     own setup.py locates the header via `import nlohmann_json; Path(
#     nlohmann_json.__file__).parent / "include"` -- a Python package
#     layout that conda-forge's nlohmann_json (a plain C++ header-only
#     library with no importable module) doesn't provide. Passing
#     -Dnlohmann_json_INCLUDE_DIR via CMAKE_ARGS as a workaround does
#     NOT reach cmake's configure step through pip's build backend in
#     this setup (observed empirically -- the env var never shows up in
#     the cmake configure log), so just `pip install nlohmann_json`.
#   - booz_xform's cmake/FindNetCDF.cmake only checks the NETCDF_DIR/
#     NETCDF_HOME/NETCDFDIR env vars (not CMAKE_PREFIX_PATH), so
#     NETCDF_DIR is exported to point at the conda env's libnetcdf.
#   - firm3d itself is installed in editable mode so the repo stays live.
#
# GPU note: this script targets the CPU-only path used for orbit-
# integrator comparisons against ASCOT5. CMakeLists.txt auto-detects
# CUDA via check_language(CUDA) -- if an nvcc is on PATH (e.g. via a
# loaded cudatoolkit module) the CUDA extension gets compiled REGARDLESS
# of whether a physical GPU is present on the node (check_language only
# checks for a working compiler, not for GPU hardware), which just adds
# unnecessary build time here. For an actual GPU build/run, do this from
# an interactive GPU node allocation (e.g. `srun --account=<gpu account>
# --partition=gpuA100x4-interactive --nodes=1 --gpus-per-node=1 --pty
# bash`) so `-arch=native` picks up the right compute capability.

set -euo pipefail

check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: $1. Exiting."
        exit 1
    fi
}

module load PrgEnv-gnu cray-mpich
module load miniforge3-python

type conda >/dev/null 2>&1 || { echo "conda not found. Please load the miniforge3-python module first."; exit 1; }

MPICC=$(which mpicc)
MPICXX=$(which mpicxx)
echo "Using cray-mpich compiler wrappers: $MPICC / $MPICXX"

echo "Enter the name for the new conda environment (e.g., firm3d):"
read -r -p "Environment name: " env_name

echo "Creating conda environment: $env_name"
conda create -n "$env_name" python=3.11 -y
check_success "Failed to create conda environment $env_name"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$env_name"
check_success "Failed to activate conda environment $env_name"

# Note: no `openmpi` package here -- see the module docstring above for
# why conda-forge's own MPI breaks Slurm's srun launcher on Delta.
echo "Installing compilers, CMake, and NetCDF..."
conda install -y -c conda-forge cmake ninja libnetcdf hdf5 'pybind11<=2.13.6' numpy
check_success "Failed to install FIRM3D dependencies"

# See module docstring: pip's nlohmann_json (not conda-forge's) is
# required so setup.py's `import nlohmann_json` include-path detection
# works.
pip install nlohmann_json
check_success "Failed to install nlohmann_json"

# mpi4py must link against cray-mpich too (it ships its own bundled MPI
# otherwise via conda-forge's binary wheel).
pip uninstall -y mpi4py 2>/dev/null || true
env CC="$MPICC" MPICC="$MPICC" pip install --no-cache-dir --no-binary mpi4py mpi4py
check_success "Failed to build mpi4py against cray-mpich"

# FIRM3D Installation
export NETCDF_DIR="$CONDA_PREFIX"
cd firm3d || { echo "Error: firm3d directory not found. Exiting."; exit 1; }
env CC="$MPICC" CXX="$MPICXX" NETCDF_DIR="$CONDA_PREFIX" pip install --no-cache-dir --no-build-isolation -e ".[dev]"
check_success "Failed to install FIRM3D"
cd ..

echo "=== Install check ==="
python -c "
from mpi4py import MPI
import firm3d
from firm3d.field.tracing import trace_particles_boozer
from firm3d.field.collisions import trace_particles_boozer_with_collisions
print(f'firm3d CPU bindings import OK, rank={MPI.COMM_WORLD.Get_rank()} size={MPI.COMM_WORLD.Get_size()}')
"
check_success "FIRM3D import check failed"

echo "Successfully installed FIRM3D into the conda environment '$env_name'"
echo "To activate, run: conda activate $env_name"
echo ""
echo "IMPORTANT: at runtime (e.g. in a Slurm job script), also load"
echo "  module load PrgEnv-gnu cray-mpich"
echo "before activating the conda env and running srun -- omitting this"
echo "reproduces the singleton-MPI issue described above even though the"
echo "build itself succeeded."
