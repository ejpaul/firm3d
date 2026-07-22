#!/bin/bash
# Install firm3d (with CUDA) on NCSA Delta.
#
# Run this from an interactive GPU node allocation (e.g. via
# `srun --account=<your-gpu-account> --partition=gpuA100x4-interactive
#   --nodes=1 --gpus-per-node=1 --pty bash`), from the directory where
# firm3d has been cloned, e.g.:
#   cd /path/to/parent && bash firm3d/install/delta/install_firm3d_delta.sh
#
# Usage notes:
#   - Must run on a node with a physical GPU present. CMakeLists.txt
#     auto-detects CUDA via check_language(CUDA) and compiles with
#     `-arch=native`, which queries the device at build time, so a
#     login/CPU-only node will silently skip the CUDA extension.
#   - Uses conda-forge openmpi's mpicc/mpicxx in place of Perlmutter's
#     Cray cc/CC compiler wrappers.
#   - booz_xform's cmake/FindNetCDF.cmake only checks the NETCDF_DIR/
#     NETCDF_HOME/NETCDFDIR env vars (not CMAKE_PREFIX_PATH), so
#     NETCDF_DIR is exported to point at the conda env's libnetcdf.
#   - firm3d itself is installed in editable mode so the repo stays live.

set -euo pipefail

check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: $1. Exiting."
        exit 1
    fi
}

nvidia-smi

module load miniforge3-python

type conda >/dev/null 2>&1 || { echo "conda not found. Please load the miniforge3-python module first."; exit 1; }

echo "Enter the name for the new conda environment (e.g., firm3d):"
read -r -p "Environment name: " env_name

echo "Creating conda environment: $env_name"
conda create -n "$env_name" python=3.11 -y
check_success "Failed to create conda environment $env_name"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$env_name"
check_success "Failed to activate conda environment $env_name"

echo "Installing compilers, MPI, and NetCDF..."
conda install -y -c conda-forge compilers openmpi cmake ninja libnetcdf hdf5
check_success "Failed to install FIRM3D dependencies"

which nvcc
nvcc --version

# FIRM3D Installation
export NETCDF_DIR="$CONDA_PREFIX"
cd firm3d || { echo "Error: firm3d directory not found. Exiting."; exit 1; }
env CC="$CONDA_PREFIX/bin/mpicc" CXX="$CONDA_PREFIX/bin/mpicxx" NETCDF_DIR="$CONDA_PREFIX" pip install -e ".[dev]"
check_success "Failed to install FIRM3D"
cd ..

echo "=== Install check ==="
python -c "
from mpi4py import MPI
import firm3d
from firm3d.field.tracing import trace_particles_boozer
from firm3d.catapult.tracing import trace_particles_boozer_gpu
print('firm3d CPU + GPU bindings import OK')
"
check_success "FIRM3D import check failed"

echo "Successfully installed FIRM3D into the conda environment '$env_name'"
echo "To activate, run: conda activate $env_name"
