#!/bin/bash
# Non-interactive build of firm3d (CPU + GPU bindings) in a conda env on
# NCSA Delta. Companion to install_firm3d_delta.sh (the interactive,
# Perlmutter-style version) -- use this one for unattended/scripted setup.
#
# Must run on a GPU-allocated compute node: CMakeLists.txt auto-detects CUDA
# via check_language(CUDA) and compiles with `-arch=native`, which queries
# the physical GPU at build time. Building on a login/CPU-only node will
# silently skip the CUDA extension, and GPU examples will fail at import.
#
# Usage:
#   cd /path/to/parent && sbatch firm3d/install/delta/build_firm3d_delta.sh
# (submit from the directory containing/to contain the firm3d clone; if
# firm3d/ doesn't exist there yet, it will be cloned)
#SBATCH --job-name=firm3d-build
#SBATCH --account=bhvw-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.log

set -euo pipefail
set -x

ENV_NAME=firm3d

# --- repo + submodules -----------------------------------------------------
if [ ! -d firm3d ]; then
    git clone https://github.com/ColumbiaStellaratorTheory/firm3d.git
fi
cd firm3d
git submodule update --init --recursive

nvidia-smi

# --- conda env ---------------------------------------------------------
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

# compilers + openmpi (mpicc/mpicxx replace Perlmutter's Cray cc/CC)
# libnetcdf (NOT netcdf-c, which doesn't exist on conda-forge) is needed
# because firm3d's booz_xform dependency builds against a system NetCDF-C lib
conda install -y -c conda-forge compilers openmpi cmake ninja libnetcdf hdf5

which nvcc
nvcc --version

# --- build firm3d + booz_xform ---------------------------------------------
# booz_xform's cmake/FindNetCDF.cmake only checks NETCDF_DIR/NETCDF_HOME/
# NETCDFDIR env vars (not CMAKE_PREFIX_PATH), so point it at the conda env
export NETCDF_DIR="$CONDA_PREFIX"

env CC="$CONDA_PREFIX/bin/mpicc" CXX="$CONDA_PREFIX/bin/mpicxx" \
    NETCDF_DIR="$CONDA_PREFIX" \
    pip install -e ".[dev]"

# gpu_boozer_tracing.py example needs pandas; not a declared firm3d dependency
pip install pandas

# --- verify ------------------------------------------------------------
echo "=== Install check ==="
python -c "
from mpi4py import MPI
import firm3d
from firm3d.field.tracing import trace_particles_boozer
from firm3d.catapult.tracing import trace_particles_boozer_gpu
print('firm3d CPU + GPU bindings import OK')
"

echo "Build complete. Activate with: conda activate $ENV_NAME"
