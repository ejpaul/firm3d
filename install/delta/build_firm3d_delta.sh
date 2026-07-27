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
# Uses the SYSTEM cray-mpich (via PrgEnv-gnu), NOT conda-forge's openmpi
# package -- conda-forge's openmpi silently breaks Slurm's srun launcher on
# Delta (falls back to N independent singleton MPI processes instead of one
# N-rank job; confirmed 5/5 across two independent investigations, see
# firm3d_delta_mpi_reliability_findings.md). Building against cray-mpich
# fixes this. See install_firm3d_delta.sh's header for the full rationale
# and the nlohmann_json pip-vs-conda gotcha.
#
# Usage:
#   cd /path/to/parent && sbatch firm3d/install/delta/build_firm3d_delta.sh
# (submit from the directory containing/to contain the firm3d clone; if
# firm3d/ doesn't exist there yet, it will be cloned)
#
# IMPORTANT: at runtime (e.g. in a Slurm run script), also
# `module load PrgEnv-gnu cray-mpich` before activating the conda env and
# running srun -- omitting this reproduces the singleton-MPI issue even
# though the build itself succeeded.
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

# --- modules + conda env -------------------------------------------------
module load PrgEnv-gnu cray-mpich
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"

MPICC=$(which mpicc)
MPICXX=$(which mpicxx)
echo "Using cray-mpich compiler wrappers: $MPICC / $MPICXX"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

# Defensively remove any leftover conda-forge openmpi from a prior run of
# this script (e.g. before this cray-mpich switch) -- its libmpi.so on
# LD_LIBRARY_PATH would otherwise be picked up ahead of cray-mpich's at
# runtime, silently reintroducing the singleton-MPI bug even though the
# extension was compiled against cray-mpich.
conda remove -y openmpi mpi 2>/dev/null || true

# Note: no `openmpi` package here -- see the header above for why
# conda-forge's own MPI breaks Slurm's srun launcher on Delta.
# libnetcdf (NOT netcdf-c, which doesn't exist on conda-forge) is needed
# because firm3d's booz_xform dependency builds against a system NetCDF-C lib
conda install -y -c conda-forge cmake ninja libnetcdf hdf5 'pybind11<=2.13.6' numpy

# pip's nlohmann_json (not conda-forge's) is required so setup.py's
# `import nlohmann_json` include-path detection works -- see header above.
pip install nlohmann_json

which nvcc
nvcc --version

# --- mpi4py against cray-mpich ---------------------------------------------
# mpi4py ships its own bundled MPI via conda-forge's binary wheel otherwise,
# which would silently defeat the point of building against cray-mpich.
pip uninstall -y mpi4py 2>/dev/null || true
env CC="$MPICC" MPICC="$MPICC" pip install --no-cache-dir --no-binary mpi4py mpi4py

# --- build firm3d + booz_xform ---------------------------------------------
# booz_xform's cmake/FindNetCDF.cmake only checks NETCDF_DIR/NETCDF_HOME/
# NETCDFDIR env vars (not CMAKE_PREFIX_PATH), so point it at the conda env
export NETCDF_DIR="$CONDA_PREFIX"

env CC="$MPICC" CXX="$MPICXX" \
    NETCDF_DIR="$CONDA_PREFIX" \
    pip install --no-cache-dir --no-build-isolation -e ".[dev]"

# gpu_boozer_tracing.py example needs pandas; not a declared firm3d dependency
pip install pandas

# --- verify ------------------------------------------------------------
echo "=== Install check ==="
python -c "
from mpi4py import MPI
import firm3d
from firm3d.field.tracing import trace_particles_boozer
from firm3d.catapult.tracing import trace_particles_boozer_gpu
print(f'firm3d CPU + GPU bindings import OK, rank={MPI.COMM_WORLD.Get_rank()} size={MPI.COMM_WORLD.Get_size()}')
"

echo "Build complete. Activate with:"
echo "  module load PrgEnv-gnu cray-mpich"
echo "  conda activate $ENV_NAME"
