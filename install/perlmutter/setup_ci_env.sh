#!/bin/bash
# One-time setup of the "firm3d-ci" conda environment on Perlmutter.
# Run this interactively from a login node before using perlmutter-ci.yml.
#
# Usage:
#   bash install/perlmutter/setup_ci_env.sh
#
# What it does:
#   1. Loads required modules
#   2. Creates a conda env named firm3d-ci (cloned from nersc-python)
#   3. Installs Python dependencies that don't need CUDA to build
#
# firm3d itself (the CUDA extension) is built fresh in each CI job so
# the compiled .so always matches the code under test.

set -euo pipefail

ENV_NAME="firm3d-ci"

module load cudatoolkit python cray-hdf5/1.14.3.7 cray-netcdf/4.9.2.1 cray-mpich

echo "Creating conda environment '$ENV_NAME' from nersc-python..."
conda create -n "$ENV_NAME" --clone nersc-python -y

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install \
    "numpy>=2.0" \
    netCDF4 \
    cmake \
    ninja \
    "pybind11<=2.13.6" \
    setuptools \
    wheel \
    scipy \
    booz-xform \
    ndsplines \
    coverage \
    pytest \
    simsopt \
    pandas \
    plotly \
    pyevtk \
    matplotlib \
    ruamel.yaml \
    nptyping \
    Deprecated \
    mpi4py

echo ""
echo "Done. Environment '$ENV_NAME' is ready."
echo "To test locally: conda activate $ENV_NAME"
