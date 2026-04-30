#!/bin/bash
# Install firm3d (with CUDA) + simsopt into a single conda environment on
# Perlmutter, suitable for running the GPU Cartesian tracing examples.
#
# Usage (run interactively from a login node inside the firm3d repo root):
#   bash install/perlmutter/install_simsopt_firm3d_perlmutter.sh
#
# Install order matters:
#   1. Build firm3d first — this pulls in booz-xform which can pin numpy<2.
#   2. Install numpy>=2.0 and simsopt afterwards so that the final numpy
#      version is >=2.0 (required by jax 0.4+, which simsopt depends on).

set -euo pipefail

check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: $1. Exiting." >&2
        exit 1
    fi
}

# ── Modules ────────────────────────────────────────────────────────────────────
module load cudatoolkit python cray-hdf5/1.14.3.7 cray-netcdf/4.9.2.1 cray-mpich
check_success "Failed to load modules"

# ── Conda environment ──────────────────────────────────────────────────────────
type conda >/dev/null 2>&1 || { echo "conda not found. Load the python module first."; exit 1; }

echo "Enter the name for the new conda environment (e.g., firm3d-simsopt):"
read -r -p "Your input: " ENV_NAME

echo "Creating conda environment '$ENV_NAME' from nersc-python..."
conda create -n "$ENV_NAME" --clone nersc-python -y
check_success "Failed to create conda environment"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
check_success "Failed to activate conda environment"

pip install --upgrade pip

# ── Base dependencies (before firm3d build) ────────────────────────────────────
echo "Installing base dependencies..."
pip install \
    netCDF4 \
    cmake \
    ninja \
    "pybind11<=2.13.6" \
    setuptools \
    wheel \
    scipy \
    ndsplines \
    mpi4py \
    pandas \
    plotly \
    pyevtk \
    matplotlib \
    ruamel.yaml \
    nptyping \
    Deprecated \
    coverage \
    pytest
check_success "Failed to install base dependencies"

# ── firm3d with CUDA ───────────────────────────────────────────────────────────
# Must be built before simsopt/numpy pin so the booz-xform numpy constraint
# does not break jax later.
echo "Building firm3d with CUDA support..."
env CC=cc CXX=CC pip install --no-build-isolation -e "." 2>&1 | tee /tmp/firm3d_build.log
check_success "Failed to build firm3d"

if ! grep -q "CUDA found. GPU bindings will be compiled." /tmp/firm3d_build.log; then
    echo "WARNING: CUDA GPU bindings were NOT compiled." >&2
    echo "Ensure the cudatoolkit module is loaded and a GPU is available for testing." >&2
fi

# ── numpy + simsopt (installed after firm3d to win the numpy version pin) ─────
# jax 0.4+ requires numpy>=2.0.  booz-xform (pulled by firm3d) can downgrade
# numpy to 1.26.x, so we re-pin here after the firm3d build is done.
echo "Installing numpy>=2.0 and simsopt..."
pip install "numpy>=2.0" simsopt
check_success "Failed to install simsopt"

# ── Verify ─────────────────────────────────────────────────────────────────────
echo ""
python - <<'PYCHECK'
import numpy, simsopt, firm3dpp
print(f"numpy   : {numpy.__version__}")
print(f"simsopt : {simsopt.__version__}")
print(f"firm3dpp: loaded OK")
try:
    from firm3d.catapult.tracing import trace_particles_cartesian_gpu
    print("trace_particles_cartesian_gpu: available")
except ImportError:
    print("trace_particles_cartesian_gpu: NOT available (gpu_cartesian branch required)")
PYCHECK

echo ""
echo "Done. Environment '$ENV_NAME' is ready."
echo "To activate: conda activate $ENV_NAME"
