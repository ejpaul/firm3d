#!/bin/bash
#SBATCH --job-name=firm3d-gpu-ci
#SBATCH --partition=gpu
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --licenses=scratch

# Usage: sbatch run_gpu_tests.sh <work_dir>
#   work_dir  - directory containing the firm3d checkout (e.g. $PSCRATCH/firm3d-ci/<sha>)
#
# Expects a conda environment named "firm3d-ci" to already exist.
# Create it once with install/perlmutter/setup_ci_env.sh.

set -euo pipefail

WORK_DIR="${1:?Usage: $0 <work_dir>}"
FIRM3D_DIR="$WORK_DIR/firm3d"
EXIT_CODE_FILE="$WORK_DIR/test_exit_code.txt"

echo "=== firm3d GPU CI on $(hostname) ==="
echo "Work dir : $WORK_DIR"
echo "Commit   : $(git -C $FIRM3D_DIR rev-parse HEAD)"
echo "Started  : $(date)"

# ── Modules ───────────────────────────────────────────────────────────────────
module load cudatoolkit python cray-hdf5 cray-netcdf

# ── Conda environment ──────────────────────────────────────────────────────────
# Conda isn't available in non-interactive shells without sourcing its init.
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/.conda")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate firm3d-ci

# ── Build firm3d with CUDA ────────────────────────────────────────────────────
cd "$FIRM3D_DIR"
echo "--- Installing firm3d ---"
env CC=cc CXX=CC pip install -v --no-build-isolation -e ".[dev]" 2>&1 | tee "$WORK_DIR/build.log"

if ! grep -q "CUDA found. GPU bindings will be compiled." "$WORK_DIR/build.log"; then
    echo "ERROR: GPU bindings were not compiled." >&2
    echo "1" > "$EXIT_CODE_FILE"
    exit 1
fi

# ── GPU sanity check ──────────────────────────────────────────────────────────
nvidia-smi
python -c "import firm3dpp; print('firm3dpp loaded OK')"

# ── Run tests ─────────────────────────────────────────────────────────────────
echo "--- Running GPU tests ---"
python -m coverage run -m unittest tests.field.test_gpu
TEST_EXIT=$?

python -m coverage xml -o "$WORK_DIR/coverage.xml" || true

echo "$TEST_EXIT" > "$EXIT_CODE_FILE"
echo "Finished: $(date)  (exit $TEST_EXIT)"
exit $TEST_EXIT
