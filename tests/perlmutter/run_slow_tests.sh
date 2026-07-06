#!/bin/bash
#SBATCH -J firm3d-slow-tests
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH -t 00:30:00
#SBATCH -o slow-tests-%j.out

# Run the slow collision physics tests with MPI-parallel particle
# tracing.  Every rank runs the identical pytest session; inside each
# test the tracer distributes particles over ranks and allgathers the
# results (bit-identical to a serial run), so the assertions pass or
# fail identically on all ranks.  srun propagates any rank's nonzero
# exit code.
#
# One-time CPU-only build (login node, from the repo root):
#   module load cpu python cray-hdf5/1.14.3.7 cray-netcdf/4.9.2.1
#   conda activate firm3d-ci
#   env CC=cc CXX=CC pip install --no-build-isolation -e .
# (No cudatoolkit / craype-accel-nvidia80 at build time; cudatoolkit at
# runtime only.)
#
# Usage (from the repo root): sbatch tests/perlmutter/run_slow_tests.sh

module load python cray-hdf5/1.14.3.7 cray-netcdf/4.9.2.1
module load cudatoolkit 2>/dev/null
source $(conda info --base)/etc/profile.d/conda.sh
conda activate firm3d-ci

WORK=${SLURM_SUBMIT_DIR:-.}/slow-tests-$SLURM_JOB_ID
mkdir -p "$WORK"

srun -n 64 --output="$WORK/pytest-%t.log" \
    python -m pytest tests/field/test_collisions.py -m slow -v
EXIT=$?

echo "=== rank 0 pytest output ==="
cat "$WORK/pytest-0.log"
exit $EXIT
