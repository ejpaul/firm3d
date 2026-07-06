#!/bin/bash
#SBATCH -J collision-validation
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 00:30:00
#SBATCH -o validation-%j.out

# firm3d side of the ASCOT5 collision validation at the published test
# point (n = 1e20 m^-3) with high statistics, MPI over particles.
#
# One-time CPU-only build (login node, from the repo root):
#   module load cpu python cray-hdf5/1.14.3.7 cray-netcdf/4.9.2.1
#   conda activate firm3d-ci
#   env CC=cc CXX=CC pip install --no-build-isolation -e .
# (Do NOT load cudatoolkit / craype-accel-nvidia80: the GPU transport
# layer makes the Cray cc wrapper require libcudart at link time.)
#
# Usage: sbatch sample_slurm.sh [outdir]

module load python cray-hdf5/1.14.3.7 cray-netcdf/4.9.2.1
# cudatoolkit at RUNTIME only (booz_xform in the environment links
# libcudart); keep it out of the BUILD environment per the note above.
module load cudatoolkit 2>/dev/null
source $(conda info --base)/etc/profile.d/conda.sh
conda activate firm3d-ci

OUT=${1:-$SCRATCH/collision_validation}
mkdir -p "$OUT"
srun -n 128 python run_firm3d.py --outdir "$OUT" --density 1e20 --nmarkers 1024
