#!/bin/bash
#SBATCH -J fusion-alphas-collisional
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 03:00:00
#SBATCH -o fusion-collisional-%j.out

# Reactor-scale alpha slowing-down in Wistell-A (ARIES-CS scale) with
# Coulomb collisions: 1024 fusion-born alphas, 150 ms, with a
# collisionless companion run of the same ensemble.
#
# Build firm3d CPU-only first (see tests/perlmutter/run_slow_tests.sh).

module load python cray-hdf5/1.14.3.7 cray-netcdf/4.9.2.1
module load cudatoolkit 2>/dev/null
source $(conda info --base)/etc/profile.d/conda.sh
conda activate firm3d-ci

srun -n 128 python fusion_distribution_collisional.py
