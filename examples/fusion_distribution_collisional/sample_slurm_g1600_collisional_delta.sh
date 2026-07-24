#!/bin/bash
#SBATCH -J g1600-collisional-delta
#SBATCH -A <your-cpu-account>
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 06:00:00
#SBATCH -o g1600-collisional-delta-%j.out

# G1600 reactor-scale alpha slowing-down with Coulomb collisions, on
# NCSA Delta: 10000 fusion-born alphas, 150 ms, res=48/tol=1e-8/128
# ranks. DP_hmin=1e-8 floors the Dormand-Prince step size so particles
# that thermalize (pitch-scattering rate ~1/v^3 diverges near the
# background thermal speed) don't grind the wall-clock time and memory
# use to grow without bound as billions of ever-shrinking steps are
# attempted (see install/delta/install_firm3d_delta.sh's module
# docstring for the build itself, and the DP_hmin docstring in
# firm3d.field.collisions.trace_particles_boozer_with_collisions for
# the physics).
#
# IMPORTANT: cray-mpich must be loaded here too (not just at build
# time) -- without it, srun silently falls back to launching N
# independent single-process ("singleton") MPI jobs instead of one
# N-rank job, so firm3d's particle distribution across ranks silently
# breaks (every rank redundantly traces the FULL particle set).

module load PrgEnv-gnu cray-mpich
module load miniforge3-python
source $(conda info --base)/etc/profile.d/conda.sh
conda activate firm3d

srun -n 128 python g1600_collisional_validation.py \
  --nparticles 10000 \
  --shared-ensemble g1600_shared_birth_ensemble_10k.npz \
  --forget-exact-path \
  --dp-hmin 1e-8 \
  --tag 10k_res48
