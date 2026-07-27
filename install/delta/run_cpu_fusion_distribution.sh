#!/bin/bash
# Run the CPU fusion birth distribution example (examples/fusion_distribution)
# on NCSA Delta. Requires firm3d already built into the "firm3d" conda env
# via build_firm3d_delta.sh or install_firm3d_delta.sh.
#
# Usage: cd /path/to/parent && sbatch firm3d/install/delta/run_cpu_fusion_distribution.sh
#
# Notes:
#   - --mem=0 (request the full node's memory pool) is required: Delta CPU
#     nodes have ~2 GB/core vs. Perlmutter's ~4 GB/core, and a fixed
#     per-task memory slice OOM-kills outlier particles' adaptive stepping.
#   - OMP/OPENBLAS/MKL_NUM_THREADS=1 pin each MPI rank to one thread;
#     without this, BLAS/OpenMP oversubscription also causes OOM.
#   - ntasks=32 here (vs. Perlmutter's reference n=128) because this
#     account's cpu-interactive QOS has a shared TRES billing budget
#     (TRESBillingWeights CPU=2000/min) across all users on the account;
#     bump --ntasks up (and adjust -n in srun to match) if your account has
#     more headroom -- 128 ranks matches the Perlmutter reference exactly.
#   - Must load `PrgEnv-gnu cray-mpich` (matching the build) before
#     activating the conda env. conda-forge's own openmpi package does NOT
#     integrate with Slurm's srun launcher on Delta: srun silently falls
#     back to N independent singleton MPI processes ("No PMIx server was
#     reachable... N singletons will be started") instead of one N-rank
#     job, so every rank redundantly traces the FULL particle set and the
#     job hangs to the time limit without producing a valid result. This
#     reproduced 5/5 times across two independent investigations (not a
#     transient/intermittent issue as earlier believed) -- see
#     firm3d_delta_mpi_reliability_findings.md. Building/running against
#     cray-mpich instead fixes it deterministically.
#SBATCH --job-name=firm3d-cpu
#SBATCH --account=bhvw-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.log

set -x
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

module load PrgEnv-gnu cray-mpich
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate firm3d

cd firm3d/examples/fusion_distribution

START=$(date +%s.%N)
srun -n 32 -c 1 python -u fusion_distribution.py
END=$(date +%s.%N)
echo "CPU_WALLCLOCK_SECONDS: $(echo "$END - $START" | bc)"
