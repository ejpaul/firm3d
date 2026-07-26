#!/bin/bash
# Run the GPU fusion birth distribution example (examples/gpu_boozer_tracing)
# on NCSA Delta. Requires firm3d already built (with CUDA) into the "firm3d"
# conda env via build_firm3d_delta.sh or install_firm3d_delta.sh.
#
# Usage: cd /path/to/parent && sbatch firm3d/install/delta/run_gpu_boozer_tracing.sh
#
# Single process, no MPI -- matches the Perlmutter reference script.
#SBATCH --job-name=firm3d-gpu-run
#SBATCH --account=bhvw-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.log

set -x
nvidia-smi -L

module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate firm3d

cd firm3d/examples/gpu_boozer_tracing

START=$(date +%s.%N)
python gpu_boozer_tracing.py
END=$(date +%s.%N)
echo "GPU_WALLCLOCK_SECONDS: $(echo "$END - $START" | bc)"
