#!/bin/bash
#SBATCH --nodes=1
# 10000 particles is ~1250 blocks of 8 against 108 SMs, so this no longer fits
# the 30 minute ceiling on the debug qos.
#SBATCH --time=2:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4680 # Change to your account number

module load python cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13
conda activate firm3d-dev # Change to the name of your environment
python gpu_cartesian_collisional_tracing.py
