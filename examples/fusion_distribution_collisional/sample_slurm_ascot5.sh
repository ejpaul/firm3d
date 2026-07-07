#!/bin/bash
#SBATCH -J wistell-ascot5
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 128
#SBATCH -t 03:00:00
#SBATCH -o wistell-ascot5-%j.out

# ASCOT5 companion for the reactor-scale Wistell-A collisional run
# (see run_ascot5_wistell.py).  Field import (a5py's vmec_field, scipy
# Delaunay triangulation per toroidal slice) is single-threaded and
# dominates at ~50 min for nphi=180 on comparable hardware; the
# GC-adaptive tracing phase after that uses OpenMP threading.
#
# One-time setup on a Perlmutter login node:
#   git clone https://github.com/ascot4fusion/ascot5.git ~/ascot5
#   sed -i 's/#define WIENERSLOTS 20/#define WIENERSLOTS 200/' \
#       ~/ascot5/src/ascot5.h   # default array is too small; see
#                               # run_ascot5_wistell.py's module docstring
#   module load cpu cray-hdf5/1.14.3.7
#   cd ~/ascot5 && make ascot5_main CC=cc -j8 -C src ascot5_main \
#       && make libascot CC=cc -j8
#   mkdir -p build && mv src/ascot5_main build/
#   module load python
#   source $(conda info --base)/etc/profile.d/conda.sh
#   conda activate firm3d-ci   # reuses firm3d's env
#   pip install -e ~/ascot5 --no-deps
#   pip install unyt xmlschema wurlitzer netCDF4
#
# Usage: sbatch sample_slurm_ascot5.sh [outdir]

module load cpu 2>/dev/null
module load cray-hdf5/1.14.3.7 python
source $(conda info --base)/etc/profile.d/conda.sh
conda activate firm3d-ci

OUT=${1:-$SCRATCH/wistell_ascot5}
mkdir -p "$OUT"
env OMP_NUM_THREADS=128 python run_ascot5_wistell.py \
    --ascot5-main "$HOME/ascot5/build/ascot5_main" --outdir "$OUT"
