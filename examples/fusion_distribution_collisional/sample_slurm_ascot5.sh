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
# One-time setup on a Perlmutter login node.  Two source patches are
# required beyond a stock ASCOT5 checkout; see run_ascot5_wistell.py's
# module docstring for the full diagnosis of each (a Wiener-array
# capacity limit, and a near-axis floating-point edge case in the
# generic rho evaluator that has no compile-time escape hatch):
#   git clone https://github.com/ascot4fusion/ascot5.git ~/ascot5
#   sed -i 's/#define WIENERSLOTS 20/#define WIENERSLOTS 200/' \
#       ~/ascot5/src/ascot5.h
#   # In ~/ascot5/src/B_field.c, B_field_eval_rho(): replace
#   #     if( (psi - psi0) / delta < 0 ) {
#   #          err = error_raise( ERR_INPUT_UNPHYSICAL, __LINE__, EF_B_FIELD );
#   #     } else {
#   # with
#   #     if( (psi - psi0) / delta < 0 ) {
#   #         rho[0] = 0.0;
#   #         rho[1] = 0.0;
#   #     } else {
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
