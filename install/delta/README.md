# Running firm3d on NCSA Delta

This directory has everything needed to build firm3d (CPU + GPU/CUDA) and
run the two reference examples on NCSA Delta (`dt-login*.delta.ncsa.illinois.edu`),
and to compare timings against the Perlmutter reference numbers in the
example READMEs.

## Cluster facts that matter here

- SLURM scheduler. Relevant partitions: `cpu` / `cpu-interactive` (CPU-only,
  128 cores/node, ~2 GB/core -- about half of Perlmutter's ~4 GB/core),
  `gpuA100x4` / `gpuA100x4-interactive` (64 CPUs + 4x A100/node).
- `module load miniforge3-python` provides conda.
- No generic Linux install path is documented in the top-level firm3d
  `README.md` (only macOS and Perlmutter) -- these scripts fill that gap,
  modeled on `install/perlmutter/install_firm3d_perlmutter.sh`.
- **CUDA build requires a physical GPU at build time.** `CMakeLists.txt`
  detects CUDA via `check_language(CUDA)` and compiles with `-arch=native`,
  which queries the device. Build on a GPU-allocated compute node, not a
  login node, or the CUDA extension is silently skipped.
- **MPI: use the system `cray-mpich`, not conda-forge's `openmpi` package.**
  conda-forge's `openmpi` does not reliably integrate with Slurm's `srun`
  launcher on Delta -- `srun -n N ...` can silently fall back to N
  independent singleton MPI processes (`rank=0 size=1` each) instead of one
  real N-rank job, which breaks firm3d's particle-to-rank distribution
  without raising an error. Confirmed failing 5/5 times across two
  independent investigations (see
  `firm3d_delta_mpi_reliability_findings.md`); `module load PrgEnv-gnu
  cray-mpich` fixes it deterministically (verified: unique ranks 0..N-1).
  All scripts here build and run against cray-mpich.

## 1. Build

Two options, both produce a conda env named `firm3d` with CPU + GPU bindings:

- **`install_firm3d_delta.sh`** -- interactive, mirrors the Perlmutter
  install script's UX (prompts for an env name). Run manually from a
  GPU-node shell.
- **`build_firm3d_delta.sh`** -- non-interactive `sbatch` script, fixed env
  name `firm3d`. Preferred for scripted/agent use.

```bash
cd /path/to/parent   # directory that contains (or will contain) firm3d/
sbatch firm3d/install/delta/build_firm3d_delta.sh
```

Key adaptations from the Perlmutter script:

| Perlmutter | Delta |
|---|---|
| `module load python cray-hdf5 cray-netcdf`, clone `nersc-python` | `module load miniforge3-python`, fresh `conda create -n firm3d python=3.11` |
| Cray `cc`/`CC` compiler wrappers | system `cray-mpich`'s `mpicc`/`mpicxx` (`module load PrgEnv-gnu cray-mpich`) -- **not** conda-forge's `openmpi` package, see the MPI gotcha above |
| `cray-netcdf` module | conda-forge `libnetcdf` + `hdf5` packages (note: the package is named `libnetcdf`, **not** `netcdf-c`) |
| n/a | `booz_xform`'s `cmake/FindNetCDF.cmake` only checks the `NETCDF_DIR`/`NETCDF_HOME`/`NETCDFDIR` env vars (not `CMAKE_PREFIX_PATH`), so `NETCDF_DIR=$CONDA_PREFIX` must be exported before `pip install` |
| n/a | `mpi4py` must be rebuilt from source (`pip install --no-binary mpi4py mpi4py`) against cray-mpich's `mpicc` -- the conda-forge binary wheel bundles its own (openmpi) MPI |
| n/a | `nlohmann_json` must come from **pip**, not conda-forge -- `setup.py` locates the header via `import nlohmann_json; Path(nlohmann_json.__file__).parent / "include"`, a package layout conda-forge's header-only build doesn't provide |
| n/a | `gpu_boozer_tracing.py` imports `pandas`, which isn't a declared firm3d dependency -- installed separately |

Build takes about 15-20 min (mostly the booz_xform + firm3dpp CMake/Ninja
compiles). Verify with:

```bash
module load PrgEnv-gnu cray-mpich
conda activate firm3d
python -c "from firm3d.field.tracing import trace_particles_boozer; from firm3d.catapult.tracing import trace_particles_boozer_gpu; print('OK')"
```

## 2. Run

```bash
sbatch firm3d/install/delta/run_cpu_fusion_distribution.sh   # examples/fusion_distribution
sbatch firm3d/install/delta/run_gpu_boozer_tracing.sh        # examples/gpu_boozer_tracing
```

Both examples initialize particles proportional to the D-T fusion
reactivity profile (`reactivity(s) = nD(s)*nT(s)*sigmav(T(s))`) in the
Wistell-A/ARIES-CS Boozer field and trace until loss or `tmax`. Each script
times only the tracing call itself and prints `CPU_WALLCLOCK_SECONDS` /
`GPU_WALLCLOCK_SECONDS` at the end of its log.

### CPU run gotchas (all already handled in `run_cpu_fusion_distribution.sh`)

1. **OOM at 128 ranks**: Delta's ~2 GB/core (vs Perlmutter's ~4 GB/core)
   means the Perlmutter reference's `-n 128` config OOMs some ranks unless
   you request the full node memory pool: `#SBATCH --mem=0`.
2. **Thread oversubscription**: without `OMP_NUM_THREADS=1`,
   `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, BLAS/OpenMP threads
   oversubscribe cores under many MPI ranks and cause OOM. Perlmutter's
   `module load python` reportedly sets these by default; Delta's
   `miniforge3-python` does not.
3. **Shared account billing quota**: the `cpu-interactive` QOS has a
   `GrpTRESMins` budget shared across every user on the account
   (`TRESBillingWeights=CPU=2000/min`, so a 128-task job costs
   256,000 billing-minutes per minute of wall time). If another user on the
   account has recently run large jobs, a 128-rank request can sit
   `PENDING` with reason `QOSGrpBillingMinutes` indefinitely. Check
   `sacctmgr show qos` / `sacct -o AllocTRES%60` to gauge headroom, or just
   drop `--ntasks` (and the matching `srun -n`) until the job clears the
   queue. **This is why the script here defaults to `n=32` rather than
   Perlmutter's `n=128`** -- bump it back up if your account has headroom;
   it's a straight `--ntasks`/`srun -n` edit.
4. **conda-forge `openmpi` breaks `srun`'s rank coordination.** If you see
   `No PMIx server was reachable... N singletons will be started` in the
   log, the job silently ran as N independent singleton processes (each
   `rank=0 size=1`) rather than one real N-rank job, and it will hang to
   the time limit without producing a valid result. This was originally
   believed to be a rare/transient node issue ("just resubmit") but was
   later shown to fail deterministically (5/5) when built against
   conda-forge's `openmpi` package, across two independent investigations
   (see `firm3d_delta_mpi_reliability_findings.md`). **The real fix is to
   build and run against the system `cray-mpich`** (`module load
   PrgEnv-gnu cray-mpich`), which all scripts in this directory now do --
   this is no longer a "just resubmit" gotcha, it's a build-configuration
   requirement.

## 3. Results (Delta A100 / CPU node, vs. Perlmutter reference)

| Example | Config | Delta wallclock | Perlmutter reference | Notes |
|---|---|---|---|---|
| `gpu_boozer_tracing` (GPU) | 1000 particles, single process, 1x A100 | **35.8 s** | 125 s (04.20.26) | Same particle count/config -- clean comparison. ~3.5x faster. |
| `fusion_distribution` (CPU) | 5000 particles, `n=32` MPI ranks, cray-mpich | **455.6 s** total (439.1 s tracing-only) | 84 s (06.11.25) at `n=128` | **Not apples-to-apples** -- Delta ran at 1/4 the rank count (see quota gotcha above), so each rank did ~4x the particles. Rerun at `n=128` if quota allows for a like-for-like number. |

This CPU number was produced with the cray-mpich build (job `20526596`),
confirmed via a standalone rank/size probe (job `20526546`: ranks 0..31,
size=32 -- no singletons) and via the run's own log filename
(`stdout_5000_48_32.txt` encodes `comm_size=32`) immediately beforehand.
An earlier n=32 run against conda-forge `openmpi` (job `20493592`)
happened to also land on a working size-32 communicator and reported a
consistent tracing time (435.4 s) in the same log file -- but per the MPI
gotcha above, that success was not guaranteed to reproduce, which is why
this result is now pinned to the cray-mpich build instead.

To reproduce the Perlmutter-equivalent CPU comparison, edit
`run_cpu_fusion_distribution.sh`: set `--ntasks=128` and `srun -n 128`
(keep `--mem=0` and the thread-pinning exports), and expect to need to
navigate the billing-quota gotcha above.

## References

- `firm3d_delta_mpi_reliability_findings.md` -- the conda-forge-openmpi
  vs. cray-mpich reliability investigation referenced above (4/4 openmpi
  failures at `n=32` on both `cpu` and `cpu-interactive`). Written to
  `/projects/bhvw/epaul/research/20260723_g1600_collisional_delta/` on
  the cluster, outside this repo.
- `firm3d_collisional_oom_diagnostic.md` (same directory) -- an
  independent diagnosis of the same singleton-MPI failure mode in a
  separate collisional-transport benchmarking project, plus an unrelated
  but useful confirmation that firm3d's physics (not the Delta launch
  environment) was the actual source of truth once cray-mpich + thread
  pinning + `--mem=0` were all applied (firm3d vs. ASCOT5 agreement:
  0.58σ).
- `delta-install-script-cpu` branch (commit `28f3b3ed`,
  `install/delta/install_firm3d_delta_cpu.sh`) -- the CPU-only,
  interactive cray-mpich install script this branch's scripts were
  aligned with.
