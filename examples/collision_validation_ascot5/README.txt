Collision-operator validation against ASCOT5
=============================================

Cross-code validation of firm3d's Monte Carlo Coulomb collision
implementation (trace_particles_boozer_with_collisions) against ASCOT5,
mirroring the alpha slowing-down branch of ASCOT5's built-in physics
test (a5py/testascot/physicstests.py, "ccoll").

Case: N fusion alphas (3.5 MeV, uniform random pitch, fixed seed shared
by both codes) slow down in a uniform hydrogen plasma (n_e = n_i,
T_e = T_i = 1 keV).  The geometries differ (ASCOT5: analytical ITER-like
circular tokamak; firm3d: BoozerAnalytic near-axis field at the same
R0 = 6.2 m, B0 = 5.3 T scale), but the plasma is uniform, so the
velocity-space moments compared here are geometry-insensitive:

  * mean slowing-down time to 50 keV (vs each other and vs the Spitzer
    estimate 0.5 ts ln(E0/Emin)),
  * <E(t)> and <xi(t)> over the slowing-down window,
  * the distribution of slowing-down times.

All collision rates scale linearly with density, so params.py defaults
to n = 1e21 m^-3 (a ~3 ms slowing window, minutes of local wall time).
Physics is identical to the published ASCOT5 test point (1e20) up to
the ~10 % change in ln Lambda, which both codes compute consistently.

Prerequisites
-------------
* firm3d built and importable (any environment).
* ASCOT5 (https://github.com/ascot4fusion/ascot5) built:
    make ascot5_main libascot     # see ASCOT5 docs; on macOS use a
                                  # conda env with hdf5 + clang
  and a5py installed (pip install -e . in the ascot5 repo).

Running
-------
    # 1. ASCOT5 side (in the environment with a5py):
    python run_ascot5.py --ascot5-main /path/to/ascot5/build/ascot5_main \
                         --outdir /tmp/validation

    # 2. firm3d side (in the environment with firm3d):
    python run_firm3d.py --outdir /tmp/validation

    # 3. Compare (either environment; needs matplotlib):
    python compare.py --outdir /tmp/validation

compare.py prints a summary table, writes validation_comparison.png,
and exits nonzero if the mean slowing-down times disagree by more than
3 combined standard errors or 10 %.

Note that only the energy moments validate the collision operator
quantitatively.  <xi(t)> is shown for qualitative comparison: pitch is
sampled along orbits and the two codes' fields have different mirror
ratios and trapped fractions, so an O(0.1) offset between the curves is
geometric rather than collisional.

Reference local result (2026-07-05, N = 100, n = 1e21 m^-3):
  mean slowing time  firm3d 2.680 +/- 0.009 ms,
                     ASCOT5 2.690 +/- 0.011 ms   (-0.4 %, 0.7 sigma)
  max |<E>_firm3d - <E>_ascot5| / E0 = 0.4 %

Reference high-statistics result at the published ASCOT5 test point
(2026-07-06, N = 1024, n = 1e20 m^-3; firm3d on one Perlmutter CPU node
with 128 MPI ranks via sample_slurm.sh, ~15 min):
  mean slowing time  firm3d 24.915 +/- 0.027 ms,
                     ASCOT5 24.885 +/- 0.029 ms   (+0.1 %, 0.7 sigma)
  max |<E>_firm3d - <E>_ascot5| / E0 = 0.4 %
  (analytic 0.5 ts ln(E0/Emin) = 26.855 ms)

Perlmutter
----------
Both scripts are serial and self-contained; for higher statistics
(N_MARKERS in params.py) submit each side as its own job.  ASCOT5 is
available on Perlmutter via its own build (CC=cc MPI=1 make ascot5_main)
or apptainer image; firm3d builds per the repository CI workflow.
