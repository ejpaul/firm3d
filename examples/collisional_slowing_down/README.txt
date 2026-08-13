This example traces 1000 alpha particles in the Wistell-A configuration, including Monte Carlo Coulomb collisions with a 50/50 DT background and its electrons. Particles are initialized proportional to the fusion reactivity profile and traced until they reach the boundary (s=1) or the elapsed time is 2e-1 seconds. This is the collisional counterpart of the fusion_distribution example, and uses the same equilibrium as the two GPU collisional examples.

Over 2e-1 seconds the confined population thermalizes to about 24% of the birth energy, 4.3% of particles are lost, and those carry 1.1% of the birth energy to the wall. The figure plots both cumulative fractions against time: each lost particle takes only the energy it still had when it crossed the boundary, which collisions reduce below the birth energy, so the energy curve falls below the particle curve.

On perlmutter (08.13.26), the wallclock time is about 4 minutes using the attached slurm script.
