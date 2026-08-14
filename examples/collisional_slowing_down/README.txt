This example traces 1000 alpha particles in the Wistell-A configuration, including Monte Carlo Coulomb collisions with a 50/50 DT background and its electrons. Particles are initialized proportional to the fusion reactivity profile and traced until they reach the boundary (s=1) or the elapsed time is 2e-1 seconds. This is the collisional counterpart of the fusion_distribution example, and uses the same equilibrium as the two GPU collisional examples.

Over 2e-1 seconds the confined population thermalizes to about 24% of the birth energy, while 0.2% of particles are lost, carrying 0.1% of the birth energy to the wall. This configuration confines alphas well, so the loss curves in the figure are correspondingly small. The figure plots both cumulative fractions against time: each lost particle takes only the energy it still had when it crossed the boundary, which collisions reduce below the birth energy, so the energy curve falls below the particle curve.

On perlmutter (08.13.26), the wallclock time is about 4 minutes using the attached slurm script.

All three collisional examples trace the same equilibrium with the same
equations and background, and agree: 0.2% of particles lost and about 24% of
the birth energy retained by the confined population, whether traced on the
CPU, on the GPU in Boozer coordinates, or on the GPU in Cartesian
coordinates with the profiles reaching the kick through an interpolated flux
label.

