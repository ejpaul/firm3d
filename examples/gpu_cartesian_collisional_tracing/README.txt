This example traces 1000 alpha particles in the Wistell-A configuration, including Monte Carlo Coulomb collisions with a 50/50 DT background and its electrons. Particles are initialized proportional to the fusion reactivity profile and traced until they leave the plasma boundary or the elapsed time is 2e-1 seconds. Over that window the confined population thermalizes to about 24% of the birth energy, while 0.2% of particles are lost, carrying 0.1% of the birth energy to the wall.

The thermal profiles are functions of the normalized flux s, which a Cartesian state does not carry, so this entry point takes a flux_label callable mapping cylindrical points (r, phi, z) to s. Its values are interpolated alongside the magnetic field and evaluated at each particle after every accepted step. Here the label is built by mapping a dense Boozer grid of the matching equilibrium forward to cylindrical coordinates and answering queries with the nearest sample; any callable finite over the interpolation grid will do, and values above 1 outside the plasma are clamped by the profile lookup.

The coil field and the equilibrium supplying the label must describe the same device. The label here is built from wout_aten_rescaled.nc, the file the coil set and the boundary classifier come from.

On perlmutter (08.13.26), the wallclock time is about 5 minutes using the attached slurm script.

All three collisional examples trace the same equilibrium with the same
equations and background, and agree: 0.2% of particles lost and about 24% of
the birth energy retained by the confined population, whether traced on the
CPU, on the GPU in Boozer coordinates, or on the GPU in Cartesian
coordinates with the profiles reaching the kick through an interpolated flux
label.

