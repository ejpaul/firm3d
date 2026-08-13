This example traces 1000 alpha particles in the Wistell-A configuration scaled to the size and field strength of ARIES-CS, including Monte Carlo Coulomb collisions with a 50/50 DT background and its electrons. Particles are initialized on the s=0.3 surface and traced until they leave the plasma boundary or the elapsed time is 1e-2 seconds. Over that window the confined population retains about 93% of the birth energy.

The thermal profiles are functions of the normalized flux s, which a Cartesian state does not carry, so this entry point takes a flux_label callable mapping cylindrical points (r, phi, z) to s. Its values are interpolated alongside the magnetic field and evaluated at each particle after every accepted step. Here the label is built by mapping a dense Boozer grid of the matching equilibrium forward to cylindrical coordinates and answering queries with the nearest sample; any callable finite over the interpolation grid will do, and values above 1 outside the plasma are clamped by the profile lookup.

Note that the coil field and the equilibrium supplying the label must describe the same device. The boozmn files in examples/inputs are differently rescaled versions of this configuration, so the label here is built from wout_aten_rescaled.nc, the same file the coil set and the boundary classifier come from.

The output has seven columns, [t, x, y, z, v_par, v, dt], one more than the collisionless tracer: collisions change the total speed, so it is reported rather than inferred from the launch energy.

On perlmutter (08.13.26), the wallclock time is about 60 seconds using the attached slurm script, of which roughly 40 seconds is the Boozer transformation and flux-label construction.
