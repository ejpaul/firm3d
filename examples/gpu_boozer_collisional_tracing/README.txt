This example traces 1000 alpha particles in the Wistell-A configuration, including Monte Carlo Coulomb collisions with a 50/50 DT background and its electrons. Particles are initialized proportional to the fusion reactivity profile and traced until they reach the boundary (s=1) or the elapsed time is 2e-1 seconds.

The integration time is much longer than in the collisionless gpu_boozer_tracing example because the alpha slowing-down time in this background is of order 0.1 seconds: over the 1e-5 seconds used there, collisions would change the energy by a part in 10^4. Over 2e-1 seconds the confined population thermalizes to about 24% of the birth energy, while 0.2% of particles are lost, carrying 0.1% of the birth energy to the wall.

Each lost particle takes to the wall only the energy it still had when it crossed the boundary, which collisions reduce below the birth energy, so the energy loss falls below the particle loss.

On perlmutter (08.13.26), the wallclock time is about 4 minutes using the attached slurm script.

All three collisional examples trace the same equilibrium with the same
equations and background, on the CPU, on the GPU in Boozer coordinates, and on
the GPU in Cartesian coordinates with the profiles reaching the kick through an
interpolated flux label. The two Boozer examples take |B| from the equilibrium
and agree with each other: 0.2% of particles lost and about 24% of the birth
energy retained by the confined population.

That agreement does not extend to the Cartesian leg, which builds |B| from the
coil set rather than taking it from the equilibrium. Two separate problems hid
behind the claim.

First, the Cartesian example applied the coil symmetries once too often and ran
with |B| about 4x too strong. That has been fixed, and it now asserts the mean
LCFS |B| against the equilibrium before tracing.

Second, and not fixed, the two do not agree even with the correct field. At
1000 particles a 0.2% loss is two events, so the old comparison could not
resolve anything; run both at 10000 particles and the picture is unambiguous:

    Boozer,    equilibrium |B|:  8 lost, 8.000e-04 +/- 2.8e-04, retained 0.2356
    Cartesian, coil |B|:        88 lost, 8.800e-03 +/- 9.4e-04, retained 0.2380

An eleven-fold difference in the loss fraction at 8 events against 88 is not
counting noise. The retained energy still agrees, as expected for a quantity
the collision profiles set rather than the field. Treat the CPU and GPU Boozer
examples as the pair that cross-checks the tracer, and see
gpu_cartesian_collisional_tracing/README.txt for what is known about the
remaining difference.
