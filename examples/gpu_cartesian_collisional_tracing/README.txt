This example traces 10000 alpha particles in the Wistell-A configuration, including Monte Carlo Coulomb collisions with a 50/50 DT background and its electrons. Particles are initialized proportional to the fusion reactivity profile and traced until they leave the plasma boundary or the elapsed time is 2e-1 seconds.

The thermal profiles are functions of the normalized flux s, which a Cartesian state does not carry, so this entry point takes a flux_label callable mapping cylindrical points (r, phi, z) to s. Its values are interpolated alongside the magnetic field and evaluated at each particle after every accepted step. Here the label is built by mapping a dense Boozer grid of the matching equilibrium forward to cylindrical coordinates and answering queries with the nearest sample; any callable finite over the interpolation grid will do, and values above 1 outside the plasma are clamped by the profile lookup.

The coil field and the equilibrium supplying the label must describe the same device. The label here is built from wout_aten_rescaled.nc, the file the coil set and the boundary classifier come from.

The coil field is built from coils.curves_22_7_21, which holds 20 coils: the stellarator-symmetric half of a full-torus 40-coil set. Only stellsym is applied, so coils_via_symmetries is called with nfp=1 and returns 40 coils. Passing surf.nfp instead replicates an already complete coil set nfp times and makes |B| about nfp times too strong. Nothing in the output announces that: a stronger field shrinks the gyroradius and improves confinement, so the run simply reports fewer losses. The example therefore asserts the mean |B| on the LCFS against the equilibrium before tracing, printing

    coil field check: mean |B| on LCFS = 5.9410 T (equilibrium 5.9410 T, ratio 1.0000)

and aborting if the ratio leaves 3% of unity.

The particle count was raised from 1000 to 10000 because the loss fraction is the quantity that regression-tests the field strength, and at 1000 particles a loss of a few parts in 1000 is a couple of events. Poisson noise on two events is large enough to hide a factor-of-several error in |B| entirely, which is how the mis-scaled field survived earlier validation: the agreement across the CPU Boozer, GPU Boozer, and GPU Cartesian collisional examples rested on that handful of events, and on the retained-energy figure, which is set by the collision profiles and barely depends on the field strength. The loss fraction is now printed with its counting uncertainty.

The reference numbers below have not yet been regenerated on the corrected coil field. The previously quoted figures -- 0.2% of particles lost carrying 0.1% of the birth energy -- were produced with the field about 4x too strong and should not be used. The thermalization figure, about 24% of the birth energy retained by the confined population, is set by the collision profiles and is expected to survive, but it has not been re-measured either. Rerun this example and the two Boozer collisional examples before quoting any three-way agreement.

On perlmutter (08.13.26), the wallclock time was about 5 minutes at 1000 particles. At 10000 particles the run no longer fits the 30 minute ceiling on the debug qos, so the attached slurm script asks for the regular qos and two hours.
