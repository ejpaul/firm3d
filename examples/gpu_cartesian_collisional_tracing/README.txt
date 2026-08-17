This example traces 10000 alpha particles in the Wistell-A configuration, including Monte Carlo Coulomb collisions with a 50/50 DT background and its electrons. Particles are initialized proportional to the fusion reactivity profile and traced until they leave the plasma boundary or the elapsed time is 2e-1 seconds.

The thermal profiles are functions of the normalized flux s, which a Cartesian state does not carry, so this entry point takes a flux_label callable mapping cylindrical points (r, phi, z) to s. Its values are interpolated alongside the magnetic field and evaluated at each particle after every accepted step. Here the label is built by mapping a dense Boozer grid of the matching equilibrium forward to cylindrical coordinates and answering queries with the nearest sample; any callable finite over the interpolation grid will do, and values above 1 outside the plasma are clamped by the profile lookup.

The coil field and the equilibrium supplying the label must describe the same device. The label here is built from wout_aten_rescaled.nc, the file the coil set and the boundary classifier come from.

The coil field is built from coils.curves_22_7_21, which holds 20 coils: the stellarator-symmetric half of a full-torus 40-coil set. Only stellsym is applied, so coils_via_symmetries is called with nfp=1 and returns 40 coils. Passing surf.nfp instead replicates an already complete coil set nfp times and makes |B| about nfp times too strong. Nothing in the output announces that: a stronger field shrinks the gyroradius and improves confinement, so the run simply reports fewer losses. The example therefore asserts the mean |B| on the LCFS against the equilibrium before tracing, and aborts if the ratio leaves 3% of unity.

Reference output on perlmutter (08.17.26), 10000 particles on one A100:

    coil field check: mean |B| on LCFS = 5.9410 T (equilibrium 5.9410 T, ratio 1.0000)
    Number of particles= 10000
    Particles lost: 88 (8.800e-03 +/- 9.4e-04)
    Energy loss fraction: 4.962e-03
    Mean energy fraction of confined: 0.2380

The particle count is 10000 rather than 1000 because the loss fraction is the quantity that regression-tests the field strength, and at 1000 particles it was a couple of events. That is how the mis-scaled field survived earlier validation: with |B| about 4x too strong this example reported 0.2% lost carrying 0.1% of the birth energy, and Poisson noise on two events is far too large to distinguish that from the correct answer. Correcting the field moves the loss fraction to 0.88% +/- 0.09% and the energy loss to 0.50%, both about a factor of four to five up, in the direction a weaker and therefore correct field predicts. The retained-energy figure barely moves -- 0.2380 against the earlier 0.24 -- because it is set by the collision profiles rather than by |B|, which is the other reason the earlier agreement looked convincing.

The loss fraction here does not agree with the Boozer collisional examples, and correcting the coil field did not make it agree. Running the GPU Boozer example at the same 10000 particles, so that both sides have real statistics, gives

    Boozer,    equilibrium |B|:  8 lost, 8.000e-04 +/- 2.8e-04, energy 3.860e-04, retained 0.2356
    Cartesian, coil |B|:        88 lost, 8.800e-03 +/- 9.4e-04, energy 4.962e-03, retained 0.2380

Same equilibrium, tmax, tolerance and background profiles. The loss fractions differ by about a factor of eleven, at 8 events against 88, so this is not counting noise. The retained energy agrees to a percent, as expected for a quantity the collision profiles set.

So the three-way agreement the examples used to claim does not hold, and the coil symmetry was only one of the reasons. What remains is a genuine difference between tracing the vacuum field of a discrete 40-coil set and tracing the vacuum-enforced Boozer representation of the same equilibrium. Coil ripple is the obvious candidate, since the Boozer representation carries only mnmax=128 modes and ripple generally costs alpha confinement, but the coarse Cartesian interpolant (n=16 over the whole LCFS bounding box), the SurfaceClassifier level set with h=0.1 as the boundary, and the nearest-neighbour flux label are all also cruder here than their Boozer counterparts. Which of these dominates has not been established.

End-to-end wallclock, interpolant and flux-label construction included, is about 10 minutes using the attached slurm script -- about twice the 5 minutes it took at 1000 particles, so it still fits the debug qos comfortably.
