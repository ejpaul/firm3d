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

So the three-way agreement the examples used to claim does not hold, and the coil symmetry was only one of the reasons.

Note that this comparison changes two things at once, the coordinate system and the source of |B|, so on its own it does not say which is responsible. The field is not: comparing the 40-coil Biot-Savart field against the equilibrium at the same physical points, over flux surfaces from s=0.1 to the LCFS, they agree to 2e-4 rms rising to 6.5e-4 at the boundary, worst case 2.4e-3. The coil ripple is visible and sits where 10 coils per field period put it, around toroidal mode n=40, but its amplitude is only 3e-4 of |B| and the equilibrium's own |B| representation reaches n=72, so it carries that ripple rather than missing it. A 0.03% ripple does not move alpha losses elevenfold.

Nor is it any of the numerical approximations this path makes. Each was refined in turn, everything else held fixed, at 10000 particles throughout:

    baseline                                     88 lost, 8.800e-03 +/- 9.4e-04
    field interpolant   (16,32,8) -> (32,64,16)  86 lost, 8.600e-03 +/- 9.3e-04
    boundary classifier h=0.1     -> h=0.05      89 lost, 8.900e-03 +/- 9.4e-04
    flux label grid     48        -> 96          84 lost, 8.400e-03 +/- 9.2e-04
    ODE tolerance       1e-8      -> 1e-10       95 lost, 9.500e-03 +/- 9.7e-04

Every one of them lands inside the counting error of the baseline. The Cartesian answer is converged: it does not depend on the field representation, the interpolant resolution, the boundary level set, the granularity of the flux label, or the integrator tolerance. It is a well-converged 8.4e-03 to 9.5e-03, and the Boozer answer is a well-converged 8.0e-04.

Both answers being converged, and the numerics being exhausted, what is left is that the two are not quite the same physical problem. The coils do not exactly reproduce the equilibrium, and energetic-particle confinement is sensitive to differences far smaller than they look against |B|.

The right denominator is not |B| but the symmetry-breaking content, since this configuration is quasi-helically symmetric and it is the departure from that symmetry that drives alpha transport. On the LCFS the dominant helical mode is (m, n) = (1, -4) at 0.168 of B_00, and the rms content off that family is 1.7e-2. The coil-vs-equilibrium difference is 6.5e-4 rms, so it is about 4% of the field content that governs the losses, not 0.03% of anything relevant. In the band where the coil ripple sits the comparison is starker still: the equilibrium carries 2.2e-4 rms over 32 <= |n| < 48 and the ripple contributes 3.3e-4, so the coils are the dominant source of structure at those mode numbers rather than a small correction to it.

A loss fraction of a few parts in a thousand is also a tail quantity, set by the small population near the loss boundary in phase space, and tails are not linear in the size of the perturbation. A few percent added to the symmetry-breaking content moving that tail by an order of magnitude is an ordinary outcome, not a suspicious one.

So the two paths are expected to differ, and the earlier three-way agreement was never physically justified: it compared a coil field against an equilibrium field and read the result as a check on the tracer. Use the Boozer examples when the question is the equilibrium's own confinement, and this one when the question is what a particular coil set does to it. Attributing the factor of eleven precisely would need a dedicated study rather than an example, for instance tracing the equilibrium field through the Cartesian path, or adding the ripple to the Boozer field, so that only the field changes.

End-to-end wallclock, interpolant and flux-label construction included, is about 10 minutes using the attached slurm script -- about twice the 5 minutes it took at 1000 particles, so it still fits the debug qos comfortably.
