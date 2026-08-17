This example traces 10000 particles in the Wistell-A configuration scaled to the size and field strength of ARIES-CS. Particles are launched uniformly over the s=0.3 flux surface with the pitch v_par/v drawn uniformly in [-1, 1], and traced until they reach the boundary (s=1) or the elapsed time is 1e-5 seconds.

The coil field is built from coils.curves_22_7_21, which holds 20 coils: the stellarator-symmetric half of a full-torus 40-coil set. Only stellsym is applied, so coils_via_symmetries is called with nfp=1 and returns 40 coils. Passing surf.nfp here instead replicates an already complete coil set nfp times and makes |B| about nfp times too strong -- a field that looks perfectly well formed but shrinks the gyroradius and quietly improves confinement. The example asserts the mean |B| on the LCFS against the equilibrium before tracing, and aborts if the ratio leaves 3% of unity.

Reference output on perlmutter (08.17.26), 10000 particles on one A100:

    coil field check: mean |B| on LCFS = 5.9410 T (equilibrium 5.9410 T, ratio 1.0000)
    Number of particles= 10000
    Particles lost: 0 (< 3.0e-04 at 95% CL)

Note that 1e-5 seconds is far too short for prompt losses from s=0.3 in this configuration: the loss count is zero, so it cannot by itself detect a mis-scaled field, and the |B| assertion is what pins the field down here. The longer collisional examples are where the loss fraction carries information. The particle count is 10000 rather than 1000 only to tighten the zero-count bound from 3e-3 to 3e-4; it costs almost nothing, because 1000 particles is 125 blocks of 8 against 108 SMs and leaves the GPU largely idle.

End-to-end wallclock, interpolant construction included, is about 40 seconds using the attached slurm script -- it was about 30 seconds at 1000 particles, so ten times the particles cost a third more time, for the reason just given.
