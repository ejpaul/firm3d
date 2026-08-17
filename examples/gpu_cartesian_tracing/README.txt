This example traces 10000 particles in the Wistell-A configuration scaled to the size and field strength of ARIES-CS. Particles are launched uniformly over the s=0.3 flux surface with the pitch v_par/v drawn uniformly in [-1, 1], and traced until they reach the boundary (s=1) or the elapsed time is 1e-5 seconds.

The coil field is built from coils.curves_22_7_21, which holds 20 coils: the stellarator-symmetric half of a full-torus 40-coil set. Only stellsym is applied, so coils_via_symmetries is called with nfp=1 and returns 40 coils. Passing surf.nfp here instead replicates an already complete coil set nfp times and makes |B| about nfp times too strong -- a field that looks perfectly well formed but shrinks the gyroradius and quietly improves confinement. The example asserts the mean |B| on the LCFS against the equilibrium before tracing so that this cannot pass unnoticed; it prints

    coil field check: mean |B| on LCFS = 5.9410 T (equilibrium 5.9410 T, ratio 1.0000)

and aborts if the ratio leaves 3% of unity.

The reported loss fraction carries its counting uncertainty, since that is the figure a mis-scaled field moves and it means nothing until it is many events.

On perlmutter (04.20.26), the wallclock time was about 30 seconds at 1000 particles using the attached slurm script; the particle count has since been raised to 10000, so expect roughly ten times that. The loss fraction has not been regenerated since the coil field was corrected.
