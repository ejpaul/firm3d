Reactor-scale collisional alpha slowing-down (Wistell-A / ARIES-CS scale)
=========================================================================

Extends examples/fusion_distribution with Monte Carlo Coulomb collisions:
1024 fusion-born alphas (birth positions ~ fusion reactivity, isotropic
pitch, 3.52 MeV) are traced for 150 ms in the Wistell-A configuration
scaled to ARIES-CS size and field strength, against a D-T-electron
background with Bader et al. (Nucl. Fusion 61, 116060 (2021)) profile
shapes at reactor scale (core n_D = n_T = 1e20 m^-3, T0 = 11.5 keV).
The identical birth ensemble is also traced collisionlessly to separate
collisional transport from prompt orbit losses.

Outputs: loss-fraction comparison (with/without collisions), the
confined-alpha energy spectrum vs the classical slowing-down
distribution, the lost-alpha energy histogram, and an npz with final
states.

Run on Perlmutter:   sbatch sample_slurm.sh     (~1-2 h on one CPU node)
Local smoke test:    python fusion_distribution_collisional.py --smoke

Reference results (2026-07-06, Perlmutter, 1 CPU node x 128 ranks,
13 min in the debug queue; commit 3499735e):
  particle losses at 150 ms:  19-21 % with collisions
                              1.2 %  collisionless (prompt orbit losses)
  energy loss fraction (wall loading): 7.6 % of total birth energy
  (particle vs energy loss differ because late losses are increasingly
  thermalized ash)
  lost-alpha energies: bimodal -- prompt losses at 3.3-3.5 MeV plus a
  slowed population peaking near 0.5 MeV
  time-integrated slowing-down spectrum vs the classical distribution
  (E_c = 266 keV): mean |relative residual| 4.7 % over 0.9-3.3 MeV
  (below ~0.8 MeV the 150 ms window has not yet fully populated the
  steady-state equivalent spectrum)

Literature benchmark: Bader et al. (Nucl. Fusion 61, 116060 (2021),
arXiv:2106.00716) ran the same Wistell-A/ARIES-CS-scale configuration
with the ANTS code (collisional guiding-center, reactivity-profile
source) and report a collisional alpha ENERGY loss fraction of ~8 %
(their figure 3, plateau near 100 ms), with Wistell-A performing on par
with W7-X.  The firm3d result above (7.6 % at 150 ms) is consistent,
with caveats: Bader et al. use n_e0 = 4.8e20 m^-3 with Z_eff = 1.13
(this example: n_e0 = 2e20, Z_eff = 1) and follow particles to full
thermalization; the field representations (ANTS real-space vs firm3d
Boozer spline) and sourcing details also differ.  This is a literature
consistency check of the full pipeline, not a marker-level cross-code
validation -- for that see examples/collision_validation_ascot5 (a
uniform-plasma case, decoupled from geometry) and the direct
same-geometry ASCOT5 comparison described below.


Direct marker-level comparison against ASCOT5 in the SAME geometry
====================================================================

run_ascot5_wistell.py builds an ASCOT5 input for this same Wistell-A
equilibrium directly from the VMEC wout file (a5py's vmec_field ->
a hand-built, padded B_STS real-space spline; see the script's module
docstring for two required ASCOT5 source patches -- a Wiener-array
capacity limit and a near-axis floating-point edge case in the generic
rho evaluator -- neither of which can be committed here since they
live in the separate, non-vendored ascot5 repo).  Same birth
distribution shape, same D-T-electron profiles, same 3.52 MeV isotropic
alphas, same tmax.  compare_wistell.py (collisional) and
compare_wistell_collisionless.py (--collisionless companion) produce
the head-to-head comparison.  Perlmutter: sample_slurm_ascot5.sh.

Reference results (2026-07-07/08, Perlmutter, N = 1024, ASCOT5 field
import at nphi=180/nr=nz=120, 0 aborted markers after the source
patches; commits d6fa3353/dc2b470f):

  COLLISIONAL
                          firm3d    ASCOT5
    particle loss frac.    19.4%     22.6%   (1.7 sigma -- consistent)
    energy loss frac.       7.6%     13.4%   (large discrepancy)
  Loss-fraction-vs-time curves have the same shape/timescale in both
  codes.  The energy-loss gap traces to ASCOT5's lost population
  skewing much more toward near-birth-energy ("prompt") losses than
  firm3d's, which skews toward thermalized, low-energy losses.

  COLLISIONLESS (orbit-following only, isolates field/geometry from
  the collision operator)
                          firm3d    ASCOT5
    final loss fraction    1.2%     27.2%   (18.2 sigma)
  The two codes' loss curves agree reasonably well at EARLY times
  (both ~0.3-1% within the first ms, consistent prompt-loss physics).
  The divergence is entirely a LATE-TIME, slow "stochastic" loss
  channel with zero collisions: firm3d's curve goes flat at ~1.2% for
  the rest of the 150 ms window; ASCOT5's keeps climbing the whole
  time, reaching 27%.

  This is a bigger and more fundamental discrepancy than the
  collisional case's, and it is NOT simply "the codes use different
  field models": both are supposed to represent the SAME VMEC
  equilibrium (firm3d via a Boozer-coordinate spline of the
  booz_xform-transformed field, ASCOT5 via a real-space (R,phi,Z)
  spline built directly from the VMEC file), so an 18-sigma gap in a
  real physical effect (secular guiding-center orbit diffusion from
  broken exact quasisymmetry) more likely reflects a numerical/
  methodological difference than two different physical models.
  Leading candidates, NONE confirmed:
    - each spline's resolution of the fine, non-quasisymmetric ripple
      harmonics that actually drive this slow diffusion (firm3d's
      Boozer grid vs ASCOT5's real-space grid don't necessarily
      resolve that ripple to the same fidelity);
    - boundary-handling differences (firm3d's hard s=1 stop vs
      ASCOT5's rho > 1.02 in the padded field) for orbits that graze
      the edge repeatedly over many transits;
    - a genuine difference in the guiding-center equations/order each
      code integrates.
  Natural next diagnostics (not yet done): a field-resolution
  sensitivity scan on one or both codes, or characterizing which
  orbits ASCOT5 loses late (radius, pitch, proximity to the trapped-
  passing boundary) to check for the ripple-driven-diffusion
  signature.  As of 2026-07-08 the user is looking for a better-suited
  comparison case with ASCOT5 rather than chasing this one further for
  now.
