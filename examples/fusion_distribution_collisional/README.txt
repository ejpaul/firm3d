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
