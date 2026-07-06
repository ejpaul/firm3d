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
