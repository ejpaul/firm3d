This example computes the trapped particle Poincare map in the beta = 2.5% QA configuration from Landreman & Buller. 

Landreman, Matt, Stefan Buller, and Michael Drevlak. "Optimization of quasi-symmetric stellarators with self-consistent bootstrap current and energetic particle confinement." Physics of Plasmas 29.8 (2022).

The particle is assumed to mirror at (s,theta,zeta)=(0.5,pi/2,0) with the alpha particle birth energy. 
This is the same example as the Trapped_map example, however the Poincare is colored by Digit Accuracy of the WBA. 
Weighted Birkhoff Averaging is added as a setting to the Trapped map and applied to particle momentum to create a numerical 
metric (Digit Accuracy) of chaos, implemented as described in:

N. Duignan and J. D. Meiss. "Distinguishing between regular and chaotic orbits of flows by the weighted birkhoff average." Physical Nonlinear Phenomena. (2023): 449:133749.
