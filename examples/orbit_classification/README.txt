This example traces 5000 alpha particles in the Wistell-A configuration scaled to the size and field strength of ARIES-CS. Particles are initialized proportional to the fusion reactivity profile and traced until they reach the boundary (s=1) or the elapsed time is 1e-2 seconds. 

Particles that are lost to the wall are classified into three trapping regimes based on their bounce dynamics:
- Banana trapped (0): Particles trapped in the lowest-order toroidal magnetic well
- Barely trapped (1): Particles near the trapped-passing boundary with large helical angle excursions
- Ripple trapped (2): Particles trapped in higher-order magnetic ripples

The classification uses helicity M=1, N=0 to analyze the M=1,N=0 helical component of the magnetic field strength. For each lost particle, the code saves trajectory data (particle_i_traj.txt), hit point data (particle_i_hits.txt), and classification diagnostics (particle_i.npz) including the trapping state for each bounce segment, parallel action variable J_||, orbit width parameter gamma_c, and transition statistics.

The main script is fusion_distribution_classification.py. The OrbitClassification class in orbit_classification.py provides the classification algorithm and is documented with detailed descriptions of the physics and output diagnostics.

