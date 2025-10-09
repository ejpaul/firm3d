---
title: 'FIRM3D: Fast ion reduced models in 3D'
tags:
  - Python
  - plasma physics
  - stellarators 
  - fusion
authors:
  - name: Elizabeth Paul
    orcid: 0000-0002-9355-5595
    # equal-contrib: true
    corresponding: true 
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Alexey Knyazev
    # equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
    orcid: 0000-0002-9355-5595
  - name: Michael Czekanski
    affiliation: 2
    orcid: 0000-0002-9355-5595
  - name: William Fei
    affiliation: 1
    orcid: 0000-0002-9355-5595
affiliations:
 - name: Department of Applied Physics and Applied Mathematics, Columbia University
   index: 1
 - name: Department of Statistics and Data Science, Cornell University
   index: 2
date: 06 October 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Submission notes
- 250-1000 words
- A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.
- A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work.
- A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline.
- Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it.
- Acknowledgement of any financial support.
- Sections from simsopt paper: summary, statement of need, capabilities, acknowledgements, references

# Summary

The dynamics of energetic particle species, born from fusion reactions or plasma heating schemes, are critical for predicting the behavior of magnetic confinement fusion experiments and future fusion reactors. Given that energetic particles are largely collisionless, the orbits of Monte-Carlo samples drawn from a given distribution function can be efficiently integrated in given electromagnetic fields. In addition to the static magneto-hydrodynamic (MHD) equilibrium magnetic fields produced due to the electromagetic coils in a fusion device, MHD waves are excited by and can transport energetic particle populations. 

FIRM3D is a software suite for modeling of energetic particle dynamics in 3D magnetic fields. The core routines are based on SIMSOPT [cite], but have been extended to include additional physics and diagnostics that are not typically required in the optimization context. This standalone framework enables more modular development of FIRM3D with minimal dependencies. 

Components of FIRM3D include: 
- Interfaces with MHD equilibrium and wave stability software.
- CPU and GPU parallelized integration of the guiding center orbit equation.  
- Orbit visualization and transport diagnostics, including Poincare\'{e} maps and weighted Birkhoff averaging. 


<!-- The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration). -->

# Statement of need

Given recent advances by the stellarator optimization community [cite], stellarator equilibria have now been identified which satisfy many physics and engineering constraints for a fusion reactor. One of the critical features of a stellarator equilibrium is the ability to confine the guiding center trajectories of energetic particle species. Through concepts such as quasisymmetry, the presence of a hidden symmetry of the field strength which provides integrability of guiding center motion, the magnetic fields of stellarators can now be designed to have excellent energetic particle confinement [cite].

However, there are likely to be perturbing electromagnetic fields that can transport energetic particles, such as magneto-hydrodynamic (MHD) waves. The class of MHD waves of primary concern for interaction with EP species are Alfv\'{e}n eigenmodes (AEs). AEs are driven unstable by free energy in the EP distribution function, and they can resonantly transport EPs. Alfv´enic ac-
tivity is considered the major limitation to alpha confinement in a burning tokamak plasma [@2014Gorelenkov]. The interaction of Alfv´en eigenmodes (AEs) with energetic particles has been shown to drive substantial flattening of the fast-ion profile in tokamak experiments [@2008Heidbrink]. Alv´enic activity has also been
observed on several stellarator configurations, including HSX [@2009Deng], CHS [@2002Takechi], LHD [@2011Toi], W7-AS [@1994Weller], TJ-II [@2014Melnikov], W7-X [@2020Rahbarnia], and Heliotron-J [@2007Yamamoto]. Given the recent growth of the private fusion industry, several start-up companies pursuing the stellarator path to fusion are interested in assessing the stability of EP-driven waves and their impact on EP transport. The development of FIRM3D is, therefore, timely. 

FIRM3D grew out of the guiding center integration routines in SIMSOPT, but has been extended to include additional physics and diagnostics specifically needed for energetic particle studies. The standalone framework enables more focused development of energetic particle physics capabilities with minimal dependencies, making it accessible to the broader stellarator and plasma physics community. 



# Acknowledgements

We acknowledge the SIMSOPT development team for providing the foundational guiding center integration routines. This work was supported by [funding sources to be added].

# References

