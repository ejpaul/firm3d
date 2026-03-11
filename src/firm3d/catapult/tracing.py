__all__ = ['trace_particles_boozer']
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenWavesSuperposition,
)
import firm3dpp
import numpy as np
from firm3d.catapult.utils import boozer_interpolant, boozer_saw_interpolant

def trace_particles_boozer(field, stz_inits, parallel_speeds, tmax, mass, charge, vtotal, tol, ns, ntheta, nzeta, dt=None):
    """ 
    Trace particles in Boozer coordinates using CATAPULT
    field: a magnetic field object representing the field in Boozer coordinates
    stz_inits: initial conditions for particles in (s, theta, zeta) coordinates
    parallel_speeds: initial parallel speeds of the particles
    tmax: maximum time to trace particles
    mass: mass of each particle
    charge: charge of each particle
    vtotal: total velocity of each particle
    tol: tolerance for the ODE solver
    dt: the initial time step size for the solver (optional)
    """
    nparticles = stz_inits.shape[0]
    if field.field_type == "vac":
        srange, trange, zrange, quad_info, maxJ = boozer_interpolant(field, field.nfp, ns, ntheta, nzeta, vacuum=True)
        vacuum = True
        psi0 = field.psi0
        last_time = firm3dpp.boozer_gpu_tracing(
            quad_pts=quad_info,
            srange=srange,
            trange=trange,
            zrange=zrange,
            stz_init=stz_inits,
            m=mass,
            q=charge,
            vtotal=vtotal,
            vtang=parallel_speeds,
            tmax=tmax,
            tol=tol,
            dt_in=-np.ones(nparticles),
            psi0=field.psi0,
            nparticles=nparticles,
            vacuum=True,
        )
        last_time = np.reshape(last_time, (nparticles, 5))
        return last_time