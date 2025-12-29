from warnings import warn

import numpy as np
from scipy import integrate
from os.path import exists

from .._core.util import parallel_loop_bounds
from ..field.boozermagneticfield import (
    ShearAlfvenHarmonic,
    ShearAlfvenWave,
    ShearAlfvenWavesSuperposition,
)
from ..field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    MinToroidalFluxStoppingCriterion,
    trace_particles_boozer,
    trace_particles_boozer_perturbed,
)
from ..field.tracing_helpers import (
    initialize_position_uniform_surf, 
    initialize_position_uniform_vol, 
    initialize_velocity_uniform,
)

__all__ = [
    "compute_loss_fraction",
    "compute_trajectory_cylindrical",
    "PassingPoincare",
    "PassingPerturbedPoincare",
    "trajectory_to_vtk",
]


def compute_loss_fraction(res_tys, tmin=1e-7, tmax=1e-2, ntime=1000):
    r"""
    Compute the fraction of particles lost as a function of time.

    Args:
        res_tys : List of particle trajectories, where each trajectory is a 2D
                  array with shape (nsteps, 5) containing time and coordinates
                  (t, s, theta, zeta, vpar).
        tmin : Minimum time to consider for loss fraction (default: 1e-7)
        tmax : Maximum time to consider for loss fraction (default: 1e-2)
        ntime : Number of time points to evaluate the loss fraction (default: 1000)
    Returns:
        times : A numpy array of shape (ntime,) containing the time points at
                which the loss fraction is evaluated.
        loss_frac : A numpy array of shape (ntime,) containing the fraction of
                    particles lost at each time point.
    """
    nparticles = len(res_tys)

    timelost = np.zeros((nparticles,))
    for ip in range(nparticles):
        timelost[ip] = res_tys[ip][-1, 0]

    times = np.logspace(np.log10(tmin), np.log10(tmax), ntime)

    loss_frac = np.zeros_like(times)
    for it in range(ntime):
        loss_frac[it] = np.count_nonzero(timelost < times[it] - 1e-15) / nparticles

    return times, loss_frac


def compute_trajectory_cylindrical(res_ty, field):
    r"""
    Compute the cylindrical coordinates (R, Z, phi) in a given
    BoozerMagneticField for each particle trajectory.

    Args:
        res_ty : A 2D numpy array of shape (nsteps, 5) containing the
                 trajectory of a single particle in Boozer coordinates
                 (s, theta, zeta, vpar). Tracing should be performed with
                 forget_exact_path=False to save the trajectory information.
        field : The :class:`BoozerMagneticField` instance used to set the
                points for the field.

    Returns:
        R_traj : A numpy array with shape (nsteps,) containing the radial
                 coordinate R for the particle trajectory.
        Z_traj : A numpy array with shape (nsteps,) containing the vertical
                 coordinate Z for the particle trajectory.
        phi_traj : A numpy array with shape (nsteps,) containing the
                   azimuthal angle phi for the particle trajectory.
    """
    nsteps = len(res_ty[:, 0])
    points = np.zeros((nsteps, 3))
    points[:, 0] = res_ty[:, 1]
    points[:, 1] = res_ty[:, 2]
    points[:, 2] = res_ty[:, 3]
    field.set_points(points)

    R_traj = field.R()[:, 0]
    Z_traj = field.Z()[:, 0]
    nu = field.nu()[:, 0]
    phi_traj = res_ty[:, 3] - nu

    return R_traj, phi_traj, Z_traj


class PassingPoincare:
    """
    Class to compute and store passing Poincare maps and related quantities
    for a given BoozerMagneticField.
    """

    def __init__(
        self,
        field,
        lam,
        sign_vpar,
        mass,
        charge,
        Ekin,
        ns_poinc=None,
        ntheta_poinc=None,
        s_init=None,
        thetas_init=None,
        Nmaps=500,
        comm=None,
        tmax=1e-2,
        solver_options=None,
    ):
        """
        Initialize and compute the passing Poincare map, evaluated by
        integrating the guiding center equations until the trajectory returns
        to the zeta = 0 plane.
        We assume that the particle is passing, so the parallel velocity does
        not change sign.

        Args:
            field : The :class:`BoozerMagneticField` instance.
            mass : Particle mass.
            charge : Particle charge.
            Ekin : Particle total energy.
            s_init : List of initial s coordinates for the Poincare map.
                     (default: None, ns_poinc is used instead)
            thetas_init : List of initial theta coordinates for the Poincare
                          map. (default: None, ntheta_poinc is used instead)
            ns_poinc : Number of initial conditions in s for Poincare plot
                       (default: 120).
            ntheta_poinc : Number of initial conditions in theta for Poincare
                           plot (default: 2).
            Nmaps : Number of Poincare return maps to compute for each initial
                    condition (default: 500).
            comm : MPI communicator for parallel execution (default: None).
            tmax : Maximum integration time for each segment of the Poincare
                   map (default: 1e-2 s).
            solver_options : Dictionary of options to pass to the ODE solver
                             (default: {}).
        """
        if solver_options is None:
            solver_options = {}
        if sign_vpar not in [-1, 1]:
            raise ValueError("sign_vpar should be either -1 or +1")

        self.field = field
        self.lam = lam
        self.sign_vpar = sign_vpar
        self.mass = mass
        self.charge = charge
        self.Ekin = Ekin
        if s_init is not None and thetas_init is not None:
            s = s_init
            thetas = thetas_init
        else:
            if ns_poinc is None:
                ns_poinc = 120
            if ntheta_poinc is None:
                ntheta_poinc = 2
            s = np.linspace(0, 1, ns_poinc + 1, endpoint=False)[1::]
            thetas = np.linspace(0, 2 * np.pi, ntheta_poinc)
        s, thetas = np.meshgrid(s, thetas)
        s_flat = s.flatten()
        thetas_flat = thetas.flatten()
        self.Nmaps = Nmaps
        self.comm = comm
        self.tmax = tmax
        self.solver_options = solver_options
        self.vpars_init, self.s_init, self.thetas_init = self.initialize_passing_map(s_flat,thetas_flat)
        (
            self.s_all,
            self.thetas_all,
            self.vpars_all,
            self.t_all,
        ) = self.compute_passing_map()

    def initialize_passing_map(self,s_flat,thetas_flat):
        r"""
        Given a :class:`BoozerMagneticField` instance, this function generates
        initial positions for the passing Poincare return map. Particles are
        initialized on the zeta = 0 plane such that the parallel velocity is
        consistent with the prescribed total energy and pitch-angle
        variable, :math:`\lambda = v_\perp^2/(v^2 B)`.

        Returns:
            s_init : List of initial s coordinates for the Poincare map.
            thetas_init : List of initial theta coordinates for the Poincare map.
            vpars_init : List of initial parallel velocities for the Poincare map.
        """
        vtotal = np.sqrt(
            2 * self.Ekin / self.mass
        )  # Total velocity from kinetic energy

        def vpar_func(s, theta):
            point = np.zeros((1, 3))
            point[0, 0] = s
            point[0, 1] = theta
            self.field.set_points(point)
            modB = self.field.modB()[0, 0]
            # Skip any trapped particles
            if 1 - self.lam * modB < 0:
                return None
            else:
                return self.sign_vpar * vtotal * np.sqrt(1 - self.lam * modB)

        first, last = parallel_loop_bounds(self.comm, len(s_flat))
        # For each point, find value of vpar such that lambda = vperp^2/(v^2 B)
        vpars_init = []
        s_init = []
        thetas_init = []
        for i in range(first, last):
            vpar = vpar_func(s_flat[i], thetas_flat[i])
            if vpar is not None:
                vpars_init.append(vpar)
                s_init.append(s_flat[i])
                thetas_init.append(thetas_flat[i])

        if self.comm is not None:
            vpars_init = [i for o in self.comm.allgather(vpars_init) for i in o]
            s_init = [i for o in self.comm.allgather(s_init) for i in o]
            thetas_init = [i for o in self.comm.allgather(thetas_init) for i in o]

        return vpars_init, s_init, thetas_init

    def passing_map(self, point):
        r"""
        Given the coordinates (s,theta,vpar) at zeta = 0, integrates the guiding
        center equations until the trajectory returns to the zeta = 0 plane.
        We assume that the particle is passing, so if vpar crosses through 0
        along the trajectory, a RuntimeError is raised. A RuntimeError is also
        raised if the particle leaves the s = 1 surface. The coordinates (s,theta,vpar)
        for the return map are returned.

        Args:
            point : A numpy array of shape (3,) containing the initial coordinates
                (s,theta,vpar).

        Returns:
            point : A numpy array of shape (3,) containing the coordinates
                (s,theta,vpar) when the trajectory returns to the zeta = 0 plane.
            time : The time taken to return to the zeta = 0 plane.
        """

        points = np.zeros((1, 3))
        points[:, 0] = point[0]
        points[:, 1] = point[1]
        points[:, 2] = 0
        # Set solver options needed for passing map
        res_tys, res_hits = trace_particles_boozer(
            self.field,
            points,
            [point[2]],
            tmax=self.tmax,
            mass=self.mass,
            charge=self.charge,
            Ekin=self.Ekin,
            vpars=[0.0],
            phases=[0.0],
            n_zetas=[1.0],
            m_thetas=[0.0],
            omegas=[0.0],
            stopping_criteria=[
                MaxToroidalFluxStoppingCriterion(1.0),
            ],
            forget_exact_path=False,
            vpars_stop=True,
            phases_stop=True,
            **self.solver_options,
        )
        if len(res_hits[0]) == 0:
            raise RuntimeError("No stopping criterion reached in passing_map.")

        res_hit = res_hits[0][0, :]  # Only check the first hit or stopping criterion

        if res_hit[1] == 0:  # Check that the zetas=[0] plane was hit
            point[0] = res_hit[2]
            point[1] = res_hit[3]
            point[2] = res_hit[5]
            time = res_hit[0]
            return point, time
        else:
            raise RuntimeError("Alternative stopping criterion reached in passing_map.")

    def compute_passing_map(self):
        r"""
        Evaluates the passing Poincare return map for the initialized particle
        positions.
        """
        Ntrj = len(self.s_init)

        s_all = []
        thetas_all = []
        vpars_all = []
        t_all = []
        if Ntrj == 1:
            first = 0
            last = 1
        else: 
            first, last = parallel_loop_bounds(self.comm, Ntrj)
        for itrj in range(first, last):
            tr = [self.s_init[itrj], self.thetas_init[itrj], self.vpars_init[itrj]]
            s_traj = [tr[0]]
            thetas_traj = [tr[1]]
            vpars_traj = [tr[2]]
            t_traj = [0]
            for _jj in range(self.Nmaps):
                try:
                    tr, time = self.passing_map(tr)
                    s_traj.append(tr[0])
                    thetas_traj.append(tr[1])
                    vpars_traj.append(tr[2])
                    t_traj.append(time)
                except RuntimeError:
                    break
            s_all.append(s_traj)
            thetas_all.append(thetas_traj)
            vpars_all.append(vpars_traj)
            t_all.append(t_traj)

        if self.comm is not None:
            s_all = [i for o in self.comm.allgather(s_all) for i in o]
            thetas_all = [i for o in self.comm.allgather(thetas_all) for i in o]
            vpars_all = [i for o in self.comm.allgather(vpars_all) for i in o]
            t_all = [i for o in self.comm.allgather(t_all) for i in o]

        return s_all, thetas_all, vpars_all, t_all

    def compute_frequencies(self):
        """
        Compute the passing particle poloidal and toroidal transit frequencies
        and mean radial position.
        The frequency is only computed if the trajectory has completed at least
        one full Poincare return map before the trajectory is terminated due to
        a time limit or stopping criterion. Since the frequency will depend on
        field-line label and trapping well in a general 3D field, an average is
        taken over all trajectories initialized on the same flux surface and
        all Poincare return maps along each trajectory.
        Since the frequency profile will flatten in a phase-space island, it is
        sometimes easier to interpret the frequency profile in a field with
        enforced quasisymmetry (i.e., initialize BoozerRadialInterpolant with N
        prescribed).

        Returns:
            omega_theta : List of poloidal transit frequencies.
            omega_zeta : List of toroidal transit frequencies.
            init_s : List of initial s values for each trajectory.
        """
        if "axis" in self.solver_options and self.solver_options["axis"] != 0:
            raise ValueError(
                'ODE solver must integrate with solver_options["axis"]=0 to '
                "compute passing frequencies."
            )

        self.field.set_points(np.array([[1], [0], [0]]).T)
        sign_G = np.sign(self.field.G()[0])

        omega_theta = []
        omega_zeta = []
        init_s = []
        for s_traj, theta_traj, _vpar_traj, t_traj in zip(
            self.s_all, self.thetas_all, self.vpars_all, self.t_all
        ):
            if (
                len(s_traj) < 2
            ):  # Need at least one full Poincare return maps to compute frequency
                continue
            delta_theta = np.array(theta_traj[1:]) - np.array(theta_traj[0:-1])
            delta_s = np.array(s_traj[1:]) - np.array(s_traj[0:-1])
            delta_t = t_traj[1::]
            delta_zeta = 2 * np.pi * self.sign_vpar * sign_G

            # Average over wells along one field line
            freq_theta = np.mean(delta_theta) / np.mean(delta_t)
            freq_zeta = delta_zeta / np.mean(delta_t)
            freq_s = delta_s / np.mean(delta_t)

            omega_theta.append(freq_theta)
            omega_zeta.append(freq_zeta)
            init_s.append(np.mean(s_traj))

        omega_theta = np.array(omega_theta)
        omega_zeta = np.array(omega_zeta)
        init_s = np.array(init_s)

        s_prof = np.unique(init_s)
        omega_theta_prof = np.zeros((len(s_prof),))
        omega_zeta_prof = np.zeros((len(s_prof),))

        # Average over field-line label
        for i, s in enumerate(s_prof):
            omega_theta_prof[i] = np.mean(omega_theta[np.where(init_s == s)])
            omega_zeta_prof[i] = np.mean(omega_zeta[np.where(init_s == s)])

        return omega_theta_prof, omega_zeta_prof, s_prof

    def compute_ds_dangle(self, helicity_M, helicity_N):
        if "axis" in self.solver_options and self.solver_options["axis"] != 0:
            raise ValueError(
                'ODE solver must integrate with solver_options["axis"]=0 to '
                "compute passing frequencies."
            )

        self.field.set_points(np.array([[1], [0], [0]]).T)
        sign_G = np.sign(self.field.G()[0])

        omega_theta = []
        omega_zeta = []
        init_s = []
        for s_traj, theta_traj, _vpar_traj, t_traj in zip(
            self.s_all, self.thetas_all, self.vpars_all, self.t_all
        ):
            if (
                len(s_traj) < 2
            ):  # Need at least one full Poincare return maps to compute frequency
                continue
            delta_theta = np.array(theta_traj[1:]) - np.array(theta_traj[0:-1])
            delta_s = np.array(s_traj[1:]) - np.array(s_traj[0:-1])
            delta_t = t_traj[1::]
            delta_zeta = 2 * np.pi * self.sign_vpar * sign_G
            delta_chi = helicity_M * delta_theta + helicity_N * delta_zeta


        omega_theta = np.array(omega_theta)
        omega_zeta = np.array(omega_zeta)
        init_s = np.array(init_s)

        s_prof = np.unique(init_s)
        omega_theta_prof = np.zeros((len(s_prof),))
        omega_zeta_prof = np.zeros((len(s_prof),))

        # Average over field-line label
        for i, s in enumerate(s_prof):
            omega_theta_prof[i] = np.mean(omega_theta[np.where(init_s == s)])
            omega_zeta_prof[i] = np.mean(omega_zeta[np.where(init_s == s)])

        return omega_theta_prof, omega_zeta_prof, s_prof

    def get_poincare_data(self):
        """
        Return the Poincare map data.

        Returns:
            s_all, thetas_all, vpars_all, t_all : Lists of trajectory data.
        """
        return self.s_all, self.thetas_all, self.vpars_all, self.t_all

    def plot_poincare(self, ax=None, filename="passing_poincare.pdf"):
        r"""
        Plot the passing Poincare map and save to a file. It is recommended to only
        call this function on MPI rank 0.

        Args:
            ax : Matplotlib axis to plot on. If None, a new figure and axis are
                 created.
            filename : Name of the file to save the plot
                       (default: 'passing_poincare.pdf').
        Returns:
            ax : The Matplotlib axis containing the plot.
        """
        import matplotlib

        matplotlib.use("Agg")  # Don't use interactive backend
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$s$")
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 1])
        for i in range(len(self.thetas_all)):
            ax.scatter(
                np.mod(self.thetas_all[i], 2 * np.pi),
                self.s_all[i],
                marker="o",
                s=0.5,
                edgecolors="none",
            )
            plt.savefig(filename)

        return ax


class TrappedPoincare:
    """
    Class to compute and store trapped Poincare maps and related quantities
    for a given BoozerMagneticField.
    """

    def __init__(
        self,
        field,
        helicity_M,
        helicity_N,
        s_mirror,
        theta_mirror,
        zeta_mirror,
        mass,
        charge,
        Ekin,
        ns_poinc=None,
        neta_poinc=None,
        s_init=None,
        etas_init=None,
        Nmaps=500,
        comm=None,
        tmax=1e-2,
        solver_options=None,
    ):
        r"""
        Initialize and compute the trapped Poincare map, evaluated by
        integrating the guiding center equations from the v_{\|} = 0 plane
        until the trajectory returns to the v_{\|} = 0 plane.
        The field strength contours are assumed to have helicity (M,N) in
        Boozer coordinates. The mapping coordinate, eta, is chosen based on
        the helicity of the field strength contours: if M = 0 (e.g., QP or
        OP), then eta=theta is used, if N = 0 (e.g., QA/OA, QH/OA), then
        eta = nfp*zeta is used.

        The particle mu is selected by providing a single mirror point,
        (s_mirror, theta_mirror, zeta_mirror), where the parallel velocity is
        zero. The pitch-angle variable, :math:`\lambda = v_\perp^2/(v^2 B)`,
        is computed and held fixed for all trajectories.

        Args:
            field : The :class:`BoozerMagneticField` instance.
            mass : Particle mass.
            charge : Particle charge.
            Ekin : Particle total energy.
            s_init : List of initial s coordinates for the Poincare map.
                     (default: None, ns_poinc is used instead)
            etas_init : List of initial eta coordinates for the Poincare map.
                        (default: None, neta_poinc is used instead)
            ns_poinc : Number of initial conditions in s for Poincare plot
                       (default: 120).
            neta_poinc : Number of initial conditions in eta for Poincare plot
                         (default: 2).
            Nmaps : Number of Poincare return maps to compute for each initial
                    condition (default: 500).
            comm : MPI communicator for parallel execution (default: None).
            tmax : Maximum integration time for each segment of the Poincare
                   map (default: 1e-2 s).
            solver_options : Dictionary of options to pass to the ODE solver
                             (default: {}).
        """
        if solver_options is None:
            solver_options = {}
        self.field = field

        self.helicity_M = helicity_M
        self.helicity_N = helicity_N
        # If modB contours close poloidally, then use theta as mapping coordinate
        if self.helicity_M == 0:
            self.helicity_Mp = 1
            self.helicity_Np = 0
        # Otherwise, use zeta as mapping coordinate
        else:
            self.helicity_Mp = 0
            self.helicity_Np = self.field.nfp

        self.s_mirror = s_mirror
        self.theta_mirror = theta_mirror
        self.zeta_mirror = zeta_mirror
        field.set_points(np.array([[s_mirror], [theta_mirror], [zeta_mirror]]).T)
        self.modBcrit = field.modB()[0, 0]  # Magnetic field at mirror point
        self.lam = 1 / self.modBcrit  # lambda = v_perp^2/(v^2 B) = 1/modBcrit
        self.mass = mass
        self.charge = charge
        self.Ekin = Ekin
        if s_init is not None and etas_init is not None:
            self.s_init = s_init
            self.etas_init = etas_init
        else:
            if ns_poinc is None:
                ns_poinc = 120
            if neta_poinc is None:
                neta_poinc = 2
            self.ns_poinc = ns_poinc
            self.neta_poinc = neta_poinc
        self.Nmaps = Nmaps
        self.comm = comm
        self.tmax = tmax
        self.solver_options = solver_options

        denom = self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        self.dtheta_dchi = self.helicity_Np / denom
        self.dzeta_dchi = self.helicity_Mp / denom

        (
            self.s_all,
            self.chis_all,
            self.etas_all,
            self.t_all,
        ) = self.compute_trapped_map()

    def chi(self, theta, zeta):
        r"""
        Compute the helical angle chi = M*theta - N*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            chi : The helical angle.
        """
        return self.helicity_M * theta - self.helicity_N * zeta

    def eta(self, theta, zeta):
        r"""
        Compute the mapping angle eta = Mp*theta - Np*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            eta : The mapping angle.
        """
        return self.helicity_Mp * theta - self.helicity_Np * zeta

    def chi_eta_to_theta_zeta(self, chi, eta):
        r"""
        Convert helical angles (chi, eta) to (theta, zeta).

        Args:
            chi : Helical angle chi.
            eta : Mapping angle eta.
        Returns:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        """
        denom = self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        theta = (self.helicity_Np * chi - self.helicity_N * eta) / denom
        zeta = (self.helicity_Mp * chi - self.helicity_M * eta) / denom

        return theta, zeta

    def trapped_map(self, point):
        r"""
        Integrates the gc equations from one mirror point to the next mirror point.
        point contains the [s, theta, zeta] coordinates and returns the same coordinates
        after mapping.

        Args:
            point : A numpy array of shape (3,) containing the initial coordinates
                (s,theta,zeta).
        Returns:
            point : A numpy array of shape (3,) containing the coordinates
                (s,theta,zeta) when the trajectory returns to the vpar = 0 plane.
            time : The time taken to return to the vpar = 0 plane.
        """
        theta, zeta = self.chi_eta_to_theta_zeta(point[1], point[2])
        points = np.zeros((1, 3))
        points[:, 0] = point[0]
        points[:, 1] = theta
        points[:, 2] = zeta

        # Set solver options needed for passing map
        res_tys, res_hits = trace_particles_boozer(
            self.field,
            points,
            [0],
            tmax=self.tmax,
            mass=self.mass,
            charge=self.charge,
            Ekin=self.Ekin,
            vpars=[0],
            stopping_criteria=[
                MinToroidalFluxStoppingCriterion(0.01),
                MaxToroidalFluxStoppingCriterion(1.0),
            ],
            forget_exact_path=False,
            vpars_stop=True,
            **self.solver_options,
        )

        if len(res_hits[0]) == 0:
            raise RuntimeError("No stopping criterion reached in trapped_map.")

        res_hit = res_hits[0][0, :]  # Only check the first hit or stopping criterion

        if res_hit[1] == 0:  # Check that the vpars=[0] plane was hit
            point[0] = res_hit[2]
            point[1] = self.chi(res_hit[3], res_hit[4])
            point[2] = self.eta(res_hit[3], res_hit[4])
            time = res_hit[0]
            return point, time
        else:
            raise RuntimeError("Alternative stopping criterion reached in passing_map.")

    def initialize_trapped_map(self):
        r"""
        For the given :class:`BoozerMagneticField` instance, this function
        generates initial positions for the trapped Poincare return map.
        Particles are initialized on the vpar = 0 plane such that the parallel
        velocity is consistent with the prescribed total energy and pitch-angle
        variable, :math:`\lambda = v_\perp^2/(v^2 B)` (i.e., all points are
        mirror points).

        Returns:
            s_init : List of initial s coordinates for the Poincare map.
            thetas_init : List of initial theta coordinates for the Poincare map.
            zetas_init : List of initial zeta coordinates for the Poincare map.
        """

        # We now compute all of the chi mirror points for each s and eta
        def chi_mirror_func(s, eta):
            from scipy.optimize import root_scalar

            point = np.zeros((1, 3))
            point[:, 0] = s

            # Peform root solve to find bounce point
            def diffmodB(chi):
                return modB_func(chi) - self.modBcrit

            def graddiffmodB(chi):
                theta, zeta = self.chi_eta_to_theta_zeta(chi, eta)
                point[:, 1] = theta
                point[:, 2] = zeta
                self.field.set_points(point)
                return (
                    self.field.dmodBdtheta()[0, 0] * self.dtheta_dchi
                    + self.field.dmodBdzeta()[0, 0] * self.dzeta_dchi
                )

            def modB_func(chi):
                theta, zeta = self.chi_eta_to_theta_zeta(chi, eta)
                point[:, 1] = theta
                point[:, 2] = zeta
                self.field.set_points(point)
                return self.field.modB()[0, 0]

            chi_mirror = self.chi(self.theta_mirror, self.zeta_mirror)
            try:
                sol = root_scalar(
                    diffmodB,
                    fprime=graddiffmodB,
                    x0=chi_mirror,
                    method="toms748",
                    bracket=[0, np.pi],
                )
            except Exception as err:
                raise RuntimeError(
                    f"Root solve for chi_mirror failed! s = {s}, "
                    f"eta/(2*pi) = {eta / (2 * np.pi)}"
                ) from err
            if not sol.converged:
                raise RuntimeError(
                    f"Root solve for chi_mirror did not converge! s = {s}, "
                    f"eta/(2*pi) = {eta / (2 * np.pi)}"
                )
            return sol.root

        # Create mesh grid if not provided directly
        if not hasattr(self, "s_init") or not hasattr(self, "etas_init"):
            etas = np.linspace(0, 2 * np.pi, self.neta_poinc, endpoint=False)
            s = np.linspace(0, 1.0, self.ns_poinc + 1, endpoint=False)[1::]
            etas2d, s2d = np.meshgrid(etas, s)
            etas2d = etas2d.flatten()
            s2d = s2d.flatten()
        else:
            s2d = self.s_init
            etas2d = self.etas_init

        s_init = []
        chis_init = []
        etas_init = []
        first, last = parallel_loop_bounds(self.comm, len(etas2d))
        # For each point, find the mirror point in chi
        for i in range(first, last):
            try:
                chi = chi_mirror_func(s2d[i], etas2d[i])
                s_init.append(s2d[i])
                chis_init.append(chi)
                etas_init.append(etas2d[i])
            except RuntimeError:
                warn(
                    f"Root solve for chi_mirror failed! s = {s2d[i]}, "
                    f"eta/(2*pi) = {etas2d[i] / (2 * np.pi)}",
                    stacklevel=2,
                )

        if self.comm is not None:
            s_init = [i for o in self.comm.allgather(s_init) for i in o]
            chis_init = [i for o in self.comm.allgather(chis_init) for i in o]
            etas_init = [i for o in self.comm.allgather(etas_init) for i in o]

        return s_init, chis_init, etas_init

    def compute_trapped_map(self):
        r"""
        Evaluates the trapped Poincare return map for the initialized particle
        positions.
        """
        self.s_init, self.chis_init, self.etas_init = self.initialize_trapped_map()
        Ntrj = len(self.s_init)

        s_all = []
        chis_all = []
        etas_all = []
        t_all = []
        first, last = parallel_loop_bounds(self.comm, Ntrj)
        for itrj in range(first, last):
            tr = [self.s_init[itrj], self.chis_init[itrj], self.etas_init[itrj]]
            s_traj = [tr[0]]
            chis_traj = [tr[1]]
            etas_traj = [tr[2]]
            t_traj = [0]
            broken = False
            for _jj in range(self.Nmaps):
                try:
                    # Apply trapped map twice to return to same vpar = 0 plane
                    tr, time1 = self.trapped_map(tr)
                    tr, time2 = self.trapped_map(tr)
                    if np.abs(tr[1] - chis_traj[-1]) > 2 * np.pi:
                        warn(
                            "Barely trapped particle detected in trapped_map.",
                            stacklevel=2,
                        )
                        broken = True
                        break
                    s_traj.append(tr[0])
                    chis_traj.append(tr[1])
                    etas_traj.append(tr[2])
                    t_traj.append(time1 + time2)
                except RuntimeError:
                    broken = True
                    break
            if not broken:
                s_all.append(s_traj)
                chis_all.append(chis_traj)
                etas_all.append(etas_traj)
                t_all.append(t_traj)

        if self.comm is not None:
            s_all = [i for o in self.comm.allgather(s_all) for i in o]
            chis_all = [i for o in self.comm.allgather(chis_all) for i in o]
            etas_all = [i for o in self.comm.allgather(etas_all) for i in o]
            t_all = [i for o in self.comm.allgather(t_all) for i in o]

        return s_all, chis_all, etas_all, t_all

    def plot_poincare(self, ax=None, filename="trapped_poincare.pdf"):
        r"""
        Plot the trapped Poincare map and save to a file. It is recommended to only
        call this function on MPI rank 0.

        Args:
            ax : Matplotlib axis to plot on. If None, a new figure and axis are
                 created.
            filename : Name of the file to save the plot
                       (default: 'trapped_poincare.pdf').
        Returns:
            ax : The Matplotlib axis containing the plot.
        """
        import matplotlib

        matplotlib.use("Agg")  # Don't use interactive backend
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$s$")
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 1])
        for i in range(len(self.etas_all)):
            ax.scatter(
                np.mod(self.etas_all[i], 2 * np.pi),
                self.s_all[i],
                marker="o",
                s=0.5,
                edgecolors="none",
            )
        plt.savefig(filename)

        return ax

    def get_poincare_data(self):
        """
        Return the Poincare map data.

        Returns:
            s_all, chis_all, etas_all, t_all : Lists of trajectory data.
        """
        return self.s_all, self.chis_all, self.etas_all, self.t_all

    def compute_frequencies(self):
        """
        Compute the trapped particle eta and bounce frequencies and mean
        radial position.
        The frequency is only computed if the trajectory has completed at least
        one full Poincare return map before the trajectory is terminated due to
        a time limit or stopping criterion. Since the frequency will depend on
        field-line label and trapping well in a general 3D field, an average is
        taken over all trajectories initialized on the same flux surface and
        all Poincare return maps along each trajectory.
        Since the frequency profile will flatten in a phase-space island, it is
        sometimes easier to interpret the frequency profile in a field with
        enforced quasisymmetry (i.e., initialize BoozerRadialInterpolant with N
        prescribed).

        Returns:
            omega_eta : List of mapping angle eta frequencies.
            omega_b : List of bounce frequencies.
            init_s : List of initial s values for each trajectory.
        """
        if self.solver_options["axis"] != 0:
            raise ValueError(
                'ODE solver must integrate with solver_options["axis"]=0 to '
                "compute trapped frequencies."
            )

        omega_eta = []
        omega_b = []
        init_s = []
        for s_traj, _chi_traj, eta_traj, t_traj in zip(
            self.s_all, self.chis_all, self.etas_all, self.t_all
        ):
            if (
                len(s_traj) < 2
            ):  # Need at least one full Poincare return maps to compute frequency
                continue
            delta_eta = np.array(eta_traj[1::]) - np.array(eta_traj[0:-1])
            delta_t = t_traj[1::]

            # Average over wells along one field line
            omega_eta.append(np.mean(delta_eta) / np.mean(delta_t))
            omega_b.append(2 * np.pi / np.mean(delta_t))  # bounce frequency
            init_s.append(np.mean(s_traj))

        omega_eta = np.array(omega_eta)
        omega_b = np.array(omega_b)
        init_s = np.array(init_s)

        s_prof = np.unique(init_s)
        omega_eta_prof = np.zeros((len(s_prof),))
        omega_b_prof = np.zeros((len(s_prof),))

        # Average over field-line label
        for i, s in enumerate(s_prof):
            omega_eta_prof[i] = np.mean(omega_eta[np.where(init_s == s)])
            omega_b_prof[i] = np.mean(omega_b[np.where(init_s == s)])

        return omega_eta_prof, omega_b_prof, s_prof


def compute_peta(
    field_or_saw,
    points,
    vpar,
    mass,
    charge,
    helicity_M,
    helicity_N,
    helicity_Mp=None,
    helicity_Np=None,
):
    r"""
    Given a ShearAlfvenWave or BoozerMagneticField instance, a point in
    Boozer coordinates, and particle properties, compute the value of the
    canonical momentum, :math:`p_{\eta}`. This quantity is conserved under
    the unperturbed guiding center equations if the field strength is exactly
    quasisymmetric with helicity (M,N) and :math:`\alpha = 0`.

    :math:`p_{\eta} = (M G + N I) \left(\frac{m v_{\|\|}}{B} + q \alpha \right) +
    q (M \psi - M \psi')`

    If field_or_saw is a BoozerMagneticField instance, then alpha = 0.

    Args:
        field_or_saw : The BoozerMagneticField or ShearAlfvenWave instance.
        points : A numpy array of shape (npoints,4) containing the coordinates
                 (s,theta,zeta,t).
            If field_or_saw is a ShearAlfvenWave, then t is the time coordinate.
            If field_or_saw is a BoozerMagneticField, then t is ignored, and
            points is allowed to have shape (npoints,3) for (s,theta,zeta).
        vpar : A numpy array of shape (npoints,) containing the parallel velocity.
        mass : Mass of the particle.
        charge : Charge of the particle.
        helicity_M : Poloidal helicity of the magnetic field.
        helicity_N : Toroidal helicity of the magnetic field.
        helicity_Mp : Poloidal helicity of the mapping coordinate eta.
            If None, then eta is chosen based on the helicity of the field strength.
        helicity_Np : Toroidal helicity of the mapping coordinate eta.
            If None, then eta is chosen based on the helicity of the field strength.

    Returns:
        peta : A numpy array of shape (npoints,) containing the value of the canonical
            momentum :math:`p_{\eta}` at each point.
    """
    if points.shape[1] not in [3, 4]:
        raise ValueError(
            "Points must have shape (npoints, 4) for (s, theta, zeta, t) or "
            "(npoints, 3) for (s, theta, zeta)"
        )
    if isinstance(vpar, float):
        vpar = np.array([vpar])
    if isinstance(vpar, list):
        vpar = np.array(vpar)
    assert vpar.shape[0] == points.shape[0], (
        "vpar must have the same number of points as points"
    )

    if isinstance(field_or_saw, ShearAlfvenWave):
        field = field_or_saw.B0
        field_or_saw.set_points(points)
        alpha = field_or_saw.alpha()[:, 0]
    else:
        field = field_or_saw
        alpha = 0.0
        if points.shape == 4:
            points = points[:, :3]
        field.set_points(points)

    modB = field.modB()[:, 0]
    G = field.G()[:, 0]
    I = field.I()[:, 0]
    psi = field.psi0 * points[:, 0]
    psip = field.psip()[:, 0]

    if helicity_Mp is None and helicity_Np is None:
        # If modB contours close poloidally, then use theta as mapping coordinate
        if helicity_M == 0:
            helicity_Mp = 1
            helicity_Np = 0
        # Otherwise, use zeta as mapping coordinate
        else:
            helicity_Mp = 0
            helicity_Np = -1
    else:
        if (helicity_Mp * helicity_N) == (helicity_Np * helicity_M):
            raise ValueError(
                "Chosen helicities (N, M, N', M') do not create a well "
                "defined Jacobian."
            )
    denom = helicity_Np * helicity_M - helicity_N * helicity_Mp
    peta = (
        -(
            (helicity_M * G + helicity_N * I) * (mass * vpar / modB + charge * alpha)
            + charge * (helicity_N * psi - helicity_M * psip)
        )
        / denom
    )
    return peta


def compute_Eprime(saw, points, vpar, mu, mass, charge, helicity_M, helicity_N):
    r"""
    Compute the invariant Eprime for a ShearAlfvenHarmonic instance given
    points in Boozer coordinates.

    Args:
        saw : An instance of ShearAlfvenHarmonic.
        points : A numpy array of shape (npoints,4) containing the coordinates
                 (s,theta,zeta,t).
        vpar : A numpy array of shape (npoints,) containing the parallel velocity.
        mu : Magnetic moment of the particle, vperp^2/(2 B).
        mass : Mass of the particle.
        charge : Charge of the particle.
        helicity_M : Poloidal helicity of the magnetic field strength.
        helicity_N : Toroidal helicity of the magnetic field strength.
    """
    if points.shape[1] != 4:
        raise ValueError("Points must have shape (npoints, 4) for (s, theta, zeta, t)")
    if isinstance(vpar, float):
        vpar = np.array([vpar])
    if isinstance(vpar, list):
        vpar = np.array(vpar)
    assert vpar.shape[0] == points.shape[0], (
        "vpar must have the same number of points as points"
    )
    if vpar.shape[0] != points.shape[0]:
        raise ValueError("vpar must have the same number of points as points")
    if isinstance(saw, ShearAlfvenHarmonic) is False:
        raise TypeError("Expected saw to be an instance of ShearAlfvenHarmonic")

    # If modB contours close poloidally, then use theta as mapping coordinate
    if helicity_M == 0:
        helicity_Mp = 1
        helicity_Np = 0
    # Otherwise, use zeta as mapping coordinate
    else:
        helicity_Mp = 0
        helicity_Np = -1

    Phim = saw.Phim
    Phin = saw.Phin
    omega = saw.omega

    # Compute the canonical momentum p_eta
    p_eta = compute_peta(saw, points, vpar, mass, charge, helicity_M, helicity_N)

    # Compute the energy E
    modB = saw.B0.modB()[:, 0]
    E = 0.5 * mass * vpar**2 + mass * mu * modB + charge * saw.Phi()[:, 0]

    # Compute the invariant Eprime
    nprime = (Phim * helicity_N - Phin * helicity_M) / (
        helicity_Np * helicity_M - helicity_N * helicity_Mp
    )
    Eprime = nprime * E - omega * p_eta
    return Eprime


def g(t, T):
    """
    Smooth bump weight on (0, T) with g=0 at t<=0 or t>=T.
    """
    t = np.asarray(t, dtype=float)
    T = float(T)

    s = t / T

    # denom = s*(1-s); positive only for s in (0,1)
    denom = s * (1.0 - s)
    w = np.zeros_like(s)

    interior = (s > 0.0) & (s < 1.0)
    denom = s[interior] * (1.0 - s[interior])
    w[interior] = np.exp(-1.0 / denom)

    return w


def return_DA(array):
    """
    Compute the DA metric for a given momentum 2D array.
    Computes the normalized digit difference between the
    WBA at the first half of the trajectory and the final
    given point.
    Args:
        array : A numpy array of shape (npoints, 2) containing the time and
                momentum values.
    Returns:
        T : Final computed time of the trajectory.
        da_c : The computed DA metric value.
    """
    time_m = array[:, 0]
    momentum = array[:, 1]

    if len(time_m) < 4:
        return 0, np.nan

    # full trajectory
    T_idx = len(time_m)
    T_time = time_m
    T_mom = momentum
    T = float(T_time[-1])

    t_idx = int(T_idx / 2)
    t_time = T_time[:t_idx]
    t_mom = T_mom[:t_idx]
    t = float(t_time[-1])

    # weight arrays
    g_t = g(t_time, t)
    g_T = g(T_time, T)

    g_t_int = integrate.trapezoid(g_t, t_time)
    g_T_int = integrate.trapezoid(g_T, T_time)

    if np.isclose(g_t_int, 0.0) and np.isclose(g_T_int, 0.0):
        return T, np.nan

    t_wavg = integrate.trapezoid(g_t * t_mom, t_time) / g_t_int
    T_wavg = integrate.trapezoid(g_T * T_mom, T_time) / g_T_int

    # DA metric
    diff = np.abs(t_wavg - T_wavg)
    denom = 0.5 * (np.abs(t_wavg) + np.abs(T_wavg))

    if diff == 0.0:
        return T, 16

    ratio = diff / denom
    da_c = -np.log10(ratio)

    return T, da_c


class PassingPerturbedPoincare:
    def __init__(
        self,
        saw,
        sign_vpar,
        mass,
        charge,
        helicity_M,
        helicity_N,
        helicity_Mp=None,
        helicity_Np=None,
        Eprime=None,
        mu=None,
        Ekin=None,
        p0=None,
        lam=None,
        ns_poinc=None,
        nchi_poinc=None,
        DA_poinc=False,
        nconvergence_points=None,
        s_init=None,
        chis_init=None,
        Nmaps=500,
        comm=None,
        tmax=1e-2,
        solver_options=None,
    ):
        """
        Initialize the PassingPerturbedPoincare class, which computes the
        Poincare return map for passing particles in a ShearAlfvenHarmonic
        magnetic field.

        The field strength contours are assumed to have helicity (M,N) in
        Boozer coordinates such that the field strength can be expressed as
        B(s,chi), where chi = M*theta - N*zeta is the helical angle.
        The mapping coordinate, eta, is chosen based on the helicity of the
        field strength contours: if M = 0 (e.g., QP or OP), then eta=theta is
        used, if N = 0 (e.g., QA/OA, QH/OA), then eta = zeta is used.

        The ShearAlfvenHarmonic can then be expressed in terms of the mapping
        coordinates with phase, m'*chi - n'*eta + omega * t.

        The map is evaluated by integrating the guiding center equations from
        the eta - omega/n' * t = 0 plane until the trajectory returns to the
        same plane.

        The map is well-defined (i.e., trajectories don't cross in the
        (s,chi=M*theta - N*zeta) plane) if the unperturbed field is
        quasisymmetric with helicity (M,N). However, the map can still be
        computed for a non-quasisymmetric field, but the trajectories may
        cross in the (s,chi) plane.

        The constants of motion are the magnetic moment, mu = vperp^2/(2 B),
        and the shifted energy, Eprime = n' * E - omega * p_eta, E is the
        total energy, and p_eta is the canonical momentum.

        These constants of motion can be prescribed directly with the Eprime
        and mu parameters. Alternatively, they can be computed from the
        prescribed pitch-angle variable, lam = vperp^2/(v^2 B), total
        unperturbed kinetic energy, Ekin, and a given point p0 in Boozer
        coordinates.

        Args:
            saw : An instance of ShearAlfvenHarmonic or ShearAlfvenWavesSuperposition.
            sign_vpar : Sign of the parallel velocity, either -1 or +1.
            mass : Mass of the particle.
            charge : Charge of the particle.
            helicity_M : Poloidal helicity of the magnetic field.
            helicity_N : Toroidal helicity of the magnetic field.
            helicity_Mp : Poloidal helicity of the phase variable eta.
            Defaults to None. If no value is given, Mp and Np are determined
            by field helicity.
            helicity_Np : Toroidal helicity of the phase variable eta.
            Defaults to None. If no value is given, Mp and Np are determined by field
            helicity.
            Eprime: Shifted energy, Eprime = n' * E - omega * p_eta.
            mu: Magnetic moment, mu = vperp^2/(2 B).
            Ekin: Total unperturbed kinetic energy of the particle, used to
                compute Eprime if not provided.
            p0: Initial point in Boozer coordinates for evaluation of Eprime.
            lam: Pitch angle variable, lambda = vperp^2/(v^2 B), used to
                compute mu if not provided.
            s_init : List of initial s coordinates for the Poincare map.
                (default: None, ns_poinc is used instead)
            chis_init : List of initial chi coordinates for the Poincare map.
                (default: None, nchi_poinc is used instead)
            ns_poinc : Number of initial conditions in s for Poincare plot
                (default: 120).
            nchi_poinc : Number of initial conditions in chi for Poincare plot
                (default: 2).
            Nmaps : Number of Poincare return maps to compute for each initial
                condition (default: 500).
            DA_poinc : Boolean value indicating whether chaos detection is desired
                (default: False)
            nconvergence_points : Integer value indicating the number of Weighted
                Birkhoff Average evaluations to assess convergence.
            comm : MPI communicator for parallel execution (default: None).
            tmax : Maximum integration time for each segment of the Poincare
                map (default: 1e-2 s).
            solver_options : Dictionary of options to pass to the ODE solver
                (default: {}).
        """
        if solver_options is None:
            solver_options = {}
        if not isinstance(saw, ShearAlfvenHarmonic) and not isinstance(
            saw, ShearAlfvenWavesSuperposition
        ):
            raise TypeError(
                "Expected saw to be an instance of ShearAlfvenHarmonic "
                "or ShearAlfvenWavesSuperposition"
            )
        if not isinstance(saw, ShearAlfvenHarmonic):
            dominant_saw = saw[0]
            raise Warning(
                "Expected saw to be an instance of ShearAlfvenHarmonic - "
                "Perturbed Energy Invariant may not be valid."
            )
        else:
            dominant_saw = None
        if sign_vpar not in [-1, 1]:
            raise ValueError("sign_vpar should be either -1 or +1")

        self.saw = saw
        self.B0 = saw.B0
        self.helicity_M = helicity_M
        self.helicity_N = helicity_N
        self._set_helicity_Np_Mp(helicity_Np, helicity_Mp)
        self.mass = mass
        self.charge = charge
        self.sign_vpar = sign_vpar

        self.DA_poinc = DA_poinc
        if DA_poinc:
            if nconvergence_points is None:
                self.nconvergence_points = 1
                self.WBA_transit_steps = [Nmaps - 1]
            else:
                self.nconvergence_points = nconvergence_points
                # set list of transits for each WBA evaluation
                transits_per_average = int(Nmaps / (nconvergence_points))
                self.WBA_transit_steps = np.linspace(
                    transits_per_average, Nmaps - 1, num=nconvergence_points, dtype=int
                ).tolist()
        else:
            self.nconvergence_points = 1
            self.WBA_transit_steps = 0

        if s_init is not None and chis_init is not None:
            self.s_init = s_init
            self.chis_init = chis_init
        else:
            if ns_poinc is None:
                ns_poinc = 120
            if nchi_poinc is None:
                nchi_poinc = 2
            self.ns_poinc = ns_poinc
            self.nchi_poinc = nchi_poinc
        self.Nmaps = Nmaps
        self.comm = comm
        self.tmax = tmax
        self.solver_options = solver_options

        # if using a ShearAlfvenWavesSuperposition, use the test_saw for
        # computing Eprime, Phin, Phim, and omega as the largest mode
        # this does not produce a poincare plot in the strict sense,
        # but you will be unable to visualize surfaces
        if dominant_saw is None:
            self.Phin = saw.Phin
            self.Phim = saw.Phim
            self.omega = saw.omega
        else:
            self.Phin = dominant_saw.Phin
            self.Phim = dominant_saw.Phim
            self.omega = dominant_saw.omega

        self.nprime = (self.Phim * self.helicity_N - self.Phin * self.helicity_M) / (
            self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        )

        if Eprime is not None and mu is not None:
            self.Eprime = Eprime
            self.mu = mu
            self.Ekin = None
        elif Ekin is not None and lam is not None and p0 is not None:
            """
            Compute unperturbed values of mu, p_eta, and Eprime from the given
            parameters.
            """
            self.v0 = np.sqrt(2 * Ekin / mass)  # Total velocity from kinetic energy
            self.mu = 0.5 * lam * self.v0**2  # mu = vperp^2/(2 B)
            self.Ekin = Ekin  # Total kinetic energy
            saw.B0.set_points(p0)
            modB = saw.B0.modB()[0, 0]
            if 1 - lam * modB < 0:
                raise ValueError(
                    "Invalid parameter p0: 1 - lambda * modB must be non-negative."
                )
            vpar = sign_vpar * self.v0 * np.sqrt(1 - lam * modB)  # Parallel velocity
            Peta0 = compute_peta(
                saw.B0,
                p0,
                vpar,
                mass,
                charge,
                helicity_M,
                helicity_N,
            )
            self.Eprime = self.nprime * Ekin - self.omega * Peta0
        else:
            raise ValueError(
                "Either Eprime and mu must be provided, or Ekin, lam, and p0 "
                "must be provided."
            )

        # Initialize the passing map
        self.s_init, self.chis_init, self.vpars_init = self.initialize_passing_map()
        # If Ekin is not provided, compute it from the initial parallel velocity
        # this is only used for computing maximum time step in the ODE solver
        if self.Ekin is None:
            self.Ekin = 0.5 * self.mass * self.vpars_init[0] ** 2

        (
            self.s_all,
            self.chis_all,
            self.etas_all,
            self.vpars_all,
            self.t_all,
            self.DA_all,
            self.DA_times,
        ) = self.compute_passing_map()

    def _set_helicity_Np_Mp(self, helicity_Np, helicity_Mp):
        """
        Sets helicity of the phase variable eta based on user inputs.
        """
        if not ((helicity_Np is None) or (helicity_Mp is None)):
            # User specified both helicities
            self.helicity_Mp = helicity_Mp
            self.helicity_Np = helicity_Np
            return

        if (helicity_Np is None) and (helicity_Mp is None):
            # User did not specify helicity, choose default
            if self.helicity_M == 0:
                # modB contours close poloidally,
                # so can use theta as mapping coordinate
                self.helicity_Mp = 1
                self.helicity_Np = 0
            else:
                # use zeta as mapping coordinate
                self.helicity_Mp = 0
                self.helicity_Np = -1
            return

        raise ValueError(
            f"User must either specify both helicity_Np and helicity_Mp or leave both "
            f"of them None. Currently {helicity_Np=} while {helicity_Mp=}."
        )
    def vpar_func_perturbed(self, s, chi):
            # Choose initial conditions on the eta = 0 plane
            theta, zeta = self.chi_eta_to_theta_zeta(chi, 0)
            point = np.zeros((1, 4))  # initialize with t = 0
            point[0, 0] = s
            point[0, 1] = theta
            point[0, 2] = zeta
            self.saw.set_points(point)
            modB = self.B0.modB()[0, 0]
            G = self.B0.G()[0, 0]
            I = self.B0.I()[0, 0]
            psi = self.B0.psi0 * s
            psip = self.B0.psip()[0, 0]
            Phi = self.saw.Phi()[0, 0]
            alpha = self.saw.alpha()[0, 0]
            denom = (
                self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
            )  # - 1 in QA
            d_peta_d_vpar = (
                -((self.helicity_M * G + self.helicity_N * I) * (self.mass / modB))
                / denom
            )  # G m/ modB in QA
            d_E_d_vpar2 = 0.5 * self.mass
            a = self.nprime * d_E_d_vpar2  # Coefficient of vpar^2
            b = -self.omega * d_peta_d_vpar  # Coefficient of vpar
            # Constant term
            c = (
                self.nprime * (self.mass * self.mu * modB + self.charge * Phi)
                + self.omega
                * (
                    (self.helicity_M * G + self.helicity_N * I) * self.charge * alpha
                    + self.charge * (self.helicity_N * psi - self.helicity_M * psip)
                )
                / denom
                - self.Eprime
            )
            if (b**2 - 4 * a * c) < 0:
                raise RuntimeError(
                    "No solution for vpar found! Check the parameters and "
                    "initial conditions."
                )
            elif a != 0:
                return (-b + self.sign_vpar * np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            else:
                return (-c / b) * self.sign_vpar

    def initialize_passing_map(self):
        """
        Compute vpar given (s,chi) such that Eprime = Eprime0
        """

        def vpar_func_perturbed(s, chi):
            # Choose initial conditions on the eta = 0 plane
            theta, zeta = self.chi_eta_to_theta_zeta(chi, 0)
            point = np.zeros((1, 4))  # initialize with t = 0
            point[0, 0] = s
            point[0, 1] = theta
            point[0, 2] = zeta
            self.saw.set_points(point)
            modB = self.B0.modB()[0, 0]
            G = self.B0.G()[0, 0]
            I = self.B0.I()[0, 0]
            psi = self.B0.psi0 * s
            psip = self.B0.psip()[0, 0]
            Phi = self.saw.Phi()[0, 0]
            alpha = self.saw.alpha()[0, 0]
            denom = (
                self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
            )  # - 1 in QA
            d_peta_d_vpar = (
                -((self.helicity_M * G + self.helicity_N * I) * (self.mass / modB))
                / denom
            )  # G m/ modB in QA
            d_E_d_vpar2 = 0.5 * self.mass
            a = self.nprime * d_E_d_vpar2  # Coefficient of vpar^2
            b = -self.omega * d_peta_d_vpar  # Coefficient of vpar
            # Constant term
            c = (
                self.nprime * (self.mass * self.mu * modB + self.charge * Phi)
                + self.omega
                * (
                    (self.helicity_M * G + self.helicity_N * I) * self.charge * alpha
                    + self.charge * (self.helicity_N * psi - self.helicity_M * psip)
                )
                / denom
                - self.Eprime
            )
            if (b**2 - 4 * a * c) < 0:
                raise RuntimeError(
                    "No solution for vpar found! Check the parameters and "
                    "initial conditions."
                )
            elif a != 0:
                return (-b + self.sign_vpar * np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            else:
                return (-c / b) * self.sign_vpar

        # Create mesh grid if not provided directly
        if not hasattr(self, "s_init") or not hasattr(self, "chis_init"):
            s = np.linspace(0, 1, self.ns_poinc + 1, endpoint=False)[1::]
            chis = np.linspace(0, 2 * np.pi, self.nchi_poinc)
            s, chis = np.meshgrid(s, chis)
            s = s.flatten()
            chis = chis.flatten()
        else:
            s = self.s_init
            chis = self.chis_init

        first, last = parallel_loop_bounds(self.comm, len(s))
        # For each point, find value of vpar such that lambda = vperp^2/(v^2 B)
        s_init = []
        chis_init = []
        vpars_init = []
        for i in range(first, last):
            try:
                vpar = vpar_func_perturbed(s[i], chis[i])
                s_init.append(s[i])
                chis_init.append(chis[i])
                vpars_init.append(vpar)
            except RuntimeError:
                continue

        if self.comm is not None:
            s_init = [i for o in self.comm.allgather(s_init) for i in o]
            chis_init = [i for o in self.comm.allgather(chis_init) for i in o]
            vpars_init = [i for o in self.comm.allgather(vpars_init) for i in o]

        return s_init, chis_init, vpars_init

    def chi(self, theta, zeta):
        r"""
        Compute the helical angle chi = M*theta - N*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            chi : The helical angle.
        """
        return self.helicity_M * theta - self.helicity_N * zeta

    def eta(self, theta, zeta):
        r"""
        Compute the mapping angle eta = Mp*theta - Np*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            eta : The mapping angle.
        """
        return self.helicity_Mp * theta - self.helicity_Np * zeta

    def chi_eta_to_theta_zeta(self, chi, eta):
        r"""
        Convert helical angles (chi, eta) to (theta, zeta).

        Args:
            chi : Helical angle chi.
            eta : Mapping angle eta.
        Returns:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        """
        denom = self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        theta = (self.helicity_Np * chi - self.helicity_N * eta) / denom
        zeta = (self.helicity_Mp * chi - self.helicity_M * eta) / denom

        return theta, zeta

    def passing_map(self, point, t, eta):
        r"""
        Integrates the GC equations from the provided point on the eta -
        omega/n' * t plane to the next intersection with this plane. An
        assumption is made that the particle is passing, so a RuntimeError
        is raised if the particle mirrors.

        Since the phase of the ShearAlfvenWave depends on time, the initial
        time, t, is passed as an argument.

        Args:
            point : A numpy array of shape (3,) containing the initial
                coordinates (s,chi,eta).
            t : Initial time at which the map is evaluated
        Returns:
            point : A numpy array of shape (3,) containing the coordinates
                (s,chi,eta).
            time : The time at which the trajectory returns to the eta -
                omega/n' * t plane.
            peta : Timeseries of the canonical momentum p_eta along the trajectory.
        """
        phase = self.omega * t
        self.saw.phase = phase
        theta, zeta = self.chi_eta_to_theta_zeta(point[1], eta)
        points = np.zeros((1, 3))
        points[:, 0] = point[0]
        points[:, 1] = theta
        points[:, 2] = zeta

        if self.helicity_M != 0:
            phases = [zeta * self.nprime]
            n_zetas = [self.nprime]
            m_thetas = [0]
            omegas = [self.omega]
        else:
            phases = [theta * self.nprime]
            n_zetas = [0]
            m_thetas = [self.nprime]
            omegas = [self.omega]
        try:
            res_tys, res_hits = trace_particles_boozer_perturbed(
                perturbed_field=self.saw,
                stz_inits=points,
                parallel_speeds=[point[2]],
                mus=[self.mu],
                tmax=self.tmax,
                mass=self.mass,
                charge=self.charge,
                phases=phases,
                n_zetas=n_zetas,
                m_thetas=m_thetas,
                omegas=omegas,
                vpars=[0],
                axis=0,
                stopping_criteria=[
                    MinToroidalFluxStoppingCriterion(0.001),
                    MaxToroidalFluxStoppingCriterion(0.99),
                ],
                forget_exact_path=True,
                vpars_stop=True,
                phases_stop=True,
                **self.solver_options,
                )
        except: 
            print(f"Error integrating guiding center equations at {point=}, {t=}, {eta=}", flush=True)
            raise RuntimeError()
        if len(res_hits[0]) == 0:
            raise RuntimeError("No stopping criterion reached in passing_map.")

        res_hit = res_hits[0][0, :]  # Only check the first hit or stopping criterion

        # Check that the phases plane was hit (index 0 for first phase)
        if res_hit[1] == 0:
            point[0] = res_hit[2]
            point[1] = self.chi(res_hit[3], res_hit[4])
            point[2] = res_hit[5]
        else:
            raise RuntimeError("Alternative stopping criterion reached in passing_map.")

        if not self.DA_poinc:
            return point, res_hit[0] + t, self.eta(res_hit[3], res_hit[4])
        else:
            # define trajectories
            time_momentum = res_tys[0][:, 0]
            s_path = res_tys[0][:, 1]
            theta_path = res_tys[0][:, 2]
            zeta_path = res_tys[0][:, 3]
            vpar_path = res_tys[0][:, 4]

            # set points for trajectories:
            points_traj = np.zeros((len(time_momentum), 4))
            points_traj[:, 0] = s_path
            points_traj[:, 1] = theta_path
            points_traj[:, 2] = zeta_path
            points_traj[:, 3] = time_momentum

            Peta = compute_peta(
                self.saw,
                points_traj,
                vpar_path,
                self.mass,
                self.charge,
                self.helicity_M,
                self.helicity_N,
                helicity_Mp=self.helicity_Mp,
                helicity_Np=self.helicity_Np,
            )
            Peta = np.column_stack((time_momentum, Peta))
            return point, res_hit[0] + t, self.eta(res_hit[3], res_hit[4]), Peta

    def compute_passing_map(self):
        r"""
        Evaluates the passing Poincare return map for the initialized particle
        positions.
        """
        Ntrj = len(self.s_init)

        s_all = []
        chis_all = []
        etas_all = []
        vpars_all = []
        t_all = []
        DA_all = []
        DA_times = []
        first, last = parallel_loop_bounds(self.comm, Ntrj)
        for itrj in range(first, last):
            tr = [self.s_init[itrj], self.chis_init[itrj], self.vpars_init[itrj]]
            s_traj = [tr[0]]
            chis_traj = [tr[1]]
            eta_traj = [0]
            vpars_traj = [tr[2]]
            t_traj = [0]
            particle_DAs = []
            particle_DA_times = []
            for jj in range(self.Nmaps):
                try:
                    if self.DA_poinc:
                        if jj == 0:
                            tr, time, eta, Peta = self.passing_map(
                                tr, t_traj[-1], eta_traj[-1]
                            )
                        else:
                            tr, time, eta, Peta_iter = self.passing_map(
                                tr, t_traj[-1], eta_traj[-1]
                            )
                            Peta_iter[:, 0] += Peta[-1, 0]
                            Peta = np.vstack((Peta, Peta_iter[1:, :]))
                    else:
                        tr, time, eta = self.passing_map(tr, t_traj[-1], eta_traj[-1])
                    s_traj.append(tr[0])
                    chis_traj.append(tr[1])
                    vpars_traj.append(tr[2])
                    t_traj.append(time)
                    eta_traj.append(eta)
                    if self.DA_poinc and jj in self.WBA_transit_steps:
                        time_at_evaluation, DA_at_evaluation = return_DA(Peta)
                        particle_DAs.append(DA_at_evaluation)
                        particle_DA_times.append(jj)
                except RuntimeError:
                    if self.DA_poinc:
                        particle_DAs.append(np.nan)
                        particle_DA_times.append(np.nan)
                    break
            DA_all.append(particle_DAs)
            DA_times.append(particle_DA_times)
            s_all.append(s_traj)
            chis_all.append(chis_traj)
            etas_all.append(eta_traj)
            vpars_all.append(vpars_traj)
            t_all.append(t_traj)

        if self.comm is not None:
            s_all = [i for o in self.comm.allgather(s_all) for i in o]
            chis_all = [i for o in self.comm.allgather(chis_all) for i in o]
            etas_all = [i for o in self.comm.allgather(etas_all) for i in o]
            vpars_all = [i for o in self.comm.allgather(vpars_all) for i in o]
            t_all = [i for o in self.comm.allgather(t_all) for i in o]
            DA_all = [i for o in self.comm.allgather(DA_all) for i in o]
            DA_times = [i for o in self.comm.allgather(DA_times) for i in o]

        return s_all, chis_all, etas_all, vpars_all, t_all, DA_all, DA_times


    def plot_poincare(
        self,
        ax=None,
        filename="passing_poincare.pdf",
        convergence_test_indicies=None,
        DA_max=7,
        lines=None,
        ylims=(0, 1),
        colorbar=True,
        s_axis_label = True
    ):
        r"""
        Plot the passing Poincare map and save to a file. It is recommended to only
        call this function on MPI rank 0.
        Args:
            ax : Matplotlib axis to plot on. If None, a new figure and axis are
                 created.
            filename : Name of the file to save the plot
                       (default: 'passing_poincare.pdf').
            convergence_test_indicies : Indices of initial conditions to show
            in convergence plot.
            DA_max : Maximum value of Digit Accuracy to show on colorbar
            ylims : Tuple specifying y-axis limits for the Poincare plot.
        Returns:
            ax : The Matplotlib axis containing the plot.
        """

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable

        mpl.use("Agg")  # Don't use interactive backend

        try:
            import cmcrameri.cm as cmc  # noqa: F401

            cmap = "cmc.managua"
        except ImportError:
            cmap = "viridis"

        star_ICs = False

        if convergence_test_indicies is None:
            convergence_test_indicies = list(range(len(self.s_all)))
        else:
            star_ICs = True

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if self.DA_poinc and self.nconvergence_points > 1:
            s_itrj_map = {}
            for itrj in convergence_test_indicies:
                s_itrj_map[itrj] = self.s_all[itrj][0]

            min_s = min(list(s_itrj_map.values()))
            max_s = max(list(s_itrj_map.values()))
            s_lst_true = list(s_itrj_map.values())
            cmap_s = mpl.colormaps["copper"].resampled(len(s_lst_true) ** 2)

        def normalize(numbers):
            if not numbers:
                return []
            min_val, max_val = 0, DA_max
            normalized_numbers = [(x - min_val) / (max_val - min_val) for x in numbers]
            return normalized_numbers

        if self.DA_poinc:
            final_DAs = []
            # retrieve final DA for each trajectory if the particle is not lost
            # put it into a list
            for elem in self.DA_all:
                if len(elem) == self.nconvergence_points:
                    final_DAs.append(elem[self.nconvergence_points - 1])
                else:
                    final_DAs.append(np.nan)
            # normalized DA values for colormap
            DA_norm_all = normalize(final_DAs)
            cmap_object = mpl.colormaps[cmap].resampled(len(self.DA_all) ** 2)

        ax.set_xlabel(r"$\chi$")
        if s_axis_label:
            ax.set_ylabel(r"$s$")
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([ylims[0], ylims[1]])

        for i in range(len(self.chis_all)):
            #if len(self.chis_all[i]) < Nmaps
            if self.DA_poinc:
                ax.scatter(
                    np.mod(self.chis_all[i], 2 * np.pi),
                    self.s_all[i],
                    marker="o",
                    s=2,
                    c=cmap_object(DA_norm_all[i]),
                    edgecolors="none",
                )
            else:
                ax.scatter(
                    np.mod(self.chis_all[i], 2 * np.pi),
                    self.s_all[i],
                    marker="o",
                    s=2,
                    edgecolors="none",
                )

        if star_ICs:
            # scatter initial conditions observed in convergence plot onto poincare plot
            for i in s_itrj_map:
                s_norm = (s_itrj_map[i] - min_s) / (max_s - min_s)
                ax.scatter(
                    np.mod(self.chis_all[i][0], 2 * np.pi),
                    self.s_all[i][0],
                    marker="*",
                    s=25,
                    color=cmap_s(s_norm),
                    edgecolors="magenta",
                )

        if self.DA_poinc:
            # make colorbar for DA values
            max_val = DA_max
            norm = plt.Normalize(0, max_val)
            if colorbar:
                fig.colorbar(
                    ScalarMappable(norm=norm, cmap=mpl.colormaps[cmap]),
                    ax=ax,
                    orientation="vertical",
                    label="Digit Accuracy",
                )
        print('Lines to plot:', lines, flush=True)
        lines_2 = []
        if lines is not None:
            cmap = plt.get_cmap('Wistia')
            n_lines = len(lines)
            for i,line in enumerate(lines):
                print(f'i={i}, {line}',flush=True)
                ell,arr = line[0], line[1]
                color = cmap(i / max(n_lines - 1, 1))
                lines_2.append((line[0], line[1], color))
                vp = self.vpar_func_perturbed(arr, self.chi(np.pi/2,0))
                self.B0.set_points(np.array([[arr, np.pi/2, 0]]).T)
                unperturbed_path_map = PassingPoincare(
                    field=self.B0,
                    lam=(self.v0**2 - vp**2)/(self.v0**2* self.B0.modB()[0,0]),
                    sign_vpar=self.sign_vpar,
                    mass=self.mass,
                    charge=self.charge,
                    Ekin=self.Ekin,
                    s_init=[arr],
                    comm=None,
                    Nmaps=250,
                    thetas_init=[np.pi/2],
                    solver_options={'axis':0},
                )

                s_upt, theta_upt, vpar_upt, t_upt = unperturbed_path_map.get_poincare_data()
                chis = self.chi(np.array(theta_upt[0]), np.array([2 * np.pi * i for i in range(len(theta_upt[0]))]))
                s_upt = np.array(s_upt[0])
                pa_data = np.column_stack((chis,s_upt))
                #pa_data = pa_data[pa_data[:, 0].argsort()]
                pa_data[:,0] = np.mod(pa_data[:,0], (2 * np.pi))
                pa_data = pa_data[pa_data[:, 0].argsort()]
                if i>0:
                    if lines[i][0] == lines[i-1][0]:
                        print('second zero activated', flush=True)
                        ax.plot(pa_data[:,0], pa_data[:,1], lw=5,color=color)
                        continue
                ax.plot(pa_data[:,0], pa_data[:,1], label=r'$\ell$='+f'{ell}',lw=5, color=color)
            ax.legend()
        fig.tight_layout()
        fig.savefig(filename + ".png", dpi = 400)

        # convergence plot - change in DA with number of transit evaluations
        # histogram of final DA values
        if self.DA_poinc and self.nconvergence_points > 1:
            fig, ax2 = plt.subplots(1, 1)
            ax2.set_ylabel(r"Digit Accuracy")
            ax2.set_xlabel(r"Toroidal Periods")

            for itrj in s_itrj_map:
                ax2.plot(
                    self.DA_times[itrj],
                    self.DA_all[itrj],
                    color=cmap_s((s_itrj_map[itrj] - min_s) / (max_s - min_s)),
                    alpha=0.75,
                    label=f"{s_itrj_map[itrj]}",
                )
            norm = plt.Normalize(min(s_lst_true), max(s_lst_true))
            fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap_s),
                ax=ax2,
                orientation="vertical",
                label="$s$",
            )

            fig.tight_layout()
            plt.savefig(filename+"_convergence.png", dpi=300)
            plt.clf()
            for elem in final_DAs:
                if not np.isnan(elem):
                    final_DAs.remove(elem)
            plt.hist(final_DAs)
            plt.xlabel("Digit Accuracy")
            plt.title("Distribution of Digit Accuracy")
            plt.tight_layout()
            plt.savefig(filename + "_DA_histogram.png")
        return ax, lines_2
    def get_poincare_data(self):
        """
        Return the Poincare map data.

        Returns:
            s_all, chis_all, etas_all, vpars_all, t_all : Lists of trajectory data.
        """
        return (
            self.s_all,
            self.chis_all,
            self.etas_all,
            self.vpars_all,
            self.t_all,
            self.DA_all,
            self.DA_times,
        )
    

class PassingPerturbedPetaPoincare:
    def __init__(
        self,
        saw,
        sign_vpar,
        mass,
        charge,
        helicity_M,
        helicity_N,
        helicity_Mp=None,
        helicity_Np=None,
        Eprime=None,
        mu=None,
        Ekin=None,
        p0=None,
        lam=None,
        ns_poinc=None,
        nchi_poinc=None,
        DA_poinc=False,
        nconvergence_points=None,
        s_init=None,
        chis_init=None,
        Nmaps=500,
        comm=None,
        tmax=1e-2,
        solver_options=None,
    ):
        """
        Initialize the PassingPerturbedPoincare class, which computes the
        Poincare return map for passing particles in a ShearAlfvenHarmonic
        magnetic field.

        The field strength contours are assumed to have helicity (M,N) in
        Boozer coordinates such that the field strength can be expressed as
        B(s,chi), where chi = M*theta - N*zeta is the helical angle.
        The mapping coordinate, eta, is chosen based on the helicity of the
        field strength contours: if M = 0 (e.g., QP or OP), then eta=theta is
        used, if N = 0 (e.g., QA/OA, QH/OA), then eta = zeta is used.

        The ShearAlfvenHarmonic can then be expressed in terms of the mapping
        coordinates with phase, m'*chi - n'*eta + omega * t.

        The map is evaluated by integrating the guiding center equations from
        the eta - omega/n' * t = 0 plane until the trajectory returns to the
        same plane.

        The map is well-defined (i.e., trajectories don't cross in the
        (s,chi=M*theta - N*zeta) plane) if the unperturbed field is
        quasisymmetric with helicity (M,N). However, the map can still be
        computed for a non-quasisymmetric field, but the trajectories may
        cross in the (s,chi) plane.

        The constants of motion are the magnetic moment, mu = vperp^2/(2 B),
        and the shifted energy, Eprime = n' * E - omega * p_eta, E is the
        total energy, and p_eta is the canonical momentum.

        These constants of motion can be prescribed directly with the Eprime
        and mu parameters. Alternatively, they can be computed from the
        prescribed pitch-angle variable, lam = vperp^2/(v^2 B), total
        unperturbed kinetic energy, Ekin, and a given point p0 in Boozer
        coordinates.

        Args:
            saw : An instance of ShearAlfvenHarmonic or ShearAlfvenWavesSuperposition.
            sign_vpar : Sign of the parallel velocity, either -1 or +1.
            mass : Mass of the particle.
            charge : Charge of the particle.
            helicity_M : Poloidal helicity of the magnetic field.
            helicity_N : Toroidal helicity of the magnetic field.
            helicity_Mp : Poloidal helicity of the phase variable eta.
            Defaults to None. If no value is given, Mp and Np are determined
            by field helicity.
            helicity_Np : Toroidal helicity of the phase variable eta.
            Defaults to None. If no value is given, Mp and Np are determined by field
            helicity.
            Eprime: Shifted energy, Eprime = n' * E - omega * p_eta.
            mu: Magnetic moment, mu = vperp^2/(2 B).
            Ekin: Total unperturbed kinetic energy of the particle, used to
                compute Eprime if not provided.
            p0: Initial point in Boozer coordinates for evaluation of Eprime.
            lam: Pitch angle variable, lambda = vperp^2/(v^2 B), used to
                compute mu if not provided.
            s_init : List of initial s coordinates for the Poincare map.
                (default: None, ns_poinc is used instead)
            chis_init : List of initial chi coordinates for the Poincare map.
                (default: None, nchi_poinc is used instead)
            ns_poinc : Number of initial conditions in s for Poincare plot
                (default: 120).
            nchi_poinc : Number of initial conditions in chi for Poincare plot
                (default: 2).
            Nmaps : Number of Poincare return maps to compute for each initial
                condition (default: 500).
            DA_poinc : Boolean value indicating whether chaos detection is desired
                (default: False)
            nconvergence_points : Integer value indicating the number of Weighted
                Birkhoff Average evaluations to assess convergence.
            comm : MPI communicator for parallel execution (default: None).
            tmax : Maximum integration time for each segment of the Poincare
                map (default: 1e-2 s).
            solver_options : Dictionary of options to pass to the ODE solver
                (default: {}).
        """
        if solver_options is None:
            solver_options = {}
        if not isinstance(saw, ShearAlfvenHarmonic) and not isinstance(
            saw, ShearAlfvenWavesSuperposition
        ):
            raise TypeError(
                "Expected saw to be an instance of ShearAlfvenHarmonic "
                "or ShearAlfvenWavesSuperposition"
            )
        if not isinstance(saw, ShearAlfvenHarmonic):
            dominant_saw = saw[0]
            raise Warning(
                "Expected saw to be an instance of ShearAlfvenHarmonic - "
                "Perturbed Energy Invariant may not be valid."
            )
        else:
            dominant_saw = None
        if sign_vpar not in [-1, 1]:
            raise ValueError("sign_vpar should be either -1 or +1")

        self.saw = saw
        self.B0 = saw.B0
        self.helicity_M = helicity_M
        self.helicity_N = helicity_N
        self._set_helicity_Np_Mp(helicity_Np, helicity_Mp)
        self.mass = mass
        self.charge = charge
        self.sign_vpar = sign_vpar

        self.DA_poinc = DA_poinc
        if DA_poinc:
            if nconvergence_points is None:
                self.nconvergence_points = 1
                self.WBA_transit_steps = [Nmaps - 1]
            else:
                self.nconvergence_points = nconvergence_points
                # set list of transits for each WBA evaluation
                transits_per_average = int(Nmaps / (nconvergence_points))
                self.WBA_transit_steps = np.linspace(
                    transits_per_average, Nmaps - 1, num=nconvergence_points, dtype=int
                ).tolist()
        else:
            self.nconvergence_points = 1
            self.WBA_transit_steps = 0

        if s_init is not None and chis_init is not None:
            self.s_init = s_init
            self.chis_init = chis_init
        else:
            if ns_poinc is None:
                ns_poinc = 120
            if nchi_poinc is None:
                nchi_poinc = 2
            self.ns_poinc = ns_poinc
            self.nchi_poinc = nchi_poinc
        self.Nmaps = Nmaps
        self.comm = comm
        self.tmax = tmax
        self.solver_options = solver_options

        # if using a ShearAlfvenWavesSuperposition, use the test_saw for
        # computing Eprime, Phin, Phim, and omega as the largest mode
        # this does not produce a poincare plot in the strict sense,
        # but you will be unable to visualize surfaces
        if dominant_saw is None:
            self.Phin = saw.Phin
            self.Phim = saw.Phim
            self.omega = saw.omega
        else:
            self.Phin = dominant_saw.Phin
            self.Phim = dominant_saw.Phim
            self.omega = dominant_saw.omega

        self.nprime = (self.Phim * self.helicity_N - self.Phin * self.helicity_M) / (
            self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        )

        if Eprime is not None and mu is not None:
            self.Eprime = Eprime
            self.mu = mu
            self.Ekin = None
        elif Ekin is not None and lam is not None and p0 is not None:
            """
            Compute unperturbed values of mu, p_eta, and Eprime from the given
            parameters.
            """
            v0 = np.sqrt(2 * Ekin / mass)  # Total velocity from kinetic energy
            self.mu = 0.5 * lam * v0**2  # mu = vperp^2/(2 B)
            self.Ekin = Ekin  # Total kinetic energy
            saw.B0.set_points(p0)
            modB = saw.B0.modB()[0, 0]
            if 1 - lam * modB < 0:
                raise ValueError(
                    "Invalid parameter p0: 1 - lambda * modB must be non-negative."
                )
            vpar = sign_vpar * v0 * np.sqrt(1 - lam * modB)  # Parallel velocity
            Peta0 = compute_peta(
                saw.B0,
                p0,
                vpar,
                mass,
                charge,
                helicity_M,
                helicity_N,
            )
            self.Eprime = self.nprime * Ekin - self.omega * Peta0
        else:
            raise ValueError(
                "Either Eprime and mu must be provided, or Ekin, lam, and p0 "
                "must be provided."
            )

        # Initialize the passing map
        self.s_init, self.chis_init, self.vpars_init = self.initialize_passing_map()
        np_chis = np.array(self.chis_init)
        np_etas = np.zeros_like(np_chis)

        theta, zeta = self.chi_eta_to_theta_zeta(np_chis, np_etas)
        point = np.zeros((len(self.s_init), 4))
        point[:, 0] = self.s_init
        point[:, 1] = theta
        point[:, 2] = zeta
        vp = np.array(self.vpars_init)
        vp = vp.reshape(-1)
        self.Petas_init = compute_peta(
                saw,
                point,
                vp,
                mass,
                charge,
                helicity_M,
                helicity_N,
            )


        # If Ekin is not provided, compute it from the initial parallel velocity
        # this is only used for computing maximum time step in the ODE solver
        if self.Ekin is None:
            self.Ekin = 0.5 * self.mass * self.vpars_init[0] ** 2

        (
            self.s_all,
            self.chis_all,
            self.etas_all,
            self.vpars_all,
            self.t_all,
            self.Peta_all,
            self.DA_all,
            self.DA_times,
            self.thetas_all,
            self.zetas_all
        ) = self.compute_passing_map()
        self.final_Petas = [self.Peta_all[i][-1]for i in range(len(self.Peta_all)) if len(self.Peta_all[i])>self.Nmaps-3]

    def _set_helicity_Np_Mp(self, helicity_Np, helicity_Mp):
        """
        Sets helicity of the phase variable eta based on user inputs.
        """
        if not ((helicity_Np is None) or (helicity_Mp is None)):
            # User specified both helicities
            self.helicity_Mp = helicity_Mp
            self.helicity_Np = helicity_Np
            return

        if (helicity_Np is None) and (helicity_Mp is None):
            # User did not specify helicity, choose default
            if self.helicity_M == 0:
                # modB contours close poloidally,
                # so can use theta as mapping coordinate
                self.helicity_Mp = 1
                self.helicity_Np = 0
            else:
                # use zeta as mapping coordinate
                self.helicity_Mp = 0
                self.helicity_Np = -1
            return

        raise ValueError(
            f"User must either specify both helicity_Np and helicity_Mp or leave both "
            f"of them None. Currently {helicity_Np=} while {helicity_Mp=}."
        )

    def initialize_passing_map(self):
        """
        Compute vpar given (s,chi) such that Eprime = Eprime0
        """

        def vpar_func_perturbed(s, chi):
            # Choose initial conditions on the eta = 0 plane
            theta, zeta = self.chi_eta_to_theta_zeta(chi, 0)
            point = np.zeros((1, 4))  # initialize with t = 0
            point[0, 0] = s
            point[0, 1] = theta
            point[0, 2] = zeta
            self.saw.set_points(point)
            modB = self.B0.modB()[0, 0]
            G = self.B0.G()[0, 0]
            I = self.B0.I()[0, 0]
            psi = self.B0.psi0 * s
            psip = self.B0.psip()[0, 0]
            Phi = self.saw.Phi()[0, 0]
            alpha = self.saw.alpha()[0, 0]
            denom = (
                self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
            )  # - 1 in QA
            d_peta_d_vpar = (
                -((self.helicity_M * G + self.helicity_N * I) * (self.mass / modB))
                / denom
            )  # G m/ modB in QA
            d_E_d_vpar2 = 0.5 * self.mass
            a = self.nprime * d_E_d_vpar2  # Coefficient of vpar^2
            b = -self.omega * d_peta_d_vpar  # Coefficient of vpar
            # Constant term
            c = (
                self.nprime * (self.mass * self.mu * modB + self.charge * Phi)
                + self.omega
                * (
                    (self.helicity_M * G + self.helicity_N * I) * self.charge * alpha
                    + self.charge * (self.helicity_N * psi - self.helicity_M * psip)
                )
                / denom
                - self.Eprime
            )
            if (b**2 - 4 * a * c) < 0:
                raise RuntimeError(
                    "No solution for vpar found! Check the parameters and "
                    "initial conditions."
                )
            elif a != 0:
                return (-b + self.sign_vpar * np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            else:
                return (-c / b) * self.sign_vpar

        # Create mesh grid if not provided directly
        if not hasattr(self, "s_init") or not hasattr(self, "chis_init"):
            s = np.linspace(0, 1, self.ns_poinc + 1, endpoint=False)[1::]
            chis = np.linspace(0, 2 * np.pi, self.nchi_poinc)
            s, chis = np.meshgrid(s, chis)
            s = s.flatten()
            chis = chis.flatten()
        else:
            s = self.s_init
            chis = self.chis_init

        first, last = parallel_loop_bounds(self.comm, len(s))
        # For each point, find value of vpar such that lambda = vperp^2/(v^2 B)
        s_init = []
        chis_init = []
        vpars_init = []
        for i in range(first, last):
            try:
                vpar = vpar_func_perturbed(s[i], chis[i])
                s_init.append(s[i])
                chis_init.append(chis[i])
                vpars_init.append(vpar)
            except RuntimeError:
                continue

        if self.comm is not None:
            s_init = [i for o in self.comm.allgather(s_init) for i in o]
            chis_init = [i for o in self.comm.allgather(chis_init) for i in o]
            vpars_init = [i for o in self.comm.allgather(vpars_init) for i in o]

        return s_init, chis_init, vpars_init

    def chi(self, theta, zeta):
        r"""
        Compute the helical angle chi = M*theta - N*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            chi : The helical angle.
        """
        return self.helicity_M * theta - self.helicity_N * zeta

    def eta(self, theta, zeta):
        r"""
        Compute the mapping angle eta = Mp*theta - Np*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            eta : The mapping angle.
        """
        return self.helicity_Mp * theta - self.helicity_Np * zeta

    def chi_eta_to_theta_zeta(self, chi, eta):
        r"""
        Convert helical angles (chi, eta) to (theta, zeta).

        Args:
            chi : Helical angle chi.
            eta : Mapping angle eta.
        Returns:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        """
        denom = self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        theta = (self.helicity_Np * chi - self.helicity_N * eta) / denom
        zeta = (self.helicity_Mp * chi - self.helicity_M * eta) / denom

        return theta, zeta

    def passing_map(self, point, t, eta):
        r"""
        Integrates the GC equations from the provided point on the eta -
        omega/n' * t plane to the next intersection with this plane. An
        assumption is made that the particle is passing, so a RuntimeError
        is raised if the particle mirrors.

        Since the phase of the ShearAlfvenWave depends on time, the initial
        time, t, is passed as an argument.

        Args:
            point : A numpy array of shape (3,) containing the initial
                coordinates (s,chi,eta).
            t : Initial time at which the map is evaluated
        Returns:
            point : A numpy array of shape (3,) containing the coordinates
                (s,chi,eta).
            time : The time at which the trajectory returns to the eta -
                omega/n' * t plane.
            peta : Timeseries of the canonical momentum p_eta along the trajectory.
        """
        phase = self.omega * t
        self.saw.phase = phase
        theta, zeta = self.chi_eta_to_theta_zeta(point[1], eta)
        points = np.zeros((1, 3))
        points[:, 0] = point[0]
        points[:, 1] = theta
        points[:, 2] = zeta

        if self.helicity_M != 0:
            phases = [zeta * self.nprime]
            n_zetas = [self.nprime]
            m_thetas = [0]
            omegas = [self.omega]
        else:
            phases = [theta * self.nprime]
            n_zetas = [0]
            m_thetas = [self.nprime]
            omegas = [self.omega]
        try:
            res_tys, res_hits = trace_particles_boozer_perturbed(
                perturbed_field=self.saw,
                stz_inits=points,
                parallel_speeds=[point[2]],
                mus=[self.mu],
                tmax=self.tmax,
                mass=self.mass,
                charge=self.charge,
                phases=phases,
                n_zetas=n_zetas,
                m_thetas=m_thetas,
                omegas=omegas,
                vpars=[0],
                axis = 0,
                ODE_solver="dormand_prince",
                stopping_criteria=[
                    MinToroidalFluxStoppingCriterion(0.001),
                    MaxToroidalFluxStoppingCriterion(1.0),
                ],
                forget_exact_path=True,
                vpars_stop=True,
                phases_stop=True,
                **self.solver_options,
            )
        except: 
            print(f"Error integrating guiding center equations at {point=}, {t=}")
            raise RuntimeError(f"Error integrating guiding center equations at {point=}, {t=}, {eta=}")
        if len(res_hits[0]) == 0:
            raise RuntimeError("No stopping criterion reached in passing_map.")

        res_hit = res_hits[0][0, :]  # Only check the first hit or stopping criterion

        # Check that the phases plane was hit (index 0 for first phase)
        if res_hit[1] == 0:
            point[0] = res_hit[2]
            point[1] = self.chi(res_hit[3], res_hit[4])
            point[2] = res_hit[5]
        else:
            raise RuntimeError("Alternative stopping criterion reached in passing_map.")

        if not self.DA_poinc:
            return point, res_hit[0] + t, self.eta(res_hit[3], res_hit[4])
        else:
            # define trajectories
            time_momentum = res_tys[0][:, 0]
            s_path = res_tys[0][:, 1]
            theta_path = res_tys[0][:, 2]
            zeta_path = res_tys[0][:, 3]
            vpar_path = res_tys[0][:, 4]

            # set points for trajectories:
            points_traj = np.zeros((len(time_momentum), 4))
            points_traj[:, 0] = s_path
            points_traj[:, 1] = theta_path
            points_traj[:, 2] = zeta_path
            points_traj[:, 3] = time_momentum

            Peta = compute_peta(
                self.saw,
                points_traj,
                vpar_path,
                self.mass,
                self.charge,
                self.helicity_M,
                self.helicity_N,
                helicity_Mp=self.helicity_Mp,
                helicity_Np=self.helicity_Np,
            )
            Peta = np.column_stack((time_momentum, Peta))
            return point, res_hit[0] + t, self.eta(res_hit[3], res_hit[4]), Peta, (res_hit[3], res_hit[4])

    def compute_passing_map(self):
        r"""
        Evaluates the passing Poincare return map for the initialized particle
        positions.
        """
        Ntrj = len(self.s_init)

        s_all = []
        chis_all = []
        etas_all = []
        vpars_all = []
        Peta_all = []
        t_all = []
        DA_all = []
        DA_times = []
        thetas_all = []
        zetas_all = []
        first, last = parallel_loop_bounds(self.comm, Ntrj)
        for itrj in range(first, last):
            tr = [self.s_init[itrj], self.chis_init[itrj], self.vpars_init[itrj]]
            s_traj = [tr[0]]
            chis_traj = [tr[1]]
            t, z = self.chi_eta_to_theta_zeta(self.chis_init[itrj],0)
            thetas_traj = [t]
            zetas_traj = [z]
            eta_traj = [0]
            vpars_traj = [tr[2][0]]
            Peta_traj = [self.Petas_init[itrj]]
            t_traj = [0]

            particle_DAs = []
            particle_DA_times = []
            for jj in range(self.Nmaps):
                try:
                    if self.DA_poinc:
                        if jj == 0:
                            tr, time, eta, Peta, angles = self.passing_map(
                                tr, t_traj[-1], eta_traj[-1]
                            )
                        else:
                            tr, time, eta, Peta_iter, angles = self.passing_map(
                                tr, t_traj[-1], eta_traj[-1]
                            )
                            Peta_iter[:, 0] += Peta[-1, 0]
                            Peta = np.vstack((Peta, Peta_iter[1:, :]))
                    else:
                        tr, time, eta = self.passing_map(tr, t_traj[-1], eta_traj[-1])
                    s_traj.append(tr[0])
                    chis_traj.append(tr[1])
                    vpars_traj.append(tr[2])
                    Peta_traj.append(Peta[-1, 1])
                    t_traj.append(time)
                    eta_traj.append(eta)
                    t, z = angles[0], angles[1]
                    thetas_traj.append(t)
                    zetas_traj.append(z)
                    if self.DA_poinc and jj in self.WBA_transit_steps:
                        time_at_evaluation, DA_at_evaluation = return_DA(Peta)
                        particle_DAs.append(DA_at_evaluation)
                        particle_DA_times.append(jj)
                except RuntimeError:
                    if self.DA_poinc:
                        particle_DAs.append(np.nan)
                        particle_DA_times.append(np.nan)
                    break
            DA_all.append(particle_DAs)
            DA_times.append(particle_DA_times)
            s_all.append(s_traj)
            Peta_all.append(Peta_traj)
            chis_all.append(chis_traj)
            etas_all.append(eta_traj)
            thetas_all.append(thetas_traj)
            zetas_all.append(zetas_traj)
            vpars_all.append(vpars_traj)
            t_all.append(t_traj)

        if self.comm is not None:
            s_all = [i for o in self.comm.allgather(s_all) for i in o]
            chis_all = [i for o in self.comm.allgather(chis_all) for i in o]
            etas_all = [i for o in self.comm.allgather(etas_all) for i in o]
            vpars_all = [i for o in self.comm.allgather(vpars_all) for i in o]
            t_all = [i for o in self.comm.allgather(t_all) for i in o]
            Peta_all = [i for o in self.comm.allgather(Peta_all) for i in o]
            DA_all = [i for o in self.comm.allgather(DA_all) for i in o]
            thetas_all = [i for o in self.comm.allgather(thetas_all) for i in o]
            zetas_all = [i for o in self.comm.allgather(zetas_all) for i in o]
            DA_times = [i for o in self.comm.allgather(DA_times) for i in o]

        return s_all, chis_all, etas_all, vpars_all, t_all, Peta_all, DA_all, DA_times, thetas_all, zetas_all
    
    def compute_frequencies(self):

        if "axis" in self.solver_options and self.solver_options["axis"] != 0:
            raise ValueError(
                'ODE solver must integrate with solver_options["axis"]=0 to '
                "compute passing frequencies."
            )

        self.saw.set_points(np.array([[1], [0], [0], [0]]).T)
        sign_G = np.sign(self.saw.B0.G()[0])

        omega_theta = []
        omega_zeta = []
        init_s = []
        omega_zeta_exp = []
        diff_eprime = []
        std_eprime = []
        Peta_prof = []
        ds_all = []

        for s_traj, chis_traj, etas_traj, _vpar_traj, Peta_traj, t_traj, thetas_traj, zetas_traj in zip(
            self.s_all, self.chis_all, self.etas_all, self.vpars_all, self.Peta_all, self.t_all, self.thetas_all, self.zetas_all
        ):
            if (
                len(s_traj) < 2
            ):  # Need at least one full Poincare return maps to compute frequency
                continue

            theta, zeta = self.chi_eta_to_theta_zeta(np.array(chis_traj), np.array(etas_traj))

            delta_theta = np.array(thetas_traj[1:]) - np.array(thetas_traj[0:-1])
            delta_s = np.array(s_traj[1:]) - np.array(s_traj[0:-1])
            delta_t = t_traj[1::]
            delta_zeta = 2 * np.pi * self.sign_vpar * sign_G
            delta_zeta_exp = np.array(zetas_traj[1:]) - np.array(zetas_traj[0:-1])

            points = np.zeros((len(s_traj), 4))
            points[:, 0] = s_traj
            points[:, 1] = theta
            points[:, 2] = zeta

            trajectory_eprime = compute_Eprime(self.saw, points, _vpar_traj, self.mu, self.mass, self.charge, self.helicity_M, self.helicity_N)
            eprime_std = np.std(trajectory_eprime)

            eprime_pert = (np.mean(trajectory_eprime) - self.Eprime)/eprime_std

            # Average over wells along one field line
            freq_theta = np.mean(delta_theta) / np.mean(delta_t)
            freq_zeta = delta_zeta / np.mean(delta_t)
            freq_zeta_exp = np.mean(np.mod(delta_zeta_exp, 2 * np.pi)) / np.mean(delta_t)

            diff_eprime.append(eprime_pert)
            std_eprime.append(eprime_std/np.abs(np.mean(trajectory_eprime)))

            omega_theta.append(freq_theta)
            omega_zeta.append(freq_zeta)
            omega_zeta_exp.append(freq_zeta_exp)
            init_s.append(np.mean(s_traj))
            ds_all.append(np.mean(delta_s))
            Peta_prof.append(Peta_traj[-1])

        omega_theta = np.array(omega_theta)
        omega_zeta = np.array(omega_zeta)
        omega_zeta_exp = np.array(omega_zeta_exp)
        std_eprime = np.array(std_eprime)
        ds_all = np.array(ds_all)
        diff_eprime = np.array(diff_eprime)
        Peta_prof = np.array(Peta_prof)

        init_s = np.array(init_s)

        s_prof = np.unique(init_s)
        omega_theta_prof = np.zeros((len(s_prof),))
        omega_zeta_prof = np.zeros((len(s_prof),))
        omega_zeta_exp_prof = np.zeros((len(s_prof),))
        ds_all_prof = np.zeros((len(s_prof),))
        diff_eprime_prof = np.zeros((len(s_prof),))
        std_eprime_prof = np.zeros((len(s_prof),))
        Peta_s_prof = np.zeros((len(s_prof),))

        # Average over field-line label
        for i, s in enumerate(s_prof):
            indicies = omega_theta[np.where(init_s == s)]
            omega_theta_prof[i] = np.mean(omega_theta[np.where(init_s == s)])
            omega_zeta_prof[i] = np.mean(omega_zeta[np.where(init_s == s)])
            omega_zeta_exp_prof[i] = np.mean(omega_zeta_exp[np.where(init_s == s)])
            ds_all_prof[i] = np.mean(ds_all[np.where(init_s == s)])
            diff_eprime_prof[i] = np.mean(diff_eprime[np.where(init_s == s)])
            std_eprime_prof[i] = np.mean(std_eprime[np.where(init_s == s)])
            Peta_s_prof[i] = np.mean(Peta_prof[np.where(init_s == s)])

        return omega_theta_prof, omega_zeta_prof, s_prof, omega_zeta_exp_prof, ds_all_prof, diff_eprime_prof, std_eprime_prof, Peta_s_prof

    def plot_poincare(
        self,
        ax=None,
        filename="passing_poincare.pdf",
        convergence_test_indicies=None,
        DA_max=7,
        lines = None,
        ylims=(-0.5e-18, 0.5e-18),
    ):
        r"""
        Plot the passing Poincare map and save to a file. It is recommended to only
        call this function on MPI rank 0.
        Args:
            ax : Matplotlib axis to plot on. If None, a new figure and axis are
                 created.
            filename : Name of the file to save the plot
                       (default: 'passing_poincare.pdf').
            convergence_test_indicies : Indices of initial conditions to show
            in convergence plot.
            DA_max : Maximum value of Digit Accuracy to show on colorbar
            ylims : Tuple specifying y-axis limits for the Poincare plot.
        Returns:
            ax : The Matplotlib axis containing the plot.
        """

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable

        mpl.use("Agg")  # Don't use interactive backend

        try:
            import cmcrameri.cm as cmc  # noqa: F401

            cmap = "cmc.managua"
        except ImportError:
            cmap = "viridis"

        star_ICs = False

        if convergence_test_indicies is None:
            convergence_test_indicies = list(range(len(self.Peta_all)))
        else:
            star_ICs = True

        if self.DA_poinc and self.nconvergence_points > 1:
    
            min_Peta = min(self.Petas_init)
            max_Peta = max(self.Petas_init)
            cmap_s = mpl.colormaps["copper"].resampled(len(self.Petas_init) ** 2)

        if ax is None:
            fig, ax = plt.subplots()

        def normalize(numbers):
            if not numbers:
                return []
            min_val, max_val = 0, DA_max
            normalized_numbers = [(x - min_val) / (max_val - min_val) for x in numbers]
            return normalized_numbers

        if self.DA_poinc:
            final_DAs = []
            # retrieve final DA for each trajectory if the particle is not lost
            # put it into a list
            for elem in self.DA_all:
                if len(elem) == self.nconvergence_points:
                    final_DAs.append(elem[self.nconvergence_points - 1])
                else:
                    final_DAs.append(np.nan)
            # normalized DA values for colormap
            DA_norm_all = normalize(final_DAs)
            cmap_object = mpl.colormaps[cmap].resampled(len(self.DA_all) ** 2)
        ax.set_ylabel(r"$\chi$")
        ax.set_xlabel(r"$P_\eta$")
        ax.set_ylim([0, 2 * np.pi])
        ax.set_xlim(min(self.final_Petas), max(self.final_Petas))

        for i in range(len(self.chis_all)):
            if self.DA_poinc:
                ax.scatter(
                    self.Peta_all[i],
                    np.mod(self.chis_all[i], 2 * np.pi),
                    marker="o",
                    s=0.75,
                    c=cmap_object(DA_norm_all[i]),
                    edgecolors="none",
                )
            else:
                ax.scatter(
                    self.Peta_all[i],
                    np.mod(self.chis_all[i], 2 * np.pi),
                    marker="o",
                    s=0.75,
                    edgecolors="none",
                )

        if star_ICs:
            # scatter initial conditions observed in convergence plot onto poincare plot
            for i in range(len(self.Petas_init)):
                s_norm = (self.Petas_init[i] - min_Peta) / (max_Peta - min_Peta)
                ax.scatter(
                    self.Peta_all[i][0],
                    np.mod(self.chis_all[i][0], 2 * np.pi),
                    marker="*",
                    s=25,
                    color=cmap_s(s_norm),
                    edgecolors="magenta",
                )

        if self.DA_poinc:
            # make colorbar for DA values
            max_val = DA_max
            norm = plt.Normalize(0, max_val)
            fig.colorbar(
                ScalarMappable(norm=norm, cmap=mpl.colormaps[cmap]),
                ax=ax,
                orientation="vertical",
                label="Digit Accuracy",
            )
        
        plt.tight_layout()
        plt.savefig(filename + ".png", dpi = 400)

        # convergence plot - change in DA with number of transit evaluations
        # histogram of final DA values
        if self.DA_poinc and self.nconvergence_points > 1:
            fig, ax2 = plt.subplots(1, 1)
            ax2.set_ylabel(r"Digit Accuracy")
            ax2.set_xlabel(r"Toroidal Periods")

            for itrj in range(len(self.Petas_init)):
                ax2.plot(
                    self.DA_times[itrj],
                    self.DA_all[itrj],
                    color=cmap_s((self.Petas_init[itrj] - min_Peta) / (max_Peta - min_Peta)),
                    alpha=0.75,
                    label=f"{self.Petas_init[itrj]}",
                )
            norm = plt.Normalize(min(self.Petas_init), max(self.Petas_init))
            fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap_s),
                ax=ax,
                orientation="vertical",
                label="$s$",
            )

            fig.tight_layout()
            plt.savefig(filename+"_convergence.png", dpi=300)

        plt.clf()
        final_Petas = [self.Peta_all[i][-1]for i in range(len(self.Peta_all)) if len(self.Peta_all[i])>self.Nmaps-3]
        plt.hist(self.Petas_init, alpha=0.5,density=True,bins=30, label='init')
        plt.hist(final_Petas, alpha=0.5, density=True,bins=30, label='final')
        plt.xlabel(r"$P_\eta$")
        plt.ylabel("Distribution")
        plt.legend()
        fig.tight_layout()
        plt.savefig(filename+"_Peta_hist.png", dpi=300)
        return ax
    """
    
    """
    def get_poincare_data(self):
        """
        Return the Poincare map data.

        Returns:
            s_all, chis_all, etas_all, vpars_all, t_all : Lists of trajectory data.
        """
        return (
            self.s_all,
            self.chis_all,
            self.etas_all,
            self.vpars_all,
            self.t_all,
            self.DA_all,
            self.DA_times,
        )

class MapPhaseSpace:
    def __init__(
        self,
        saw,
        Phin_max,
        Phim_max,
        omega,
        mass,
        charge,
        Ekin,
        helicity_N,
        helicity_M,
        helicity_Mp=None,
        helicity_Np=None,
        ns_points=20,
        particles_per_surface = 20,
        nlambda_points=20,
        randomize_particles = False,
        number_of_particles = 10000,
        initial_conditions = None,
        initial_vpar = None,
        initial_mu_per_particle = None,
        Eprime = None,
        Eprime_slice = False,
        min_timestep=1e-6,
        s_lims=[0.01,0.95],
        diffusion = False,
        mean=True,
        comm=None,
        tmax=1e-2,
        tol=1e-10,
        skip = [],
        solver_options={},
        unperturbed=False,
        savedata=[True, 'DATA/'],
        nconvergence_points = 1
    ):
        # @TODO: add user checks for saw
        # @TODO: Eprime for random init 
        # @TODO: add option to feed in ICs directly
        # @TODO: add convergence points support
        # @TODO: implement saving 


        # set field parameters
        self.saw = saw
        self.B0 = saw.B0
        self.helicity_M = helicity_M
        self.helicity_N = helicity_N

        if helicity_Mp is None and helicity_Np is None:
            # If modB contours close poloidally, then use theta as mapping coordinate
            if helicity_M == 0:
                helicity_Mp = 1
                helicity_Np = 0
            # Otherwise, use zeta as mapping coordinate
            else:
                helicity_Mp = 0
                helicity_Np = -1
        else:
            if (helicity_Mp * helicity_N) == (helicity_Np * helicity_M):
                raise ValueError(
                    "Chosen helicities (N, M, N', M') do not create a well "
                    "defined Jacobian."
                )
            
        self.helicity_Mp = helicity_Mp
        self.helicity_Np = helicity_Np
        #self.Phihat = max(saw.Phihat_value_or_tuple)

        self.Phim = Phim_max
        self.Phin = Phin_max
        self.Phimp = (self.Phim * self.helicity_Np - self.Phin * self.helicity_Mp)/( self.helicity_M * self.helicity_Np - self.helicity_Mp * self.helicity_N)
        self.Phinp = (self.Phim * self.helicity_Np - self.Phin * self.helicity_Mp)/( self.helicity_M * self.helicity_Np - self.helicity_Mp * self.helicity_N)
        
        # Compute the invariant Eprime
        if unperturbed: 
            self.nprime = ( helicity_N - helicity_M) / (
                helicity_Np * helicity_M - helicity_N * helicity_Mp
            )
        else:
            self.nprime = (self.Phim * helicity_N - self.Phin * helicity_M) / (
                helicity_Np * helicity_M - helicity_N * helicity_Mp
            )

        self.omega = omega
        self.omegan = self.omega / self.nprime

        self.tol = tol

        # set timing parameters
        self.tmax = tmax
        self.min_timestep = min_timestep
        
        self.Ekin = Ekin
        self.vtotal = np.sqrt(2*self.Ekin/mass)
        self.mass = mass
        self.charge = charge
        self.Eprime_slice = Eprime_slice
        self.Eprime = Eprime

        # set communicator parameters
        self.comm = comm
        self.verbose=False
        if self.comm==None:
            self.verbose = True
        elif self.comm.rank==0:
            self.verbose = True

        self.solver_options = solver_options

        def min_volumemodB():
            resolution = 100 
            points = np.zeros((resolution * resolution,3))

            for surface in np.linspace(0,0.99,resolution):
                if surface == 0:
                    points = initialize_position_uniform_surf(self.B0,resolution ,surface)
                else:
                    sampled_surface = initialize_position_uniform_surf(self.B0, resolution ,surface)
                    points = np.concatenate((points,sampled_surface), axis=0)
                    
            self.B0.set_points(points)
            modB = self.B0.modB()[:,0]
            return np.min(modB)
        
        self.min_volmodB=min_volumemodB()

        # plotting settings
        self.mean = mean
        self.savedata = savedata[0]
        self.savepath = savedata[1]
        self.convergence_points = nconvergence_points

        self.s_min = s_lims[0]
        self.s_max = s_lims[1]

        if initial_conditions is None: 
            self.randomize_particles = randomize_particles
            # instantiate ICs 
            if randomize_particles:
                self.nParticles = number_of_particles
                s, thetas, zetas, vpar, mu = self.instantiate_uniform_particles(self.nParticles)

            else:
                self.ns_points = ns_points
                self.particles_per_surface = particles_per_surface
                self.nlambda_points = nlambda_points
                self.nParticles = ns_points * particles_per_surface * nlambda_points
                s, thetas, zetas, vpar, mu = self.instantiate_gridded_particles()
        else:
            s = initial_conditions[:,0]
            thetas = initial_conditions[:,1]
            zetas = initial_conditions[:,2]
            if initial_vpar is not None and initial_mu_per_particle is not None:
                vpar = initial_vpar
                mu = initial_mu_per_particle
            else:
                raise ValueError("If providing initial conditions, must provide both initial_vpar and initial_mu_per_particle")

        # set parameters for convergence plot
        self.diffusion = diffusion
        expected_length = int(self.tmax/self.min_timestep)
        expected_step = int(expected_length/self.convergence_points)
        self.WBA_transit_steps = np.linspace(expected_step, expected_length - 1, num=nconvergence_points, dtype=int).tolist()
        self.convergence_plot = True if nconvergence_points>1 else False
        initial_point = np.zeros((len(s), 3))  # initialize with t = 0
        initial_point[:, 0] = s
        initial_point[:, 1] = thetas
        initial_point[:, 2] = zetas

        if self.savedata:
            self.IC_filepaths = {
                's0': self.savepath + '_s0.txt',
                'theta0': self.savepath + '_theta0.txt',
                'zeta0': self.savepath + '_zeta0.txt',
                'vpar0': self.savepath + '_vpar0.txt',
                'mu_per_mass': self.savepath + '_mu_per_mass.txt'
            }
            if diffusion:
                self.final_filepaths = {
                    'D': self.savepath + f'Deff.txt',
                    'wall_lost': self.savepath + f'wall_lost.txt',
                }
            else:
                self.final_filepaths = {
                    'D': self.savepath + f'DA.txt',
                    'wall_lost': self.savepath + f'wall_lost.txt',
                }

        self.equilibrium_lost = skip
        if self.check_filepaths(self,self.final_filepaths):
            self.DAs = np.loadtxt(self.final_filepaths['DA']).tolist()
            wall_lost = np.loadtxt(self.final_filepaths['wall_lost']).astype(int)
            self.wall_lost_indicies = wall_lost[:,0].tolist()
            self.wall_lost_times = wall_lost[:,1].tolist()
        else:
            self.s, self.thetas, self.zetas, self.vpar, self.mus, self.equilibrium_lost = self.remove_equilibrium_lost_particles(initial_point, vpar, mu)
            self.da_values, self.wall_lost, self.surfaces, self.pitch_angles = self.trace_particles()


    def check_filepaths(self,filepaths):
        return all([exists(fp) for fp in filepaths.values()])
    
    def chi(self, theta, zeta):
        r"""
        Compute the helical angle chi = M*theta - N*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            chi : The helical angle.
        """
        return self.helicity_M * theta - self.helicity_N * zeta
    
    def instantiate_uniform_particles(self, nParticles):
        tracing_points = initialize_position_uniform_vol(
            self.B0,
            nParticles,
            comm=self.comm,
            seed=None,
        )

        vpars_init = initialize_velocity_uniform(
            self.vtotal,
            nParticles,
            comm=self.comm,
            seed=None,
        )
        
        self.B0.set_points(tracing_points)
        modB = self.B0.modB()[:,0]
        mus_per_mass = (1/(2 * modB)) * (self.vtotal**2 - vpars_init**2)
        return tracing_points[:,0],tracing_points[:,1],tracing_points[:,2], vpars_init, mus_per_mass

    def chi_eta_to_theta_zeta(self, chi, eta):
        r"""
        Convert helical angles (chi, eta) to (theta, zeta).

        Args:
            chi : Helical angle chi.
            eta : Mapping angle eta.
        Returns:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        """
        denom = self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        theta = (self.helicity_Np * chi - self.helicity_N * eta) / denom
        zeta = (self.helicity_Mp * chi - self.helicity_M * eta) / denom

        return theta, zeta

    def vpar_func(self, s, theta, zeta, p_a, sgn):
        point = np.zeros((len(s), 3))
        point[:, 0] = s
        point[:, 1] = theta
        point[:, 2] = zeta
        self.B0.set_points(point)
        modB = self.B0.modB()[0, 0]
        if 1 - p_a * modB < 0:
            return np.nan
        else:
            return sgn * self.vtotal * np.sqrt(1 - p_a * modB)
    
    def vpar_func_perturbed(self, s, theta, zeta, mu, sgn):
        # Choose initial conditions on the eta = 0 plane
        point = np.zeros((1, 4))  # initialize with t = 0
        point[:, 0] = s
        point[:, 1] = theta
        point[:, 2] = zeta
        self.saw.set_points(point)
        modB = self.B0.modB()[:, 0]
        G = self.B0.G()[:, 0]
        I = self.B0.I()[:, 0]
        psi = self.B0.psi0 * s
        psip = self.B0.psip()[:, 0]
        Phi = self.saw.Phi()[:, 0]
        alpha = self.saw.alpha()[:, 0]
        denom = (
            self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        )  # - 1 in QA
        d_peta_d_vpar = (
            -((self.helicity_M * G + self.helicity_N * I) * (self.mass / modB))
            / denom
        )  # G m/ modB in QA
        d_E_d_vpar2 = 0.5 * self.mass
        a = self.nprime * d_E_d_vpar2  # Coefficient of vpar^2
        b = -self.omega * d_peta_d_vpar  # Coefficient of vpar
        # Constant term
        c = (
            self.nprime * (self.mass * mu * modB + self.charge * Phi)
            + self.omega
            * (
                (self.helicity_M * G + self.helicity_N * I) * self.charge * alpha
                + self.charge * (self.helicity_N * psi - self.helicity_M * psip)
            )
            / denom
            - self.Eprime
        )
        if (b**2 - 4 * a * c) < 0:
            print(
                "No solution for vpar found! Check the parameters and "
                "initial conditions."
            )
            return np.nan
        elif a != 0:
            return (-b + sgn * np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        else:
            return (-c / b) * sgn
    
    def instantiate_gridded_particles(self):
        surfaces = np.linspace(self.s_min,self.s_max,self.ns_points)
        pitch_angle = np.linspace(-1,1,self.nlambda_points)

        surfaces,pitch_angle = np.meshgrid(surfaces,pitch_angle)
        
        surfaces_flat = surfaces.flatten()
        pitch_angle_flat = pitch_angle.flatten()

        vpars = []
        mus = []
        for particle_index in range(len(surfaces_flat)):
            points_temp = initialize_position_uniform_surf(
                    self.B0, 
                    self.particles_per_surface,
                    surfaces_flat[particle_index],
                    comm=self.comm)
            if self.Eprime_slice:
                sgn = np.sign(pitch_angle_flat[particle_index])
                mu = (np.abs(pitch_angle_flat[particle_index]) * self.Ekin)
                vpars_temp = []
                for i in range(points_temp.shape[0]):
                    vp_temp = self.vpar_func_perturbed(
                        points_temp[i,0],
                        points_temp[i,1],
                        points_temp[i,2],
                        mu,
                        sgn
                    )
                    vpars_temp.append(vp_temp[0])
                vpars_temp = np.array(vpars_temp)
                mus_temp = mu/self.mass * np.ones_like(vpars_temp)
            else:
                sgn = np.sign(pitch_angle_flat[particle_index])
                vpars_temp = initialize_velocity_uniform(
                    self.vtotal,
                    points_temp.shape[0],
                    comm=self.comm,
                    seed=None,
                )
                self.B0.set_points(points_temp)
                modB = self.B0.modB()[:,0]
                mus_temp = (1/(2 * modB)) * (self.vtotal**2 - vpars_temp**2)
            # remove unphysical particles
            mask = ~np.isnan(vpars_temp)
            vpars_temp = vpars_temp[mask]
            points_temp = points_temp[mask]
            vpars_temp = vpars_temp.tolist()
            vpars += vpars_temp
            mus_temp = mus_temp.tolist()
            mus += mus_temp
            if particle_index == 0:
                points = points_temp
            else: 
                points = np.concatenate((points, points_temp), axis=0)
        print(f'{points.shape=}')
        return points[:,0].tolist(), points[:,1].tolist(), points[:,2].tolist(), vpars, mus

    def chi(self, theta, zeta):
        r"""
        Compute the helical angle chi = M*theta - N*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            chi : The helical angle.
        """
        return self.helicity_M * theta - self.helicity_N * zeta

    def eta(self, theta, zeta):
        r"""
        Compute the mapping angle eta = Mp*theta - Np*zeta.

        Args:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        Returns:
            eta : The mapping angle.
        """
        return self.helicity_Mp * theta - self.helicity_Np * zeta

    def chi_eta_to_theta_zeta(self, chi, eta):
        r"""
        Convert helical angles (chi, eta) to (theta, zeta).

        Args:
            chi : Helical angle chi.
            eta : Mapping angle eta.
        Returns:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        """
        denom = self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        theta = (self.helicity_Np * chi - self.helicity_N * eta) / denom
        zeta = (self.helicity_Mp * chi - self.helicity_M * eta) / denom 
        return theta, zeta
    
    def remove_equilibrium_lost_particles(self, points, vpars_init, mus):

        # trace particles in equilibrium field to see if any are lost
        gc_tys, gc_zeta_hits = trace_particles_boozer(
            field = self.B0, 
            stz_inits = points, 
            parallel_speeds = vpars_init,
            tmax=2e-3,
            mass=self.mass, 
            charge=self.charge,
            Ekin=self.Ekin,
            comm = self.comm,
            dt_save=self.min_timestep,
            tol=self.tol,
            stopping_criteria=[
                MaxToroidalFluxStoppingCriterion(1.0)
            ],
            mode='gc_noK',
            **self.solver_options
            )

        # check if any particles were lost to the wall 
        lost_total = []
        for i in range(len(gc_zeta_hits)):
            if gc_zeta_hits[i].size>0:
                if int(gc_zeta_hits[i][0][1]) ==-1:
                    lost_total.append(i)
        
        # remove wall lost particles from the list of evaluated particles
        points = np.delete(points, lost_total, axis=0)
        vpars_init = np.delete(vpars_init, lost_total, axis=0)
        mus = np.delete(mus, lost_total, axis=0)

        return points[:, 0], points[:, 1], points[:, 2], vpars_init, mus, lost_total

    def trace_particles(self):
        print("Tracing particles in perturbed field...", flush=True)

        first, last = parallel_loop_bounds(self.comm, len(self.s))
        res_tys = []  
        res_hits = []

        for itrj in range(first, last):
            point = np.zeros((1, 4))  # initialize with t = 0
            point[:, 0] = self.s[itrj]
            point[:, 1] = self.thetas[itrj]
            point[:, 2] = self.zetas[itrj]
            point[:, 3] = 0.0

            vpar = [self.vpar[itrj]]
            mu = [self.mus[itrj]]
            self.saw.set_points(point)
            gc_tys, gc_zeta_hits = trace_particles_boozer_perturbed(
                    perturbed_field = self.saw, 
                    stz_inits = point, 
                    parallel_speeds = vpar,
                    mus = mu, 
                    tmax=self.tmax, 
                    mass=self.mass, 
                    charge=self.charge,
                    Ekin=self.Ekin,
                    abstol=1e-9,
                    reltol=1e-9,
                    stopping_criteria=[
                        MaxToroidalFluxStoppingCriterion(1.0)
                    ],
                    mode='gc_noK',
                    ODE_solver="dormand_prince",
                    **self.solver_options,
                    )
            points_trajectory = gc_tys[0]
            time_momentum, s_path, theta_path, zeta_path, vpar_path = points_trajectory[:, 0], points_trajectory[:, 1], points_trajectory[:, 2], points_trajectory[:, 3], points_trajectory[:, 4]
            points_trajectory = np.column_stack(
                (s_path, theta_path, zeta_path, time_momentum)
            )
            idx_wall = np.argmax(s_path >= 1) if np.any(s_path >= 1) else None
            if idx_wall is not None and s_path[idx_wall] >= 1:
                print(f"Particle {itrj} hit the wall at time {time_momentum[idx_wall]}, s = {s_path[idx_wall]}",flush=True)
                idx_wall -= 1
                points_trajectory = points_trajectory[:idx_wall, :]
                vpar_path = vpar_path[:idx_wall]

            if points_trajectory.shape[0] < 10:
                print(f"Particle {itrj} has no valid trajectory points \t {point[-1, :]}", flush=True)
                start_state = [0, point[0, 0], point[0, 1], point[0, 2], None, None, None]
                particle_out = [start_state, start_state]
                res_tys.append(particle_out)
                res_hits.append(gc_zeta_hits[0])
                continue

            self.saw.set_points(points_trajectory)
            modB = self.saw.B0.modB()[:, 0]
            weighted_muB = self.mus[itrj] * self.mass * modB[0]
            E = 0.5 * self.mass * vpar_path**2 + self.mass * mu[0] * modB + self.charge * self.saw.Phi()[:, 0]
            Peta_values  = compute_peta(
                self.saw,
                points_trajectory,
                vpar_path,
                self.mass,
                self.charge,
                self.helicity_M,
                self.helicity_N,
                self.helicity_Mp,
                self.helicity_Np
            )

            trunc_Peta = Peta_values[::3]
            trunc_s = s_path[::3]
            trunc_t = time_momentum[::3]

            if len(trunc_Peta) != len(trunc_t):
                trunc_t = trunc_t[:len(trunc_Peta)]
                trunc_s = trunc_s[:len(trunc_Peta)]

            trunc_path_t = np.column_stack((trunc_t,trunc_Peta, trunc_s))
            #np.savetxt(f'{self.folder}particle_{itrj}_path.txt', trunc_path_t)

            s_mean = np.mean(s_path)

            dt = np.diff(trunc_t)
            ds = np.diff(trunc_s)  
            dt_diff = trunc_t[1:] - trunc_t[:-1]

            d_eff_0 = 0.5 * np.mean((trunc_Peta[1:] - trunc_Peta[0])**2  / dt_diff)
            d_eff_path =  0.5 * (np.mean(ds)**2) / np.mean(dt_diff)

            average_peta = np.mean(Peta_values)
            time_eval, DA_eval = return_DA(np.column_stack((points_trajectory[:, 3], Peta_values)))

            end_points = points_trajectory[-1,:-1] 
            start_points = points_trajectory[0,:-1]

            diffusion_data = [d_eff_0, d_eff_path]
            mean_data = [s_mean,d_eff_0,  average_peta, np.mean(E)]

            v_par_signs = np.sign(vpar_path)
            bounces = np.sum(v_par_signs[1:] * v_par_signs[:-1] < 0)

            start_phasespace = [vpar_path[0],Peta_values[0], E[0], weighted_muB]
            end_phasespace = [vpar_path[-1],Peta_values[-1], E[-1]]
            # puts time back in front 
            # t, s, theta, zeta, vpar, peta, E
            end_state = [points_trajectory[-1,-1].tolist()] + end_points.tolist() + end_phasespace
            start_state = [points_trajectory[0,-1].tolist()] + start_points.tolist() + start_phasespace
            particle_out = [start_state, end_state, diffusion_data, mean_data, DA_eval, bounces, weighted_muB]
            res_tys.append(particle_out)
            res_hits.append(gc_zeta_hits[0])
        print(f'{self.comm.rank=} done tracing particles', flush=True)

        if self.comm is not None:
            res_tys = [i for o in self.comm.allgather(res_tys) for i in o]
            res_hits = [i for o in self.comm.allgather(res_hits) for i in o]

        lost_total = []
        for i in range(len(res_hits)):
            if res_hits[i].size>0:     
                if int(res_hits[i][0][1]) ==-1:
                    lost_total.append(i)

        DAs = []
        Peta_start = []
        pitch_initial = []
        bounces = []

        for elem in res_tys:
            # index 0: start state: 
            # index 1: end state
            # index 2: diffusion data
            # index 3: mean data
            # index 4: DA value
            # index 5: number of bounces
            # start state vector:  [t, s, theta, zeta, vpar, peta, E, mu]
            if len(elem)>2:
                start = elem[0]
                end = elem[1]
                
                if self.diffusion: 
                    DAs.append(elem[2][0])  # diffusion data
                else:
                    DAs.append(elem[4])
                Peta_start.append(start[1])
                pitch_initial.append((start[7]/self.Ekin) * np.sign(start[4]))  # peta / E
                bounces.append(elem[5])

        
        self.DA_final = DAs
        self.res_tys = res_tys
        self.bounces = bounces
        #self.da_values, self.wall_lost, self.surfaces, self.pitch_angles =
        return DAs, lost_total, Peta_start, pitch_initial

    def plot_surfaces(self, nx=20, ny=20, savepath = 'heatmap_digit_accuracy.png',  ax=None, DA_max=7):
        import matplotlib as mpl
        from matplotlib.cm import ScalarMappable
        import matplotlib.pyplot as plt
        from scipy.stats import binned_statistic_2d
        import cmcrameri.cm as cmc 

        if ax is None:
            fig, ax = plt.subplots(figsize=(14,12))
        else:
            fig = ax.get_figure()

        if not self.randomize_particles:
            ny = self.ns_points
            nx = self.nlambda_points
        
        norm = mpl.colors.Normalize(vmin=0, vmax=DA_max)

        #print(f"{(self.pitch_angles)=} {(self.surfaces)=} {(self.DA_final)=}")

        stat, x_edges, y_edges, binnumber = binned_statistic_2d(
            np.array(self.pitch_angles), np.array(self.surfaces), np.array(self.DA_final),
            statistic='mean',
            bins=[nx, ny]
        )

        # Meshgrid
        X2, Y2 = np.meshgrid(x_edges, y_edges)

        im2 = ax.pcolormesh(X2, Y2, stat.T, shading='auto', cmap="cmc.managua", norm=norm)
        
        if self.diffusion:
            colorlabel = r'$D_{eff}$'
        else:
            colorlabel = 'Digit Accuracy'
        ax.set_xlabel(r'$\lambda = \frac{\mu}{E} \text{sign}(v_{||})$')
        ax.set_ylabel(r'$s_0$')
        ax.set_xlim([-1, 1])

        fig.tight_layout()
        fig.colorbar(im2, ax=ax, label=colorlabel)
        plt.savefig(savepath, dpi=400)

    def plot_ratio(self, ax=None,nx=20, ny=20, savepath = 'heatmap_ratio.png'):
        import matplotlib as mpl
        from matplotlib.cm import ScalarMappable
        import matplotlib.pyplot as plt
        from scipy.stats import binned_statistic_2d

        if ax is None:
            fig, ax = plt.subplots(figsize=(12,12))
        else:
            fig = ax.get_figure()

        if not self.randomize_particles:
            ny = self.ns_points
            nx = self.nlambda_points
    

        #print(f"{(self.pitch_angles)=} {(self.surfaces)=} {(self.DA_final)=}")

        stat, x_edges, y_edges, binnumber = binned_statistic_2d(
            np.array(self.pitch_angles), np.array(self.surfaces), np.array(self.DA_final),
            statistic='mean',
            bins=[nx, ny]
        )

        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        # Meshgrid for bin centers
        Xc, Yc = np.meshgrid(x_centers, y_centers)

        RATIO_grid = np.zeros_like(stat)
        num_list = []
        denom_list = []
        s_lst = []
        ratio_pre_weighted = []
        from scipy.constants import m_e, epsilon_0, e
        Z_alpha = 2
        Z_i = 1
        lnLambda = 18

        plt.rcParams.update({'font.size': 16})

        Te_eV = 12e3  * np.abs(e)      # 12 keV to J
        n_i = 2e20  # m^-3  
        def pa_frequency(s):
            v_Te = np.sqrt(2 * Te_eV / m_e)

            # prefactor
            prefactor = (n_i * Z_i**2 * Z_alpha**2 * e**4 * lnLambda) \
                        / (6 * np.sqrt(np.pi) * epsilon_0**2 * self.mass**2 * self.vtotal**3)
            
            # temperature/mass correction
            temp_factor = (m_e**1.5 * v_Te**3) / (Te_eV**1.5)

            # final rate
            v_d = prefactor * temp_factor
            return v_d

        def compute_jacobian_prime(points):
            self.B0.set_points(points)
            dGds = self.B0.dGds()[:, 0]
            G = self.B0.G()[:, 0]
            modB = self.B0.modB()[:, 0]
            dBds = self.B0.dmodBds()[:, 0]
            diotads = self.B0.diotads()[:, 0]
            dIds = self.B0.dIds()[:, 0]
            I = self.B0.I()[:, 0]
            iota = self.B0.iota()[:, 0]

            denom = (G + iota * I)
            inner = 2 * dBds - modB * (dGds + I * diotads + iota * dIds) / denom
            return modB * inner / denom
        def dpeta_ds(points, lam):
            self.B0.set_points(points)
            G = self.B0.G()[:, 0]
            I = self.B0.I()[:, 0]
            modB = self.B0.modB()[:, 0]
            dBds = self.B0.dmodBds()[:, 0]
            dIds = self.B0.dIds()[:, 0]
            dGds = self.B0.dGds()[:, 0]
            psi = self.B0.psi0 * points[:, 0]
            psip = self.B0.psip()[:, 0]

            alpha = self.saw.alpha()[:, 0]
            dalphads = self.saw.dalphadpsi()[:, 0]

            constant = (self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp)

            geometry = (self.helicity_M * G + self.helicity_N * I)
            geometry_prime = (self.helicity_M * dGds + self.helicity_N * dIds) 
            vpar = np.sqrt(self.vtotal * (1-lam))

            field_term = (self.mass * vpar / modB + self.charge * alpha + self.charge * self.helicity_N * psi + self.charge * self.helicity_M * psip)

            ke_prime = (-1* (self.mass * vpar * dBds)/(modB**2))
            pertubation_prime = self.charge * dalphads
            field_terms_prime = self.charge * self.helicity_N * self.B0.psi0 + self.charge

            dpeta_ds = (1/constant) * (geometry_prime * field_term)     
            dpeta_ds +=  (1/constant) * (geometry * (ke_prime + pertubation_prime) + geometry * field_terms_prime)
            return dpeta_ds

        def RATIO(D, points,pitch,  m,n):

            self.B0.set_points(points)
            G = self.B0.G()[:, 0]
            I = self.B0.I()[:, 0]
            modB = self.B0.modB()[:, 0]
            iota = self.B0.iota()[:, 0]
            diotads = self.B0.diotads()[:, 0]

            diotads_fraction = (diotads * m)/ (iota * m - n)

            dPdE = 2 * self.mass / modB
            dTds_term = ( diotads_fraction + compute_jacobian_prime(points) * compute_jacobian(points))
            numerator = D * (dpeta_ds(points, pitch))

            denom = ((self.helicity_M*G + self.helicity_N*I) / (self.helicity_Np*self.helicity_M - self.helicity_Mp + self.helicity_N ))**2 * pa_frequency()**2*(dPdE)**2
            RATIO = -1 * numerator / denom
            return RATIO, numerator, denom

        def compute_jacobian(points):
            self.B0.set_points(points)
            G = self.B0.G()[:, 0]
            modB = self.B0.modB()[:, 0]
            I = self.B0.I()[:, 0]
            iota = self.B0.iota()[:, 0]

            denom = (G + iota * I)

            return modB**2 / denom

        for i in range(nx):
            for j in range(ny):
                D_val = stat[j, i]   # binned D in this cell
                s_val = Yc[j, i]
                lam_val = Xc[j, i]

                pts = initialize_position_uniform_surf(self.B0, 100, s_val)
                ratio, numerator, denom =  RATIO(D_val, pts, np.abs(lam_val), self.Phim, self.Phin)
                ratio_pre_weighted.append(ratio)
                ratio *= compute_jacobian(pts)
                ratio_weighted  = np.sum(ratio) / np.sum(compute_jacobian(pts))

                num_list.append(numerator)
                denom_list.append(denom)
                s_lst.append(s_val)
                RATIO_grid[j, i] = ratio_weighted

        Xc, Yc = np.meshgrid(x_centers, y_centers)

        levels = 100
        cont = ax.contourf(
            Xc, Yc, RATIO_grid,
            levels=levels,
            cmap="plasma"
        )
        ax.set_title(r'R = $\frac{\nu_{stoch}}{\nu_{\text{Pitch Angle}}}$')
        fig.colorbar(cont, ax=ax, label='R')
        ax.set_ylabel(r'$s_0$')
        ax.set_xlabel(r'$\lambda = \frac{\mu}{E} \text{sign}(v_{||})$')
        plt.savefig(savepath, dpi=400)

class WBAParticles:
    def __init__(self, 
            saw,
            initial_conditions,
            v_pars,
            mu_per_mass,
            mass,
            charge,
            Ekin,
            helicity_N,
            helicity_M,
            helicity_Mp=None,
            helicity_Np=None,
            mean=True,
            savedata=(False,'DATA/'),
            tmax = 1e-2,
            min_timestep = 1e-6,
            comm=None,
            DA_cutoff=3,
            skipped_particles = [],
            solver_options = {},
            nconvergence_points=1):

        self.saw = saw
        self.B0 = saw.B0
        self.helicity_M = helicity_M
        self.helicity_N = helicity_N

        if helicity_Mp is None and helicity_Np is None:
            # If modB contours close poloidally, then use theta as mapping coordinate
            if helicity_M == 0:
                self.helicity_Mp = 1
                self.helicity_Np = 0
            # Otherwise, use zeta as mapping coordinate
            else:
                self.helicity_Mp = 0
                self.helicity_Np = -1
        else:
            if (helicity_Mp * helicity_N) == (helicity_Np * helicity_M):
                raise ValueError(
                    "Chosen helicities (N, M, N', M') do not create a well "
                    "defined Jacobian."
                )
            self.helicity_Mp = helicity_Mp
            self.helicity_Np = helicity_Np
        
        self.mass = mass
        self.charge = charge
        self.Ekin = Ekin

        self.comm = comm
        self.verbose=False
        if self.comm==None:
            self.verbose = True
        elif self.comm.rank==0:
            self.verbose = True

        self.solver_options = solver_options
        self.tmax = tmax

        self.mean = mean
        self.savedata = savedata[0]
        self.savepath = savedata[1]
        self.convergence_points = nconvergence_points

        if self.savedata:
            self.IC_filepaths = {
                's0': self.savepath + 'uniform_s0.txt',
                'theta0': self.savepath + 'uniform_theta0.txt',
                'zeta0': self.savepath + 'uniform_zeta0.txt',
                'vpar0': self.savepath + 'uniform_vpar0.txt',
                'mu_per_mass': self.savepath + 'uniform_mu_per_mass.txt'
            }
            self.final_filepaths = {
                'DA': self.savepath + f'DA.txt',
                'wall_lost': self.savepath + f'wall_lost.txt',
            }
        
        self.skip = skipped_particles
        if not self.check_filepaths(self.final_filepaths):
            points_phase = np.append(initial_conditions, np.zeros((initial_conditions.shape[0],1)), axis=1)
            self.gc_tys = self.trace_particles(saw, points_phase, v_pars, mu_per_mass)
            self.DAs, self.wall_lost_indicies, self.wall_lost_times = self.quantify_chaos_and_losses(
                    trajectories=self.gc_tys,
                    equilibrium_lost_indicies = self.skip)
            np.savetxt(self.final_filepaths['DA'], np.array(self.DAs))
            np.savetxt(self.final_filepaths['wall_lost'], np.column_stack((self.wall_lost_indicies, self.wall_lost_times)))
        else:
            self.DAs = np.loadtxt(self.final_filepaths['DA']).tolist()
            wall_lost = np.loadtxt(self.final_filepaths['wall_lost']).astype(int)
            self.wall_lost_indicies = wall_lost[:,0].tolist()
            self.wall_lost_times = wall_lost[:,1].tolist()
        self.numparticle = len(self.DAs)-len(self.skip)
        fractional_chaotic = self.compute_fractions(DA_cutoff=DA_cutoff)
        
    def compute_fractions(self, DA_cutoff=3):
        uniform_fractional_chaotic = [(sum([1 for i in range(len(self.DAs)) if ((self.DAs[i]<DA_cutoff) or (i in self.wall_lost_indicies))])/(self.numparticle)) * 100]
        return uniform_fractional_chaotic          

    def quantify_chaos_and_losses(self, trajectories, equilibrium_lost_indicies):
        lost_total = []
        DA_list = []
        lost_times = []

        for i in range(len(trajectories)):
            # trajectories are in format: 
            # first slice, last slice, DA
            # slice = [s, theta, zeta, time]
            if i in equilibrium_lost_indicies:
                DA_list.append(np.nan)
                continue

            final_time = trajectories[i][1][3]
            final_s = trajectories[i][1][0]
            DA = trajectories[i][2]
            
                
            # check if particle lost to wall
            if final_time < (self.tmax-2e-6):
                lost_total.append(int(i))
                lost_times.append(final_time)
            DA_list.append(DA)
        return DA_list, lost_total, lost_times

    def check_filepaths(self,filepaths):
        return all([exists(fp) for fp in filepaths.values()])
    
    def trace_particles(self, saw, points_phase, vpars, mus, ):
        first, last = parallel_loop_bounds(self.comm, points_phase.shape[0])
        res_tys = []  
        res_hits = []

        for itrj in range(first, last):
            if itrj in self.skip:
                start_state = [points_phase[itrj, 0], points_phase[itrj, 1], points_phase[itrj, 2],0]
                particle_out = [start_state, start_state, np.nan]
                res_tys.append(particle_out)
                res_hits.append(np.array([]))
                continue
            #print(f'{points_phase[itrj].shape=} \t{vpars[itrj]=} \t{mus[itrj]=}', flush=True)
            gc_tys, gc_zeta_hits = trace_particles_boozer_perturbed(
                    perturbed_field = saw, 
                    stz_inits = points_phase[itrj,:].reshape(1,4), 
                    parallel_speeds = [vpars[itrj]],
                    mus = [mus[itrj]], 
                    tmax=self.tmax, 
                    mass=self.mass, 
                    charge=self.charge,
                    Ekin=self.Ekin,
                    abstol=1e-9,
                    reltol=1e-9,
                    stopping_criteria=[
                        MaxToroidalFluxStoppingCriterion(1.0)
                    ],
                    mode='gc_noK',
                    ODE_solver="dormand_prince",
                    **self.solver_options,)
            
            points_trajectory = gc_tys[0]
            time_momentum, s_path, theta_path, zeta_path, vpar_path = points_trajectory[:, 0], points_trajectory[:, 1], points_trajectory[:, 2], points_trajectory[:, 3], points_trajectory[:, 4]
            points_trajectory = np.column_stack(
                (s_path, theta_path, zeta_path, time_momentum)
            )
            idx_wall = np.argmax(s_path >= 1) if np.any(s_path >= 1) else None
            if idx_wall is not None and s_path[idx_wall] >= 1:
                print(f"Particle {itrj} hit the wall at time {time_momentum[idx_wall]}, s = {s_path[idx_wall]}",flush=True)
                idx_wall -= 1
                points_trajectory = points_trajectory[:idx_wall, :]
                vpar_path = vpar_path[:idx_wall]

            if points_trajectory.shape[0] < 8:
                print(f"Particle {itrj} has no valid trajectory points \t {points_trajectory[-1, :]}", flush=True)
                start_state = points_trajectory[0, :].tolist()
                end_state = points_trajectory[-1, :].tolist()
                particle_out = [start_state, end_state, np.nan]
                res_tys.append(particle_out)
                res_hits.append(gc_zeta_hits[0])
                continue

            Peta_values  = compute_peta(
                saw,
                points_trajectory,
                vpar_path,
                self.mass,
                self.charge,
                self.helicity_M,
                self.helicity_N,
                self.helicity_Mp,
                self.helicity_Np
            )
            time_eval, DA_eval = return_DA(np.column_stack((points_trajectory[:, 3], Peta_values)))

            first_slice = points_trajectory[0,:]
            last_slice = points_trajectory[-1,:]
            particle_out = [first_slice.tolist(), last_slice.tolist(), DA_eval]
            res_tys.append(particle_out)
            res_hits.append(gc_zeta_hits[0])
        if self.comm is not None:
            res_tys = [i for o in self.comm.allgather(res_tys) for i in o]
        return res_tys



def trajectory_to_vtk(res_ty, field, filename="trajectory"):
    r"""
    Save a single particle trajectory in Cartesian coordinates to a VTK file.
    Requires the pyevtk package to be installed.

    Args:
        res_ty : A 2D numpy array of shape (nsteps, 5) containing the trajectory of a
                 single particle in Boozer coordinates.
        field : The :class:`BoozerMagneticField` instance used for field evaluation.
        filename : The name of the output VTK file.
    """
    from pyevtk.hl import polyLinesToVTK

    R_traj, phi_traj, Z_traj = compute_trajectory_cylindrical(res_ty, field)

    X_traj = R_traj * np.cos(phi_traj)
    Y_traj = R_traj * np.sin(phi_traj)

    ppl = np.asarray([len(R_traj)])  # Number of points along trajectory
    polyLinesToVTK(filename, X_traj, Y_traj, Z_traj, pointsPerLine=ppl)
