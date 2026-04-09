from os.path import exists
from warnings import warn

import numpy as np
from scipy import integrate

from .._core.util import parallel_loop_bounds
from ..field.boozermagneticfield import (
    BoozerMagneticField,
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
    "TrappedPoincare",
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
        mu=None,
        Eprime=None,
        nprime=None,
        omega=None,
        helicity_N=None,
        helicity_M=None,
        helicity_Np=None,
        helicity_Mp=None,
    ):
        r"""
        Initialize and compute the passing Poincare map, evaluated by
        integrating the guiding center equations until the trajectory returns
        to the zeta = 0 plane.
        We assume that the particle is passing, so the parallel velocity does
        not change sign.

        Args:
            field : The :class:`BoozerMagneticField` instance.
            lam : Pitch-angle variable :math:`\lambda = v_\perp^2/(v^2 B)`.
            sign_vpar : Sign of the parallel velocity.
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

        self.helicity_N = helicity_N
        self.helicity_M = helicity_M
        self.helicity_Np = helicity_Np
        self.helicity_Mp = helicity_Mp

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
        self.vpars_init, self.s_init, self.thetas_init = self.initialize_passing_map(
            s_flat, thetas_flat
        )
        (self.s_all, self.thetas_all, self.vpars_all, self.t_all, self.peta_all) = (
            self.compute_passing_map()
        )

    def initialize_passing_map(self, s_flat, thetas_flat):
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

        points_traj = np.zeros((1, 3))
        points_traj[:, 0] = res_tys[0][-1, 1]
        points_traj[:, 1] = res_tys[0][-1, 2]
        points_traj[:, 2] = res_tys[0][-1, 3]
        vpar_path = res_tys[0][-1, 4]
        peta = compute_peta(
            self.field,
            points_traj,
            vpar_path,
            self.mass,
            self.charge,
            self.helicity_M,
            self.helicity_N,
            helicity_Mp=self.helicity_Mp,
            helicity_Np=self.helicity_Np,
        )

        if res_hit[1] == 0:  # Check that the zetas=[0] plane was hit
            point[0] = res_hit[2]
            point[1] = res_hit[3]
            point[2] = res_hit[5]
            time = res_hit[0]
            return point, time, peta[0]
        else:
            raise RuntimeError("Alternative stopping criterion reached in passing_map.")

    def compute_passing_map(self):
        r"""
        Evaluates the passing Poincare return map for the initialized particle
        positions.
        """
        Ntrj = len(self.s_init)

        s_all = []
        peta_all = []
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
            points_traj = np.zeros((1, 3))
            points_traj[:, 0] = self.s_init[itrj]
            points_traj[:, 1] = self.thetas_init[itrj]
            points_traj[:, 2] = 0
            peta = compute_peta(
                self.field,
                points_traj,
                self.vpars_init[itrj],
                self.mass,
                self.charge,
                self.helicity_M,
                self.helicity_N,
                helicity_Mp=self.helicity_Mp,
                helicity_Np=self.helicity_Np,
            )
            peta_traj = [peta[0]]
            thetas_traj = [tr[1]]
            vpars_traj = [tr[2]]
            t_traj = [0]
            for _jj in range(self.Nmaps):
                try:
                    tr, time, peta = self.passing_map(tr)
                    s_traj.append(tr[0])
                    thetas_traj.append(tr[1])
                    vpars_traj.append(tr[2])
                    t_traj.append(time)
                    peta_traj.append(peta)
                except RuntimeError:
                    break
            peta_all.append(peta_traj)
            s_all.append(s_traj)
            thetas_all.append(thetas_traj)
            vpars_all.append(vpars_traj)
            t_all.append(t_traj)

        if self.comm is not None:
            peta_all = [i for o in self.comm.allgather(peta_all) for i in o]
            s_all = [i for o in self.comm.allgather(s_all) for i in o]
            thetas_all = [i for o in self.comm.allgather(thetas_all) for i in o]
            vpars_all = [i for o in self.comm.allgather(vpars_all) for i in o]
            t_all = [i for o in self.comm.allgather(t_all) for i in o]

        return s_all, thetas_all, vpars_all, t_all, peta_all

    def compute_frequencies(self, s_profile=True):
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
        init_peta = []
        for s_traj, theta_traj, _vpar_traj, t_traj, peta_traj in zip(
            self.s_all, self.thetas_all, self.vpars_all, self.t_all, self.peta_all
        ):
            if (
                len(peta_traj) < 2
            ):  # Need at least one full Poincare return maps to compute frequency
                continue
            delta_theta = np.array(theta_traj[1:]) - np.array(theta_traj[0:-1])

            delta_t = t_traj[1::]
            delta_zeta = 2 * np.pi * self.sign_vpar * sign_G

            # Average over wells along one field line
            freq_theta = np.mean(delta_theta) / np.mean(delta_t)
            freq_zeta = delta_zeta / np.mean(delta_t)

            omega_theta.append(freq_theta)
            omega_zeta.append(freq_zeta)
            init_s.append(np.mean(s_traj))
            init_peta.append(np.mean(peta_traj))

        omega_theta = np.array(omega_theta)
        omega_zeta = np.array(omega_zeta)
        init_s = np.array(init_s)
        init_peta = np.array(init_peta)

        s_prof = np.unique(init_s)
        peta_prof = np.unique(init_peta)

        # Average over field-line label
        if s_profile:
            omega_theta_prof = np.zeros((len(s_prof),))
            omega_zeta_prof = np.zeros((len(s_prof),))
            for i, s in enumerate(s_prof):
                omega_theta_prof[i] = np.mean(omega_theta[np.where(init_s == s)])
                omega_zeta_prof[i] = np.mean(omega_zeta[np.where(init_s == s)])
            return omega_theta_prof, omega_zeta_prof, s_prof
        else:
            omega_theta_prof = np.zeros((len(peta_prof),))
            omega_zeta_prof = np.zeros((len(peta_prof),))
            s_prof = np.zeros((len(peta_prof),))
            for i, s in enumerate(peta_prof):
                omega_theta_prof[i] = np.mean(omega_theta[np.where(init_peta == s)])
                omega_zeta_prof[i] = np.mean(omega_zeta[np.where(init_peta == s)])
                s_prof[i] = np.mean(init_s[np.where(init_peta == s)])
            return omega_theta_prof, omega_zeta_prof, peta_prof, s_prof

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
            np.array(s_traj[1:]) - np.array(s_traj[0:-1])
            t_traj[1::]
            delta_zeta = 2 * np.pi * self.sign_vpar * sign_G
            helicity_M * delta_theta + helicity_N * delta_zeta

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
        mass,
        charge,
        Ekin,
        s_mirror=None,
        theta_mirror=None,
        zeta_mirror=None,
        lam=None,
        ns_poinc=None,
        neta_poinc=None,
        s_init=None,
        etas_init=None,
        Nmaps=500,
        comm=None,
        tmax=1e-2,
        chaos_detection=False,
        nconvergence_points=None,
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
            helicity_M : Approximate poloidal helicity of the field strength for
                         classifying ripple and barely-trapped particles.
            helicity_N : Approximate toroidal helicity of the field strength for
                         classifying ripple and barely-trapped particles.
            mass : Particle mass.
            charge : Particle charge.
            Ekin : Particle total energy.
            s_mirror : Initial s coordinate for the mirror point. If None, the
                       pitch-angle variable lam is used to find the mirror point.
            theta_mirror : Initial theta coordinate for the mirror point. If None
                           when lam is provided, the default value of chi = pi/2 is
                           used as an initial guess for the mirror point.
            zeta_mirror : Initial zeta coordinate for the mirror point. If None
                           when lam is provided, the default value of chi = pi/2 is
                           used as an initial guess for the mirror point.
            lam : Pitch-angle variable :math:`\lambda = v_\perp^2/(v^2 B)`.
                  If s_mirror, theta_mirror, zeta_mirror and lam are all provided,
                  then lam is used to find the mirror point, but s_mirror,
                  theta_mirror, zeta_mirror are still used to specify an initial
                  guess for the root solve to find the mirror point.
            ns_poinc : Number of initial conditions in s for Poincare plot
                       (default: 120).
            neta_poinc : Number of initial conditions in eta for Poincare plot
                         (default: 2).
            s_init : List of initial s coordinates for the Poincare map.
                     (default: None, ns_poinc is used instead)
            etas_init : List of initial eta coordinates for the Poincare map.
                        (default: None, neta_poinc is used instead)
            Nmaps : Number of Poincare return maps to compute for each initial
                    condition (default: 500).
            comm : MPI communicator for parallel execution (default: None).
            tmax : Maximum integration time for each segment of the Poincare
                   map (default: 1e-2 s).
            solver_options : Dictionary of options to pass to the ODE solver
                             (default: {}).
            chaos_detection : Whether to perform chaos detection (default: False).
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

        if lam is not None:
            if lam <= 0:
                raise ValueError("lam must be positive.")
            self.lam = lam
            self.modBcrit = 1 / self.lam
            if theta_mirror is None or zeta_mirror is None:
                # Default value for mirror point initial guess
                self.chi_mirror = np.pi / 2
            else:
                self.chi_mirror = self.chi(theta_mirror, zeta_mirror)
        elif (
            s_mirror is not None
            and theta_mirror is not None
            and zeta_mirror is not None
        ):
            field.set_points(np.array([[s_mirror], [theta_mirror], [zeta_mirror]]).T)
            self.modBcrit = field.modB()[0, 0]  # Magnetic field at mirror point
            self.lam = 1 / self.modBcrit  # lambda = v_perp^2/(v^2 B) = 1/modBcrit
            self.chi_mirror = self.chi(theta_mirror, zeta_mirror)
        else:
            raise ValueError(
                "Either lam or s_mirror, theta_mirror, zeta_mirror must be provided."
            )

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

        self.DA_poinc = chaos_detection
        if self.DA_poinc:
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
            self.WBA_transit_steps = [Nmaps - 1]

        (
            self.s_all,
            self.chis_all,
            self.etas_all,
            self.t_all,
            self.DA_all,
            self.DA_times,
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
            #return point, time
        else:
            raise RuntimeError("Alternative stopping criterion reached in passing_map.")

        if not self.DA_poinc:
            return point, time
        
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

        peta = compute_peta(
            self.field,
            points_traj,
            vpar_path,
            self.mass,
            self.charge,
            self.helicity_M,
            self.helicity_N,
            self.helicity_Mp,
            self.helicity_Np,
        )
        peta = np.column_stack((time_momentum, peta))
        return point, time, self.eta(res_hit[3], res_hit[4]), peta

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

            try:
                sol = root_scalar(
                    diffmodB,
                    fprime=graddiffmodB,
                    x0=self.chi_mirror,
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
        DA_all = []
        DA_times = []
        t_all = []
        first, last = parallel_loop_bounds(self.comm, Ntrj)
        for itrj in range(first, last):
            tr = [self.s_init[itrj], self.chis_init[itrj], self.etas_init[itrj]]
            s_traj = [tr[0]]
            chis_traj = [tr[1]]
            etas_traj = [tr[2]]
            t_traj = [0]
            broken = False
            particle_DAs = []
            particle_DA_times = []
            for jj in range(self.Nmaps):
                try:
                    if self.DA_poinc:
                        if jj == 0: 
                            tr, time1, Peta = self.trapped_map(tr)
                        else:
                            tr, time1, Peta_iter = self.trapped_map(tr)
                            Peta_iter[:, 0] += Peta[-1, 0]
                            Peta = np.vstack((Peta, Peta_iter[1:, :]))

                        tr, time2, Peta_iter = self.trapped_map(tr)
                        Peta_iter[:, 0] += Peta[-1, 0]
                        Peta = np.vstack((Peta, Peta_iter[1:, :]))
                    else:
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
                    if self.DA_poinc and jj in self.WBA_transit_steps:
                        time_at_evaluation, DA_at_evaluation = return_DA(Peta)
                        particle_DAs.append(DA_at_evaluation)
                        particle_DA_times.append(jj)
                except RuntimeError:
                    broken = True
                    # @NOTE: not returning DA 
                    break
            if not broken:
                s_all.append(s_traj)
                chis_all.append(chis_traj)
                etas_all.append(etas_traj)
                t_all.append(t_traj)
                DA_all.append(particle_DAs)
                DA_times.append(particle_DA_times)

        if self.comm is not None:
            s_all = [i for o in self.comm.allgather(s_all) for i in o]
            chis_all = [i for o in self.comm.allgather(chis_all) for i in o]
            etas_all = [i for o in self.comm.allgather(etas_all) for i in o]
            t_all = [i for o in self.comm.allgather(t_all) for i in o]
            DA_all = [i for o in self.comm.allgather(DA_all) for i in o]
            DA_times = [i for o in self.comm.allgather(DA_times) for i in o]

        return s_all, chis_all, etas_all, t_all, DA_all, DA_times

    def plot_poincare(self, 
                      ax=None, 
                      filename="trapped_poincare.pdf", 
                      convergence_test_indicies=None, 
                      DA_max=7):
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
        import matplotlib as mpl

        mpl.use("Agg")  # Don't use interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable

        try:
            import cmcrameri.cm as cmc  # noqa: F401
            cmap = "cmc.managua"
        except ImportError:
            cmap = "viridis"

        if ax is None:
            fig, ax = plt.subplots()
        
        if convergence_test_indicies is None:
            convergence_test_indicies = list(range(len(self.s_all)))
        
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


        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$s$")
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 1])
        for i in range(len(self.etas_all)):
            if self.DA_poinc: 
                ax.scatter(
                    np.mod(self.etas_all[i], 2 * np.pi),
                    self.s_all[i],
                    marker="o",
                    s=0.5,
                    c=cmap_object(DA_norm_all[i]),
                    edgecolors="none",
                )
            else:
                ax.scatter(
                    np.mod(self.etas_all[i], 2 * np.pi),
                    self.s_all[i],
                    marker="o",
                    s=0.5,
                    edgecolors="none",
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


def compute_Eprime(
    saw, points, vpar, mu, mass, charge, helicity_M, helicity_N, nprime=None
):
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
    # if isinstance(saw, ShearAlfvenHarmonic) is False:
    #    raise TypeError("Expected saw to be an instance of ShearAlfvenHarmonic")

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


def vpar_func_perturbed(
    field_or_saw,
    s,
    theta,
    zeta,
    mass,
    helicity_M,
    helicity_N,
    helicity_Mp,
    helicity_Np,
    Eprime,
    mu,
    charge,
    nprime,
    omega,
    sign_vpar=1,
):
    point = np.zeros((1, 4))  # initialize with t = 0
    point[0, 0] = s
    point[0, 1] = theta
    point[0, 2] = zeta
    if isinstance(field_or_saw, ShearAlfvenWave):
        field = field_or_saw.B0
        field_or_saw.set_points(point)
        alpha = field_or_saw.alpha()[:, 0]
        Phi = field_or_saw.Phi()[:, 0]
    else:
        field = field_or_saw
        alpha = 0.0
        Phi = 0
        if point.shape == 4:
            point = point[:, :3]
        field.set_points(point)

    # Choose initial conditions on the eta = 0 plane
    modB = field.modB()[0, 0]
    G = field.G()[0, 0]
    I = field.I()[0, 0]
    psi = field.psi0 * s
    psip = field.psip()[0, 0]
    denom = helicity_Np * helicity_M - helicity_N * helicity_Mp  # - 1 in QA
    d_peta_d_vpar = (
        -((helicity_M * G + helicity_N * I) * (mass / modB)) / denom
    )  # G m/ modB in QA
    d_E_d_vpar2 = 0.5 * mass
    a = nprime * d_E_d_vpar2  # Coefficient of vpar^2
    b = -omega * d_peta_d_vpar  # Coefficient of vpar
    # Constant term
    c = (
        nprime * (mass * mu * modB + charge * Phi)
        + omega
        * (
            (helicity_M * G + helicity_N * I) * charge * alpha
            + charge * (helicity_N * psi - helicity_M * psip)
        )
        / denom
        - Eprime
    )
    if (b**2 - 4 * a * c) < 0:
        return None
    elif a != 0:
        return (-b + sign_vpar * np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    else:
        return (-c / b) * sign_vpar


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
            -((self.helicity_M * G + self.helicity_N * I) * (self.mass / modB)) / denom
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
        linecolors=None,
        ylims=(0, 1),
        colorbar=True,
        s_axis_label=True,
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
            # if len(self.chis_all[i]) < Nmaps
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

        lines_2 = []
        if lines is not None:
            cmap = plt.get_cmap("Wistia")
            n_lines = len(lines)
            for i, line in enumerate(lines):
                ell, arr = line[0], line[1]
                if linecolors is not None:
                    color = linecolors[i]
                else:
                    color = cmap(i / max(n_lines - 1, 1))
                lines_2.append((line[0], line[1], color))
                vp = self.vpar_func_perturbed(arr, self.chi(np.pi / 2, 0))
                self.B0.set_points(np.array([[arr, np.pi / 2, 0]]).T)
                unperturbed_path_map = PassingPoincare(
                    field=self.B0,
                    lam=(self.v0**2 - vp**2) / (self.v0**2 * self.B0.modB()[0, 0]),
                    sign_vpar=self.sign_vpar,
                    mass=self.mass,
                    charge=self.charge,
                    Ekin=self.Ekin,
                    s_init=[arr],
                    comm=None,
                    Nmaps=250,
                    helicity_N=self.helicity_N,
                    helicity_M=self.helicity_M,
                    helicity_Mp=self.helicity_Mp,
                    helicity_Np=self.helicity_Np,
                    thetas_init=[np.pi / 2],
                    solver_options={"axis": 0},
                )

                s_upt, theta_upt, vpar_upt, t_upt = (
                    unperturbed_path_map.get_poincare_data()
                )
                chis = self.chi(
                    np.array(theta_upt[0]),
                    np.array([2 * np.pi * i for i in range(len(theta_upt[0]))]),
                )
                s_upt = np.array(s_upt[0])
                pa_data = np.column_stack((chis, s_upt))
                # pa_data = pa_data[pa_data[:, 0].argsort()]
                pa_data[:, 0] = np.mod(pa_data[:, 0], (2 * np.pi))
                pa_data = pa_data[pa_data[:, 0].argsort()]
                if i > 0 and lines[i][0] == lines[i - 1][0]:
                    ax.plot(pa_data[:, 0], pa_data[:, 1], lw=5, color=color)
                    continue
                ax.plot(
                    pa_data[:, 0],
                    pa_data[:, 1],
                    label=r"$\ell$=" + f"{ell}",
                    lw=5,
                    color=color,
                )
            ax.legend()
        fig.tight_layout()
        fig.savefig(filename + ".png", dpi=400)

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
            plt.savefig(filename + "_convergence.png", dpi=300)
            plt.clf()
            final_DAs = [x for x in final_DAs if not np.isnan(x)]
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
        ns_points=25,
        particles_per_surface=15,
        nlambda_points=25,
        randomize_particles=False,
        number_of_particles=10000,
        initial_conditions=None,
        initial_vpar=None,
        initial_mu_per_particle=None,
        Eprime=None,
        sign=None,
        min_timestep=1e-6,
        s_lims=None,
        diffusion=False,
        mean=True,
        comm=None,
        tmax=1e-2,
        tol=1e-10,
        skip=None,
        solver_options=None,
        unperturbed=False,
        savedata=None,
        nconvergence_points=1,
        plot_s=False,
        mu_max=None,
    ):
        """
        Initialize phase-space sampling, particle tracing, and diagnostic evaluation
        for guiding-center orbits in a perturbed Boozer magnetic field.

        This class generates or accepts a collection of initial particle conditions,
        removes particles that are already lost in the equilibrium field, then traces
        the remaining particles in the perturbed field and computes per-particle
        diagnostics such as digit accuracy or effective diffusion, wall-loss status,
        and phase-space coordinates used for plotting.

        The sampled phase space is parameterized by Boozer coordinates together with
        either:
            - a fixed total kinetic energy Ekin, or
            - a fixed shifted-energy slice Eprime.

        If Eprime is provided, the class samples magnetic moment mu and solves for the
        initial parallel velocity vpar from the perturbed invariant constraint. If
        Eprime is not provided, particles are initialized at fixed Ekin with velocities
        sampled directly and magnetic moment inferred from the local magnetic field.
        This uses a reference kinetic energy Ekin.

        Initial conditions may be supplied directly or generated internally in one of
        two ways:
            - randomly throughout the plasma volume, or
            - on a structured grid in surface label and pitch
              coordinate, with multiple particles sampled on each surface.


        After initialization, the class:
            1. constructs or reads particle initial conditions,
            2. removes particles lost in the unperturbed equilibrium,
            3. traces the remaining particles in the perturbed field,
            4. computes summary diagnostics and stores quantities used for phase-space
               plots.

        Args:
            saw:
                Perturbed field object used for particle tracing. Must provide the
                equilibrium field through `saw.B0` and the perturbation-dependent
                quantities needed by the tracing and invariant routines.
            Phin_max:
                Toroidal mode-number-like coefficient associated with the perturbation.
            Phim_max:
                Poloidal mode-number-like coefficient associated with the perturbation.
            omega:
                Perturbation frequency.
            mass:
                Particle mass.
            charge:
                Particle charge.
            Ekin:
                Reference kinetic energy used for initialization and for diagnostics.
                If `Eprime` is not provided, particles are initialized at fixed `Ekin`.
            helicity_N:
                Toroidal helicity N entering chi = M * theta - N * zeta.
            helicity_M:
                Poloidal helicity M entering chi = M * theta - N * zeta.
            helicity_Mp:
                Poloidal helicity Mp entering eta = Mp * theta - Np * zeta. If None,
                it is chosen automatically together with `helicity_Np`.
            helicity_Np:
                Toroidal helicity Np entering eta = Mp * theta - Np * zeta. If None,
                it is chosen automatically together with `helicity_Mp`.
            ns_points:
                Number of sampled surfaces used when constructing gridded initial
                conditions.
            particles_per_surface:
                Number of spatial initial conditions sampled on each surface for each
                grid point in the other phase-space variables.
            nlambda_points:
                Number of sampled magnetic-moment / pitch-like points used in gridded
                initialization.
            randomize_particles:
                If True, generate random initial conditions in the plasma volume.
                Otherwise generate structured gridded initial conditions.
            number_of_particles:
                Number of particles to generate when `randomize_particles=True`.
            initial_conditions:
                Optional user-supplied initial spatial coordinates. Expected to contain
                columns `[s, theta, zeta]`. If supplied, internal initialization is
                skipped.
            initial_vpar:
                Initial parallel velocities corresponding to `initial_conditions`.
                Required if `initial_conditions` is supplied.
            initial_mu_per_particle:
                Initial magnetic moments per particle corresponding to
                `initial_conditions`. Required if `initial_conditions` is supplied.
            Eprime:
                Optional shifted-energy value defining an invariant slice. If provided,
                the initialization solves for vpar from the perturbed invariant rather
                than sampling vpar directly from fixed Ekin.
            sign:
                Optional fixed sign of vpar used during initialization.
                If None, both signs are sampled.
            min_timestep:
                Minimum timestep used for intergration.
            s_lims:
                Minimum and maximum surface labels used during gridded initialization.
            mean:
                Controls averaging-related plotting or output behavior.
            comm:
                MPI communicator used to distribute particle work across ranks.
            tmax:
                Maximum integration time for perturbed orbit tracing.
            tol:
                Tolerance used in equilibrium tracing.
            skip:
                Optional list of particles or indices to mark as equilibrium-lost or
                exclude.
            solver_options:
                Additional keyword arguments passed to the orbit integrators.
            unperturbed:
                If True, use an alternate definition of nprime corresponding to the
                unperturbed mapping.
            savedata:
                Two-element list/tuple controlling whether data is saved and where.
                The first entry is a boolean flag and the second is the output prefix
                or directory.
            nconvergence_points:
                Number of convergence checkpoints used for time-series based
                diagnostics.
            plot_s:
                If True, use surface label `s` as the plotted vertical coordinate in
                later diagnostics; otherwise use `P_eta`.
            mu_max:
                Maximum magnetic moment sampled during initialization. If None, defaults
                to Ekin/minmodB of the volume.
        """
        if not isinstance(saw, ShearAlfvenWave) and not isinstance(
            saw, ShearAlfvenWavesSuperposition
        ):
            raise ValueError(
                "saw must be an instance of ShearAlfvenWave "
                "or ShearAlfvenWavesSuperposition."
            )
        # @TODO: add convergence points support

        if savedata is None:
            savedata = [True, "DATA/"]
        if solver_options is None:
            solver_options = {}
        if skip is None:
            skip = []
        if s_lims is None:
            s_lims = [0.01, 0.95]
        if mu_max is None:
            mu_max = Ekin
        self.mu_max = mu_max

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

        self.Phim = Phim_max
        self.Phin = Phin_max
        self.Phimp = (self.Phim * self.helicity_Np - self.Phin * self.helicity_Mp) / (
            self.helicity_M * self.helicity_Np - self.helicity_Mp * self.helicity_N
        )
        self.Phinp = (self.Phim * self.helicity_Np - self.Phin * self.helicity_Mp) / (
            self.helicity_M * self.helicity_Np - self.helicity_Mp * self.helicity_N
        )

        if unperturbed:
            self.nprime = (helicity_N - helicity_M) / (
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
        self.vtotal = np.sqrt(2 * self.Ekin / mass)
        self.mass = mass
        self.charge = charge
        if Eprime is None:
            self.Eprime_slice = False
        else:
            self.Eprime_slice = True
        self.Eprime = Eprime

        # set communicator parameters
        self.comm = comm
        self.verbose = False
        if self.comm is None or self.comm.rank == 0:
            self.verbose = True

        self.sign = sign
        self.plot_s = plot_s

        self.solver_options = solver_options

        def min_volumemodB():
            r"""
            Estimate minimum magnetic-field magnitude over sampled surfaces.

            Returns:
                min_modB : Approximate minimum value of modB in the sampled volume.
            """
            resolution = 100
            points = np.zeros((resolution * resolution, 3))

            for surface in np.linspace(0, 0.99, resolution):
                if surface == 0:
                    points = initialize_position_uniform_surf(
                        self.B0, resolution, surface
                    )
                else:
                    sampled_surface = initialize_position_uniform_surf(
                        self.B0, resolution, surface
                    )
                    points = np.concatenate((points, sampled_surface), axis=0)

            self.B0.set_points(points)
            modB = self.B0.modB()[:, 0]
            return np.min(modB)

        self.min_volmodB = min_volumemodB()

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

            if exists(self.savepath + "_initial_conditions.txt"):
                initial_conditions = np.loadtxt(
                    self.savepath + "_initial_conditions.txt"
                )
                s, thetas, zetas, vpar, mu = (
                    initial_conditions[:, 0],
                    initial_conditions[:, 1],
                    initial_conditions[:, 2],
                    initial_conditions[:, 3],
                    initial_conditions[:, 4],
                )
            else:
                if randomize_particles:
                    self.nParticles = number_of_particles
                    s, thetas, zetas, vpar, mu = self.instantiate_uniform_particles(
                        self.nParticles
                    )
                else:
                    self.ns_points = ns_points
                    self.particles_per_surface = particles_per_surface
                    self.nlambda_points = nlambda_points
                    self.nParticles = ns_points * particles_per_surface * nlambda_points
                    s, thetas, zetas, vpar, mu = self.instantiate_gridded_particles()

                np.savetxt(
                    self.savepath + "_initial_conditions.txt",
                    np.column_stack((s, thetas, zetas, vpar, mu)),
                )
        else:
            s = initial_conditions[:, 0]
            thetas = initial_conditions[:, 1]
            zetas = initial_conditions[:, 2]
            if initial_vpar is not None and initial_mu_per_particle is not None:
                vpar = initial_vpar
                mu = initial_mu_per_particle
            else:
                raise ValueError(
                    "If providing initial conditions, " \
                    "must provide both initial_vpar and initial_mu_per_particle"
                )

        # set parameters for convergence plot
        self.diffusion = diffusion
        expected_length = int(self.tmax / self.min_timestep)
        expected_step = int(expected_length / self.convergence_points)
        self.WBA_transit_steps = np.linspace(
            expected_step, expected_length - 1, num=nconvergence_points, dtype=int
        ).tolist()
        self.convergence_plot = nconvergence_points > 1
        initial_point = np.zeros((len(s), 3))  # initialize with t = 0
        initial_point[:, 0] = s
        initial_point[:, 1] = thetas
        initial_point[:, 2] = zetas

        if self.savedata:
            self.IC_filepaths = {
                "s0": self.savepath + "_s0.txt",
                "theta0": self.savepath + "_theta0.txt",
                "zeta0": self.savepath + "_zeta0.txt",
                "vpar0": self.savepath + "_vpar0.txt",
                "mu_per_mass": self.savepath + "_mu_per_mass.txt",
            }
            if diffusion:
                self.final_filepaths = {
                    "D": self.savepath + "Deff.txt",
                    "wall_lost": self.savepath + "wall_lost.txt",
                }
            else:
                self.final_filepaths = {
                    "D": self.savepath + "DA.txt",
                    "wall_lost": self.savepath + "wall_lost.txt",
                }
            self.res_filepaths = {
                "tys": self.savepath + "res_tys.txt",
                "hits": self.savepath + "res_hits.txt",
            }

        self.equilibrium_lost = skip
        self.s, self.thetas, self.zetas, self.vpar, self.mus, self.equilibrium_lost = (
            self.remove_equilibrium_lost_particles(initial_point, vpar, mu)
        )
        self.da_values, self.wall_lost, self.surfaces, self.pitch_angles = (
            self.trace_particles()
        )

    def check_filepaths(self, filepaths):
        r"""
        Check whether all provided output file paths exist.

        Args:
            filepaths : Dictionary of file labels to filesystem paths.
        Returns:
            exists_all : True if every path exists, otherwise False.
        """
        return all(exists(fp) for fp in filepaths.values())

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

    def instantiate_uniform_particles(self, nParticles):
        r"""
        Generate uniformly distributed initial particles and velocities.

        Args:
            nParticles : Number of particles to initialize.
        Returns:
            s : Initial radial-like coordinates.
            theta : Initial poloidal angles.
            zeta : Initial toroidal angles.
            vpars_init : Initial parallel velocities.
            mus_per_mass : Initial magnetic moments normalized by mass.
        """
        tracing_points = initialize_position_uniform_vol(
            self.B0,
            nParticles,
            comm=self.comm,
            seed=None,
        )

        if self.Eprime_slice:
            mu = np.random.uniform(0, self.mu_max, nParticles)
            if self.sign is not None:
                sgn = self.sign * np.ones(nParticles)
            else:
                sgn = np.random.choice([-1, 1], size=nParticles)
            vpars_temp = []
            for i in range(tracing_points.shape[0]):
                vp_temp = self.vpar_func_perturbed(
                    tracing_points[i, 0],
                    tracing_points[i, 1],
                    tracing_points[i, 2],
                    mu[i],
                    sgn[i],
                )
                vpars_temp.append(vp_temp[0])
            vpars_init = np.array(vpars_temp)

            mask = ~np.isnan(vpars_init)
            vpars_init = vpars_init[mask]
            mus_per_mass = mu[mask]
            mus_per_mass /= self.mass
            tracing_points = tracing_points[mask]
        else:
            vpars_init = initialize_velocity_uniform(
                self.vtotal,
                nParticles,
                comm=self.comm,
                seed=None,
            )
            self.B0.set_points(tracing_points)
            modB = self.B0.modB()[:, 0]
            mus_per_mass = (1 / (2 * modB)) * (self.vtotal**2 - vpars_init**2)
        return (
            tracing_points[:, 0],
            tracing_points[:, 1],
            tracing_points[:, 2],
            vpars_init,
            mus_per_mass,
        )

    def vpar_func(self, s, theta, zeta, p_a, sgn):
        r"""
        Compute unperturbed parallel velocity from pitch parameter and sign.

        Args:
            s : Radial-like Boozer coordinate(s).
            theta : Poloidal Boozer angle(s).
            zeta : Toroidal Boozer angle(s).
            p_a : Pitch-angle parameter.
            sgn : Desired velocity sign.
        Returns:
            vpar : Parallel velocity value(s), or nan for invalid states.
        """
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
        r"""
        Compute perturbed parallel velocity from the shifted-energy constraint.

        Args:
            s : Radial-like Boozer coordinate.
            theta : Poloidal Boozer angle.
            zeta : Toroidal Boozer angle.
            mu : Magnetic moment.
            sgn : Desired velocity sign.
        Returns:
            vpar : Parallel velocity solution, or nan when no solution exists.
        """
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
            -((self.helicity_M * G + self.helicity_N * I) * (self.mass / modB)) / denom
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
            return [np.nan]
        elif a != 0:
            return (-b + sgn * np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        else:
            return (-c / b) * sgn

    def instantiate_gridded_particles(self):
        r"""
        Generate gridded initial particles over surfaces and pitch space.

        Returns:
            s : Initial radial-like coordinates.
            theta : Initial poloidal angles.
            zeta : Initial toroidal angles.
            vpars : Initial parallel velocities.
            mus : Initial magnetic moments normalized by mass.
        """
        surfaces = np.linspace(self.s_min, self.s_max, self.ns_points)
        if self.verbose:
            print(f"self.s_min,self.s_max={self.s_min, self.s_max}")
        mu = np.linspace(0, self.mu_max, self.nlambda_points)

        surfaces, mu = np.meshgrid(surfaces, mu)

        surfaces_flat = surfaces.flatten()
        mus_flat = mu.flatten()

        vpars = []
        mus = []
        for particle_index in range(len(surfaces_flat)):
            points_temp = initialize_position_uniform_surf(
                self.B0,
                self.particles_per_surface,
                surfaces_flat[particle_index],
                comm=self.comm,
            )
            if self.Eprime_slice:
                if self.sign is not None:
                    sgn = self.sign * np.ones(self.particles_per_surface)
                else:
                    sgn = np.random.choice([-1, 1], size=self.particles_per_surface)
                mu_particle = mus_flat[particle_index]
                vpars_temp = []

                for i in range(points_temp.shape[0]):
                    vp_temp = self.vpar_func_perturbed(
                        points_temp[i, 0],
                        points_temp[i, 1],
                        points_temp[i, 2],
                        mu_particle,
                        sgn[i],
                    )
                    if isinstance(vp_temp, float):
                        continue
                    else:
                        vpars_temp.append(vp_temp[0])
                vpars_temp = np.array(vpars_temp)
                mus_temp = mu_particle / self.mass * np.ones(points_temp.shape[0])
            else:
                vpars_temp = initialize_velocity_uniform(
                    self.vtotal,
                    points_temp.shape[0],
                    comm=self.comm,
                    seed=None,
                )
                if self.sign is not None:
                    vpars_temp = self.sign * np.abs(vpars_temp)
                self.B0.set_points(points_temp)
                modB = self.B0.modB()[:, 0]
                mus_temp = (1 / (2 * modB)) * (self.vtotal**2 - vpars_temp**2)
            # remove unphysical particles
            mask = ~np.isnan(vpars_temp)
            vpars_temp = vpars_temp[mask]
            points_temp = points_temp[mask]
            vpars_temp = vpars_temp.tolist()

            vpars += vpars_temp

            mus_temp = mus_temp[mask]
            mus_temp = mus_temp.tolist()
            mus += mus_temp
            if particle_index == 0:
                points = points_temp
            else:
                points = np.concatenate((points, points_temp), axis=0)
        return (
            points[:, 0].tolist(),
            points[:, 1].tolist(),
            points[:, 2].tolist(),
            vpars,
            mus,
        )

    def remove_equilibrium_lost_particles(self, points, vpars_init, mus):
        r"""
        Filter particles lost in equilibrium before perturbed tracing.

        Args:
            points : Initial particle coordinates.
            vpars_init : Initial parallel velocities.
            mus : Initial magnetic moments.
        Returns:
            s : Filtered radial-like coordinates.
            theta : Filtered poloidal angles.
            zeta : Filtered toroidal angles.
            vpars_init : Filtered parallel velocities.
            mus : Filtered magnetic moments.
            lost_total : Indices of particles removed as equilibrium losses.
        """

        # trace particles in equilibrium field to see if any are lost
        gc_tys, gc_zeta_hits = trace_particles_boozer(
            field=self.B0,
            stz_inits=points,
            parallel_speeds=vpars_init,
            tmax=2e-3,
            mass=self.mass,
            charge=self.charge,
            Ekin=self.Ekin,
            comm=self.comm,
            forget_exact_path=True,
            dt_save=self.min_timestep,
            tol=self.tol,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
            mode="gc_noK",
            **self.solver_options,
        )

        # check if any particles were lost to the wall
        lost_total = []
        for i in range(len(gc_zeta_hits)):
            if isinstance(gc_zeta_hits[i], np.ndarray):
                if gc_zeta_hits[i].size > 0:
                    if int(gc_zeta_hits[i][0][1]) == -1:
                        lost_total.append(i)

        if self.Eprime_slice:
            self.equilibrium_lost_pitch = (np.array(mus)[lost_total] * self.mass) / (
                self.Eprime
            )
        else:
            self.equilibrium_lost_pitch = np.array(mus)[lost_total] / self.Ekin

        petas = compute_peta(
            self.B0,
            points,
            vpars_init,
            self.mass,
            self.charge,
            self.helicity_M,
            self.helicity_N,
            self.helicity_Mp,
            self.helicity_Np,
        )
        if self.plot_s:
            self.equilibrium_lost_surfaces = points[lost_total, 0]
        else:
            self.equilibrium_lost_surfaces = petas[lost_total]
        # remove wall lost particles from the list of evaluated particles
        points = np.delete(points, lost_total, axis=0)
        vpars_init = np.delete(vpars_init, lost_total, axis=0)
        mus = np.delete(mus, lost_total, axis=0)

        return points[:, 0], points[:, 1], points[:, 2], vpars_init, mus, lost_total

    def trace_particles(self):
        r"""
        Trace all particles and compute per-particle chaos/loss diagnostics.

        Returns:
            DAs : Per-particle chaos metric values.
            lost_total : Indices of wall-lost particles.
            Peta_start : Initial map-coordinate values per particle.
            pitch_initial : Initial pitch-like values per particle.
        """
        import pickle

        if self.check_filepaths(self.res_filepaths):
            if self.verbose:
                print("Reading File", flush=True)
            with open(self.res_filepaths["tys"], "rb") as f:
                res_tys = pickle.load(f)
            with open(self.res_filepaths["hits"], "rb") as f:
                res_hits = pickle.load(f)
            if self.verbose:
                print("Read Files", flush=True)
            DAs, lost_total, Peta_start, pitch_initial = self.build_lists(
                res_tys, res_hits
            )

            return DAs, lost_total, Peta_start, pitch_initial
        if self.verbose:
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
                perturbed_field=self.saw,
                stz_inits=point,
                parallel_speeds=vpar,
                mus=mu,
                tmax=self.tmax,
                mass=self.mass,
                charge=self.charge,
                Ekin=self.Ekin,
                abstol=1e-9,
                reltol=1e-9,
                stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
                mode="gc_noK",
                ODE_solver="dormand_prince",
                **self.solver_options,
            )
            points_trajectory = gc_tys[0]
            time_momentum, s_path, theta_path, zeta_path, vpar_path = (
                points_trajectory[:, 0],
                points_trajectory[:, 1],
                points_trajectory[:, 2],
                points_trajectory[:, 3],
                points_trajectory[:, 4],
            )
            points_trajectory = np.column_stack(
                (s_path, theta_path, zeta_path, time_momentum)
            )
            idx_wall = np.argmax(s_path >= 1) if np.any(s_path >= 1) else None
            if idx_wall is not None and s_path[idx_wall] >= 1:
                idx_wall -= 1
                points_trajectory = points_trajectory[:idx_wall, :]
                vpar_path = vpar_path[:idx_wall]

            self.saw.set_points(points_trajectory)
            modB = self.saw.B0.modB()[:, 0]
            weighted_mu = self.mus[itrj] * self.mass
            E = (
                0.5 * self.mass * vpar_path**2
                + self.mass * mu[0] * modB
                + self.charge * self.saw.Phi()[:, 0]
            )
            Peta_values = compute_peta(
                self.saw,
                points_trajectory,
                vpar_path,
                self.mass,
                self.charge,
                self.helicity_M,
                self.helicity_N,
                self.helicity_Mp,
                self.helicity_Np,
            )
            Eprime = self.nprime * E - self.omega * Peta_values

            if points_trajectory.shape[0] < 8:
                # start state vector:  [t, s, theta, zeta, vpar, peta, E, mu, Eprime]
                # end state vector:   [t, s, theta, zeta, vpar, peta, E]
                start_state = [
                    0,
                    point[0, 0],
                    point[0, 1],
                    point[0, 2],
                    vpar[0],
                    Peta_values[0],
                    E[0],
                    weighted_mu,
                    Eprime[0],
                ]

                if gc_zeta_hits[0][0][1] == -1:
                    end_state = [
                        gc_zeta_hits[0][0][0],
                        point[-1, 0],
                        point[-1, 1],
                        point[-1, 2],
                        vpar[-1],
                        Peta_values[-1],
                        E[-1],
                    ]
                else:
                    end_state = [
                        point[-1, 3],
                        point[-1, 0],
                        point[-1, 1],
                        point[-1, 2],
                        vpar[-1],
                        Peta_values[-1],
                        E[-1],
                    ]

                particle_out = [start_state, end_state]
                res_tys.append(particle_out)
                res_hits.append(gc_zeta_hits[0])
                continue

            s_mean = np.mean(s_path)

            dt = np.diff(time_momentum)

            d_eff_0 = (
                0.5 * np.mean((Peta_values[1:] - Peta_values[0]) ** 2) / np.mean(dt)
            )

            average_peta = np.mean(Peta_values)
            stack_data = np.column_stack((points_trajectory[:, 3], Peta_values))
            time_eval, DA_eval = return_DA(stack_data)

            end_points = points_trajectory[-1, :-1]
            start_points = points_trajectory[0, :-1]

            diffusion_data = [d_eff_0]
            mean_data = [s_mean, d_eff_0, average_peta, np.mean(E)]

            v_par_signs = np.sign(vpar_path)
            v = v_par_signs[v_par_signs != 0]
            bounces = np.sum(v[1:] * v[:-1] < 0) if len(v) > 2 else 1

            start_phasespace = [
                vpar_path[0],
                Peta_values[0],
                E[0],
                weighted_mu,
                Eprime[0],
            ]
            end_phasespace = [vpar_path[-1], Peta_values[-1], E[-1], Eprime[-1]]
            # puts time back in front
            # t, s, theta, zeta, vpar, peta, E, Eprime
            end_state = (
                [points_trajectory[-1, -1].tolist()]
                + end_points.tolist()
                + end_phasespace
            )
            # t, s, theta, zeta, vpar, peta, E, mu, Eprime
            start_state = (
                [points_trajectory[0, -1].tolist()]
                + start_points.tolist()
                + start_phasespace
            )
            particle_out = [
                start_state,
                end_state,
                diffusion_data,
                mean_data,
                DA_eval,
                bounces,
                weighted_mu,
            ]
            res_tys.append(particle_out)
            res_hits.append(gc_zeta_hits[0])
        print(f"{self.comm.rank=} done tracing particles", flush=True)

        if self.comm is not None:
            res_tys = [i for o in self.comm.allgather(res_tys) for i in o]
            res_hits = [i for o in self.comm.allgather(res_hits) for i in o]

        with open(self.res_filepaths["tys"], "wb") as f:
            pickle.dump(res_tys, f)
        with open(self.res_filepaths["hits"], "wb") as f:
            pickle.dump(res_hits, f)
        DAs, lost_total, Peta_start, pitch_initial = self.build_lists(res_tys, res_hits)
        return DAs, lost_total, Peta_start, pitch_initial

    def build_lists(self, res_tys, res_hits):
        r"""
        Build output metric lists from raw trajectory and hit arrays.

        Args:
            res_tys : Stored per-particle trajectory summaries.
            res_hits : Stored stopping/hit summaries.
        Returns:
            DAs : Per-particle chaos metric values.
            lost_total : Indices of wall-lost particles.
            Peta_start : Initial map-coordinate values per particle.
            pitch_initial : Initial pitch-like values per particle.
        """
        if self.verbose:
            print("Building Lists", flush=True)
        lost_total = []
        for i in range(len(res_hits)):
            if res_hits[i].size > 0 and int(res_hits[i][0][1]) == -1:
                lost_total.append(i)

        DAs = []
        Peta_start = []
        pitch_initial = []
        bounces = []
        lost = []

        for elem in res_tys:
            # index 0: start state:
            # index 1: end state
            # index 2: diffusion data
            # index 3: mean data
            # index 4: DA value
            # index 5: number of bounces
            # start state vector:  [t, s, theta, zeta, vpar, peta, E, mu, Eprime]
            # end state vector:   [t, s, theta, zeta, vpar, peta, E]
            start = elem[0]
            end = elem[1]

            if self.plot_s:
                Peta_start.append(start[1])
            else:
                Peta_start.append(start[5])

            if self.Eprime_slice:
                pitch_val = float(start[7]) / self.Eprime
            else:
                pitch_val = float(start[7]) / self.Ekin
            pitch_val *= np.sign(start[4])
            pitch_initial.append(pitch_val)
            if end[0] < (self.tmax - 2e-7):
                lost.append(1)
            else:
                lost.append(0)
            if len(elem) > 2:
                if self.diffusion:
                    DAs.append(elem[2][0])  # diffusion data
                else:
                    DAs.append(elem[4])
                bounces.append(elem[5])
            else:
                DAs.append(np.nan)
                bounces.append(0)

        self.DA_final = DAs
        self.res_tys = res_tys
        self.bounces = bounces

        # self.lost_pitch =
        self.lost = lost
        if self.verbose:
            print("Done Building Lists", flush=True)
        # self.da_values, self.wall_lost, self.surfaces, self.pitch_angles =
        return DAs, lost_total, Peta_start, pitch_initial

    def s_peta_map(self, s, mu, sign):
        r"""
        Map a point to canonical momentum p_eta using current settings.

        Args:
            s : Radial-like coordinate.
            mu : Magnetic moment.
            sign : Desired sign for parallel velocity.
        Returns:
            peta : Canonical momentum value at the requested point.
        """
        points = np.zeros((3, 1))
        points[0, :] = s

        vp_temp = self.vpar_func_perturbed(
            points[0, 0], points[0, 1], points[0, 2], mu, sign
        )

        peta = compute_peta(
            self.B0,
            points,
            vp_temp,
            self.mass,
            self.charge,
            self.helicity_M,
            self.helicity_N,
        )
        return peta

    def plot_surfaces(
        self,
        nx=20,
        ny=20,
        savepath="heatmap_digit_accuracy.png",
        ax=None,
        DA_max=7,
        plot_losses=False,
    ):
        r"""
        Plot 2D phase-space heatmaps and trapped/loss boundaries.

        Args:
            nx : Number of bins in the pitch-like direction.
            ny : Number of bins in the surface/peta direction.
            savepath : Output path for the saved figure.
            ax : Optional Matplotlib axis.
            DA_max : Maximum DA value used for colormap normalization.
            plot_losses : Overlay wall-loss markers
        Returns:
            ax : Matplotlib axis containing the rendered plot.
        """
        import cmcrameri.cm as cmc
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from scipy.stats import binned_statistic_2d

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 12))
        else:
            fig = ax.get_figure()

        # def surf_trapped_s(pitch_angle, surface):

        def surf_trapped_func(pitch_angle, surface):
            r"""
            Evaluate trapped condition and map coordinate at one surface slice.

            Args:
                pitch_angle : Pitch-like parameter value.
                surface : Flux-surface label.
            Returns:
                trapped : Trapped mask/list for sampled points.
                map_coord : Corresponding mapped peta list.
            """
            resolution = 100
            points = initialize_position_uniform_surf(self.B0, resolution, surface)
            self.B0.set_points(points)
            modB = self.B0.modB()[:, 0]
            mmbB = np.max(modB)

            points = initialize_position_uniform_surf(
                self.B0, int(resolution / 5), surface
            )

            self.B0.set_points(points)
            modB = self.B0.modB()[:, 0]
            sgn = np.sign(pitch_angle)
            if self.Eprime_slice:
                mu = np.abs(pitch_angle * self.Eprime)
            else:
                mu = np.abs(pitch_angle * self.Ekin) / self.min_volmodB
            pitch_angle = mu / self.Ekin

            vpars_temp = []
            if self.Eprime_slice:
                for i in range(points.shape[0]):
                    vp_temp = self.vpar_func_perturbed(
                        points[i, 0], points[i, 1], points[i, 2], mu, sgn
                    )
                    if vp_temp is not None:
                        vpars_temp.append(vp_temp[0])
                    else:
                        vpars_temp.append(np.nan)
                vpars = np.array(vpars_temp)

                mask = ~np.isnan(vpars)
                vpars = vpars[mask]
                points = points[mask]
            else:
                vpars = np.sqrt(self.Ekin - mu * modB)

            peta = compute_peta(
                self.B0,
                points,
                vpars,
                self.mass,
                self.charge,
                self.helicity_M,
                self.helicity_N,
                self.helicity_Mp,
                self.helicity_Np,
            )
            mmbB = np.max(modB)
            if self.Eprime_slice:
                E = (1 / self.nprime) * (self.Eprime + self.omega * peta)
            else:
                E = np.array(self.Ekin)
            if self.plot_s:
                if np.min(E) - mu * mmbB < 0:
                    return [1], [surface]
                else:
                    return [0], [surface]
            else:
                return ((E - mu * mmbB) < 0).astype(int).tolist(), peta.tolist()

        if self.diffusion:
            norm = mpl.colors.Normalize(
                vmin=min(self.DA_final), vmax=max(self.DA_final)
            )
            cmap = "viridis"
        else:
            norm = mpl.colors.Normalize(vmin=0, vmax=DA_max)
            cmap = "cmc.managua"

        stat, x_edges, y_edges, binnumber = binned_statistic_2d(
            np.array(self.pitch_angles),
            np.array(self.surfaces),
            np.array(self.DA_final),
            statistic="mean",
            bins=[nx, ny],
        )

        X2, Y2 = np.meshgrid(x_edges, y_edges)
        im2 = ax.pcolormesh(X2, Y2, stat.T, shading="auto", cmap=cmap, norm=norm)
        oldtrapped = False
        plotrapped = True

        if plotrapped:
            if oldtrapped:
                stat_bounce, x_edges_bounce, y_edges_bounce, binnumber_bounce = (
                    binned_statistic_2d(
                        np.array(self.pitch_angles),
                        np.array(self.surfaces),
                        np.array(self.bounces),
                        statistic="max",
                        bins=[nx, ny],
                    )
                )
                bounce_mask = (stat_bounce >= 1).astype(float)
                x_centers = 0.5 * (x_edges_bounce[:-1] + x_edges_bounce[1:])
                y_centers = 0.5 * (y_edges_bounce[:-1] + y_edges_bounce[1:])
                Xc, Yc = np.meshgrid(x_centers, y_centers)

            else:
                volume_boundary_peta = []
                volume_boundary_pitch = []
                volume_trapped = []
                space = 50
                if self.verbose:
                    print("making trapped boundary...", flush=True)
                for s_val in np.linspace(0, 1, space):
                    for pitch_val in np.linspace(
                        np.min(self.pitch_angles), np.max(self.pitch_angles), space
                    ):
                        trapped, peta = surf_trapped_func(pitch_val, s_val)
                        if trapped is None:
                            continue
                        if any(peta > max(np.array(self.surfaces))):
                            continue
                        if any(peta < min(np.array(self.surfaces))):
                            continue

                        pitch_lst = [pitch_val] * len(peta)
                        if self.plot_s:
                            volume_boundary_peta.append(s_val)
                            volume_trapped.append(np.sum(trapped))
                            volume_boundary_pitch.append(pitch_val)
                        else:
                            volume_boundary_peta += peta
                            volume_trapped += trapped
                            pitch_lst = [pitch_val] * len(peta)
                            volume_boundary_pitch += pitch_lst
                if self.verbose:
                    print("making trapped boundary...", flush=True)

                stat_bounce, x_edges_bounce, y_edges_bounce, binnumber_bounce = (
                    binned_statistic_2d(
                        np.array(volume_boundary_pitch),
                        np.array(volume_boundary_peta),
                        np.array(volume_trapped),
                        statistic="max",
                        bins=[space, space],
                    )
                )
            bounce_mask = (stat_bounce >= 1).astype(float)
            x_centers = 0.5 * (x_edges_bounce[:-1] + x_edges_bounce[1:])
            y_centers = 0.5 * (y_edges_bounce[:-1] + y_edges_bounce[1:])
            Xc, Yc = np.meshgrid(x_centers, y_centers)

            CS = ax.contour(
                Xc,
                Yc,
                bounce_mask.T,
                levels=[0.5],
                colors="gray",
                linewidths=10,
                linestyles="-",
            )
            self._pt_contour = CS

            paths = CS.collections[0].get_paths()
            if len(paths) > 1:

                def poly_area(verts):
                    # verts: (N,2) array
                    x = verts[:, 0]
                    y = verts[:, 1]
                    return 0.5 * abs(
                        (x[:-1] * y[1:]).sum()
                        + x[-1] * y[0]
                        - (y[:-1] * x[1:]).sum()
                        - y[-1] * x[0]
                    )

                main_path = max(paths, key=lambda p: poly_area(p.vertices))
                main_verts = main_path.vertices
                ax.plot(main_verts[:, 0], main_verts[:, 1], color="white", lw=10)
                for coll in CS.collections:
                    coll.remove()
                # store for clipping later if you want
                self._pt_contour = main_path
            from scipy.interpolate import RegularGridInterpolator

            self._pt_mask_interp = RegularGridInterpolator(
                (x_centers, y_centers),  # (pitch grid, peta grid)
                bounce_mask,  # shape (len(x_centers), len(y_centers))
                bounds_error=False,
                fill_value=0.0,
            )
            self._pt_mask_level = 0.5

            # ax.scatter(
            #    self.equilibrium_lost_pitch, self.equilibrium_lost_surfaces,
            #    marker='x',
            #    s=200,
            #    c='red',
            #    label='Equilibrium lost particles',
            # )
        colorlabel = r"$D_{eff}$" if self.diffusion else "Digit Accuracy"

        if plot_losses:
            lost_frac, x_edges, y_edges, _ = binned_statistic_2d(
                np.array(self.pitch_angles),
                np.array(self.surfaces),
                np.array(self.lost),
                statistic="mean",
                bins=[nx, ny],
            )
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            Xc, Yc = np.meshgrid(x_centers, y_centers)
            xf = Xc.ravel()
            yf = Yc.ravel()
            af = lost_frac.T.ravel()
            mask = np.isfinite(af)
            af_plot = af[mask]
            ax.scatter(
                xf[mask],
                yf[mask],
                marker="^",
                s=500,
                c="white",
                edgecolors="k",
                linewidths=1,
                alpha=af_plot,
                zorder=10,
            )
        if self.Eprime_slice:
            ax.set_xlabel(r"$\lambda = \frac{\mu}{E^\prime} \text{sign}(v_{||})$")
        else:
            ax.set_xlabel(r"$\lambda = \frac{\mu}{E} \text{sign}(v_{||})$")
        if self.plot_s:
            ax.set_ylabel(r"$s$")
        else:
            ax.set_ylabel(r"$P_\eta$")

        fig.tight_layout()
        fig.colorbar(im2, ax=ax, label=colorlabel)
        plt.savefig(savepath, dpi=400)
        return ax


class WBAParticles:
    def __init__(
        self,
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
        savedata=(False, "DATA/"),
        tmax=1e-2,
        min_timestep=1e-6,
        comm=None,
        DA_cutoff=3,
        skipped_particles=None,
        solver_options=None,
        nconvergence_points=1,
    ):
        """
        Initialize weighted Birkhoff analysis/tracing for perturbed
        guiding-center trajectories.

        This class takes a user-supplied set of initial particle conditions,
        traces each particle in a SAW or ShearAlfvenWaveSuperposition, computes
        a weighted Birkhoff average diagnostic from the resulting
        time series of P_eta, and records which particles are lost to the wall
        before the final tracing time.

        The class does not generate initial conditions internally, but expects:
            - initial spatial coordinates in Boozer variables,
            - initial parallel velocities,
            - initial magnetic moments per unit mass.

        If saved output files already exist, previously computed DA values and
        wall-loss metadata are loaded instead of retracing particles. Otherwise,
        trajectories are computed and the resulting output saved.

        Args:
            saw:
                SAW or ShearAlfvenWaveSuperposition for trajectory integration.
            initial_conditions:
                Array of initial spatial particle coordinates, expected to have
                columns `[s, theta, zeta]`.
            v_pars:
                Initial parallel velocities.
            mu_per_mass:
                Initial magnetic moments divided by particle mass.
            mass:
                Particle mass.
            charge:
                Particle charge.
            Ekin:
                Reference kinetic energy passed to the orbit integrator.
            helicity_N:
                Toroidal helicity N entering chi = M * theta - N * zeta.
            helicity_M:
                Poloidal helicity M entering chi = M * theta - N * zeta.
            helicity_Mp:
                Poloidal helicity Mp entering eta = Mp * theta - Np * zeta.
                If None, it is chosen automatically together with
                `helicity_Np`.
            helicity_Np:
                Toroidal helicity Np entering eta = Mp * theta - Np * zeta.
                If None, it is chosen automatically together with
                `helicity_Mp`.
            mean:
                Flag controlling averaging-related binning behavior.
            savedata:
                Two-element tuple/list of the form `(save_flag, savepath)`.
                If `save_flag` is True, DA values and wall losses are
                read from / written to files in `savepath`.
            tmax:
                Maximum integration time for each particle trajectory.
            min_timestep:
                Reserved timestep-related parameter for consistency with other
                interfaces.
            comm:
                MPI communicator.
            DA_cutoff:
                Threshold used after tracing to classify particles as chaotic
                when computing  fractions.
            skipped_particles:
                List of particle indices to exclude from tracing. Typically,
                these are particles that are equilibrium lost. These are
                assigned NaN DA values and treated as pre-skipped entries.
            solver_options:
                Additional keyword arguments passed to the
                integrator.
        """

        if solver_options is None:
            solver_options = {}
        if skipped_particles is None:
            skipped_particles = []
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
        self.verbose = False
        if self.comm is None or self.comm.rank == 0:
            self.verbose = True

        self.solver_options = solver_options
        self.tmax = tmax

        self.mean = mean
        self.savedata = savedata[0]
        self.savepath = savedata[1]
        self.convergence_points = nconvergence_points

        if self.savedata:
            self.IC_filepaths = {
                "s0": self.savepath + "uniform_s0.txt",
                "theta0": self.savepath + "uniform_theta0.txt",
                "zeta0": self.savepath + "uniform_zeta0.txt",
                "vpar0": self.savepath + "uniform_vpar0.txt",
                "mu_per_mass": self.savepath + "uniform_mu_per_mass.txt",
            }
            self.final_filepaths = {
                "DA": self.savepath + "DA.txt",
                "wall_lost": self.savepath + "wall_lost.txt",
            }

        self.skip = skipped_particles
        if not self.check_filepaths(self.final_filepaths):
            points_phase = np.append(
                initial_conditions, np.zeros((initial_conditions.shape[0], 1)), axis=1
            )
            self.gc_tys = self.trace_particles(saw, points_phase, v_pars, mu_per_mass)
            self.DAs, self.wall_lost_indicies, self.wall_lost_times = (
                self.quantify_chaos_and_losses(
                    trajectories=self.gc_tys, equilibrium_lost_indicies=self.skip
                )
            )
            np.savetxt(self.final_filepaths["DA"], np.array(self.DAs))
            np.savetxt(
                self.final_filepaths["wall_lost"],
                np.column_stack((self.wall_lost_indicies, self.wall_lost_times)),
            )
        else:
            if self.verbose:
                print("loaded existing data files", flush=True)
            self.DAs = np.loadtxt(self.final_filepaths["DA"]).tolist()
            wall_lost = np.loadtxt(self.final_filepaths["wall_lost"]).astype(int)
            self.wall_lost_indicies = wall_lost[:, 0].tolist()
            self.wall_lost_times = wall_lost[:, 1].tolist()
        self.numparticle = len(self.DAs) - len(self.skip)
        self.compute_fractions(DA_cutoff=DA_cutoff)

    def compute_fractions(self, DA_cutoff=3):
        r"""
        Return chaotic-particle fraction percentage for a DA threshold.

        Args:
            DA_cutoff : DA threshold used to classify chaos.
        Returns:
            uniform_fractional_chaotic : Percent chaotic fraction in the sample.
        """
        uniform_fractional_chaotic = [
            (
                sum(
                    [
                        1
                        for i in range(len(self.DAs))
                        if ((self.DAs[i] < DA_cutoff) or (i in self.wall_lost_indicies))
                    ]
                )
                / (self.numparticle)
            )
            * 100
        ]
        return uniform_fractional_chaotic

    def quantify_chaos_and_losses(self, trajectories, equilibrium_lost_indicies):
        r"""
        Compute per-particle DA values and wall-loss metadata.

        Args:
            trajectories : Stored trajectory summary per particle.
            equilibrium_lost_indicies : Indices removed by equilibrium losses.
        Returns:
            DA_list : DA value for each particle index.
            lost_total : Indices of particles that hit the wall.
            lost_times : Wall-hit times for lost particles.
        """
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
            trajectories[i][1][0]
            DA = trajectories[i][2]

            # check if particle lost to wall
            if final_time < (self.tmax - 2e-6):
                lost_total.append(int(i))
                lost_times.append(final_time)
            DA_list.append(DA)
        return DA_list, lost_total, lost_times

    def check_filepaths(self, filepaths):
        r"""
        Check whether all provided output file paths exist.

        Args:
            filepaths : Dictionary of file labels to filesystem paths.
        Returns:
            exists_all : True if every path exists, otherwise False.
        """
        return all(exists(fp) for fp in filepaths.values())

    def trace_particles(
        self,
        saw,
        points_phase,
        vpars,
        mus,
    ):
        r"""
        Trace perturbed particle trajectories and compute DA outputs.

        Args:
            saw : Perturbed field object used for tracing.
            points_phase : Initial states including phase/time column.
            vpars : Initial parallel velocities.
            mus : Initial magnetic moments.
        Returns:
            res_tys : Per-particle trajectory summaries.
        """
        first, last = parallel_loop_bounds(self.comm, points_phase.shape[0])
        res_tys = []
        res_hits = []

        for itrj in range(first, last):
            if itrj in self.skip:
                start_state = [
                    points_phase[itrj, 0],
                    points_phase[itrj, 1],
                    points_phase[itrj, 2],
                    0,
                ]
                particle_out = [start_state, start_state, np.nan]
                res_tys.append(particle_out)
                res_hits.append(np.array([]))
                continue
            gc_tys, gc_zeta_hits = trace_particles_boozer_perturbed(
                perturbed_field=saw,
                stz_inits=points_phase[itrj, :].reshape(1, 4),
                parallel_speeds=[vpars[itrj]],
                mus=[mus[itrj]],
                tmax=self.tmax,
                mass=self.mass,
                charge=self.charge,
                Ekin=self.Ekin,
                abstol=1e-9,
                reltol=1e-9,
                stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
                mode="gc_noK",
                ODE_solver="dormand_prince",
                **self.solver_options,
            )

            points_trajectory = gc_tys[0]
            time_momentum, s_path, theta_path, zeta_path, vpar_path = (
                points_trajectory[:, 0],
                points_trajectory[:, 1],
                points_trajectory[:, 2],
                points_trajectory[:, 3],
                points_trajectory[:, 4],
            )
            points_trajectory = np.column_stack(
                (s_path, theta_path, zeta_path, time_momentum)
            )
            idx_wall = np.argmax(s_path >= 1) if np.any(s_path >= 1) else None
            if idx_wall is not None and s_path[idx_wall] >= 1:
                idx_wall -= 1
                points_trajectory = points_trajectory[:idx_wall, :]
                vpar_path = vpar_path[:idx_wall]

            if points_trajectory.shape[0] < 8:
                start_state = points_trajectory[0, :].tolist()
                end_state = points_trajectory[-1, :].tolist()
                particle_out = [start_state, end_state, np.nan]
                res_tys.append(particle_out)
                res_hits.append(gc_zeta_hits[0])
                continue

            Peta_values = compute_peta(
                saw,
                points_trajectory,
                vpar_path,
                self.mass,
                self.charge,
                self.helicity_M,
                self.helicity_N,
                self.helicity_Mp,
                self.helicity_Np,
            )

            stack_data = np.column_stack((points_trajectory[:, 3], Peta_values))
            time_eval, DA_eval = return_DA(stack_data)

            first_slice = points_trajectory[0, :]
            last_slice = points_trajectory[-1, :]
            particle_out = [first_slice.tolist(), last_slice.tolist(), DA_eval]
            res_tys.append(particle_out)
            res_hits.append(gc_zeta_hits[0])
        if self.comm is not None:
            res_tys = [i for o in self.comm.allgather(res_tys) for i in o]
        if self.verbose:
            with open(self.savepath + "_data.pkl", "wb") as f:
                import pickle

                pickle.dump(res_tys, f)
        return res_tys


class WBAUnPertParticles:
    def __init__(
        self,
        B0,
        initial_conditions,
        v_pars,
        mass,
        charge,
        Ekin,
        helicity_N,
        helicity_M,
        helicity_Mp=None,
        helicity_Np=None,
        mean=True,
        savedata=(False, "DATA/"),
        tmax=1e-2,
        min_timestep=1e-6,
        comm=None,
        DA_cutoff=3,
        skipped_particles=None,
        solver_options=None,
        # nconvergence_points=1
    ):
        """
        Initialize weighted-Birkhoff analysis for an ensemble of unperturbed
        guiding center trajectories.

        This class takes a user-supplied set of initial particle conditions,
        traces each particle in an equilibrium Boozer magnetic field, computes a
        weighted Birkhoff Digit accuracy (DA) diagnostic from the resulting
        time series of P_eta, and records which particles are lost to the wall
        before the requested final integration time.

        Unlike `WBAParticles`, this class uses the unperturbed tracer
        `trace_particles_boozer`.

        If saved output files already exist, previously computed DA values and
        wall-loss metadata are loaded instead of retracing particles. Otherwise,
        trajectories are computed and the resulting output is saved.

        Args:
            B0:
                Equilibrium Boozer magnetic field used for particle
                tracing and for evaluation of `P_eta`.
            initial_conditions:
                Array of initial spatial particle coordinates, expected to have
                columns `[s, theta, zeta]`.
            v_pars:
                Initial parallel velocities.
            mass:
                Particle mass.
            charge:
                Particle charge.
            Ekin:
                Reference kinetic energy passed to the orbit integrator.
            helicity_N:
                Toroidal helicity N entering chi = M * theta - N * zeta.
            helicity_M:
                Poloidal helicity M entering chi = M * theta - N * zeta.
            helicity_Mp:
                Poloidal helicity Mp entering eta = Mp * theta - Np * zeta.
                If None, it is chosen automatically together with
                `helicity_Np`.
            helicity_Np:
                Toroidal helicity Np entering eta = Mp * theta - Np * zeta.
                If None, it is chosen automatically together with
                `helicity_Mp`.
            mean:
                Flag controlling averaging-related behavior in binning.
            savedata:
                Two-element tuple/list of the form `(save_flag, savepath)`.
                If `save_flag` is True, DA values and lost particles are
                read from / written to files in `savepath`.
            tmax:
                Maximum integration time for each particle trajectory.
            min_timestep:
                Integration timestep.
            comm:
                MPI communicator.
            DA_cutoff:
                Threshold used after tracing to classify particles as chaotic
                when computing aggregate fractions.
            skipped_particles:
                List of particle indices to exclude from tracing. These are
                assigned NaN DA values and treated as pre-skipped entries.
            solver_options:
                Additional keyword arguments passed to the unperturbed orbit
                integrator.
        """

        if solver_options is None:
            solver_options = {}
        if skipped_particles is None:
            skipped_particles = []
        self.B0 = B0
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
        self.verbose = False
        if self.comm is None or self.comm.rank == 0:
            self.verbose = True

        self.solver_options = solver_options
        self.tmax = tmax

        self.mean = mean
        self.savedata = savedata[0]
        self.savepath = savedata[1]
        self.convergence_points = 1  # nconvergence_points

        if self.savedata:
            self.IC_filepaths = {
                "s0": self.savepath + "uniform_s0.txt",
                "theta0": self.savepath + "uniform_theta0.txt",
                "zeta0": self.savepath + "uniform_zeta0.txt",
                "vpar0": self.savepath + "uniform_vpar0.txt",
                "mu_per_mass": self.savepath + "uniform_mu_per_mass.txt",
            }
            self.final_filepaths = {
                "DA": self.savepath + "DA.txt",
                "wall_lost": self.savepath + "wall_lost.txt",
            }

        self.skip = skipped_particles
        if not self.check_filepaths(self.final_filepaths):
            self.gc_tys = self.trace_particles(B0, initial_conditions, v_pars)
            self.DAs, self.wall_lost_indicies, self.wall_lost_times = (
                self.quantify_chaos_and_losses(
                    trajectories=self.gc_tys, equilibrium_lost_indicies=self.skip
                )
            )
            np.savetxt(self.final_filepaths["DA"], np.array(self.DAs))
            np.savetxt(
                self.final_filepaths["wall_lost"],
                np.column_stack((self.wall_lost_indicies, self.wall_lost_times)),
            )
        else:
            if self.verbose:
                print("loaded existing data files", flush=True)
            self.DAs = np.loadtxt(self.final_filepaths["DA"]).tolist()
            wall_lost = np.loadtxt(self.final_filepaths["wall_lost"]).astype(int)
            self.wall_lost_indicies = wall_lost[:, 0].tolist()
            self.wall_lost_times = wall_lost[:, 1].tolist()
        self.numparticle = len(self.DAs) - len(self.skip)
        self.compute_fractions(DA_cutoff=DA_cutoff)

    def compute_fractions(self, DA_cutoff=3):
        r"""
        Return chaotic-particle fraction percentage for a DA threshold.

        Args:
            DA_cutoff : DA threshold used to classify chaos.
        Returns:
            uniform_fractional_chaotic : Percent chaotic fraction in the sample.
        """
        uniform_fractional_chaotic = [
            (
                sum(
                    [
                        1
                        for i in range(len(self.DAs))
                        if ((self.DAs[i] < DA_cutoff) or (i in self.wall_lost_indicies))
                    ]
                )
                / (self.numparticle)
            )
            * 100
        ]
        return uniform_fractional_chaotic

    def quantify_chaos_and_losses(self, trajectories, equilibrium_lost_indicies):
        r"""
        Compute per-particle DA values and wall-loss metadata.

        Args:
            trajectories : Stored trajectory summary per particle.
            equilibrium_lost_indicies : Indices removed by equilibrium losses.
        Returns:
            DA_list : DA value for each particle index.
            lost_total : Indices of particles that hit the wall.
            lost_times : Wall-hit times for lost particles.
        """
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
            trajectories[i][1][0]
            DA = trajectories[i][2]

            # check if particle lost to wall
            if final_time < (self.tmax - 2e-6):
                lost_total.append(int(i))
                lost_times.append(final_time)
            DA_list.append(DA)
        return DA_list, lost_total, lost_times

    def check_filepaths(self, filepaths):
        r"""
        Check whether all provided output file paths exist.

        Args:
            filepaths : Dictionary of file labels to filesystem paths.
        Returns:
            exists_all : True if every path exists, otherwise False.
        """
        return all(exists(fp) for fp in filepaths.values())

    def trace_particles(self, field, points_phase, vpars):
        r"""
        Trace unperturbed particle trajectories and compute DA outputs.

        Args:
            field : Equilibrium field object used for tracing.
            points_phase : Initial particle positions in Boozer coordinates.
            vpars : Initial parallel velocities.
        Returns:
            res_tys : Per-particle trajectory summaries.
        """
        first, last = parallel_loop_bounds(self.comm, points_phase.shape[0])
        res_tys = []
        res_hits = []

        for itrj in range(first, last):
            if itrj in self.skip:
                start_state = [
                    points_phase[itrj, 0],
                    points_phase[itrj, 1],
                    points_phase[itrj, 2],
                ]
                particle_out = [start_state, start_state, np.nan]
                res_tys.append(particle_out)
                res_hits.append(np.array([]))
                continue

            pt = np.zeros((1, 3))
            pt[0, 0] = points_phase[itrj, 0]
            pt[0, 1] = points_phase[itrj, 1]
            pt[0, 2] = points_phase[itrj, 2]
            self.vtotal = np.sqrt(2 * self.Ekin / self.mass)
            gc_tys, gc_zeta_hits = trace_particles_boozer(
                field,
                stz_inits=pt,
                parallel_speeds=[vpars[itrj]],
                tmax=self.tmax,
                mass=self.mass,
                charge=self.charge,
                Ekin=self.Ekin,
                abstol=1e-9,
                reltol=1e-9,
                stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
                mode="gc_noK",
                ODE_solver="dormand_prince",
                **self.solver_options,
            )

            points_trajectory = gc_tys[0]
            time_momentum = points_trajectory[:, 0]
            s_path = points_trajectory[:, 1]
            theta_path = points_trajectory[:, 2]
            zeta_path = points_trajectory[:, 3]
            vpar_path = points_trajectory[:, 4]
            points_trajectory = np.column_stack(
                (s_path, theta_path, zeta_path, time_momentum)
            )
            idx_wall = np.argmax(s_path >= 1) if np.any(s_path >= 1) else None
            if idx_wall is not None and s_path[idx_wall] >= 1:
                idx_wall -= 1
                points_trajectory = points_trajectory[:idx_wall, :]
                vpar_path = vpar_path[:idx_wall]

            if points_trajectory.shape[0] < 8:
                start_state = points_trajectory[0, :].tolist()
                end_state = points_trajectory[-1, :].tolist()
                particle_out = [start_state, end_state, np.nan]
                res_tys.append(particle_out)
                res_hits.append(gc_zeta_hits[0])
                continue

            traj = points_trajectory[:, :-1]

            Peta_values = compute_peta(
                self.B0,
                points=traj,
                vpar=vpar_path,
                mass=self.mass,
                charge=self.charge,
                helicity_M=self.helicity_M,
                helicity_N=self.helicity_N,
                helicity_Mp=self.helicity_Mp,
                helicity_Np=self.helicity_Np,
            )
            stack_data = np.column_stack((points_trajectory[:, 3], Peta_values))
            time_eval, DA_eval = return_DA(stack_data)

            first_slice = points_trajectory[0, :]
            last_slice = points_trajectory[-1, :]
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
