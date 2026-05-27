from os.path import exists
from warnings import warn

import numpy as np
from scipy import integrate
from scipy.stats import binned_statistic_2d

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


def min_volumemodB(B0):
    r"""
    Estimate minimum magnetic-field magnitude over sampled surfaces
    by evaluating |B| on a uniform grid of flux surfaces.

    Args:
        B0 : The :class:`BoozerMagneticField` instance to evaluate.

    Returns:
        min_modB : Approximate minimum value of |B| in the sampled volume.
    """
    resolution = 1000

    for surface in np.linspace(0.01, 0.99, resolution):
        if surface == 0.01:
            points = initialize_position_uniform_surf(B0, resolution, surface)
        else:
            sampled_surface = initialize_position_uniform_surf(B0, resolution, surface)
            points = np.concatenate((points, sampled_surface), axis=0)

    B0.set_points(points)
    modB = B0.modB()[:, 0]
    return np.min(modB)


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
        chaos_detection=False,
        nconvergence_points=None,
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
            sign_vpar : Sign of the parallel velocity (+1 or -1).
            mass : Particle mass.
            charge : Particle charge.
            Ekin : Particle total kinetic energy.
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
            helicity_M : Poloidal helicity of the field-strength contours.
                         Required when computing the canonical momentum
                         :math:`p_{\eta}` or performing chaos detection.
            helicity_N : Toroidal helicity of the field-strength contours.
                         Required when computing the canonical momentum
                         :math:`p_{\eta}` or performing chaos detection.
            helicity_Mp : Poloidal helicity of the mapping coordinate eta.
                          If None, determined automatically from helicity_M.
            helicity_Np : Toroidal helicity of the mapping coordinate eta.
                          If None, determined automatically from helicity_N.
            chaos_detection : If True, compute the Weighted Birkhoff Average
                              (WBA) digit accuracy along each trajectory
                              (default: False).
            nconvergence_points : Number of WBA evaluations per trajectory
                                  used to assess convergence of the chaos
                                  detection metric. If None and
                                  chaos_detection=True, a single evaluation
                                  at the end of the trajectory is used.
        """
        if solver_options is None:
            solver_options = {}
        if sign_vpar not in [-1, 1]:
            raise ValueError("sign_vpar should be either -1 or +1")

        self.helicity_N = helicity_N
        self.helicity_M = helicity_M

        if helicity_N is None or helicity_M is None:
            self.peta_profile = False
        else:
            self.peta_profile = True
            if helicity_Mp is None or helicity_Np is None:
                # If modB contours close poloidally,
                # use theta as mapping coordinate
                if self.helicity_M == 0:
                    helicity_Mp = 1
                    helicity_Np = 0
                # Otherwise, use zeta as mapping coordinate
                else:
                    helicity_Mp = 0
                    helicity_Np = field.nfp
            self.helicity_Mp = helicity_Mp
            self.helicity_Np = helicity_Np

        if (self.helicity_M is None or self.helicity_N is None) and chaos_detection:
            raise ValueError(
                "helicity_M and helicity_N must be provided for chaos detection."
            )

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
        self.DA_poinc = chaos_detection
        self.nconvergence_points = nconvergence_points
        self.solver_options = solver_options
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

        self.vpars_init, self.s_init, self.thetas_init = self.initialize_passing_map(
            s_flat, thetas_flat
        )
        (
            self.s_all,
            self.thetas_all,
            self.vpars_all,
            self.t_all,
            self.peta_all,
            self.DA_all,
            self.DA_times,
        ) = self.compute_passing_map()

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
                MaxToroidalFluxStoppingCriterion(0.99),
            ],
            forget_exact_path=False,
            vpars_stop=True,
            phases_stop=True,
            **self.solver_options,
        )
        if len(res_hits[0]) == 0:
            raise RuntimeError("No stopping criterion reached in passing_map.")

        res_hit = res_hits[0][0, :]  # Only check the first hit or stopping criterion
        time_momentum = res_tys[0][:, 0]

        points_traj = np.zeros((res_tys[0].shape[0], 3))
        points_traj[:, 0] = res_tys[0][:, 1]
        points_traj[:, 1] = res_tys[0][:, 2]
        points_traj[:, 2] = res_tys[0][:, 3]
        vpar_path = res_tys[0][:, 4]
        if self.peta_profile:
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
            peta = np.column_stack((time_momentum, peta))

        if res_hit[1] == 0:  # Check that the zetas=[0] plane was hit
            point[0] = res_hit[2]
            point[1] = res_hit[3]
            point[2] = res_hit[5]
            time = res_hit[0]
            if self.peta_profile:
                return point, time, peta
            else:
                return point, time, []
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
        DA_all = []
        DA_times = []
        t_all = []
        if (Ntrj == 1) and self.comm is None:
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

            if self.peta_profile:
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
                Peta = np.array([[0, peta[0]]])
            else:
                peta_traj = []

            thetas_traj = [tr[1]]
            vpars_traj = [tr[2]]
            particle_DAs = []
            particle_DA_times = []
            t_traj = [0]
            for _jj in range(self.Nmaps):
                try:
                    tr, time, Peta_iter = self.passing_map(tr)

                    if self.peta_profile:
                        peta_traj.append(Peta_iter[-1, 1])
                        # shift time column by orbit time
                        Peta_iter[:, 0] += Peta[-1, 0]
                        Peta = np.vstack((Peta, Peta_iter[1:, :]))
                    else:
                        peta_traj.append(np.nan)

                    t_traj.append(time)
                    s_traj.append(tr[0])

                    thetas_traj.append(tr[1])
                    vpars_traj.append(tr[2])
                    if self.DA_poinc and _jj in self.WBA_transit_steps:
                        time_at_evaluation, DA_at_evaluation = return_DA(Peta)
                        particle_DAs.append(DA_at_evaluation)
                        particle_DA_times.append(_jj)
                except RuntimeError:
                    break
            if self.peta_profile:
                peta_all.append(peta_traj)
            s_all.append(s_traj)
            thetas_all.append(thetas_traj)
            vpars_all.append(vpars_traj)
            t_all.append(t_traj)
            DA_all.append(particle_DAs)
            DA_times.append(particle_DA_times)

        if self.comm is not None:
            peta_all = [i for o in self.comm.allgather(peta_all) for i in o]
            s_all = [i for o in self.comm.allgather(s_all) for i in o]
            thetas_all = [i for o in self.comm.allgather(thetas_all) for i in o]
            vpars_all = [i for o in self.comm.allgather(vpars_all) for i in o]
            t_all = [i for o in self.comm.allgather(t_all) for i in o]
            DA_all = [i for o in self.comm.allgather(DA_all) for i in o]
            DA_times = [i for o in self.comm.allgather(DA_times) for i in o]

        return s_all, thetas_all, vpars_all, t_all, peta_all, DA_all, DA_times

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

            sort_idx = np.argsort(peta_prof)
            peta_prof = peta_prof[sort_idx]
            omega_theta_prof = omega_theta_prof[sort_idx]
            omega_zeta_prof = omega_zeta_prof[sort_idx]
            s_prof = s_prof[sort_idx]
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

    def plot_poincare(
        self,
        ax=None,
        filename="passing_poincare.pdf",
        colorbar=True,
        DA_max=7,
        title="",
    ):
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
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable

        mpl.use("Agg")  # Don't use interactive backend
        try:
            import cmcrameri.cm as cmc  # noqa: F401

            cmap = "cmc.managua"
        except ImportError:
            cmap = "viridis"

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        def normalize(numbers):
            if not numbers:
                return []
            min_val, max_val = 0, DA_max
            normalized_numbers = [(x - min_val) / (max_val - min_val) for x in numbers]
            return normalized_numbers

        convergence_test_indicies = list(range(len(self.s_all)))
        if self.DA_poinc and self.nconvergence_points > 1:
            s_itrj_map = {}
            for itrj in convergence_test_indicies:
                s_itrj_map[itrj] = self.s_all[itrj][0]

            min_s = min(list(s_itrj_map.values()))
            max_s = max(list(s_itrj_map.values()))
            s_lst_true = list(s_itrj_map.values())
            cmap_s = mpl.colormaps["copper"].resampled(len(s_lst_true) ** 2)

        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$s$")
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 1])

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

        if self.DA_poinc:
            for i in range(len(self.thetas_all)):
                ax.scatter(
                    np.mod(self.thetas_all[i], 2 * np.pi),
                    self.s_all[i],
                    marker="o",
                    s=0.5,
                    c=cmap_object(DA_norm_all[i]),
                    edgecolors="none",
                )
            if colorbar:
                fig.colorbar(
                    ScalarMappable(
                        norm=plt.Normalize(0, DA_max), cmap=mpl.colormaps[cmap]
                    ),
                    ax=ax,
                    orientation="vertical",
                    label="Digit Accuracy",
                )
        else:
            for i in range(len(self.thetas_all)):
                ax.scatter(
                    np.mod(self.thetas_all[i], 2 * np.pi),
                    self.s_all[i],
                    marker="o",
                    s=2,
                    edgecolors="none",
                )
        if title != "":
            ax.set_title(title)
        fig.tight_layout()
        plt.savefig(filename)

        if self.DA_poinc and self.nconvergence_points > 1:
            fig_convergence, ax2 = plt.subplots(1, 1)
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
            fig_convergence.colorbar(
                ScalarMappable(norm=norm, cmap=cmap_s),
                ax=ax2,
                orientation="vertical",
                label="$s$",
            )

            fig_convergence.tight_layout()
            plt.savefig(filename[:-4] + "_convergence.pdf")
            plt.clf()

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
        helicity_Mp=None,
        helicity_Np=None,
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
        if (self.helicity_M is None) and (self.helicity_N is None) and chaos_detection:
            raise ValueError(
                "helicity_M and helicity_N must be provided for chaos detection."
            )

        if helicity_Mp is None or helicity_Np is None:
            # If modB contours close poloidally, then use theta as mapping coordinate
            if self.helicity_M == 0:
                helicity_Mp = 1
                helicity_Np = 0
            # Otherwise, use zeta as mapping coordinate
            else:
                helicity_Mp = 0
                helicity_Np = self.field.nfp
        self.helicity_Mp = helicity_Mp
        self.helicity_Np = helicity_Np

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
            peta : A numpy array of shape (N, 2) containing trajectory time and peta.
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
        points_traj = np.zeros((len(time_momentum), 3))
        points_traj[:, 0] = s_path
        points_traj[:, 1] = theta_path
        points_traj[:, 2] = zeta_path

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
        return point, time, peta

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

    def plot_poincare(
        self,
        ax=None,
        filename="trapped_poincare.pdf",
        convergence_test_indicies=None,
        DA_max=7,
    ):
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
        else:
            fig = ax.get_figure()

        if convergence_test_indicies is None:
            convergence_test_indicies = list(range(len(self.s_all)))

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
        if points.shape[1] == 4:
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


def g(t, T):
    """
    Smooth bump weight on (0, T) with g=0 at t<=0 or t>=T.
    """
    t = np.asarray(t, dtype=float)
    T = float(T)

    s = t / T

    # denom = s*(1-s); positive only for s in (0,1)
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
        dt_save=1e-6,
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
            self.WBA_transit_steps = [0]

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
        self.dt_save = dt_save
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
            Peta0_arr = np.asarray(Peta0)
            if Peta0_arr.size != 1:
                raise ValueError(
                    f"Peta0 must be scalar-like (size 1), got shape "
                    f"{Peta0_arr.shape} and size {Peta0_arr.size}"
                )
            self.Eprime = float(self.nprime * Ekin - self.omega * Peta0_arr.item())
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
            Ekin=self.Ekin,
            dt_save=self.dt_save,
            phases=phases,
            n_zetas=n_zetas,
            m_thetas=m_thetas,
            omegas=omegas,
            vpars=[0],
            axis=0,
            stopping_criteria=[
                MinToroidalFluxStoppingCriterion(0.01),
                MaxToroidalFluxStoppingCriterion(0.999),
            ],
            forget_exact_path=not self.DA_poinc,
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
        resonance_lines=None,
        line_plotting_kwargs=None,
        ylims=(0, 1),
        DA_colorbar=True,
        plot_legend=True,
        bg_field=None,
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
                in DA convergence plot.
            DA_max : Maximum value of Digit Accuracy to show on colorbar
            resonance_lines : List of resonance lines to plot on Poincare map.
                Each element should be a s value.
            line_plotting_kwargs : List of dictionaries of keyword arguments for
                plotting resonance lines. Should be the same length as 
                `resonance_lines`.
            ylims : Tuple specifying y-axis limits for the Poincare plot.
            DA_colorbar : Boolean indicating for DA colorbar inclusion.
            plot_legend : Boolean indicating whether to include a legend for the
                resonance lines.
            bg_field : Magnetic field to use for plotting resonance lines. Should
                be the background field of the pertubation with perfect QS enforced.
                If None, the unperturbed field, B0, is used.
            s_axis_label : Boolean for s-axis label. True by default.
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

        if bg_field is None:
            bg_field = self.B0

        if line_plotting_kwargs is None and resonance_lines is not None:
            line_plotting_kwargs = [
                {} for _ in resonance_lines if resonance_lines is not None
            ]

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
            if DA_colorbar:
                fig.colorbar(
                    ScalarMappable(norm=norm, cmap=mpl.colormaps[cmap]),
                    ax=ax,
                    orientation="vertical",
                    label="Digit Accuracy",
                )

        lines_2 = []
        if resonance_lines is not None:
            cmap = plt.get_cmap("Wistia")
            n_lines = len(resonance_lines)
            for i, arr in enumerate(resonance_lines):
                if "color" not in line_plotting_kwargs[i]:
                    line_plotting_kwargs[i]["color"] = cmap(i / max(n_lines - 1, 1))
                lines_2.append((arr, line_plotting_kwargs[i]["color"]))
                vp = self.vpar_func_perturbed(arr, self.chi(np.pi / 2, 0))
                bg_field.set_points(np.array([[arr, np.pi / 2, 0]]).T)
                unperturbed_path_map = PassingPoincare(
                    field=bg_field,
                    lam=(self.v0**2 - vp**2) / (self.v0**2 * bg_field.modB()[0, 0]),
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
                pa_data[:, 0] = np.mod(pa_data[:, 0], (2 * np.pi))
                pa_data = pa_data[pa_data[:, 0].argsort()]
                ax.plot(
                    pa_data[:, 0],
                    pa_data[:, 1],
                    **line_plotting_kwargs[i],
                )
            if plot_legend:
                ax.legend()
        fig.tight_layout()
        fig.savefig(filename[:-4] + ".pdf")

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
            plt.savefig(filename[:-4] + "_convergence.pdf")
            plt.clf()
            final_DAs = [x for x in final_DAs if not np.isnan(x)]
            plt.hist(final_DAs)
            plt.xlabel("Digit Accuracy")
            plt.title("Distribution of Digit Accuracy")
            plt.tight_layout()
            plt.savefig(filename[:-4] + "_DA_histogram.pdf")
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


class MapEquilibrium:
    def __init__(
        self,
        B0,
        mass,
        charge,
        Ekin,
        helicity_N,
        helicity_M,
        plot_s=True,
        helicity_Mp=None,
        helicity_Np=None,
        mu_lims=None,
        sign=1,
        ns_points=25,
        particles_per_surface=25,
        nlambda_points=25,
        savedata=True,
        savepath="",
        randomize_particles=False,
        number_of_particles=10000,
        min_timestep=1e-7,
        nconvergence_points=100,
        s_lims=None,
        mean=True,
        comm=None,
        tmax=1e-2,
        tol=1e-10,
        solver_options=None,
    ):
        """
        Initialize phase-space sampling, particle tracing, and diagnostic evaluation
        for guiding-center orbits in an equilibrium Boozer magnetic field.

        This class generates or accepts a collection of initial particle conditions
        and traces the corresponding guiding-center orbits in the equilibrium field.
        Per-particle diagnostics such as digit accuracy, wall-loss
        status, and phase-space coordinates used for plotting are computed and stored.

        The sampled phase space is parameterized by Boozer coordinates together with
        either:
            - a fixed total kinetic energy Ekin, or
            - a fixed shifted-energy slice Eprime.

        If Eprime is provided, the initialization solves for vpar from the invariant
        constraint rather than sampling vpar directly from fixed Ekin. This uses a
        reference kinetic energy Ekin for computing mu from vperp.

        Initial conditions may be supplied directly or generated internally in one of
        two ways:
            - randomly throughout the plasma volume, or
            - on a structured grid in surface label and pitch coordinate, with multiple
              particles sampled on each surface.
        """
        if solver_options is None:
            solver_options = {}
        self.solver_options=solver_options
        if s_lims is None:
            s_lims = [0.05, 0.95]

        self.B0 = B0
        self.mass = mass
        self.charge = charge
        self.Ekin = Ekin
        self.helicity_N = helicity_N
        self.helicity_M = helicity_M
        self.min_volmodB = min_volumemodB(self.B0)

        if mu_lims is None:
            mu_max = (Ekin) / self.min_volmodB
            mu_min = 0
        else:
            mu_max = mu_lims[1]
            mu_min = mu_lims[0]
        self.mu_min = mu_min
        self.mu_max = mu_max

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
        self.tol = tol
        self.plot_s = plot_s

        # set timing parameters
        self.tmax = tmax
        self.min_timestep = min_timestep
        self.vtotal = np.sqrt(2 * self.Ekin / mass)

        # set communicator parameters
        self.comm = comm
        self.verbose = False
        if self.comm is None or self.comm.rank == 0:
            self.verbose = True

        self.sign = sign

        self.s_min = s_lims[0]
        self.s_max = s_lims[1]

        # plotting settings
        self.savedata = savedata
        if savepath != "":
            savepath += "_"
        self.savepath = savepath
        self.convergence_points = nconvergence_points

        self.randomize = randomize_particles

        if exists(self.savepath + "initial_conditions.txt"):
            initial_conditions = np.loadtxt(self.savepath + "initial_conditions.txt")
            s, thetas, zetas, vpar, mu = (
                initial_conditions[:, 0],
                initial_conditions[:, 1],
                initial_conditions[:, 2],
                initial_conditions[:, 3],
                initial_conditions[:, 4],
            )
        else:
            if not randomize_particles:
                self.ns_points = ns_points
                self.nlambda_points = nlambda_points
                self.nParticles = ns_points * particles_per_surface * nlambda_points
            else:
                self.nParticles = number_of_particles
                xy_pts = int(np.sqrt(number_of_particles / particles_per_surface))
                self.ns_points = xy_pts
                self.nlambda_points = xy_pts

            self.particles_per_surface = particles_per_surface
            s, thetas, zetas, vpar, mu = self.initialize_particles()

        self.s, self.thetas, self.zetas, self.vpar, self.mu = s, thetas, zetas, vpar, mu

        self.res_filepaths = {
            "tys": self.savepath + "DA_data.txt",
        }

        self.expected_length = int(self.tmax / self.min_timestep)
        self.expected_step = int(self.expected_length / self.convergence_points)
        self.WBA_transit_indicies = np.linspace(
            self.expected_step,
            self.expected_length - 1,
            num=self.convergence_points,
            dtype=int,
        ).tolist()
        self.convergence_plot = self.convergence_points > 1

        self.trace_particles()
        return

    def initialize_particles(self):
        r"""
        Generate initial particle positions, parallel velocities, and magnetic
        moments for tracing.

        Initial flux surfaces and magnetic moments are sampled either on a
        structured (s, mu) grid or uniformly at random, depending on
        self.randomize. On each surface, particles are scattered uniformly using
        initialize_position_uniform_surf. Points with vpar^2 < 0 from energy
        conservation are dropped.

        Returns:
            s : List of initial s coordinates.
            thetas : List of initial theta coordinates.
            zetas : List of initial zeta coordinates.
            vpar : List of initial parallel velocities (signed by self.sign).
            mu : List of initial magnetic moments.
        """
        if self.randomize:
            mus = np.random.uniform(self.mu_min, self.mu_max, self.nlambda_points)

            s_linspace = np.random.uniform(self.s_min, self.s_max, self.ns_points)
        else:
            mus = np.linspace(self.mu_min, self.mu_max, self.nlambda_points)

            s_linspace = np.linspace(self.s_min, self.s_max, self.ns_points)

        surfaces, mus = np.meshgrid(s_linspace, mus)

        surfaces_flat = surfaces.flatten()
        mus_flat = mus.flatten()

        s = []
        thetas = []
        zetas = []
        vpars = []
        mus_tot = []
        for particle_index in range(len(surfaces_flat)):
            points_temp = initialize_position_uniform_surf(
                self.B0,
                self.particles_per_surface,
                surfaces_flat[particle_index],
                comm=self.comm,
            )

            self.B0.set_points(points_temp)
            modB = self.B0.modB()[:, 0]

            mu = np.ones_like(modB) * mus_flat[particle_index]

            vpar_energy = self.Ekin - (mu * modB)

            # remove unphysical points
            neg_idx = np.where(vpar_energy < 0)[0]
            vpar_energy = np.delete(vpar_energy, neg_idx)
            points_temp = np.delete(points_temp, neg_idx, axis=0)
            mu = np.delete(mu, neg_idx, axis=0)

            vpar = self.sign * np.sqrt((2 * vpar_energy) / self.mass)
            vpar_list = vpar.tolist()

            vpars += vpar_list
            s += list(points_temp[:, 0])
            thetas += list(points_temp[:, 1])
            zetas += list(points_temp[:, 2])
            mus_tot += mu.tolist()

        return s, thetas, zetas, vpars, mus_tot

    def check_filepaths(self, filepaths):
        r"""
        Check whether all provided output file paths exist.

        Args:
            filepaths : Dictionary of file labels to filesystem paths.
        Returns:
            exists_all : True if every path exists, otherwise False.
        """
        return all(exists(fp) for fp in filepaths.values())

    def trace_particles(self):
        """
        Trace particles in the equilibrium field and compute diagnostics.
        Initialises build_lists for data processing.
        """
        import pickle

        if self.check_filepaths(self.res_filepaths):
            if self.verbose:
                print("Reading File", flush=True)
            with open(self.res_filepaths["tys"], "rb") as f:
                res_tys = pickle.load(f)
            if self.verbose:
                print("Read Files", flush=True)
            self.build_lists(res_tys)
            return

        first, last = parallel_loop_bounds(self.comm, len(self.s))
        gc_tys = []

        for itrj in range(first, last):
            point = np.zeros((1, 3))  # initialize with t = 0
            point[:, 0] = self.s[itrj]
            point[:, 1] = self.thetas[itrj]
            point[:, 2] = self.zetas[itrj]

            vpar = [self.vpar[itrj]]
            mu = self.mu[itrj]
            res_tys, res_zeta_hits = trace_particles_boozer(
                self.B0,
                point,
                vpar,
                tmax=self.tmax,
                mass=self.mass,
                charge=self.charge,
                Ekin=self.Ekin,
                stopping_criteria=[MaxToroidalFluxStoppingCriterion(0.999)],
                forget_exact_path=False,
                dt_save=self.min_timestep,
                abstol=self.tol,
                reltol=self.tol,
                **self.solver_options,
            )
            points_trajectory = res_tys[0]

            if isinstance(points_trajectory, list):
                print(f"points trajectory is list: {points_trajectory=}")
                for i, p in enumerate(points_trajectory):
                    print(i, len(p))
                continue

            if points_trajectory.ndim != 2:
                continue

            time_momentum, s_path, theta_path, zeta_path, vpar_path = (
                points_trajectory[:, 0],
                points_trajectory[:, 1],
                points_trajectory[:, 2],
                points_trajectory[:, 3],
                points_trajectory[:, 4],
            )
            points_trajectory = np.column_stack((s_path, theta_path, zeta_path))

            # avoid tracing bug that sometimes causes last pt to be outside wall
            idx_wall = np.argmax(s_path >= 1) if np.any(s_path >= 1) else None
            if idx_wall is not None and s_path[idx_wall] >= 1:
                idx_wall -= 1
                points_trajectory = points_trajectory[:idx_wall, :]
                vpar_path = vpar_path[:idx_wall]
                time_momentum = time_momentum[:idx_wall]

            Peta_values = compute_peta(
                self.B0,
                points_trajectory,
                vpar_path,
                self.mass,
                self.charge,
                self.helicity_M,
                self.helicity_N,
                self.helicity_Mp,
                self.helicity_Np,
            )

            v_par_signs = np.sign(vpar_path)

            mask = v_par_signs != 0
            v = v_par_signs[mask]
            orig_idx = np.where(mask)[0]

            bounce_local = np.where(v[1:] * v[:-1] < 0)[0]
            bounce_indices = orig_idx[bounce_local + 1]

            bounces = len(bounce_indices) if len(v) > 1 else 0

            zeta_path = np.mod(points_trajectory[:, 2], 2 * np.pi)
            dzeta = np.diff(zeta_path)
            dzeta_idx = np.where(dzeta < -np.pi)[0]

            true_passes = []

            for passing_index in range(len(dzeta_idx) - 1):
                pass1 = dzeta_idx[passing_index]
                pass2 = dzeta_idx[passing_index + 1]
                if not np.any((bounce_indices > pass1) & (bounce_indices < pass2)):
                    true_passes.append(dzeta_idx[passing_index])

            true_passes = np.array(true_passes)
            passes = len(true_passes) if len(dzeta) > 0 else 0

            # start_state = [s, theta, zeta, vpar, p_eta_0, mu]
            start_state = [
                points_trajectory[-1, 0],
                points_trajectory[-1, 1],
                points_trajectory[-1, 2],
                vpar[0],
                Peta_values[0],
                mu,
            ]

            if len(Peta_values) > 10:
                stack_data = np.column_stack((time_momentum, Peta_values))
                time_eval, DA_eval = return_DA(stack_data)
                final_DA = DA_eval
            else:
                final_DA = np.nan

            # end_state = [time, s, theta, zeta, vpar, p_eta_f, bounces, passes, DA]
            end_state = [
                time_momentum[-1],
                points_trajectory[-1, 0],
                points_trajectory[-1, 1],
                points_trajectory[-1, 2],
                vpar_path[-1],
                Peta_values[-1],
                bounces,
                passes,
                final_DA,
            ]

            convergence_times = []
            convergence_petas = []
            convergence_bounces = []
            convergence_passes = []
            convergence_DAs = []

            for _conv_index, timing_index in enumerate(self.WBA_transit_indicies):
                if timing_index > len(time_momentum):
                    break
                convergence_times.append(time_momentum[timing_index])
                convergence_petas.append(Peta_values[timing_index])

                bounce_enum = np.searchsorted(bounce_indices, timing_index, side="left")
                pass_enum = np.searchsorted(true_passes, timing_index, side="left")

                convergence_bounces.append(bounce_enum)
                convergence_passes.append(pass_enum)

                stack_data = np.column_stack(
                    (time_momentum[:timing_index], Peta_values[:timing_index])
                )
                time_eval, DA_eval = return_DA(stack_data)
                convergence_DAs.append(DA_eval)

            convergence_data = [
                convergence_times,
                convergence_petas,
                convergence_bounces,
                convergence_passes,
                convergence_DAs,
            ]

            gc_tys.append([start_state, end_state, convergence_data])
        print(f"{self.comm.rank=} done tracing particles", flush=True)
        if self.comm is not None:
            gc_tys = [i for o in self.comm.allgather(gc_tys) for i in o]

        if self.verbose:
            import pickle

            with open(self.res_filepaths["tys"], "wb") as f:
                pickle.dump(gc_tys, f)

        self.build_lists(gc_tys)
        return

    def build_lists(self, res_tys):
        r"""
        Unpack per-particle trajectory summaries into flat instance attributes
        suitable for plotting and aggregation.

        Populates self.DAs_at_loss, self.DA_at_tfinal, self.bounces, self.passes,
        self.pitch, self.lost_total, self.final_times, self.trapped,
        self.Peta_start, self.s0, self.mu0, and the convergence_* arrays from
        the list produced by trace_particles. If self.verbose, initial conditions
        are written to disk.

        Args:
            res_tys : List of per-particle summaries, each of the form
                    [start_state, end_state, convergence_data].
        """
        DAs_at_loss = []

        DA_tfinal = []
        bounces = []
        passes = []

        lost_total = []
        final_times = []
        trapped = []
        Peta_start = []
        pitch = []

        s0 = []
        theta0 = []
        zeta0 = []
        vpar0 = []
        mu0 = []

        convergence_bounces = []
        convergence_passes = []
        convergence_times = []
        convergence_petas = []
        convergence_DAs = []

        for i in range(len(res_tys)):
            # start_state = [s, theta, zeta, vpar, p_eta_0, mu]
            # end_state = [time, s, theta, zeta, vpar, p_eta_f, bounces, passes, DA]

            start_state = res_tys[i][0]
            end_state = res_tys[i][1]
            convergence_data = res_tys[i][2]

            pitch.append(start_state[5] / self.Ekin)
            if self.plot_s:
                Peta_start.append(start_state[0])
            else:
                Peta_start.append(start_state[4])
            final_time = end_state[0]
            final_times.append(final_time)
            DAs_at_loss.append(end_state[8])
            trapped.append(end_state[6])

            s0.append(start_state[0])
            theta0.append(start_state[1])
            zeta0.append(start_state[2])
            vpar0.append(start_state[3])
            mu0.append(start_state[5])

            bounces.append(end_state[6])
            passes.append(end_state[7])

            # params that depend on loss
            if final_time < (self.tmax - (5 * self.min_timestep)):
                lost_total.append(1)
                DA_tfinal.append(np.nan)
            else:
                lost_total.append(0)
                DA_tfinal.append(end_state[8])
            convergence_times.append(convergence_data[0])
            convergence_petas.append(convergence_data[1])
            convergence_bounces.append(convergence_data[2])
            convergence_passes.append(convergence_data[3])
            convergence_DAs.append(convergence_data[4])

        self.DAs_at_loss = DAs_at_loss
        self.DA_at_tfinal = DA_tfinal
        self.bounces = bounces
        self.passes = passes
        self.pitch = pitch

        self.lost_total = lost_total
        self.final_times = final_times
        self.trapped = trapped
        self.Peta_start = Peta_start

        self.s0 = s0
        self.mu0 = mu0

        self.convergence_bounces = convergence_bounces
        self.convergence_passes = convergence_passes
        self.convergence_times = convergence_times
        self.convergence_petas = convergence_petas
        self.convergence_DAs = convergence_DAs

        if self.verbose:
            np.savetxt(
                self.savepath + "initial_conditions.txt",
                np.column_stack((s0, theta0, zeta0, vpar0, mu0)),
            )

        if self.verbose:
            print("Done Building Lists", flush=True)
        return

    def vpar_func(self, s, theta, zeta, mu, sgn):
        r"""
        Solve for the parallel velocity at (s, theta, zeta) from the
        shifted-energy invariant Eprime = n' * E - omega * p_eta.

        The constraint is quadratic in vpar; the root matching sgn is selected.
        Returns NaN where the discriminant is negative.

        Args:
            s : Flux-surface label (scalar or array-like).
            theta : Boozer poloidal angle.
            zeta : Boozer toroidal angle.
            mu : Magnetic moment per mass.
            sgn : Desired sign of the parallel velocity.

        Returns:
            vpar : Parallel velocity consistent with self.Eprime, or NaN.
        """
        point = np.zeros((len(s), 3)) if hasattr(s, "__len__") else np.zeros((1, 3))
        point[:, 0] = s
        point[:, 1] = theta
        point[:, 2] = zeta
        self.B0.set_points(point)
        modB = self.B0.modB()[:, 0]

        energy = self.Ekin - mu * modB
        vpar = sgn * np.sqrt(np.maximum(energy, 0))
        # condtion, x, y
        # returns x (vpar) if energy > 0, else nan
        return np.where(energy > 0, vpar, np.nan)

    def plot_surfaces(
        self,
        nx=30,
        ny=30,
        savepath="heatmap_digit_accuracy.pdf",
        plot_at_loss=True,
        ax=None,
        DA_max=7,
        statistic="mean",
        plot_losses=False,
    ):
        r"""
        Create and save a 2D heatmap of digit accuracy in the
        (pitch angle, flux-surface label) plane, with the trapped-passing
        boundary overlaid as a fitted curve.

        Args:
            nx          : Number of bins along the pitch-angle axis (default: 30).
            ny          : Number of bins along the radial axis (default: 30).
            savepath    : File path for the output heatmap image
                        (default: 'heatmap_digit_accuracy.pdf').
            plot_at_loss : If True, use the digit accuracy value at the time of
                        loss; otherwise use the value at the end of the full
                        integration (default: True).
            ax          : Matplotlib axis to plot on. If None, a new figure and
                        axis are created.
            DA_max      : Maximum digit accuracy value shown on the colorbar
                        (default: 7).
            minimum_DA  : If True, show the minimum DA within each bin instead of
                        the mean (default: False).
            plot_losses : Currently unused (default: False).

        Returns:
            None
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from scipy.stats import binned_statistic_2d

        if plot_at_loss:
            fDA = np.array(self.DAs_at_loss)
        else:
            fDA = np.array(self.DA_at_tfinal)

        mpl.use("Agg")  # Don't use interactive backend

        try:
            import cmcrameri.cm as cmc  # noqa: F401

            cmap = "cmc.managua"
        except ImportError:
            cmap = "viridis"

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if not self.randomize:
            nx = int(self.ns_points - 1)
            ny = int(self.nlambda_points - 1)

        def trapped_passing_function(s, pitch):
            resolution = 500
            points_temp = initialize_position_uniform_surf(
                self.B0,
                resolution,
                s,
                comm=None,
            )
            self.B0.set_points(points_temp)
            modB = self.B0.modB()[:, 0]

            if np.any(1 - (pitch * modB) < 0):
                return 1
            return 0

        def s_peta_map(s, mu, sign):
            r"""
            Map a point to canonical momentum p_eta using current settings.

            Args:
                s : Radial-like coordinate.
                mu : Magnetic moment.
                sign : Desired sign for parallel velocity.
            Returns:
                peta : Canonical momentum value at the requested point.
            """
            points = np.zeros((3, 1)) if np.isscalar(s) else np.zeros((3, len(s)))
            points[0, :] = s

            vp_temp = self.vpar_func(points[0, 0], points[0, 1], points[0, 2], mu, sign)

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

        normalized_pitch = np.array(self.pitch) * self.min_volmodB * self.sign
        peta_start = np.array(self.Peta_start)

        stat, x_edges, y_edges, binnumber = binned_statistic_2d(
            normalized_pitch,
            peta_start,
            fDA,
            statistic=statistic,
            bins=[nx, ny],
        )
        norm = mpl.colors.Normalize(vmin=0, vmax=DA_max)
        X2, Y2 = np.meshgrid(x_edges, y_edges)
        im2 = ax.pcolormesh(X2, Y2, stat.T, shading="auto", cmap=cmap, norm=norm)
        fig.colorbar(im2, ax=ax, label="Digit Accuracy")

        ax.set_xlabel(r"$\lambda = \frac{\mu}{E} \text{sign}(v_{||})$")

        s_scope = np.linspace(self.s_min, self.s_max, 75)[::-1]
        pa_min, pa_max = (self.mu_min / self.Ekin), (self.mu_max / self.Ekin)
        pa_scope = np.linspace(pa_min, pa_max, 75)

        s_tp = []
        pa_tp = []

        for s in s_scope:
            for pa in pa_scope:
                if trapped_passing_function(s, pa) == 1:
                    s_tp.append(s)
                    pa_tp.append(pa * self.sign * self.min_volmodB)
                    break

        s_tp = np.array(s_tp)
        pa_tp = np.array(pa_tp)

        if self.plot_s:
            ax.set_ylabel(r"$s$")
        else:
            ax.set_ylabel(r"$P_\eta$")
            peta_tp = []
            for s, mu in zip(s_tp, (pa_tp * self.Ekin) / self.min_volmodB):
                peta_tp.append(s_peta_map(s, mu, self.sign))
            s_tp = np.array(peta_tp)

        coeffs = np.polyfit(pa_tp, s_tp, 2)
        poly = np.poly1d(coeffs)
        pa_fit = np.linspace(min(pa_tp), max(pa_tp), 100)
        s_fit = poly(pa_fit)

        ax.plot(
            pa_fit, s_fit, color="grey", linewidth=5, label="Trapped-passing boundary"
        )

        plt.tight_layout()
        plt.savefig(savepath)
        plt.clf()
        for i in range(len(self.convergence_times)):
            plt.plot(self.convergence_times[i], self.convergence_DAs[i], alpha=0.5)
        plt.savefig(savepath[:-4] + "_convergence.pdf")

def return_bounces_and_passes(vpar_path, zeta_path):
    r"""
    Count guiding-center bounces and toroidal transits along a trajectory.

    Bounces are detected as sign changes of vpar. Transits are detected as
    negative jumps in zeta mod 2*pi; a candidate transit is rejected if a
    bounce occurs between it and the next wrap.

    Args:
        vpar_path : Array of parallel velocity samples.
        zeta_path : Array of zeta samples (radians).

    Returns:
        bounce_indices : Trajectory indices where vpar changes sign.
        true_passes : Trajectory indices of confirmed toroidal transits.
    """
    v_par_signs = np.sign(vpar_path)

    mask = v_par_signs != 0
    v = v_par_signs[mask]
    # Keep track of original indices after removing zeros
    orig_idx = np.where(mask)[0]

    # find vpar sign changes
    bounce_local = np.where(v[1:] * v[:-1] < 0)[0]
    # map back to original trajectory indexing, pre zero removal
    bounce_indices = orig_idx[bounce_local + 1]

    zeta_path = np.mod(zeta_path, 2 * np.pi)
    dzeta = np.diff(zeta_path)

    # find large negative jump, this is where mod
    # brings factors of 2pi back to zero, and pass
    wrap_idx = np.where(dzeta < -np.pi)[0]

    # isolate transits across zeta of 2pi
    true_passes = []
    for passing_index in range(len(wrap_idx) - 1):
        pass1 = wrap_idx[passing_index]
        pass2 = wrap_idx[passing_index + 1]

        # ensure no bounce between these two toroidal passes
        if not np.any((bounce_indices > pass1) & (bounce_indices < pass2)):
            true_passes.append(wrap_idx[passing_index])

    return bounce_indices, true_passes

class MapPhaseSpace:
    r"""
    Phase-space mapping and digit-accuracy diagnostics for guiding-center
    orbits in a perturbed (ShearAlfvenWave) magnetic field.

    This class traces particles in a ShearAlfvenHarmonic or
    ShearAlfvenWavesSuperposition field and computes per-particle diagnostics
    including the Weighted Birkhoff Average (WBA) digit accuracy, wall-loss
    status, bounce and transit counts, and the perturbed energy invariant
    E' = n' * E - omega * p_eta.

    Initial conditions are generated either on a structured grid in
    (s, \mu) space or drawn uniformly from the
    plasma volume. Particles lost in the equilibrium field before tracing in
    the perturbed field are removed.
    """

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
        Eprime=None,
        sign_vpar=1,
        tmax=1e-2,
        plot_s=False,
        min_timestep=1e-6,
        ns_points=35,
        particles_per_surface=15,
        nlambda_points=35,
        randomize_particles=False,
        number_of_particles=10000,
        s_lims=None,
        mu_lims=None,
        comm=None,
        tol=1e-9,
        solver_options=None,
        savedata=True,
        file_name="",
        convergence_points=1,
    ):
        r"""
        Initialize the MapPhaseSpace instance, generate initial conditions, and
        trace particles in the perturbed field.

        Args:
            saw                  : A :class:`ShearAlfvenHarmonic` or
                                :class:`ShearAlfvenWavesSuperposition` instance.
            Phin_max             : Toroidal mode number of the dominant wave harmonic,
                                used to compute nprime.
            Phim_max             : Poloidal mode number of the dominant wave harmonic,
                                used to compute nprime.
            omega                : Wave frequency.
            mass                 : Particle mass.
            charge               : Particle charge.
            Ekin                 : Total unperturbed kinetic energy.
            helicity_N           : Toroidal helicity of the field-strength contours.
            helicity_M           : Poloidal helicity of the field-strength contours.
            helicity_Mp          : Poloidal helicity of the mapping coordinate eta.
                                If None, determined automatically from helicity_M.
            helicity_Np          : Toroidal helicity of the mapping coordinate eta.
                                If None, determined automatically from helicity_N.
            Eprime               : Fixed value of the shifted energy invariant
                                n' * E - omega * p_eta. If provided, initial
                                parallel velocities are solved from this constraint
                                rather than from Ekin directly (default: None).
            sign_vpar            : Sign of the parallel velocity, either +1 or -1
                                (default: 1).
            tmax                 : Maximum integration time per particle
                                (default: 1e-2 s).
            plot_s               : If True, use the flux-surface label s as the radial
                                plot coordinate; otherwise use p_eta (default: False).
            min_timestep         : Minimum time-step size used as the save interval and
                                for computing the expected trajectory length
                                (default: 1e-6 s).
            ns_points            : Number of flux surfaces in the structured grid
                                (default: 25).
            particles_per_surface : Number of particles sampled on each flux surface
                                (default: 15).
            nlambda_points       : Number of magnetic-moment values in the structured
                                grid (default: 25).
            randomize_particles  : If True, sample initial conditions uniformly at
                                random rather than on a structured grid
                                (default: False).
            number_of_particles  : Number of particles to sample when
                                randomize_particles is True (default: 10000).
            s_lims               : Two-element list [s_min, s_max] bounding the
                                sampled flux surfaces (default: [0.01, 0.975]).
            mu_lims              : Two-element list [mu_min, mu_max] bounding the
                                sampled magnetic moments. If [None, None], limits
                                are set from Ekin and the minimum |B| in the volume
                                (default: [None, None]).
            comm                 : MPI communicator for parallel execution
                                (default: None).
            tol                  : Absolute and relative ODE solver tolerance
                                (default: 1e-10).
            solver_options       : Dictionary of additional options passed to the ODE
                                solver (default: {}).
            savedata             : If True, save initial conditions and final
                                diagnostics to disk (default: True).
            file_name            : Prefix for output file names (default: '').
            convergence_points   : Number of intermediate WBA evaluations per
                                trajectory used to assess convergence
                                (default: 1).
        """
        if not isinstance(saw, ShearAlfvenHarmonic) and not isinstance(
            saw, ShearAlfvenWavesSuperposition
        ):
            raise TypeError(
                "Expected saw to be an instance of ShearAlfvenHarmonic "
                "or ShearAlfvenWavesSuperposition"
            )

        if not isinstance(saw, ShearAlfvenHarmonic):
            raise Warning(
                "Expected saw to be an instance of ShearAlfvenHarmonic - "
                "Perturbed Energy Invariant may not be valid."
            )

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
        self.sign = sign_vpar
        self.vtotal = np.sqrt(2 * self.Ekin / mass)
        self.mass = mass
        self.charge = charge

        if Eprime is None:
            self.Eprime_slice = False
        else:
            self.Eprime_slice = True
        self.Eprime = Eprime

        if s_lims is None:
            self.s_min = 0.01
            self.s_max = 0.975
        else:
            self.s_min = s_lims[0]
            self.s_max = s_lims[1]

        # set communicator parameters
        self.comm = comm
        self.verbose = False
        if self.comm is None or self.comm.rank == 0:
            self.verbose = True

        self.solver_options = solver_options if solver_options is not None else {}

        self.min_volmodB = min_volumemodB(self.B0)
        self.plot_s = plot_s

        # plotting settings
        self.savedata = savedata
        self.savepath = file_name + "_"
        self.convergence_points = convergence_points

        if mu_lims is None:
            self.mu_min = 0
            self.mu_max = Ekin / self.min_volmodB
        else:
            self.mu_min = mu_lims[0]
            self.mu_max = mu_lims[1]

        if self.savedata:
            self.final_filepaths = {
                "DA": self.savepath + "DA_losttimes.txt",
                "ICs": self.savepath + "initial_conditions.txt",
            }
            self.res_filepaths = {
                "tys": self.savepath + "DA_data.txt",
            }

        self.randomize = randomize_particles

        if exists(self.final_filepaths["ICs"]):
            initial_conditions = np.loadtxt(self.final_filepaths["ICs"])
            self.s, self.thetas, self.zetas, self.vpar, self.mus_per_mass = (
                initial_conditions[:, 0],
                initial_conditions[:, 1],
                initial_conditions[:, 2],
                initial_conditions[:, 3],
                initial_conditions[:, 4],
            )
        else:
            if randomize_particles:
                self.nParticles = number_of_particles
                xy_pts = int(np.sqrt(number_of_particles / particles_per_surface))
                self.ns_points = xy_pts
                self.nlambda_points = xy_pts
            else:
                self.ns_points = ns_points
                self.nlambda_points = nlambda_points
                self.nParticles = ns_points * particles_per_surface * nlambda_points

            self.particles_per_surface = particles_per_surface
            s, thetas, zetas, vpar, mu_per_mass = self.initialize_particles()

            initial_points = np.zeros((len(s), 3))  # initialize with t = 0
            initial_points[:, 0] = s
            initial_points[:, 1] = thetas
            initial_points[:, 2] = zetas
            self.s, self.thetas, self.zetas, self.vpar, self.mus_per_mass = (
                self.remove_equilibrium_lost_particles(
                    initial_points, vpar, mu_per_mass
                )
            )

        # set parameters for convergence plot
        expected_length = int(self.tmax / self.min_timestep)
        expected_step = int(expected_length / self.convergence_points)
        self.WBA_transit_indicies = np.linspace(
            expected_step, expected_length - 1, num=self.convergence_points, dtype=int
        ).tolist()
        self.convergence_plot = self.convergence_points > 1

        self.trace_particles()
        return

    def initialize_particles(self):
        if self.randomize:
            mus = np.random.uniform(self.mu_min, self.mu_max, self.nlambda_points)

            s_linspace = np.random.uniform(self.s_min, self.s_max, self.ns_points)
        else:
            mus = np.linspace(self.mu_min, self.mu_max, self.nlambda_points)

            s_linspace = np.linspace(self.s_min, self.s_max, self.ns_points)

        surfaces, mus = np.meshgrid(s_linspace, mus)

        surfaces_flat = surfaces.flatten()
        mus_flat = mus.flatten()

        s = []
        thetas = []
        zetas = []
        vpars = []
        mus_per_mass = []
        for particle_index in range(len(surfaces_flat)):
            points_temp = initialize_position_uniform_surf(
                self.B0,
                self.particles_per_surface,
                surfaces_flat[particle_index],
                comm=self.comm,
            )

            self.B0.set_points(points_temp)
            modB = self.B0.modB()[:, 0]

            mu = np.ones_like(modB) * mus_flat[particle_index]
            mu_per_mass = mu / self.mass

            if self.Eprime_slice:
                vpar_temp = self.vpar_func_perturbed(
                    points_temp[:, 0],
                    points_temp[:, 1],
                    points_temp[:, 2],
                    mu_per_mass,
                    self.sign,
                )
            else:
                vpar_temp = self.vpar_func(
                    points_temp[:, 0],
                    points_temp[:, 1],
                    points_temp[:, 2],
                    mu,
                    self.sign,
                )
            # remove unphysical points
            neg_idx = np.where(np.isnan(vpar_temp))[0]
            vpar_temp = np.delete(vpar_temp, neg_idx)
            points_temp = np.delete(points_temp, neg_idx, axis=0)
            mu = np.delete(mu, neg_idx, axis=0)
            mu_per_mass = np.delete(mu_per_mass, neg_idx, axis=0)

            s += list(points_temp[:, 0])
            thetas += list(points_temp[:, 1])
            zetas += list(points_temp[:, 2])
            vpars += vpar_temp.tolist()
            mus_per_mass += mu_per_mass.tolist()
        return (
            s,
            thetas,
            zetas,
            vpars,
            mus_per_mass,
        )

    def check_filepaths(self, filepaths):
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

    def vpar_func(self, s, theta, zeta, mu, sgn):
        point = np.zeros((len(s), 3))
        point[:, 0] = s
        point[:, 1] = theta
        point[:, 2] = zeta
        self.B0.set_points(point)
        modB = self.B0.modB()[:, 0]

        energy_par = self.Ekin - mu * modB
        energy_par_norm = np.maximum(energy_par, 0)
        vpar = sgn * np.sqrt(2 * energy_par_norm / self.mass)
        # condtion, x, y
        # returns x (vpar) if energy > 0, else nan
        return np.where(energy_par > 0, vpar, np.nan)

    def vpar_func_perturbed(self, s, theta, zeta, mu, sgn):
        point = np.zeros((len(s), 4)) if hasattr(s, "__len__") else np.zeros((1, 4))
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
        discriminant = b**2 - 4 * a * c
        valid = discriminant >= 0

        if a != 0:
            # mask negatives before sqrt so we don't get warnings
            safe_disc = np.where(valid, discriminant, 0.0)
            result = (-b + sgn * np.sqrt(safe_disc)) / (2 * a)
        else:
            # discriminant = b**2 >= 0 always
            result = (-c / b) * sgn

        return np.where(valid, result, np.nan)

    def remove_equilibrium_lost_particles(self, points, vpars_init, mus_per_mass):
        r"""
        Trace particles briefly in the unperturbed equilibrium field and drop any
        that hit the s = 1 wall before tracing in the perturbed field.

        Args:
            points : Array of shape (N, 3) of initial (s, theta, zeta).
            vpars_init : Array of initial parallel velocities.
            mus_per_mass : Array of initial magnetic moments per mass.

        Returns:
            s, thetas, zetas, vpars, mus_per_mass : Filtered arrays with
                equilibrium-lost particles removed.
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
            if isinstance(gc_zeta_hits[i], np.ndarray):  # noqa: SIM102
                if gc_zeta_hits[i].size > 0:  # noqa: SIM102
                    if int(gc_zeta_hits[i][0][1]) == -1:  # noqa: SIM102
                        lost_total.append(i)

        # remove wall lost particles from the list of evaluated particles
        points = np.delete(points, lost_total, axis=0)
        vpars_init = np.delete(vpars_init, lost_total, axis=0)
        mus_per_mass = np.delete(mus_per_mass, lost_total, axis=0)

        return points[:, 0], points[:, 1], points[:, 2], vpars_init, mus_per_mass


    def trace_particles(self):
        r"""
        Trace all initialized particles in the perturbed field and compute
        per-particle diagnostics.

        For each particle, integrates the guiding-center equations in the
        ShearAlfvenWave field, computes the canonical momentum p_eta, the total
        energy E, the shifted energy Eprime, bounce and transit counts, and the
        WBA digit accuracy. Results are collected across MPI ranks, saved to
        disk if self.savedata is True, and passed to build_lists.
        """
        import pickle

        if self.check_filepaths(self.res_filepaths):
            if self.verbose:
                print("Reading File", flush=True)
            with open(self.res_filepaths["tys"], "rb") as f:
                res_tys = pickle.load(f)
            if self.verbose:
                print("Read Files", flush=True)
            self.build_lists(res_tys)
            return

        if self.verbose:
            print("Tracing particles in perturbed field...", flush=True)

        first, last = parallel_loop_bounds(self.comm, len(self.s))

        res_tys = []

        for itrj in range(first, last):
            point = np.zeros((1, 3))  # initialize with t = 0
            point[:, 0] = self.s[itrj]
            point[:, 1] = self.thetas[itrj]
            point[:, 2] = self.zetas[itrj]

            vpar = [self.vpar[itrj]]
            mu_pm = [self.mus_per_mass[itrj]]

            gc_tys, gc_zeta_hits = trace_particles_boozer_perturbed(
                perturbed_field=self.saw,
                stz_inits=point,
                parallel_speeds=vpar,
                mus=mu_pm,
                tmax=self.tmax,
                mass=self.mass,
                charge=self.charge,
                Ekin=self.Ekin,
                abstol=1e-9,
                reltol=1e-9,
                stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
                dt_save=self.min_timestep,
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

            # avoid tracing bug that sometimes causes last pt to be outside wall
            idx_wall = np.argmax(s_path >= 1) if np.any(s_path >= 1) else None
            if idx_wall is not None and s_path[idx_wall] >= 1:
                idx_wall -= 1
                points_trajectory = points_trajectory[:idx_wall, :]
                vpar_path = vpar_path[:idx_wall]
                time_momentum = time_momentum[:idx_wall]

            self.saw.set_points(points_trajectory)

            modB = self.saw.B0.modB()[:, 0]
            Phi = self.saw.Phi()[:, 0]

            weighted_mu = mu_pm[0] * self.mass

            E = 0.5 * self.mass * vpar_path**2 + weighted_mu * modB + self.charge * Phi

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

            if points_trajectory.shape[0] > 8:
                stack_data = np.column_stack((points_trajectory[:, -1], Peta_values))
                time_eval, DA_eval = return_DA(stack_data)
                final_DA = DA_eval
            else:
                final_DA = np.nan

            bounce_indices, passing_indicies = return_bounces_and_passes(
                vpar_path, zeta_path
            )

            # start state vector:  [s, theta, zeta, vpar, peta, E, mu, Eprime]
            # end state vector:
            # [t, s, theta, zeta, vpar, peta, E, mu, Eprime, bounces, passes, DA]
            # mean state vector:  [s_mean, peta_mean, E_mean, Eprime_mean]
            start_state = [
                points_trajectory[0, 0],
                points_trajectory[0, 1],
                points_trajectory[0, 2],
                vpar[0],
                Peta_values[0],
                E[0],
                weighted_mu,
                Eprime[0],
            ]

            end_state = [
                points_trajectory[-1, -1],
                points_trajectory[-1, 0],
                points_trajectory[-1, 1],
                points_trajectory[-1, 2],
                vpar[-1],
                Peta_values[-1],
                E[-1],
                weighted_mu,
                Eprime[-1],
                len(bounce_indices),
                len(passing_indicies),
                final_DA,
            ]

            mean_state = [
                np.mean(points_trajectory[:, 0]),
                np.mean(Peta_values),
                np.mean(E),
                np.mean(Eprime),
            ]

            convergence_times = []
            convergence_petas = []
            convergence_energies = []
            convergence_bounces = []
            convergence_passes = []
            convergence_DAs = []

            for _conv_index, timing_index in enumerate(self.WBA_transit_indicies):
                if timing_index > len(time_momentum):
                    break
                convergence_times.append(time_momentum[timing_index])
                convergence_petas.append(Peta_values[timing_index])
                convergence_energies.append(E[timing_index])
                bounce_enum = np.searchsorted(bounce_indices, timing_index, side="left")

                pass_enum = np.searchsorted(passing_indicies, timing_index, side="left")

                convergence_bounces.append(bounce_enum)
                convergence_passes.append(pass_enum)

                stack_data = np.column_stack(
                    (time_momentum[:timing_index], Peta_values[:timing_index])
                )
                time_eval, DA_eval = return_DA(stack_data)
                convergence_DAs.append(DA_eval)

            convergence_data = [
                convergence_times,
                convergence_petas,
                convergence_bounces,
                convergence_passes,
                convergence_DAs,
                convergence_energies,
            ]

            particle_out = [start_state, end_state, mean_state, convergence_data]
            res_tys.append(particle_out)

        if self.comm is not None:
            print(f"{self.comm.rank=} done tracing particles", flush=True)
            res_tys = [i for o in self.comm.allgather(res_tys) for i in o]

        if self.verbose:
            with open(self.res_filepaths["tys"], "wb") as f:
                pickle.dump(res_tys, f)

        self.build_lists(res_tys)
        return

    def build_lists(self, res_tys):
        r"""
        Process trajectory summaries into organized diagnostic
        lists stored as instance attributes.

        Populates self.DAs_at_loss, self.DA_at_tfinal, self.lost_total,
        self.final_times, self.bounces, self.passes, self.pitch,
        self.Plot_Radial, self.Peta_init/mean/final, self.E_init/mean/final,
        and the convergence_* arrays.

        Args:
            res_tys : List of per-particle summaries, each of the form
                    [start_state, end_state, mean_state, convergence_data].
        """
        if self.verbose:
            print("Building Lists", flush=True)

        DAs_at_loss = []
        DA_tfinal = []

        lost_total = []
        final_times = []
        bounces = []
        passes = []
        pitch = []

        Plot_Radial = []

        Peta_init = []
        Peta_mean = []
        Peta_final = []
        E_mean = []
        E_final = []
        E_init = []

        s0 = []
        theta0 = []
        zeta0 = []
        vpar0 = []
        mus0 = []

        convergence_bounces = []
        convergence_passes = []
        convergence_times = []
        convergence_petas = []
        convergence_energies = []
        convergence_DAs = []

        for elem in res_tys:
            # start state vector:
            #   [s, theta, zeta, vpar, peta, E, mu, Eprime]
            # end state vector:
            #   [t, s, theta, zeta, vpar, peta(5), E, mu, Eprime, bounces, passes, DA]
            # mean state vector:
            #   [s_mean, peta_mean, E_mean, Eprime_mean]
            # convergence state vector:
            #   [times, petas, bounces, passes, DAs, energies]

            start_state = elem[0]
            end_state = elem[1]
            means = elem[2]
            convergence = elem[3]

            final_time = end_state[0]

            final_times.append(final_time)

            pitch_val = float(start_state[6]) / self.Ekin
            pitch_val *= self.sign
            pitch.append(pitch_val)

            if self.plot_s:
                Plot_Radial.append(start_state[0])
            else:
                Plot_Radial.append(start_state[4])

            Peta_init.append(start_state[4])
            Peta_mean.append(means[1])
            Peta_final.append(end_state[5])
            E_mean.append(means[2])
            E_final.append(end_state[6])
            E_init.append(start_state[5])

            if final_time < (self.tmax - (5 * self.min_timestep)):
                lost_total.append(1)
                DA_tfinal.append(np.nan)
            else:
                lost_total.append(0)
                DA_tfinal.append(end_state[11])

            DAs_at_loss.append(end_state[11])
            bounces.append(end_state[9])
            passes.append(end_state[10])

            s0.append(start_state[0])
            theta0.append(start_state[1])
            zeta0.append(start_state[2])
            vpar0.append(start_state[3])
            mus0.append(start_state[6])

            convergence_bounces.append(convergence[2])
            convergence_passes.append(convergence[3])
            convergence_times.append(convergence[0])
            convergence_petas.append(convergence[1])
            convergence_energies.append(convergence[5])
            convergence_DAs.append(convergence[4])

        self.DAs_at_loss = DAs_at_loss
        self.DA_at_tfinal = DA_tfinal
        self.lost_total = lost_total
        self.final_times = final_times

        self.bounces = bounces
        self.passes = passes
        self.pitch = pitch
        self.Plot_Radial = Plot_Radial

        self.Peta_init = Peta_init
        self.Peta_mean = Peta_mean
        self.Peta_final = Peta_final
        self.E_mean = E_mean
        self.E_final = E_final
        self.E_init = E_init

        self.convergence_bounces = convergence_bounces
        self.convergence_passes = convergence_passes
        self.convergence_times = convergence_times
        self.convergence_petas = convergence_petas
        self.convergence_DAs = convergence_DAs
        self.convergence_energies = convergence_energies

        if self.verbose:
            mu_per_mass0 = np.array(mus0) / self.mass
            if not exists(self.final_filepaths["ICs"]):
                np.savetxt(
                    self.final_filepaths["ICs"],
                    np.column_stack((s0, theta0, zeta0, vpar0, mu_per_mass0)),
                )
            if not exists(self.final_filepaths["DA"]):
                np.savetxt(
                    self.final_filepaths["DA"],
                    np.column_stack((DAs_at_loss, final_times)),
                )

        if self.verbose:
            print("Done Building Lists", flush=True)
        return

    def surface_trapped_func_Eprime(self, mu, surface):
        r"""
        `    Determine whether a particle with the given pitch angle is trapped on a
            specified flux surface.

            Samples modB over the surface and checks whether the parallel velocity
            calculated via the Eprime rootsolve would become imaginary at the maximum
            field strength on that surface, indicating trapping.
            The prescribed reference Ekin is used.

            Args:
                mu : Magnetic moment.
                surface     : Flux-surface label s at which to evaluate trapping.

            Returns:
                trapped : List of integers (0 or 1) indicating whether each sampled
                        point is trapped.
                peta   : List of radial like coordinates corresponding to each sampled
                        point.
        """
        resolution = 500
        points = initialize_position_uniform_surf(self.B0, resolution, surface)
        points_phase = np.column_stack((points, np.zeros(points.shape[0])))  # add time column
        self.saw.set_points(points_phase)
        modB = self.saw.B0.modB()[:, 0]
        Phi = self.saw.Phi()[:, 0]
        max_modB = np.max(modB)

        sign_arrs = np.ones_like(modB) * self.sign

        mu_pm = mu / self.mass

        vp_temp = self.vpar_func_perturbed(
            points[:, 0], points[:, 1], points[:, 2], mu_pm, sign_arrs
        )

        mask = ~np.isnan(vp_temp)
        points = points[mask]
        points_phase = points_phase[mask]
        modB = modB[mask]
        Phi = Phi[mask]
        vp_temp = vp_temp[mask]

        E = 0.5 * self.mass * vp_temp**2 + mu * modB + self.charge * Phi

        E_potential = mu * max_modB + self.charge * Phi

        output = ((E - E_potential) < 0).astype(int)
        output = output.tolist()
        if self.plot_s:
            return np.sum(output), surface
        else:
            if points.shape[0] == 0:
                return [], []
            
            peta = compute_peta(
                    self.B0,
                    points,
                    vp_temp,
                    self.mass,
                    self.charge,
                    self.helicity_M,
                    self.helicity_N,
                    self.helicity_Mp,
                    self.helicity_Np,
                )
            if np.isnan(peta).any():
                print(f"Warning: NaN values encountered in p_eta calculation at s={surface} PA = {mu/self.Eprime}", flush=True)
            if np.isinf(peta).any():
                print(f"Warning: Inf values encountered in p_eta calculation at s={surface} PA = {mu/self.Eprime}", flush=True)
            
            return output, peta.tolist()


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
        points = np.zeros((1, 3)) if np.isscalar(s) else np.zeros((len(s), 3))
        points[:, 0] = s

        if self.Eprime_slice:
            vp_temp = self.vpar_func_perturbed(
                points[:, 0], points[:, 1], points[:, 2], mu, sign
            )
        else:
            vp_temp = self.vpar_func(points[:, 0], points[:, 1], points[:, 2], mu, sign)

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

    def return_peta_trapped_contoured_boundary(self):
        r"""
        Estimate the trapped-passing boundary in the (pitch, p_eta) plane by
        sampling many points on flux surfaces, binning the trapped/passing
        indicator, and tracing the column-wise transition. The boundary is then
        fit with a quadratic polynomial.

        Args:
            binned_statistic_2d : The scipy.stats.binned_statistic_2d callable
                (injected to avoid re-importing).

        Returns:
            poly : numpy.poly1d quadratic fit of the boundary.
            pitch_fit : Pitch coordinates of the fitted curve.
            radlike_fit : Radial-like (s or p_eta) values of the fitted curve.
        """
        volume_boundary_radlike = []
        volume_boundary_pitch = []
        volume_trapped = []
        radial_space = 100
        pa_space = 100

        s_vals = np.linspace(self.s_min, self.s_max, radial_space)
        mu_vals = np.linspace(self.mu_min, self.mu_max, pa_space)

        for s_val in s_vals:
            for mu_val in mu_vals:
                trapped, radial_like = self.surface_trapped_func_Eprime(mu_val, s_val)

                pitch_val = (mu_val / self.Ekin) * self.min_volmodB
                pitch_val *= self.sign

                if self.plot_s:
                    volume_boundary_radlike.append(s_val)
                    volume_trapped.append(trapped)
                    volume_boundary_pitch.append(pitch_val)
                    continue

                volume_boundary_radlike += radial_like
                volume_trapped += trapped
                pitch_lst = [pitch_val] * len(radial_like)
                volume_boundary_pitch += pitch_lst

        volume_boundary_pitch = np.array(volume_boundary_pitch)
        volume_boundary_radlike = np.array(volume_boundary_radlike)
        volume_trapped = np.array(volume_trapped)

        if np.isnan(volume_trapped).any():
            print(f"volume trapped nan", flush=True)

        trapped_vals, pitch_edges, radlike_edges, binnumber = binned_statistic_2d(
            volume_boundary_pitch,
            volume_boundary_radlike,
            volume_trapped,
            statistic="max",
            bins=[int(pa_space*0.80), int(radial_space * 0.80)],
        )

        pitch_c = 0.5 * (pitch_edges[:-1] + pitch_edges[1:])
        radlike_c = 0.5 * (radlike_edges[:-1] + radlike_edges[1:])

        T = np.nan_to_num(trapped_vals, nan=0.0).astype(int)
        # x, y -> (pitch, peta) dimensions

        boundary_pitch, boundary_radlike = [], []

        for pitch_i in range(0, T.shape[0]):
            peta_data = T[pitch_i, :]
            if not peta_data.any() or peta_data.all():
                continue
            peta_i = int(np.argmax(peta_data > 0.5))
            peta_value = radlike_c[peta_i] if peta_i == 0 else 0.5 * (radlike_c[peta_i - 1] + radlike_c[peta_i])
            boundary_pitch.append(pitch_c[pitch_i])
            boundary_radlike.append(peta_value)

        boundary_pitch = np.array(boundary_pitch)
        boundary_radlike = np.array(boundary_radlike)

        order = np.argsort(boundary_pitch)
        boundary_pitch = boundary_pitch[order]
        boundary_radlike = boundary_radlike[order]

        coeffs = np.polyfit(boundary_pitch, boundary_radlike, 2)
        poly = np.poly1d(coeffs)

        pitch_fit = np.linspace(boundary_pitch.min(), boundary_pitch.max(), 300)
        radlike_fit = poly(pitch_fit)
        trunc = int(np.argmin(radlike_fit))
        pitch_fit = pitch_fit[:trunc]
        radlike_fit = radlike_fit[:trunc]

        return poly, pitch_fit, radlike_fit

    def plot_heatmap(
        self,
        nx=20,
        ny=20,
        savepath="heatmap_digit_accuracy.pdf",
        ax=None,
        DA_max=7,
        statistic="mean",
        DA_at_loss=False,
        plot_losses=False,
        negate_peta=False,
        smoothing=3,
    ):
        r"""
        Plot a 2D heatmap of digit accuracy in the (pitch, radial-like) plane and
        overlay the fitted trapped-passing boundary. Optionally overlay loss
        fractions as triangle markers per bin.

        Args:
            nx : Number of pitch bins.
            ny : Number of radial bins.
            savepath : Output file path for the heatmap.
            ax : Matplotlib axis. If None, a new figure and axis are created.
            DA_max : Maximum DA value shown on the colorbar.
            statistic : Aggregation statistic passed to binned_statistic_2d.
            DA_at_loss : If True, use the DA value at loss; otherwise the final
                integration DA.
            plot_losses : If True, overlay loss-fraction markers per bin.
            negate_peta : If True, flip the sign of the y axis (useful when
                plotting against -p_eta).
            smoothing : Currently unused; retained for API compatibility.

        Returns:
            ax : The Matplotlib axis containing the plot.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        if self.verbose:
            print("plotting...", flush=True)

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 12))
        else:
            fig = ax.get_figure()

        try:
            import cmcrameri.cm as cmc  # noqa: F401

            cmap = "cmc.managua"

        except ImportError:
            cmap = "viridis"

        DA_values = self.DAs_at_loss if DA_at_loss else self.DA_at_tfinal

        norm = mpl.colors.Normalize(vmin=0, vmax=DA_max)

        plotting_pitch_normalized = np.array(self.pitch) * self.min_volmodB

        DA_stats, x_edges, y_edges, binnumber = binned_statistic_2d(
            plotting_pitch_normalized,
            np.array(self.Plot_Radial),
            np.array(DA_values),
            statistic=statistic,
            bins=[nx, ny],
        )

        X, Y = np.meshgrid(x_edges, y_edges)
        if negate_peta:
            Y *= -1
        im2 = ax.pcolormesh(X, Y, DA_stats.T, shading="auto", cmap=cmap, norm=norm)

        poly, pa_fit, rad_fit = self.return_peta_trapped_contoured_boundary()
        self.trapped_boundary_fit = poly
        ax.plot(
            pa_fit,
            rad_fit,
            color="grey",
            linewidth=10,
            label="Trapped-passing boundary",
        )

        colorlabel = "Digit Accuracy"

        if plot_losses:
            lost_frac, x_edges, y_edges, _ = binned_statistic_2d(
                plotting_pitch_normalized,
                np.array(self.Plot_Radial),
                np.array(self.lost_total),
                statistic="mean",
                bins=[nx, ny],
            )
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            Xc, Yc = np.meshgrid(x_centers, y_centers)
            xf = Xc.ravel()
            yf = Yc.ravel()
            lost_frac = np.nan_to_num(lost_frac, nan=0.0).astype(int)
            af = lost_frac.T.ravel()
            if negate_peta:
                yf = yf * -1
            ax.scatter(
                xf,
                yf,
                marker="^",
                s=500,
                c="darkorange",
                edgecolors="k",
                alpha=af,
                linewidths=1,
                zorder=10,
            )
        ax.set_xlabel(r"$\lambda = \frac{\mu}{E} \text{sign}(v_{\|})$")

        if self.plot_s:
            ax.set_ylabel(r"$s$")
        else:
            if negate_peta:
                ax.set_ylabel(r"$-P_\eta$")
            else:
                ax.set_ylabel(r"$P_\eta$")

        fig.tight_layout()
        fig.colorbar(im2, ax=ax, label=colorlabel)
        plt.savefig(savepath, dpi=400)
        return ax


class WBAPerturbedParticles:
    def __init__(
        self,
        saw,
        mass,
        charge,
        Ekin,
        Phin,
        Phim,
        omega,
        helicity_N,
        helicity_M,
        helicity_Mp=None,
        helicity_Np=None,
        points=None,
        v_pars=None,
        mu_per_mass=None,
        tmax=1e-2,
        min_timestep=1e-7,
        DA_cutoff=3,
        tol=1e-9,
        gc_tys=None,
        savedata=False,
        save_gc_trajectories=False,
        savepath="",
        comm=None,
        solver_options=None,
        nconvergence_points=1,
    ):
        r"""
        Initialize the WBAPerturbedParticles instance for computing the
        Weighted Birkhoff Average (WBA) digit accuracy of guiding-center
        trajectories in a perturbed ShearAlfvenWave field.

        Either pre-traced trajectories (gc_tys) or initial conditions (points,
        v_pars, mu_per_mass) must be provided. If initial conditions are given,
        particles are traced using trace_particles_boozer_perturbed.

        Args:
            saw                  : A :class:`ShearAlfvenHarmonic` or compatible
                                perturbed-field instance.
            mass                 : Particle mass.
            charge               : Particle charge.
            Ekin                 : Total kinetic energy.
            helicity_N           : Toroidal helicity of the field-strength contours.
            helicity_M           : Poloidal helicity of the field-strength contours.
            helicity_Mp          : Poloidal helicity of the mapping coordinate eta.
                                If None, determined automatically from helicity_M.
            helicity_Np          : Toroidal helicity of the mapping coordinate eta.
                                If None, determined automatically from helicity_N.
            points               : Array of shape (N, 4) containing initial
                                coordinates (s, theta, zeta, t). Required if
                                gc_tys is None.
            v_pars               : Array of initial parallel velocities. Required
                                if gc_tys is None.
            mu_per_mass          : Array of initial magnetic moments divided by
                                mass. Required if gc_tys is None.
            tmax                 : Maximum integration time per particle
                                (default: 1e-2 s).
            min_timestep         : Minimum time-step size used as the save
                                interval (default: 1e-7 s).
            DA_cutoff            : Digit accuracy threshold below which a
                                trajectory is classified as chaotic
                                (default: 3).
            tol                  : ODE solver tolerance (default: 1e-9).
            gc_tys               : List of pre-traced trajectory arrays. If
                                provided, tracing is skipped (default: None).
            savedata             : If True, save diagnostics to disk
                                (default: False).
            save_gc_trajectories : If True, save raw trajectory arrays to disk
                                (default: False).
            savepath             : Prefix for output file names (default: '').
            comm                 : MPI communicator for parallel execution
                                (default: None).
            solver_options       : Dictionary of additional options passed to the
                                ODE solver (default: {}).
            nconvergence_points  : Number of intermediate WBA evaluations per
                                trajectory (default: 1).
        """
        if gc_tys is None and points is None:
            raise ValueError("Need to provide traced trajectories or points to trace.")

        if points is not None and (v_pars is None or mu_per_mass is None):
            raise ValueError(
                "If providing points to trace, need v_pars and mu_per_mass."
            )

        if points.shape[1] not in [3, 4]:
            raise ValueError(
                "Points must have shape (npoints, 4) for (s, theta, zeta, t) or "
                "(npoints, 3) for (s, theta, zeta)"
            )
        if points.shape[1] == 4:
            points = points[:, :3]

        if not isinstance(saw, ShearAlfvenHarmonic) and not isinstance(
            saw, ShearAlfvenWavesSuperposition
        ):
            raise TypeError(
                "Expected saw to be an instance of ShearAlfvenHarmonic "
                "or ShearAlfvenWavesSuperposition"
            )

        if solver_options is None:
            solver_options = {}

        self.saw = saw
        self.B0 = saw.B0
        self.helicity_M = helicity_M
        self.helicity_N = helicity_N

        if helicity_Mp is None or helicity_Np is None:
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

        self.Phin = Phin
        self.Phim = Phim
        self.omega = omega
        self.mass = mass
        self.charge = charge
        self.Ekin = Ekin
        self.nprime = (self.Phim * self.helicity_N - self.Phin * self.helicity_M) / (
            self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        )
        self.vtotal = np.sqrt(2 * self.Ekin / mass)

        self.comm = comm
        self.verbose = False
        if self.comm is None or self.comm.rank == 0:
            self.verbose = True

        self.solver_options = solver_options

        self.tmax = tmax
        self.min_timestep = min_timestep
        self.convergence_points = nconvergence_points

        self.savedata = savedata
        self.save_gc_trajectories = save_gc_trajectories

        # set parameters for convergence plot
        expected_length = int(self.tmax / self.min_timestep)
        expected_step = int(expected_length / self.convergence_points)
        self.WBA_transit_indicies = np.linspace(
            expected_step, expected_length - 1, num=nconvergence_points, dtype=int
        ).tolist()
        self.convergence_plot = nconvergence_points > 1

        self.tol = tol

        if self.savedata:
            self.IC_filepaths = "initial_conditions.txt"
            self.final_filepaths = {
                "DA": savepath + "DA_walllosttimes.txt",
                "TRAJS": savepath + "gc_tys.txt",
                "DATA": savepath + "DATA.txt",
            }

        if gc_tys is None:
            self.trace = True
        else:
            self.trace = False
            self.gc_tys = gc_tys

            if points is None:
                s_ic = []
                theta_ic = []
                zeta_ic = []
                v_pars = []

                for elem in gc_tys:
                    s_ic.append(elem[-1, 1])
                    theta_ic.append(elem[-1, 2])
                    zeta_ic.append(elem[-1, 3])
                    v_pars.append(elem[-1, 4])

                points = np.zeros((len(gc_tys), 3))
                points[:, 0] = s_ic
                points[:, 1] = theta_ic
                points[:, 2] = zeta_ic

            if mu_per_mass is None:
                raise Warning(
                    "Expected mu_per_mass to be provided with gc_tys",
                    "Computing mu_per_mass from gc_tys with reference energy, this"
                    " may be inaccurate if not provided directly.",
                )
                mu_per_mass = []
                for i in range(len(gc_tys)):
                    self.B0.set_points(points[i, :])
                    modB = self.B0.modB()[:, 0]

                    eperp_per_mass = (self.Ekin / self.mass) - 0.5 * v_pars[i] ** 2
                    mu_per_mass.append(eperp_per_mass / modB[0])

        self.points0 = points
        self.v_pars0 = v_pars
        self.mu_per_mass0 = mu_per_mass

        (self.gc_tys, self.DAs, self.wall_lost, self.dense_output) = (
            self.trace_particles()
        )
        return 

    def check_filepaths(self, filepaths):
        r"""
        Check whether all provided output file paths exist.

        Args:
            filepaths : Dictionary of file labels to filesystem paths.
        Returns:
            exists_all : True if every path exists, otherwise False.
        """
        return all(exists(fp) for fp in filepaths.values())

    def trace_particles(self):
        r"""
        Trace perturbed particle trajectories and compute DA outputs.

        Returns:
            res_tys : Per-particle trajectory summaries.
        """
        first, last = parallel_loop_bounds(self.comm, self.points0.shape[0])

        DA_data = []
        dense_output = []
        wall_lost = []
        res_tys = []

        for itrj in range(first, last):
            if self.trace:
                pts = np.zeros((1,3))
                pts[:,0] = self.points0[itrj, 0]
                pts[:,1] = self.points0[itrj, 1]
                pts[:,2] = self.points0[itrj, 2]
                gc_tys, gc_zeta_hits = trace_particles_boozer_perturbed(
                    perturbed_field=self.saw,
                    stz_inits=pts,
                    parallel_speeds=[self.v_pars0[itrj]],
                    mus=[self.mu_per_mass0[itrj]],
                    tmax=self.tmax,
                    mass=self.mass,
                    charge=self.charge,
                    Ekin=self.Ekin,
                    abstol=self.tol,
                    reltol=self.tol,
                    dt_save=self.min_timestep,
                    stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
                    mode="gc_noK",
                    ODE_solver="dormand_prince",
                    **self.solver_options,
                )
                points_trajectory = gc_tys[0]
            else:
                points_trajectory = self.gc_tys[itrj]

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

            bounce_indices, passing_indicies = return_bounces_and_passes(
                vpar_path, zeta_path
            )

            self.saw.set_points(points_trajectory)

            modB = self.saw.B0.modB()[:, 0]
            Phi = self.saw.Phi()[:, 0]

            weighted_mu = self.mu_per_mass0[itrj] * self.mass

            E = 0.5 * self.mass * vpar_path**2 + weighted_mu * modB + self.charge * Phi

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

            if points_trajectory.shape[0] > 10:
                stack_data = np.column_stack((points_trajectory[:, -1], Peta_values))
                time_eval, DA_eval = return_DA(stack_data)
                final_DA = DA_eval
            else:
                final_DA = np.nan

            convergence_times = []
            convergence_petas = []
            convergence_energies = []
            convergence_bounces = []
            convergence_passes = []
            convergence_DAs = []

            for _conv_index, timing_index in enumerate(self.WBA_transit_indicies):
                if timing_index > len(time_momentum):
                    break
                convergence_times.append(time_momentum[timing_index])
                convergence_petas.append(Peta_values[timing_index])
                convergence_energies.append(E[timing_index])
                bounce_enum = np.searchsorted(bounce_indices, timing_index, side="left")

                pass_enum = np.searchsorted(passing_indicies, timing_index, side="left")

                convergence_bounces.append(bounce_enum)
                convergence_passes.append(pass_enum)

                stack_data = np.column_stack(
                    (time_momentum[:timing_index], Peta_values[:timing_index])
                )
                time_eval, DA_eval = return_DA(stack_data)
                convergence_DAs.append(DA_eval)

            convergence_data = [
                convergence_times,
                convergence_petas,
                convergence_bounces,
                convergence_passes,
                convergence_DAs,
                convergence_energies,
            ]
            # start state:
            # [s, theta, zeta, vpar, peta, E, Eprime]
            # end state:
            # [t, s, theta, zeta, vpar, peta, E, Eprime, bounces, passes, DA]
            # mean state:
            # [s_mean, peta_mean, E_mean, Eprime_mean]
            start_state = [
                points_trajectory[0, 0],
                points_trajectory[0, 1],
                points_trajectory[0, 2],
                vpar_path[0],
                weighted_mu,
                Peta_values[0],
                E[0],
                Eprime[0],
            ]

            end_state = [
                points_trajectory[-1, 3],
                points_trajectory[-1, 0],
                points_trajectory[-1, 1],
                points_trajectory[-1, 2],
                vpar_path[-1],
                Peta_values[-1],
                E[-1],
                Eprime[-1],
                len(bounce_indices),
                len(passing_indicies),
                final_DA,
            ]

            mean_state = [
                np.mean(points_trajectory[:, 0]),
                np.mean(Peta_values),
                np.mean(E),
                np.mean(Eprime),
            ]

            particle_out = [start_state, end_state, mean_state, convergence_data]
            DA_data.append(final_DA)
            wall_lost.append(points_trajectory[-1:3])
            dense_output.append(particle_out)
            if self.save_gc_trajectories:
                res_tys.append(points_trajectory)

        if self.comm is not None:
            res_tys = [i for o in self.comm.allgather(res_tys) for i in o]
            DA_data = [i for o in self.comm.allgather(DA_data) for i in o]
            wall_lost = [i for o in self.comm.allgather(wall_lost) for i in o]
            dense_output = [i for o in self.comm.allgather(dense_output) for i in o]

        if self.verbose:
            import pickle

            if self.save_gc_trajectories:
                self.gc_tys = res_tys
                with open(self.final_filepaths["TRAJS"], "wb") as f:
                    pickle.dump(res_tys, f)
            if self.savedata:
                np.savetxt(
                    self.final_filepaths["DA"], np.column_stack((DA_data, wall_lost))
                )
                with open(self.final_filepaths["DATA"], "wb") as f:
                    pickle.dump(dense_output, f)
        if not self.trace:
            res_tys = self.gc_tys
        self.build_lists(dense_output)
        return res_tys, DA_data, wall_lost, dense_output
    def build_lists(self, dense_output):
        r"""
        Process trajectory summary data into lists that can be compared
        across many particles.

        Populates self.DAs_at_loss, self.DA_at_tfinal, self.lost_total,
        self.final_times, self.bounces, self.passes, self.Peta_init/mean/final,
        self.E_init/mean/final, self.Eprime_init/mean/final, and the
        convergence_* arrays from the list of per-particle state tuples
        produced by trace_particles.

        Args:
            dense_output : List of per-particle trajectory summaries, each a
                list of the form
                [start_state, end_state, mean_state, convergence_data].
        """
        if self.verbose:
            print("Building Lists", flush=True)

        DAs_at_loss = []
        DA_tfinal = []

        lost_total = []
        final_times = []
        bounces = []
        passes = []

        mus = []
        Peta_init = []
        Peta_mean = []
        Peta_final = []
        E_init = []
        E_mean = []
        E_final = []
        Eprime_init = []
        Eprime_mean = []
        Eprime_final = []

        s0 = []
        theta0 = []
        zeta0 = []
        vpar0 = []

        convergence_bounces = []
        convergence_passes = []
        convergence_times = []
        convergence_petas = []
        convergence_energies = []
        convergence_DAs = []

        for elem in dense_output:
            # start state vector:
            #   [s, theta, zeta, vpar, mu, peta, E, Eprime]
            # end state vector:
            #   [t, s, theta, zeta, vpar, peta, E, Eprime, bounces, passes, DA]
            # mean state vector:
            #   [s_mean, peta_mean, E_mean, Eprime_mean]
            # convergence state vector:
            #   [times, petas, bounces, passes, DAs, energies]

            start_state = elem[0]
            end_state = elem[1]
            means = elem[2]
            convergence = elem[3]

            final_time = end_state[0]

            final_times.append(final_time)

            if final_time < (self.tmax - (5 * self.min_timestep)):
                lost_total.append(1)
                DA_tfinal.append(np.nan)
            else:
                lost_total.append(0)
                DA_tfinal.append(end_state[10])

            DAs_at_loss.append(end_state[10])
            bounces.append(end_state[8])
            passes.append(end_state[9])

            s0.append(start_state[0])
            theta0.append(start_state[1])
            zeta0.append(start_state[2])
            vpar0.append(start_state[3])

            mus.append(start_state[4])

            Peta_init.append(start_state[5])
            Peta_mean.append(means[1])
            Peta_final.append(end_state[5])

            E_init.append(start_state[6])
            E_mean.append(means[2])
            E_final.append(end_state[6])

            Eprime_init.append(start_state[7])
            Eprime_mean.append(means[3])
            Eprime_final.append(end_state[7])

            convergence_times.append(convergence[0])
            convergence_petas.append(convergence[1])
            convergence_bounces.append(convergence[2])
            convergence_passes.append(convergence[3])
            convergence_DAs.append(convergence[4])
            convergence_energies.append(convergence[5])

        self.mus = mus
        self.DAs_at_loss = DAs_at_loss
        self.DA_at_tfinal = DA_tfinal
        self.lost_total = lost_total
        self.final_times = final_times

        self.bounces = bounces
        self.passes = passes

        self.Peta_init = Peta_init
        self.Peta_mean = Peta_mean
        self.Peta_final = Peta_final
        self.E_init = E_init
        self.E_mean = E_mean
        self.E_final = E_final
        self.Eprime_init = Eprime_init
        self.Eprime_mean = Eprime_mean
        self.Eprime_final = Eprime_final

        self.convergence_bounces = convergence_bounces
        self.convergence_passes = convergence_passes
        self.convergence_times = convergence_times
        self.convergence_petas = convergence_petas
        self.convergence_DAs = convergence_DAs
        self.convergence_energies = convergence_energies

        if self.verbose:
            print("Done Building Lists", flush=True)
        return


class WBAParticles:
    def __init__(
        self,
        B0,
        mass,
        charge,
        Ekin,
        helicity_N,
        helicity_M,
        helicity_Mp=None,
        helicity_Np=None,
        points=None,
        v_pars=None,
        gc_tys=None,
        tmax=1e-2,
        min_timestep=1e-7,
        savedata=False,
        save_gc_trajectories=False,
        savepath="",
        comm=None,
        DA_cutoff=3,
        solver_options=None,
        tol=1e-9,
        convergence_points=1,
    ):
        r"""
        Initialize a Weighted Birkhoff Average (WBA) computation for guiding-
        center trajectories in an unperturbed BoozerMagneticField.

        Either pre-traced trajectories (gc_tys) or initial conditions
        (points, v_pars) must be provided. If initial conditions are given,
        particles are traced with trace_particles_boozer.

        Args:
            B0 : The :class:`BoozerMagneticField` instance.
            mass : Particle mass.
            charge : Particle charge.
            Ekin : Total kinetic energy.
            helicity_N : Toroidal helicity of the field-strength contours.
            helicity_M : Poloidal helicity of the field-strength contours.
            helicity_Mp : Poloidal helicity of the mapping coordinate eta. If
                None, determined automatically from helicity_M.
            helicity_Np : Toroidal helicity of the mapping coordinate eta. If
                None, determined automatically from helicity_N.
            points : Array of shape (N, 3) of initial (s, theta, zeta). Required
                if gc_tys is None.
            v_pars : Array of initial parallel velocities. Required if gc_tys is
                None.
            gc_tys : Pre-traced trajectory arrays; if provided, tracing is
                skipped.
            tmax : Maximum integration time per particle (s).
            min_timestep : Save interval for the ODE solver (s).
            savedata : If True, save DA arrays and trajectories to disk.
            save_gc_trajectories : If True, persist raw trajectories.
            savepath : Prefix for output filenames.
            comm : MPI communicator for parallel execution.
            DA_cutoff : Digit accuracy threshold for classifying chaos.
            solver_options : Extra options passed to the ODE solver.
            tol : Absolute and relative ODE tolerance.
            convergence_points : Number of intermediate WBA evaluations per
                trajectory.
        """
        self.B0 = B0
        self.helicity_M = helicity_M
        self.helicity_N = helicity_N

        if gc_tys is None and points is None:
            raise ValueError("Need to provide trajctories or points.")

        if gc_tys is None and points is None:
            raise ValueError("Need to provide traced trajectories or points to trace.")

        if points is not None and (v_pars is None):
            raise ValueError(
                "If providing points to trace, need v_pars and mu_per_mass."
            )

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

        self.mass = mass
        self.charge = charge
        self.Ekin = Ekin
        self.convergence_points = convergence_points
        self.vtotal = np.sqrt(2 * self.Ekin / mass)

        self.comm = comm
        self.verbose = False
        if self.comm is None or self.comm.rank == 0:
            self.verbose = True

        if solver_options is None:
            solver_options = {}
        self.solver_options = solver_options
        self.tmax = tmax

        self.min_timestep = min_timestep
        self.savedata = savedata
        self.savepath = savepath
        self.save_gc_trajectories = save_gc_trajectories

        # set parameters for convergence plot
        expected_length = int(self.tmax / self.min_timestep)
        expected_step = int(expected_length / self.convergence_points)
        self.WBA_transit_indicies = np.linspace(
            expected_step, expected_length - 1, num=self.convergence_points, dtype=int
        ).tolist()
        self.convergence_plot = self.convergence_points > 1

        self.points0 = points
        self.v_pars0 = v_pars

        self.tol = tol

        if gc_tys is None:
            self.trace = True
        else:
            self.trace = False
            self.gc_tys = gc_tys

        if self.savedata:
            self.IC_filepaths = "initial_conditions.txt"
            self.final_filepaths = {
                "DA": self.savepath + "DA_walllosttimes.txt",
                "TRAJS": self.savepath + "gc_tys.txt",
                "DATA": self.savepath + "DATA.txt",
            }
        self.gc_tys, self.DAs, self.wall_lost, self.dense_output = (
            self.trace_particles()
        )
        self.build_lists(self.dense_output)

    def check_filepaths(self, filepaths):
        r"""
        Check whether all provided output file paths exist.

        Args:
            filepaths : Dictionary of file labels to filesystem paths.
        Returns:
            exists_all : True if every path exists, otherwise False.
        """
        return all(exists(fp) for fp in filepaths.values())


    def trace_particles(self):
        r"""
        Trace unperturbed particle trajectories and compute per-particle WBA
        digit-accuracy diagnostics.

        If self.trace is True, particles are integrated with
        trace_particles_boozer from the initial conditions (points_phase, vpars).
        Otherwise, pre-traced trajectories in self.gc_tys are reused. For each
        trajectory, the canonical momentum p_eta is evaluated, bounces and
        toroidal transits are counted, and the WBA digit accuracy is computed at
        every index in self.WBA_transit_indicies as well as at the final time.

        Args:
            points_phase : Array of shape (N, 3) of initial (s, theta, zeta)
                coordinates. Ignored when self.trace is False.
            vpars : Array of initial parallel velocities. Ignored when self.trace
                is False.

        Returns:
            res_tys : List of raw trajectory arrays (only populated when
                self.save_gc_trajectories is True; otherwise an empty list, or
                self.gc_tys when self.trace is False).
            DA_data : List of final digit-accuracy values, one per particle.
            wall_lost : List of final integration times per particle (used to
                flag wall-loss events).
            dense_output : List of per-particle summaries of the form
                [start_state, end_state, convergence_data].
        """

        shape = self.points0.shape[0] if self.trace else len(self.gc_tys)

        first, last = parallel_loop_bounds(self.comm, shape)
        res_tys = []

        DA_data = []
        dense_output = []
        wall_lost = []
        res_tys = []

        for itrj in range(first, last):
            if self.trace:
                pt = np.zeros((1, 3))
                pt[0, 0] = self.points0[itrj, 0]
                pt[0, 1] = self.points0[itrj, 1]
                pt[0, 2] = self.points0[itrj, 2]
                self.vtotal = np.sqrt(2 * self.Ekin / self.mass)
                gc_tys, gc_zeta_hits = trace_particles_boozer(
                    self.B0,
                    stz_inits=pt,
                    parallel_speeds=[self.v_pars0[itrj]],
                    tmax=self.tmax,
                    mass=self.mass,
                    charge=self.charge,
                    Ekin=self.Ekin,
                    abstol=self.tol,
                    reltol=self.tol,
                    dt_save=self.min_timestep,
                    stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
                    mode="gc_noK",
                    ODE_solver="dormand_prince",
                    **self.solver_options,
                )
                points_trajectory = gc_tys[0]
            else:
                points_trajectory = self.gc_tys[itrj]

            time_momentum = points_trajectory[:, 0]
            s_path = points_trajectory[:, 1]
            theta_path = points_trajectory[:, 2]
            zeta_path = points_trajectory[:, 3]
            vpar_path = points_trajectory[:, 4]

            points_trajectory = np.column_stack((s_path, theta_path, zeta_path))
            idx_wall = np.argmax(s_path >= 1) if np.any(s_path >= 1) else None
            if idx_wall is not None and s_path[idx_wall] >= 1:
                idx_wall -= 1
                points_trajectory = points_trajectory[:idx_wall, :]
                time_momentum = time_momentum[:idx_wall]
                vpar_path = vpar_path[:idx_wall]

            Peta_values = compute_peta(
                self.B0,
                points=points_trajectory,
                vpar=vpar_path,
                mass=self.mass,
                charge=self.charge,
                helicity_M=self.helicity_M,
                helicity_N=self.helicity_N,
            )

            if points_trajectory.shape[0] > 8:
                stack_data = np.column_stack((points_trajectory[:, -1], Peta_values))
                time_eval, DA_eval = return_DA(stack_data)
                final_DA = DA_eval
            else:
                final_DA = np.nan

            bounce_indices, passing_indicies = return_bounces_and_passes(
                vpar_path, points_trajectory[:, 2]
            )

            convergence_times = []
            convergence_petas = []
            convergence_bounces = []
            convergence_passes = []
            convergence_DAs = []

            for _conv_index, timing_index in enumerate(self.WBA_transit_indicies):
                if timing_index > len(time_momentum):
                    break
                convergence_times.append(time_momentum[timing_index])
                convergence_petas.append(Peta_values[timing_index])

                bounce_enum = np.searchsorted(bounce_indices, timing_index, side="left")

                pass_enum = np.searchsorted(passing_indicies, timing_index, side="left")

                convergence_bounces.append(bounce_enum)
                convergence_passes.append(pass_enum)

                stack_data = np.column_stack(
                    (time_momentum[:timing_index], Peta_values[:timing_index])
                )
                time_eval, DA_eval = return_DA(stack_data)
                convergence_DAs.append(DA_eval)

            convergence_data = [
                convergence_times,
                convergence_petas,
                convergence_bounces,
                convergence_passes,
                convergence_DAs,
            ]
            points = np.zeros((1, 3))
            points[:, 0] = points_trajectory[-1, 0]
            points[:, 1] = points_trajectory[-1, 1]
            points[:, 2] = points_trajectory[-1, 2]
            self.B0.set_points(points)
            B = self.B0.modB()[0, 0]
            mu = (1 / 2) * self.mass * (self.vtotal**2 - vpar_path[0] ** 2) / B
            # start state vector:  [s, theta, zeta, vpar, mu]
            # end state vector:   [t, s, theta, zeta, vpar,  bounces, passes, DA]
            start_state = [
                points_trajectory[0, 0],
                points_trajectory[0, 1],
                points_trajectory[0, 2],
                vpar_path[0],
                mu,
            ]

            end_state = [
                time_momentum[-1],
                points_trajectory[-1, 0],
                points_trajectory[-1, 1],
                points_trajectory[-1, 2],
                vpar_path[-1],
                len(bounce_indices),
                len(passing_indicies),
                final_DA,
            ]

            particle_out = [start_state, end_state, convergence_data]
            DA_data.append(final_DA)
            wall_lost.append(time_momentum[-1])
            dense_output.append(particle_out)

        if self.comm is not None:
            res_tys = [i for o in self.comm.allgather(res_tys) for i in o]
            DA_data = [i for o in self.comm.allgather(DA_data) for i in o]
            wall_lost = [i for o in self.comm.allgather(wall_lost) for i in o]
            dense_output = [i for o in self.comm.allgather(dense_output) for i in o]

        if self.verbose:
            import pickle

            if self.save_gc_trajectories and self.trace:
                with open(self.final_filepaths["TRAJS"], "wb") as f:
                    pickle.dump(res_tys, f)
            if self.savedata:
                np.savetxt(
                    self.final_filepaths["DA"], np.column_stack((DA_data, wall_lost))
                )
                with open(self.final_filepaths["DATA"], "wb") as f:
                    pickle.dump(dense_output, f)
        if not self.trace:
            res_tys = self.gc_tys
        return res_tys, DA_data, wall_lost, dense_output

    def build_lists(self, dense_output):
        r"""
        Process trajectory summary data into lists that can be compared
        across many particles.

        Populates self.DAs_at_loss, self.DA_at_tfinal, self.lost_total,
        self.final_times, self.trapped, self.Peta_start, self.pitch, and the
        convergence_* arrays from the list of per-particle state tuples produced
        by trace_particles.

        Args:
            res_tys : List of per-particle trajectory summaries, each a list of
                    the form [start_state, end_state, convergence_data].
        """
        if self.verbose:
            print("Building Lists", flush=True)

        DAs_at_loss = []
        DA_tfinal = []

        lost_total = []
        final_times = []
        bounces = []
        passes = []

        s0 = []
        theta0 = []
        zeta0 = []
        vpar0 = []
        mu0 = []

        convergence_bounces = []
        convergence_passes = []
        convergence_times = []
        convergence_petas = []
        convergence_DAs = []

        for elem in dense_output:
            # start state vector:
            #   [s, theta, zeta, vpar, mu]
            # end state vector:
            #   [t, s, theta, zeta, vpar,  bounces, passes, DA]
            # convergence state vector:
            #   [times, petas, bounces, passes, DAs]

            start_state = elem[0]
            end_state = elem[1]
            convergence = elem[2]

            final_time = end_state[0]

            final_times.append(final_time)

            if final_time < (self.tmax - (5 * self.min_timestep)):
                lost_total.append(1)
                DA_tfinal.append(np.nan)
            else:
                lost_total.append(0)
                DA_tfinal.append(end_state[7])

            DAs_at_loss.append(end_state[7])
            bounces.append(end_state[5])
            passes.append(end_state[6])

            s0.append(start_state[0])
            theta0.append(start_state[1])
            zeta0.append(start_state[2])
            vpar0.append(start_state[3])
            mu0.append(start_state[4])

            convergence_bounces.append(convergence[2])
            convergence_passes.append(convergence[3])
            convergence_times.append(convergence[0])
            convergence_petas.append(convergence[1])
            convergence_DAs.append(convergence[4])

        self.DAs_at_loss = DAs_at_loss
        self.DA_at_tfinal = DA_tfinal
        self.lost_total = lost_total
        self.final_times = final_times

        self.bounces = bounces
        self.passes = passes

        self.convergence_bounces = convergence_bounces
        self.convergence_passes = convergence_passes
        self.convergence_times = convergence_times
        self.convergence_petas = convergence_petas
        self.convergence_DAs = convergence_DAs

        if self.verbose:
            print("Done Building Lists", flush=True)
        return


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
