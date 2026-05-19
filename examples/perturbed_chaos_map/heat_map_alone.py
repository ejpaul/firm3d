import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenHarmonic,
)
from firm3d.field.trajectory_helpers import (
    MapPhaseSpace,
    PassingPoincare,
    compute_peta,
    min_volumemodB,
)
from firm3d.saw.ae3d import AE3DEigenvector
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_size = comm.size
    verbose = comm.rank == 0
except ImportError:
    comm = None
    comm_size = 1
    verbose = True


# harmonic to isolate for this case
harmonic = 1
sign_vpar = 1
harmonics = {harmonic: {}}
max_ell = 2
# list of possible line styles for the resonance lines, indexed by ell value
# (shifted by max_ell to avoid negative indices)
#  ell =                 -2           -1      0           1           2
possible_linestyles = [(0, (1, 5)), "dotted", "solid", "dashed", (0, (5, 5))]

helicity_M = 1
helicity_N = -4
helicity_Mp = 0
helicity_Np = -1

boozmn_filename = "../inputs/boozmn_beta2.5_QH.nc"
perfect = True  # enforce perfect QS?
AE_filename = "QH_10harmonics_scale0_00464159.npy"
plot_losses = False

# poincare parameters
nchi_poinc = 5
ns_poinc = 500
Nmaps = 2000

# Eprime parameters
lam = 0.0
p0_int = 0.1

order = 3
degree = 3
resolution = 45
# resolution for perfect QS enforced if needed
res_p = 40

mpl.rcParams["font.size"] = 25  # base font size
mpl.rcParams["axes.labelsize"] = 25  # x/y labels
mpl.rcParams["axes.titlesize"] = 25
mpl.rcParams["xtick.labelsize"] = 25
mpl.rcParams["ytick.labelsize"] = 25
mpl.rcParams["legend.fontsize"] = 25
mpl.rcParams["figure.titlesize"] = 25

ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation


dir = "coprop" if sign_vpar else "counter"

if perfect:
    bri = BoozerRadialInterpolant(
        boozmn_filename,
        order,
        no_K=True,
        helicity_M=helicity_M,
        helicity_N=helicity_N,
        comm=comm,
    )
else:
    bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm)
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

if perfect:
    bri_p, field_p = bri, field
else:
    bri_p = BoozerRadialInterpolant(
        boozmn_filename,
        order,
        no_K=True,
        helicity_M=helicity_M,
        helicity_N=helicity_N,
        comm=comm,
    )
    field_p = InterpolatedBoozerField(
        bri_p,
        degree,
        ns_interp=res_p,
        ntheta_interp=res_p,
        nzeta_interp=res_p,
    )


AE_temp = AE3DEigenvector.load_from_numpy(AE_filename)
omega = np.sqrt(AE_temp.eigenvalue) * 1000
Phihat = (AE_temp.s_coords, AE_temp.harmonics[harmonic].amplitudes)
Phin = AE_temp.harmonics[harmonic].n
Phim = AE_temp.harmonics[harmonic].m
saw = ShearAlfvenHarmonic(Phihat, Phim=Phim, Phin=Phin, omega=omega, B0=field, phase=0)

filepath = f"qa_{Phim}_{Phin}_{dir}" if perfect else f"bqa_{Phim}_{Phin}_{dir}"

mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
Ekin = FUSION_ALPHA_PARTICLE_ENERGY

vtotal = np.sqrt(2 * Ekin / mass)

# Calculate Eprime for the given parameters
p0 = np.zeros((1, 3))
p0[0, 0] = p0_int  # s
v0 = np.sqrt(2 * Ekin / mass)  # Total velocity from kinetic energy
mu = 0.5 * lam * v0**2  # mu = vperp^2/(2 B)
saw.B0.set_points(p0)
modB = saw.B0.modB()[0, 0]
if 1 - lam * modB < 0:
    raise ValueError("Invalid parameter p0: 1 - lambda * modB must be non-negative.")
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
nprime = (Phim * helicity_N - Phin * helicity_M) / (
    helicity_Np * helicity_M - helicity_N * helicity_Mp
)
Eprime = nprime * Ekin - omega * Peta0
Eprime = Eprime[0]

min_volmodB = min_volumemodB(field)

heat_map = MapPhaseSpace(
    saw,
    Phin,
    Phim,
    omega,
    mass,
    charge,
    Ekin,
    helicity_N,
    helicity_M,
    helicity_Mp,
    helicity_Np,
    Eprime=Eprime,
    sign_vpar=sign_vpar,
    tmax=1e-2,
    comm=comm,
    savedata=True,
    file_name="",
    convergence_points=5,
)


def compute_rotational_profile(pitch, sgn, s_profile, comm):
    poinc = PassingPoincare(
        field_p,
        np.abs(pitch),
        sgn,
        mass,
        charge,
        Ekin,
        ns_poinc=120,
        ntheta_poinc=1,
        Nmaps=50,
        comm=comm,
        tmax=1e-2,
        solver_options={"axis": 0},
        helicity_M=helicity_M,
        helicity_N=helicity_N,
        helicity_Mp=helicity_Mp,
        helicity_Np=helicity_Np,
    )
    data = poinc.compute_frequencies(s_profile=s_profile)
    # returns omega_theta_prof, omega_zeta_prof, peta_prof, s_prof
    # or omega_theta_prof, omega_zeta_prof, s_prof if s_profile is True
    data = np.column_stack(
        [
            data[2],
            data[0],
            data[1],
            [data[0][i] / data[1][i] for i in range(len(data[0]))],
        ]
    )
    profiles = data[data[:, 0].argsort()]
    # sort by radial coordinate
    # return radial_position, omega_theta, omega_zeta, orbit_helicity
    return profiles


def calculate_crossings(drift_helicity, h_res, radial_position):
    diff = drift_helicity - h_res
    sign_changes = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    crossings = []
    for i in sign_changes:
        s = radial_position[i]
        crossings.append(s)
    return crossings


def calculate_QS_resonance(Phim, Phin, M, N, omega, drift_omega_zeta, ell):
    return (Phin - N * Phim - omega / drift_omega_zeta) / (Phim + ell) + N


max_mu = Ekin / min_volmodB
# iterate through pitch angles
mu_harmonics = np.linspace(0, max_mu, 50)
perturbed_pitch_angle = []
resonance_loc = []


plt.clf()
plt.xlabel("s", fontsize=12)
plt.ylabel("h", fontsize=12)
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12

for plot_counter, mu_h in enumerate(mu_harmonics):
    # compute rotational profile for given pitch angle
    profile = compute_rotational_profile(
        mu_h / (min_volmodB * Ekin), sign_vpar, False, comm=comm
    )
    perturbed_pitch_angle.append(sign_vpar * np.abs(mu_h) / Eprime)
    drift_helicity = profile[:, 3]
    radial_position = profile[:, 0]

    if verbose and (
        plot_counter % 5 == 0
    ):  # only plot some pitch angles to avoid overcrowding
        plt.plot(
            radial_position,
            drift_helicity,
            label=f"$\\lambda$={sign_vpar * np.abs(mu_h) / Eprime:.2f}",
        )

    # creates a dictionary of the form
    #  {harmonic: {ell: [[resonance_peta], [resonance_radius]]}}
    # to store the resonance locations for each harmonic and ell value
    for h in harmonics:
        Phim_h = AE_temp.harmonics[h].m
        Phin_h = AE_temp.harmonics[h].n

        crossings = []
        for ell in range(-max_ell, max_ell + 1):
            h_res = calculate_QS_resonance(
                Phim_h,
                Phin_h,
                helicity_M,
                helicity_N,
                omega,
                np.mean(profile[:, 2]),
                ell=ell,
            )
            crossings = calculate_crossings(drift_helicity, h_res, radial_position)
            if len(crossings) != 0:
                for crossing_index, radius in enumerate(crossings):
                    if ell in harmonics[h]:
                        # if the resonance location intercepts the rotational
                        # profile multiple times, we want to store all of the
                        # crossing locations. if this is the first entry, start
                        # empty lists
                        if crossing_index > (len(harmonics[h][ell]) - 1):
                            harmonics[h][ell].append([[], []])
                        harmonics[h][ell][crossing_index][0].append(
                            sign_vpar * np.abs(mu_h) / Eprime
                        )
                        harmonics[h][ell][crossing_index][1].append(radius)
                    else:
                        harmonics[h][ell] = [
                            [[sign_vpar * np.abs(mu_h) / Eprime], [radius]]
                        ]

    if verbose:
        # plot the rotational profile
        plt.tight_layout()
        plt.legend(fontsize=14, markerscale=1.5)
        plt.savefig(filepath + "harmonic_profile.png", dpi=400)
    if comm is not None:
        comm.Barrier()


plt.clf()

lines_modes = []
mpl.rcParams["xtick.labelsize"] = 25
mpl.rcParams["ytick.labelsize"] = 25

# compute rotational profile for single pitch angle for poincare plot
ell_list = []
rad_list = []
lines_modes = []
profile = compute_rotational_profile(lam, sign_vpar, True, comm)
for ell in range(-max_ell, max_ell + 1):
    h_res = calculate_QS_resonance(
        Phim, Phin, helicity_M, helicity_N, omega, np.mean(profile[:, 2]), ell=ell
    )
    crossings = calculate_crossings(profile[:, 3], h_res, profile[:, 0])
    for radius in crossings:
        ell_list.append(ell)
        rad_list.append(radius)
        lines_modes.append(harmonic)

# list of harmonics that we calculated resonances for
hlist = list(harmonics.keys())

# create a colormap for the lines based on the harmonic number
harmonic_cmap = mpl.cm.cividis
harmonic_cmap = harmonic_cmap.reversed()
if len(hlist) > 1:
    norm = mpl.colors.Normalize(vmin=hlist[0], vmax=hlist[-1])
else:
    norm = mpl.colors.Normalize(vmin=hlist[0], vmax=hlist[0] + 1)

if comm is not None:
    comm.Barrier()

if verbose:
    # make a list of line colors for each harmonic
    linecolors = [harmonic_cmap(norm(h)) for h in lines_modes]

    # create subplots
    fig, (ax_right, ax_dummy) = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [4, 0.61]}, figsize=(17, 12)
    )

    # plot heatmap
    ax_right = heat_map.plot_heatmap(
        ax=ax_right, savepath="heatmap.png", plot_losses=plot_losses
    )

    fig = ax_right.get_figure()

    lines = []
    # iterate through harmonics to plot
    for h in harmonics:
        Phim_h = AE_temp.harmonics[h].m
        Phin_h = AE_temp.harmonics[h].n

        for ell in harmonics[h]:
            for crossing_line_index, crossing_line in enumerate(harmonics[h][ell]):
                resonance_peta = np.asarray(crossing_line[0])
                resonance_pitch = np.asarray(crossing_line[1])
                color = harmonic_cmap(norm(h))

                # don't repeat label if resonance line crosses multiple times
                if crossing_line_index == 0:
                    label = rf"m,n={Phim_h},{Phin_h} $\ell$={ell}"
                else:
                    label = None

                # fit a curve to the resonance points to plot on the heatmap
                trapped_passing_fit = heat_map.trapped_boundary_fit(resonance_pitch)

                diff = trapped_passing_fit - resonance_peta
                sign_changes = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]

                (line,) = ax_right.plot(
                    resonance_pitch[: sign_changes[0]],
                    resonance_peta[: sign_changes[0]],
                    linewidth=5,
                    linestyle=possible_linestyles[ell + max_ell],
                    label=label,
                    color=color,
                )

                if crossing_line_index == 0:
                    lines.append(line)
    legend_handles, legend_labels = ax_right.get_legend_handles_labels()
    ax_right.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(1.2, 1.0),
        borderaxespad=0.0,
    )
    ax_dummy.remove()
    fig.savefig("heatmap.png", dpi=400)

    plt.clf()
if comm is not None:
    comm.Barrier()
