import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
    ShearAlfvenHarmonic,
)
from firm3d.trajectory_helpers import (
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

from firm3d.util.functions import in_github_actions
from firm3d.util.mpi import comm_world, verbose

# harmonic to isolate for this case
harmonic = 1
sign_vpar = 1
harmonics = {harmonic: {}}
max_ell = 1
# list of possible line styles for the resonance lines, indexed by ell value
# (shifted by max_ell to avoid negative indices)
#  ell =                 -1      0           1
possible_linestyles = ["dotted", "solid", "dashed"]

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
ns_poinc = 5 if in_github_actions else 100
Nmaps = 5 if in_github_actions else 1000  # number of maps for poincare

ns_points = 5 if in_github_actions else 30  # number of radial grid points for heatmap
particles_per_surface = (
    2 if in_github_actions else 20
)  # number of particles per radial grid point for heatmap
nlambda_points = 5 if in_github_actions else 30  # number of lambda points for heatmap
tmax = 1e-4 if in_github_actions else 1e-2  # maximum time for trajectory integration

# Eprime parameters
lam = 0.0
p0_int = 0.1

order = 3
degree = 3
resolution = 10 if in_github_actions else 48
# resolution for perfect QS enforced if needed
res_p = 10 if in_github_actions else 48

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
        comm=comm_world,
    )
else:
    bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world)
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
        comm=comm_world,
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

filepath = f"qh_{Phim}_{Phin}_{dir}" if perfect else f"bqh_{Phim}_{Phin}_{dir}"

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
    ns_points=ns_points,
    particles_per_surface=particles_per_surface,
    nlambda_points=nlambda_points,
    sign_vpar=sign_vpar,
    tmax=tmax,
    comm=comm_world,
    savedata=not in_github_actions,
    file_name=filepath,
)


def compute_rotational_profile(pitch, sgn, s_profile, comm):
    poinc = PassingPoincare(
        field_p,
        np.abs(pitch),
        sgn,
        mass,
        charge,
        Ekin,
        ns_poinc=resolution * 2,
        ntheta_poinc=1,
        Nmaps=resolution,
        comm=comm,
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
mu_harmonics = np.linspace(0, max_mu, resolution)
perturbed_pitch_angle = []
resonance_loc = []

for plot_counter, mu_h in enumerate(mu_harmonics):
    # compute rotational profile for given pitch angle
    profile = compute_rotational_profile(mu_h / Ekin, sign_vpar, False, comm=comm_world)
    if profile.shape[0] < 2:
        continue  # skip if not enough points to compute resonance:
    pitch_angle_h = (sign_vpar * np.abs(mu_h) / Ekin) * min_volmodB
    perturbed_pitch_angle.append(pitch_angle_h)
    drift_helicity = profile[:, 3]
    radial_position = profile[:, 0]

    if verbose and (
        plot_counter % 5 == 0
    ):  # only plot some pitch angles to avoid overcrowding
        plt.plot(
            radial_position,
            drift_helicity,
            label=f"$\\lambda$={pitch_angle_h:.2f}",
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
                        harmonics[h][ell][crossing_index][0].append(pitch_angle_h)
                        harmonics[h][ell][crossing_index][1].append(radius)
                    else:
                        harmonics[h][ell] = [[[pitch_angle_h], [radius]]]
    if comm_world is not None:
        comm_world.Barrier()


plt.clf()

lines_modes = []
mpl.rcParams["xtick.labelsize"] = 25
mpl.rcParams["ytick.labelsize"] = 25

# list of harmonics that we calculated resonances for
hlist = list(harmonics.keys())

# create a colormap for the lines based on the harmonic number
harmonic_cmap = mpl.cm.cividis
harmonic_cmap = harmonic_cmap.reversed()
if len(hlist) > 1:
    norm = mpl.colors.Normalize(vmin=hlist[0], vmax=hlist[-1])
else:
    norm = mpl.colors.Normalize(vmin=hlist[0], vmax=hlist[0] + 1)

if comm_world is not None:
    comm_world.Barrier()

if verbose and not in_github_actions:
    # make a list of line colors for each harmonic
    linecolors = [harmonic_cmap(norm(h)) for h in lines_modes]

    fig, (ax, ax_dummy) = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [4, 0.61]}, figsize=(17, 12)
    )

    # plot heatmap
    ax = heat_map.plot_heatmap(
        ax=ax,
        savepath=filepath + "heatmap.png",
        plot_losses=plot_losses,
        DA_at_loss=plot_losses,
    )

    fig = ax.get_figure()

    lines = []
    labels_lines = []
    labels_text = []
    # iterate through harmonics to plot
    for h in harmonics:
        Phim_h = AE_temp.harmonics[h].m
        Phin_h = AE_temp.harmonics[h].n

        for ell in harmonics[h]:
            for crossing_line_index, crossing_line in enumerate(harmonics[h][ell]):
                resonance_peta = np.asarray(crossing_line[1])
                resonance_pitch = np.asarray(crossing_line[0])

                if len(crossing_line[1]) < 3:
                    continue  # skip if not enough points to fit a curve:

                # smooth resonance lines with a polynomial fit
                coeffs = np.polyfit(resonance_pitch, resonance_peta, 1)
                poly = np.poly1d(coeffs)
                pa_fit = np.linspace(
                    min(resonance_pitch), max(resonance_pitch), resolution * 2
                )
                s_fit = poly(pa_fit)

                color = harmonic_cmap(norm(h))

                # fit a curve to the resonance points to plot on the heatmap
                trapped_passing_fit = heat_map.trapped_boundary_fit(pa_fit)
                trapped_passing_line_rad = heat_map.trapped_boundary_fit_radial
                trapped_passing_line_pitch = heat_map.trapped_boundary_fit_pitch

                # ignore resonance lines which start near the trapped-passing boundary
                # in the region where the fit is inaccurate due to numerical noise
                if trapped_passing_line_pitch[0] > resonance_pitch[0]:
                    continue

                diff = trapped_passing_fit - s_fit
                sign_changes = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]

                # don't repeat label if resonance line crosses multiple times
                if crossing_line_index == 0:
                    label = rf"m,n={Phim_h},{Phin_h} $\ell$={ell}"
                else:
                    label = None

                stop_index = len(pa_fit) if len(sign_changes) == 0 else sign_changes[0]

                if stop_index == 0:
                    continue  # skip if resonance line is entirely in trapped region
                (line,) = ax.plot(
                    pa_fit[:stop_index],
                    s_fit[:stop_index],
                    linewidth=5,
                    linestyle=possible_linestyles[ell + max_ell],
                    label=label,
                    color=color,
                )

                if crossing_line_index == 0:
                    lines.append(line)
                if crossing_line_index == 0:
                    label = rf"m,n={Phim_h},{Phin_h} $\ell$={ell}"
                    labels_lines.append(line)
                    labels_text.append(label)
    ax_dummy.remove()
    ax.legend(
        labels_lines,
        labels_text,
        loc="upper left",
        bbox_to_anchor=(1.2, 1.0),
        borderaxespad=0.0,
    )
    fig.savefig(filepath + f"{filepath[:-1]}.png", dpi=400)

    plt.clf()

if comm_world is not None:
    comm_world.Barrier()
