import simsoptpp as sopp
from simsopt.field import (BoozerRadialInterpolant, InterpolatedBoozerField, trace_particles,
                           MinToroidalFluxStoppingCriterion, MaxToroidalFluxStoppingCriterion,
                           ToroidalTransitStoppingCriterion, IterationStoppingCriterion, compute_resonances)
from simsopt.mhd import Vmec
import numpy as np
from simsopt.util.constants import (
        ALPHA_PARTICLE_MASS as MASS,
        FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
        ALPHA_PARTICLE_CHARGE as CHARGE
        )
import os

from simsopt.configs import get_ncsx_data
from simsopt.field import BiotSavart, coils_via_symmetries, InterpolatedField
from simsopt.geo import SurfaceRZFourier, SurfaceClassifier
from firm3d.util.gpu_utils import cartesian_interpolant  
import firm3dpp

np.random.seed(1800)


def cartesian_rhs(position, vpar, field, mass, charge, velocity):
        field.set_points_cyl(position.reshape(-1, 3))
        B = field.B()
        # print(B)
        GradAbsB = field.GradAbsB()

        # print(B)
        AbsB = np.linalg.norm(B)
        # print(AbsB)
        # exit()

        BcrossGradAbsB = [0]*3 
        BcrossGradAbsB[0] = B[0,1]*GradAbsB[0,2] - B[0,2]*GradAbsB[0,1]
        BcrossGradAbsB[1] = B[0,2]*GradAbsB[0,0] - B[0,0]*GradAbsB[0,2]
        BcrossGradAbsB[2] = B[0,0]*GradAbsB[0,1] - B[0,1]*GradAbsB[0,0]

        v_perp2 = velocity**2 - vpar**2
        mu = v_perp2/(2*AbsB)
        fak1 = vpar / AbsB
        fak2 = (mass / (charge * AbsB**3)) * (0.5*v_perp2 + vpar**2)

        out = [0]*4
        for i in range(3):
                out[i] = fak1 * B[0,i] + fak2*BcrossGradAbsB[i]
        out[3] = -mu*np.sum([B[0,i]*GradAbsB[0,i] for i in range(3)]) / AbsB
        return out


def test_derivs(field, sc_particle, nfp, n_metagrid_pts, n_test_pts, verify=True):
        # generate test points
        r_range = field.r_range
        phi_range = field.phi_range
        z_range = field.z_range

        # generate test points
        r = np.random.uniform(low=r_range[0]+0.1, high=r_range[1]-0.1, size=(n_test_pts,1))
        phi = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
        z = np.random.uniform(low=-z_range[1]+0.1, high=z_range[1]-0.1, size=(n_test_pts,1))
        rphiz = np.hstack((r,phi,z))
        # rphiz = np.array([[1.64982937,  4.57359021, -0.05896898]])
        rphiz = np.ascontiguousarray(rphiz)

        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        # vpar_init = np.array([4355542.313737903])

      

        ### NEW INTERPOLANT
        r_range, phi_range, z_range, quad_info = cartesian_interpolant(field, sc_particle, nfp, n_metagrid_pts)

        # psi0 =field.psi0

        # print("calculating new derivatives")
        new_derivs = firm3dpp.test_derivatives_cartesian(quad_info, r_range, phi_range, z_range, rphiz, vpar_init, VELOCITY, MASS, CHARGE, rphiz.shape[0])
        new_derivs = np.reshape(new_derivs, (rphiz.shape[0], 4))

        if verify:
                # print("computing simsopt derivatives")
                old_derivs = np.empty((n_test_pts, 4))
                for i in range(n_test_pts):
                        old_derivs[i,:] = cartesian_rhs(rphiz[i,:], vpar_init[i], field, MASS, CHARGE, VELOCITY)

                dist_fn = sc_particle.evaluate_rphiz(rphiz)[:, 0]
                # print(dist_fn)
                rel_err = np.abs((old_derivs - new_derivs) / old_derivs)
                rel_err = rel_err[dist_fn > 0, :] # only consider particles inside the device
                diff = np.max(rel_err)
                # print(np.abs(old_derivs - new_derivs) / old_derivs)

                print("Maximum relative error in derivative values on {} points: {}".format(rel_err.shape[0], diff))
                if diff > 1e-7:
                        print("CARTESIAN DERIVS TEST FAILED")
                        print("culprit particle:")
                        row_index = np.argmax(rel_err) // rel_err.shape[1]
                        print(rphiz[row_index, :])
                        print(vpar_init[row_index])
                        print("simsopt", old_derivs[row_index, :])
                        print("new", new_derivs[row_index, :])
                        print(rel_err[row_index, :])
                else:
                        print("CARTESIAN DERIVS TEST SUCCESS")





def test_timestep(field, sc_particle, nfp, n_metagrid_pts, n_test_pts, verify=True):

        # generate test points
        r_range = field.r_range
        phi_range = field.phi_range
        z_range = field.z_range

        # generate test points
        r = np.random.uniform(low=r_range[0]+0.1, high=r_range[1]-0.1, size=(n_test_pts,1))
        phi = np.random.uniform(low=phi_range[0], high=phi_range[1], size=(n_test_pts,1))
        z = np.random.uniform(low=z_range[0]+0.1, high=z_range[1]-0.1, size=(n_test_pts,1))
        rphiz = np.hstack((r,phi,z))
        rphiz = np.ascontiguousarray(rphiz)

        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        vpar_init = np.ascontiguousarray(vpar_init)



        ### NEW INTERPOLANT
        # print("setting up new interpolant")
        r_range, phi_range, z_range, quad_info = cartesian_interpolant(field, sc_particle, nfp, n_metagrid_pts)
        quad_info = np.ascontiguousarray(quad_info)
        # # for i in range(n_test_pts):
        # print(r_range)
        # print(phi_range),
        # print(z_range)
        # rphiz_test = rphiz[i, :]
        # vpar_init_test = vpar_init[i]

        # rphiz = np.array([[1.36825919, 0.61279615, 0.26253024]])
        # vpar_init = [-6203622.275269774]
        # print("rphiz", rphiz)

        # print(i, rphiz_test, vpar_init_test)
        # print(quad_info.shape)
        # print("testing new timstep")
        
        last_time = sopp.test_timestep_cartesian(
                quad_pts=quad_info, 
                srange=r_range,
                trange=phi_range,
                zrange=z_range, 
                stz_init=rphiz,
                m=MASS, 
                q=CHARGE, 
                vtotal=np.sqrt(2*ENERGY/MASS),  
                vtang=vpar_init, 
                tol=1e-9, 
                nparticles=n_test_pts)


        # print("last_time", last_time)

        last_time = np.reshape(last_time, (n_test_pts, 7))

        # print("last_time", last_time)
        # print(last_time[:, 4])

        new_final_positions = np.array([[x[4], x[0], x[1], x[2], x[3]] for x in last_time])

        # print("new final positions", new_final_positions)

        # print("computing simsopt timestep")

        if verify:
                r = rphiz[:, 0].reshape(-1,1)
                phi = rphiz[:, 1].reshape(-1,1)
                z = rphiz[:, 2].reshape(-1,1)
                x = r*np.cos(phi)
                y = r*np.sin(phi)
                # print(x,y,z)
                xyz = np.hstack((x,y,z))
                # print(xyz)
                # print(vpar_init)
                # print(xyz.shape)
                # print(len(vpar_init))
                gc_tys, gc_zeta_hits = trace_particles(
                        field, xyz, vpar_init, tmax=1e-2, mass=MASS, charge=CHARGE,
                        Ekin=ENERGY, tol=1e-9, stopping_criteria=[IterationStoppingCriterion(1)],
                        forget_exact_path=True)
                # print("done with simsopt timestep")
                # print(gc_tys)
                # print(gc_zeta_hits)
                final_positions = np.array([x[-1] for x in gc_tys])
                rel_err = np.abs((final_positions - new_final_positions) / final_positions)
                diff = np.max(rel_err)
                # print(np.abs(final_positions - new_final_positions) / final_positions)

                print("Maximum relative error in final positions on {} points: {}".format(n_test_pts, diff))
                if diff > 1e-6:
                        print("CARTESIAN TIMESTEP TEST FAILURE")
                        print("culprit particle:")
                        row_index = np.argmax(rel_err) // rel_err.shape[1]
                        print(rphiz[row_index, :])
                        print(vpar_init[row_index])
                        print("simsopt", final_positions[row_index, :])
                        print("new", new_final_positions[row_index, :])
                        print(rel_err[row_index, :])
                else:
                        print("CARTESIAN TIMESTEP TEST SUCCESS")


if __name__ == "__main__":
        ### CREATE A FIELD FOR TRACING
        nfp = 3
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, nfp, True)
        curves = [c.curve for c in coils]
        bs = BiotSavart(coils)
        # proc0_print("Mean(|B|) on axis =", np.mean(np.linalg.norm(bs.set_points(ma.gamma()).B(), axis=1)))
        # proc0_print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)))

        mpol = 5
        ntor = 5
        stellsym = True
        s = SurfaceRZFourier.from_nphi_ntheta(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
                                                range="full torus", nphi=64, ntheta=24)
        s.fit_to_curve(ma, 0.20, flip_theta=False)

        n_metagrid_pts = 30
        degree = 3
        rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
        zs = s.gamma()[:, :, 2]
        sc_particle = SurfaceClassifier(s, h=0.1, p=2)


        rrange = (np.min(rs), np.max(rs), n_metagrid_pts)
        phirange = (0, 2*np.pi/nfp, n_metagrid_pts)
        # exploit stellarator symmetry and only consider positive z values:
        zrange = (0, np.max(zs), n_metagrid_pts)
        print(rrange, phirange, zrange)
        bsh = InterpolatedField(
                bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True
        )
        np.random.seed(1800)
        # test_interpolant_bfield(bsh, sc_particle, nfp, n_metagrid_pts, 100000)

        test_derivs(bsh, sc_particle, nfp, n_metagrid_pts, 100000)

        # test_timestep(bsh, sc_particle, nfp, n_metagrid_pts, 100000)