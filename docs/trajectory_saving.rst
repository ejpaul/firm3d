.. _trajectory_saving:

Trajectory Saving
=================

There are two ways the trajectory information can be saved: by recording "hits" of user-defined coordinate planes (e.g., Poincaré sections), or by recording uniform time intervals of the trajectory. The routines ``trace_particles_boozer`` or ``trace_particles_boozer_perturbed`` return this information in the tuple ``(res_tys,res_hits)``.

Trajectory Data (res_tys)
-------------------------

If ``forget_exact_path=False``, the parameter ``dt_save`` determines the time interval for trajectory saving. (Note that if this parameter is made too small, one may run into memory issues.) This trajectory information is returned in ``res_tys``, which is a list (length = number of particles) of numpy arrays with shape ``(nsave,5)``. Here ``nsave`` is the number of timesteps saved. Each row contains the time and the state, ``[t, s, theta, zeta, v_par]``. If ``forget_exact_path=True``, only the state at the initial and final time will be returned.

.. code-block:: python

   # Trace with trajectory saving
   res_tys, res_hits = trace_particles_boozer(
       field=field,
       stz_inits=points,
       parallel_speeds=vpars,
       tmax=1e-3,
       dt_save=1e-6,
       forget_exact_path=False
   )

   # Access trajectory data
   trajectory = res_tys[0]  # First particle
   times = trajectory[:, 0]
   s_values = trajectory[:, 1]
   theta_values = trajectory[:, 2]
   zeta_values = trajectory[:, 3]
   vpar_values = trajectory[:, 4]

   print(f"Saved {len(times)} trajectory points")
   print(f"Time range: {times[0]:.3f} to {times[-1]:.3f}")

Hits Data (res_hits)
--------------------

The "hits" are defined through the input lists ``phases``, ``n_zetas``, ``m_thetas``, ``omegas``, and ``vpars``.

**Hit Planes:**

- **vpars**: If specified, the trajectory will be recorded when the parallel velocity hits a given value. For example, the Poincaré map for trapped particles is defined by recording the points with :math:`v_{||} = 0`.

- **phases**, **n_zetas**, **m_zetas** and **omegas**: If ``phases`` is specified, the trajectory will be recorded when :math:`n_\zeta * \zeta + m_\theta * theta - \omega t` hits the values given in the ``phases`` array, with the frequency :math:`\omega` given by the ``omegas`` array. All lists must have the same length.


**Hit Data Structure:**

The hits are returned in ``res_hits``, which is a list (length = number of particles) of numpy arrays with shape ``(nhits,6)``, where ``nhits`` is the number of hits of a coordinate plane or stopping criteria. Each row of the array contains ``[time] + [idx] + state]``, where ``idx`` tells us which of the hit planes or stopping criteria was hit:

- If ``idx >= 0`` and ``idx < len(zetas)``: the ``zetas[idx]`` plane was hit
- If ``len(zetas) <= idx < len(zetas) + len(vpars)``: the ``vpars[idx-len(zetas)]`` plane was hit
- If ``len(zetas) + len(vpars) <= idx < len(zetas) + len(vpars) + len(thetas)``: the ``thetas[idx-len(zetas)-len(vpars)]`` plane was hit
- If ``idx < 0``: ``stopping_criteria[int(-idx)-1]`` was hit

The state vector is ``[t, s, theta, zeta, v_par]``.

Multiple Hit Planes
-------------------

For custom analysis beyond the standard Poincaré maps, you can specify multiple hit planes manually:

.. code-block:: python

   # Record hits at multiple toroidal angles
   zetas = [0.0, np.pi/2, np.pi, 3*np.pi/2]
   omega_zetas = [0.0, 0.0, 0.0, 0.0]

   # Record hits at multiple poloidal angles
   thetas = [0.0, np.pi/2]
   omega_thetas = [0.0, 0.0]

   # Record hits at multiple parallel velocities
   vpars = [0.0, 0.5, -0.5]

   res_tys, res_hits = trace_particles_boozer(
       field=field,
       stz_inits=points,
       parallel_speeds=vpars,
       tmax=1e-3,
       phases=phases,
       n_zetas=n_zetas,
       m_thetas=m_thetas,
       omegas=omegas,
       vpars=vpars
   )

   # Analyze hits
   hits = res_hits[0]
   for hit in hits:
       time, idx, s, theta, zeta, vpar = hit
       if idx < len(zetas):
           print(f"Hit zeta plane {idx} at t={time:.3f}")
       elif idx < len(zetas) + len(vpars):
           vpar_idx = idx - len(zetas)
           print(f"Hit v_parallel plane {vpar_idx} at t={time:.3f}")
       elif idx < len(zetas) + len(vpars) + len(thetas):
           theta_idx = idx - len(zetas) - len(vpars)
           print(f"Hit theta plane {theta_idx} at t={time:.3f}")
       else:
           print(f"Hit stopping criterion at t={time:.3f}")

Memory Management
-----------------

For long integrations, memory usage can become an issue. Use ``forget_exact_path=True`` to save only initial and final states:

.. code-block:: python

   # Save only initial and final states to save memory
   res_tys, res_hits = trace_particles_boozer(
       field=field,
       stz_inits=points,
       parallel_speeds=vpars,
       tmax=1e-3,
       forget_exact_path=True,  # Only save initial/final states
       vpars=[0.0]  # Still record hits
   )

   # res_tys will contain only 2 points per particle
   trajectory = res_tys[0]
   print(f"Saved {len(trajectory)} points (initial and final only)")
