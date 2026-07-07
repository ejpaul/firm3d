# Code Review: `trajectory_helpers.py`

## 1. Confirmed Bugs

### 1.1 `MapEquilibrium.__init__` — `self.savepath` used before assignment
**Location:** Lines 2793–2795

```python
if savepath != "":
    savepath += "_"
    self.res_filepaths = {
        "tys": self.savepath + "DA_data.txt",   # BUG: self.savepath not yet set
        "ICs": self.savepath + "initial_conditions.txt"
    }

self.savepath = savepath  # assigned here, too late
```

`self.savepath` is referenced inside the `if` block but is only assigned two lines later.
The result is that if `savepath != ""` this will raise `AttributeError` on first run (no cached `self.savepath`),
or silently use a stale value if the instance has been reused. Replace with the local variable `savepath`:

```python
self.res_filepaths = {
    "tys": savepath + "DA_data.txt",
    "ICs": savepath + "initial_conditions.txt",
}
```

### 1.2 `MapEquilibrium.__init__` — `self.res_filepaths` accessed when never set
**Location:** Lines 2801–2805

```python
if savedata:
    if exists(self.res_filepaths["ICs"]):   # AttributeError if savepath == ""
        load_ics = True
    if exists(self.res_filepaths["tys"]):
        load_files = True
```

`self.res_filepaths` is only created when `savepath != ""`, but the block above runs whenever
`savedata=True` regardless. Passing `savedata=True, savepath=""` will raise `AttributeError`.
`res_filepaths` should be initialised unconditionally (even with empty-string paths) before this block,
or the `exists` checks should be guarded by `hasattr(self, "res_filepaths")`.

### 1.3 `MapEquilibrium.trace_particles` — `start_state` stores the **final** position, not the initial
**Location:** Lines 3011–3018

```python
# start_state = [s, theta, zeta, vpar, p_eta_0, mu]
start_state = [
    points_trajectory[-1, 0],  # last s  ← should be [0, 0]
    points_trajectory[-1, 1],  # last theta ← should be [0, 1]
    points_trajectory[-1, 2],  # last zeta ← should be [0, 2]
    vpar[0],                   # initial vpar (correct)
    Peta_values[0],            # initial peta (correct)
    mu,
]
```

`points_trajectory[-1]` is the final (possibly wall-clipped) position, not the starting position.
This means the heat map `radial_coordinate_start` axis plots the final s, not the initial s — mixing
initial and final phase-space coordinates. Fix: use `points_trajectory[0]`.

### 1.4 `WBAParticles.trace_particles` — DA computed from **zeta**, not **time**
**Location:** Line 5623

```python
points_trajectory = np.column_stack((s_path, theta_path, zeta_path))  # shape (N, 3)
...
stack_data = np.column_stack((points_trajectory[:, -1], Peta_values))  # col -1 = zeta!
time_eval, DA_eval = return_DA(stack_data)
```

After the reassignment, `points_trajectory[:, -1]` is `zeta_path`, not `time_momentum`.
`return_DA` expects `(time, momentum)` pairs; feeding it zeta instead of time produces
incorrect WBA digit-accuracy values for all `WBAParticles` instances.

Compare with `WBAPerturbedParticles.trace_particles` (line ~5067) where `points_trajectory`
retains the time column and `points_trajectory[:, -1]` is correctly the time. Fix:

```python
stack_data = np.column_stack((time_momentum, Peta_values))
```

### 1.5 `PassingPerturbedPoincare.__init__` — `raise Warning(...)` instead of `warnings.warn(...)`
**Location:** Lines 1822–1825

```python
if not isinstance(saw, ShearAlfvenHarmonic):
    dominant_saw = saw[0]
    raise Warning(
        "Expected saw to be an instance of ShearAlfvenHarmonic - ..."
    )
```

`raise Warning(...)` raises a `Warning` as an exception, which terminates the constructor
before any of the subsequent setup runs. The intent is clearly to emit a soft warning while
continuing. The code after this line (setting `dominant_saw`, computing `nprime`, etc.) is
unreachable when `saw` is a `ShearAlfvenWavesSuperposition`. Fix:

```python
import warnings
if not isinstance(saw, ShearAlfvenHarmonic):
    dominant_saw = saw[0]
    warnings.warn(
        "Expected saw to be an instance of ShearAlfvenHarmonic - "
        "Perturbed Energy Invariant may not be valid."
    )
else:
    dominant_saw = None
```

### 1.6 `PassingPerturbedPoincare` — `WBA_transit_steps` starts at `[0]` when chaos detection is off
**Location:** Lines 1852–1854

```python
else:
    self.nconvergence_points = 1
    self.WBA_transit_steps = [0]   # transit 0, i.e. before any map is computed
```

Compare `PassingPoincare.__init__` (line 276–278) and `TrappedPoincare.__init__` (line 934–936):
both correctly use `[Nmaps - 1]` (the last transit) when `chaos_detection=False`. Using `[0]`
here means the WBA DA is evaluated after zero transits for all non-chaos-detection runs.
Fix: `self.WBA_transit_steps = [Nmaps - 1]`.

---

## 2. Performance Issues

### 2.1 `min_volumemodB` — 3D meshgrid at `resolution=1000` creates 10⁹ points
**Location:** Lines 116–130

```python
resolution = 1000
s_grid     = np.linspace(0, 1, resolution)           # 1 000 values
theta_grid = np.linspace(0, 2π, resolution, ...)     # 1 000 values
zeta_grid  = np.linspace(0, 2π/NFP, resolution, ...) # 1 000 values
[zeta_grid, theta_grid, s_grid] = np.meshgrid(...)   # 1 000³ = 10⁹ points
points = np.zeros((len(theta_grid.flatten()), 3))    # 24 GB array
```

This will crash with an `MemoryError` on any practical machine. The intent is to estimate
the volume minimum of `|B|`, so a coarse grid of ~30 per dimension (27 000 points) is
more than sufficient and runs in milliseconds. Either reduce `resolution` to ~30 before
meshing, or use a single 1D random sample:

```python
resolution = 10  # per-dimension; 10³ = 1 000 field evaluations
```

---

## 3. Code Redundancies / Structural Duplication

### 3.1 `chi`, `eta`, `chi_eta_to_theta_zeta` are defined identically in three classes

The following three methods have byte-for-byte identical bodies in
`TrappedPoincare`, `PassingPerturbedPoincare`, and `MapPhaseSpace`:

```python
def chi(self, theta, zeta):
    return self.helicity_M * theta - self.helicity_N * zeta

def eta(self, theta, zeta):
    return self.helicity_Mp * theta - self.helicity_Np * zeta

def chi_eta_to_theta_zeta(self, chi, eta):
    denom = self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
    theta = (self.helicity_Np * chi - self.helicity_N * eta) / denom
    zeta  = (self.helicity_Mp * chi - self.helicity_M * eta) / denom
    return theta, zeta
```

These should be factored into a `HelicityMixin` (or standalone module-level functions)
and inherited/imported wherever needed.

### 3.2 `check_filepaths` is duplicated identically in four classes

`MapEquilibrium`, `MapPhaseSpace`, `WBAPerturbedParticles`, and `WBAParticles` all contain:

```python
def check_filepaths(self, filepaths):
    return all(exists(fp) for fp in filepaths.values())
```

This is a pure utility with no `self` dependency and should be a single module-level function.

### 3.3 WBA chaos-detection initialisation block is copy-pasted five times

The following pattern appears with near-identical logic in `PassingPoincare.__init__`,
`TrappedPoincare.__init__`, `PassingPerturbedPoincare.__init__`,
`MapEquilibrium.__init__`, and `MapPhaseSpace.__init__`:

```python
if chaos_detection:
    if nconvergence_points is None:
        self.nconvergence_points = 1
        self.WBA_transit_steps = [Nmaps - 1]
    else:
        self.nconvergence_points = nconvergence_points
        transits_per_average = int(Nmaps / nconvergence_points)
        self.WBA_transit_steps = np.linspace(
            transits_per_average, Nmaps - 1, num=nconvergence_points, dtype=int
        ).tolist()
else:
    self.nconvergence_points = 1
    self.WBA_transit_steps = [Nmaps - 1]
```

Extract into a helper `_init_wba_transit_steps(self, Nmaps, nconvergence_points, chaos_detection)`.

### 3.4 `build_lists` / `trace_particles` structural duplication across classes

`MapEquilibrium`, `MapPhaseSpace`, `WBAPerturbedParticles`, and `WBAParticles`
all implement near-identical `build_lists` and `trace_particles` methods that differ
mainly in the start/end state vector lengths and which diagnostics are stored.
Consider a shared base class or dataclass-based state representation.

### 3.5 `compute_Eprime` — `nprime` parameter is unconditionally overwritten internally

**Location:** Lines 1561, 1618

```python
def compute_Eprime(saw, points, vpar, mu, mass, charge, helicity_M, helicity_N, nprime=None):
    ...
    nprime = (Phim * helicity_N - Phin * helicity_M) / (...)  # overwrites parameter
```

The `nprime=None` parameter in the signature is never used and is silently discarded.
Either remove it from the signature, or use it as an override when provided.

---

## 4. Physics Concerns

### 4.1 `TrappedPoincare.initialize_trapped_map` — mirror-point bracket assumes `chi ∈ [0, π]`
**Location:** Line 1155

```python
sol = root_scalar(diffmodB, ..., method="toms748", bracket=[0, np.pi])
```

`toms748` requires a bracket `[a, b]` where `diffmodB(a)` and `diffmodB(b)` have opposite signs.
This is only guaranteed if the field-strength maximum in the chi direction lies in `[0, π]`.
For a general non-quasisymmetric field — or for ripple wells shifted away from the outboard
midplane — this bracket will fail. The fallback initial guess `self.chi_mirror = π/2` has
the same assumption. A more robust approach would scan chi to find a sign change before bracketing,
or try multiple brackets (e.g., `[0, π]`, `[π, 2π]`).

### 4.2 `TrappedPoincare.compute_frequencies` — unconditional dict access raises `KeyError`
**Location:** Line 1404

```python
if self.solver_options["axis"] != 0:
```

This raises `KeyError` if the user never set `"axis"` in `solver_options`. Should be:

```python
if self.solver_options.get("axis", 0) != 0:
```

### 4.3 `return_bounces_and_passes` — last toroidal transit is always dropped
**Location:** Lines 3480–3489

```python
for passing_index in range(len(wrap_idx) - 1):   # stops one short
    pass1 = wrap_idx[passing_index]
    pass2 = wrap_idx[passing_index + 1]
    if not np.any((bounce_indices > pass1) & (bounce_indices < pass2)):
        true_passes.append(wrap_idx[passing_index])
```

The loop cannot count the last wrap because it would need a `pass2` to check for
intervening bounces. For long trajectories this is negligible, but for short integration
windows (or resonance-width scans where the number of passes matters) it consistently
under-counts by one. One fix: append the last index unconditionally if no bounce follows it:

```python
if len(wrap_idx) > 0:
    last = wrap_idx[-1]
    if not np.any(bounce_indices > last):
        true_passes.append(last)
```

### 4.4 `PassingPerturbedPoincare.__init__` — `Eprime` uses unperturbed `Ekin`, not perturbed energy
**Location:** Line 1923

```python
self.Eprime = float(self.nprime * Ekin - self.omega * Peta0_arr.item())
```

`Ekin` here is the unperturbed kinetic energy (no wave potential), so `E = Ekin` ignores
the electrostatic perturbation `charge * Phi` at the reference point `p0`. This is correct
only if the perturbation is initialised at `t=0` with zero phase (`Phi(p0, 0) = 0`).
If the wave has a non-zero phase at the reference time, `Eprime` will be offset. A comment
explaining this assumption would prevent future confusion; alternatively, evaluate the full
perturbed energy:

```python
saw.set_points(p0)
E_perturbed = Ekin + charge * saw.Phi()[0, 0]
self.Eprime = float(self.nprime * E_perturbed - self.omega * Peta0_arr.item())
```

### 4.5 `MapEquilibrium.vpar_func` — redundant `np.maximum` before `np.where`
**Location:** Lines 3181–3185

```python
rhs = 2 * (self.Ekin - mu * modB) / self.mass
vpar = sgn * np.sqrt(np.maximum(rhs, 0))  # np.maximum ensures no NaN in sqrt...
return np.where(rhs > 0, vpar, np.nan)    # ...but the result is replaced by NaN anyway
```

`np.maximum(rhs, 0)` prevents a `sqrt` domain warning, but `np.where` then discards those
values in favour of `nan`. This is not a correctness bug, but it is inconsistent with
`MapPhaseSpace.vpar_func` which clips with `np.maximum` and returns `nan` for non-positive
energy. Unifying the two implementations reduces the maintenance surface.

### 4.6 `return_DA` — mixing absolute and relative DA metrics
**Location:** Lines 1743–1746

```python
da_absolute = -np.log10(diff / denom)         # relative to mean value
da_relative = -np.log10(diff / np.nanmax(np.abs(T_mom)))  # relative to amplitude
return T, max(da_relative, da_absolute)
```

Taking `max` of the two metrics returns the **more optimistic** estimate (higher digit
accuracy). For near-zero mean `p_eta` (e.g., on a flux surface near the magnetic axis),
`denom → 0` making `da_absolute` diverge, so the `max` will almost always select
`da_absolute`. This produces artificially high digit-accuracy values near the axis,
potentially masking genuine chaos. Consider using only `da_relative` (or `da_absolute`
with a floor on `denom`) and documenting the choice.

### 4.7 `min_volumemodB` is called during `__init__` for every `MapEquilibrium` and `MapPhaseSpace` instance

`min_volumemodB` triggers a full-volume field evaluation on construction. For expensive
VMEC or Boozer-interpolated fields this can dominate the constructor cost. Consider caching
the result as an attribute of the field object, or accepting it as an optional parameter
(`min_modB=None`) so callers who already know it can skip the scan.

---

## 5. Minor Issues

### 5.1 `trapped_map` error message says "passing_map"
**Location:** Line 1040

```python
raise RuntimeError("Alternative stopping criterion reached in passing_map.")
```

This is inside `TrappedPoincare.trapped_map`. The message should read `"in trapped_map"`.

### 5.2 `WBAParticles.trace_particles` — recomputes `self.vtotal` inside loop
**Location:** Line 5383 (inside loop)

```python
self.vtotal = np.sqrt(2 * self.Ekin / self.mass)
```

This was already computed identically in `__init__` (line 5279). Move it outside the loop
or remove the redundant assignment.

### 5.3 `WBAParticles.__init__` docstring still mentions `DA_cutoff`
**Location:** Line 5433

```
DA_cutoff : Digit accuracy threshold for classifying chaos.
```

`DA_cutoff` does not appear in the function signature. Remove from the docstring.

### 5.4 `PassingPerturbedPoincare.compute_frequencies` — `s_profile` input not validated
`PassingPerturbedPoincare` does not implement `compute_frequencies`, while `PassingPoincare`
and `TrappedPoincare` do. If a user accidentally calls `compute_frequencies` on a
`PassingPerturbedPoincare` instance they will get an `AttributeError`. A `NotImplementedError`
with a helpful message would be clearer.

### 5.5 Inconsistent spelling of `indicies` vs `indices`
`passing_indicies`, `convergence_test_indicies` — should be `indices` throughout.

---

## Summary Table

| # | Class / Function | Type | Severity |
|---|---|---|---|
| 1.1 | `MapEquilibrium.__init__` | Bug — `AttributeError` on `self.savepath` | High |
| 1.2 | `MapEquilibrium.__init__` | Bug — `AttributeError` on `res_filepaths` when `savepath=""` | High |
| 1.3 | `MapEquilibrium.trace_particles` | Bug — final position stored as start state | High |
| 1.4 | `WBAParticles.trace_particles` | Bug — zeta passed to `return_DA` instead of time | High |
| 1.5 | `PassingPerturbedPoincare.__init__` | Bug — `raise Warning` terminates constructor | High |
| 1.6 | `PassingPerturbedPoincare.__init__` | Bug — WBA evaluated at transit 0, not final | Medium |
| 2.1 | `min_volumemodB` | Performance — 10⁹-point meshgrid crashes memory | Critical |
| 3.1 | `TrappedPoincare`, `PassingPerturbedPoincare`, `MapPhaseSpace` | Duplication — 3× identical helicity methods | Low |
| 3.2 | Four classes | Duplication — `check_filepaths` | Low |
| 3.3 | Five `__init__` methods | Duplication — WBA transit step initialisation | Low |
| 3.4 | Four classes | Duplication — `build_lists` / `trace_particles` structure | Medium |
| 3.5 | `compute_Eprime` | Redundancy — unused `nprime` parameter | Low |
| 4.1 | `TrappedPoincare.initialize_trapped_map` | Physics — bracket assumes `chi ∈ [0,π]` | Medium |
| 4.2 | `TrappedPoincare.compute_frequencies` | Bug — `KeyError` on missing `"axis"` key | Medium |
| 4.3 | `return_bounces_and_passes` | Physics — last transit always undercounted | Low |
| 4.4 | `PassingPerturbedPoincare.__init__` | Physics — Eprime ignores wave potential at `p0` | Medium |
| 4.5 | `MapEquilibrium.vpar_func` | Style — redundant `np.maximum` before `np.where` | Low |
| 4.6 | `return_DA` | Physics — `max(da_relative, da_absolute)` inflates DA near axis | Medium |
| 4.7 | `MapEquilibrium`, `MapPhaseSpace` | Performance — `min_volumemodB` on every construction | Low |
| 5.1 | `TrappedPoincare.trapped_map` | Cosmetic — wrong name in error message | Low |
| 5.2 | `WBAParticles.trace_particles` | Redundancy — `self.vtotal` recomputed in loop | Low |
| 5.3 | `WBAParticles.__init__` | Docstring — phantom `DA_cutoff` parameter | Low |
| 5.4 | `PassingPerturbedPoincare` | Usability — missing `compute_frequencies` raises `AttributeError` | Low |
| 5.5 | Multiple | Style — `indicies` misspelling throughout | Low |