# PIPELINE.md

**The source of truth for how data flows through this project.**

This document exists so that anyone — you in six months, a collaborator, or a fresh
Claude Code session with no memory of how the system was built — can understand and
operate every stage without narration. For each stage it states: what it **consumes**,
what it **produces**, the **format/schema** of the hand-off, where the data **lives**,
and the **command** to run it.

> If you change what a stage produces or consumes, update this file in the same commit.
> A drifted PIPELINE.md is worse than none.

---

## The pipeline at a glance

```
[1] Geometry            analytic shape + surface-station set (id, location, normal, curvature)
        │                        │
        │                        ├────────────────────────────┐
        ▼                        ▼                            ▼
[2] Sampling regions     [4a] AERO-F mesh              (stations are the shared
    (per station,              (hyperbolic extrusion)    registration layer:
     wall-normal-thin)          │                        both solvers report to
        │                       ▼                         the same station ids)
        ▼                  [4b] AERO-F run (laminar no-slip NS, steady)
[3] SPARTA setup + runs        │
    (custom interface,         ▼
     union of regions)    [4c] Base-state extraction @ wall
        │                      + a-posteriori 1st-order Maxwell slip
        ▼                      + near-wall gradients
   3D VDF data per station     │
        │                      │
        └───────────┬──────────┘
                    ▼
[5] Moments: fit Hermite expansion about the base Maxwellian;
    compute low-order moments via hand-derived functions
                    │
                    ▼
[6] Reconstruction: moments + run info (density/mass) -> dimensional
    surface stress tensor M_ij and heat flux Q_i
                    │
                    ▼
        (surface fluxes = the deliverable / FST boundary data)

  ---- planned / downstream ----
[7] Feature assembly  ->  [8] GP training (foundation + fine-tune)  ->  [9] coupling
```

The core, working today (in some form, scattered): **[1]–[6]**.
Planned/on the horizon: **[7]–[9]**.

---

## Key modeling decisions that constrain this pipeline

These are summarized here because they determine what each stage is *allowed* to do.
The full record and rationale live in `docs/decisions/`.

- **Base predictor = laminar, no-slip Navier–Stokes (AERO-F) + a-posteriori
  first-order Maxwell slip.** No turbulence model (Re ~ 10³ worst case → laminar).
  No slip BC inside AERO-F; slip velocity and temperature jump are applied as a
  **post-processing step** on the converged wall gradients. The base Maxwellian is
  read **at the wall** (non-degenerate because of the applied slip).
- **Single-Maxwellian base is deliberate**, not a simplification: it maps exactly to
  what a continuum solver produces (one ρ, u, T). The GP learns only the correction.
- **Shared surface stations are the registration layer.** Geometry, AERO-F, and
  SPARTA all key to the same station ids defined on one analytic geometry. This is
  what makes the DSMC-VDF ↔ base-state pairing one-to-one.
- **DSMC sampling regions are wall-normal-thin, tangentially-elongated**, at a frozen
  normal offset; statistics come from **time-averaging the steady ensemble**, not
  from enlarging the region.
- **Only low-order moments are needed** (stress = 2nd, heat flux = 3rd). These are
  exact orthogonal projections onto low-order Hermite coefficients — full-VDF
  convergence is neither claimed nor required.
- **Declared scope is quasi-steady** (fluid relaxes far faster than the structure
  moves/heats). Validation is quasi-steady conjugate aeroheating, not flutter.

---

## Stage specifications

Each stage lists: **Consumes · Produces · Format · Location · Command · Code · Status.**
`FILL IN` marks things only the existing code/cluster knows.

### [1] Geometry & surface stations
- **Consumes:** case config (shape family + parameters).
- **Produces:** the analytic surface, plus a **station manifest** — the ~N surface
  points both solvers will report to.
- **Format:** see *Contract A* below.
- **Location:** currently written to a local dataset folder (registered as
  `shape_dataset_2026_02_06_09_28_35` in `data/registry.yaml`), not committed —
  this is data, not config, so it follows the registry rule like everything else.
- **Command:** no `scripts/` entry point exists yet; the notebook's own driver cells
  (see below) are the only way to run this today. `FILL IN`
- **Code:** `src/rarefied/geometry/` — **promoted** (2026-09-12 cleanup):
  `mesh.py` (mesh → SPARTA-format conversion), `sparta_shapes.py` (SPARTA
  shape-file reader), `regions.py` (circle-union region/curvature/AoA computation
  — see the terminology caveat under Contract A), `plotting.py` (visualization
  helpers `generate_circle_shape` depends on directly), `circle.py`
  (`generate_circle_shape`, the analytic-circle generator). Plotting-only
  functions with no load-bearing caller remain in
  `src/rarefied/geometry/process_shape_data.ipynb` per the README's
  define-vs-inspect rule.
- **Status:** promoted from `process_shape_data.ipynb`; relocation verified against
  `tests/characterization/geometry_reference.npz` (bit-for-bit on a small local
  input).

**Added 2026-09-12 — `src/rarefied/geometry/curves.py`,** the curve front-end
for the AERO-F side. A curve is an `(N, 2)` closed loop, **clockwise** (M13),
recentred on its area centroid, resampled to `n_surf` points equispaced in arc
length, with index 0 anchored where the +x ray from the centroid crosses the
surface. Sources: `circle(diameter, n_pts)` (analytic, exact — the validation
target) and `from_shape(shape)` (the upstream Bezier generator, loaded by path
via `aerof/shapes_bridge.py` so `$SHAPES_DIR` stays a pristine clone with its
commit hash recorded per run). `validate()` is the cheap first gate — it runs
before gmsh is ever invoked and rejects self-intersections, near-duplicate
points, wrong orientation, and ambiguous station origins.

**Stations (M6).** `aerof/geo.stations(curve, n_station)` returns indices into
the wall curve, so every station id is an **exact mesh node** — stage [4c] can
read the base state at a station with no interpolation, because
`Transfinite Curve {…} = 2` pins the mesh wall nodes to exactly the curve
points (verified to 7.85e-17 on the circle).
`n_surf` (mesh resolution, M4) and `n_station` (registration count, M6) are
**independent**: an earlier version derived the wall cell size as
`perimeter/n_surf`, which coupled them, and that is now a documented fallback
only. The on-disk station **manifest format is still open** — only indices are
recorded today (in `<case>.record.json` under `stations`).

**Not yet promoted:** `generate_circle_shape` (SPARTA side) and `curves.circle`
(AERO-F side) are two analytic circle generators. Both are now clockwise and
both start at `(+r, 0)`, so they agree — but they have not been unified, and
whether they should be is open.

### [2] Sampling-region generation (SPARTA side)
- **Consumes:** station manifest [1]; sampling params (normal offset, band size).
- **Produces:** per-station sampling-region definitions (wall-normal-thin,
  tangentially-elongated) + the union set for successive collection runs.
- **Format:** the custom-interface region format. See *Contract B*. `FILL IN`
- **Location:** written to the run dir on the cluster. `FILL IN`
- **Command:** `python scripts/gen_regions.py --case <case.yaml>`  `FILL IN`
- **Code:** `src/rarefied/sparta/`
- **Status:** **CONFIRM, do not assume:** the "regions" promoted to
  `src/rarefied/geometry/regions.py` in stage [1] (circle-union per-point regions
  used for curvature/AoA/collection-area estimation) were found during the
  2026-09-12 cleanup and are **not confirmed to be the same thing** as this stage's
  wall-normal-thin, tangentially-elongated DSMC sampling regions (decision M6). The
  naming overlap ("region") looks coincidental rather than the same concept wearing
  two hats — no code implementing M6's specific band geometry was found anywhere in
  this subtree. `src/rarefied/sparta/` currently holds only the C++ custom interface
  (documented in `src/rarefied/sparta/README.md`, 2026-09-12) and no Python
  region-generation wrapper. Interface documented; this stage's actual Python
  implementation is still unlocated/unwritten.

### [3] SPARTA setup & runs
- **Consumes:** region definitions [2]; the **custom config file**; freestream +
  gas + wall params from the case config.
- **Produces:** **3D VDF data per station** (velocity-space histograms), time-averaged
  over the steady ensemble.
- **Format:** currently **pickle**. See *Contract C* — now fully documented below.
- **Location:** cluster run dirs (`FILL IN` — no cluster access from this cleanup
  session); archived pickles also stored locally at
  `~/Documents/Meteor_Modelling/code/`, registered in `data/registry.yaml`
  (`kn_eq_*_data` entries, 2026-09-12).
- **Command:** custom interface + SPARTA launch. `FILL IN exact invocation`
- **Code:** the custom interface is `src/rarefied/sparta/SpartaDistributionGenerator/`
  (this subtree) / `SpartaAeroInterface/` (parent repo) — named and documented in
  `src/rarefied/sparta/README.md` (2026-09-12). The loader is
  `src/rarefied/io/histograms.py` (`read_all_timesteps_histogram_3D`/`_2D`) +
  `src/rarefied/io/sparta_runs.py` (region metadata, ITEM tables, full-run cache),
  promoted from the moments notebook.
- **Status:** **open scope question, flagged not resolved (2026-09-12):** the two
  C++ copies have diverged into different tools (see `src/rarefied/sparta/README.md`)
  — evidence suggests the parent repo's single-run tool, not this subtree's batch
  tool, produced the argon-cylinder pickles this project's own characterization
  reference depends on. That tool lives outside this subtree's declared scope
  boundary (see this repo's `README.md`). Not resolved; do not assume either tool
  supersedes the other.

### [4] AERO-F base predictor
Sub-stages share one solver run.

- **[4a] Mesh — implemented 2026-09-12.** Graded unstructured **gmsh** mesh
  (M4 as amended; hyperbolic extrusion is no longer required — see the M4
  Amendment for why orthogonality is irrelevant to the consumed product).
  A `.geo` is generated from the curve and meshed by the **gmsh CLI as a
  subprocess** (`aerof/gmsh_env.sh`), never by importing gmsh — the gmsh module
  forces `gcc/10.1.0`, which breaks the py312 numeric stack.
  Two hard format constraints, both set by `gmsh2top`:
  **MSH 2.2 only** (it cannot read 4.1, and its version-4 branch targets 4.0)
  and **no `Recombine`** (it accepts only element types 2 and 4, so the
  extrusion must yield tetrahedra).
  Wall spacing is `size_min_factor` — per M4 from a wall-gradient convergence
  study, **not yet run**, so cases carry `FILL_IN` and the code falls back to
  `perimeter/n_surf`. Far-field is a fixed `20 D` stand-in for M4's
  shock-standoff rule.
  - **Code:** `src/rarefied/aerof/{geo,mshio,quality,topfile,partition,pipeline}.py`
  - **Command:** `python3 scripts/gen_mesh.py --case config/cases/circle_1m.yaml`
    (batch: `scripts/gen_mesh_batch.py`, `scripts/gen_mesh_batch.sbatch`)
  - **Produces:** `.geo`, `.msh`, `.top`, `.top.dec.N`, sower's
    `.msh1/.dec1/.con/.Ncpu`, plus `.png`/`.vtu`/`.quality.txt`/`.record.json`
  - **Gate:** `aerof/quality.py` checks group areas against analytic values,
    total volume, **zero non-positive tet volumes**, wall node count
    `= 2 x n_surf`, wall nodes lying on the input curve, and exactly two
    extrusion planes. The area checks do double duty: they are the only thing
    verifying gmsh's `Extrude` lateral-surface ordering, so a change there
    fails loudly instead of silently mislabelling boundary conditions.

- **[4b] Run:** laminar, no-slip, compressible steady NS.
  - **Command:** `scripts/prep_aerof_case.sh` → `scripts/run_aerof.sh` →
    `scripts/postpro_aerof.sh`
  - **Status:** the chain is **verified end to end** on the reference circle
    (see `docs/handoffs/aerof_mesh_and_case2.md`), but with the *tutorial's*
    settings — air, turbulent closure, wall functions. That is a smoke test,
    not this project's physics.
  - **Open (user-owned):** solver settings — flux, CFL law, limiter,
    reconstruction, supersonic startup. Not a pipeline concern; see the M4
    Amendment. AERO-F's flux options are `Roe`, `VanLeer`, `HLLE`, `HLLC`,
    `RotatedRiemann` (`IoDataCore.C:3313-3315`); there is no AUSM.
  - **Note:** `StickMoving` is an *adiabatic* wall (`BcDef.h:23`), so wall heat
    flux is **zero by definition** unless a `SurfaceData` block promotes it to
    isothermal (`SubDomainCore.C:2914`). That promotion requires the physical
    group name to carry a trailing `_N` surface id, which `aerof/geo.py`
    emits for exactly this reason.

- **[4c] Base-state extraction — not yet implemented.** At each station, read
  (ρ, u, T) at the frozen wall reference; apply a-posteriori first-order
  Maxwell slip from the near-wall gradients (M3); emit the gradients as
  features. Mechanism is now pinned down — see *Contract D*.

- **Consumes:** analytic geometry + stations [1]; case config.
- **Produces:** per-station **base state** + slip-corrected wall state + gradients.
- **Location:** cluster, under `$SCRATCH`; registered in `data/registry.yaml`.

#### Capability tiers (documented, not enforced — M12)

| tier | modules | runs |
|---|---|---|
| pure Python | `geometry/curves`, `aerof/mshio`, `aerof/quality` | anywhere |
| needs gmsh | `aerof/geo` | cluster (or a local gmsh) |
| needs the group binaries | `aerof/topfile` (gmsh2top), `aerof/partition` (mpmetis, sower), [4b] (cd2tet, aerof2) | **cluster only** |

`gmsh2top`, `sower`, `mpmetis`, `cd2tet` and `aerof2` live in
`/home/groups/cfarhat/bin` and have no local equivalent, so everything from the
`.top` conversion onward is cluster-only regardless of Python environment. The
pure-Python tier imports none of them on purpose, which is what keeps curve
generation, mesh-quality metrics and the Contract D reconstruction locally
testable.

### [5] Moments
- **Consumes:** 3D VDF data [3]; base state [4c] (defines the Maxwellian to expand
  about).
- **Produces:** Hermite correction coefficients `a_klm` per station (nondimensional),
  plus the low-order moments (normalized stress, heat flux).
- **Format:** see *Contract E* — now fully documented below.
- **Location:** derived in-memory only today — `calculate_and_compare_moments`
  returns its result dict directly, nothing is cached to disk or registered yet.
  `FILL IN` if/when that changes.
- **Command:** no `scripts/` entry point exists yet. `FILL IN`
- **Code:** `src/rarefied/moments/hermite.py` (the hand-derived fitting functions —
  canonical, relocated verbatim, math untouched) + `pipeline.py`
  (`calculate_and_compare_moments`, the orchestration glue). Promoted from
  `proess_3D_distributions.ipynb` (2026-09-12); relocation verified bit-for-bit
  against `tests/characterization/kn_eq_1p0_reference.npz`.
- **Status:** promoted and verified. **Not the same as PIPELINE.md's originally
  envisioned regression test** — see the *Regression test* section below; the
  characterization reference records current behavior, it does not independently
  validate it. Whether this is "the corrupted-Maxwellian experiment" referenced
  above remains unconfirmed (`docs/CLEANUP_TODO.md` CONFIRM #2).

### [6] Reconstruction → dimensional surface fluxes
- **Consumes:** moments [5]; run info (number density / mass) for redimensionalization.
- **Produces:** dimensional surface **stress tensor M_ij** and **heat flux Q_i**
  (the momentum/energy flux to the surface). Note density enters as an external
  multiplier (see decisions: fluxes are linear in n).
- **Format:** per-station dimensional quantities — see *Contract E* below (fields
  named `pn_model`/`pt_model`/`e_model` etc.).
- **Command:** no `scripts/` entry point exists yet. `FILL IN`
- **Code:** `src/rarefied/reconstruct/flux.py` — promoted verbatim from
  `proess_3D_distributions.ipynb` cell 2 (2026-09-12); relocation verified
  bit-for-bit against `tests/characterization/kn_eq_1p0_reference.npz`.
- **Status:** promoted and verified. **Note a real gap between this doc's stage
  split and the current implementation:** stages [5] and [6] are not two separate
  hand-offs today — `calculate_and_compare_moments` (stage [5]'s code) calls
  `reconstruct/flux.py`'s functions internally and returns the *already-dimensional*
  flux values directly; there is no serialized intermediate "raw Hermite
  coefficients" artifact between the two stages in the live code path. If a clean
  two-stage hand-off is wanted later, that's a real (not yet done) refactor, not
  just a relocation.

### [7]–[9] Planned (features → GP → coupling)
- **[7] Feature assembly:** local geometry (α, φ, normal, κ) + global freestream
  (global Kn, Mach, altitude, T_wall) + base-solver locals (gradient-length-local
  Kn, near-wall ∂T/∂n and ∂u_t/∂n, local pressure/Mach). Admissible iff computable
  from geometry+freestream+base — **never from DSMC**. ARD-prune. Code:
  `src/rarefied/features/`.
- **[8] GP training:** SVGP/NN **foundation** over pooled geometries; exact residual
  GP **fine-tune** per geometry (Kennedy–O'Hagan form). Code: `src/rarefied/models/`.
- **[9] Coupling:** quasi-steady conjugate aeroheating; surrogate supplies steady
  wall flux to FEM. (This is where the surrogate meets the FST problem.)

---

## Interface contracts

These are the hand-off formats that currently live only in your head. Pinning them
down here is the single most valuable thing this file does. **`FILL IN` the exact
fields against the existing code.**

### Contract A — Station manifest ([1] → [2],[4])
**Reconciled against the actual code (2026-09-12), with an important caveat: the
schema below is what `src/rarefied/geometry/regions.py`'s `read_region_metadata`
actually parses today, not a confirmed implementation of the "station manifest"
concept above.** It has no `s` (arc-length) or `phi` (angle-about-centroid) field —
only `Tangent_Angle_Rad`/`AoA_Rad`, which are related but not proven equivalent.
See stage [2]'s status note: whether this "region" concept is the same thing
PIPELINE.md's stage [1]/[2] split imagines is itself unconfirmed.

One record per region, text format (comma-separated after a header block), written
by `process_sparta_folder`/`generate_circle_shape`, read by `read_region_metadata`:
| field | meaning |
|---|---|
| `Region_Index` | integer index (analogous to `station_id`, not confirmed identical in scope) |
| `Num_Points` | number of boundary vertices in this region |
| `BBox_Min_X/Max_X/Min_Y/Max_Y` | bounding box of the region's circle union |
| `Normal_X`, `Normal_Y` | outward unit normal (≈ `nx`, `ny`) |
| `Tangent_X`, `Tangent_Y` | tangent vector |
| `Fitted_Circle_X/Y/Radius` | curvature-fitting circle (`1/Fitted_Circle_Radius` ≈ `kappa`, signed) |
| `Local_AoA_Rad` | local angle of attack (radians) |
| `Union_Area`, `Collection_Area` | circle-union area / area available for data collection |

The in-memory `region_df` (from the moments-side pickle, Contract C) additionally
carries `Tangent_Angle_Rad` (derived: `arctan2(Tangent_Y, Tangent_X)`) and `AoA_Rad`
(aliased from `Local_AoA_Rad`) — added by `rarefied.io.sparta_runs.read_region_metadata`,
a **different function from the one above**, parsing a differently-shaped metadata
file (see `src/rarefied/geometry/regions.py` and `src/rarefied/io/sparta_runs.py`
docstrings for the full explanation of why there are two `read_region_metadata`s).
Storage format: plain text, not YAML/JSON/CSV-with-header — see either module's
source for the exact parse.

### Contract B — Sampling-region definition ([2] → [3])
Per-station region geometry (wall-normal-thin, tangentially-elongated band) + the
union set, in the **custom-interface format**.
<!-- FILL IN: still open. src/rarefied/sparta/README.md (2026-09-12) documents the
     custom interface's *config file* schema (both the batch and single-run
     variants), but no code implementing this specific wall-normal-thin banded
     region format (decision M6) was located anywhere in this subtree during the
     2026-09-12 cleanup -- do not assume Contract A's circle-union regions are it. -->

### Contract C — 3D VDF data ([3] → [5])
**Fully reconciled against the actual pickle (2026-09-12), by direct
`pickle.load` of `kn_eq_1p0_data` (see `data/registry.yaml`).** Each pickle is a
2-tuple `(region_cache, region_df)`:
- `region_cache`: `dict[int, dict]` keyed by region/station index. Each value:
  `{"ts": <int, final timestep>, "data": <dict, see below>, "grid_df": <DataFrame>, "surf_df": <DataFrame>}`.
  - `data`: `{"header": {...}, "bin_coords": [...], "bin_normalized": ndarray, "Tdata": ndarray, "Ndata": ndarray, "Zdata": ndarray, "joint_pmf": ndarray}` —
    the velocity-space histogram (3D: tangential/normal/spanwise bin centers +
    normalized joint PMF). Produced by
    `rarefied.io.histograms.read_all_timesteps_histogram_3D`.
  - `grid_df` columns (via `rarefied.io.sparta_runs.relabel_grid_columns`): `id`,
    `n`, `nrho`, `mass`, `massrho`, `u`, `v`, `ke`, `temp`, `pxrho`, `pyrho`,
    `kerho`, `temp_thermal`, `pressure_thermal`.
  - `surf_df` columns (via `relabel_surface_columns`): `id`, `n`, `nflux`,
    `nflux_incident`, `mflux`, `mflux_incident`, `fx`, `fy`, `press`, `px`, `py`,
    `shx`, `shy`, `ke`, `etot` — this is the DSMC ground-truth stress/heat-flux
    data that `calculate_and_compare_moments` compares the Hermite-fit
    reconstruction against.
- `region_df`: `DataFrame`, one row per region — see Contract A above for columns.

Loader: `rarefied.io.sparta_runs.load_run_data_3D` builds this structure from a raw
SPARTA run directory; the pickle is a cached/archived copy of that. Read through
`data/registry.yaml`, never a hard-coded path.

### Contract D — Base state ([4c] → [5],[7])
Per-station continuum base + slip correction + gradients.
**Mechanism pinned down 2026-09-12; field list and storage format still open.**

The important finding: **AERO-F cannot give you the tangential derivatives.**
All its surface outputs are normal-projected — `HeatFluxPerUnitSurface` is
`q·n = -κ ∂T/∂n`, `Force` is `τ·n` — and there is no gradient-tensor output
(`PostFcn.h:40-43` exposes only `D2WALLGRAD`; `TEMPERATURENORMALDERIVATIVE = 33`
at `PostFcn.h:32` is a dead enum slot with no keyword and no computation behind
it).

But the information exists: AERO-F builds the **full** tensor internally and
only contracts at the last step (`PostFcn.C:1834` heat flux, `PostFcn.C:1256`
viscous traction). So stage [4c] reconstructs the same operator in
post-processing:

```
vol, dp1dxj = P1_gradient(tet_coords)    # == Elem::computeGradientP1Function
grad_u = sum_i u_i (x) dp1dxj[i]         # full 3x3 tensor
grad_T = sum_i T_i *  dp1dxj[i]          # full 3-vector
```

then project onto a local `(n, t1, t2)` basis per station. This is **not an
approximation relative to the solver's own view** — it is bit-for-bit the
operator AERO-F uses. Inputs needed: nodal `Velocity`/`Temperature` output
(merged via `sower -fluid -merge` to ASCII xpost) plus the mesh connectivity,
which this pipeline already has because it generated the mesh.

Two consequences worth recording:
- **Free correctness check:** in the one-cell extrusion the spanwise derivative
  must be ≈ 0. Any appreciable value means the extraction is wrong.
- **`SkinFrictionCoefficient` is unusable as-is:** `PostFcn.C:1246` hardcodes
  `Vec3D t(1.0, 0.0, 0.0)`, so it reports the *x-component* of traction, not
  the component tangential to the local surface. Acceptable for an airfoil at
  small incidence; wrong for an arbitrary shape, where the tangent sweeps
  through every direction. Use `Force` and decompose against each station's own
  tangents.

<!-- FILL IN, still open: the exact field list (rho, u, T at the wall reference;
     slip velocity; temperature jump; dT/dn and du_t/dn plus their tangential
     counterparts; local pressure/Mach), units, and on-disk storage format.
     Also open: how station indices map onto a manifest file (Contract A). -->

### Contract E — Moments ([5] → [6])
**Fully reconciled against the actual code (2026-09-12).** Two related structures:

1. **The Hermite fit itself** — `fit, error = fit_hermite_distribution_3d(...)`
   (`src/rarefied/moments/hermite.py`):
   - `fit`: `{"terms": ndarray[n_terms, n_terms, n_terms], "mean_T": float, "mean_N": float, "mean_Z": float, "var_T": float, "var_N": float, "var_Z": float}`.
     `terms[k, l, m]` is the `a_klm` Hermite coefficient (nondimensional, about the
     shifted/scaled `mean_*`/`var_*`); array shape is exactly `(n_terms, n_terms, n_terms)`,
     not `n_terms + 1`.
   - `error`: `{"mean_L1": float, "mean_L2": float, "L_inf": float}` — fit-quality
     diagnostics, not part of the coefficient vector itself.
2. **The per-region moments dict** — return value of `calculate_and_compare_moments`
   (`src/rarefied/moments/pipeline.py`), one array per key, each shape `(n_regions,)`:
   `pn_model`, `pt_model` (normal/tangential momentum flux from the Hermite fit, via
   `reconstruct/flux.py`'s `fast_momentum_flux_from_3D_hermite_fit`), `pn_dist`,
   `pt_dist` (same, computed directly from the raw distribution, bypassing the fit —
   a ground-truth-adjacent comparison, not the model output), `pn_dsmc`, `pt_dsmc`
   (DSMC ground truth from `surf_df`), `e_model`, `e_dist`, `e_dsmc` (heat-flux
   analogs). These are **already dimensional** (see stage [6]'s status note above —
   the stress/heat-flux "reconstruction" happens inside this same call, there is no
   separate serialized nondimensional-moments artifact today).

---

## Data locations & the registry

- Authoritative large data (VDF, run outputs) lives on the **cluster**.
- `data/registry.yaml` maps every dataset name → absolute path + format + provenance
  (case, run, config, date). **All code resolves paths through the registry.**
- Archived pickles are registered with provenance and read through the single loader
  in `src/rarefied/io/`.

**Status (2026-09-12): local paths populated, cluster paths still open.**
`data/registry.yaml` now exists with 10 entries: the 9 local pickles under
`~/Documents/Meteor_Modelling/code/` (`kn_eq_*_data` + `_3d_coarse` variants +
`my_data`, the last marked `incompatible_format` — do not use it) and the local
shapes dataset (`shape_dataset_2026_02_06_09_28_35`). Every entry's `cluster_path`
is `FILL_IN` — no cluster-connected session has confirmed those yet. Provenance is
recorded with explicit confidence levels (`verified`/`unverified`/
`incompatible_format`) rather than asserted uniformly — see the registry file's own
schema comment and `docs/CLEANUP_TODO.md`'s CONFIRM list for what's still open
(identical file sizes across several pickles, unconfirmed which SPARTA tool
produced them, etc.).

---

## Regression test (the safety net for everything above)

Before refactoring anything, freeze one known-good result from the corrupted-Maxwellian
experiment (a specific case: its moments and final dimensional fluxes) and write a test
that runs [5]→[6] on the archived pickle and checks it still reproduces those numbers.
Once green, relocation and refactors — yours or Claude Code's — are guarded.

- **Code:** `tests/regression/`
- **Status:** **Partially done, and it's important not to conflate the two:**
  `tests/characterization/` (`kn_eq_1p0_reference.npz`, `geometry_reference.npz`)
  now exists and *was* used to guard every relocation in Phases 3–5 of the
  2026-09-12 cleanup (`docs/CLEANUP_TODO.md`) — bit-for-bit diffs before/after every
  function move. **This is not the regression test this section originally asked
  for.** A characterization reference records "what the code currently outputs";
  it has no independently-derived "known-good" oracle value and doesn't claim the
  numbers are physically correct. `tests/regression/` is still empty — writing an
  actual regression test (frozen expected values from a specific validated case,
  ideally reconciled with "the corrupted-Maxwellian experiment" once CONFIRM #2 in
  `docs/CLEANUP_TODO.md` is resolved) is still `TODO`.

---

## Open items

- [ ] **AERO-F solver parameters** (flux, CFL law, limiter, reconstruction,
      supersonic startup). **User-owned**, informed by the AERO-F tutorials —
      explicitly not a pipeline concern (see the M4 Amendment). Available flux
      options: `Roe`, `VanLeer`, `HLLE`, `HLLC`, `RotatedRiemann`; no AUSM.
- [ ] **`size_min_factor`** — M4's wall-gradient convergence study has not been
      run, so case configs carry `FILL_IN` and the code falls back to
      `perimeter/n_surf`. This is the one value blocking M4's mesh clause from
      being fully satisfied.
- [ ] **`n_station`** — M6 says ~30; `circle_1m.yaml` sets 30 against
      `n_surf = 300`. The two are now independent but the number is a
      placeholder.
- [ ] **Station manifest format** (Contract A/D) — only indices are emitted
      today. Interacts with the unresolved question of whether the geometry
      notebook's circle-union "regions" are the same concept as M6's stations.
- [ ] **Far-field rule** — M4 says "keyed to the shock-standoff correlation";
      the mesh uses a fixed `20 D`.
- [ ] **Shape family vs M10's convex scope** — the Bezier population produces
      non-convex shapes with near-cusp spikes. M4 says sharp-edged geometries
      get a hand-checked mesh, not the automated pipeline. `curves.validate()`
      is the natural gate; the curvature threshold is a modeling choice.
      Measured: M13's +x-ray check already rejects ~25% of the population
      (2 of 8 on a pilot run) as non-star-convex, so the effective dataset
      yield is ~75% before any curvature gate is added.
- [ ] **Case config vs AERO-F input deck** — `config/cases/*.yaml` covers mesh
      and geometry only; flow conditions and solver settings still live in the
      `FluidFile` handled by `scripts/prep_aerof_case.sh`. Unifying them is
      deliberate future work.
- [x] Exact fields/formats for Contracts A–E (reconcile against existing code) —
      **C and E fully reconciled; A reconciled but flagged as possibly not the
      concept originally envisioned; B and D still open** (2026-09-12).
- [x] Populate `data/registry.yaml` — **local paths done; cluster paths still
      `FILL_IN`** (2026-09-12).
- [x] Name + document the SPARTA custom interface and its config schema — done in
      `src/rarefied/sparta/README.md`, but surfaced a real open scope question
      (see stage [3]'s status) that is not resolved by the documentation
      (2026-09-12).
- [x] Promote logic out of notebooks into `src/` — **done for moments, reconstruct,
      io, and geometry** (`src/rarefied/aerof/`, `features/`, `models/` remain
      empty — no corresponding notebook code was found to promote). **Write the
      regression test — still not done**, see the *Regression test* section above
      (2026-09-12).
- [ ] New from the 2026-09-12 cleanup, not yet resolved — see `docs/CLEANUP_TODO.md`
      for the full list with context: whether `SpartaAeroInterface` (parent repo)
      should be considered in-scope for this project; whether the geometry
      notebook's circle-union "regions" are the same concept as this pipeline's
      stage [2] sampling regions; which local pickle (if any) is "the
      corrupted-Maxwellian experiment"; several pickles' suspiciously identical
      file sizes; `my_data.pkl`'s incompatible format; the missing
      `docs/handoffs/argon_cylinder_cfd.md`.

---

*Draft v0.1 — the structure is settled; the `FILL IN`s are the reconciliation pass
against the real code and cluster layout.*

*2026-09-12: first reconciliation pass against the actual code (Contracts A/C/E,
`data/registry.yaml` local entries, the SPARTA interface split) — see
`docs/CLEANUP_TODO.md` for the full record and the open items it surfaced. Cluster
paths, AERO-F stage [4], and Contracts B/D remain unreconciled `FILL IN`s.*
