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
  input). **Not yet promoted:** no CLI entry point in `scripts/`.

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
- **[4a] Mesh:** hyperbolic extrusion on the (convex) analytic geometry; far-field
  keyed to shock standoff; wall spacing set by a **wall-gradient convergence study**
  (not y+). Code: `src/rarefied/aerof/`. `FILL IN mesh tool: GMSH? built-in?`
- **[4b] Run:** laminar, no-slip, compressible steady NS. `FILL IN: solver params for
  convergence` (this is an open item — see the argon-cylinder handoff).
- **[4c] Base-state extraction:** at each station, read (ρ, u, T) at the frozen wall
  reference; apply a-posteriori first-order Maxwell slip (velocity + temperature
  jump) from the near-wall gradients; also emit the near-wall gradients as features.
- **Consumes:** analytic geometry + station manifest [1]; case config.
- **Produces:** per-station **base state** + slip-corrected wall state + gradients.
- **Format:** see *Contract D*.
- **Location:** cluster; register outputs. `FILL IN`
- **Command:** `python scripts/run_aerof.py --case <case.yaml>`  `FILL IN`
- **Status:** setup infrastructure exists (built with Claude Code on the cluster);
  **first task is the argon-cylinder CFD reproduction.**

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
<!-- FILL IN: unchanged by the 2026-09-12 cleanup -- AERO-F work was out of scope
     for that pass. fields -- (rho, u, T) at the wall reference; slip velocity;
     temperature jump; near-wall dT/dn, du_t/dn; local pressure/Mach; units;
     storage format. -->

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

- [ ] AERO-F solver parameters for compressible-flow convergence (argon cylinder).
      Unchanged — out of scope for the 2026-09-12 cleanup.
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
