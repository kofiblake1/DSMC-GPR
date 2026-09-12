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
- **Location:** small; committed with the case or written to the run dir. `FILL IN`
- **Command:** `python scripts/gen_geometry.py --case <case.yaml>`  `FILL IN`
- **Code:** `src/rarefied/geometry/`
- **Status:** exists in notebooks; **promote to `src/`.** `FILL IN: source notebook(s)`

### [2] Sampling-region generation (SPARTA side)
- **Consumes:** station manifest [1]; sampling params (normal offset, band size).
- **Produces:** per-station sampling-region definitions (wall-normal-thin,
  tangentially-elongated) + the union set for successive collection runs.
- **Format:** the custom-interface region format. See *Contract B*. `FILL IN`
- **Location:** written to the run dir on the cluster. `FILL IN`
- **Command:** `python scripts/gen_regions.py --case <case.yaml>`  `FILL IN`
- **Code:** `src/rarefied/sparta/`
- **Status:** exists (notebooks + custom interface). **Promote / document interface.**

### [3] SPARTA setup & runs
- **Consumes:** region definitions [2]; the **custom config file**; freestream +
  gas + wall params from the case config.
- **Produces:** **3D VDF data per station** (velocity-space histograms), time-averaged
  over the steady ensemble.
- **Format:** currently **pickle**. See *Contract C* — document its internal structure.
- **Location:** cluster run dirs; archived pickles also stored locally. Register both
  in `data/registry.yaml`. `FILL IN paths`
- **Command:** custom interface + SPARTA launch. `FILL IN exact invocation`
- **Code:** `src/rarefied/sparta/` (wrappers) + the custom interface. `FILL IN name/loc`
- **Status:** exists and works. Wrap + document; do not rewrite.

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
- **Format:** see *Contract E*.
- **Location:** derived; cache to the run dir + register. `FILL IN`
- **Command:** `python scripts/compute_moments.py --case <case.yaml>`  `FILL IN`
- **Code:** `src/rarefied/moments/` — **the hand-derived functions. Canonical.
  Wrap and test; do not "clean up" the math.**
- **Status:** exists (hand-derived, validated in the corrupted-Maxwellian experiment).

### [6] Reconstruction → dimensional surface fluxes
- **Consumes:** moments [5]; run info (number density / mass) for redimensionalization.
- **Produces:** dimensional surface **stress tensor M_ij** and **heat flux Q_i**
  (the momentum/energy flux to the surface). Note density enters as an external
  multiplier (see decisions: fluxes are linear in n).
- **Format:** per-station dimensional quantities. `FILL IN exact fields`
- **Command:** `python scripts/reconstruct.py --case <case.yaml>`  `FILL IN`
- **Code:** `src/rarefied/reconstruct/`
- **Status:** exists; promote + test.

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
One record per surface station. Proposed fields:
| field | meaning |
|---|---|
| `station_id` | stable integer/string index (the shared key across all stages) |
| `s` | arc-length along the surface |
| `phi` | angle about centroid |
| `x`, `y` | location on the analytic surface |
| `nx`, `ny` | outward unit normal |
| `kappa` | local curvature |
<!-- FILL IN: exact field names/units in the existing geometry code; storage format
     (YAML/JSON/structured array/CSV)? -->

### Contract B — Sampling-region definition ([2] → [3])
Per-station region geometry (wall-normal-thin, tangentially-elongated band) + the
union set, in the **custom-interface format**.
<!-- FILL IN: the custom interface's expected schema; the custom config file's fields;
     the frozen normal offset and band dimensions; the interface's name/location. -->

### Contract C — 3D VDF data ([3] → [5])
Per-station, time-averaged velocity-space distribution. Currently **pickle**.
<!-- FILL IN: the pickle's internal structure — keys, array shapes, velocity-grid
     definition (bin edges/centers, units), how stations are indexed, what run
     metadata travels with it (density, timestep, sample count). Then write ONE
     loader in src/rarefied/io/ that reads it into a documented in-memory object. -->

### Contract D — Base state ([4c] → [5],[7])
Per-station continuum base + slip correction + gradients.
<!-- FILL IN: fields — (rho, u, T) at the wall reference; slip velocity; temperature
     jump; near-wall dT/dn, du_t/dn; local pressure/Mach; units; storage format. -->

### Contract E — Moments ([5] → [6])
Per-station Hermite coefficients and low-order moments.
<!-- FILL IN: which a_klm are kept and their ordering/normalization; the nondimensional
     stress and heat-flux definitions; how the coefficient vector is stored. -->

---

## Data locations & the registry

- Authoritative large data (VDF, run outputs) lives on the **cluster**.
- `data/registry.yaml` maps every dataset name → absolute path + format + provenance
  (case, run, config, date). **All code resolves paths through the registry.**
- Archived pickles are registered with provenance and read through the single loader
  in `src/rarefied/io/`.

<!-- FILL IN: populate data/registry.yaml with real cluster paths, local pickle
     locations, and provenance for each existing dataset. -->

---

## Regression test (the safety net for everything above)

Before refactoring anything, freeze one known-good result from the corrupted-Maxwellian
experiment (a specific case: its moments and final dimensional fluxes) and write a test
that runs [5]→[6] on the archived pickle and checks it still reproduces those numbers.
Once green, relocation and refactors — yours or Claude Code's — are guarded.

- **Code:** `tests/regression/`
- **Status:** `TODO` — write immediately after the mechanical relocation, before any
  refactor. `FILL IN: which case + expected values.`

---

## Open items

- [ ] AERO-F solver parameters for compressible-flow convergence (argon cylinder).
- [ ] Exact fields/formats for Contracts A–E (reconcile against existing code).
- [ ] Populate `data/registry.yaml` (cluster + local paths, provenance).
- [ ] Name + document the SPARTA custom interface and its config schema.
- [ ] Promote logic out of notebooks into `src/`; write the regression test.

---

*Draft v0.1 — the structure is settled; the `FILL IN`s are the reconciliation pass
against the real code and cluster layout.*
