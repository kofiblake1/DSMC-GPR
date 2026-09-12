# Decision Log — DSMC-Informed GP Surrogate (rarefied-surrogate)

**Intended location:** `rarefied-surrogate/docs/decisions/README.md`
**Last updated:** 2026-09-11
**Status of this file:** draft, assembled from design conversation — reconcile each entry against the actual code/config before trusting it.

---

## How to use this log

Each entry records one locked (or proposed) decision: the **context** that forced it, the **decision** itself, and its **consequences** — plus, where relevant, what it **supersedes** or which alternatives were **rejected** and why. Recording the rejected paths is deliberate: it stops us (and any Claude Code session) from re-litigating settled questions.

This is a single-file log for now because the repo is mid-reorganization. When it stabilizes, each entry can be promoted to its own numbered ADR file (`0001-repo-layout.md`, …) with no rewriting.

**Status legend:**
- **Accepted** — decided and in force.
- **Proposed** — leaning this way, but gated on a diagnostic or not yet committed.
- **Superseded** — replaced by a later entry (noted inline).

**Scope tags:** `[eng]` engineering/repository · `[model]` method/physics.

---

# A. Engineering / repository decisions

## E1 — `rarefied-surrogate` is a self-contained, extractable subtree of `DSMC-GPR` `[eng]`
**Status:** Accepted

**Context.** `DSMC-GPR` currently holds both this surrogate project and the separate direct DSMC–FEM coupling project. They share a *domain* (rarefied, DSMC, FST) but the actually-shared *code* is narrow (SPARTA setup, geometry/mesh generation, argon properties, some I/O). A repository is the unit of context for a Claude Code session; a two-project repo taxes every session.

**Decision.** Keep both projects physically in `DSMC-GPR` for now, but develop the surrogate work as a self-contained subtree (`rarefied-surrogate/`) with its own `README.md`, `PIPELINE.md`, `src/` package namespace, and tests — structured **as if it will be extracted**. Organize `src/` **by pipeline stage** (geometry → sampling → sparta → moments → reconstruct), not by file type, so the layout mirrors the data flow.

**Consequences.** Focused context for Claude Code today (point a session at the subtree, declare the sibling project out of scope) without paying the split cost. Splitting later is a `git filter-repo` + directory move, not a disentangling project.

**Split trigger (watch for any of these):** a Claude Code session edits the wrong project; incompatible dependency versions between the two; the SciTech artifact needs a clean standalone repo; the work is made public. On the first trigger, extract — and factor the genuinely shared code into a small `dsmc-common` package rather than duplicating it.

---

## E2 — Notebooks are an interface; `src/` holds the implementation `[eng]`
**Status:** Accepted

**Context.** Most existing work lives in Jupyter notebooks, which gave fast iteration but strand load-bearing logic (moment math, shape/region generation, reconstruction) in a form that is hard to reuse, test, diff, or hand to Claude Code.

**Decision.** **If a line of code defines behavior, it belongs in `src/rarefied/`; if it inspects behavior, it belongs in a notebook.** Notebooks import from the package and look at results; they do not define pipeline logic. Two notebook classes: `notebooks/exploratory/` (unpoliced scratchpad, not part of the pipeline) and `notebooks/validation/` (thin, reproducible, e.g. the corrupted-Maxwellian figures). A **promotion path** governs migration: develop freely in an exploratory notebook, and when a piece of logic earns its keep (will be called again by a stage, a test, or future-you), promote it into `src/`, replace the cell with an import, and add a test if it is load-bearing. Do **not** over-promote genuine one-offs.

**Consequences.** Fast loop preserved via `%autoreload 2` (edit `src/`, re-run cell, see change with no restart). Use `jupytext` on **validation** notebooks only, so they diff cleanly and Claude Code can read/edit the paired `.py`; exploratory notebooks stay unpaired to avoid friction.

---

## E3 — Data stays out of git; a registry points to it `[eng]`
**Status:** Accepted

**Context.** VDF/distribution data (including existing local pickles) is large and lives authoritatively on the cluster, where the compute is. Local and cluster must stay in sync via git, which is for small versioned code/config.

**Decision.** Commit code and config; **do not** commit data. Maintain `data/registry.yaml` as a manifest mapping dataset name → absolute path, format, and provenance (which case/run/config produced it). All code resolves data paths **through the registry**, never hard-coded. Treat the existing pickles as **precious archived inputs**: register them with provenance and read them through a single documented loader in `src/rarefied/io/`.

**Consequences.** One place answers "where is dataset X"; a cluster session can resolve paths a local session cannot. Registry is the natural future hook for DVC/git-annex if reproducibility guarantees are later needed — not adopted now (a documented manifest is ~90% of the benefit).

**Later hardening (optional):** migrate the archival format from pickle to a language-agnostic, self-describing format (HDF5 / NPZ with a documented schema) to escape pickle's version-coupling. Deferred.

---

## E4 — A regression test guards the known-good results before any refactor `[eng]`
**Status:** Accepted (to be created)

**Context.** The hand-derived moment functions and reconstruction are validated crown jewels; the danger in reorganizing is silently breaking them while moving them.

**Decision.** Freeze one trusted case (a specific Kn, its computed moments, its final dimensional fluxes) from the corrupted-Maxwellian experiment as expected values, and write an end-to-end regression test that runs the pipeline on the archived pickle and checks it still reproduces them. Write this **immediately after the mechanical relocation and before any refactoring**.

**Consequences.** Once green in the new structure, relocation is proven non-destructive, and every later change (including edits by a Claude Code session on the cluster) is guarded. This is what makes it safe to let a remote session modify the moment/reconstruction code.

---

# B. Method / modeling decisions

## M1 — Single-Maxwellian base representation `[model]`
**Status:** Accepted

**Context.** The VDF is represented as a base Maxwellian times a Hermite correction series; the corrector learns the correction coefficients.

**Decision.** The base is a **single** Maxwellian, defined by one density, one bulk velocity, one temperature. This is a deliberate architectural choice, **not a simplification**: it maps exactly onto what any continuum solver (Euler/NS) produces at the wall, so the corrector can sit on top of a cheap base the host simulation already computes.

**Rejected alternative — multi-Maxwellian / Mott-Smith (two-stream) base.** Although a bimodal base better matches near-free-molecular VDFs, it carries extra parameters (split fraction, two drift/temperature pairs) that **no continuum solver produces**, so the base would have to be *learned* rather than *read from the solver* — ill-posed in exactly the way this architecture avoids. The single Maxwellian's inability to represent bimodality is what the correction terms are for (see M2).

---

## M2 — Surface fluxes are exact orthogonal projections onto low-order Hermite coefficients `[model]`
**Status:** Accepted — **load-bearing**

**Context.** Reviewers will invoke Grad/Hermite convergence limits (e.g. Cai–Torrilhon: the series about a single Maxwellian diverges once tail temperature exceeds ~2× the base). Our high-Kn cases plausibly violate that.

**Decision.** We do **not** reconstruct the pointwise VDF; we compute a finite set of low-order velocity moments (stress = 2nd, heat flux = 3rd). Because the Hermite basis is orthogonal against the Maxwellian weight, each such moment is an **exact linear functional of the low-order coefficients**, insensitive to truncation error or divergence of the full series (which live entirely in the functional's kernel). Coefficients are obtained by **orthogonal projection** (direct moments of the sample), **not** by nonlinear least-squares fitting of the pointwise VDF — projection is what guarantees the low-order coefficients equal the empirical low-order moments.

**Consequences.** Series divergence and negative/non-realizable truncated `f` are irrelevant to the delivered fluxes, provided we never evaluate `f` pointwise (we don't, for the FST use case). This is the one-line reviewer rebuttal. It also means the implementation must keep using projection, never a curve-fit, or the guarantee is lost.

---

## M3 — Base predictor: laminar no-slip AERO-F + a-posteriori first-order Maxwell slip, sampled at the wall `[model]`
**Status:** Accepted — supersedes earlier base-predictor framings

**Context.** AERO-F has no built-in slip model and no easy Robin-BC path, and we will not implement one now. No-slip *read at the wall* is degenerate (u=0, T=T_wall → stationary Maxwellian carrying zero convective flux).

**Decision.** Run **laminar no-slip** NS in AERO-F to convergence, then apply **first-order Maxwell slip-velocity and temperature-jump** algebra as a **post-processing step** from the converged near-wall gradients, and read the base Maxwellian **at the wall** (now non-degenerate). Match the accommodation coefficient to SPARTA's wall model.

**Supersedes:** (a) NS-with-slip via Robin BC — no easy AERO-F path; (b) reading the base at the boundary-layer edge / a fixed off-wall offset — the a-posteriori slip removes the need for an arbitrary reference.

**Consequences.** Known consistent bias (no-slip over-stiffens the wall gradient → slip slightly high, growing with Kn) is smooth and absorbed by the corrector. The corrector's usable Kn ceiling is now **empirical**, set by the Week-1 base-smoothness diagnostic; above it, hand over to the freestream-married corrector (M8). AERO-F doubles as the fluid side of the eventual coupling.

**Second-order slip:** rejected for now (no agreed coefficients; needs a noisier 2nd normal derivative; benefit only above Kn~0.5 where the base is dubious anyway). Journal-scope at most.

**Corroborated 2026-09-12 — the premise is stronger than stated.** This entry's context says "AERO-F has no built-in slip model." The source in fact contains a *partially implemented but entirely dead* Maxwell slip model, which supports the a-posteriori decision rather than undermining it. Do not be misled into thinking the built-in path is usable:

- `Face::computeMaxwellSlipTerm` (`Face.C:806-809`) is an **empty stub**, and it is the only possible caller of `Elem::computeMaxwellSlipTerm`.
- `SubDomain::computeMaxwellSlipTerm` (`SubDomain.C:2315`) has **no callers anywhere** in the tree.
- `MaxwellSlipData::zeroconstant` **defaults to 0.0** and multiplies *both* terms in `MaxwellSlipFcn::compute` (`SlipFcn.h:35`), so the kernel returns zero slip velocity even if reached.
- There is **no temperature jump** at all (no `TemperatureJump`/`Smoluchowski` anywhere). First-order Maxwell slip needs both; velocity slip alone over-predicts wall heat flux.
- The intended integration point is the ROM hyperreduction path (`sampleFaces`, assignment rather than accumulation), not the standard residual assembly.
- It exists only in `aero-f-2`, not `aero-f` or `aero-f-lite`.

Making it work would be a solver project — implement the face hook, wire it into residual assembly over all wall faces, set `ZeroConstant`, and add the missing temperature jump — not a configuration change. M3's post-processing route remains the correct call.

---

## M4 — Laminar (no turbulence model); wall-gradient convergence study replaces y+; no solution-adaptive refinement `[model]`
**Status:** Accepted — **mesh-topology clause amended 2026-09-12** (see *Amendment* at the end of this entry)

**Context.** Reynolds number across 90–120 km is ~10^3 at the dense-end worst case (checked: 90 km, L=10 m, M=20 gives Re ~1.5e4; typical meter-scale ~200) versus transition ~5e5–1e6. Rarefaction forces low Re.

**Decision.** Use **laminar** compressible NS — no RANS, no LES. Because turbulence is absent, the **y+ apparatus does not apply**; instead a **wall-gradient convergence study** (refine near-wall spacing until wall heat flux and shear stop changing) sets and freezes the spacing, since the wall-normal gradients are the consumed product. Use a fixed **graded unstructured (gmsh) mesh topology** on convex bodies, with a **scale-relative sizing rule** so the discretization bias is consistent across the geometry sweep, and stagnation-region resolution; far-field distance keyed to the shock-standoff correlation. **No solution-adaptive mesh refinement.** *(Amended 2026-09-12 — was "hyperbolic-extrusion … with a carbuncle-resistant flux (AUSM-type)". See the Amendment note below.)*

**Rejected — RANS/LES:** would inject spurious eddy viscosity into a flow with none, corrupting the near-wall gradients. **Rejected — adaptive refinement:** makes discretization error (hence base bias) a solution-dependent function per case, breaking the base-consistency the corrector marriage depends on. Consistency across the geometry sweep beats peak per-case shock sharpness.

**Consequences.** The residual meshing difficulty concentrates at the leading-edge/stagnation station (shortest gradient length) — run the W1 wall-gradient study at the bluntest, highest-Mach case. Sharp-edged geometries (flat panel) get a hand-checked mesh, not the automated pipeline.

---

### Amendment (2026-09-12): mesh topology and flux

Two clauses of the decision above were written before the AERO-F source and option set had been read. Both are corrected here rather than silently overridden; the laminar, no-adaptive-refinement, and wall-gradient-study clauses are **unchanged and still in force**.

**1. Hyperbolic extrusion → graded unstructured gmsh.**

The original rationale for hyperbolic extrusion was to guarantee clean wall-normal gradients by making mesh lines orthogonal to the wall. That turns out to be unnecessary, because **AERO-F never uses mesh orthogonality to form them.** For each boundary face it builds the *full* gradient tensor from the attached tetrahedron's P1 shape-function gradients and only then contracts with the face normal:

- heat flux — `PostFcnNS::computeHeatPower`, `PostFcn.C:1834`: `computeTemperatureGradient(dp1dxj, T, dTdxj)` → `computeHeatFluxVector(kappa, dTdxj, qj)` → `q·n`
- viscous traction — `computeViscousForce`, `PostFcn.C:1256`: `computeVelocityGradient(dp1dxj, u, dudxj)` → `computeStressTensor` → `tij·n`

The P1 gradient is exact for a linear field on any non-degenerate tet, so element orthogonality never enters the consumed product. What *does* set the error is near-wall **spacing** — which is precisely what M4's wall-gradient convergence study already governs.

The replacement serves M4's real intent (consistent discretization bias across the sweep, so the corrector's base stays consistent) by a different mechanism: every mesh length is expressed as a fraction of the shape's characteristic diameter `D`, making the sizing rule scale-invariant. See `config/schema.md`.

Practical consequence: this removes the pyHyp toolchain (CGNS + PETSc + cgnsutilities, none of which are available as Sherlock modules) from the critical path entirely.

**2. AUSM-type flux → not used, and not available.**

AERO-F offers only `Roe`, `VanLeer`, `HLLE`, `HLLC`, `RotatedRiemann` (`IoDataCore.C:3313-3315`). There is no AUSM option, so the original clause was unimplementable as written.

**We are not using an AUSM flux.** AERO-F handles shock waves and discontinuities perfectly well with its available schemes; the earlier clause reflected a concern about blunt-body carbuncles that is better addressed through solver configuration than through a flux the code does not have.

**Selecting the solver settings** — flux, CFL law, limiter, reconstruction, and the startup strategy for supersonic cases — is a separate task, owned by the user and informed by the AERO-F tutorials. It is deliberately **not** a pipeline concern and is not decided here. The mesh pipeline is agnostic to all of it.

**3. One further gap this exposed, recorded not decided.** Stage [4c] needs *tangential* wall derivatives as well as normal ones, and AERO-F exposes neither a gradient tensor nor a tangential derivative: the `VectorType` enum (`PostFcn.h:40-43`) offers only `D2WALLGRAD`, and `TEMPERATURENORMALDERIVATIVE = 33` (`PostFcn.h:32`) is a dead enum slot with no input keyword and no computation behind it. So stage [4c] must reconstruct the P1 tet gradient itself in post-processing — bit-for-bit the same operator quoted above. See `PIPELINE.md` Contract D.

---

## M5 — Feature vector: geometry + freestream + ARD-pruned base-solver locals `[model]`
**Status:** Accepted

**Context.** Pure local geometry is a complete point description at high Kn (free-molecular locality) but incomplete at moderate Kn, where non-local/collisional history matters.

**Decision.** Keep the **local geometric block** (local angle of attack, angle about centroid, surface normal, curvature) and the **global freestream block** (global Kn, Mach/speed, altitude, wall temperature). **Add base-solver-derived local features**, led by the **gradient-length-local Knudsen number** `Kn_local = lambda_local / L_Q` (local mean free path from the base wall state over a local gradient length `L_Q = Q/|grad Q|`) — the natural coordinate for how much correction a point needs — plus near-wall `dT/dn` and `du_t/dn` (already computed for slip) and local pressure/Mach.

**Admissibility rule (hard):** a feature is allowed **iff** it is computable from (geometry + freestream + base solve) without running DSMC. Anything derived from the DSMC/true VDF is an **output** and is forbidden as an input (target leakage).

**Consequences.** Nondimensionalize features so ARD lengthscales are comparable and transfer across scale; let **ARD prune** redundancy (Kn_local, local pressure, dT/dn are correlated — keep the most physical, drop the rest). Note the two-level data structure: one run yields ~30 *correlated* local points at one freestream, so effective sample size tracks the number of **distinct runs**, not rows.

---

## M6 — DSMC↔CFD registration via shared surface stations; wall-normal-thin sampling; statistics by time-averaging `[model]`
**Status:** Accepted

**Context.** AERO-F (body-fitted, nodal) and SPARTA (region-sampled) have different surface discretizations, so pairing the DSMC VDF with the AERO-F base state is not naturally one-to-one. The earlier SPARTA sampling used a union of small isotropic circles.

**Decision.** Neither grid is master. Generate both the AERO-F surface and the SPARTA surface elements from a **single analytic geometry**, and define a shared set of **surface stations** (~30, indexed by arc-length / angle-about-centroid, each with location + normal) on that analytic surface. Both codes report to the station id → pairing is one-to-one by construction; the station set is also where the feature vector lives, and (at deployment) the station→FEM transfer reuses the AERO-suite matcher. For the DSMC sampling itself, use a **wall-normal-thin, tangentially-elongated** region at a **frozen normal offset**, and buy statistics by **time-averaging the steady ensemble** and particle weighting — **not** by enlarging the region.

**Rejected — isotropic / enlarged sampling regions:** they average over ~a mean free path in the wall-normal direction, smearing the incident/reflected bimodality and the gradients we need.

**Consequences.** Size the region against `L_Q` (stagnation station is the worst case). Freeze the offset via a **heat-flux-monitored offset/size sensitivity study** (heat flux is the tail-sensitive moment that degrades first; low-order moments won't warn you) — this belongs in the Week-1 pipeline freeze.

---

## M7 — Two-level design of experiments: LHS seed + active learning on GP variance `[model]`
**Status:** Accepted

**Context.** Expanding the feature space makes sampling design matter; each expensive DSMC run yields ~30 local points for free but only **one** global (geometry × freestream) condition.

**Decision.** Design at two levels. **Global level** (which geometry × freestream to run — where the DSMC budget lives): seed with a space-filling **Latin hypercube**, then **active learning** — place subsequent runs where the GP posterior variance is highest. **Local level** (surface stations): comes free from discretizing each body.

**Rejected — full-factorial / even-partition grid:** dies in the ~6–10-D global space.

**Consequences.** Same variance signal drives per-geometry fine-tuning (M8). Acquisition targets *global* conditions; take the local sweep for free.

---

## M8 — Learning architecture: two regime correctors; foundation model + per-geometry exact-GP residual `[model]`
**Status:** Accepted (abstract-scope small; scaled version journal-scope)

**Context.** No single base spans the full Kn range, and we want a model that trains on general geometries yet sharpens on a specific one.

**Decision.** **Two correctors by regime:** a no-slip-married corrector (continuum → transition, up to the empirical ceiling from M3) and a freestream-married corrector (free-molecular, base = analytic freestream Maxwellian). **Transfer via the same recursion as the method:** a **foundation model** `F(x)` trained on pooled multi-geometry data with a scalable backbone (**SVGP** with inducing points, or a small NN), then **per-geometry fine-tuning** by fitting an **exact residual GP** `G(x)` to `r = a - F(x)` on a small active-learning-selected target set, deployed as `a_hat = rho*F(x) + G(x)` (Kennedy–O'Hagan autoregressive form; warm-start G's lengthscales from F). The exact residual GP restores calibrated uncertainty on the specific object.

**Scope.** Abstract: demonstrate the mechanism on a handful of geometries with a held-out target (zero-shot vs few-shot vs from-scratch). Journal: the scaled random-geometry foundation model.

**Note on naive vs sparse GP:** an exact GP is O(N^3) and caps ~1e4 points; SVGP funnels the data through M << N inducing points for ~O(N·M^2), hence the split (SVGP foundation for scale, exact GP for the small residual with UQ).

---

## M9 — Declared scope is quasi-steady `[model]`
**Status:** Accepted

**Context.** The surrogate returns a steady flux for the instantaneous surface state; this is only valid when the fluid relaxes far faster than the structure moves or heats.

**Decision.** State the **quasi-steady assumption** explicitly as the method's declared domain of applicability: time-scale separation `tau_fluid << tau_structure`, equivalently reduced frequency `k = omega*L/U << 1`. Put this up front in the paper.

**Consequences.** Turns "we don't do unsteady problems" from an apparent limitation into a scoped modeling choice. It is the criterion that admits M10's aeroheating case and excludes panel flutter.

---

## M10 — Validation case & geometry: quasi-steady conjugate aeroheating on a convex body `[model]`
**Status:** Accepted — supersedes panel flutter as the primary case

**Context.** The primary validation must live inside the quasi-steady domain (M9) and on a geometry the surrogate is trained for.

**Decision.** Primary validation/demonstration = **quasi-steady conjugate aeroheating on a convex (reentry-capsule-like) body**: the fluid relaxes in microseconds while the wall heats over seconds–minutes, and it maps onto the published DSMC–FEM conjugate benchmark (external reference). The **geometry** is a distribution of convex-ish shapes for the foundation set plus a **held-out capsule-like body** that doubles as the generality target (M8) and the aeroheating-demo geometry. Standing validation ladder: MMS verification → a-priori static moments → one-way steady conjugate (external ref) → quasi-steady march stability/consistency audit → analytic high-Kn flux check → (stretch) one minimal coupled reference run + speedup.

**Superseded / deferred.** **Panel flutter demoted** to a journal boundary-of-validity study — flutter's physics *is* the unsteady aerodynamic phase lag (reduced frequency not small), which a quasi-steady surrogate structurally cannot represent; a flat panel is also geometrically out-of-distribution with near-singular edges. **Micrometeoroid entry** deferred to a journal regime-spanning showcase.

---

## M11 — Density handling `[model]`
**Status:** **Proposed** — gated on the Week-1 near-wall-density diagnostic

**Context.** The normalized VDF integrates to 1, so dimensional fluxes scale linearly with the externally supplied density `n` (equivalently the zeroth Hermite coefficient `a_0`). Flux error inherits density error one-to-one — true of any method, not a defect, but it depends on how well the base predicts near-wall density.

**Decision (tentative).** Default to taking `a_0`/density **externally from the base solver** (keeps the normalized shape corrections O(1) and interpretable). **If** the W1 diagnostic shows the base's near-wall density error is not smooth/small at the target Kn, add a **learned `a_0` correction** — a density correction on top of the base density, using the same GP machinery as the higher moments — rather than folding scale into the shape regression.

**Consequences.** Decision finalizes after the W1 measurement. Keep scale (`a_0`) and shape (`a_{k>=1}`) corrections separable either way, to preserve the interpretable decomposition.

---

## M12 — Mesh generation is gmsh-driven; the pipeline is cluster-only from stage [4a] `[eng]`
**Status:** Accepted (2026-09-12)

**Context.** M4's amended mesh clause needs a concrete tool. Two constraints decided it. First, `gmsh2top` — the only route from a mesh to an AERO-F `.top` — accepts **only** element types 2 (triangle) and 4 (tetrahedron), and cannot read MSH 4.1 (its version-4 branch was written against 4.0). Second, Sherlock's `gmsh/4.10.1` module forces `gcc/10.1.0`, which breaks the py312 numeric stack the rest of the package runs on, and the pip `gmsh` wheel cannot load at all because `libGLU.so.1` is absent cluster-wide.

**Decision.** Generate a `.geo` from the curve and invoke the **gmsh CLI as a subprocess** in its own module environment (`src/rarefied/aerof/gmsh_env.sh`); never import gmsh into the Python process. Always write **MSH 2.2**, and never use `Recombine`, so extruding a triangulated surface yields tetrahedra.

**Consequences.** The `.geo` is a human-inspectable artifact that mirrors the group's working `NACA0012.geo`, which is a side benefit rather than a cost. It also fixes the capability boundary: stages from [4a] onward need `gmsh`, `gmsh2top`, `mpmetis`, `sower`, `cd2tet`, `aerof2` — all group-installed in `/home/groups/cfarhat/bin`, with **no local equivalent**. The pure-Python layer (`geometry/curves`, `aerof/mshio`, `aerof/quality`) deliberately imports none of them and runs anywhere, so curve generation, mesh-quality metrics, and the eventual Contract D reconstruction stay locally testable. Documented, not enforced in code.

---

## M13 — Surface stations are ordered clockwise, anchored on the +x ray `[model]`
**Status:** Accepted (2026-09-12)

**Context.** M6 makes shared surface stations the DSMC↔CFD registration layer, pairing the two solvers by station id. That only works if both sides agree on the ordering. They did not: the SPARTA-side generator `geometry/circle.py:51` emits points **clockwise** (`np.linspace(0, -2*np.pi, …)`), while the new AERO-F-side curve module was written **counter-clockwise**. An orientation mismatch reverses station ids everywhere except index 0 — and on a symmetric test shape it would look fine in aggregate while being wrong point-by-point.

**Decision.** **Clockwise, project-wide.** Station 0 is anchored where the **+x ray from the area centroid** crosses the surface, proceeding clockwise.

The constraint ordering made this nearly forced:

| side | requirement | source |
|---|---|---|
| SPARTA | **clockwise, hard** | external requirement on surface node definition |
| AERO-S / existing coupling | **clockwise** | `notebooks/mesh_pipeline/mesh_export.py:76`, `np.argsort(-angles)` |
| AERO-F | **indifferent** | repairs orientation itself, see below |

AERO-F is orientation-agnostic, demonstrated rather than assumed: `GeoSource.C:340` calls `Elem::checkVolume` (`ElemCore.C:150-160`), which swaps nodes 1↔2 on any negative-volume tet at load time, and the reference circle run reported `changed the orientation of 11004 boundary faces` (of 21152) while still producing correct physics (stagnation at the nose, M≈0.87 over the shoulders, Cd≈0.19, `Lz/|F| = 2e-6`).

One side has a hard requirement, one already complies, the third is free. Clockwise also **invalidates nothing**: `circle.py`, the archived `KN_eq_*` pickles, and `tests/characterization/geometry_reference.npz` are all untouched.

**Consequences.**
- `signed_area()` is **negative** for a valid curve; callers wanting magnitude must use `abs()`.
- `aerof/geo.py` reverses to counter-clockwise internally for gmsh's `Plane Surface`. That is private and non-load-bearing (AERO-F repairs face orientation anyway); station indices are always defined on the clockwise input.
- The +x-ray anchor is required for station ids to be **reproducible across shapes** — without it, index 0 falls wherever the input curve happened to start, which for a Bezier shape is arbitrary.
- Shapes whose +x ray crosses the surface more than once have an **ambiguous** origin and are **rejected** by `curves.validate`. This is consistent with M10's convex-body scope and drops into the batch driver's existing reject/retry loop. Note a single closed loop always has an odd crossing count, so the failure mode is 3+, not 0 — and a shape that is star-convex about its centroid always has exactly 1.

**Rejected alternative — standardize on counter-clockwise** (the mathematical convention, and what positive shoelace area implies): impossible, because SPARTA's requirement is external and non-negotiable, and flipping the SPARTA side would invalidate the archived pickles this project's own characterization reference depends on.

**Rejected alternative — keep both conventions with an index-mapping helper** (`i → (n-i) % n`): preserves existing data but leaves two live conventions in a codebase whose whole premise is that station ids pair one-to-one. The mapping would be correct exactly until someone forgot to call it.

---

# C. Related open items (not yet decisions)

These are pending confirmations, tracked here so they aren't lost — resolve into decisions as they settle:

- **AERO-F argon transport model** for the argon-cylinder reproduction: match Lofthouse's collision-integral viscosity via a power-law with omega = 0.734 anchored to the VHS reference, plus Prandtl = 2/3 (Eucken, monatomic) — *to derive/confirm*.
- **Domain extent / far-field placement** rule vs. existing AERO-F setup infrastructure — *to confirm*.
- **Which Kn cases to run** (all four vs. focus), symmetry (half vs. full cylinder), 2-D planar confirmation — *to confirm*.
- **Argon constants** (molecular mass, gamma = 5/3) stated but to be confirmed against the papers.

---

*End of log. Promote entries to individual numbered ADR files once the reorganization settles.*
