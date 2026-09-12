# Case config schema

One YAML per case in `config/cases/`. Consumed by `scripts/gen_mesh.py`
(stage [4a]); explicit command-line flags always override the file.

Only mesh/geometry fields are defined so far. Flow conditions and solver
settings are **not** in these files yet — the AERO-F input deck (`FluidFile`)
is still handled separately by `scripts/prep_aerof_case.sh`. Unifying the two
is deliberate future work, not an oversight; see the open items in
`PIPELINE.md`.

`FILL_IN` marks a value that must come from a study that has not been run.
Leaving it as `FILL_IN` (or 0, where that triggers a documented fallback) is
correct; guessing a plausible number is not.

---

## `name`
Case identifier. Becomes the output directory under `$MESH_RUN_ROOT` and the
stem of every artifact.

## `geometry`

| field | meaning |
|---|---|
| `source` | `circle` (analytic, exact — the validation target) or `bezier` (random, from the upstream `shapes` package via `$SHAPES_DIR`) |
| `diameter` | metres. `circle` only. |
| `n_surf` | wall points = **mesh wall nodes**. Sets how well the wall is resolved. `Transfinite Curve {…} = 2` pins the mesh nodes to exactly these points, so this is also the wall discretisation. |
| `n_station` | **shared surface stations** (M6's DSMC↔CFD registration layer, ~30). A coarse subsample of the wall nodes, so every station id is an exact mesh node and stage [4c] needs no interpolation. `0` ⇒ every wall node is a station. |

`n_surf` and `n_station` are **deliberately independent**. M4 sets wall
resolution from a gradient convergence study; M6 sets station count from the
registration requirement. Earlier code derived the near-wall cell size as
`perimeter / n_surf`, which coupled them — that coupling is now a fallback
only (see `size_min_factor`).

Curves are **clockwise**, with station 0 anchored where the +x ray from the
area centroid crosses the surface (decision M13). Shapes whose +x ray crosses
more than once are **rejected** by `curves.validate` — the station origin would
be ambiguous. That is consistent with M10's convex-body scope.

## `mesh`

Every length is a fraction of the characteristic diameter **D** (the diameter
of the circle of equal area), so the sizing rule is scale-invariant across a
geometry sweep. That invariance is the point: it keeps the discretisation bias
consistent from shape to shape, which is what M4 actually requires of the base
predictor.

| field | meaning |
|---|---|
| `reference_diameter` | pin D to an exact value instead of deriving it from the enclosed area. `0` ⇒ derive. Use it for reference cases: a 300-gon on a 1 m circle measures D = 0.99996, which would make every derived length untidy. |
| `size_min_factor` | **near-wall cell size / D.** Per M4 this must come from a wall-gradient convergence study (refine until wall heat flux and shear stop changing) — *not* from y⁺, and *not* from the station count. `0` ⇒ fall back to `perimeter / n_surf`. The resolved value and which path produced it are both recorded in the run record as `size_min` / `size_min_source`. |
| `size_max_factor` | far-field target cell size / D. |
| `growth_rate` | target cell-to-cell growth ratio away from the wall. `dist_max` is **derived** from it as `(size_max - size_min) / growth_rate`, because for a linear size law `h(d) = h₀ + m·d` the growth ratio at the wall is exactly `m`. Setting `dist_max` independently (as `NACA0012.geo` does, at 2.0 on a unit chord) implies ~50% growth per cell. 0.10–0.20 is the usual range. |
| `farfield_factor` | outer radius / D. **Stand-in:** M4 says this should be keyed to the shock-standoff correlation; 20 is a fixed value that was adequate for the cases run so far. |
| `thickness_factor` | extrusion span / D. Governs near-wall element quality, because the one-cell extrusion makes tets anisotropic when the span differs greatly from the wall spacing. Measured: at `1.0` the aspect ratio is median 29 / max 288; at `0.05` it is median 7.8 / max 28.5. |
| `extrude_axis` | `z` (shape in x–y) or `y` (shape in x–z). **`z` is required for AERO-F**: the freestream is `u = V cos α cos β`, `v = V cos α sin β`, `w = V sin α` (`DistTimeState.C:321-323`), so with `Alpha = 0` the flow lies in x–y and `w = 0`, meaning nothing crosses the side planes. `y` reproduces `NACA0012.geo`'s orientation instead and would send the flow through them. |
| `algorithm` | gmsh 2D meshing algorithm; 6 = Frontal-Delaunay. |
| `optimize` | run gmsh's mesh optimiser. |

## `partition`

| field | meaning |
|---|---|
| `nparts` | subdomains for `mpmetis` → `sower`. Without a decomposition sower produces a single subdomain regardless of `-cpu`. |
