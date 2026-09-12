# Handoff: AERO-F mesh generation and the Case2 smoke test

Stage [4a]-[4b] on the cluster, as built and verified 2026-09-12.
Written so a fresh session can operate this without rediscovering the traps.

> **Scope warning.** The Case2 run below is a **smoke test of the solver
> chain**, not this project's physics. It uses the FRG tutorial's *air*
> (gamma 1.4, R 287.1, Pr 0.72) with a *turbulent* closure
> (`TurbulenceClosure Type = TurbulenceModel`, `Integration = WallFunction`)
> at Re 4.5e5 -- all three contrary to M3/M4's laminar-argon requirement.
> For the real case see `argon_cylinder_cfd.md`.

---

### Mesh pipeline (stage [4a])

Curve → gmsh mesh → AERO-F `.top` → partition → sower, on Sherlock.

The output of a case is the complete file set AERO-F reads, plus artefacts for
visual inspection and a JSON record of every metric and parameter.

### Quick start

Never on the login node — gmsh and sower are real compute.

```bash
cd ~/DSMC-GPR/rarefied-surrogate
srun -p dev -c 4 --mem=16G -t 00:30:00 --pty bash
source env.sh

python3 scripts/gen_mesh.py --case config/cases/circle_1m.yaml   # reference case
python3 scripts/gen_mesh.py --shape bezier --seed 3              # one random shape
python3 scripts/gen_mesh_batch.py --n 24 --run-name pilot --no-vtu --contact-sheet
sbatch --export=ALL,RUN_NAME=set1,N_PER_SHARD=25 scripts/gen_mesh_batch.sbatch
```

Explicit flags override the case file. Parameter meanings: `config/schema.md`.

Code lives here (in git); all data goes to `$MESH_RUN_ROOT`, default
`$SCRATCH/shape_meshes/<name>/`. **`$SCRATCH` is purged after 90 days without
content modification** — meshes are cheap to regenerate, datasets are not, so
move anything worth keeping to `$GROUP_HOME`. There is no `$OAK` on this
account, and `$HOME` is at 71% of its 15 GB, so nothing is written there.

### The environment has two incompatible halves

`gmsh/4.10.1` forces `gcc/10.1.0`, which breaks the py312 numpy stack (built
against `gcc/12.4.0`). So **gmsh is never imported into the Python process**:
`geo.py` writes a `.geo` and invokes the gmsh CLI through `gmsh_env.sh`, which
loads its own modules. `env.sh` sets up the Python half only.

The gmsh pip wheel is not an option either — it needs `libGLU.so.1`, which is
not present anywhere on this cluster. That is also why `shapes_bridge.py`
stubs out `pygmsh` (it depends on the gmsh Python module) rather than
installing it.

Also note: on this cluster `ml <category> <module>` in a single call silently
fails. The category must be loaded in its own `ml` call first. `env.sh` does.

### Pipeline

| Module | Role |
|---|---|
| `curves.py` | closed 2D curves: analytic circle, `Shape` adapter, arc-length resample, validation, point-to-curve distance |
| `geo.py` | writes the `.geo` and drives the gmsh CLI |
| `mshio.py` | MSH 2.2 reader, element quality, VTU writer |
| `quality.py` | the mesh gate: hard `problems` vs advisory `notes` |
| `topfile.py` | drives `gmsh2top`, cross-checks the `.top` against the mesh |
| `partition.py` | `mpmetis` → `.dec` → `sower`, plus partition coherence metrics |
| `quicklook.py` | triage PNG, volume/surface VTUs |
| `pipeline.py` | one case end to end, gated, with a JSON record |
| `scripts/gen_mesh.py` / `scripts/gen_mesh_batch.py` | drivers |
| `shapes_bridge.py` | loads `Shape` from `$SHAPES_DIR` without its mesher deps |

`$SHAPES_DIR` (default `~/shapes`) stays a **pristine clone** of
`jviquerat/shapes`; nothing here modifies it, and its commit hash is recorded
in every run record.

### Geometry conventions

- curves are **clockwise**, station 0 on the +x ray from the centroid (M13)
- shape in the **x–y plane**, extruded one layer in **+z** (`extrude_axis: z`).
  This is required, not cosmetic: AERO-F's freestream is
  `u = V cos α cos β`, `v = V cos α sin β`, `w = V sin α`
  (`DistTimeState.C:321-323`), so with `Alpha = 0` the flow lies in x–y and
  `w = 0` — nothing crosses the side planes. It also matches the short-course
  mesh (`2DNaca.top`: x–y with z ∈ [0, 0.05]). `extrude_axis: y` reproduces
  `NACA0012.geo`'s x–z orientation instead, which would send the flow *through*
  the side planes.
- outer boundary a single circle, all one physical group
- **no `Recombine`**, so extrusion of a triangulated surface yields
  tetrahedra — required, because `gmsh2top` accepts only element types 2
  (triangle) and 4 (tetrahedron)
- written as **MSH 2.2**, because `gmsh2top` cannot read 4.1 and its
  version-4 branch was written against 4.0

Every length is expressed relative to the shape's characteristic diameter `D`
(the diameter of the circle of equal area), so the sizing rule is
scale-invariant across a dataset — which is what keeps the discretisation bias
consistent from shape to shape (M4's base-consistency requirement).
Defaults: farfield `20 D`, extrusion `0.05 D`, far-field cell size `1 D`, wall
cell size from `size_min_factor` (falling back to perimeter / `n_surf` until
M4's convergence study supplies a value).

`Transfinite Curve {...} = 2` pins the wall nodes to exactly the `n_surf`
input curve points on each of the two extruded planes. For the 1 m circle they
land at r = 0.5 to within 8e-17. Those same points are the **station** set
(M6) — `geo.stations()` subsamples them, so every station id is an exact mesh
node and stage [4c] needs no interpolation.

#### Size grading

`DistMax` is **derived, not set**. For a linear size law `h(d) = h₀ + m·d`, the
cell-to-cell growth ratio at the wall is exactly `m`, so

    dist_max = (size_max - size_min) / growth_rate

with `growth_rate = 0.15` by default. Setting `DistMax` independently — as
`NACA0012.geo` does, at 2.0 on a unit chord — implies a growth rate near 0.5,
i.e. 50% per cell. `quality.py` measures the realised slope and reports it
next to the requested one (0.143 vs 0.150 on the circle).

### Physical groups and why they are named that way

| Group | sower code | AERO-F meaning |
|---|---|---|
| `StickMoving_1` | −3 | `BC_ADIABATIC_WALL_MOVING` — the shape surface |
| `InletFixed_2` | 4 | `BC_INLET_FIXED` — farfield |
| `SlipFixed_3` | 2 | `BC_SLIP_WALL_FIXED` — side plane at z = 0 |
| `SlipFixed_4` | 2 | `BC_SLIP_WALL_FIXED` — side plane at z = thickness |
| `FluidMesh` | volume | the tets |

Three deliberate choices:

**Trailing `_N` is mandatory.** sower reads the surface id from the text after
the last underscore in the element-set name (`FluidDomain.C:202-217`). Plain
`StickMoving` yields id 0 and cannot be targeted by an AERO-F `SurfaceData`
block. That matters because `StickMoving` is an *adiabatic* wall
(`BcDef.h:23`) and **an adiabatic wall has zero heat flux by definition** —
only a `SurfaceData { Type = Isothermal; Temperature = ...; }` override
promotes it (`SubDomainCore.C:2914`). Note `Symmetry_1`/`Symmetry_2` in
`NACA0012.geo` pick up ids 1 and 2 by accident, which would collide.

**`SlipFixed`, not `Symmetry`.** sower maps names beginning `Symmetry` to code
6 (`FluidDomain.C:63`), but AERO-F defines 6 as `BC_POROUS_WALL_FIXED` and 11
as `BC_SYMMETRY` (`BcDef.h`, identical across `aero-f`, `aero-f-2` and
`aero-f-lite`), with no remap on the read path. In `FluxFcn.h` codes 2 and 11
share a case label while 6 is handled separately. Code 2 is unambiguous and is
physically identical to a symmetry plane for a 2D-equivalent run.

**`StickMoving`, not `StickFixed`.** The moving-wall variant means no mesh
change is needed when FSI arrives.

### Verification

`quality.py` gates on: all five groups present; each group's area matching its
analytic value; total volume; **no non-positive tet volumes**; wall node count
exactly `2 × n_surf`; wall nodes lying on the input curve; wall nodes on
exactly two y-planes.

The area checks do double duty — they are the only thing that verifies
`geo.py`'s assumption about the order in which gmsh's `Extrude` returns
lateral surfaces. If that ever changes, the wall and farfield areas swap and
the check fails loudly instead of silently putting the boundary conditions on
the wrong surfaces.

`partition.py` reports an **interface node fraction**. Any permutation of
element ids is a legal partition, so sower would accept a scrambled
decomposition silently; coherence is the only signal. Measured: 0.046 for
metis vs 0.999 for a deliberately scrambled control.

#### On `.dec` element ordering

The format is undocumented; it was read off sower's parser
(`Decomp3D.C:13-77`) and mpmetis's writer (`io.c:491-515`):

```
Decomposition ...        <- header, matched on its first 13 chars
<numSub>
<n_elems_in_sub_0>
<e>                      <- 1-based element ids, tetrahedra only
...
```

Ordering is **preserved**, despite appearances. sower reads elements onto a
*stack* and stores them at `elements[numElem-i-1]` (`FluidDomain.C:339-345`);
the pop reverses the read order and the descending index reverses it again, so
`elements[0]` is the first tet in the `.top` file. Confirmed empirically by the
coherence metric above.

The `mpmetis` at `$FRG_BIN` is **not stock METIS** — it is patched to read a
`.top` file directly (filtering on element type 5) and to write sower's
decomposition format itself as `<top>.dec.<nparts>`.

### Outputs per case

| File | For |
|---|---|
| `<name>.geo`, `<name>.msh` | geometry and mesh; open the `.msh` in gmsh |
| `<name>.top` | AERO-F mesh, plain text |
| `<name>.top.dec.N` | decomposition |
| `<name>.msh1`, `.dec1`, `.con`, `.Ncpu` | sower output — what AERO-F reads |
| `<name>.png` | 3-panel triage render (full domain / near field / wall) |
| `<name>_volume.vtu` | tets, coloured by `subdomain`, `tet_volume`, `aspect_ratio` |
| `<name>_surface.vtu` | boundary tris coloured by `group` (legend in the record) |
| `<name>.quality.txt` | human-readable gate report |
| `<name>.record.json` | every metric, timing and parameter |
| `manifest.jsonl` | one line per batch attempt, successes and rejections |

### Reference result — 1 m circle, `n_surf` 300, 4 subdomains

`python3 scripts/gen_mesh.py --case config/cases/circle_1m.yaml`

```
nodes 10570   tets 30426   boundary tris 21140

group                  tris           area       expected   rel err
StickMoving_1           600       0.157077       0.157077   -0.000%
InletFixed_2            256        6.28255        6.28319   -0.010%
SlipFixed_3           10142        1255.35        1255.85   -0.040%
SlipFixed_4           10142        1255.35        1255.85   -0.040%

volume 62.7674   expected 62.7926   rel err -0.0402%
tet volume min 5.42e-07  max 0.00981  non-positive 0
aspect ratio  min 3.26  median 7.79  p99 24.1  max 28.5
wall nodes 600 (expected 600)  max deviation from curve 7.85e-17  extrude planes [0.0, 0.05]
size at wall 0.0109 (requested 0.01047)   growth measured 0.143 (requested 0.150)

partition        [7616, 7623, 7565, 7622]  balance 1.0022  interface nodes 507 (4.80%)
sower bc codes   StickMoving_1 -3   InletFixed_2 4   SlipFixed_3 2   SlipFixed_4 2
stations         30 (indices 0, 10, 20, ...)
total 3.6 s
```

The `-0.000%` wall area is exact, and worth understanding: 0.157077 is the
300-gon's perimeter (3.14154, **not** π) times the extrusion thickness (0.05).
Comparing against `π·D·t` instead would hide a real 2e-5 discrepancy, because
the mesh represents the polygon, not the circle it approximates.

Frozen as `tests/characterization/mesh_reference.npz` by
`tests/characterization/build_mesh_reference.py` — the same freeze-then-diff
pattern used for the moments relocation. Note the `.npz` is gitignored and
rebuilt on demand, like the existing references.

### Known limitations

- **Near-wall aspect ratio is set by `thickness_factor`**, and it matters. The
  one-cell extrusion makes tets anisotropic whenever the extrusion span differs
  greatly from the wall spacing. Measured on the circle at `n_surf = 300`:

  | `thickness_factor` | AR median | AR max |
  |---|---|---|
  | 1.00 D (`NACA0012.geo`-like) | 28.7 | 288 |
  | **0.05 D (default, short-course-like)** | **7.8** | **28.5** |

  `0.05` is now the default for this reason. Reported, not gated.
- **`size_min` is still the fallback.** M4 requires it from a wall-gradient
  convergence study that has not been run, so the code uses
  `perimeter/n_surf`. The run record states which path was taken
  (`size_min_source`), so this can never be mistaken for a converged value.
- **The farfield is one group.** For supersonic runs you may want the
  downstream portion as `OutletFixed`. That is a predicate on the outer
  circle's angle in `geo.py` — a mesh change, not an input-file change.
  (The short-course mesh does split it: `OutletFixedSurface_4` /
  `InletFixedSurface_5`.)
- **M13's station-origin check costs ~25% of the shape population.** Measured
  on a 6-shape pilot: 2 of 8 attempts rejected with
  `+x ray from the centroid crosses the surface 3 times`, i.e. 75% yield
  (it was 100% before the check existed). Those are the non-star-convex
  shapes — the rejection is intended, and consistent with M10's convex-body
  scope, but budget for it when sizing a dataset run. The batch driver's
  reject/retry loop handles it automatically and logs the reason per attempt
  in `manifest.jsonl`.
- **Sharp cusps are still not filtered.** `radius`/`edgy` uniform on [0,1]
  with 3–7 control points produces near-degenerate spikes that the +x-ray
  check does *not* catch — it tests star-convexity about the centroid, not
  curvature. A minimum-curvature-radius gate in `curves.validate` is the
  natural place; the threshold is a modeling choice, still open.
- **The BC codes are verified as far as sower and one AERO-F run** — the
  Case2 section below exercised them end to end, but with the tutorial's
  physics, not this project's.


---

### AERO-F Case2 on the 1 m circle (stage [4b])

Reproduces `$GROUP_HOME/FRG_ShortCourses/FolderCases1To4` Case2
(`SteadyViscousPG`) with a pipeline-generated circle instead of `2DNaca.top`.
The FluidFile is **byte-identical to the tutorial's apart from its geometry
comment** — the directory layout was mirrored so nothing else had to change.

```bash
srun -p dev -c 4 --mem=16G -t 00:30:00 --pty bash   # or an existing allocation
cd rarefied-surrogate/aerof_case2
scripts/prep_aerof_case.sh          # mesh + cd2tet + sower  (~20 s)
scripts/run_aerof.sh           # mpirun -np 4 aerof2    (~88 s)
scripts/postpro_aerof.sh       # sower -merge + xp2exo
```

Result: exit 0, 1000 iterations in **~88 s** on 4 ranks, residual
1.0 → **3.743647e-07**. Bit-for-bit reproducible across two
independent from-scratch runs.

> This number changed from an earlier `7.031645e-06` when the M13 clockwise
> flip reordered the curve points: gmsh's triangulation shifted slightly
> (10570 nodes / 30426 tets, previously 10576 / 30444), so the discrete
> solution differs. The mesh gate passes identically either way — the residual
> is a property of the specific mesh, not a regression.

### Mesh differences from the pipeline default

Two parameters were changed to match the tutorial mesh, and both are
improvements worth keeping:

**`--extrude-axis z`** (shape in x–y). This is not cosmetic. AERO-F builds the
freestream as `u = V cos α cos β`, `v = V cos α sin β`, `w = V sin α`
(`DistTimeState.C:321-323`), so Case2's `Alpha = 0, Beta = 5` puts the flow in
the **x–y** plane. `2DNaca.top` is x–y with z ∈ [0, 0.05]; `NACA0012.geo` is
x–z, which would have sent the flow through the side planes. Confirmed correct
by the run: `Lz / |F| = 2e-6`.

**`--thickness-factor 0.05`** (tutorial uses z ∈ [0, 0.05] on a unit chord).
This also fixes the aspect-ratio concern from the mesh work — element quality
improves sharply, because the extrusion span is now comparable to the
near-wall spacing rather than to the diameter:

| thickness | aspect ratio median | max |
|---|---|---|
| 1.00 D | 28.7 | 288 |
| 0.05 D | **7.8** | **28.5** |

`z` is now the pipeline default; see `config/schema.md`.

### Environment gotchas

- **`cmake/3.8.1` no longer exists** on Sherlock (only 3.11.1+), and Lmod
  aborts the entire `module load` if any member is missing — so the tutorial's
  module line silently loads nothing. cmake is build-time only; dropping it
  leaves `gcc/9.1.0 openmpi/4.0.3 imkl/2019`, which resolves every `aerof2`
  shared library.
- **Load the module category in its own `ml` call first.** `ml devel math` then
  `ml gcc/9.1.0 ...`; combining them silently loads nothing.
- **Never pipe `ml` into another command.** It is a shell function that `eval`s
  its output; a pipe runs it in a subshell and the environment change is lost.
  This cost me two debugging rounds.
- **Create `postpro/` before running sower `-merge`.** It reports success and
  writes nothing if the output directory is missing.

### Preprocessing substitutions

- `partnmesh` → `mpmetis`. Both write `<top>.dec.N` in sower's format; the
  pipeline already produces this file, so `prep.sh` reuses it.
- `cd2tet -mesh sources/circle.top -output sources/circle.sinus` is unchanged.
  The `.dwall` file is required by both `Integration = WallFunction` and the
  Spalart–Allmaras model, so it is not optional.

### Physics: read the result with care

The flow field is correct — stagnation at the nose, acceleration to M ≈ 0.87
over the shoulders, suction lobes top and bottom, a wake. And the drag is in
the right range:

```
T = 267.9 K   V = 131.3 m/s   q = 11200 Pa   Re = 4.5e5
|F| = 105.8 N   frontal area = D x t = 0.05 m^2
Cd = |F| / (q A) = 0.19        smooth cylinder above the drag crisis: ~0.2-0.4
```

**But there is a persistent side force.** The final force is
`(Lx, Ly) = (99.9, -34.8) N`, i.e. 19.2° below the x-axis, while the
freestream is at +5°. A circle is symmetric, so the force *must* be parallel to
the freestream; a 24° misalignment means a real, spurious side force. It is not
a transient — over the last 100 iterations `Lx` drifts 0.08% and `Ly` 3.6%,
settled around −35 N.

This is the setup, not the mesh. A cylinder at Re 4.5e5 sheds vortices; the
true flow is unsteady and has no steady solution. Forcing `Type = Steady`
makes the solver lock into one arbitrary asymmetric separation pattern. The
residual stalling at ~4e-7 rather than reaching the requested `Eps = 1e-8` is
the same symptom. Case2's NACA airfoil at 5° has attached flow and converges
cleanly; a bluff body does not.

So this is a valid **smoke test** — mesh, BCs, decomposition, wall distance and
solver all work end to end — but not a physically meaningful cylinder solution.
For that you would want `Type = Unsteady`, or a streamlined shape.

Two further caveats:

- `Surface = 1.0` in the FluidFile is the tutorial's reference area, not this
  geometry's `D x t = 0.05 m²`. The `Lx Ly Lz` columns are raw force
  components, so this only matters if you read the coefficients directly.
- **`Delta = 0.004`** is the wall-function offset inherited verbatim from
  Case2, but this mesh's near-wall spacing is `π/300 = 0.0105 m` — the first
  cell is ~2.6x larger than the assumed offset. The wall function is therefore
  being applied outside its intended regime. Harmless for a smoke test;
  it must be revisited before any quantitative viscous result.
- AERO-F reported `changed the orientation of 11004 boundary faces` (of 21152).
  It self-heals boundary face orientation just as it does tet orientation
  (`GeoSource.C:340` → `Elem::checkVolume`), so this is informational — but the
  pipeline could emit consistently outward-facing triangles and remove the
  warning.

### Files

Data lands in `$SCRATCH/aerof_case2/` (regenerate with `prep.sh`; `$SCRATCH`
purges after 90 days). `simulations/Case2/postpro/Case2.exo` opens in ParaView;
`solution.png` there is a quick Mach/pressure render.
