# Beam-coupling example (1D AERO-S beam <-> 2D SPARTA surface)

A small, self-contained, working example of the 1D-beam coupling path added
alongside the existing 2D<->2D (`DSMCSTRUC`/`SpartaExchanger`) coupling. It
couples a thin rectangular SPARTA outline (representing the swept
thickness-*t* cross-section of a body) to a straight 6-node AERO-S beam
running along its centerline - the two meshes do **not** share nodes, unlike
the original coupling path.

This example was verified end-to-end on Sherlock: it completes a short
coupled run cleanly (no errors/NaNs) and produces a beam that deflects
smoothly under real DSMC-computed surface forces.

## What's new here vs. the existing coupling

- **AERO-S side**: `Hetero.d/SpartaBeamExchanger.{h,C}` - reuses
  `EulerBeam::getFlLoad`/`computeDisp` (the existing rotating-offset,
  moment-generating conservative transfer) driven by a per-SPARTA-node
  `InterpPoint` built from the mapping file below. Selected via a new
  `DSMCBEAM "<mapping file>"` line in the AERO-S `StructureFile` (parallel
  to, and mutually exclusive with, the existing `DSMCSTRUC` line).
- **SPARTA side**: `src/fix_AERO_S_beam.{h,cpp}` - a new, standalone
  `fix AERO_S_beam` (parallel to, and never combined in the same run with,
  the existing `fix AERO_S`). Force still flows per-SPARTA-*line* (px, py),
  exactly like the existing fix; AERO-S lumps each SPARTA node's two
  adjacent lines' forces before calling `getFlLoad`. Position/velocity come
  back per-SPARTA-*node*.
- **Python**: `beam_projection.py` / `beam_export.py` (in this directory's
  parent) - point-to-segment projection of the SPARTA outline onto the beam
  centerline, and the writers for the beam struct/mapping files.

Neither change touches a single line of the existing `SpartaExchanger`,
`fix_AERO_S.{h,cpp}`, or `mesh_export.py` - this is a fully additive path,
always run as its own separate MPMD job (never combined with an existing
`DSMCSTRUC` coupling in the same AERO-S process; AERO-S will refuse to start
if both flags are set).

## Directory contents

| File | Role |
|---|---|
| `beam_coupling_example_sparta.txt` | The SPARTA boundary outline (a thin rectangle, half-width 0.1, half-span 0.5), same format as any other `{title}_sparta.txt`. |
| `beam_coupling_example_beam_struct.txt` | AERO-S beam mesh: 6 nodes, 5 `EulerBeam` (type 6) elements, cantilevered at node 1, out-of-plane DOFs pinned everywhere. |
| `beam_coupling_example_beam_mapping.txt` | The new 3-section per-node mapping file (`NODE_PROJECTIONS` / `SPARTA_LINE_TOPOLOGY` / `NODE_ADJACENT_ELEMENTS`) consumed by `SpartaBeamExchanger` and `fix_AERO_S_beam`. |
| `beam_coupling_example_beam_summary.txt` | Human-readable projection stats (xi range, gap magnitudes, clamped-node count). |
| `StructureFile` | AERO-S input deck - `DSMCBEAM`, linear `DYNAMICS`. |
| `coupled_sparta_input` | SPARTA input script - `fix AERO_S_beam`, a real (small) rarefied-air flow. |
| `config.txt` | Read by `SpartaAeroInterface/main` at launch; `sparta_input_file` points at `coupled_sparta_input`. Several numeric fields (`dt`, `total_timesteps`, `u_inf`, ...) are present but not actually consulted by this deck - `coupled_sparta_input`'s own `timestep`/`run` commands are authoritative. Present only because `SpartaAeroInterface` requires the keys to exist. |
| `air.species`, `air.vss` | Argon/air species data, copied so this example has no dependency on files outside this directory. |
| `run.sh` | Slurm job that launches the coupled run. |
| `template_beam_example.py` (parent dir) | The script that generated the 4 files above with a `_beam_`/`_sparta.txt` suffix - re-run it to regenerate or modify the geometry/material. |

## Regenerating the mesh/mapping files

```
cd ..   # mesh_pipeline/
python template_beam_example.py
```

Needs `numpy` (no `gmsh` dependency - the outline is built directly). This
overwrites the 4 generated files in this directory; `StructureFile`,
`coupled_sparta_input`, `config.txt`, `run.sh` do not need to change unless
you rename the `TITLE`/`OUT_DIR` in the script.

## Running it

Prerequisites: AERO-S built with `Hetero.d/SpartaBeamExchanger.{h,C}`
(`Element.d/Beam.d/EulerBeam` element type 6), and SPARTA's shared library
(`libsparta_mpi.so`) built with `src/fix_AERO_S_beam.{h,cpp}` - both are
picked up automatically by their respective build systems once the files
exist in `Hetero.d/`/`src/` (SPARTA's build auto-regenerates `style_fix.h`;
AERO-S's `Hetero.d/CMakeLists.txt` lists `SpartaBeamExchanger.C` explicitly).

`EXAMPLE/` and `results/` (AERO-S/SPARTA output directories) must exist
before running - `results/` is covered by this repo's top-level
`.gitignore` (`results/` is excluded as "large output data"), so a fresh
clone won't have it:

```
mkdir -p EXAMPLE results
```

```
sbatch run.sh
```

or, from an interactive allocation:

```
bash run.sh
```

It requests 9 ranks (1 AERO-S + 8 SPARTA) and ~10 minutes; the actual run
(500 SPARTA steps, 5 structural coupling exchanges) takes well under a
minute once particles/grid setup is done.

## What to expect

- AERO-S prints `Initializing SpartaBeamExchanger for 1D-beam structural
  coupling`; SPARTA prints `Initializing SPARTA with AERO-S: 1D-beam
  structural coupling`.
- `results/gdisplac` accumulates the beam's nodal displacement history. The
  root node (node 1) stays exactly `0` (cantilever constraint); the free
  nodes start at `0` and grow smoothly and monotonically once the first
  coupling exchange lands (step 100/500), as the beam responds to the real
  (small, rarefied) DSMC-computed surface force. Values are small (this
  example's material/flow parameters were picked for a quick, bounded,
  visibly-nonzero response, not physical realism) but should never be NaN
  or diverge within this short run.
- SPARTA's own stats/timing summary prints normally; exit code `0`.

**If AERO-S hangs instead of exiting cleanly**: check that
`StructureFile`'s `DYNAMICS TIME <dt> <dt> <total>` gives
`total/dt` exactly equal to `(SPARTA's run length) / (fix AERO_S_beam's
nevery_structural)` - AERO-S calls the coupling exchange every one of its
own timesteps and simply blocks until SPARTA's next matching collective
call, so a mismatch here (more AERO-S steps than SPARTA will ever provide)
is the one way to make it hang rather than error out.

## Known v1 limitations

- Only AERO-S element type 6 (`EulerBeam`) is supported as the coupled
  element - `TimoshenkoBeam` (type 7) has no `getFlLoad`/`computeDisp`
  override and would silently transfer zero force if used here.
- The force wire protocol assumes a 2D SPARTA run (2 components/line, px
  and py).
- This coupling path and the existing `DSMCSTRUC` path are mutually
  exclusive within one AERO-S run/process (by design - always launch as a
  separate job).

## Mapping-file schema (for reference)

```
# SPARTA_BEAM_MAPPING v1
# SECTION NODE_PROJECTIONS
# sparta_node_id  beam_elem  xi  gap_x  gap_y  active_flag
...
# SECTION SPARTA_LINE_TOPOLOGY
# sparta_elem  sparta_node1  sparta_node2
...
# SECTION NODE_ADJACENT_ELEMENTS
# sparta_node_id  sparta_elem_prev  sparta_elem_next
...
```

`beam_elem` is 0-based (matches the `TOPO` element index in
`*_beam_struct.txt`); `xi` is the natural coordinate (0..1) along that beam
element; `gap_x`/`gap_y` are the fixed reference-configuration offset from
the beam axis to that SPARTA node (this is what `EulerBeam::getFlLoad`/
`computeDisp` rotate with the beam's current frame to get the moment
channel - see `Hetero.d/SpartaBeamExchanger.C`).
