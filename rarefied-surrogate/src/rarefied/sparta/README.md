# SPARTA custom interface — sync status and scope note

This directory (`SpartaDistributionGenerator/`) holds a copy of a custom SPARTA
interface tool. A related copy, `SpartaAeroInterface/`, lives in the parent
`DSMC-GPR` repo, outside this subtree. Per PIPELINE.md ("exists and works. Wrap +
document; do not rewrite.") this note documents the sync status between the two
copies and the config schema — it does not change any C++ code.

## File-by-file sync status

- **`SpartaAeroInterface.cpp` / `.hpp`** — byte-identical between the two copies.
  In sync; no action needed.
- **`sparta_utils.cpp` / `.hpp`** — differ only trivially: an include-path fix for
  this subtree's extra directory nesting, plus one small added wrapper
  (`execute_command`) in this copy. Effectively the same tool; no action needed.
- **`main.cpp` — substantially diverged (~619-line diff) into two different
  tools:**
  - **This subtree's `SpartaDistributionGenerator/main.cpp`** is a **batch
    multi-shape VDF-distribution generator**: `BoundingBox`/`MicroCircle` structs,
    `surface_folder`/`surface_stem`/`num_samples`, randomized `mach_min`/`mach_max`,
    `t0_min`/`t0_max`. Its `config_default.txt` (`run_name noisy_debris_1`,
    `num_samples 10`, `mach_min/max`, `t0_min/max`) matches a batch-random-shape
    workflow — plausibly (not confirmed) the tool behind the 100-shape dataset
    registered as `shape_dataset_2026_02_06_09_28_35` in `data/registry.yaml`
    (Phase 5), though the Python shape generation
    (`scripts/shapes/shape_random.py`, parent repo) and the SPARTA VDF-sampling
    step are separate pipeline stages and this link is not independently verified.
  - **The parent repo's `SpartaAeroInterface/main.cpp`** is a **single-shape,
    single-run tool**: fixed `surf_file`, mesh-refinement-level params
    (`x_level_1_ref`...`z_level_4_ref`), `u_inf`/`t_inf`/`rho_inf`. Its
    `config.txt` has `run_name KN_eq_10p0_3D`, `sim_mode FT`, `seed 12345`,
    `dim 2` — naming that matches the argon-cylinder Kn-sweep pickles
    (`KN_eq_{0.01,0.1,1.0,5.0,10.0}_data.pkl`, registered in
    `data/registry.yaml`) that this project's moments/reconstruction
    characterization (`tests/characterization/kn_eq_1p0_reference.npz`) is built
    on.

## Open scope question (flagged, not resolved here)

This subtree's `README.md` states the sibling direct-coupling project in the
parent repo is out of scope: *"it does not import from, and should not modify,
the direct-coupling code"* / *"work only within this subtree."* But the evidence
above suggests the tool that most plausibly produced **this project's own**
validation-case VDF data (the argon-cylinder Kn sweep) is
`SpartaAeroInterface/main.cpp`, which lives outside this subtree's directory
boundary.

**This is flagged, not resolved, per the project's own rule against guessing on
canonical-vs-superseded questions.** Possibilities, none assumed:
1. `SpartaAeroInterface` (parent) should be treated as in-scope for this project
   specifically for the argon-cylinder case, despite its physical location, because
   the argon-cylinder validation case may be genuinely shared between the
   direct-coupling and surrogate projects.
2. The argon-cylinder pickles were produced by some other configuration/run not
   captured by the one `config.txt` found locally, and the naming match is
   coincidental or from an unrelated exploratory run.
3. Something else not yet identified.

**Do not backport `SpartaAeroInterface/main.cpp` into this subtree, and do not
treat either tool as superseding the other, until this is confirmed.**

Related parent-only files not present in this subtree's copy: `config.txt`,
`coupled_sparta_input`, `modified_from_dist_gen` (a SPARTA input-deck template
referenced by `config.txt`'s `sparta_input_file` field). Whether this subtree
needs its own copies of these depends on the scope question above.

## Config schema (partial Contract B documentation)

Fields observed directly in each copy's config file (PIPELINE.md Contract B is
still marked `FILL IN` for the rest — this covers only what's visible here):

**This subtree's `SpartaDistributionGenerator/config_default.txt`** (batch
generator):
| field | meaning (as named; semantics not independently verified beyond the name) |
|---|---|
| `sparta_input_file` | SPARTA input-deck template to use (`vdf_generator`) |
| `run_name` | run/output label |
| `data` | output data prefix (`out`) |
| `surface_folder`, `surface_stem` | where per-shape surface files live and their filename stem |
| `num_samples` | number of random shapes/conditions to sample per batch |
| `mach_min`, `mach_max` | freestream Mach number sampling range |
| `t0_min`, `t0_max` | freestream temperature sampling range |
| `to_paraview` | whether to emit Paraview-compatible output |
| `dim` | simulation dimensionality (2) |
| `x_min/max`, `y_min/max`, `z_min/max` | domain bounding box |

**Parent repo's `SpartaAeroInterface/config.txt`** (single-run tool):
| field | meaning (as named; semantics not independently verified beyond the name) |
|---|---|
| `run_name`, `sparta_input_file`, `sparta_struct_file`, `data_prefix` | run identity and I/O naming |
| `sim_mode` | simulation mode (`FT`) |
| `u_inf`, `t_inf`, `rho_inf` | freestream velocity/temperature/density (all `0` in this snapshot — possibly overridden per-case) |
| `seed` | RNG seed |
| `dim` | simulation dimensionality (2) |
| `x/y/z_min/max` | domain bounding box |
| `x/y/z_level_{1-4}_ref` | mesh-refinement level flags per axis (all `0` in this snapshot) |
| `surf_ref`, `surf_ref_level` | surface mesh refinement toggle/level |
| `dt`, `predata_timesteps`, `total_timesteps` | timestep size and run length |
| `do_grid`/`do_surf`/`do_therm`/`do_2D`/`do_3D` + matching `Nevery_*`/`Nrepeat_*`/`Nfreq_*` | which diagnostics to collect and their SPARTA time-averaging windows |
| `lo_2D`/`hi_2D`/`nbin_2D`, `lo_3D`/`hi_3D`/`nbin_3D` | velocity-space histogram grid bounds/resolution — this is what produces the `bin_coords`/`joint_pmf` structure `rarefied.io.histograms.read_all_timesteps_histogram_3D`/`_2D` parse |

## Not otherwise analyzed

`main_legacy_1D_separable.cpp`, `mainlegacy.cpp`, `compile.sh`, `makefile`, and the
compiled `main` binary are present in this subtree's copy and were not further
inspected in this pass — out of scope for this cleanup round.
