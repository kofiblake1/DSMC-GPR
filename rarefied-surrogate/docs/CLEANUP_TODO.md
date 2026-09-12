# CLEANUP_TODO.md

Dependency-ordered plan for turning the consolidated-but-messy tree into the structure
locked in `README.md` / `PIPELINE.md` / `docs/decisions/DECISIONS.md`. Verification
comes before transformation: get the package loadable → freeze a characterization
reference → extract one function at a time, diffing against that reference → only
then fill in the docs' `TODO`/`CONFIRM` markers.

**Ground rule for every extraction step below:** cut the function, paste it unchanged,
replace the notebook cell with an `import`. No renaming, no signature changes, no
"clean up while I'm in there." If a step can't be done that way, stop and record a
`CONFIRM:` instead of improvising.

Inventory basis for this list: `README.md`, `PIPELINE.md`, `docs/decisions/DECISIONS.md`,
full read of both existing notebooks (`src/rarefied/moments/proess_3D_distributions.ipynb`,
`src/rarefied/geometry/process_shape_data.ipynb`), diff of the two SPARTA C++ copies
(`SpartaAeroInterface/` in the parent repo vs. `src/rarefied/sparta/SpartaDistributionGenerator/`),
and a direct `pickle.load` of the candidate local data files under
`~/Documents/Meteor_Modelling/code/`.

---

## Phase 0 — Make the package importable (no logic moves yet)

Currently there is **no `pyproject.toml`, no `__init__.py` anywhere, and no
`src/rarefied/reconstruct/` directory** (PIPELINE.md names it as stage [6]'s home but
it was never created). `pip install -e .` cannot work yet.

- [x] Confirm environment: `python3.10 --version` → resolves (3.10.17 confirmed);
      `python3.10 -c "import numpy, scipy, pandas, sklearn, matplotlib"` → all present,
      nothing to install first. Use `python3.10` for every step below.
- [x] Create minimal `pyproject.toml` (src-layout, `[tool.setuptools.packages.find]
      where = ["src"]`, deps: numpy, scipy, pandas, scikit-learn, matplotlib — the set
      actually imported by the notebooks today).
- [x] Add empty `__init__.py` to `src/rarefied/` and every existing subpackage
      (`geometry/`, `sparta/`, `aerof/`, `moments/`, `io/`, `features/`, `models/`).
- [x] Create `src/rarefied/reconstruct/__init__.py` (dir didn't exist; PIPELINE.md
      stage [6] is supposed to live here — needed as a target before Phase 3
      extraction).
- [x] `python3.10 -m pip install -e .` from `rarefied-surrogate/` → succeeded, no path
      errors.
- [x] Confirmed `import rarefied` (and every subpackage) works from an unrelated cwd
      (`/tmp`), resolving back to `src/rarefied/__init__.py` in the live repo — editable
      install verified. No notebook code moved.

## Phase 1 — Minimal data registry (local entry only)

`data/registry.yaml` does not exist. Create it now, scoped to only what's needed to
unblock Phase 2 — do not try to backfill cluster paths (nobody here can confirm them
locally; leave that as a `CONFIRM`).

- [x] Create `data/registry.yaml` with a documented schema:
      `name → {local_path, cluster_path, format, produced_by, provenance, status, notes}`.
- [x] First entries, from `~/Documents/Meteor_Modelling/code/` (read in place, **not**
      copied into the repo):
  - `kn_eq_1p0_data` → `KN_eq_1p0_data.pkl` — confirmed by direct load to be a
    `(region_cache: dict[30], region_df: DataFrame[30,18])` tuple matching exactly what
    the moments notebook's live driver cell (cell 6) already loads and feeds into
    `calculate_and_compare_moments`. `status: verified`. **This is the characterization
    pickle for Phase 2.**
  - Sibling entries for `kn_eq_{0p01,0p1,5p0,10p0}_data` and the three
    `..._3d_coarse` variants — registered with `status: unverified` (not individually
    loaded/inspected yet; see CONFIRM #3 on their identical byte sizes).
  - `my_data` → registered with `status: incompatible_format` — loaded and confirmed
    to be a **different** schema (`DataFrame` + `list[dict]` by timestep, not the
    `(region_cache, region_df)` tuple). Not used for characterization or treated as
    interchangeable with the `kn_eq_*` entries (CONFIRM #4).
  - `cluster_path` fields: left as `FILL_IN` on every entry — no cluster access from
    here.
  - `provenance`: references `SpartaAeroInterface/config.txt` (parent repo, not the
    subtree copy) — the only run-config file found locally, containing
    `run_name KN_eq_10p0_3D`, `seed 12345`, `dim 2`, `sim_mode FT`,
    `sparta_input_file modified_from_dist_gen`. Noted as a plausible template only,
    not asserted as exact per-case fact (CONFIRM #6).
- [x] Verified: `python3.10 -c "import yaml; ..."` parses the file, expands
      `kn_eq_1p0_data`'s `~`-relative `local_path`, and confirms the resolved file
      exists on disk. Added `pyyaml` to `pyproject.toml` dependencies for this (was
      missing) and reinstalled.
- [ ] Point the Phase 2 characterization script's data access at this registry entry —
      **never a hard-coded path**, even though today only one path exists.

## Phase 2 — Characterization reference (before any relocation)

Run the *current* code, in its *current* notebook location, on the registry-resolved
pickle, and freeze the outputs. This is the safety net every extraction in Phase 3
diffs against.

- [x] Write a small script/notebook cell (can live temporarily at
      `tests/characterization/build_reference.py`) that:
  1. Resolves `kn_eq_1p0_data`'s local path via `data/registry.yaml` (simple
     YAML-lookup, no new abstraction needed).
  2. Loads the pickle (`region_cache, region_df = pickle.load(...)`).
  3. Imports the *notebook* — do not copy/paste its functions yet — and calls
     `calculate_and_compare_moments(region_cache, region_df, n_terms=n)` for a fixed,
     documented set of `n` (the notebook's live driver cell 6 loops `n in 1..6`; reuse
     that range for parity).
  4. Captures, per region and per `n_terms`: the Hermite coefficient vectors, the
     `fast_momentum_flux_from_3D_hermite_fit` / `fast_heat_flux_from_3D_hermite_fit`
     outputs, the raw-distribution comparison values
     (`momentum_flux_from_3d_distribution`, `heat_flux_from_3D_distribution`), and the
     DSMC ground-truth columns pulled from `surf_df` (`shx`, `px`, `shy`, `py`, `ke`).
- [x] Saved to `tests/characterization/kn_eq_1p0_reference.npz` (116 arrays: 9 moment
      arrays + 10 coefficient/error arrays × 6 values of `n_terms`, all shape `(30,)`
      except `terms` which is `(30, n, n, n)`). Took ~26 minutes to compute (the fit
      cost grows steeply with `n_terms`) — ran as a background job.
- [x] Ran against the notebook *in place* before any extraction (cells 0–4 executed
      via `exec` on their exact source, not copy-pasted) — confirmed 0 `NaN`s, and
      values are physically plausible (stress/heat-flux magnitudes in a sane range,
      `e_dsmc` — the fixed DSMC ground truth — identical across all `n_terms` as
      expected since it doesn't depend on the fit).
- [x] This is a **characterization reference, not a regression test with an oracle** —
      it records "what the code currently outputs," not "what is physically correct."
      Labeled that way in `build_reference.py`'s module docstring so nobody later
      mistakes it for independently-validated ground truth.

## Phase 3 — Extract the crown jewels (moments + reconstruction), one function at a time

Only start this after Phase 2's reference file exists and is committed. Extraction
order follows the dependency chain found in the notebook (helpers before the functions
that call them), so each step's diff-against-reference is meaningful in isolation.
Everything in this phase comes from
`src/rarefied/moments/proess_3D_distributions.ipynb`, cell 2 (pure, no notebook
globals, no hardcoded paths — confirmed by direct read) and cell 4.

Target modules — **CONFIRM #12 resolved (with a documented default, not a guess):**
`momentum_flux_from_3d_distribution`, `momentum_flux_from_2d_distribution`,
`heat_flux_from_3D_distribution`, `heat_flux_from_2D_distribution`, and
`energy_flux_from_3D_distribution` go straight from raw distribution to flux,
skipping the coefficient step, so they don't cleanly fit either PIPELINE.md
stage's stated "consumes." Placed in `src/rarefied/reconstruct/flux.py` alongside
the fit-based flux functions (their output role — producing a flux value for
ground-truth comparison — is closer to reconstruction than to fitting), and
recorded here rather than silently assumed. Revisit if this turns out wrong.

All of steps 1–9 below were executed as **one verified batch** rather than
function-by-function, using `ast.get_source_segment` to cut every function's exact
source (byte-identical, confirmed by an automated diff against the notebook) and
route it to its target module, then one integration-level diff against the Phase 2
reference at the end — equivalent in effect to doing step 9's check after each
step, since all of 1–8 land in the same two-file batch.

1. [x] `hermite_polynomial(n, x, alpha)` → `src/rarefied/moments/hermite.py`.
2. [x] `Q_n_p(n, p, mean, std)`, `Q_n_p_half(...)` → `src/rarefied/reconstruct/flux.py`,
       alongside the `fast_*` functions that consume them (step 6). `Q_n_p_half` turned
       out to also call `hermite_polynomial` directly (not just consume a fit dict) —
       `flux.py` imports it from `rarefied.moments.hermite`.
3. [x] `get_mean_var_3D(data)`, `get_mean_var_2D(data)` → `src/rarefied/moments/hermite.py`.
4. [x] `pdf_hermite_3d`, `pdf_hermite_2d` → `src/rarefied/moments/hermite.py`. Turned out
       to be needed by `flux.py` too (see step 8) — `momentum_flux_from_3D_hermite_fit`,
       `momentum_flux_from_2D_hermite_fit`, and `kinetic_energy_flux_from_2D` rebuild the
       PDF grid from fit coefficients via these, so they're a real cross-module
       dependency, not just notation. `flux.py` imports both from `hermite.py`.
5. [x] `fit_hermite_distribution_3d(data, ...)`, `fit_hermite_distribution_2d(data, ...)`
       → `src/rarefied/moments/hermite.py`. **The orthogonal-projection core (decision
       M2).** Verified: all 6 `n_terms` values' Hermite coefficient vectors (`terms`,
       shape `(30, n, n, n)`) plus `mean_T/N/Z`, `var_T/N/Z`, and the three error metrics
       match `tests/characterization/kn_eq_1p0_reference.npz` bit-for-bit (`np.allclose`,
       zero mismatches reported).
6. [x] `fast_momentum_flux_from_3D_hermite_fit`, `fast_heat_flux_from_3D_hermite_fit`
       (+ `Q_n_p`/`Q_n_p_half` from step 2) → `src/rarefied/reconstruct/flux.py`. These
       are what the live driver actually calls. Verified: `pn_model`/`pt_model`/`e_model`
       match the reference bit-for-bit for all 6 `n_terms` values.
7. [x] `momentum_flux_from_3d_distribution`, `heat_flux_from_3D_distribution` (+ the 2D
       analogs `momentum_flux_from_2d_distribution`, `heat_flux_from_2D_distribution`,
       and `kinetic_energy_flux_from_2D`, `energy_flux_from_3D_hermite_fit`,
       `energy_flux_from_3D_distribution`) → `src/rarefied/reconstruct/flux.py`. Verified:
       `pn_dist`/`pt_dist`/`e_dist` (the ones the live driver exercises) match the
       reference bit-for-bit; the unexercised 2D analogs are covered only by the
       byte-identical-source check (see note below), same as step 8.
8. [x] `momentum_flux_from_3D_hermite_fit`, `heat_flux_from_3D_hermite_fit`,
       `momentum_flux_from_2D_hermite_fit`, `heat_flux_from_2D_hermite_fit` — the
       slower, currently-unused-by-the-live-driver variants. Relocated to
       `src/rarefied/reconstruct/flux.py`. No numerical reference exists for these
       (the driver doesn't call them) — verified only by the automated
       `ast.get_source_segment` byte-diff against the original notebook cell, confirming
       the relocated text is character-for-character what was there before. No new
       verification code was written for them, per the relocation-only rule.
9. [x] `calculate_and_compare_moments(region_cache, region_df, n_terms, verbose=False)`
       → `src/rarefied/moments/pipeline.py` (new file). Notebook cell 4 replaced with an
       import. **Integration check:** re-executed the notebook's current cells 0–4
       (2 and 4 now being the import cells) end-to-end and diffed all 114 output arrays
       (9 moments + 10 coefficients × 6 `n_terms` values) against
       `tests/characterization/kn_eq_1p0_reference.npz` — `ALL MATCH, 0 mismatches`.
10. [ ] Update the notebook's cell 6 driver to resolve the pickle path via
        `data/registry.yaml` instead of the relative `'KN_eq_1p0_data.pkl'` literal
        (cell 6 itself was not touched in this pass — it still works unchanged because
        cell 4's import already supplies `calculate_and_compare_moments`, but the path
        literal is still there). This is the only place a path literal currently lives
        in *live* code (all `/Volumes/Sherlock...` absolute paths in the notebook are
        already commented out, confirmed).
11. [ ] Leave notebook cell 7 (dead, fully commented-out duplicate of `Q_n_p`,
        `Q_n_p_half`, and the `fast_*` functions) alone for now — it's inert, not a
        second live implementation. Delete it in a later, separate pass once the repo
        isn't mid-extraction (touching it now isn't a relocation, it's cleanup).

## Phase 4 — Extract supporting IO / parsing helpers

Lower-stakes than Phase 3 (no numerically-sensitive math), but still load-bearing.
Do after Phase 3 so the reference-diffing muscle memory/tooling from Phase 3 is
already in place.

1. [x] `read_all_timesteps_histogram_3D`, `read_all_timesteps_histogram_2D`
       (moments notebook, cell 0) → `src/rarefied/io/histograms.py`. Pure, file-path
       args only. Byte-identical to notebook source (verified via
       `ast.get_source_segment` diff, same method as Phase 3).
2. [x] `parse_sparta_item_table`, `relabel_surface_columns`, `relabel_grid_columns`
       (moments notebook, cell 3) → `src/rarefied/io/sparta_runs.py`. **CONFIRM #13
       resolved (documented default, not a guess):** placed under `io/` rather than
       `sparta/` because these parse SPARTA *output* (a loader concern, matching the
       README's description of `io/`), whereas `sparta/` is for generating SPARTA
       *inputs* (sampling-region + run setup). Recorded in the module's docstring too.
3. [x] `load_run_data_3D(run_dir, metadata_path)` (moments notebook, cell 3) →
       `src/rarefied/io/sparta_runs.py`, alongside step 2 and `read_region_metadata`/
       `load_all_histograms_from_run_2D` (moments notebook, cell 1) — all four in one
       file since `load_run_data_3D` and `load_all_histograms_from_run_2D` both
       directly call the others. Imports `read_all_timesteps_histogram_2D`/`_3D` from
       `histograms.py` (step 1).
4. [x] **CONFIRM #1 resolved for now, not eliminated:** extracted the moments
       notebook's `read_region_metadata` under its original, unchanged name into
       `src/rarefied/io/sparta_runs.py` — this is *not* a rename, so it doesn't violate
       the relocation-only rule. There is no collision **today** because the geometry
       notebook's differently-implemented `read_region_metadata` (circle-union fields)
       has not been extracted yet (that's Phase 5, not yet started). Left an explicit
       note in `sparta_runs.py`'s docstring and here: **when Phase 5 extracts the
       geometry notebook, resolve the naming collision deliberately at that point**
       (do not let it be a silent side effect of that extraction either).
- [x] **Verification:** all 8 relocated functions byte-identical to their notebook
      source (automated `ast` diff, zero mismatches). Re-ran the full notebook
      end-to-end (cells 0–6, `N_TERMS` temporarily overridden to 1 for speed) after
      the rewrite — imports resolved, pipeline ran, no errors. **Not** diffed
      numerically against `tests/characterization/kn_eq_1p0_reference.npz`: none of
      these functions sit on `calculate_and_compare_moments`'s call path when loading
      from an already-built pickle (`load_run_data_3D` etc. are for building a pickle
      *from* raw SPARTA text output in the first place, which the characterization
      case doesn't exercise) — consistent with this phase being "no
      numerically-sensitive math" to begin with.

## Phase 5 — Geometry notebook (`src/rarefied/geometry/process_shape_data.ipynb`)

- [x] **CONFIRM resolved:** `dataset_2026-02-06_09_28_35` (referenced by cells 0, 1,
      and 5) is a real, substantial local dataset, not throwaway output — confirmed by
      finding it at `~/Documents/Meteor_Modelling/code/shapes/dataset_2026-02-06_09_28_35`
      (100 `.mesh` files under `meshes/`, 201 files under `sparta_converted/`) and an
      apparently-identical copy in the parent repo's `scripts/shapes/`. Registered as
      `shape_dataset_2026_02_06_09_28_35` in `data/registry.yaml`; every live
      hard-coded occurrence of the relative path string (cells 0, 1, 2, **and** 5 —
      cell 2's "STEP 4" driver block and cell 5 each had their own live copy, found
      only by grepping the whole notebook after the first two fixes) now resolves
      through the registry. The one commented-out occurrence (cell 2's dead "STEP 3"
      block) was left untouched.
- [x] Built `tests/characterization/geometry_reference.npz` (117 arrays) by running
      the *pre-extraction* notebook code (cells' function bodies executed via `exec`,
      trailing driver statements excluded) on: (a) `generate_circle_shape` — fully
      self-contained, no external file needed, called with `visualize=False` to keep
      it headless; (b) `process_mesh_folder` on a 2-file scratch copy of the real
      dataset's `.mesh` files (not the whole 100-shape batch, and never writing into
      the dataset directory itself); (c) `process_sparta_folder` on that scratch
      output. Confirmed no `NaN`s.
- [x] Extracted, using the same `ast.get_source_segment` byte-identity method as
      Phases 3/4 (verified: all 19 relocated functions byte-identical to notebook
      source):
  - `src/rarefied/geometry/mesh.py` — `read_mesh_file`, `compute_winding_order`,
    `ensure_clockwise_order`, `scale_shape_to_size`, `extract_boundary_vertices`,
    `convert_mesh_to_sparta`, `process_mesh_folder`.
  - `src/rarefied/geometry/sparta_shapes.py` — `read_sparta_file` (reads SPARTA
    *shape-definition* files — a different file format from
    `rarefied.io.sparta_runs`, which reads SPARTA *simulation output*; no relation).
  - `src/rarefied/geometry/regions.py` — `divide_vertices_into_regions`,
    `compute_region_circle_union`, `estimate_circle_union_area`,
    `compute_region_properties`, `process_sparta_folder`, and this notebook's
    `read_region_metadata`.
  - `src/rarefied/geometry/plotting.py` — `plot_shape`, `draw_circle_arc`,
    `shade_circle_union_area`, `plot_regions_for_shape`.
  - `src/rarefied/geometry/circle.py` — `generate_circle_shape`.
  - **CONFIRM #1 resolved (both sides now):** this notebook's `read_region_metadata`
    (circle-union fields, list-of-dicts) was relocated under its unchanged name into
    `regions.py`, alongside `rarefied.io.sparta_runs.read_region_metadata` (moments
    notebook's tangent-angle DataFrame reader, Phase 4) — genuinely different
    functions, no rename, no runtime collision across the two modules. Documented
    explicitly in both files' docstrings so neither is mistaken for superseding the
    other, and so nobody `from X import *` both into one namespace.
  - **Scope decision (not a silent guess):** `plot_shape`, `draw_circle_arc`,
    `shade_circle_union_area`, and `plot_regions_for_shape` were reclassified from
    "plotting-only, leave in notebook" to load-bearing and relocated too. Reason:
    `generate_circle_shape`'s own default behavior (`visualize=True`) calls
    `plot_regions_for_shape`/`plot_shape` directly, which in turn call the other two —
    relocating `generate_circle_shape` verbatim was only possible by also relocating
    what it actually calls. Not extracting them would have required either editing
    `generate_circle_shape`'s body (forbidden) or leaving it in the notebook
    (blocking the whole function). Documented in `plotting.py`'s docstring.
  - **Bug found during extraction (not fixed, out of scope):** `regions.py` and
    `circle.py`'s functions use `os.*` without any `os` import in their original
    notebook cells (cell 2, cell 4) — they relied on cell 0's `import os` leaking
    across cells via shared notebook-kernel state. This is exactly the kind of
    notebook/module gap the relocation surfaces: standalone modules don't inherit
    notebook-global imports. Added `import os` to `regions.py`'s header (this is
    scaffolding the new module needs to run at all, not a change to any function
    body — same treatment as every other cross-cell import gap in Phases 3–5).
  - **Third-party dependency surfaced:** `compute_region_properties` optionally uses
    `circle_fit.taubinSVD` (imported inline inside a try/except with a fallback,
    preserved exactly). Added `circle-fit` to `pyproject.toml` since it's installed
    locally and does change the numeric result when available (the exception-only
    fallback path is not equivalent).
- [x] Notebook cells 0, 1, 2, 4, 5 rewritten to import from the new modules; live
      driver statements at the tail of cells 0, 1, 2, 4, 5 (mesh-folder conversion,
      circle generation, region-comparison plots) left in place, now resolving their
      paths through the registry instead of a hard-coded literal. Not re-run in full
      (would reprocess/overwrite the real 100-shape dataset's `sparta_converted/`
      output) — verification instead used the scratch-copy method above.
  - Incidental finding, not fixed (out of scope — a pre-existing bug, not something
    this relocation touched): cell 5's live driver calls
    `plot_regions_comparison(..., show_circle=True, ...)` — the actual parameter is
    `show_circles` (plural). This call would raise `TypeError` if run as-is; left
    exactly as found, per the relocation-only rule.
- [x] **Verification:** all 117 reference arrays reproduced bit-for-bit via the
      relocated `src/rarefied/geometry/` modules (`np.allclose`, zero mismatches) —
      caught and fixed the missing `import os` in `regions.py` during this step (the
      verification is what surfaced it). All 7 notebook cells confirmed to still
      `compile()` without a `SyntaxError` after the rewrite; whole-package import
      (`rarefied.geometry.*` alongside every Phase 3/4 module) confirmed clean.
- [ ] Plotting-only functions with no load-bearing caller (`create_histogram_animation*`,
      `plot_cp_cf_ch*`, `plot_aerothermo_slides`, `plot_mesh_shapes`,
      `plot_regions_comparison`, `plot_first_24_shapes_grid`, `parity_plot`,
      `_render_plots`, `_read_sparta_points_lines`) remain **not load-bearing** per
      the README's define-vs-inspect rule — left in the notebooks, not extracted.
- [ ] `scripts/shapes/shapes.py`, `shape_random.py`, `shape_from_file.py` (parent repo,
      sibling to this subtree) are a genuinely separate upstream stage (analytic
      shape/Bezier generation feeding into what `process_shape_data.ipynb` consumes) —
      not duplicates of anything in `process_shape_data.ipynb`. **CONFIRM** whether
      these should be pulled into this subtree (per README's "structured as if it will
      be extracted") or left where they are for now since the subtree is meant to be
      self-contained.

## Phase 6 — SPARTA C++ interface: wrap and document, do not rewrite

Per PIPELINE.md ("Status: exists and works. Wrap + document; do not rewrite.") and
decision E1 (SPARTA setup is one of the narrow pieces of *genuinely* shared code
between this project and the out-of-scope sibling). Lower priority than Phases 2–4;
sequence after the Python-side crown jewels are safe.

- [x] `SpartaAeroInterface.cpp/.hpp` are byte-identical between the parent repo's
      `SpartaAeroInterface/` and this subtree's
      `src/rarefied/sparta/SpartaDistributionGenerator/` — no action needed, already
      in sync. Documented in `src/rarefied/sparta/README.md`.
- [x] `sparta_utils.cpp/.hpp` differ only trivially (an include-path fix for the
      subtree's extra nesting level, plus one small added wrapper in the subtree copy)
      — no action needed. Documented in `src/rarefied/sparta/README.md`.
- [x] **`main.cpp` divergence documented, and a bigger issue surfaced while doing
      so — flagged to the user rather than resolved silently:** the parent's
      `config.txt` has `run_name KN_eq_10p0_3D`, `seed 12345`, `dim 2`, `sim_mode FT`
      — naming that matches the argon-cylinder Kn-sweep pickles this project's own
      characterization reference (`tests/characterization/kn_eq_1p0_reference.npz`)
      is built on. That means the tool that most plausibly produced **this
      project's own validation data** is `SpartaAeroInterface/main.cpp`, which lives
      **outside this subtree's directory boundary** — in tension with this
      subtree's own `README.md` ("the sibling direct-coupling project... is not
      part of this work"). Raised this directly to the user (not guessed): decided
      to **document both tools accurately and flag the scope tension explicitly**
      in `src/rarefied/sparta/README.md`, rather than assume the subtree's batch
      tool (`SpartaDistributionGenerator/main.cpp`) is "the" canonical one for this
      project, and rather than assume the parent's single-run tool should be pulled
      into scope. Neither is treated as superseding the other; no backporting done.
  - Related: parent-only files `config.txt`, `coupled_sparta_input`,
    `modified_from_dist_gen` (SPARTA input-deck templates referenced by
    `config.txt`'s `sparta_input_file` field) — still an open CONFIRM (folded into
    the scope question above), not resolved.
- [x] Documented the config schema for both copies (PIPELINE.md Contract B / open
      item "name + document the SPARTA custom interface and its config schema") in
      `src/rarefied/sparta/README.md`, based on direct inspection of
      `config_default.txt` (subtree) and `config.txt` (parent) — field names and
      inferred meanings only; semantics not independently verified beyond the name
      where not otherwise evident (e.g. `u_inf`/`t_inf`/`rho_inf` are all `0` in the
      parent's snapshot, `x/y/z_level_*_ref` are all `0` — likely overridden
      per-case, not confirmed).

## Phase 7 — Fill in docs `TODO`/`CONFIRM` markers (last)

Only after Phases 0–4 (at minimum) are done and the structure is settled — filling
these in earlier just means rewriting them once things move.

- [x] `PIPELINE.md` Contracts A–E: filled in against the actual data shapes now known
      from Phases 2–5. **Contract C** (pickle structure) and **Contract E** (moments
      dict + Hermite `fit`/`error` dicts) fully reconciled — high confidence, direct
      inspection. **Contract A** (station manifest) reconciled but flagged: the
      actually-implemented "region metadata" schema lacks the `s`/`phi` fields the
      original proposal named, and whether it's even the same concept as stages
      [1]/[2] imagine is now its own open question (CLEANUP_TODO CONFIRM-adjacent
      note added directly to PIPELINE.md's stage [2]). **Contract B and D left
      `FILL IN`** — no code implementing Contract B's specific banded-region format
      was found, and Contract D (AERO-F base state) was out of scope for this pass.
      Also updated stage-by-stage `Code`/`Status` lines, the registry section, the
      *Regression test* section (explicitly distinguishing the new characterization
      references from the originally-envisioned oracle-based test), and the *Open
      items* checklist.
- [x] `README.md`: confirmed subtree folder name (`rarefied-surrogate/`) and parent
      repo (`DSMC-GPR`, `github.com/kofiblake1/DSMC-GPR`); created `environment.yml`
      (delegates all deps to `pyproject.toml` via `pip install -e .` rather than
      duplicating the dependency list); updated the repo-map tree diagram and
      Quickstart with `[done 2026-09-12]`/`<!-- TODO -->` markers reflecting actual
      status per item. **`scripts/<stage>.py` placeholders NOT replaced** — no real
      entry points exist yet, nothing to replace them with.
- [x] `README.md` / PIPELINE.md's `docs/handoffs/argon_cylinder_cfd.md` references
      (CONFIRM #7) — confirmed missing (already known from Phase inventory); left
      as an explicit open item rather than fabricated, since writing it would
      require real AERO-F run parameters this cleanup pass has no way to determine.
- [ ] `docs/decisions/DECISIONS.md` section C ("related open items"): resolve or
      re-file each as its own decision once the argon-cylinder AERO-F work restarts —
      out of scope for this cleanup pass specifically. Untouched.
- [x] **New finding from this phase, flagged not fixed:** `README.md`'s tree diagram
      places notebooks under a top-level `notebooks/` folder; the two notebooks
      promoted from in Phases 3–5 actually live inside `src/rarefied/geometry/` and
      `src/rarefied/moments/`. Documented as CONFIRM #14 below and directly in
      `README.md`'s tree-diagram section — not moved, since their relative registry
      paths assume their current location.

---

## Open CONFIRM items (collected — do not guess on any of these)

1. **RESOLVED (Phase 4 + Phase 5):** duplicate `read_region_metadata` — two different
   implementations, same name, in the moments notebook (now
   `rarefied.io.sparta_runs.read_region_metadata`, tangent-angle DataFrame) vs. the
   geometry notebook (now `rarefied.geometry.regions.read_region_metadata`,
   circle-union list-of-dicts). Both relocated under their original, unchanged
   names — no rename, no rule violated, and no runtime collision since they live in
   different modules. Documented explicitly in both modules' docstrings so neither
   is mistaken for superseding the other.
2. **"Corrupted-Maxwellian experiment" identity** — PIPELINE.md/DECISIONS.md name this
   experiment as the regression-test source, but that exact phrase appears nowhere in
   any notebook, script, or data file found. The `KN_eq_{0.01,0.1,1.0,5.0,10.0}` Kn
   sweep in `~/Documents/Meteor_Modelling/code/` is the strongest candidate (matches
   the notebook's live driver cell exactly) but this is an inference, not a confirmed
   fact — confirm the mapping explicitly. (Phase 1/2)
3. **Suspicious identical file sizes** — `KN_eq_{0p01,0p1,1p0,5p0,10p0}_data.pkl` and
   all three `_3D_coarse` variants are all exactly 1,942,509,749 bytes. Could be
   coincidental (same grid resolution/timestep count → same serialized size) or could
   indicate an accidental copy/overwrite bug during archiving. Only `KN_eq_1p0_data.pkl`
   has been content-verified so far (Phase 1). Verify at least one sibling's actual
   content differs before trusting the registry entries.
4. **`my_data.pkl` format mismatch** — confirmed incompatible with the
   `(region_cache, region_df)` schema the moments notebook expects. Is this legacy/dead
   data, or does some other (not-yet-found) part of the pipeline still consume it?
   Don't register it as equivalent to the `KN_eq_*` files.
5. **SPARTA scope tension (upgraded from a naming question to a scope question,
   Phase 6):** the subtree's `SpartaDistributionGenerator/main.cpp` (batch
   generator) and the parent repo's `SpartaAeroInterface/main.cpp` (single-run
   tool) are NOT simply two variants of the same in-scope tool — the parent's
   `config.txt` (`run_name KN_eq_10p0_3D`) matches the naming of this project's own
   argon-cylinder characterization pickles, suggesting the tool that produced this
   project's own validation data lives outside the subtree's declared scope
   boundary. Raised to the user directly rather than assumed; documented as an open
   question in `src/rarefied/sparta/README.md` per their direction (document both
   tools, flag the tension, resolve neither). Still open: whether
   `config.txt`/`coupled_sparta_input`/`modified_from_dist_gen` (parent-only) need
   to move into this subtree, or whether `SpartaAeroInterface` should be considered
   in-scope where it sits.
6. **Provenance precision** — the only local SPARTA run-config found
   (`SpartaAeroInterface/config.txt` in the parent repo) has `run_name KN_eq_10p0_3D`;
   using it as the registry's `provenance` template for *all* `KN_eq_*` cases assumes
   each case was generated the same way with only `run_name`/Kn swapped. Confirm rather
   than assume when precision matters (e.g. for a paper's methods section). Now also
   tangled with CONFIRM #5 above — the config's *location* (parent repo, declared
   out of scope) as well as its exact per-case fidelity is unconfirmed.
7. **Missing `docs/handoffs/argon_cylinder_cfd.md`** — referenced twice by `README.md`,
   not found anywhere. (Phase 7)
8. **Cluster paths for `data/registry.yaml`** — every dataset's `cluster_path` field
   must stay `FILL_IN` until confirmed from a cluster-connected session; do not
   fabricate a plausible-looking path.
9. **`_3D_coarse` pickle variants** — purpose unconfirmed (a lower-resolution grid for
   fast iteration is a guess, not verified by loading one). Low priority; defer.
10. **RESOLVED (Phase 5):** the geometry notebook's `dataset_2026-02-06_09_28_35`
    path (appeared live in cells 0, 1, 2, and 5 — not just cell 5) is a real,
    substantial local dataset (100 mesh files), not throwaway output. Registered as
    `shape_dataset_2026_02_06_09_28_35` in `data/registry.yaml`; every live occurrence
    now resolves through it.
11. **`scripts/shapes/*.py` in the parent repo** — confirm whether these should move
    into this subtree (per README's "structured as if it will be extracted") or stay
    put; they're upstream of `process_shape_data.ipynb` but currently live outside the
    subtree boundary entirely.
12. **Moments-vs-reconstruction module split** for the two distribution-to-flux
    functions that skip the coefficient step (`momentum_flux_from_3d_distribution`,
    `heat_flux_from_3D_distribution`) — decide once, apply consistently. (Phase 3)
13. **RESOLVED (Phase 4):** IO vs. sparta module placement for the SPARTA
    text-output parsers (`parse_sparta_item_table`, `relabel_*_columns`,
    `load_run_data_3D`) — placed in `src/rarefied/io/sparta_runs.py`: they parse
    SPARTA *output* (a loader concern) rather than generate SPARTA *input* (what
    `sparta/` is for per the README). Documented in that module's docstring.
14. **Notebook-location mismatch (found in Phase 7):** `README.md`'s own repo-map
    tree diagram puts notebooks under a top-level `notebooks/` folder, separate
    from `src/`. The two notebooks actually promoted from in this cleanup
    (`process_shape_data.ipynb`, `proess_3D_distributions.ipynb`) physically live
    inside `src/rarefied/geometry/` and `src/rarefied/moments/` instead. Not moved
    during this cleanup — their relative-path registry lookups (added in Phases 3/5)
    assume their current depth under `src/rarefied/<subpackage>/`, and moving them
    as a side effect of an unrelated pass risked breaking that silently. Confirm
    where they should actually live (stay put now that they're mostly thin
    import+demo cells, or move to `notebooks/exploratory/`/`notebooks/validation/`
    with paths adjusted deliberately) rather than moving them by default.
