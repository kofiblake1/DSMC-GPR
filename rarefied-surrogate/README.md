# rarefied-surrogate

**A DSMC-informed Gaussian Process surrogate for surface fluxes in rarefied external flow.**

Given an object in a rarefied hypersonic flow, this project predicts the near-wall
velocity distribution function (VDF) as a Hermite correction on top of a cheap
continuum base predictor, then takes moments of that VDF to produce the surface
stress and heat flux that a structural/thermal solver needs — at a fraction of the
cost of running DSMC in a coupled loop.

> **Status:** work in progress. First validation target is the 2D argon-cylinder
> CFD case (Lofthouse, Boyd & Wright 2007). See `docs/handoffs/argon_cylinder_cfd.md`.

---

## Where this sits

This is a **self-contained subtree** inside the parent `DSMC-GPR` repository. The
parent repo also holds a separate *direct* DSMC–FEM coupling project. **That sibling
project is out of scope here.** Everything under this directory is the ML-surrogate
work only; it does not import from, and should not modify, the direct-coupling code.

The subtree is structured *as if it will one day be extracted into its own repo*, so
that separation (if/when it happens) is a directory move rather than a disentangling
project. Until then, it shares the parent repo's git history and remote.

**Confirmed (2026-09-12):** the subtree folder name is `rarefied-surrogate/` (not
`rarefied/`), living at the top level of the parent repo
`DSMC-GPR` (`github.com/kofiblake1/DSMC-GPR`, remote `origin`) — i.e. this file's
path is `DSMC-GPR/rarefied-surrogate/README.md`. The sibling direct-coupling
project's code lives in `DSMC-GPR`'s other top-level directories (`SpartaAeroInterface/`,
`scripts/`, `notebooks/`, etc.) — see `src/rarefied/sparta/README.md` for one
concrete place this boundary was tested during the 2026-09-12 cleanup (and where
it turned out to be less clean than assumed).

---

## Read this first (especially Claude Code sessions)

1. **`PIPELINE.md`** — the source of truth for the data flow: every stage, what it
   consumes and produces, the format of each hand-off, where data physically lives,
   and how to run it. Read it before touching any stage.
2. **`docs/decisions/`** — the locked modeling decisions (base predictor, sampling,
   validation case, etc.) and *why*. These constrain what the pipeline is allowed to
   do; don't silently override them.
3. This README — the map and the working conventions.

**Scope note for any automated session:** work only within this subtree unless
explicitly told otherwise. The sibling direct-coupling project in the parent repo is
not part of this work.

---

## Repository map

```
rarefied-surrogate/
  README.md                  # this file — the map and conventions
  PIPELINE.md                # the data-flow spec (keystone; read first)
  environment.yml            # conda env / dependencies      [done 2026-09-12]
  pyproject.toml             # makes src/ installable (pip install -e .)  [done 2026-09-12]

  env.sh                     # CLUSTER environment (Lmod modules + small venv)  [2026-09-12]
                             #   environment.yml is the LOCAL one; they differ
                             #   deliberately -- see both files' headers

  config/
    cases/
      circle_1m.yaml         #   reference mesh case (stage 4a)  [2026-09-12]
      argon_cylinder.yaml    #   the first validation case    <!-- TODO: create -->
    schema.md                # what every config field means   [2026-09-12]

  src/rarefied/              # ALL logic lives here — importable, tested
    geometry/                # analytic shape + surface-station generation  [populated 2026-09-12]
                             #   + curves.py, contact_sheet.py (AERO-F side)  [2026-09-12]
    sparta/                  # sampling-region + run setup for the custom interface
                             #   [C++ interface documented 2026-09-12, see its README.md;
                             #    no Python wrapper code yet]
    aerof/                   # mesh generation + partition  [populated 2026-09-12]
                             #   geo, mshio, quality, topfile, partition, pipeline,
                             #   quicklook, shapes_bridge, gmsh_env.sh
                             #   base-state extraction (stage 4c) NOT yet written
    moments/                 # the hand-derived moment functions (canonical)  [populated 2026-09-12]
    reconstruct/             # moments + run info -> dimensional surface fluxes  [populated 2026-09-12]
    io/                      # loaders: pickles, VDF data, configs, registry  [populated 2026-09-12]
    features/                # feature-vector assembly (planned)  (still empty)
    models/                  # GP foundation + fine-tune (planned)  (still empty)

  scripts/                   # thin CLI entry points; each calls into src/  [2026-09-12]
    gen_mesh.py              #   stage 4a, one case
    gen_mesh_batch.py        #   stage 4a, a dataset (+ .sbatch for arrays)
    prep_aerof_case.sh       #   mesh + cd2tet + sower -> AERO-F file set
    run_aerof.sh             #   mpirun aerof2
    postpro_aerof.sh         #   sower -merge + xp2exo
    run_aerof_variant.sh     #   solver-setting exploration (user-owned)

  notebooks/
    exploratory/             # fast, messy, disposable scratchpads (not pipeline)
    validation/              # thin notebooks that reproduce paper figures

  tests/
    characterization/        # [new, 2026-09-12] frozen current-behavior snapshots,
                             #   used to verify each Phase 3-5 relocation was non-destructive --
                             #   NOT the same as a validated regression test, see below
    regression/              # "does it still reproduce known-good results?"  still empty

  data/
    README.md                # WHERE the real data lives (paths), not the data  <!-- TODO -->
    registry.yaml            # manifest: dataset -> path, format, provenance  [populated 2026-09-12,
                             #   local paths only -- cluster paths still FILL_IN]

  docs/
    decisions/               # locked modeling decisions + rationale
                             #   M4 mesh clause amended, M12/M13 added  [2026-09-12]
    handoffs/                # [populated 2026-09-12]
      aerof_mesh_and_case2.md    #   stages 4a-4b: the verified recipe + traps
      argon_cylinder_cfd.md      #   the first validation case (skeleton; resolves CONFIRM #7)
    CLEANUP_TODO.md           # [new, 2026-09-12] the dependency-ordered cleanup plan and its
                             #   full record of what was found, moved, and left as open CONFIRMs
```

The directory tree deliberately **mirrors the pipeline stages** (geometry → sampling
→ runs → moments → reconstruction). `ls src/rarefied/` should let a new reader infer
the architecture.

**Known mismatch, flagged 2026-09-12, not yet resolved:** the two notebooks that
were promoted from (`process_shape_data.ipynb`, `proess_3D_distributions.ipynb`)
physically live inside `src/rarefied/geometry/` and `src/rarefied/moments/`
respectively — not under a top-level `notebooks/` folder as this tree diagram
shows. Moving them would risk breaking the relative paths added during the
2026-09-12 relocation (e.g. each notebook's registry lookup assumes its specific
depth under `src/rarefied/<subpackage>/`), so they were left in place rather than
moved as a side effect of an unrelated cleanup pass. Whether they belong in
`notebooks/exploratory/`, `notebooks/validation/`, or should just stay where they
are (since they now only contain the non-load-bearing plotting/demo cells) is an
open question — see `docs/CLEANUP_TODO.md`.

---

## Development workflow: notebooks vs. `src/`

We keep both fast iteration *and* scalable, testable code by treating them as two
**layers**, not as a choice.

**The rule:** if a line of code *defines behavior*, it belongs in `src/rarefied/`.
If a line of code *inspects or explores behavior*, it belongs in a notebook.

- **`src/rarefied/`** holds every function that does real work — the moment math,
  shape/region generation, loaders, reconstruction. This is what the regression
  tests call, what `scripts/` invoke, and what Claude Code reads and edits. A
  notebook must never be the only place a piece of logic is defined.

- **`notebooks/exploratory/`** is your scratchpad. Fast, messy, disposable; cells may
  define things temporarily. Not part of the pipeline and not guaranteed to run.
  Don't police these — their value is that they're unconstrained.

- **`notebooks/validation/`** are *thin*: they `import` from `src/`, run on a case,
  and produce a figure (e.g., the corrupted-Maxwellian validation plots). Because
  they only call library code, they stay reproducible and legible.

**The promotion path** is how the two layers connect: develop at full speed in an
exploratory notebook; when a piece of logic proves itself, *promote* it into
`src/rarefied/`, replace the cell with an import, and add a test if it's load-bearing.
Discoveries flow one direction — notebook → package — as they earn their place.
Don't over-promote: a look-once plot can live and die in a notebook. The test for
promotion is "will this be *called again* — by another stage, a test, or future me?"

**Two settings that remove the friction:**

- **autoreload** — put `%load_ext autoreload` / `%autoreload 2` at the top of every
  notebook. Then you can edit a function in `src/` and re-run a notebook cell to see
  the change instantly, with no kernel restart. This gives the notebook's fast loop
  *while editing library code*.
- **jupytext** (validation notebooks only) — pairs each notebook with a plain `.py`
  that syncs automatically, so it diffs cleanly in git and Claude Code can read/edit
  it. Skip it for exploratory notebooks; the ceremony isn't worth it there.

---

## Data: local vs. cluster

Code and config are small and version-controlled (identical local and on the
cluster via git). **Data is large, lives authoritatively on the cluster, and is
referenced — never committed.**

- `data/registry.yaml` is the manifest: dataset name → absolute path, format, and
  provenance (which case/run/config produced it). **Code reads paths through the
  registry, never hard-codes them.**
- The archived 3D VDF **pickles** are precious, expensive-to-regenerate *inputs*.
  They are registered with provenance and read through a single loader in
  `src/rarefied/io/`. (Longer-term: consider migrating the archival format to a
  self-describing, language-agnostic one — HDF5/NPZ — but that's optional hardening,
  not now.)

**Status (2026-09-12):** local pickle locations and provenance are filled in (see
`data/registry.yaml`). Cluster paths are still `<!-- TODO -->` — that needs a
cluster-connected session to confirm, not guessed at from here.

---

## Quickstart

```bash
# 1a. LOCAL environment  [environment.yml created 2026-09-12]
conda env create -f environment.yml
conda activate rarefied
python3.10 -m pip install -e .

# 1b. CLUSTER environment -- deliberately different, see env.sh's header.
#     Sherlock is CentOS 7 / glibc 2.17 and current numpy/scipy wheels need
#     glibc >= 2.28, so pip cannot build the numeric stack there; the SRCC
#     py-* modules are used instead, plus a small venv for tqdm/pyyaml.
source env.sh                              # sets PYTHONPATH; no pip install -e needed

# 2. run a stage (see PIPELINE.md for the full list)
#    Stage [4a]/[4b] are CLUSTER-ONLY and need a Slurm allocation -- gmsh,
#    gmsh2top, mpmetis, sower and aerof2 have no local equivalent.
srun -p dev -c 4 --mem=16G -t 00:30:00 --pty bash
source env.sh
python3 scripts/gen_mesh.py --case config/cases/circle_1m.yaml   # stage [4a]
./scripts/prep_aerof_case.sh && ./scripts/run_aerof.sh           # stage [4b]
# stages [1],[2],[3],[5],[6] still have no CLI entry point  <!-- TODO -->

# 4. verify a relocation didn't change behavior (2026-09-12 -- NOT the same as #5 below)
python3.10 tests/characterization/build_reference.py   # freezes current output
# (then diff a re-run against the frozen .npz -- see docs/CLEANUP_TODO.md's
# Phase 3 methodology for the actual diff script; this is characterization,
# not validation)

# 5. reproduce the known-good validation result (guards the crown jewels)
pytest tests/regression                     # <!-- TODO: still not written --
                                              #      tests/characterization/ is not
                                              #      a substitute, see PIPELINE.md's
                                              #      Regression test section -->
```

<!-- TODO: replace <stage> placeholders once scripts/ entry points exist. -->

---

## First milestone

Reproduce the **argon-cylinder CFD** results (Lofthouse et al. 2007) with the
existing AERO-F setup infrastructure, then confirm the SPARTA side, then turn on
automated shape + data generation across both solvers. See
`docs/handoffs/argon_cylinder_cfd.md` for the case parameters and targets.

---

*Draft v0.1 — expect to iterate. Placeholders marked `TODO` / `FILL IN` are the
checklist to reconcile against the existing code and cluster layout.*

*2026-09-12: first reconciliation pass — package scaffolding, `data/registry.yaml`
local entries, and the `src/rarefied/{moments,reconstruct,io,geometry}/`
promotions are done; see `docs/CLEANUP_TODO.md` for the full record, including
several open questions this pass surfaced rather than resolved (the notebook-location
mismatch above, the SPARTA scope question in `src/rarefied/sparta/README.md`, and
others in that file's CONFIRM list).*
