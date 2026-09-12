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

<!-- TODO: confirm the exact subtree folder name (rarefied-surrogate/ vs rarefied/)
     and the parent-repo path it lives under. -->

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
  environment.yml            # conda env / dependencies      <!-- TODO -->
  pyproject.toml             # makes src/ installable (pip install -e .)  <!-- TODO -->

  config/
    cases/                   # one config per case
      argon_cylinder.yaml    #   the first validation case    <!-- TODO: create -->
    schema.md                # what every config field means  <!-- TODO -->

  src/rarefied/              # ALL logic lives here — importable, tested
    geometry/                # analytic shape + surface-station generation
    sparta/                  # sampling-region + run setup for the custom interface
    aerof/                   # mesh + input-deck generation; base-state extraction
    moments/                 # the hand-derived moment functions (canonical)
    reconstruct/             # moments + run info -> dimensional surface fluxes
    io/                      # loaders: pickles, VDF data, configs, registry
    features/                # feature-vector assembly (planned)
    models/                  # GP foundation + fine-tune (planned)

  scripts/                   # thin CLI entry points; each calls into src/
                             #   one per pipeline stage (see PIPELINE.md)

  notebooks/
    exploratory/             # fast, messy, disposable scratchpads (not pipeline)
    validation/              # thin notebooks that reproduce paper figures

  tests/
    regression/              # "does it still reproduce known-good results?"

  data/
    README.md                # WHERE the real data lives (paths), not the data
    registry.yaml            # manifest: dataset -> path, format, provenance

  docs/
    decisions/               # locked modeling decisions + rationale
    handoffs/                # Claude Code handoff docs (argon-cylinder CFD, etc.)
```

The directory tree deliberately **mirrors the pipeline stages** (geometry → sampling
→ runs → moments → reconstruction). `ls src/rarefied/` should let a new reader infer
the architecture.

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

<!-- TODO: fill in data/registry.yaml with the real cluster paths and the local
     pickle locations, plus provenance for each dataset. -->

---

## Quickstart

```bash
# 1. environment
conda env create -f environment.yml        # <!-- TODO: create environment.yml -->
conda activate rarefied                     # <!-- TODO: confirm env name -->

# 2. make the package importable
pip install -e .                            # <!-- TODO: create pyproject.toml -->

# 3. run a stage (see PIPELINE.md for the full list)
python scripts/<stage>.py --case config/cases/argon_cylinder.yaml   # <!-- TODO -->

# 4. reproduce the known-good validation result (guards the crown jewels)
pytest tests/regression                     # <!-- TODO: write this test -->
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
