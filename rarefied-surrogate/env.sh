#!/bin/bash
# Cluster environment for rarefied-surrogate (Sherlock).
#
# This is the CLUSTER definition. environment.yml is the LOCAL (conda) one, and
# the two deliberately differ -- see the comment block in environment.yml.
#
# Why not pip on the cluster: Sherlock is CentOS 7 with glibc 2.17, while
# current numpy/scipy wheels target manylinux_2_28 (glibc >= 2.28). pip finds
# no compatible wheel, falls back to the source tarball, and the build fails.
# SRCC's py-* modules are compiled on-site against this glibc, so they are the
# working path. Only the pure-python remainder (tqdm, pyyaml) goes in a venv.
#
# Why gmsh is not loaded here: gmsh/4.10.1 forces gcc/10.1.0, which breaks the
# py312 numeric stack (built against gcc/12.4.0). geo.py therefore drives the
# gmsh CLI as a subprocess through src/rarefied/aerof/gmsh_env.sh, which loads
# its own modules. Do not add gmsh to this file.
#
# Two Lmod traps, both of which cost real debugging time:
#   1. `ml <category> <module>` in ONE call silently loads nothing. The category
#      must be loaded in its own `ml` call first, hence the split below.
#   2. NEVER pipe `ml` into another command. It is a shell function that evals
#      its output; a pipe runs it in a subshell and the environment change is
#      silently lost.
#
# Usage:  source env.sh

# numpy is pinned to 1.26.3, not the newer 2.2.6, because py-pandas/2.2.1_py312
# depends on it and Lmod silently downgrades numpy if you ask for 2.2.6 anyway.
# Pinning the version that actually resolves keeps the environment deterministic
# instead of quietly different from what this file claims.
ml purge >/dev/null 2>&1
ml math viz >/dev/null 2>&1
ml python/3.12.1 >/dev/null 2>&1
ml py-numpy/1.26.3_py312 py-scipy/1.16.0_py312 \
   py-matplotlib/3.10.3_py312 py-pandas/2.2.1_py312 >/dev/null 2>&1

# Pure-python extras that have no py312 module (tqdm, pyyaml). Created with:
#   python3 -m venv --system-site-packages $RAREFIED_VENV
#   $RAREFIED_VENV/bin/pip install tqdm pyyaml
export RAREFIED_VENV=${RAREFIED_VENV:-$GROUP_HOME/kofib_envs/rarefied}
if [ -f "$RAREFIED_VENV/bin/activate" ]; then
    source "$RAREFIED_VENV/bin/activate"
fi

# Make src/ importable without `pip install -e .` (which would try to resolve
# the pinned numeric stack through pip and fail, per the glibc note above).
_RAREFIED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$_RAREFIED_ROOT/src:${PYTHONPATH}"

# Group-installed solver chain: gmsh2top, sower, mpmetis, cd2tet, aerof2.
# These are cluster-only and have no local equivalent -- see PIPELINE.md's
# capability-tiers note.
export FRG_BIN=/home/groups/cfarhat/bin
export PATH=$FRG_BIN:$PATH

# Upstream Bezier shape generator: a pristine clone of jviquerat/shapes, loaded
# by path (never vendored) via aerof/shapes_bridge.py, which records its commit
# hash in every run record.
export SHAPES_DIR=${SHAPES_DIR:-$HOME/shapes}

# Keep caches off $HOME (15 GB, NFS-backed).
export XDG_CACHE_HOME=$SCRATCH/.cache
export MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export PIP_CACHE_DIR=$SCRATCH/.pip_cache

# Default root for generated data. $SCRATCH is purged after 90 days without
# content modification: meshes are regenerable by design, datasets are not, so
# move anything worth keeping to $GROUP_HOME.
export MESH_RUN_ROOT=${MESH_RUN_ROOT:-$SCRATCH/shape_meshes}
