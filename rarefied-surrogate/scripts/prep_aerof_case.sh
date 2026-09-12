#!/bin/bash
# Build the FRG short-course Case2 file set for a pipeline-generated mesh.
#
# Reproduces FolderCases1To4/README step for step, with two substitutions:
#   * partnmesh  -> mpmetis  (both write <top>.dec.N in sower's format; the
#     pipeline already produced this file, so we just reuse it)
#   * 2DNaca     -> circle
#
# Directory layout is identical to the tutorial's, which is what lets the
# FluidFile stay byte-identical apart from its geometry comment.
#
#   sources/      circle.top, circle.top.dec.4, circle.sinus.dwall
#   data/         fluidmodel.{msh1,con,dec1,dwall1,1cpu,2cpu,4cpu}
#   simulations/Case2/  FluidFile, results/, references/, postpro/
#
# WARNING: the FluidFile this copies is the tutorial's -- air (gamma 1.4,
# R 287.1, Pr 0.72) with a TURBULENT closure and wall functions. That is a
# smoke test of the solver chain, NOT this project's physics, which M3/M4
# require to be laminar argon. See docs/handoffs/argon_cylinder_cfd.md.
#
# Usage:  scripts/prep_aerof_case.sh [case_root]   default $SCRATCH/aerof_case2
#         CASE=config/cases/other.yaml scripts/prep_aerof_case.sh

set -euo pipefail

ROOT=${1:-$SCRATCH/aerof_case2}
CASE=${CASE:-config/cases/circle_1m.yaml}
FRG=/home/groups/cfarhat/bin
TUTORIAL=$GROUP_HOME/FRG_ShortCourses/FolderCases1To4
SUBTREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

echo "case root: $ROOT"
mkdir -p "$ROOT"/{sources,data,simulations/Case2/{results,references,postpro}}
cd "$ROOT"

# ---- 1. mesh the shape ------------------------------------------------------
# Mesh parameters come from the case config ($CASE). Its thickness 0.05 D and
# extrusion along +z match the tutorial mesh (2DNaca.top is x-y with
# z in [0, 0.05]). The extrusion axis matters: AERO-F sets
# u = V cos(a)cos(b), v = V cos(a)sin(b), w = V sin(a)
# (DistTimeState.C:321-323), so with Alpha = 0 the flow lies in x-y and
# nothing crosses the side planes.
MESHNAME=$(basename "$CASE" .yaml)
if [ ! -f "$MESHNAME/$MESHNAME.top" ]; then
  ( cd "$SUBTREE" && source ./env.sh \
    && MESH_RUN_ROOT="$ROOT" python3 scripts/gen_mesh.py \
         --case "$CASE" --name "$MESHNAME" --no-vtu )
fi
cp -f "$MESHNAME/$MESHNAME.top" "$MESHNAME/$MESHNAME.top.dec.4" sources/
# Downstream steps and the FluidFile refer to the mesh as circle.*; keep that
# stable regardless of the case name.
[ "$MESHNAME" = circle ] || {
  cp -f "sources/$MESHNAME.top"       sources/circle.top
  cp -f "sources/$MESHNAME.top.dec.4" sources/circle.top.dec.4
}

# ---- 2. distance to the wall ------------------------------------------------
# Required by both Integration = WallFunction and the Spalart-Allmaras model.
"$FRG"/cd2tet -mesh sources/circle.top -output sources/circle.sinus \
  > sources/cd2tet.log 2>&1
echo "cd2tet: $(ls -la sources/circle.sinus.dwall | awk '{print $5}') bytes"

# ---- 3. preprocess the mesh with sower --------------------------------------
"$FRG"/sower -fluid -mesh sources/circle.top -dec sources/circle.top.dec.4 \
  -cpu 1 -cpu 2 -cpu 4 -output data/fluidmodel -cluster 1 \
  > data/sower_mesh.log 2>&1
grep -E "Found|subdomain" data/sower_mesh.log

# ---- 4. preprocess the wall distance with sower -----------------------------
"$FRG"/sower -fluid -split -mesh data/fluidmodel.msh -con data/fluidmodel.con \
  -cluster 1 -result sources/circle.sinus.dwall -ascii \
  -output data/fluidmodel.dwall > data/sower_dwall.log 2>&1

# ---- 5. the FluidFile, byte-identical apart from the geometry comment -------
sed -e 's|// FLUID GEOMETRY:    sources/2DNaca.top|// FLUID GEOMETRY:    sources/circle.top  (rarefied-surrogate)|' \
    "$TUTORIAL/simulations/Case2/FluidFile" > simulations/Case2/FluidFile

echo
echo "diff vs the tutorial FluidFile:"
diff "$TUTORIAL/simulations/Case2/FluidFile" simulations/Case2/FluidFile || true
echo
ls -la data/
