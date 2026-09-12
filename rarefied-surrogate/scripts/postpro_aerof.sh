#!/bin/bash
# Merge the per-subdomain binary results and write an Exodus file.
# Mirrors the tutorial's postpro.sh.
#
# The postpro/ directory must exist before sower runs: sower reports success
# and writes nothing if the output directory is missing.
#
# Usage:  ./postpro.sh [case_root]        default $SCRATCH/aerof_case2

set -euo pipefail

ROOT=${1:-$SCRATCH/aerof_case2}
SOWER=/home/groups/cfarhat/bin/sower
XP2EXO=/home/groups/cfarhat/bin/xp2exo

ml purge >/dev/null 2>&1
ml devel math viz >/dev/null 2>&1
ml gcc/9.1.0 netcdf/4.4.1.1 >/dev/null 2>&1

cd "$ROOT/simulations/Case2"
mkdir -p postpro

for f in Mach Pressure; do
  "$SOWER" -fluid -merge -con ../../data/fluidmodel.con \
    -mesh ../../data/fluidmodel.msh -result "results/$f.bin" \
    -output "postpro/$f" -name "$f" -quiet
done

"$XP2EXO" ../../sources/circle.top postpro/Case2.exo \
  postpro/Mach.xpost postpro/Pressure.xpost

ls -la postpro/
