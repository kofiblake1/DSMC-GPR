#!/bin/bash
# Run a FluidFile variant of the Case2 setup on the same mesh.
#
# An impulsive start into a uniform M=3 field drives a negative-pressure
# failure at the stagnation region within ~10 iterations, because second-order
# reconstruction extrapolates across the bow shock while it is still forming.
# This script exists to try the standard remedies one at a time and report
# which are actually needed.
#
# Usage:  ./run_variant.sh <name> [sed-expression ...]
#   ./run_variant.sh failsafe 's/FailSafe = Off;/FailSafe = On;/'

set -uo pipefail

NAME=$1; shift
BASE=${BASE:-$SCRATCH/aerof_case2}
SRC_FLUID=${SRC_FLUID:-$SCRATCH/aerof_case2_m3/simulations/Case2/FluidFile}
DST=$SCRATCH/aerof_m3_$NAME

mkdir -p "$DST"/simulations/Case2/{results,references,postpro}
# sources/ and data/ are identical for every variant: only the FluidFile
# changes, so link rather than copy.
for d in sources data; do
  [ -e "$DST/$d" ] || ln -s "$BASE/$d" "$DST/$d"
done

cp "$SRC_FLUID" "$DST/simulations/Case2/FluidFile"
for expr in "$@"; do
  sed -i "$expr" "$DST/simulations/Case2/FluidFile"
done

ml purge >/dev/null 2>&1
ml devel math >/dev/null 2>&1
ml gcc/9.1.0 openmpi/4.0.3 imkl/2019 >/dev/null 2>&1

cd "$DST"
start=$(date +%s)
timeout 1800 mpirun -np 4 /home/groups/cfarhat/bin/aerof2 \
  simulations/Case2/FluidFile > simulations/Case2/log.out 2>&1
rc=$?
wall=$(( $(date +%s) - start ))

its=$(grep -c '^It ' simulations/Case2/log.out)
last=$(grep '^It ' simulations/Case2/log.out | tail -1)
negp=$(grep -c 'negative pressure' simulations/Case2/log.out)

printf '%-16s rc=%-3s %4ss  its=%-5s neg_p=%-6s %s\n' \
  "$NAME" "$rc" "$wall" "$its" "$negp" "$last"
