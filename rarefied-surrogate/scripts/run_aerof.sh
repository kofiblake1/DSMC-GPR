#!/bin/bash
# Run AERO-F Case2 (SteadyViscousPG). Mirrors the tutorial's run.sh.
#
# Two deviations from the tutorial's module line, both forced:
#   * cmake/3.8.1 is gone from Sherlock (only 3.11.1+ remain) and Lmod aborts
#     the whole `module load` if any member is missing. cmake is a build-time
#     tool, not a runtime dependency, so it is simply dropped.
#   * the module category must be loaded in its own `ml` call first, otherwise
#     `ml <category> <module>` silently loads nothing on this cluster.
#
# Also note: never pipe `ml` into another command. Lmod's `ml` is a shell
# function that evals its output, and a pipe puts it in a subshell, so the
# environment change is silently lost.
#
# Usage:  ./run.sh [case_root] [nprocs]      default $SCRATCH/aerof_case2, 4

set -euo pipefail

ROOT=${1:-$SCRATCH/aerof_case2}
NP=${2:-4}
AEROF=/home/groups/cfarhat/bin/aerof2

ml purge >/dev/null 2>&1
ml devel math >/dev/null 2>&1
ml gcc/9.1.0 openmpi/4.0.3 imkl/2019 >/dev/null 2>&1

cd "$ROOT"
echo "running $NP MPI ranks against a $(grep -c . sources/circle.top)-line .top"
time mpirun -np "$NP" "$AEROF" simulations/Case2/FluidFile \
  > simulations/Case2/log.out 2>&1

echo
grep -E "^It +(0|1000):" simulations/Case2/log.out || true
tail -3 simulations/Case2/results/Residual.out
