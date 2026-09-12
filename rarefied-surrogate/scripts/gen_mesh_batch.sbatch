#!/bin/bash
#SBATCH --job-name=shapemesh
#SBATCH --partition=normal
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.out
#
# Shard a shape-meshing batch across an array. Each task writes to its own
# shard directory under $MESH_RUN_ROOT/<run-name>/shardNN, so tasks never
# contend for the same files and each draws from a disjoint seed stream.
#
#   sbatch --export=ALL,RUN_NAME=set1,N_PER_SHARD=25 batch.sbatch
#
# Output lands on $SCRATCH. Move anything worth keeping to $GROUP_HOME:
# $SCRATCH is purged after 90 days without content modification.

set -euo pipefail

RUN_NAME=${RUN_NAME:-batch}
N_PER_SHARD=${N_PER_SHARD:-25}
N_SURF=${N_SURF:-300}
NPARTS=${NPARTS:-4}

cd "$SLURM_SUBMIT_DIR"
source ./env.sh

python3 run_batch.py \
    --n "$N_PER_SHARD" \
    --run-name "$RUN_NAME" \
    --n-surf "$N_SURF" \
    --nparts "$NPARTS" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" \
    --nshards "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --no-vtu \
    --contact-sheet
