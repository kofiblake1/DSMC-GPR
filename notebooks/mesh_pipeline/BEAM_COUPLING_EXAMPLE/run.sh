#!/bin/bash
#
#SBATCH --job-name=beam_coupling_example
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --partition=cfarhat

# Runs the beam <-> 2D-SPARTA-surface coupling example. See README.md in
# this directory for context, prerequisites, and what to expect.
#
# 1 AERO-S rank + 8 SPARTA ranks = 9 total. Adjust --ntasks-per-node and the
# "-np 8" below together if you want more SPARTA ranks (e.g. for a bigger
# outline) - AERO-S always runs with exactly 1 rank for this coupling path.

AEROS=/home/users/kofib/aero-s/bin/aeros
INTERFACE=/home/users/kofib/SpartaAeroInterface/main

module load cmake gcc openmpi openblas eigen netcdf/4.4.1.1 libpng

mpirun -np 1 $AEROS StructureFile : -np 8 $INTERFACE |& tee run.log
