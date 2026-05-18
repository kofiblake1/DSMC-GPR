#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=cfarhat

#AEROS=/home/groups/cfarhat/bin/aeros
AEROS=/home/users/kofib/aero-s/bin/aeros
INTERFACE=/home/users/kofib/SpartaAeroInterface/main
XP2EXO=/home/groups/cfarhat/bin/xp2exo
SPARTA=/home/users/kofib/sparta-20Jan2025


# run simulation
module load cmake gcc openmpi openblas eigen netcdf/4.4.1.1 libpng
#mpirun -np 2 $INTERFACE : -np 1 $AEROS StructureFile |& tee log.out 
#mpirun -np 1 $AEROS StructureFile : -np 2 $INTERFACE > all.log 2>&1
# mpirun -np 1 $AEROS -v 1 StructureFile2 : -np 23 $INTERFACE |& tee log.out
# mpirun -np 1 $AEROS -v 1 StructureFile2 |& tee log.out
mpirun -np 1 $AEROS  StructureFile2 : -np 3 $INTERFACE 
# mpirun -np 23 $INTERFACE |& tee log.out

#mpirun -np 1 $AEROS -v 1 StructureFile2 |& tee log.out


###
mpirun -np 1 $AEROS -t StructureFile2
$XP2EXO circle.top postpro/circle.exo results/gtempera results/gtempvel

module load mesa 
export PATH=~/ParaView-5.13.3-MPI-Linux-Python3.10-x86_64/bin:$PATH
cd surf_no_ablation
mpiexec -np 16 pvbatch --sym $SPARTA/tools/paraview/grid2paraview.py surf_no_ablation.txt surf_no_ablation_grid -r out.*
mpiexec -np 16 pvbatch --sym $SPARTA/tools/paraview/surf2paraview.py ../surf.circle surf_no_ablation_surf -r outsurf.*