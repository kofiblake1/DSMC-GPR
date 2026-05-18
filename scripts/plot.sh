#!/bin/bash
#
#SBATCH --job-name=spartaPlots
#
#SBATCH --time=10:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --partition=cfarhat

# run sparta
module load mesa 
export PATH=~/ParaView-5.13.3-MPI-Linux-Python3.10-x86_64/bin:$PATH
mpiexec -np 16 pvbatch --sym ../../sparta-20Jan2025/tools/paraview/grid2paraview.py surf_short.txt surf_short_grid -r out.*
mpiexec -np 16 pvbatch --sym ../../sparta-20Jan2025/tools/paraview/surf2paraview.py ../surf.circle surf_short_surf -r outsurf.*
#mpirun -np 8 pvbatch --sym ../sparta-20Jan2025/tools/paraview/grid2paraview.py surf_short.txt surf_short_grid -r out.*
#mpirun -np 16 pvbatch --sym ../sparta-20Jan2025/tools/paraview/grid2paraview.py hs_80km_201_mid_higher_coll_2_40_pll.txt hs_80km_201_mid_higher_coll_2_40_pll_grid -r out.*
