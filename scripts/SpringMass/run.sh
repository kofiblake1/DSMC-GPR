AEROS=/home/groups/cfarhat/bin/aeros
XP2EXO=/home/groups/cfarhat/bin/xp2exo


# run simulation
module load cmake/3.8.1 gcc/9.1.0 openmpi/4.0.3 imkl/2019  netcdf/4.4.1.1
mpirun -np 1 $AEROS StructureFile |& tee log.out


###
mpirun -np 1 $AEROS -t StructureFile
$XP2EXO imp69.top postpro/Case5.exo results/gdisplac # results/strainvm results/stressvm
