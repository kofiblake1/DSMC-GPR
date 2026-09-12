#!/bin/bash
# Wrapper that runs the gmsh CLI in its own module environment, isolated from
# the python stack (gmsh/4.10.1 forces gcc/10.1.0, which breaks py312 numpy).
#
# Usage:  gmsh_env.sh -3 model.geo -format msh2 -o model.msh

ml purge >/dev/null 2>&1
ml math >/dev/null 2>&1
ml gmsh/4.10.1 >/dev/null 2>&1
exec gmsh "$@"
