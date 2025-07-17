#!/usr/bin/bash
################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# example script to process pdbbind dataset
################################################################################

#SBATCH -t 1-00:00:00
#SBATCH -p pbatch

set -u

shopt -s globstar

num_jobs=$1
pdbbind_path=${2:-"/home/karen/Projects/pdbbind/PDBbind_v2020_refined"}

timestamp=$(date +%b_%d_%y_%H_%M_%e)

echo "using ${num_jobs} workers.."
echo "processing PDBbind data from: ${pdbbind_path}"

# Find all _pocket.pdb files in the refined-set directory
find ${pdbbind_path}/refined-set -name "*_pocket.pdb" | \
parallel -j${num_jobs} --timeout 600 --delay 2.5 --joblog prepare_chimera_pdbbind_2020_refined_${timestamp}.out ./prepare_complexes_chimera.sh {}

echo "done."