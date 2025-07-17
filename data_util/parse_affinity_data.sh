#!/usr/bin/bash
################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# based on implementation provide in https://gitlab.com/cheminfIBB/pafnucy/blob/master/pdbbind_data.ipynb
################################################################################


# user specify path as first arg
path=$1

echo 'pdbid,-logKd/Ki' > affinity_data.csv
cat $path/refined-set/index/INDEX_general_PL_data.2020 | while read l1 l2 l3 l4 l5; do
    if [[ ! $l1 =~ "#" ]]; then
        echo $l1,$l4
    fi
done >> affinity_data.csv


# Find affinities without structural data (i.e. with missing directories)

cut -f 1 -d ',' affinity_data.csv | tail -n +2 | while read l;
    do if [ ! -e $path/refined-set/$l ]; then
        echo $l;
    fi
done



# Note: INDEX_core_data.2020 doesn't exist in PDBbind v2020
# Using INDEX_refined_set.2020 instead for core-like functionality
grep -v '#' $path/refined-set/index/INDEX_refined_set.2020 | cut -f 1 -d ' '  > core.csv

grep -v '#' $path/refined-set/index/INDEX_refined_data.2020 | cut -f 1 -d ' ' > refined.csv
