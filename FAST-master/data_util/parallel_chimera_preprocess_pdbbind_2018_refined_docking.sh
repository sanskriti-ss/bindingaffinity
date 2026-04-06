#!/usr/bin/env bash
set -u

num_jobs=${1:-4}
timestamp=$(date +%b_%d_%y_%H_%M_%S)

echo "using ${num_jobs} workers.."

find ../data/refined-set -mindepth 1 -maxdepth 1 -type d | \
parallel -j "${num_jobs}" --timeout 600 --delay 0.5 \
--joblog "prepare_chimera_${timestamp}.log" \
./prepare_complexes_chimera.sh {}

echo "done."
