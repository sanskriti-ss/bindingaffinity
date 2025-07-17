#!/usr/bin/bash

pdbbind_path="/home/karen/Projects/pdbbind/PDBbind_v2020_refined"

echo "Checking directory structure..."
echo "Base path: ${pdbbind_path}"

if [ -d "${pdbbind_path}/refined-set" ]; then
    echo "✓ refined-set directory exists"
    
    # Count total directories in refined-set
    total_dirs=$(find ${pdbbind_path}/refined-set -maxdepth 1 -type d | wc -l)
    echo "Total directories in refined-set: $((total_dirs - 1))"
    
    # Look for _pocket.pdb files
    pocket_files=$(find ${pdbbind_path}/refined-set -name "*_pocket.pdb" | wc -l)
    echo "Found ${pocket_files} _pocket.pdb files"
    
    # Show first 5 examples
    echo "First 5 _pocket.pdb files:"
    find ${pdbbind_path}/refined-set -name "*_pocket.pdb" | head -5
    
else
    echo "✗ refined-set directory not found"
    echo "Available directories:"
    ls -la ${pdbbind_path}/
fi