#!/bin/bash
set -euo pipefail

complex_dir="$1"

process_one() {
    local pdb="$1"
    local mol2="${pdb%.pdb}.mol2"

    # Skip if mol2 already exists
    if [[ -f "$mol2" ]]; then
        echo "Skipping $(basename "$pdb") (mol2 exists)"
        return
    fi

    # Unique temp file for parallel safety
    local tmp
    tmp=$(mktemp "${pdb%.pdb}_XXXX_tmp.mol2")

    echo "Running Chimera on $(basename "$pdb")..."

    echo -e "open $pdb\naddh\naddcharge\nwrite format mol2 0 $tmp\nstop" \
        | chimera --nogui > /dev/null 2>&1

    # Only continue if Chimera produced output
    if [[ -f "$tmp" ]]; then
        sed 's/H\.t3p/H    /' "$tmp" | sed 's/O\.t3p/O\.3  /' > "$mol2"
        rm -f "$tmp"
        echo "Created $(basename "$mol2")"
    else
        echo "Chimera failed for $(basename "$pdb")"
    fi
}

# Process only the specific pocket/ligand files in this directory
for pdb in "$complex_dir"/*_pocket.pdb "$complex_dir"/*_ligand.pdb; do
    [[ -f "$pdb" ]] && process_one "$pdb"
done
