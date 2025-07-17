#!/usr/bin/env python3

import pandas as pd
import os

# Read the original affinity data
df = pd.read_csv('/home/karen/Projects/FAST/data_util/affinity_data.csv')

# Read the core and refined lists
try:
    core_df = pd.read_csv('/home/karen/Projects/FAST/data_util/core.csv', header=None, names=['pdbid'])
    core_set = set(core_df['pdbid'].tolist())
except FileNotFoundError:
    core_set = set()

try:
    refined_df = pd.read_csv('/home/karen/Projects/FAST/data_util/refined.csv', header=None, names=['pdbid'])
    refined_set = set(refined_df['pdbid'].tolist())
except FileNotFoundError:
    refined_set = set()

# Create the metadata format expected by extract_pafnucy_data_with_docking.py
metadata = []
pdbbind_base = "/home/karen/Projects/pdbbind/PDBbind_v2020_refined/refined-set"

for _, row in df.iterrows():
    pdb_id = row['pdbid']
    
    # Skip header row
    if pdb_id == 'pdbid':
        continue
        
    affinity = row['-logKd/Ki']
    
    # Determine set membership
    if pdb_id in core_set:
        dataset_set = 'core'
    elif pdb_id in refined_set:
        dataset_set = 'refined'
    else:
        dataset_set = 'refined'  # Default to refined
    
    # Check if both pocket and ligand mol2 files exist
    pocket_path = f"{pdbbind_base}/{pdb_id}/{pdb_id}_pocket.mol2"
    ligand_path = f"{pdbbind_base}/{pdb_id}/{pdb_id}_ligand.mol2"
    
    if os.path.exists(pocket_path) and os.path.exists(ligand_path):
        metadata.append({
            'name': pdb_id,
            'set': dataset_set,
            '-logKd/Ki': affinity,
            'receptor_path': pocket_path
        })

# Create DataFrame and save
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv('/home/karen/Projects/FAST/data_util/metadata_for_extract.csv', index=False)

print(f"Created metadata file with {len(metadata_df)} entries")
print(f"Core: {len(metadata_df[metadata_df['set'] == 'core'])}")
print(f"Refined: {len(metadata_df[metadata_df['set'] == 'refined'])}")