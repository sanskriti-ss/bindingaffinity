"""
Script to copy proteins from orginal dataset to the hydro dataset
"""


import os
import shutil

# Source and destination root directories
source_dir = "PDBbind_v2020_refined/refined-set"
destination_dir = "parallel_refined_2020_hydro"

# Loop through each subfolder in the source directory
for protein in os.listdir(source_dir):
    protein_folder = os.path.join(source_dir, protein)
    dest_folder = os.path.join(destination_dir, protein)

    if os.path.isdir(protein_folder) and os.path.exists(dest_folder):
        for file in os.listdir(protein_folder):
            if "_ligand" in file and (file.endswith(".mol2") or file.endswith(".sdf")):
                source_file = os.path.join(protein_folder, file)
                dest_file = os.path.join(dest_folder, file)
                shutil.copy2(source_file, dest_file)
                print(f"Copied {source_file} to {dest_file}")
