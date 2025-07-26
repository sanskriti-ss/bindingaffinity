import os
import subprocess

# Paths - update these!
input_root = r"FAST-master\data\PDBbind_v2020_refined\refined-set"
refined_2020_pll_root = r"C:\Users\user\Documents\bindingaffinity\parallel_refined_2020_hydro"

def find_protein_complexes(root_dir):
    for dirs, _, filenames in os.walk(root_dir):
        yield os.path.basename(dirs)


if __name__ == "__main__":
    refined_proteins = list(find_protein_complexes(input_root))
    print(f"Len refined : {len(refined_proteins)}, {refined_proteins[:5]}, {refined_proteins[-5:]}")
    refined_2020_parallel = list(find_protein_complexes(refined_2020_pll_root))
    print(f"Len 2020 parallel refined : {len(refined_2020_parallel)}, {list(refined_2020_parallel)[:5]}, {list(refined_2020_parallel)[-5:]}")
    print(f"Len intersection : {len(set(refined_2020_parallel).intersection(set(refined_proteins)) )}")
    print(f"Remaining for processing: {len(set(refined_proteins).difference(set(refined_2020_parallel)))}")