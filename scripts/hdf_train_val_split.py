import h5py
import random

# Set random seed for reproducibility
random.seed(42)

# Path to source file
# source_file = 'proteins.h5'
source_file = 'pdbbind_2020_refined_good_proteins.hdf'
test_file = "pdbbind2016_core_test.hdf"


# Load top-level protein group names
with h5py.File(source_file, 'r') as src:
    protein_ids = set(src.keys()) # Change 150 to other number

print("Proteins in source file:", len(protein_ids), ", file: ", source_file)
    
with h5py.File(test_file, 'r') as src:
    test_ids = set(src.keys())

print("Proteins in test file:", len(test_ids))

protein_ids = list(protein_ids - test_ids)
print("Proteins in source file after removing test proteins:", len(protein_ids))

# # Shuffle and split
random.shuffle(protein_ids)
split_idx = int(len(protein_ids) * 0.8)
train_ids = protein_ids[:split_idx]
val_ids = protein_ids[split_idx:]

print(f"Train proteins: {len(train_ids)}, preview: {list(train_ids)[:5]}")
print(f"Val proteins: {len(val_ids)}, preview: {list(val_ids)[:5]}")
print(f"Test proteins: {len(test_ids)}, preview: {list(test_ids)[:5]}")


# Helper function to copy groups recursively
def copy_proteins(protein_list, dst_file):
    print(f"Copying {len(protein_list)} proteins to {dst_file}...")
    with h5py.File(source_file, 'r') as src, h5py.File(dst_file, 'w') as dst:
        for pid in protein_list:
            src.copy(pid, dst)
    print(f"Copied {len(protein_list)} proteins to {dst_file}!!!")

# Write train, val and test files
copy_proteins(train_ids, 'pdbbind_2020_refined_train.hdf')
copy_proteins(val_ids, 'pdbbind_2020_refined_val.hdf')
