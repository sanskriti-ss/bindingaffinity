import h5py
import random

# Set random seed for reproducibility
random.seed(42)

# Path to source file
# source_file = 'proteins.h5'
source_file = "pdbbind2016_core_test.hdf"


# Load top-level protein group names
with h5py.File(source_file, 'r') as src:
    protein_ids = list(src.keys())[:150] # Change 150 to other number
    test_ids = list(src.keys())[150:200] 

# # Shuffle and split
random.shuffle(protein_ids)
split_idx = int(len(protein_ids) * 0.8)
train_ids = protein_ids[:split_idx]
val_ids = protein_ids[split_idx:]

print(f"Train proteins: {train_ids}")
print(f"Val proteins: {val_ids}")
print(f"Test proteins: {test_ids}")


# Helper function to copy groups recursively
def copy_proteins(protein_list, dst_file):
    with h5py.File(source_file, 'r') as src, h5py.File(dst_file, 'w') as dst:
        for pid in protein_list:
            src.copy(pid, dst)

# Write train, val and test files
copy_proteins(train_ids, 'train.hdf')
copy_proteins(val_ids, 'val.hdf')
copy_proteins(test_ids, 'test.hdf')
