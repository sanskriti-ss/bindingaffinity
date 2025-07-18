import h5py
import numpy as np
from openbabel import openbabel
import os

# VDW radii dictionary (Bondi radii in Angstroms) as fallback
VDW_RADII = {
    1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80, 
    16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98
}

def get_vdw_radius(atomic_num):
    """Get VDW radius using OpenBabel or fallback to Bondi radii"""
    try:
        vdw = openbabel.GetVdwRad(int(atomic_num))
        if vdw > 0:
            return vdw
    except:
        pass
    
    # Fallback to Bondi radii
    return VDW_RADII.get(int(atomic_num), 2.0)

def add_vdw_to_hdf(hdf_path):
    """Add van der Waals radii to existing HDF5 file"""
    print(f"Processing {hdf_path}")
    
    with h5py.File(hdf_path, 'r+') as f:
        for complex_id in f.keys():
            data_path = f"{complex_id}/pybel/processed/pdbbind/data"
            attrs_path = f"{complex_id}/pybel/processed/pdbbind"
            
            if data_path not in f:
                print(f"  Warning: {data_path} not found, skipping {complex_id}")
                continue
                
            # Check if van_der_waals already exists
            if "van_der_waals" in f[attrs_path].attrs:
                print(f"  {complex_id}: van_der_waals already exists, skipping")
                continue
            
            # Get the data
            data = f[data_path][:]
            
            # Extract atomic numbers (assuming they're in columns 3-11 as one-hot encoding)
            # For pybel features: columns 3-11 are element one-hot encoding
            element_cols = data[:, 3:12]  # 9 elements: H, C, N, O, F, P, S, Cl, other
            
            vdw_radii = []
            for i, row in enumerate(element_cols):
                # Find which element this atom is
                element_idx = np.argmax(row)
                
                # Map element index to atomic number
                element_mapping = {
                    0: 1,   # H
                    1: 6,   # C  
                    2: 7,   # N
                    3: 8,   # O
                    4: 9,   # F
                    5: 15,  # P
                    6: 16,  # S
                    7: 17,  # Cl
                    8: 6    # Other -> default to Carbon
                }
                
                atomic_num = element_mapping.get(element_idx, 6)
                vdw_radius = get_vdw_radius(atomic_num)
                vdw_radii.append(vdw_radius)
            
            # Store as attribute
            vdw_array = np.array(vdw_radii, dtype=np.float32)
            f[attrs_path].attrs["van_der_waals"] = vdw_array
            
            print(f"  {complex_id}: Added {len(vdw_radii)} VDW radii")

def main():
    # Add VDW radii to all your split files
    split_files = [
        "/home/karen/Projects/FAST/data/refined_splits/refined_train.hdf",
        "/home/karen/Projects/FAST/data/refined_splits/refined_val.hdf", 
        "/home/karen/Projects/FAST/data/refined_splits/refined_test.hdf"
    ]
    
    for hdf_file in split_files:
        if os.path.exists(hdf_file):
            add_vdw_to_hdf(hdf_file)
        else:
            print(f"File not found: {hdf_file}")
    
    print("✅ Finished adding VDW radii to all files")

if __name__ == "__main__":
    main()