import h5py
import numpy as np
import argparse
import sys

if __name__ != "__main__":
    print("This script is not intended to be imported.")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--input-sgcnn-hdf", default='data-self/val_sgcnn.hdf')
parser.add_argument("--input-3dcnn-fc10-npz", default='data-self/pdbbind2021_demo_val_feat.npz')
parser.add_argument("--output-file", default='data-self/val_fusion.hdf')
args = parser.parse_args()

# Open input files
fc6_f = h5py.File(args.input_sgcnn_hdf, 'r')
fc10_f = np.load(args.input_3dcnn_fc10_npz, 'r', allow_pickle=True)

print(f"FC 6 Proteins, {len(fc6_f.keys())}")
print(f"FC 10 Proteins, {len(fc10_f.keys())}")

# Create output HDF5 file
with h5py.File(args.output_file, 'w') as output_f:
    # Process each protein
    for key in fc6_f.keys():
        if key in fc10_f:
            # Get SGCNN features (fc6)
            fc6_features = fc6_f[key]['0']['hidden_features']
            fc6_features = np.array(fc6_features).reshape(-1)[-6:]
            print(f"FC6 Features for {key}:", fc6_features.shape)

            # Get affinity value
            affinity = fc6_f[key]['0'].attrs['y_true'].reshape(-1)
            print(f"Affinity for {key}: ", affinity)

            # Get 3DCNN features (fc10)
            fc10_features = fc10_f[key]
            print(f"FC10 Features for {key}:", fc10_features.shape)
            
            # Concatenate fc6 and fc10 features
            concatenated_features = np.concatenate([fc10_features, fc6_features])
            
            # Create a group for this protein
            protein_group = output_f.create_group(key)
            
            # Store concatenated features as dataset
            protein_group.create_dataset('fc16', data=concatenated_features)
            
            # Store affinity as attribute
            protein_group.attrs['affinity'] = affinity
        else:
            print(f"WARNING: Protein {key} in SG-CNN but not found in fc10_f")

# Close input files
fc6_f.close()
fc10_f.close()

print(f"Concatenated features saved to {args.output_file}")