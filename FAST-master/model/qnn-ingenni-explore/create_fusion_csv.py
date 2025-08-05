import h5py
import numpy as np
import argparse
import sys
import csv

if __name__ != "__main__":
    print("This script is not intended to be imported.")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--input-sgcnn-hdf", default='')
parser.add_argument("--input-3dcnn-fc10", default='')
parser.add_argument("--output-file", default='')
args = parser.parse_args()

"""
python create_fusion_csv.py --input-sgcnn-hdf data_refined_2020/sgcnn_pdbbind2016_core_test.hdf --input-3dcnn-fc10 data_refined_2020/pdbbind2016_core_test_feat.npz --output-file data_refined_2020/pdbbind2016_core_test_fusion.csv
python create_fusion_csv.py --input-sgcnn-hdf data_refined_2020/sgcnn_pdbbind_refined_2020_val.hdf --input-3dcnn-fc10 data_refined_2020/pdbbind_2020_refined_val_feat.npz --output-file data_refined_2020/pdbbind_2020_refined_val_fusion.csv
python create_fusion_csv.py --input-sgcnn-hdf data_refined_2020/sgcnn_pdbbind_refined_2020_train.hdf --input-3dcnn-fc10 data_refined_2020/pdbbind_2020_refined_train_feat.npz --output-file data_refined_2020/pdbbind_2020_refined_train_fusion.csv

"""

# Open input files
fc6_f = h5py.File(args.input_sgcnn_hdf, 'r')
fc10_f = np.load(args.input_3dcnn_fc10, 'r', allow_pickle=True)

print(f"FC 6 Proteins, {len(fc6_f.keys())}")
print(f"FC 10 Proteins, {len(fc10_f.keys())}")

# Create CSV file
with open(args.output_file, 'w', newline='') as csvfile:
    # Create header: name, fc0-fc15, affinity
    fieldnames = ['name'] + [f'fc{i}' for i in range(16)] + ['affinity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # Process each protein
    for key in fc6_f.keys():
        if key in fc10_f:
            # Get SGCNN features (fc6)
            fc6_features = fc6_f[key]['0']['hidden_features']
            fc6_features = np.array(fc6_features).reshape(-1)[-6:]
            print(f"FC6 Features for {key}:", fc6_features.shape)

            # Get affinity value
            affinity = fc6_f[key]['0'].attrs['y_true'].reshape(-1)[0]  # Get scalar value
            print(f"Affinity for {key}: ", affinity)

            # Get 3DCNN features (fc10)
            fc10_features = fc10_f[key]
            print(f"FC10 Features for {key}:", fc10_features.shape)
            
            # Concatenate fc6 and fc10 features
            concatenated_features = np.concatenate([fc10_features, fc6_features])
            
            # Create row dictionary
            row = {'name': key, 'affinity': affinity}
            # Add feature values
            for i, value in enumerate(concatenated_features):
                row[f'fc{i}'] = value
                
            # Write row to CSV
            writer.writerow(row)
        else:
            print(f"WARNING: Protein {key} in SG-CNN but not found in fc10_f")

# Close input files
fc6_f.close()
fc10_f.close()

print(f"Concatenated features saved to CSV file {args.output_file}")