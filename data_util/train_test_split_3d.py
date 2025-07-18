#!/usr/bin/env python3
"""
Split the refined dataset into train/test/val splits for 3D CNN training
"""

import os
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import shutil

def split_dataset(input_hdf, input_csv, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Split dataset into train/val/test splits
    
    Args:
        input_hdf: Input HDF5 file path
        input_csv: Input CSV file path
        output_dir: Output directory for split files
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_state: Random seed for reproducibility
    """
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Read CSV metadata
    print("Reading CSV metadata...")
    df = pd.read_csv(input_csv)
    print(f"Total complexes: {len(df)}")
    
    # Get unique ligand IDs
    ligand_ids = df['ligand_id'].values
    
    # First split: train vs (val + test)
    train_ids, temp_ids = train_test_split(
        ligand_ids, 
        test_size=(val_ratio + test_ratio), 
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ids, test_ids = train_test_split(
        temp_ids, 
        test_size=(test_ratio / (val_ratio + test_ratio)), 
        random_state=random_state
    )
    
    print(f"Train set: {len(train_ids)} complexes ({len(train_ids)/len(ligand_ids)*100:.1f}%)")
    print(f"Val set: {len(val_ids)} complexes ({len(val_ids)/len(ligand_ids)*100:.1f}%)")
    print(f"Test set: {len(test_ids)} complexes ({len(test_ids)/len(ligand_ids)*100:.1f}%)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output files
    splits = {
        'train': {'ids': train_ids, 'split_num': 0},
        'val': {'ids': val_ids, 'split_num': 1},
        'test': {'ids': test_ids, 'split_num': 2}
    }
    
    # Open input HDF5 file
    print("Processing HDF5 files...")
    with h5py.File(input_hdf, 'r') as input_h5:
        
        # Create output HDF5 files and CSV files
        for split_name, split_info in splits.items():
            split_ids = split_info['ids']
            split_num = split_info['split_num']
            
            # Output file paths
            output_hdf_path = os.path.join(output_dir, f"refined_3d_{split_name}.hdf")
            output_csv_path = os.path.join(output_dir, f"refined_3d_{split_name}.csv")
            
            print(f"Creating {split_name} split...")
            
            # Create output HDF5 file
            with h5py.File(output_hdf_path, 'w') as output_h5:
                
                # Copy data for this split
                for ligand_id in split_ids:
                    if ligand_id in input_h5:
                        # Copy dataset
                        input_h5.copy(ligand_id, output_h5)
                    else:
                        print(f"Warning: {ligand_id} not found in HDF5 file")
            
            # Create CSV for this split
            split_df = df[df['ligand_id'].isin(split_ids)].copy()
            split_df['train_test_split'] = split_num
            
            # Update file_prefix to point to the new split files
            split_df['file_prefix'] = split_df['ligand_id'].apply(
                lambda x: f"{split_num}/refined_3d_{split_name}/{x}"
            )
            
            # Save CSV
            split_df.to_csv(output_csv_path, index=False)
            
            print(f"  Saved {len(split_df)} complexes to {output_hdf_path}")
            print(f"  Saved metadata to {output_csv_path}")
    
    # Create combined CSV file for convenience
    print("Creating combined CSV file...")
    combined_dfs = []
    for split_name, split_info in splits.items():
        split_csv_path = os.path.join(output_dir, f"refined_3d_{split_name}.csv")
        split_df = pd.read_csv(split_csv_path)
        combined_dfs.append(split_df)
    
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    combined_csv_path = os.path.join(output_dir, "refined_3d_combined.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    
    print(f"Saved combined CSV to {combined_csv_path}")
    
    # Print statistics
    print("\n=== Split Statistics ===")
    for split_name in ['train', 'val', 'test']:
        split_df = combined_df[combined_df['train_test_split'] == splits[split_name]['split_num']]
        print(f"{split_name.upper()} SET:")
        print(f"  Count: {len(split_df)}")
        print(f"  Affinity range: {split_df['label'].min():.2f} - {split_df['label'].max():.2f}")
        print(f"  Affinity mean: {split_df['label'].mean():.2f} ± {split_df['label'].std():.2f}")
        print()
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Split refined dataset into train/val/test')
    parser.add_argument('--input-hdf', required=True, help='Input HDF5 file')
    parser.add_argument('--input-csv', required=True, help='Input CSV file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.input_hdf):
        raise FileNotFoundError(f"Input HDF5 file not found: {args.input_hdf}")
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {args.input_csv}")
    
    # Run the splitting
    output_dir = split_dataset(
        args.input_hdf,
        args.input_csv,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.random_seed
    )
    
    print(f"\n✅ Dataset splitting completed!")
    print(f"📁 Output files saved to: {output_dir}")
    print(f"🔍 Files created:")
    for split in ['train', 'val', 'test']:
        print(f"   - refined_3d_{split}.hdf")
        print(f"   - refined_3d_{split}.csv")
    print(f"   - refined_3d_combined.csv")

if __name__ == "__main__":
    main()