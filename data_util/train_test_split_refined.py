#!/usr/bin/env python3
"""
Split refined.hdf to match existing train/val/test splits
"""

import os
import h5py
import pandas as pd
import numpy as np
import argparse

def split_refined_to_match_existing(refined_hdf, splits_dir, output_dir):
    """
    Split refined.hdf to match existing train/val/test splits
    
    Args:
        refined_hdf: Path to refined.hdf file
        splits_dir: Directory containing existing split CSV files
        output_dir: Output directory for new split files
    """
    
    # Read existing split CSV files to get the complex IDs
    split_files = {
        'train': os.path.join(splits_dir, 'refined_3d_train.csv'),
        'val': os.path.join(splits_dir, 'refined_3d_val.csv'),
        'test': os.path.join(splits_dir, 'refined_3d_test.csv')
    }
    
    # Load complex IDs for each split
    split_complexes = {}
    for split_name, csv_path in split_files.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Split file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        split_complexes[split_name] = set(df['ligand_id'].values)
        print(f"{split_name.upper()} set: {len(split_complexes[split_name])} complexes")
    
    # Check for overlaps between splits
    train_ids = split_complexes['train']
    val_ids = split_complexes['val']
    test_ids = split_complexes['test']
    
    print(f"\nChecking for overlaps...")
    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)
    
    if train_val_overlap:
        print(f"WARNING: {len(train_val_overlap)} complexes overlap between train and val")
    if train_test_overlap:
        print(f"WARNING: {len(train_test_overlap)} complexes overlap between train and test")
    if val_test_overlap:
        print(f"WARNING: {len(val_test_overlap)} complexes overlap between val and test")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open refined.hdf file
    print(f"\nOpening refined.hdf: {refined_hdf}")
    with h5py.File(refined_hdf, 'r') as input_h5:
        
        # Get all available complex IDs in refined.hdf
        available_complexes = set(input_h5.keys())
        print(f"Available complexes in refined.hdf: {len(available_complexes)}")
        
        # Check which complexes from splits are available in refined.hdf
        all_split_complexes = train_ids.union(val_ids).union(test_ids)
        missing_complexes = all_split_complexes - available_complexes
        extra_complexes = available_complexes - all_split_complexes
        
        if missing_complexes:
            print(f"WARNING: {len(missing_complexes)} complexes from splits not found in refined.hdf")
            print(f"Missing complexes: {sorted(list(missing_complexes))[:10]}...")  # Show first 10
        
        if extra_complexes:
            print(f"INFO: {len(extra_complexes)} extra complexes in refined.hdf not in any split")
        
        # Create output HDF5 files for each split
        for split_name, complex_ids in split_complexes.items():
            output_hdf_path = os.path.join(output_dir, f"refined_{split_name}.hdf")
            
            print(f"\nCreating {split_name} split: {output_hdf_path}")
            
            # Find complexes that exist in both the split and refined.hdf
            valid_complexes = complex_ids.intersection(available_complexes)
            missing_in_split = complex_ids - available_complexes
            
            print(f"  Complexes to copy: {len(valid_complexes)}")
            if missing_in_split:
                print(f"  Missing from refined.hdf: {len(missing_in_split)}")
            
            # Create output HDF5 file
            with h5py.File(output_hdf_path, 'w') as output_h5:
                copied_count = 0
                
                for complex_id in valid_complexes:
                    try:
                        # Copy the entire group/dataset for this complex
                        input_h5.copy(complex_id, output_h5)
                        copied_count += 1
                    except Exception as e:
                        print(f"  Error copying {complex_id}: {e}")
                
                print(f"  Successfully copied: {copied_count} complexes")
    
    # Create summary statistics
    print(f"\n=== Summary ===")
    for split_name in ['train', 'val', 'test']:
        split_hdf_path = os.path.join(output_dir, f"refined_{split_name}.hdf")
        
        if os.path.exists(split_hdf_path):
            with h5py.File(split_hdf_path, 'r') as h5f:
                actual_count = len(h5f.keys())
                expected_count = len(split_complexes[split_name])
                print(f"{split_name.upper()}: {actual_count}/{expected_count} complexes copied")
    
    print(f"\n✅ Splitting completed!")
    print(f"📁 Output files saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Split refined.hdf to match existing splits')
    parser.add_argument('--refined-hdf', required=True, help='Path to refined.hdf file')
    parser.add_argument('--splits-dir', required=True, help='Directory containing existing split CSV files')
    parser.add_argument('--output-dir', required=True, help='Output directory for new HDF5 splits')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.refined_hdf):
        raise FileNotFoundError(f"refined.hdf not found: {args.refined_hdf}")
    if not os.path.exists(args.splits_dir):
        raise FileNotFoundError(f"Splits directory not found: {args.splits_dir}")
    
    # Run the splitting
    split_refined_to_match_existing(
        args.refined_hdf,
        args.splits_dir,
        args.output_dir
    )

if __name__ == "__main__":
    main()