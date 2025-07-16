"""
Fix NaN and Inf values in protein grid data

This script cleans the protein_grids.npy file by replacing NaN and Inf values
with appropriate finite values.
"""

import numpy as np
import os

def fix_protein_grids(input_path, output_path=None, strategy='zero'):
    """
    Fix NaN and Inf values in protein grids
    
    Args:
        input_path: Path to the input protein_grids.npy file
        output_path: Path to save the fixed file (if None, overwrites input)
        strategy: Strategy for handling NaN/Inf values
                 'zero' - replace with 0
                 'mean' - replace with channel mean
                 'median' - replace with channel median
                 'clip' - clip infinite values to reasonable bounds
    """
    print(f"Loading protein grids from: {input_path}")
    
    # Load the data
    protein_grids = np.load(input_path)
    print(f"Original shape: {protein_grids.shape}")
    print(f"Original dtype: {protein_grids.dtype}")
    
    # Check for issues
    nan_count = np.isnan(protein_grids).sum()
    inf_count = np.isinf(protein_grids).sum()
    finite_count = np.isfinite(protein_grids).sum()
    total_count = protein_grids.size
    
    print(f"\nOriginal data statistics:")
    print(f"  NaN values: {nan_count:,} ({100*nan_count/total_count:.2f}%)")
    print(f"  Inf values: {inf_count:,} ({100*inf_count/total_count:.2f}%)")
    print(f"  Finite values: {finite_count:,} ({100*finite_count/total_count:.2f}%)")
    
    if nan_count == 0 and inf_count == 0:
        print("✅ No NaN or Inf values found - data is already clean!")
        return protein_grids
    
    # Fix the data based on strategy
    print(f"\n🔧 Applying fix strategy: {strategy}")
    
    if strategy == 'zero':
        # Replace NaN and Inf with 0
        protein_grids = np.nan_to_num(protein_grids, nan=0.0, posinf=0.0, neginf=0.0)
        
    elif strategy == 'clip':
        # Replace NaN with 0, clip infinite values to reasonable bounds
        finite_mask = np.isfinite(protein_grids)
        if finite_mask.any():
            finite_min = np.min(protein_grids[finite_mask])
            finite_max = np.max(protein_grids[finite_mask])
            finite_std = np.std(protein_grids[finite_mask])
            
            # Set reasonable bounds (mean ± 5 standard deviations)
            clip_min = finite_min - 5 * finite_std
            clip_max = finite_max + 5 * finite_std
            
            print(f"  Finite range: [{finite_min:.3f}, {finite_max:.3f}]")
            print(f"  Clipping to: [{clip_min:.3f}, {clip_max:.3f}]")
            
            protein_grids = np.nan_to_num(protein_grids, nan=0.0, posinf=clip_max, neginf=clip_min)
        else:
            protein_grids = np.nan_to_num(protein_grids, nan=0.0, posinf=0.0, neginf=0.0)
            
    elif strategy == 'mean':
        # Replace with channel-wise mean of finite values
        for sample_idx in range(protein_grids.shape[0]):
            for channel_idx in range(protein_grids.shape[1]):
                channel_data = protein_grids[sample_idx, channel_idx]
                finite_mask = np.isfinite(channel_data)
                
                if finite_mask.any():
                    channel_mean = np.mean(channel_data[finite_mask])
                    channel_data[~finite_mask] = channel_mean
                else:
                    channel_data[:] = 0.0
                    
                protein_grids[sample_idx, channel_idx] = channel_data
                
    elif strategy == 'median':
        # Replace with channel-wise median of finite values
        for sample_idx in range(protein_grids.shape[0]):
            for channel_idx in range(protein_grids.shape[1]):
                channel_data = protein_grids[sample_idx, channel_idx]
                finite_mask = np.isfinite(channel_data)
                
                if finite_mask.any():
                    channel_median = np.median(channel_data[finite_mask])
                    channel_data[~finite_mask] = channel_median
                else:
                    channel_data[:] = 0.0
                    
                protein_grids[sample_idx, channel_idx] = channel_data
    
    # Verify the fix
    new_nan_count = np.isnan(protein_grids).sum()
    new_inf_count = np.isinf(protein_grids).sum()
    new_finite_count = np.isfinite(protein_grids).sum()
    
    print(f"\n✅ Fixed data statistics:")
    print(f"  NaN values: {new_nan_count:,} ({100*new_nan_count/total_count:.2f}%)")
    print(f"  Inf values: {new_inf_count:,} ({100*new_inf_count/total_count:.2f}%)")
    print(f"  Finite values: {new_finite_count:,} ({100*new_finite_count/total_count:.2f}%)")
    print(f"  Min value: {np.min(protein_grids):.6f}")
    print(f"  Max value: {np.max(protein_grids):.6f}")
    print(f"  Mean value: {np.mean(protein_grids):.6f}")
    
    # Save the fixed data
    if output_path is None:
        output_path = input_path
        print(f"\n💾 Saving fixed data to: {output_path} (overwriting original)")
    else:
        print(f"\n💾 Saving fixed data to: {output_path}")
    
    np.save(output_path, protein_grids)
    
    return protein_grids


def main():
    """Main function to fix protein grid data"""
    print("🔧 Protein Grid Data Fixer")
    print("=" * 50)
    
    input_path = "processed_protein_data/protein_grids.npy"
    backup_path = "processed_protein_data/protein_grids_backup.npy"
    
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return
    
    # Create backup
    if not os.path.exists(backup_path):
        print(f"📦 Creating backup: {backup_path}")
        import shutil
        shutil.copy2(input_path, backup_path)
    else:
        print(f"📦 Backup already exists: {backup_path}")
    
    # Try different strategies
    strategies = ['clip', 'zero', 'mean']
    
    for strategy in strategies:
        print(f"\n{'='*20} Testing strategy: {strategy} {'='*20}")
        
        # Load fresh data for each test
        test_grids = fix_protein_grids(input_path, output_path=None, strategy=strategy)
        
        # Check if this strategy worked
        if np.isfinite(test_grids).all():
            print(f"✅ Strategy '{strategy}' successfully cleaned all data!")
            break
        else:
            remaining_issues = (~np.isfinite(test_grids)).sum()
            print(f"⚠️  Strategy '{strategy}' left {remaining_issues} non-finite values")
    
    print(f"\n🎉 Data cleaning completed!")
    print(f"Original file backed up to: {backup_path}")
    print(f"Fixed file saved to: {input_path}")


if __name__ == "__main__":
    main()
