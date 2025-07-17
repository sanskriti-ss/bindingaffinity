#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter

def analyze_complex_features(hdf_file, complex_ids=None, n_complexes=3):
    """
    Analyze and visualize features from processed HDF5 file
    """
    # Feature names for the 19-channel representation
    feature_names = [
        'B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal',  # 0-8: one-hot elements
        'hybridization',  # 9: hybridization
        'heavy_bonds',    # 10: heavy-atom bonds
        'hetero_bonds',   # 11: hetero-atom bonds
        'hydrophobic',    # 12: hydrophobic (carbon)
        'aromatic',       # 13: aromatic
        'acceptor',       # 14: H-bond acceptor
        'donor',          # 15: H-bond donor
        'ring',           # 16: in ring
        'partial_charge', # 17: partial charge
        'mol_type'        # 18: molecule type (+1 ligand, -1 protein)
    ]
    
    with h5py.File(hdf_file, 'r') as f:
        # Get complex IDs to analyze
        if complex_ids is None:
            complex_ids = list(f.keys())[:n_complexes]
        
        print(f"Analyzing complexes: {complex_ids}")
        
        # Create subplots
        fig, axes = plt.subplots(len(complex_ids), 3, figsize=(18, 6*len(complex_ids)))
        if len(complex_ids) == 1:
            axes = axes.reshape(1, -1)
        
        for i, complex_id in enumerate(complex_ids):
            if complex_id not in f:
                print(f"Complex {complex_id} not found in HDF5 file")
                continue
                
            # Get the data
            data = f[complex_id]['pybel']['processed']['pdbbind']['data'][:]
            affinity = f[complex_id].attrs['affinity']
            
            coords = data[:, :3]
            features = data[:, 3:]
            
            print(f"\n=== {complex_id} ===")
            print(f"Affinity: {affinity:.2f}")
            print(f"Total atoms: {data.shape[0]}")
            print(f"Coordinates shape: {coords.shape}")
            print(f"Features shape: {features.shape}")
            
            # Separate ligand and protein atoms
            ligand_mask = features[:, 18] == 1  # mol_type == +1
            protein_mask = features[:, 18] == -1  # mol_type == -1
            
            ligand_features = features[ligand_mask]
            protein_features = features[protein_mask]
            
            print(f"Ligand atoms: {np.sum(ligand_mask)}")
            print(f"Protein atoms: {np.sum(protein_mask)}")
            
            # Plot 1: Element distribution
            ax1 = axes[i, 0]
            element_counts = np.sum(features[:, :9], axis=0)
            element_names = feature_names[:9]
            bars = ax1.bar(element_names, element_counts)
            ax1.set_title(f'{complex_id}: Element Distribution')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars, element_counts):
                if count > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{int(count)}', ha='center', va='bottom')
            
            # Plot 2: Structural features
            ax2 = axes[i, 1]
            struct_features = features[:, 9:17]  # hybridization through ring
            struct_names = feature_names[9:17]
            
            # For continuous features (hybridization, bonds, partial_charge)
            continuous_features = [9, 10, 11, 17]  # hybridization, heavy_bonds, hetero_bonds, partial_charge
            binary_features = [12, 13, 14, 15, 16]  # hydrophobic, aromatic, acceptor, donor, ring
            
            # Show binary features as counts
            binary_counts = np.sum(struct_features[:, [3, 4, 5, 6, 7]], axis=0)  # indices 12-16 -> 3-7 in struct_features
            binary_names = [struct_names[j] for j in [3, 4, 5, 6, 7]]
            
            bars2 = ax2.bar(binary_names, binary_counts)
            ax2.set_title(f'{complex_id}: Structural Features')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, count in zip(bars2, binary_counts):
                if count > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{int(count)}', ha='center', va='bottom')
            
            # Plot 3: Continuous features distribution
            ax3 = axes[i, 2]
            
            # Create subplots for continuous features
            continuous_data = []
            continuous_labels = []
            
            # Hybridization
            hyb_values = features[:, 9][features[:, 9] > 0]
            if len(hyb_values) > 0:
                continuous_data.append(hyb_values)
                continuous_labels.append('Hybridization')
            
            # Heavy bonds
            heavy_bonds = features[:, 10][features[:, 10] > 0]
            if len(heavy_bonds) > 0:
                continuous_data.append(heavy_bonds)
                continuous_labels.append('Heavy Bonds')
            
            # Hetero bonds
            hetero_bonds = features[:, 11][features[:, 11] > 0]
            if len(hetero_bonds) > 0:
                continuous_data.append(hetero_bonds)
                continuous_labels.append('Hetero Bonds')
            
            # Partial charges (non-zero)
            charges = features[:, 17][np.abs(features[:, 17]) > 0.01]
            if len(charges) > 0:
                ax3.hist(charges, bins=20, alpha=0.7, label='Partial Charges')
                ax3.set_xlabel('Partial Charge')
                ax3.set_ylabel('Count')
                ax3.set_title(f'{complex_id}: Partial Charge Distribution')
                ax3.legend()
            
            # Print summary statistics
            print(f"Hybridization stats: {Counter(features[:, 9])}")
            print(f"Heavy bonds: min={features[:, 10].min()}, max={features[:, 10].max()}, mean={features[:, 10].mean():.2f}")
            print(f"Hetero bonds: min={features[:, 11].min()}, max={features[:, 11].max()}, mean={features[:, 11].mean():.2f}")
            print(f"Partial charges: min={features[:, 17].min():.3f}, max={features[:, 17].max():.3f}, mean={features[:, 17].mean():.3f}")
        
        plt.tight_layout()
        plt.savefig('/home/karen/Projects/FAST/feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create a summary table
        summary_data = []
        for complex_id in complex_ids:
            if complex_id in f:
                data = f[complex_id]['pybel']['processed']['pdbbind']['data'][:]
                features = data[:, 3:]
                
                ligand_count = np.sum(features[:, 18] == 1)
                protein_count = np.sum(features[:, 18] == -1)
                
                summary_data.append({
                    'Complex': complex_id,
                    'Affinity': f[complex_id].attrs['affinity'],
                    'Total_Atoms': data.shape[0],
                    'Ligand_Atoms': ligand_count,
                    'Protein_Atoms': protein_count,
                    'C_atoms': np.sum(features[:, 1]),
                    'N_atoms': np.sum(features[:, 2]),
                    'O_atoms': np.sum(features[:, 3]),
                    'Aromatic_atoms': np.sum(features[:, 13]),
                    'Acceptor_atoms': np.sum(features[:, 14]),
                    'Donor_atoms': np.sum(features[:, 15]),
                    'Avg_partial_charge': np.mean(features[:, 17]),
                })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n=== Summary Table ===")
        print(summary_df.to_string(index=False))
        
        return summary_df

if __name__ == "__main__":
    # Analyze features
    hdf_file = '/home/karen/Projects/FAST/data/refined.hdf'
    
    # You can specify specific complexes or let it pick the first few
    # complex_ids = ['1abc', '2def', '3ghi']  # Specify complexes
    complex_ids = ['966c', '8cpa', '8a3h'] # Let it pick first 3
    
    summary = analyze_complex_features(hdf_file, complex_ids=complex_ids, n_complexes=3)
    
    # Save summary to CSV
    summary.to_csv('/home/karen/Projects/FAST/feature_summary.csv', index=False)
    print("\nSummary saved to feature_summary.csv")
    print("Plots saved to feature_analysis.png")