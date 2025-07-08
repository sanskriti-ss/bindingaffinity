################################################################################
# Modified data reader for processed protein grids (.npy format)
# Adapted from the original FAST data_reader.py
################################################################################
import os
import sys
import csv
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle


class ProteinGridDataset(Dataset):
    """
    Dataset class for loading processed protein, ligand, and pocket grids
    Compatible with your existing .npy files and metadata
    """
    
    def __init__(self, 
                 protein_grids_path="../processed_protein_data/protein_grids.npy",
                 ligand_grids_path="../processed_ligand_data/ligand_grids.npy", 
                 pocket_grids_path="../processed_pocket_data/pocket_grids.npy",
                 protein_metadata_path="../processed_protein_data/metadata.json",
                 ligand_metadata_path="../ligand_metadata.json",
                 pocket_metadata_path="../pocket_metadata.json",
                 binding_energy_csv="../pdbbind_with_dG.csv",
                 normalize=True,
                 max_samples=None):
        
        super(ProteinGridDataset, self).__init__()
        
        self.normalize = normalize
        self.max_samples = max_samples
        
        # Load grid data
        print("Loading grid data...")
        self.protein_grids = np.load(protein_grids_path, allow_pickle=True) 
        self.ligand_grids = np.load(ligand_grids_path, allow_pickle=True) 
        self.pocket_grids = np.load(pocket_grids_path, allow_pickle=True)
        
        # Load metadata
        print("Loading metadata...")
        self.protein_metadata = self._load_metadata(protein_metadata_path)
        self.ligand_metadata = self._load_metadata(ligand_metadata_path)
        self.pocket_metadata = self._load_metadata(pocket_metadata_path)
        
        # Load binding energies
        print("Loading binding energies...")
        self.binding_df = pd.read_csv(binding_energy_csv)
        self.binding_dict = dict(zip(self.binding_df['protein'], self.binding_df['ΔG_kcal_per_mol']))
        
        # Create matched dataset
        print("Creating matched dataset...")
        self.valid_indices = self._create_matched_dataset()
        
        # Normalize if requested
        if self.normalize:
            self._normalize_data()
            
        print(f"Dataset initialized with {len(self.valid_indices)} valid samples")
        
    def _load_metadata(self, path):
        """Load metadata from pickle or json files"""
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif path.endswith('.json'):
            import json
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported metadata format: {path}")
    
    def _create_matched_dataset(self):
        """Create indices for samples that have protein, ligand, pocket, and binding energy data"""
        valid_indices = []
        
        # Get successful IDs from each component
        ligand_ids = set()
        if isinstance(self.ligand_metadata, list):
            for i, entry in enumerate(self.ligand_metadata):
                if entry.get('success', False):
                    mol_id = entry.get('ligand_id') or entry.get('protein_id')
                    if mol_id and i < len(self.ligand_grids):
                        ligand_ids.add((mol_id, i))
        
        pocket_ids = set()
        if isinstance(self.pocket_metadata, list):
            for i, entry in enumerate(self.pocket_metadata):
                if entry.get('success', False):
                    mol_id = entry.get('pocket_id') or entry.get('protein_id')
                    if mol_id and i < len(self.pocket_grids):
                        pocket_ids.add((mol_id, i))
        
        # Find common IDs
        ligand_dict = dict(ligand_ids)
        pocket_dict = dict(pocket_ids)
        common_ids = set(ligand_dict.keys()) & set(pocket_dict.keys())
        
        # Filter by available binding energies and protein grids
        for i, mol_id in enumerate(sorted(common_ids)):
            if (mol_id in self.binding_dict and 
                i < len(self.protein_grids) and
                mol_id in ligand_dict and 
                mol_id in pocket_dict):
                
                valid_indices.append({
                    'mol_id': mol_id,
                    'protein_idx': i,
                    'ligand_idx': ligand_dict[mol_id],
                    'pocket_idx': pocket_dict[mol_id],
                    'binding_energy': self.binding_dict[mol_id]
                })
                
                if self.max_samples and len(valid_indices) >= self.max_samples:
                    break
        
        return valid_indices
    
    def _normalize_data(self):
        """Normalize grid data to zero mean, unit variance"""
        print("Normalizing grid data...")
        
        # Get valid samples for normalization
        valid_protein_indices = [item['protein_idx'] for item in self.valid_indices]
        valid_ligand_indices = [item['ligand_idx'] for item in self.valid_indices]
        valid_pocket_indices = [item['pocket_idx'] for item in self.valid_indices]
        
        # Normalize each grid type
        if len(valid_protein_indices) > 0:
            valid_proteins = self.protein_grids[valid_protein_indices]
            self.protein_mean = np.mean(valid_proteins)
            self.protein_std = np.std(valid_proteins) + 1e-8
            self.protein_grids = (self.protein_grids - self.protein_mean) / self.protein_std
        
        if len(valid_ligand_indices) > 0:
            valid_ligands = self.ligand_grids[valid_ligand_indices]
            self.ligand_mean = np.mean(valid_ligands)
            self.ligand_std = np.std(valid_ligands) + 1e-8
            self.ligand_grids = (self.ligand_grids - self.ligand_mean) / self.ligand_std
        
        if len(valid_pocket_indices) > 0:
            valid_pockets = self.pocket_grids[valid_pocket_indices]
            self.pocket_mean = np.mean(valid_pockets)
            self.pocket_std = np.std(valid_pockets) + 1e-8
            self.pocket_grids = (self.pocket_grids - self.pocket_mean) / self.pocket_std
        
        # Normalize binding energies
        binding_energies = [item['binding_energy'] for item in self.valid_indices]
        self.binding_mean = np.mean(binding_energies)
        self.binding_std = np.std(binding_energies) + 1e-8
    
    def get_normalization_params(self):
        """Return normalization parameters for denormalizing predictions"""
        if self.normalize:
            return {
                'binding_mean': self.binding_mean,
                'binding_std': self.binding_std,
                'protein_mean': getattr(self, 'protein_mean', 0),
                'protein_std': getattr(self, 'protein_std', 1),
                'ligand_mean': getattr(self, 'ligand_mean', 0),
                'ligand_std': getattr(self, 'ligand_std', 1),
                'pocket_mean': getattr(self, 'pocket_mean', 0),
                'pocket_std': getattr(self, 'pocket_std', 1)
            }
        return None
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Get a single sample: protein, ligand, pocket grids + binding energy"""
        item = self.valid_indices[idx]
        
        # Get grids
        protein_grid = torch.tensor(self.protein_grids[item['protein_idx']], dtype=torch.float32)
        ligand_grid = torch.tensor(self.ligand_grids[item['ligand_idx']], dtype=torch.float32)
        pocket_grid = torch.tensor(self.pocket_grids[item['pocket_idx']], dtype=torch.float32)
        
        # Get binding energy (normalized if requested)
        binding_energy = item['binding_energy']
        if self.normalize:
            binding_energy = (binding_energy - self.binding_mean) / self.binding_std
        
        binding_energy = torch.tensor([binding_energy], dtype=torch.float32)
        
        return {
            'protein': protein_grid,
            'ligand': ligand_grid, 
            'pocket': pocket_grid,
            'binding_energy': binding_energy,
            'mol_id': item['mol_id']
        }


class SimpleProteinDataset(Dataset):
    """
    Simplified dataset that just loads your existing matched data
    Use this if you already have matched protein/ligand/pocket arrays
    """
    
    def __init__(self, 
                 matched_proteins, 
                 matched_ligands, 
                 matched_pockets, 
                 binding_energies,
                 mol_ids,
                 normalize=True):
        
        self.matched_proteins = matched_proteins
        self.matched_ligands = matched_ligands  
        self.matched_pockets = matched_pockets
        self.binding_energies = binding_energies
        self.mol_ids = mol_ids
        self.normalize = normalize
        
        if self.normalize:
            # Normalize binding energies
            self.binding_mean = np.mean(binding_energies)
            self.binding_std = np.std(binding_energies) + 1e-8
            self.normalized_energies = (binding_energies - self.binding_mean) / self.binding_std
        else:
            self.normalized_energies = binding_energies
            self.binding_mean = 0
            self.binding_std = 1
    
    def get_normalization_params(self):
        return {
            'binding_mean': self.binding_mean,
            'binding_std': self.binding_std
        }
    
    def __len__(self):
        return len(self.matched_proteins)
    
    def __getitem__(self, idx):
        return {
            'protein': torch.tensor(self.matched_proteins[idx], dtype=torch.float32),
            'ligand': torch.tensor(self.matched_ligands[idx], dtype=torch.float32),
            'pocket': torch.tensor(self.matched_pockets[idx], dtype=torch.float32),
            'binding_energy': torch.tensor([self.normalized_energies[idx]], dtype=torch.float32),
            'mol_id': self.mol_ids[idx]
        }


def create_data_loaders(dataset, batch_size=8, train_split=0.8, random_seed=42):
    """
    Create train/validation data loaders from dataset
    """
    from torch.utils.data import DataLoader, random_split
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader


# Example usage function
def example_usage():
    """
    Example of how to use the protein dataset classes
    """
    
    # Option 1: Load from your processed files
    dataset = ProteinGridDataset(
        protein_grids_path="../processed_protein_data/protein_grids.npy",
        ligand_grids_path="../processed_ligand_data/ligand_grids.npy",
        pocket_grids_path="../processed_pocket_data/pocket_grids.npy",
        protein_metadata_path="../processed_protein_data/metadata.json",
        ligand_metadata_path="../ligand_metadata.json",
        pocket_metadata_path="../pocket_metadata.json",
        binding_energy_csv="../pdbbind_with_dG.csv",
        normalize=True,
        max_samples=10  # Small number for testing
    )
    
    # Option 2: Use existing matched arrays (if you have them)
    # dataset = SimpleProteinDataset(
    #     matched_proteins, matched_ligands, matched_pockets, 
    #     binding_energies, mol_ids, normalize=True
    # )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(dataset, batch_size=4)
    
    # Test loading a batch
    for batch in train_loader:
        print("Batch shapes:")
        print(f"  Protein: {batch['protein'].shape}")
        print(f"  Ligand: {batch['ligand'].shape}")
        print(f"  Pocket: {batch['pocket'].shape}")
        print(f"  Binding Energy: {batch['binding_energy'].shape}")
        print(f"  Mol IDs: {batch['mol_id']}")
        break
    
    return dataset, train_loader, val_loader


if __name__ == "__main__":
    # Run example
    dataset, train_loader, val_loader = example_usage()
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
