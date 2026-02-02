# Example: How to use the protein_data_reader.py with your processed data

# Import the new data reader
import sys
sys.path.append('data_new')  # Add data directory to path
from protein_data_reader import ProteinGridDataset, SimpleProteinDataset, create_data_loaders

# Option 1: Use your existing matched arrays (recommended since you already have them)
if 'matched_proteins' in locals():
    print("Using existing matched arrays...")
    
    # Create dataset from your existing variables
    dataset = SimpleProteinDataset(
        matched_proteins=matched_proteins,
        matched_ligands=matched_ligands, 
        matched_pockets=matched_pockets,
        binding_energies=binding_labels,  # Your binding energy labels
        mol_ids=common_ids_list,
        normalize=True
    )
    
    # Create train/validation data loaders
    train_loader, val_loader = create_data_loaders(
        dataset, 
        batch_size=8, 
        train_split=0.8,
        random_seed=42
    )
    
    print(f"✅ Dataset created with {len(dataset)} samples")
    print(f"📊 Training batches: {len(train_loader)}")
    print(f"📊 Validation batches: {len(val_loader)}")
    
    # Get normalization parameters for later denormalization
    norm_params = dataset.get_normalization_params()
    print(f"🔧 Normalization - Mean: {norm_params['binding_mean']:.3f}, Std: {norm_params['binding_std']:.3f}")

else:
    print("Matched arrays not found. Using file-based loading...")
    
    # Option 2: Load directly from your .npy files
    dataset = ProteinGridDataset(
        protein_grids_path="processed_protein_data/protein_grids.npy",
        ligand_grids_path="processed_ligand_data/ligand_grids.npy", 
        pocket_grids_path="processed_pocket_data/pocket_grids.npy",
        protein_metadata_path="processed_protein_data/protein_metadata.pkl",
        ligand_metadata_path="ligand_metadata.json",
        pocket_metadata_path="pocket_metadata.json",
        binding_energy_csv="pdbbind_with_dG.csv",
        normalize=True,
        max_samples=None  # Use all available samples
    )
    
    train_loader, val_loader = create_data_loaders(dataset, batch_size=8)

# Test the data loader
print("\n🧪 Testing data loader...")
for i, batch in enumerate(train_loader):
    print(f"Batch {i+1} shapes:")
    print(f"  Protein: {batch['protein'].shape}")  # Should be [batch_size, channels, depth, height, width]
    print(f"  Ligand: {batch['ligand'].shape}")
    print(f"  Pocket: {batch['pocket'].shape}")
    print(f"  Binding Energy: {batch['binding_energy'].shape}")  # Should be [batch_size, 1]
    print(f"  Sample Mol IDs: {batch['mol_id'][:3]}...")  # First 3 molecular IDs
    
    if i == 0:  # Only show first batch
        break
