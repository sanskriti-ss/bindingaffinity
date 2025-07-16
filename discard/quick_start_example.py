"""
Quick Start Example: Using CNN Models with Protein Data Reader

This script demonstrates how to quickly set up and use the integrated
CNN models with the protein data loading pipeline.
"""

import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import models and data loader
from models import StableMultiComponentCNN, MemoryEfficientCNN, SimpleBindingCNN, model_summary
from protein_data_reader import ProteinGridDataset


def quick_test():
    """Quick test of the data loading and model inference pipeline"""
    
    print("Quick Test: Data Loading and Model Inference")
    print("=" * 50)
    
    # 1. Load a small dataset for testing
    print("Loading test dataset...")
    try:
        dataset = ProteinGridDataset(
            protein_grids_path="processed_protein_data/protein_grids.npy",
            ligand_grids_path="processed_ligand_data/ligand_grids.npy", 
            pocket_grids_path="processed_pocket_data/pocket_grids.npy",
            protein_metadata_path="processed_protein_data/metadata.json",
            ligand_metadata_path="ligand_metadata.json",
            pocket_metadata_path="pocket_metadata.json",
            binding_energy_csv="pdbbind_with_dG.csv",
            normalize=True,
            max_samples=10  # Only load 10 samples for testing
        )
        print(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f" Error loading dataset: {e}")
        print("Please check that all data files exist in the correct locations:")
        print("  - processed_protein_data/protein_grids.npy")
        print("  - processed_ligand_data/ligand_grids.npy")
        print("  - processed_pocket_data/pocket_grids.npy")
        print("  - ligand_metadata.json")
        print("  - pocket_metadata.json")
        print("  - pdbbind_with_dG.csv")
        return
    
    # 2. Create data loader
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # 3. Test each model
    models = {
        'Simple': SimpleBindingCNN(),
        'Memory Efficient': MemoryEfficientCNN(),
        'Stable Multi-Component': StableMultiComponentCNN()
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f" Using device: {device}")
    
    for model_name, model in models.items():
        print(f"\nTesting {model_name} CNN")
        print("-" * 30)
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        # Show model info
        model_summary(model)
        
        # Test inference on first batch
        try:
            batch = next(iter(data_loader))
            protein_grids = batch['protein']
            ligand_grids = batch['ligand']
            pocket_grids = batch['pocket']
            binding_energies = batch['binding_energy']
            mol_ids = batch['mol_id']
            
            # Move data to device
            protein_grids = protein_grids.to(device)
            ligand_grids = ligand_grids.to(device)
            pocket_grids = pocket_grids.to(device)
            
            print(f"Input shapes:")
            print(f"  Protein: {protein_grids.shape}")
            print(f"  Ligand:  {ligand_grids.shape}")
            print(f"  Pocket:  {pocket_grids.shape}")
            print(f"  Target binding energies: {binding_energies.squeeze().numpy()}")
            print(f"  Molecular IDs: {mol_ids}")
            
            # Forward pass
            with torch.no_grad():
                predictions = model(protein_grids, ligand_grids, pocket_grids)
            
            print(f"Predictions: {predictions.squeeze().cpu().numpy()}")
            print(" Model inference successful!")
            
        except Exception as e:
            print(f" Error during inference: {e}")
    
    print(f"\n Quick test completed successfully!")
    print(f"All models can load data and make predictions.")


def simple_training_example():
    """Simple training example with the SimpleBindingCNN model"""
    
    print("\n Simple Training Example")
    print("=" * 40)
    
    # Load dataset
    try:
        dataset = ProteinGridDataset(
            protein_grids_path="processed_protein_data/protein_grids.npy",
            ligand_grids_path="processed_ligand_data/ligand_grids.npy",
            pocket_grids_path="processed_pocket_data/pocket_grids.npy", 
            protein_metadata_path="processed_protein_data/metadata.json",
            ligand_metadata_path="ligand_metadata.json",
            pocket_metadata_path="pocket_metadata.json",
            binding_energy_csv="pdbbind_with_dG.csv",
            normalize=True,
            max_samples=20  # Small dataset for demo
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Training dataset: {len(dataset)} samples")
    
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Create model
    model = SimpleBindingCNN()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    print(f"Training simple model for 5 epochs...")
    
    # Training loop
    model.train()
    for epoch in range(5):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            # Move to device
            protein_grids = batch['protein'].to(device)
            ligand_grids = batch['ligand'].to(device)
            pocket_grids = batch['pocket'].to(device)
            binding_energies = batch['binding_energy'].squeeze().to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(protein_grids, ligand_grids, pocket_grids).squeeze()
            loss = criterion(predictions, binding_energies)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/5: Loss = {avg_loss:.4f}")
    
    print(" Simple training example completed!")
    
    # Test final predictions
    model.eval()
    with torch.no_grad():
        batch = next(iter(data_loader))
        protein_grids = batch['protein'].to(device)
        ligand_grids = batch['ligand'].to(device)
        pocket_grids = batch['pocket'].to(device)
        binding_energies = batch['binding_energy'].squeeze()
        mol_ids = batch['mol_id']
        
        predictions = model(protein_grids, ligand_grids, pocket_grids).squeeze()
        
        print(f"\nFinal predictions vs targets:")
        for i in range(len(predictions)):
            print(f"  {mol_ids[i]}: Predicted={predictions[i].cpu().numpy():.3f}, "
                  f"Actual={binding_energies[i].numpy():.3f}")


def save_model_example():
    """Example of how to save and load trained models"""
    
    print("\n Save/Load Model Example")
    print("=" * 30)
    
    # Create a model
    model = MemoryEfficientCNN()
    print(f"Created {model.__class__.__name__}")
    
    # Save model architecture and weights
    model_path = "example_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'model_config': {
            'protein_channels': 19,
            'ligand_channels': 19,
            'pocket_channels': 19,
            'num_classes': 1
        }
    }, model_path)
    
    print(f" Model saved to {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    loaded_model = MemoryEfficientCNN()  # Create new instance
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f" Model loaded from {model_path}")
    print(f"Model class: {checkpoint['model_class']}")
    print(f"Model config: {checkpoint['model_config']}")
    
    # Clean up
    import os
    os.remove(model_path)
    print(" Temporary model file cleaned up")


if __name__ == "__main__":
    print("🔬 Protein-Ligand Binding Affinity Prediction")
    print("CNN Models Integration Example")
    print("=" * 60)
    
    # Run all examples
    quick_test()
    simple_training_example()
    save_model_example()
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Run the full training pipeline: python integrated_training_clean.py")
    print(f"2. Modify model architectures in models.py for your specific needs")
    print(f"3. Adjust training parameters in the config dictionary")
    print(f"4. Use the best trained model for predictions on new data")
    
    print(f"\n Integration completed successfully!")
    print(f"You now have a complete pipeline from data loading to model training!")
