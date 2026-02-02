# Training script using the new data reader

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.path.append('data')
from protein_data_reader import SimpleProteinDataset, create_data_loaders

def train_with_data_loader(model, train_loader, val_loader, norm_params, num_epochs=50, device='cpu'):
    """
    Train your model using the data loader
    """
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    train_losses = []
    val_losses = []
    val_maes = []
    val_r2s = []
    
    print(f"🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move data to device
            protein = batch['protein'].to(device)
            ligand = batch['ligand'].to(device) 
            pocket = batch['pocket'].to(device)
            target = batch['binding_energy'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Adjust this based on your model's forward method
            if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 4:
                # For models that take separate inputs
                output = model(protein, ligand, pocket)
            else:
                # For models that take concatenated input
                combined_input = torch.cat([protein, ligand, pocket], dim=1)
                output = model(combined_input)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_targets = []
        epoch_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                protein = batch['protein'].to(device)
                ligand = batch['ligand'].to(device)
                pocket = batch['pocket'].to(device) 
                target = batch['binding_energy'].to(device)
                
                # Forward pass
                if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 4:
                    output = model(protein, ligand, pocket)
                else:
                    combined_input = torch.cat([protein, ligand, pocket], dim=1)
                    output = model(combined_input)
                
                loss = criterion(output, target)
                epoch_val_loss += loss.item()
                val_batches += 1
                
                # Collect predictions for metrics
                val_predictions.extend(output.cpu().numpy().flatten())
                val_targets.extend(target.cpu().numpy().flatten())
        
        avg_val_loss = epoch_val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Calculate metrics (denormalized)
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        
        if norm_params:
            # Denormalize for meaningful metrics
            val_pred_denorm = val_predictions * norm_params['binding_std'] + norm_params['binding_mean'] 
            val_target_denorm = val_targets * norm_params['binding_std'] + norm_params['binding_mean']
        else:
            val_pred_denorm = val_predictions
            val_target_denorm = val_targets
        
        val_mae = mean_absolute_error(val_target_denorm, val_pred_denorm)
        val_r2 = r2_score(val_target_denorm, val_pred_denorm)
        
        val_maes.append(val_mae)
        val_r2s.append(val_r2)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")  
            print(f"  Val MAE: {val_mae:.3f} kcal/mol")
            print(f"  Val R²: {val_r2:.3f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses, 
        'val_maes': val_maes,
        'val_r2s': val_r2s
    }


def predict_with_data_loader(model, data_loader, norm_params, device='cpu'):
    """
    Make predictions using the trained model
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_mol_ids = []
    
    with torch.no_grad():
        for batch in data_loader:
            protein = batch['protein'].to(device)
            ligand = batch['ligand'].to(device)
            pocket = batch['pocket'].to(device)
            target = batch['binding_energy'].to(device)
            mol_ids = batch['mol_id']
            
            # Forward pass
            if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 4:
                output = model(protein, ligand, pocket)
            else:
                combined_input = torch.cat([protein, ligand, pocket], dim=1)
                output = model(combined_input)
            
            all_predictions.extend(output.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
            all_mol_ids.extend(mol_ids)
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Denormalize if needed
    if norm_params:
        predictions = predictions * norm_params['binding_std'] + norm_params['binding_mean']
        targets = targets * norm_params['binding_std'] + norm_params['binding_mean']
    
    return predictions, targets, all_mol_ids


# Example usage (to be run in your notebook)
def example_training():
    """
    Example of how to use the data reader for training
    """
    
    # Assuming you have your existing matched data
    # Replace these with your actual variables
    matched_proteins = "../processed_protein_data/protein_grids.npy"  # Your protein grids
    matched_ligands = "../processed_ligand_data/ligand_grids.npy"   # Your ligand grids  
    matched_pockets = "../processed_pocket_data/pocket_grids.npy"   # Your pocket grids
    binding_labels = "../processed_protein_data/binding_labels.npy"  # Your binding energy labels
    common_ids_list = "../processed_protein_data/common_ids.npy"     # Your molecular IDs

    # Create dataset
    dataset = SimpleProteinDataset(
        matched_proteins=matched_proteins,
        matched_ligands=matched_ligands,
        matched_pockets=matched_pockets, 
        binding_energies=binding_labels,
        mol_ids=common_ids_list,
        normalize=True
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(dataset, batch_size=8)
    
    # Get normalization parameters
    norm_params = dataset.get_normalization_params()
    
    # Initialize your model (replace with your actual model)
    # model = YourModelClass()
    
    
    # Train the model
    # results = train_with_data_loader(model, train_loader, val_loader, norm_params)
    
    # Make predictions
    # predictions, targets, mol_ids = predict_with_data_loader(model, val_loader, norm_params)
    
    return dataset, train_loader, val_loader

if __name__ == "__main__":
    print("This is a template for using the protein data reader.")
    print("Copy the relevant functions to your notebook and adapt as needed.")
