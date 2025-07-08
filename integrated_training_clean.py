"""
Integrated Training Script for Protein-Ligand Binding Affinity Prediction

This script integrates the robust data loading pipeline from protein_data_reader.py
with the best CNN architectures from step5_basicML.ipynb for end-to-end training.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import our CNN models and data reader
from models import (StableMultiComponentCNN, MemoryEfficientCNN, SimpleBindingCNN, HybridCNNGNN, 
                   AttentionEnhancedCNN, LightweightAttentionCNN, get_model, model_summary)
from protein_data_reader import ProteinGridDataset


def safe_model_initialization(model):
    """Initialize model parameters safely to avoid NaN gradients."""
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    # Check for NaN in initialized parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"Warning: NaN detected in parameter {name} after initialization")
            param.data.normal_(0, 0.01)  # Re-initialize with small normal distribution


def filter_nan_samples(dataset):
    """Remove samples with NaN values from the dataset."""
    print("Filtering NaN samples from dataset...")
    valid_indices = []
    
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            protein_grid = sample['protein']
            ligand_grid = sample['ligand'] 
            pocket_grid = sample['pocket']
            binding_energy = sample['binding_energy']
            
            # Check if all data is finite
            if (np.isfinite(protein_grid).all() and 
                np.isfinite(ligand_grid).all() and 
                np.isfinite(pocket_grid).all() and 
                np.isfinite(binding_energy).all()):
                valid_indices.append(i)
        except Exception as e:
            print(f"Error checking sample {i}: {e}")
            continue
    
    print(f"Filtered dataset: {len(valid_indices)}/{len(dataset)} samples are valid")
    return valid_indices


def check_model_parameters(model):
    """Check model parameters for NaN/Inf values."""
    nan_params = []
    inf_params = []
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
        if torch.isinf(param).any():
            inf_params.append(name)
    
    if nan_params:
        print(f"Warning: NaN parameters detected: {nan_params}")
    if inf_params:
        print(f"Warning: Inf parameters detected: {inf_params}")
    
    return len(nan_params) == 0 and len(inf_params) == 0


def train_model(model, train_loader, val_loader, config):
    """
    Train a model with the given configuration.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
    
    Returns:
        dict: Training history and final model
    """
    
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.get('scheduler_step', 10), 
        gamma=config.get('scheduler_gamma', 0.7)
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'val_mae': [],
        'val_r2': [],
        'learning_rates': []
    }
    
    num_epochs = config.get('num_epochs', 50)
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_loss = float('inf')
    patience = config.get('early_stopping_patience', 15)
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            protein_grids = batch['protein'].to(device)
            ligand_grids = batch['ligand'].to(device)
            pocket_grids = batch['pocket'].to(device)
            binding_energies = batch['binding_energy'].squeeze().to(device)
            
            # Check for NaN in inputs
            if torch.isnan(protein_grids).any() or torch.isnan(ligand_grids).any() or torch.isnan(pocket_grids).any():
                print(f"Warning: NaN detected in input data at epoch {epoch+1}, batch {batch_idx}")
                continue
                
            if torch.isnan(binding_energies).any():
                print(f"Warning: NaN detected in target data at epoch {epoch+1}, batch {batch_idx}")
                continue
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(protein_grids, ligand_grids, pocket_grids).squeeze()
            
            # Check for NaN in outputs before loss calculation
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"Warning: NaN/Inf in model outputs at epoch {epoch+1}, batch {batch_idx}")
                print(f"  Output stats: min={outputs.min():.6f}, max={outputs.max():.6f}, mean={outputs.mean():.6f}")
                continue
            
            loss = criterion(outputs, binding_energies)
            
            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected at epoch {epoch+1}, batch {batch_idx}")
                print(f"  Loss value: {loss.item()}")
                print(f"  Output range: [{outputs.min():.6f}, {outputs.max():.6f}]")
                print(f"  Target range: [{binding_energies.min():.6f}, {binding_energies.max():.6f}]")
                continue
            
            # Backward pass
            loss.backward()
            
            # Check gradients for NaN/Inf
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get('gradient_clip', 1.0))
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"Warning: NaN/Inf gradients detected at epoch {epoch+1}, batch {batch_idx}")
                continue
            
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / max(num_batches, 1)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                protein_grids = batch['protein'].to(device)
                ligand_grids = batch['ligand'].to(device)
                pocket_grids = batch['pocket'].to(device)
                binding_energies = batch['binding_energy'].squeeze().to(device)
                
                outputs = model(protein_grids, ligand_grids, pocket_grids).squeeze()
                
                # Check for NaN in validation outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"Warning: NaN/Inf in validation outputs")
                    continue
                
                loss = criterion(outputs, binding_energies)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    
                    # Handle both scalar and tensor outputs
                    if outputs.dim() == 0:  # Single scalar output
                        val_predictions.append(outputs.cpu().numpy().item())
                    else:  # Multiple outputs
                        val_predictions.extend(outputs.cpu().numpy())
                    
                    # Handle targets similarly
                    if binding_energies.dim() == 0:  # Single scalar target
                        val_targets.append(binding_energies.cpu().numpy().item())
                    else:  # Multiple targets
                        val_targets.extend(binding_energies.cpu().numpy())
        
        avg_val_loss = val_loss / max(len(val_loader), 1)
        history['val_loss'].append(avg_val_loss)
        
        # Calculate validation metrics only if we have valid predictions
        if len(val_predictions) > 0 and len(val_targets) > 0:
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            
            # Check for NaN in the arrays before computing metrics
            if not (np.isnan(val_predictions).any() or np.isnan(val_targets).any()):
                val_mse = mean_squared_error(val_targets, val_predictions)
                val_mae = mean_absolute_error(val_targets, val_predictions)
                val_r2 = r2_score(val_targets, val_predictions) if len(val_targets) > 1 else 0.0
            else:
                print(f"Warning: NaN detected in validation arrays at epoch {epoch+1}")
                val_mse = float('inf')
                val_mae = float('inf')
                val_r2 = -float('inf')
        else:
            print(f"Warning: No valid validation predictions at epoch {epoch+1}")
            val_mse = float('inf')
            val_mae = float('inf')
            val_r2 = -float('inf')
        
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val MAE: {val_mae:.4f}, "
                  f"Val R²: {val_r2:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            if config.get('save_best_model', True):
                torch.save(model.state_dict(), f"best_{model.__class__.__name__.lower()}_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Step scheduler
        scheduler.step()
    
    return history, model


def plot_training_history(history, model_name):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Validation metrics
    axes[0, 1].plot(history['val_mse'], label='MSE', color='red')
    axes[0, 1].plot(history['val_mae'], label='MAE', color='green')
    axes[0, 1].set_title('Validation Error Metrics')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # R² score
    axes[1, 0].plot(history['val_r2'], label='R² Score', color='purple')
    axes[1, 0].set_title('Validation R² Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate
    axes[1, 1].plot(history['learning_rates'], label='Learning Rate', color='orange')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale('log')
    
    plt.suptitle(f'{model_name} Training History', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            protein_grids = batch['protein'].to(device)
            ligand_grids = batch['ligand'].to(device)
            pocket_grids = batch['pocket'].to(device)
            
            # Handle different model types
            if isinstance(model, HybridCNNGNN):
                # For hybrid models, create placeholder graph data
                graph_data = create_graph_data_from_grids(protein_grids, ligand_grids, pocket_grids)
                outputs = model(protein_grids, ligand_grids, pocket_grids, graph_data).squeeze()
            else:
                # Standard CNN models
                outputs = model(protein_grids, ligand_grids, pocket_grids).squeeze()
            
            # Handle both scalar and tensor outputs
            if outputs.dim() == 0:  # Single scalar output
                predictions.append(outputs.cpu().numpy().item())
            else:  # Multiple outputs
                predictions.extend(outputs.cpu().numpy())
            
            # Handle targets similarly
            binding_energies = batch['binding_energy'].squeeze()
            if binding_energies.dim() == 0:  # Single scalar target
                targets.append(binding_energies.numpy().item())
            else:  # Multiple targets
                targets.extend(binding_energies.numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return {
        'mse': mse,
        'mae': mae, 
        'r2': r2,
        'predictions': predictions,
        'targets': targets
    }


def create_predictions_plot(targets, predictions, model_name):
    """Create actual vs predicted plot"""
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, predictions, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Binding Energy (kcal/mol)')
    plt.ylabel('Predicted Binding Energy (kcal/mol)')
    plt.title(f'{model_name}: Actual vs Predicted Binding Energies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R² annotation
    r2 = r2_score(targets, predictions)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


def validate_dataset(dataset, max_samples=10):
    """
    Validate dataset for NaN values and extreme outliers
    """
    print("🔍 Validating dataset for NaN values and outliers...")
    
    nan_count = 0
    extreme_count = 0
    
    for i in range(min(len(dataset), max_samples)):
        sample = dataset[i]
        
        # Check protein grid
        if torch.isnan(sample['protein']).any():
            print(f"  NaN found in protein grid at index {i}")
            nan_count += 1
            
        # Check ligand grid  
        if torch.isnan(sample['ligand']).any():
            print(f"  NaN found in ligand grid at index {i}")
            nan_count += 1
            
        # Check pocket grid
        if torch.isnan(sample['pocket']).any():
            print(f"  NaN found in pocket grid at index {i}")
            nan_count += 1
            
        # Check binding energy
        if torch.isnan(sample['binding_energy']).any():
            print(f"  NaN found in binding energy at index {i}")
            nan_count += 1
            
        # Check for extreme values
        binding_energy = sample['binding_energy'].item()
        if abs(binding_energy) > 100:  # Unreasonably large binding energy
            print(f"  Extreme binding energy ({binding_energy:.2f}) at index {i}")
            extreme_count += 1
    
    print(f"  Validation complete: {nan_count} NaN issues, {extreme_count} extreme values found")
    return nan_count == 0 and extreme_count == 0


def print_dataset_stats(dataset):
    """
    Print statistics about the dataset normalization
    """
    print("📊 Dataset Statistics:")
    if hasattr(dataset, 'get_normalization_params') and dataset.get_normalization_params():
        params = dataset.get_normalization_params()
        print(f"  Binding energy normalization:")
        print(f"    Mean: {params.get('binding_mean', 'N/A'):.4f}")
        print(f"    Std:  {params.get('binding_std', 'N/A'):.4f}")
        
        if 'protein_mean' in params:
            print(f"  Protein grid normalization:")
            print(f"    Mean: {params['protein_mean']:.6f}, Std: {params['protein_std']:.6f}")
        if 'ligand_mean' in params:
            print(f"  Ligand grid normalization:")
            print(f"    Mean: {params['ligand_mean']:.6f}, Std: {params['ligand_std']:.6f}")
        if 'pocket_mean' in params:
            print(f"  Pocket grid normalization:")
            print(f"    Mean: {params['pocket_mean']:.6f}, Std: {params['pocket_std']:.6f}")
    else:
        print("  No normalization parameters available")


def create_graph_data_from_grids(protein_grid, ligand_grid, pocket_grid, 
                                distance_threshold=5.0, max_nodes=1000):
    """
    Convert 3D grids to graph representation for graph neural networks.
    
    This function extracts nodes from high-density grid points and creates
    molecular graphs based on spatial proximity and chemical features.
    
    Args:
        protein_grid: 3D tensor representing protein (B, C, D, H, W)
        ligand_grid: 3D tensor representing ligand (B, C, D, H, W)
        pocket_grid: 3D tensor representing pocket (B, C, D, H, W)
        distance_threshold: Maximum distance for edge creation (Angstroms)
        max_nodes: Maximum number of nodes per graph to prevent memory issues
    
    Returns:
        torch_geometric.data.Data or torch_geometric.data.Batch: Graph data object(s)
    """
    try:
        from torch_geometric.data import Data, Batch
        import torch.nn.functional as F
    except ImportError:
        print("Warning: torch_geometric not available, returning None")
        return None
    
    # Handle batch dimensions
    if protein_grid.dim() == 5:  # Batched input (B, C, D, H, W)
        batch_size = protein_grid.size(0)
        graph_list = []
        
        for b in range(batch_size):
            # Process each sample in the batch
            single_graph = create_single_graph_from_grids(
                protein_grid[b], ligand_grid[b], pocket_grid[b],
                distance_threshold, max_nodes
            )
            if single_graph is not None:
                graph_list.append(single_graph)
        
        if len(graph_list) == 0:
            return None
        elif len(graph_list) == 1:
            return graph_list[0]
        else:
            return Batch.from_data_list(graph_list)
    
    else:  # Single sample (C, D, H, W)
        return create_single_graph_from_grids(
            protein_grid, ligand_grid, pocket_grid,
            distance_threshold, max_nodes
        )


def create_single_graph_from_grids(protein_grid, ligand_grid, pocket_grid,
                                  distance_threshold=5.0, max_nodes=1000):
    """
    Create a single graph from individual 3D grids.
    
    Args:
        protein_grid: 3D tensor (C, D, H, W)
        ligand_grid: 3D tensor (C, D, H, W) 
        pocket_grid: 3D tensor (C, D, H, W)
        distance_threshold: Maximum distance for edge creation
        max_nodes: Maximum number of nodes
    
    Returns:
        torch_geometric.data.Data: Single graph object
    """
    try:
        from torch_geometric.data import Data
        import torch
        
        device = protein_grid.device
        
        # Extract nodes and features from grids
        nodes, features = extract_nodes_from_grids(
            protein_grid, ligand_grid, pocket_grid, max_nodes
        )
        
        if nodes.size(0) == 0:
            # Create minimal graph if no nodes found
            nodes = torch.zeros(1, 3, device=device)
            features = torch.zeros(1, protein_grid.size(0) * 3, device=device)
        
        # Create edges based on spatial proximity
        edge_index, edge_attr = create_edges_from_positions(
            nodes, distance_threshold
        )
        
        # Create graph data object
        graph_data = Data(
            x=features,           # Node features
            edge_index=edge_index, # Edge connectivity  
            edge_attr=edge_attr,   # Edge features
            pos=nodes             # 3D positions (optional)
        )
        
        return graph_data
        
    except Exception as e:
        print(f"Warning: Error creating graph from grids: {e}")
        return None


def extract_nodes_from_grids(protein_grid, ligand_grid, pocket_grid, max_nodes=1000):
    """
    Extract node positions and features from 3D grids.
    
    Strategy:
    1. Find high-density grid points as potential nodes
    2. Extract multi-channel features at those positions
    3. Combine features from protein, ligand, and pocket grids
    
    Args:
        protein_grid: 3D tensor (C, D, H, W)
        ligand_grid: 3D tensor (C, D, H, W)
        pocket_grid: 3D tensor (C, D, H, W)
        max_nodes: Maximum number of nodes to extract
    
    Returns:
        tuple: (node_positions, node_features)
            - node_positions: tensor of shape (N, 3) - 3D coordinates
            - node_features: tensor of shape (N, F) - feature vectors
    """
    import torch
    
    device = protein_grid.device
    grid_shape = protein_grid.shape[1:]  # (D, H, W)
    
    # Compute density maps by summing across channels
    protein_density = torch.sum(torch.abs(protein_grid), dim=0)  # (D, H, W)
    ligand_density = torch.sum(torch.abs(ligand_grid), dim=0)
    pocket_density = torch.sum(torch.abs(pocket_grid), dim=0)
    
    # Combined density map
    combined_density = protein_density + ligand_density + pocket_density
    
    # Check if there's any significant density
    non_zero_values = combined_density[combined_density > 1e-6]
    
    if non_zero_values.numel() == 0:
        # All grids are essentially zero - create a minimal graph at the center
        center_coord = torch.tensor([[d//2, h//2, w//2] for d, h, w in [grid_shape]], 
                                   device=device, dtype=torch.float)
        grid_center = torch.tensor([(d-1)/2 for d in grid_shape], device=device)
        positions = center_coord - grid_center.unsqueeze(0)
        
        # Create minimal features (all zeros)
        num_channels_total = protein_grid.size(0) + ligand_grid.size(0) + pocket_grid.size(0)
        features = torch.zeros(1, num_channels_total, device=device)
        
        return positions, features
    
    # Find significant points (above threshold)
    density_threshold = torch.quantile(non_zero_values, min(0.7, 1.0 - 1.0/non_zero_values.numel()))
    significant_mask = combined_density > density_threshold
    
    # Get 3D coordinates of significant points
    coords = torch.nonzero(significant_mask, as_tuple=False).float()  # (N, 3)
    
    if coords.size(0) == 0:
        # Fallback: use grid center if no significant points found
        center = torch.tensor([d//2 for d in grid_shape], device=device).float().unsqueeze(0)
        coords = center
    
    # Limit number of nodes
    if coords.size(0) > max_nodes:
        # Sample randomly or take highest density points
        densities_at_coords = combined_density[significant_mask]
        _, top_indices = torch.topk(densities_at_coords, max_nodes)
        coords = coords[top_indices]
    
    # Convert grid coordinates to actual 3D positions (assuming 1 Angstrom per grid unit)
    # Center the coordinates around the grid center
    grid_center = torch.tensor([(d-1)/2 for d in grid_shape], device=device)
    positions = coords - grid_center.unsqueeze(0)  # Center around origin
    
    # Extract features at each coordinate
    num_nodes = coords.size(0)
    num_channels_total = protein_grid.size(0) + ligand_grid.size(0) + pocket_grid.size(0)
    features = torch.zeros(num_nodes, num_channels_total, device=device)
    
    for i, coord in enumerate(coords):
        d, h, w = coord.long()
        # Ensure coordinates are within bounds
        d = torch.clamp(d, 0, grid_shape[0] - 1)
        h = torch.clamp(h, 0, grid_shape[1] - 1)
        w = torch.clamp(w, 0, grid_shape[2] - 1)
        
        # Extract features from all three grids
        protein_feat = protein_grid[:, d, h, w]
        ligand_feat = ligand_grid[:, d, h, w]
        pocket_feat = pocket_grid[:, d, h, w]
        
        # Concatenate features
        features[i] = torch.cat([protein_feat, ligand_feat, pocket_feat])
    
    return positions, features


def create_edges_from_positions(positions, distance_threshold=5.0):
    """
    Create edges between nodes based on spatial proximity.
    
    Args:
        positions: tensor of shape (N, 3) - 3D coordinates
        distance_threshold: Maximum distance for edge creation
    
    Returns:
        tuple: (edge_index, edge_attr)
            - edge_index: tensor of shape (2, E) - edge connectivity
            - edge_attr: tensor of shape (E, F) - edge features
    """
    import torch
    
    device = positions.device
    num_nodes = positions.size(0)
    
    if num_nodes <= 1:
        # No edges possible
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        edge_attr = torch.empty(0, 1, device=device)
        return edge_index, edge_attr
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(positions, positions)  # (N, N)
    
    # Create adjacency matrix based on distance threshold
    adj_matrix = (dist_matrix < distance_threshold) & (dist_matrix > 0)  # Exclude self-loops
    
    # Extract edge indices
    edge_indices = torch.nonzero(adj_matrix, as_tuple=False)  # (E, 2)
    
    if edge_indices.size(0) == 0:
        # No edges found, create minimal connectivity
        # Connect each node to its nearest neighbor
        _, nearest_indices = torch.topk(dist_matrix + torch.eye(num_nodes, device=device) * 1e6, 
                                       k=min(2, num_nodes), dim=1, largest=False)
        
        edge_list = []
        for i in range(num_nodes):
            for j in range(min(2, num_nodes)):
                neighbor = nearest_indices[i, j]
                if neighbor != i:
                    edge_list.append([i, neighbor.item()])
        
        if edge_list:
            edge_indices = torch.tensor(edge_list, device=device)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            edge_attr = torch.empty(0, 1, device=device)
            return edge_index, edge_attr
    
    # Transpose to get edge_index format (2, E)
    edge_index = edge_indices.t().contiguous()
    
    # Compute edge features (distances and relative positions)
    edge_distances = dist_matrix[edge_indices[:, 0], edge_indices[:, 1]].unsqueeze(1)
    
    # Relative position vectors
    source_positions = positions[edge_indices[:, 0]]
    target_positions = positions[edge_indices[:, 1]]
    relative_positions = target_positions - source_positions
    
    # Normalize relative positions by distance
    distances_for_norm = edge_distances + 1e-8  # Avoid division by zero
    normalized_relative_pos = relative_positions / distances_for_norm
    
    # Combine distance and normalized relative position as edge features
    edge_attr = torch.cat([edge_distances, normalized_relative_pos], dim=1)  # (E, 4)
    
    return edge_index, edge_attr


def enhanced_train_model(model, train_loader, val_loader, config, use_graph_data=True):
    """
    Enhanced training function that can handle both CNN and hybrid CNN-GNN models.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader  
        config: Training configuration dictionary
        use_graph_data: Whether to use graph data (for hybrid models)
    
    Returns:
        dict: Training history and final model
    """
    
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.get('scheduler_step', 10), 
        gamma=config.get('scheduler_gamma', 0.7)
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'val_mae': [],
        'val_r2': [],
        'learning_rates': []
    }
    
    num_epochs = config.get('num_epochs', 50)
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Graph data enabled: {use_graph_data}")
    
    best_val_loss = float('inf')
    patience = config.get('early_stopping_patience', 15)
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            protein_grids = batch['protein'].to(device)
            ligand_grids = batch['ligand'].to(device)
            pocket_grids = batch['pocket'].to(device)
            binding_energies = batch['binding_energy'].squeeze().to(device)
            
            # Check for NaN in inputs
            if torch.isnan(protein_grids).any() or torch.isnan(ligand_grids).any() or torch.isnan(pocket_grids).any():
                print(f"Warning: NaN detected in input data at epoch {epoch+1}, batch {batch_idx}")
                continue
                
            if torch.isnan(binding_energies).any():
                print(f"Warning: NaN detected in target data at epoch {epoch+1}, batch {batch_idx}")
                continue
            
            # Forward pass
            optimizer.zero_grad()
            
            # Create graph data if using hybrid model
            graph_data = None
            if use_graph_data and hasattr(model, 'use_gnn') and model.use_gnn:
                graph_data = create_graph_data_from_grids(protein_grids, ligand_grids, pocket_grids)
            
            # Model forward pass
            if isinstance(model, HybridCNNGNN):
                outputs = model(protein_grids, ligand_grids, pocket_grids, graph_data).squeeze()
            else:
                outputs = model(protein_grids, ligand_grids, pocket_grids).squeeze()
            
            # Check for NaN in outputs before loss calculation
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"Warning: NaN/Inf in model outputs at epoch {epoch+1}, batch {batch_idx}")
                print(f"  Output stats: min={outputs.min():.6f}, max={outputs.max():.6f}, mean={outputs.mean():.6f}")
                continue
            
            loss = criterion(outputs, binding_energies)
            
            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected at epoch {epoch+1}, batch {batch_idx}")
                print(f"  Loss value: {loss.item()}")
                print(f"  Output range: [{outputs.min():.6f}, {outputs.max():.6f}]")
                print(f"  Target range: [{binding_energies.min():.6f}, {binding_energies.max():.6f}]")
                continue
            
            # Backward pass
            loss.backward()
            
            # Check gradients for NaN/Inf
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get('gradient_clip', 1.0))
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"Warning: NaN/Inf gradients detected at epoch {epoch+1}, batch {batch_idx}")
                continue
            
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / max(num_batches, 1)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                protein_grids = batch['protein'].to(device)
                ligand_grids = batch['ligand'].to(device)
                pocket_grids = batch['pocket'].to(device)
                binding_energies = batch['binding_energy'].squeeze().to(device)
                
                # Create graph data if using hybrid model
                graph_data = None
                if use_graph_data and hasattr(model, 'use_gnn') and model.use_gnn:
                    graph_data = create_graph_data_from_grids(protein_grids, ligand_grids, pocket_grids)
                
                # Model forward pass
                if isinstance(model, HybridCNNGNN):
                    outputs = model(protein_grids, ligand_grids, pocket_grids, graph_data).squeeze()
                else:
                    outputs = model(protein_grids, ligand_grids, pocket_grids).squeeze()
                
                # Check for NaN in validation outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"Warning: NaN/Inf in validation outputs")
                    continue
                
                loss = criterion(outputs, binding_energies)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    
                    # Handle both scalar and tensor outputs
                    if outputs.dim() == 0:  # Single scalar output
                        val_predictions.append(outputs.cpu().numpy().item())
                    else:  # Multiple outputs
                        val_predictions.extend(outputs.cpu().numpy())
                    
                    # Handle targets similarly
                    if binding_energies.dim() == 0:  # Single scalar target
                        val_targets.append(binding_energies.cpu().numpy().item())
                    else:  # Multiple targets
                        val_targets.extend(binding_energies.cpu().numpy())
        
        avg_val_loss = val_loss / max(len(val_loader), 1)
        history['val_loss'].append(avg_val_loss)
        
        # Calculate validation metrics only if we have valid predictions
        if len(val_predictions) > 0 and len(val_targets) > 0:
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            
            # Check for NaN in the arrays before computing metrics
            if not (np.isnan(val_predictions).any() or np.isnan(val_targets).any()):
                val_mse = mean_squared_error(val_targets, val_predictions)
                val_mae = mean_absolute_error(val_targets, val_predictions)
                val_r2 = r2_score(val_targets, val_predictions) if len(val_targets) > 1 else 0.0
            else:
                print(f"Warning: NaN detected in validation arrays at epoch {epoch+1}")
                val_mse = float('inf')
                val_mae = float('inf')
                val_r2 = -float('inf')
        else:
            print(f"Warning: No valid validation predictions at epoch {epoch+1}")
            val_mse = float('inf')
            val_mae = float('inf')
            val_r2 = -float('inf')
        
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val MAE: {val_mae:.4f}, "
                  f"Val R²: {val_r2:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            if config.get('save_best_model', True):
                torch.save(model.state_dict(), f"best_{model.__class__.__name__.lower()}_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Step scheduler
        scheduler.step()
    
    return history, model


def main():
    """Main training and evaluation pipeline"""
    print("Starting Integrated CNN Training Pipeline")
    print("="*60)
    
    # Configuration - Using safer parameters to avoid NaN
    config = {
        'batch_size': 4,  # Small batch size for memory efficiency
        'num_epochs': 50,
        'learning_rate': 1e-5,  # Much lower learning rate to prevent NaN
        'weight_decay': 1e-6,   # Reduced weight decay
        'scheduler_step': 15,
        'scheduler_gamma': 0.9,
        'early_stopping_patience': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_best_model': True,
        'gradient_clip': 0.5,   # More aggressive gradient clipping
        'check_gradients': True  # Enable gradient checking
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Load dataset
    print("Loading dataset...")
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
            max_samples=None  # Use all available samples
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure all data files are in the correct locations.")
        return
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Filter out NaN samples to create a clean dataset
    print("🧹 Filtering dataset to remove NaN samples...")
    valid_indices = filter_nan_samples(dataset)
    
    if len(valid_indices) == 0:
        print("❌ No valid samples found in dataset! All samples contain NaN values.")
        print("Please check your data preprocessing pipeline.")
        return
    
    if len(valid_indices) < len(dataset):
        print(f"⚠️  Filtered out {len(dataset) - len(valid_indices)} samples with NaN values")
        # Create a subset dataset with only valid samples
        from torch.utils.data import Subset
        clean_dataset = Subset(dataset, valid_indices)
        print(f"✅ Clean dataset: {len(clean_dataset)} valid samples")
    else:
        clean_dataset = dataset
        print("✅ All samples are clean - no NaN values detected")
    
    # Validate the clean dataset
    print("🔍 Validating clean dataset...")
    is_valid = validate_dataset(clean_dataset, max_samples=20)
    if not is_valid:
        print("❌ Even after filtering, dataset still contains issues!")
        return
    
    # Print dataset statistics
    print_dataset_stats(clean_dataset)
    
    # Split clean dataset
    train_size = int(0.7 * len(clean_dataset))
    val_size = int(0.15 * len(clean_dataset))
    test_size = len(clean_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        clean_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Models to train - including new attention-enhanced models
    models_to_train = [
        # ('AttentionEnhancedCNN', AttentionEnhancedCNN()),  # New attention model from paper
        ('LightweightAttentionCNN', LightweightAttentionCNN()),  # Lightweight attention model
        # ('StableMultiComponentCNN', StableMultiComponentCNN()),
        ('StableMultiComponentCNN_Attention', StableMultiComponentCNN(use_attention=True)),  # Enhanced with attention
        # ('MemoryEfficientCNN', MemoryEfficientCNN()),
        ('MemoryEfficientCNN_Attention', MemoryEfficientCNN(use_attention=True)),  # Enhanced with attention
        # ('HybridCNNGNN', HybridCNNGNN(use_gnn=True)),  
        # ('SimpleBindingCNN', SimpleBindingCNN()),
    ]
    
    results = {}
    
    # Train each model
    for model_name, model in models_to_train:
        print(f"\nTraining {model_name}")
        print("-" * 40)
        
        # Apply safe model initialization
        print(" Applying safe model initialization...")
        safe_model_initialization(model)
        
        # Check initial model parameters
        is_params_valid = check_model_parameters(model)
        if not is_params_valid:
            print("❌ Model parameters contain NaN/Inf after initialization!")
            continue
        
        # Show model summary
        model_summary(model)
        
        # Train model using enhanced training function for hybrid models
        start_time = time.time()
        if isinstance(model, HybridCNNGNN):
            # Use enhanced training for hybrid models with graph data enabled
            history, trained_model = enhanced_train_model(model, train_loader, val_loader, config, use_graph_data=True)
        else:
            # Use standard training for CNN models
            history, trained_model = train_model(model, train_loader, val_loader, config)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.1f} seconds")
        
        # Plot training history
        plot_training_history(history, model_name)
        
        # Evaluate on test set
        print(f"\nEvaluating {model_name} on test set...")
        test_results = evaluate_model(trained_model, test_loader, config['device'])
        
        print(f"Test Results:")
        print(f"  MSE: {test_results['mse']:.4f}")
        print(f"  MAE: {test_results['mae']:.4f}")
        print(f"  R²:  {test_results['r2']:.4f}")
        
        # Create predictions plot
        create_predictions_plot(test_results['targets'], test_results['predictions'], model_name)
        
        # Store results
        results[model_name] = {
            'history': history,
            'test_results': test_results,
            'training_time': training_time,
            'model': trained_model
        }
    
    # Compare models
    print("\nModel Comparison")
    print("=" * 60)
    print(f"{'Model':<25} {'Test MSE':<12} {'Test MAE':<12} {'Test R²':<12} {'Time (s)':<12}")
    print("-" * 75)
    
    for model_name, result in results.items():
        test_res = result['test_results']
        print(f"{model_name:<25} {test_res['mse']:<12.4f} {test_res['mae']:<12.4f} "
              f"{test_res['r2']:<12.4f} {result['training_time']:<12.1f}")
    
    # Find best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['test_results']['mse'])
    print(f"\n Best model: {best_model_name} (lowest MSE)")
    
    # Save results summary
    summary_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test_MSE': [results[name]['test_results']['mse'] for name in results.keys()],
        'Test_MAE': [results[name]['test_results']['mae'] for name in results.keys()],
        'Test_R2': [results[name]['test_results']['r2'] for name in results.keys()],
        'Training_Time': [results[name]['training_time'] for name in results.keys()]
    })
    
    summary_df.to_csv('model_comparison_results_1.csv', index=False)
    print(f"\nResults saved to 'model_comparison_results.csv'")
    
    print("\nTraining pipeline completed successfully!")
    return results


if __name__ == "__main__":
    results = main()
