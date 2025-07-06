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
from models import StableMultiComponentCNN, MemoryEfficientCNN, SimpleBindingCNN, get_model, model_summary
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
    
    # Models to train
    models_to_train = [
        ('StableMultiComponentCNN', StableMultiComponentCNN()),
        ('MemoryEfficientCNN', MemoryEfficientCNN()),
        ('SimpleBindingCNN', SimpleBindingCNN())
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
        
        # Train model
        start_time = time.time()
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
    
    summary_df.to_csv('model_comparison_results.csv', index=False)
    print(f"\nResults saved to 'model_comparison_results.csv'")
    
    print("\nTraining pipeline completed successfully!")
    return results


if __name__ == "__main__":
    results = main()
