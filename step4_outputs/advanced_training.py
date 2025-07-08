# advanced_training.py

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

# Import existing components
from integrated_training_clean import (
    safe_model_initialization, filter_nan_samples, check_model_parameters,
    validate_dataset, print_dataset_stats, plot_training_history, 
    evaluate_model, create_predictions_plot, create_graph_data_from_grids
)
from protein_data_reader import ProteinGridDataset

# Import advanced models
from models_2 import (
    AdvancedTransformerCNN, HybridCNNGNNTransformer, ResidualTransformerCNN,
    CombinedLoss, FocalLoss, WarmupLRScheduler, get_advanced_model, 
    advanced_model_summary, TORCH_GEOMETRIC_AVAILABLE
)

# Import existing models for comparison
from models import (
    StableMultiComponentCNN, MemoryEfficientCNN, LightweightAttentionCNN,
    AttentionEnhancedCNN, HybridCNNGNN
)

# Import quantum-enhanced models
from quantum_models import (
    QuantumEnhancedBindingAffinityPredictor, QuantumHybridEnsemble, 
    QuantumResilientCNN, QuantumAwareLoss, create_quantum_model,
    quantum_model_summary
)


def advanced_train_model(model, train_loader, val_loader, config):
    """
    Advanced training function with improved optimization strategies.
    
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
    
    # Advanced optimizer with different parameters for different parts
    base_lr = config.get('learning_rate', 1e-4)
    
    # Different learning rates for different parts of the model
    param_groups = []
    
    # CNN backbone parameters (lower learning rate)
    cnn_params = []
    # Transformer parameters (higher learning rate)
    transformer_params = []
    # Classifier parameters (medium learning rate)
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'transformer' in name.lower() or 'attention' in name.lower():
            transformer_params.append(param)
        elif 'classifier' in name.lower() or 'fusion' in name.lower():
            classifier_params.append(param)
        else:
            cnn_params.append(param)
    
    if cnn_params:
        param_groups.append({'params': cnn_params, 'lr': base_lr * 0.5})
    if transformer_params:
        param_groups.append({'params': transformer_params, 'lr': base_lr * 1.5})
    if classifier_params:
        param_groups.append({'params': classifier_params, 'lr': base_lr})
    
    if not param_groups:
        param_groups = [{'params': model.parameters(), 'lr': base_lr}]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.get('weight_decay', 1e-4),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Advanced loss function
    loss_type = config.get('loss_type', 'combined')
    if loss_type == 'combined':
        criterion = CombinedLoss()
    elif loss_type == 'focal':
        criterion = FocalLoss()
    elif loss_type == 'quantum_aware':
        criterion = QuantumAwareLoss()
    else:
        criterion = nn.MSELoss()
    
    # Learning rate scheduler with warmup
    warmup_scheduler = WarmupLRScheduler(
        optimizer, 
        warmup_epochs=config.get('warmup_epochs', 5),
        base_lr=base_lr * 0.1,
        max_lr=base_lr
    )
    
    # Main scheduler
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=config.get('scheduler_t0', 10),
        T_mult=2,
        eta_min=base_lr * 0.01
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
    
    num_epochs = config.get('num_epochs', 100)
    print(f"Starting advanced training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f" Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loss function: {loss_type}")
    
    best_val_loss = float('inf')
    best_r2 = -float('inf')
    patience = config.get('early_stopping_patience', 25)
    patience_counter = 0
    
    # Set up automatic mixed precision if enabled
    use_amp = config.get('mixed_precision', False)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # Gradient accumulation settings
    accumulation_steps = config.get('gradient_accumulation_steps', 1)
    
    for epoch in range(num_epochs):
        # Clear cache at the start of each epoch
        if config.get('memory_cleanup', False) and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            protein_grids = batch['protein'].to(device, non_blocking=True)
            ligand_grids = batch['ligand'].to(device, non_blocking=True)
            pocket_grids = batch['pocket'].to(device, non_blocking=True)
            binding_energies = batch['binding_energy'].squeeze().to(device, non_blocking=True)
            
            # Check for NaN in inputs
            if torch.isnan(protein_grids).any() or torch.isnan(ligand_grids).any() or torch.isnan(pocket_grids).any():
                print(f"  NaN detected in input data at epoch {epoch+1}, batch {batch_idx}")
                continue
                
            if torch.isnan(binding_energies).any():
                print(f"  NaN detected in target data at epoch {epoch+1}, batch {batch_idx}")
                continue
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda',enabled=use_amp):
                predictions = model(protein_grids, ligand_grids, pocket_grids)
                loss = criterion(predictions.squeeze(), binding_energies)
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
            
            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip', 1.0))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip', 1.0))
                    optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            num_batches += 1
            
            # Memory cleanup every few batches
            if config.get('memory_cleanup', False) and batch_idx % 10 == 0:
                del protein_grids, ligand_grids, pocket_grids, binding_energies, predictions
                torch.cuda.empty_cache()
        
        avg_train_loss = train_loss / max(num_batches, 1)
        history['train_loss'].append(avg_train_loss)
        
        # Learning rate scheduling
        if epoch < config.get('warmup_epochs', 5):
            current_lr = warmup_scheduler.step()
        else:
            main_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        
        history['learning_rates'].append(current_lr)
        
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
                if isinstance(model, HybridCNNGNNTransformer) and TORCH_GEOMETRIC_AVAILABLE:
                    graph_data = create_graph_data_from_grids(protein_grids, ligand_grids, pocket_grids)
                
                # Model forward pass
                if isinstance(model, (HybridCNNGNNTransformer, HybridCNNGNN)):
                    outputs = model(protein_grids, ligand_grids, pocket_grids, graph_data).squeeze()
                else:
                    outputs = model(protein_grids, ligand_grids, pocket_grids).squeeze()
                
                # Check for NaN in validation outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    continue
                
                loss = criterion(outputs, binding_energies)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    
                    # Handle both scalar and tensor outputs
                    if outputs.dim() == 0:
                        val_predictions.append(outputs.cpu().numpy().item())
                    else:
                        val_predictions.extend(outputs.cpu().numpy())
                    
                    # Handle targets similarly
                    if binding_energies.dim() == 0:
                        val_targets.append(binding_energies.cpu().numpy().item())
                    else:
                        val_targets.extend(binding_energies.cpu().numpy())
        
        avg_val_loss = val_loss / max(len(val_loader), 1)
        history['val_loss'].append(avg_val_loss)
        
        # Calculate validation metrics
        if len(val_predictions) > 0 and len(val_targets) > 0:
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            
            if not (np.isnan(val_predictions).any() or np.isnan(val_targets).any()):
                val_mse = mean_squared_error(val_targets, val_predictions)
                val_mae = mean_absolute_error(val_targets, val_predictions)
                val_r2 = r2_score(val_targets, val_predictions) if len(val_targets) > 1 else -float('inf')
            else:
                val_mse = float('inf')
                val_mae = float('inf')
                val_r2 = -float('inf')
        else:
            val_mse = float('inf')
            val_mae = float('inf')
            val_r2 = -float('inf')
        
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        
        # Print progress with emojis and better formatting
        if (epoch + 1) % 5 == 0 or epoch == 0:
            r2_emoji = "🎯" if val_r2 > 0.5 else "📈" if val_r2 > 0 else "📉"
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val MAE: {val_mae:.4f}, "
                  f"Val R²: {val_r2:.4f} {r2_emoji}, "
                  f"LR: {current_lr:.2e}")
        
        # Early stopping with R² consideration
        is_best_loss = avg_val_loss < best_val_loss
        is_best_r2 = val_r2 > best_r2
        
        if is_best_loss or is_best_r2:
            if is_best_loss:
                best_val_loss = avg_val_loss
            if is_best_r2:
                best_r2 = val_r2
            patience_counter = 0
            
            # Save best model
            if config.get('save_best_model', True):
                torch.save(model.state_dict(), f"best_{model.__class__.__name__.lower()}_model.pth")
                if val_r2 > 0.5:
                    print(f"🏆 New best R² score: {val_r2:.4f} - Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered after {epoch+1} epochs")
                print(f" Best R² score achieved: {best_r2:.4f}")
                break
    
    final_message = " Training completed successfully!" if best_r2 > 0.3 else " Training completed - consider tuning hyperparameters"
    print(f"\n{final_message}")
    print(f" Final metrics - Best R²: {best_r2:.4f}, Best Loss: {best_val_loss:.4f}")
    
    return history, model


def create_quantum_model_with_config(model_name, config):
    """
    Create a quantum model with configuration parameters.
    
    Args:
        model_name: Name of the quantum model to create
        config: Training configuration dictionary with quantum parameters
        
    Returns:
        Configured quantum model instance
    """
    quantum_config = {
        'noise_level': config.get('quantum_noise_level', 0.1),
        'error_correction': config.get('quantum_error_correction', True),
        'shots': config.get('quantum_shots', 1024),
        'depth': config.get('quantum_depth', 6),
    }
    
    if model_name == 'QuantumEnhancedBindingAffinityPredictor':
        return QuantumEnhancedBindingAffinityPredictor()
    elif model_name == 'QuantumHybridEnsemble':
        return QuantumHybridEnsemble()
    elif model_name == 'QuantumResilientCNN':
        return QuantumResilientCNN(
            noise_level=quantum_config['noise_level'],
            error_correction=quantum_config['error_correction']
        )
    else:
        # Use factory function if available
        return create_quantum_model(model_name, **quantum_config)


def main():
    """Main training pipeline with advanced models"""
    print(" Starting Advanced CNN-Transformer Training Pipeline")
    print("="*70)

    
    # Advanced configuration - Memory optimized with quantum support
    config = {
        'batch_size': 4,  # Reduced batch size for GPU memory
        'num_epochs': 100,
        'learning_rate': 1e-4,  # Learning rate
        'weight_decay': 1e-4,   # Stronger regularization
        'warmup_epochs': 10,
        'scheduler_t0': 15,
        'early_stopping_patience': 30,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_best_model': True,
        'gradient_clip': 1.0,
        'loss_type': 'combined',  # Use combined loss for better training
        'gradient_accumulation_steps': 4,  # Simulate larger batch size
        'mixed_precision': True,  # Use automatic mixed precision
        'memory_cleanup': True,   # Enable aggressive memory cleanup
        
        # Quantum-specific configurations
        'quantum_enabled': True,           # Enable quantum model training
        'quantum_noise_level': 0.1,       # For QuantumResilientCNN
        'quantum_error_correction': True,  # For robust quantum training
        'quantum_shots': 1024,            # Number of quantum measurements
        'quantum_depth': 6,               # Quantum circuit depth
    }
    
    print(f"  Advanced Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Load dataset
    print(" Loading dataset...")
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
            max_samples=None
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print(f" Dataset loaded: {len(dataset)} samples")
    
    # Filter NaN samples
    print("🧹Filtering dataset...")
    valid_indices = filter_nan_samples(dataset)
    
    if len(valid_indices) == 0:
        print(" No valid samples found!")
        return
    
    if len(valid_indices) < len(dataset):
        clean_dataset = Subset(dataset, valid_indices)
        print(f" Clean dataset: {len(clean_dataset)} valid samples")
    else:
        clean_dataset = dataset
        print(" All samples are clean")
    
    # Validate dataset
    is_valid = validate_dataset(clean_dataset, max_samples=20)
    if not is_valid:
        print(" Dataset validation failed!")
        return
    
    # Print dataset statistics
    print_dataset_stats(clean_dataset)
    
    # Split dataset
    train_size = int(0.75 * len(clean_dataset))  # Larger training set
    val_size = int(0.15 * len(clean_dataset))
    test_size = len(clean_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        clean_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f" Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create data loaders with memory optimization
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=0, pin_memory=False,
                             drop_last=True)  # Drop last incomplete batch
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0, pin_memory=False,
                           drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=0, pin_memory=False,
                            drop_last=False)
    
    # Advanced models to train
    models_to_train = [
        # Classical advanced models
        # ('ResidualTransformerCNN', ResidualTransformerCNN()),
        
        # Quantum-enhanced models
        ('QuantumEnhancedBindingAffinityPredictor', QuantumEnhancedBindingAffinityPredictor()),
        ('QuantumHybridEnsemble', QuantumHybridEnsemble()),
        ('QuantumResilientCNN', QuantumResilientCNN()),
    ]
    
    # Add hybrid model - now works with and without torch_geometric
    models_to_train.append(('HybridCNNGNNTransformer', HybridCNNGNNTransformer()))
    
    
    results = {}
    
    # Train each model
    for model_name, model in models_to_train:
        print(f"\n Training {model_name}")
        print("-" * 50)
        
        # Apply safe initialization
        safe_model_initialization(model)
        
        # Check initial parameters
        if not check_model_parameters(model):
            print(" Model parameters contain NaN/Inf after initialization!")
            continue
        
        # Show model summary (handle both quantum and classical models)
        if 'Quantum' in model_name:
            try:
                quantum_model_summary(model)
            except:
                # Fallback to standard summary for quantum models
                advanced_model_summary(model)
        else:
            advanced_model_summary(model)
        
        # Train model
        start_time = time.time()
        history, trained_model = advanced_train_model(model, train_loader, val_loader, config)
        training_time = time.time() - start_time
        
        print(f"⏱  Training completed in {training_time:.1f} seconds")
        
        # Plot training history
        plot_training_history(history, model_name)
        
        # Evaluate on test set
        print(f"\n Evaluating {model_name} on test set...")
        test_results = evaluate_model(trained_model, test_loader, config['device'])
        
        # Determine performance emoji
        r2_score = test_results['r2']
        if r2_score >= 0.7:
            performance = ">=0.7"
        elif r2_score >= 0.5:
            performance = ">=0.5"
        elif r2_score >= 0.3:
            performance = ">=0.3"
        elif r2_score >= 0:
            performance = ">=0"
        else:
            performance = "<0"
        
        print(f"Test Results {performance}:")
        print(f"   MSE: {test_results['mse']:.4f}")
        print(f"   MAE: {test_results['mae']:.4f}")
        print(f"   R²:  {test_results['r2']:.4f}")
        
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
    print(f"\n Model Comparison")
    print("=" * 80)
    print(f"{'Model':<30} {'Test MSE':<12} {'Test MAE':<12} {'Test R²':<12} {'Status':<10}")
    print("-" * 80)
    
    for model_name, result in results.items():
        test_res = result['test_results']
        r2 = test_res['r2']
        
        # Status indicator
        if r2 >= 0.7:
            status = "EXCELLENT"
        elif r2 >= 0.5:
            status = "GREAT"
        elif r2 >= 0.3:
            status = " MEH"
        elif r2 >= 0:
            status = " BAD"
        else:
            status = " POOR"
        
        print(f"{model_name:<30} {test_res['mse']:<12.4f} {test_res['mae']:<12.4f} "
              f"{r2:<12.4f} {status}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_results']['r2'])
    best_r2 = results[best_model_name]['test_results']['r2']
    
    print(f"\n Best model: {best_model_name} (R² = {best_r2:.4f})")
    
    
    # Save results
    summary_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test_MSE': [results[name]['test_results']['mse'] for name in results.keys()],
        'Test_MAE': [results[name]['test_results']['mae'] for name in results.keys()],
        'Test_R2': [results[name]['test_results']['r2'] for name in results.keys()],
        'Training_Time': [results[name]['training_time'] for name in results.keys()]
    })
    
    summary_df.to_csv('advanced_model_comparison_results.csv', index=False)
    print(f"\n Results saved to 'advanced_model_comparison_results.csv'")
    
    print(f"\n Advanced training pipeline completed!")
    return results


if __name__ == "__main__":
    results = main()
