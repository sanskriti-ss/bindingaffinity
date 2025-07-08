"""
Memory-optimized training script for protein-ligand binding affinity prediction.
Uses smaller models, reduced batch sizes, and aggressive memory management.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import gc
import time

# Import our components
from protein_data_reader import ProteinGridDataset
from memory_optimized_models import get_memory_optimized_models, count_parameters

# Loss functions
class CombinedLoss(nn.Module):
    def __init__(self, mse_weight: float = 0.7, mae_weight: float = 0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.mse_weight * self.mse(predictions, targets) + self.mae_weight * self.mae(predictions, targets)

def memory_optimized_training_loop(model, train_loader, val_loader, config):
    """Memory-optimized training loop with aggressive cleanup."""
    
    device = config['device']
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    criterion = CombinedLoss()
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] else None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'val_mae': [],
        'val_mse': []
    }
    
    best_val_loss = float('inf')
    best_r2 = -float('inf')
    patience_counter = 0
    
    print(f"Starting memory-optimized training for {config['num_epochs']} epochs...")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Mixed precision: {config['mixed_precision']}")
    
    for epoch in range(config['num_epochs']):
        # Clear cache at start of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            protein = batch['protein'].to(device, non_blocking=True)
            ligand = batch['ligand'].to(device, non_blocking=True)
            pocket = batch['pocket'].to(device, non_blocking=True)
            targets = batch['binding_energy'].squeeze().to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=config['mixed_precision']):
                predictions = model(protein, ligand, pocket).squeeze()
                loss = criterion(predictions, targets)
                loss = loss / config['accumulation_steps']
            
            # Backward pass
            if config['mixed_precision']:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config['accumulation_steps'] == 0:
                if config['mixed_precision']:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                    optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * config['accumulation_steps']
            num_batches += 1
            
            # Memory cleanup
            del protein, ligand, pocket, targets, predictions, loss
            if batch_idx % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_train_loss = train_loss / max(num_batches, 1)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                protein = batch['protein'].to(device, non_blocking=True)
                ligand = batch['ligand'].to(device, non_blocking=True)
                pocket = batch['pocket'].to(device, non_blocking=True)
                targets = batch['binding_energy'].squeeze().to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast(enabled=config['mixed_precision']):
                    predictions = model(protein, ligand, pocket).squeeze()
                    loss = criterion(predictions, targets)
                
                val_loss += loss.item()
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                
                # Memory cleanup
                del protein, ligand, pocket, targets, predictions, loss
        
        # Calculate metrics
        avg_val_loss = val_loss / len(val_loader)
        val_r2 = r2_score(val_targets, val_predictions)
        val_mae = mean_absolute_error(val_targets, val_predictions)
        val_mse = mean_squared_error(val_targets, val_predictions)
        
        # Update history
        history['val_loss'].append(avg_val_loss)
        history['val_r2'].append(val_r2)
        history['val_mae'].append(val_mae)
        history['val_mse'].append(val_mse)
        
        # Print progress
        print(f"Epoch {epoch+1}/{config['num_epochs']}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val R²: {val_r2:.4f}")
        print(f"  Val MAE: {val_mae:.4f}")
        print(f"  Val MSE: {val_mse:.4f}")
        
        # Early stopping and model saving
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_val_loss = avg_val_loss
            patience_counter = 0
            if config['save_best_model']:
                torch.save(model.state_dict(), f"best_{config['model_name']}_memory_optimized.pth")
                print(f"  New best model saved! R²: {best_r2:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= config['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Memory cleanup at end of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    return history, best_r2

def main():
    """Main training function with memory optimization."""
    
    # Memory-optimized configuration
    config = {
        'batch_size': 1,  # Ultra small batch size
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'early_stopping_patience': 25,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_best_model': True,
        'gradient_clip': 1.0,
        'accumulation_steps': 8,  # Simulate larger batch size
        'mixed_precision': True,
    }
    
    print("Loading datasets...")
    
    # Load datasets with memory optimization
    train_dataset = ProteinGridDataset('data/train.hdf')
    val_dataset = ProteinGridDataset('data/val.hdf')
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    # Create data loaders with minimal memory usage
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0,  # No multiprocessing
        pin_memory=False,  # No pinned memory
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    # Get memory-optimized models
    models = get_memory_optimized_models()
    results = []
    
    for model_name, model in models:
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        config['model_name'] = model_name.lower()
        
        try:
            # Train model
            history, best_r2 = memory_optimized_training_loop(
                model, train_loader, val_loader, config
            )
            
            # Record results
            results.append({
                'model': model_name,
                'best_r2': best_r2,
                'final_val_loss': history['val_loss'][-1],
                'final_val_mae': history['val_mae'][-1],
                'final_val_mse': history['val_mse'][-1],
                'parameters': count_parameters(model)
            })
            
            # Save training plots
            save_training_plots(history, model_name)
            
            print(f"\\n{model_name} training completed!")
            print(f"Best R²: {best_r2:.4f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Aggressive cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # Save results summary
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('memory_optimized_model_results.csv', index=False)
        print("\\nFinal Results Summary:")
        print(results_df.to_string(index=False))
        
        # Find best model
        best_model = results_df.loc[results_df['best_r2'].idxmax()]
        print(f"\\nBest performing model: {best_model['model']}")
        print(f"R²: {best_model['best_r2']:.4f}")
        print(f"Parameters: {best_model['parameters']:,}")

def save_training_plots(history, model_name):
    """Save training plots for analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{model_name} Training History')
    
    # Loss plot
    axes[0,0].plot(history['train_loss'], label='Train Loss')
    axes[0,0].plot(history['val_loss'], label='Val Loss')
    axes[0,0].set_title('Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # R² plot
    axes[0,1].plot(history['val_r2'], label='Val R²', color='green')
    axes[0,1].set_title('R² Score')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('R²')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # MAE plot
    axes[1,0].plot(history['val_mae'], label='Val MAE', color='orange')
    axes[1,0].set_title('Mean Absolute Error')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('MAE')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # MSE plot
    axes[1,1].plot(history['val_mse'], label='Val MSE', color='red')
    axes[1,1].set_title('Mean Squared Error')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('MSE')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_memory_optimized_training.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
