"""
CNN Model Architectures for Protein-Ligand Binding Affinity Prediction

This module contains the best performing CNN architectures extracted from step5_basicML.ipynb
optimized for use with the protein_data_reader.py data loading pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StableMultiComponentCNN(nn.Module):
    """
    Stable Multi-Component CNN for protein-ligand binding affinity prediction.
    
    This model processes protein, ligand, and pocket grids separately through
    dedicated encoders, then fuses the features for final prediction.
    """
    
    def __init__(self, protein_channels=19, ligand_channels=19, pocket_channels=19, num_classes=1):
        super(StableMultiComponentCNN, self).__init__()
        
        # Component-specific encoders with proper initialization
        self.protein_encoder = nn.Sequential(
            nn.Conv3d(protein_channels, 16, 3, padding=1),  # Smaller initial channels
            nn.BatchNorm3d(16),  # Add batch normalization
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(4)
        )
        
        self.ligand_encoder = nn.Sequential(
            nn.Conv3d(ligand_channels, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(4)
        )
        
        self.pocket_encoder = nn.Sequential(
            nn.Conv3d(pocket_channels, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(4)
        )
        
        # Fusion layer with dropout
        self.fusion = nn.Sequential(
            nn.Linear(64 * 4 * 4 * 4 * 3, 256),  # Smaller hidden layer
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classifier for regression (binding affinity prediction)
        self.classifier = nn.Linear(64, num_classes)
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Proper weight initialization"""
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm3d, nn.BatchNorm1d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        
    def forward(self, protein_grid, ligand_grid, pocket_grid):
        # Encode each component
        protein_features = self.protein_encoder(protein_grid).flatten(1)
        ligand_features = self.ligand_encoder(ligand_grid).flatten(1)
        pocket_features = self.pocket_encoder(pocket_grid).flatten(1)
        
        # Fuse features
        combined = torch.cat([protein_features, ligand_features, pocket_features], dim=1)
        fused = self.fusion(combined)
        
        # Predict binding affinity
        output = self.classifier(fused)
        return output


class MemoryEfficientCNN(nn.Module):
    """
    Memory-efficient CNN for protein-ligand binding affinity prediction.
    
    This model uses smaller feature maps and aggressive pooling to reduce
    memory usage while maintaining predictive performance.
    """
    
    def __init__(self, protein_channels=19, ligand_channels=19, pocket_channels=19, num_classes=1):
        super(MemoryEfficientCNN, self).__init__()
        
        # Smaller encoders to reduce memory usage
        self.protein_encoder = nn.Sequential(
            nn.Conv3d(protein_channels, 8, 3, padding=1),  # Even smaller channels
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(4),  # Larger pooling for faster reduction
            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(2)  # Smaller output
        )
        
        self.ligand_encoder = nn.Sequential(
            nn.Conv3d(ligand_channels, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(4),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(2)
        )
        
        self.pocket_encoder = nn.Sequential(
            nn.Conv3d(pocket_channels, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(4),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(2)
        )
        
        # Smaller fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(16 * 2 * 2 * 2 * 3, 64),  # Much smaller
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classifier
        self.classifier = nn.Linear(16, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm3d, nn.BatchNorm1d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        
    def forward(self, protein_grid, ligand_grid, pocket_grid):
        # Encode each component
        protein_features = self.protein_encoder(protein_grid).flatten(1)
        ligand_features = self.ligand_encoder(ligand_grid).flatten(1)
        pocket_features = self.pocket_encoder(pocket_grid).flatten(1)
        
        # Fuse features
        combined = torch.cat([protein_features, ligand_features, pocket_features], dim=1)
        fused = self.fusion(combined)
        
        # Classify
        output = self.classifier(fused)
        return output


class SimpleBindingCNN(nn.Module):
    """
    Simple and lightweight CNN for rapid prototyping and testing.
    
    This model uses minimal layers and aggressive dimensionality reduction
    for fast training and inference.
    """
    
    def __init__(self, protein_channels=19, ligand_channels=19, pocket_channels=19):
        super(SimpleBindingCNN, self).__init__()
        
        # Very small encoders
        self.protein_conv = nn.Sequential(
            nn.Conv3d(protein_channels, 4, 5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(4)
        )
        
        self.ligand_conv = nn.Sequential(
            nn.Conv3d(ligand_channels, 4, 5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(4)
        )
        
        self.pocket_conv = nn.Sequential(
            nn.Conv3d(pocket_channels, 4, 5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(4)
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(4*4*4*4*3, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )
        
    def forward(self, protein, ligand, pocket):
        p_feat = self.protein_conv(protein).flatten(1)
        l_feat = self.ligand_conv(ligand).flatten(1)
        k_feat = self.pocket_conv(pocket).flatten(1)
        
        combined = torch.cat([p_feat, l_feat, k_feat], dim=1)
        return self.classifier(combined)


def get_model(model_name, **kwargs):
    """
    Factory function to create models by name.
    
    Args:
        model_name (str): One of 'stable', 'memory_efficient', 'simple'
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        torch.nn.Module: The requested model
    """
    models = {
        'stable': StableMultiComponentCNN,
        'memory_efficient': MemoryEfficientCNN,
        'simple': SimpleBindingCNN
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)


def model_summary(model, input_shape=(1, 19, 64, 64, 64)):
    """
    Print a summary of the model including parameter count and memory usage.
    
    Args:
        model (torch.nn.Module): The model to summarize
        input_shape (tuple): Shape of input tensors (batch_size, channels, D, H, W)
    """
    model.eval()
    
    # Create dummy inputs
    device = next(model.parameters()).device
    dummy_protein = torch.randn(*input_shape).to(device)
    dummy_ligand = torch.randn(*input_shape).to(device) 
    dummy_pocket = torch.randn(*input_shape).to(device)
    
    # Forward pass to get output shape
    with torch.no_grad():
        output = model(dummy_protein, dummy_ligand, dummy_pocket)
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (rough approximation)
    input_size = np.prod(input_shape) * 4 * 3  # 3 inputs, 4 bytes per float32
    param_size = total_params * 4  # 4 bytes per parameter
    estimated_memory_mb = (input_size + param_size) / (1024 * 1024)
    
    print(f"\nModel Summary:")
    print(f"{'='*50}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {input_shape} (per component)")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Estimated memory usage: {estimated_memory_mb:.1f} MB")
    print(f"{'='*50}")
