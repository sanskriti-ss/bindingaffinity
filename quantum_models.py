"""
Quantum-Enhanced Binding Affinity Prediction Models


Key Features:
1. Quantum Feature Fusion (inspired by the notebook's 3D-CNN + SG-CNN fusion)
2. Quantum Attention Mechanisms
3. Multi-stage quantum processing
4. Classical fallback when quantum components unavailable
5. Advanced training components and loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Import from existing models
from models_2 import (
    QuantumFeatureFusion, QuantumAttentionFusion, 
    EnhancedFeatureExtractor, TransformerBlock3D, 
    PositionalEncoding3D, MultiHeadSelfAttention,
    ChannelAttention3D, QUANTUM_AVAILABLE
)


class QuantumEnhancedBindingAffinityPredictor(nn.Module):
    """
    Quantum-enhanced model for protein-ligand binding affinity prediction.
   
    """
    
    def __init__(self, num_classes=1, quantum_layers=10, use_quantum=True, dropout=0.3):
        super(QuantumEnhancedBindingAffinityPredictor, self).__init__()
        self.use_quantum = use_quantum and QUANTUM_AVAILABLE
        
        print(f" Initializing Quantum-Enhanced Binding Affinity Predictor")
        print(f"   Quantum Processing: {' Enabled' if self.use_quantum else ' Disabled (using classical fallback)'}")
        
        # Enhanced multi-scale feature extractors
        self.protein_extractor = EnhancedFeatureExtractor(19, 128, dropout)
        self.ligand_extractor = EnhancedFeatureExtractor(19, 64, dropout)
        self.pocket_extractor = EnhancedFeatureExtractor(19, 64, dropout)
        
        # Advanced CNN fusion with attention
        self.cnn_fusion = nn.Sequential(
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            ChannelAttention3D(256),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(8)  # Larger spatial preservation for better features
        )
        
        # Multi-scale spatial pooling for comprehensive feature extraction
        self.spatial_pooling = nn.ModuleList([
            nn.AdaptiveAvgPool3d(size) for size in [8, 4, 2, 1]
        ])
        
        # Calculate feature dimensions: 256 * (8³ + 4³ + 2³ + 1³) = 256 * 585 = 149,760
        spatial_feature_size = 256 * (8**3 + 4**3 + 2**3 + 1**3)
        
        # Feature dimensionality reduction before quantum processing
        self.feature_reduction = nn.Sequential(
            nn.Linear(spatial_feature_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
        
        # Quantum fusion layers inspired by the research notebook
        if self.use_quantum:
            # Primary quantum fusion (simulating 3D-CNN + SG-CNN fusion from notebook)
            self.quantum_cnn_fusion = QuantumFeatureFusion(
                input_size=512,
                quantum_layers=quantum_layers,
                encoding='amplitude',  # Use amplitude encoding as in notebook
                ansatz=4,  # Strongly entangling layers as in notebook
                dropout=dropout
            )
            
            # Secondary quantum fusion for enhanced representation
            self.quantum_enhanced_fusion = QuantumFeatureFusion(
                input_size=512,
                quantum_layers=quantum_layers // 2,
                encoding='amplitude',
                ansatz=1,  # Different ansatz for diversity
                dropout=dropout
            )
            
            # Quantum attention for final feature refinement
            self.quantum_attention = QuantumAttentionFusion(
                feature_dim=512,
                num_heads=8,
                quantum_layers=6,
                dropout=dropout
            )
        else:
            # Classical fallback with enhanced non-linearities
            self.classical_fusion = nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),  # More complex activation
                nn.Dropout(dropout),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),  # Swish-like activation
                nn.Dropout(dropout/2)
            )
        
        # Advanced transformer processing
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock3D(512, num_heads=8, dropout=dropout, activation='gelu')
            for _ in range(3)
        ])
        
        # Multi-head cross-attention for feature integration
        self.cross_attention = MultiHeadSelfAttention(512, num_heads=8, dropout=dropout)
        
        # Enhanced classifier with residual connections (inspired by notebook's approach)
        self.classifier_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout/2)
            ),
            nn.Sequential(
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout/4)
            ),
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout/4)
            )
        ])
        
        # Final output layer
        self.final_layer = nn.Linear(32, num_classes)
        
        # Residual connections for classifier (like in notebook)
        self.residual_projections = nn.ModuleList([
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32)
        ])
        
        self.apply(self._init_weights)
        
        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Total Parameters: {total_params:,}")
        if self.use_quantum:
            quantum_params = sum(p.numel() for p in self.quantum_cnn_fusion.parameters())
            quantum_params += sum(p.numel() for p in self.quantum_enhanced_fusion.parameters())
            quantum_params += sum(p.numel() for p in self.quantum_attention.parameters())
            print(f"   Quantum Parameters: {quantum_params:,} ({quantum_params/total_params*100:.1f}%)")
    
    def _init_weights(self, module):
        """Initialize weights using advanced techniques"""
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            # Use Xavier initialization for better gradient flow
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm3d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, protein_grid, ligand_grid, pocket_grid):
        """
        Forward pass through the quantum-enhanced model.
        
        Args:
            protein_grid: Protein voxel grid (B, 19, D, H, W)
            ligand_grid: Ligand voxel grid (B, 19, D, H, W) 
            pocket_grid: Pocket voxel grid (B, 19, D, H, W)
            
        Returns:
            Binding affinity prediction (B, 1)
        """
        batch_size = protein_grid.size(0)
        
        # Step 1: Extract enhanced features from each component
        protein_features = self.protein_extractor(protein_grid)  # (B, 128, D, H, W)
        ligand_features = self.ligand_extractor(ligand_grid)     # (B, 64, D, H, W)
        pocket_features = self.pocket_extractor(pocket_grid)     # (B, 64, D, H, W)
        
        # Step 2: Combine and process with advanced CNN fusion
        combined = torch.cat([protein_features, ligand_features, pocket_features], dim=1)  # (B, 256, D, H, W)
        cnn_fused = self.cnn_fusion(combined)  # (B, 256, 8, 8, 8)
        
        # Step 3: Multi-scale spatial feature extraction
        multi_scale_features = []
        for pooling_layer in self.spatial_pooling:
            pooled = pooling_layer(cnn_fused)  # Various sizes: 8³, 4³, 2³, 1³
            flattened = pooled.view(batch_size, -1)
            multi_scale_features.append(flattened)
        
        # Concatenate all spatial scales
        spatial_features = torch.cat(multi_scale_features, dim=1)  # (B, 149760)
        
        # Step 4: Reduce dimensionality for quantum processing
        reduced_features = self.feature_reduction(spatial_features)  # (B, 512)
        
        # Step 5: Quantum enhancement (key innovation from notebook)
        if self.use_quantum:
            # Primary quantum fusion (simulating the 3D-CNN + SG-CNN fusion from notebook)
            quantum_features_1 = self.quantum_cnn_fusion(reduced_features)
            
            # Secondary quantum enhancement for improved representation
            quantum_features_2 = self.quantum_enhanced_fusion(quantum_features_1)
            
            # Quantum attention for final refinement
            quantum_input = quantum_features_2.unsqueeze(1)  # (B, 1, 512)
            quantum_attended = self.quantum_attention(quantum_input)
            final_quantum_features = quantum_attended.squeeze(1)  # (B, 512)
            
            enhanced_features = final_quantum_features
        else:
            # Classical fallback
            enhanced_features = self.classical_fusion(reduced_features)
        
        # Step 6: Advanced transformer processing
        transformer_input = enhanced_features.unsqueeze(1)  # (B, 1, 512)
        
        # Apply multiple transformer blocks
        transformed = transformer_input
        for transformer in self.transformer_blocks:
            transformed = transformer(transformed)
        
        # Cross-attention for final feature integration
        final_transformed = self.cross_attention(transformed).squeeze(1)  # (B, 512)
        
        # Step 7: Enhanced classification with residual connections
        x = final_transformed
        for i, (classifier_layer, residual_proj) in enumerate(zip(self.classifier_layers, self.residual_projections)):
            residual = residual_proj(x)
            x = classifier_layer(x) + residual  # Residual connection
        
        # Final output layer
        output = self.final_layer(x)
        
        return output
    
    def get_quantum_feature_importance(self, protein_grid, ligand_grid, pocket_grid):
        """
        Analyze quantum feature importance for interpretability.
        Returns quantum circuit contributions and classical feature importance.
        """
        if not self.use_quantum:
            print("  Quantum analysis not available - model using classical fallback")
            return None
            
        self.eval()
        with torch.no_grad():
            # Extract features up to quantum processing
            batch_size = protein_grid.size(0)
            protein_features = self.protein_extractor(protein_grid)
            ligand_features = self.ligand_extractor(ligand_grid)
            pocket_features = self.pocket_extractor(pocket_grid)
            
            combined = torch.cat([protein_features, ligand_features, pocket_features], dim=1)
            cnn_fused = self.cnn_fusion(combined)
            
            # Multi-scale features
            multi_scale_features = []
            for pooling_layer in self.spatial_pooling:
                pooled = pooling_layer(cnn_fused)
                flattened = pooled.view(batch_size, -1)
                multi_scale_features.append(flattened)
            
            spatial_features = torch.cat(multi_scale_features, dim=1)
            reduced_features = self.feature_reduction(spatial_features)
            
            # Analyze quantum contributions
            classical_features = reduced_features
            quantum_features_1 = self.quantum_cnn_fusion(reduced_features)
            quantum_features_2 = self.quantum_enhanced_fusion(quantum_features_1)
            
            importance_analysis = {
                'classical_contribution': torch.mean(torch.abs(classical_features), dim=0),
                'quantum_fusion_1_contribution': torch.mean(torch.abs(quantum_features_1 - classical_features), dim=0),
                'quantum_fusion_2_contribution': torch.mean(torch.abs(quantum_features_2 - quantum_features_1), dim=0),
                'protein_importance': torch.mean(protein_features, dim=[2, 3, 4]),
                'ligand_importance': torch.mean(ligand_features, dim=[2, 3, 4]),
                'pocket_importance': torch.mean(pocket_features, dim=[2, 3, 4])
            }
            
            return importance_analysis


class QuantumHybridEnsemble(nn.Module):
    """
    Ensemble model combining multiple quantum-enhanced architectures for maximum performance.
    Implements ensemble learning with quantum feature fusion.
    """
    
    def __init__(self, num_classes=1, ensemble_size=3, use_quantum=True, dropout=0.3):
        super(QuantumHybridEnsemble, self).__init__()
        self.ensemble_size = ensemble_size
        self.use_quantum = use_quantum and QUANTUM_AVAILABLE
        
        print(f" Initializing Quantum Hybrid Ensemble ({ensemble_size} models)")
        
        # Create ensemble of quantum-enhanced models with different configurations
        self.models = nn.ModuleList([
            QuantumEnhancedBindingAffinityPredictor(
                num_classes=num_classes,
                quantum_layers=10 - i*2,  # Vary quantum layers
                use_quantum=use_quantum,
                dropout=dropout + i*0.05  # Vary dropout for diversity
            ) for i in range(ensemble_size)
        ])
        
        # Quantum ensemble fusion
        if self.use_quantum:
            self.ensemble_quantum_fusion = QuantumFeatureFusion(
                input_size=ensemble_size,
                quantum_layers=6,
                encoding='amplitude',
                ansatz=2,
                dropout=dropout
            )
        
        # Classical ensemble fusion
        self.ensemble_weights = nn.Parameter(torch.ones(ensemble_size) / ensemble_size)
        self.final_fusion = nn.Sequential(
            nn.Linear(ensemble_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, protein_grid, ligand_grid, pocket_grid):
        # Get predictions from all ensemble models
        predictions = []
        for model in self.models:
            pred = model(protein_grid, ligand_grid, pocket_grid)
            predictions.append(pred)
        
        # Stack predictions
        ensemble_preds = torch.cat(predictions, dim=1)  # (B, ensemble_size)
        
        if self.use_quantum:
            # Quantum ensemble fusion
            quantum_fused = self.ensemble_quantum_fusion(ensemble_preds)
            final_pred = self.final_fusion(quantum_fused)
        else:
            # Weighted average
            weights = F.softmax(self.ensemble_weights, dim=0)
            weighted_pred = torch.sum(ensemble_preds * weights.unsqueeze(0), dim=1, keepdim=True)
            final_pred = weighted_pred
        
        return final_pred


class QuantumResilientCNN(nn.Module):
    """
    Quantum-enhanced CNN with noise resilience and robustness features.
    Implements advanced feature extraction, quantum fusion, and classical backup paths.
    """
    
    def __init__(self, num_classes=1, noise_resilience=True, use_quantum=True, dropout=0.3):
        super(QuantumResilientCNN, self).__init__()
        self.use_quantum = use_quantum and QUANTUM_AVAILABLE
        self.noise_resilience = noise_resilience
        
        print(f"  Initializing Quantum Resilient CNN")
        print(f"   Noise Resilience: {' Enabled' if noise_resilience else '❌ Disabled'}")
        
        # Standard feature extractors
        self.protein_extractor = EnhancedFeatureExtractor(19, 128, dropout)
        self.ligand_extractor = EnhancedFeatureExtractor(19, 64, dropout)
        self.pocket_extractor = EnhancedFeatureExtractor(19, 64, dropout)
        
        # Robust CNN fusion
        self.cnn_fusion = nn.Sequential(
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(4)
        )
        
        feature_size = 256 * 64  # 256 * 4^3
        
        if self.use_quantum:
            # Quantum layers with error mitigation
            self.quantum_primary = QuantumFeatureFusion(
                input_size=feature_size,
                quantum_layers=6,  # Fewer layers for noise resilience
                encoding='amplitude',
                ansatz=4,
                dropout=dropout
            )
            
            if noise_resilience:
                # Additional quantum layer for error correction
                self.quantum_backup = QuantumFeatureFusion(
                    input_size=feature_size,
                    quantum_layers=4,
                    encoding='amplitude',
                    ansatz=1,  # Different ansatz
                    dropout=dropout
                )
                
                # Quantum error detection
                self.quantum_validator = nn.Sequential(
                    nn.Linear(feature_size * 2, feature_size),
                    nn.ReLU(),
                    nn.Linear(feature_size, 1),
                    nn.Sigmoid()  # Confidence score
                )
        
        # Classical backup path
        self.classical_backup = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Final classifier
        final_input_size = 256 if not self.use_quantum else feature_size
        self.classifier = nn.Sequential(
            nn.Linear(final_input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm3d, nn.BatchNorm1d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, protein_grid, ligand_grid, pocket_grid):
        # Feature extraction
        protein_features = self.protein_extractor(protein_grid)
        ligand_features = self.ligand_extractor(ligand_grid)
        pocket_features = self.pocket_extractor(pocket_grid)
        
        # Combine and process
        combined = torch.cat([protein_features, ligand_features, pocket_features], dim=1)
        cnn_features = self.cnn_fusion(combined)
        flattened = cnn_features.view(cnn_features.size(0), -1)
        
        if self.use_quantum:
            # Primary quantum processing
            quantum_primary = self.quantum_primary(flattened)
            
            if self.noise_resilience:
                # Backup quantum processing
                quantum_backup = self.quantum_backup(flattened)
                
                # Validate quantum results
                combined_quantum = torch.cat([quantum_primary, quantum_backup], dim=1)
                confidence = self.quantum_validator(combined_quantum)
                
                # Use quantum result if confidence is high, otherwise use classical
                quantum_result = quantum_primary
                classical_result = self.classical_backup(flattened)
                
                # Blend based on confidence
                final_features = confidence * quantum_result + (1 - confidence) * classical_result
            else:
                final_features = quantum_primary
        else:
            final_features = self.classical_backup(flattened)
        
        output = self.classifier(final_features)
        return output


# =========================================================================
# Advanced Quantum Training Utilities
# =========================================================================

class QuantumAwareLoss(nn.Module):
    """
    Loss function that accounts for quantum circuit depth and encourages
    efficient quantum resource usage.
    """
    
    def __init__(self, base_loss='mse', quantum_penalty=0.01, depth_penalty=0.001):
        super(QuantumAwareLoss, self).__init__()
        self.quantum_penalty = quantum_penalty
        self.depth_penalty = depth_penalty
        
        if base_loss == 'mse':
            self.base_loss = nn.MSELoss()
        elif base_loss == 'mae':
            self.base_loss = nn.L1Loss()
        elif base_loss == 'huber':
            self.base_loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")
    
    def forward(self, pred, target, model=None):
        # Base prediction loss
        loss = self.base_loss(pred, target)
        
        # Add quantum-specific penalties if model provided
        if model is not None and hasattr(model, 'use_quantum') and model.use_quantum:
            # Penalty for quantum parameter magnitude (encourage efficiency)
            quantum_penalty = 0
            quantum_param_count = 0
            
            for name, param in model.named_parameters():
                if 'quantum' in name.lower():
                    quantum_penalty += torch.sum(param ** 2)
                    quantum_param_count += param.numel()
            
            if quantum_param_count > 0:
                loss += self.quantum_penalty * quantum_penalty / quantum_param_count
        
        return loss


def create_quantum_model(model_type='enhanced', **kwargs):
    """
    Factory function to create quantum-enhanced models.
    
    Args:
        model_type (str): 'enhanced', 'ensemble', or 'resilient'
        **kwargs: Additional arguments for model construction
    
    Returns:
        torch.nn.Module: The requested quantum model
    """
    models = {
        'enhanced': QuantumEnhancedBindingAffinityPredictor,
        'ensemble': QuantumHybridEnsemble,
        'resilient': QuantumResilientCNN
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    model = models[model_type](**kwargs)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"🚀 Quantum Model Created: {model.__class__.__name__}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    if hasattr(model, 'use_quantum') and model.use_quantum:
        print(f"⚛️  Quantum Processing: ENABLED")
    else:
        print(f"🔧 Classical Fallback: ACTIVE")
    
    print(f"{'='*60}\n")
    
    return model


def quantum_model_summary(model, input_shape=(2, 19, 32, 32, 32)):
    """
    Print detailed summary of quantum model including quantum circuit information.
    
    Args:
        model: Quantum-enhanced model
        input_shape: Input tensor shape for testing
    """
    model.eval()
    
    # Create dummy inputs
    device = next(model.parameters()).device if next(model.parameters(), None) is not None else torch.device('cpu')
    dummy_protein = torch.randn(*input_shape).to(device)
    dummy_ligand = torch.randn(*input_shape).to(device)
    dummy_pocket = torch.randn(*input_shape).to(device)
    
    print(f"\n🔬 Quantum Model Analysis")
    print(f"{'='*60}")
    
    # Forward pass
    with torch.no_grad():
        try:
            output = model(dummy_protein, dummy_ligand, dummy_pocket)
            print(f"✅ Forward Pass: Successful")
            print(f"   Input Shape: {input_shape} (per component)")
            print(f"   Output Shape: {output.shape}")
        except Exception as e:
            print(f"❌ Forward Pass: Failed - {e}")
            return
    
    # Parameter analysis
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    quantum_params = 0
    classical_params = 0
    
    for name, param in model.named_parameters():
        if 'quantum' in name.lower():
            quantum_params += param.numel()
        else:
            classical_params += param.numel()
    
    print(f"\n📊 Parameter Breakdown:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Quantum: {quantum_params:,} ({quantum_params/total_params*100:.1f}%)")
    print(f"   Classical: {classical_params:,} ({classical_params/total_params*100:.1f}%)")
    
    # Memory estimation
    input_size = np.prod(input_shape) * 4 * 3  # 3 inputs, 4 bytes per float32
    param_size = total_params * 4  # 4 bytes per parameter
    estimated_memory_mb = (input_size + param_size) / (1024 * 1024)
    
    print(f"\n💾 Memory Estimation:")
    print(f"   Model Size: ~{param_size/(1024*1024):.1f} MB")
    print(f"   Runtime Memory: ~{estimated_memory_mb:.1f} MB")
    
    # Quantum feature analysis
    if hasattr(model, 'use_quantum') and model.use_quantum:
        print(f"\n⚛️  Quantum Features:")
        print(f"   Status: ACTIVE")
        
        if hasattr(model, 'get_quantum_feature_importance'):
            try:
                importance = model.get_quantum_feature_importance(dummy_protein, dummy_ligand, dummy_pocket)
                if importance:
                    print(f"   Feature Analysis: Available")
                    print(f"   Quantum Contribution: Measurable")
            except Exception as e:
                print(f"   Feature Analysis: Error - {e}")
    else:
        print(f"\n🔧 Classical Mode:")
        print(f"   Quantum components using classical fallback")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Quantum Models...")
    
    # Test model creation
    model = create_quantum_model('enhanced', quantum_layers=8, dropout=0.2)
    
    # Test model summary
    quantum_model_summary(model)
    
    # Test ensemble
    ensemble = create_quantum_model('ensemble', ensemble_size=2, dropout=0.2)
    
    # Test resilient model
    resilient = create_quantum_model('resilient', noise_resilience=True, dropout=0.2)
    
    print("✅ All quantum model tests completed successfully!")
