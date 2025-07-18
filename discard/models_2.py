

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Import existing components from models.py
from models import (
    MultiHeadSelfAttention, ChannelAttention3D, SpatialAttention3D,
    DilatedConvBlock3D, TORCH_GEOMETRIC_AVAILABLE
)

# Import quantum components
try:
    from layers import QuantumFCLayer
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    print("Warning: Quantum layers not available. Quantum features will be disabled.")

if TORCH_GEOMETRIC_AVAILABLE:
    from torch_geometric.nn import global_add_pool, GCNConv, GATConv, TransformerConv
    from torch_geometric.data import Data, Batch


# =========================================================================
# Quantum Components for Enhanced Feature Processing
# =========================================================================

class QuantumFeatureFusion(nn.Module):
    """
    Quantum fusion layer for enhanced feature representation learning.
    Uses quantum circuits to process and fuse multi-modal features.
    """
    
    def __init__(self, input_size, quantum_layers=6, encoding='amplitude', 
                 ansatz=4, dropout=0.1):
        super(QuantumFeatureFusion, self).__init__()
        self.input_size = input_size
        self.quantum_available = QUANTUM_AVAILABLE
        
        if self.quantum_available:
            # Quantum observables for multi-qubit measurements
            n_qubits = int(np.ceil(np.log2(input_size))) if encoding == 'amplitude' else input_size
            self.observables = [f'{"Z" * i}{"I" * (n_qubits - i)}' for i in range(1, min(n_qubits + 1, 5))]
            
            # Create quantum layer
            self.quantum_layer = QuantumFCLayer(
                input_size=input_size,
                n_layers=quantum_layers,
                encoding=encoding,
                ansatz=ansatz,
                observables=self.observables,
                backend='default.qubit'
            ).create_layer(type_layer='torch')
            
            # Classical post-processing
            self.quantum_output_size = len(self.observables)
            self.post_quantum = nn.Sequential(
                nn.Linear(self.quantum_output_size, input_size // 2),
                nn.BatchNorm1d(input_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_size // 2, input_size)
            )
        else:
            # Fallback: enhanced classical fusion
            self.classical_fusion = nn.Sequential(
                nn.Linear(input_size, input_size * 2),
                nn.BatchNorm1d(input_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_size * 2, input_size),
                nn.BatchNorm1d(input_size),
                nn.Tanh()  # Simulate quantum-like non-linearity
            )
    
    def forward(self, x):
        if self.quantum_available:
            # Normalize input for quantum processing
            x_normalized = F.normalize(x, p=2, dim=1)
            
            # Quantum processing
            quantum_features = self.quantum_layer(x_normalized)
            
            # Post-quantum classical processing
            enhanced_features = self.post_quantum(quantum_features)
            
            # Residual connection
            return x + enhanced_features
        else:
            # Classical fallback
            return self.classical_fusion(x)


class QuantumAttentionFusion(nn.Module):
    """
    Quantum-enhanced attention mechanism for feature fusion.
    Combines quantum feature processing with attention weights.
    """
    
    def __init__(self, feature_dim, num_heads=4, quantum_layers=4, dropout=0.1):
        super(QuantumAttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Classical attention components
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        # Quantum feature enhancement for attention weights
        self.quantum_fusion = QuantumFeatureFusion(
            input_size=feature_dim,
            quantum_layers=quantum_layers,
            encoding='amplitude',
            ansatz=4,
            dropout=dropout
        )
        
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Enhance features with quantum processing
        x_enhanced = []
        for i in range(seq_len):
            enhanced_feat = self.quantum_fusion(x[:, i, :])
            x_enhanced.append(enhanced_feat.unsqueeze(1))
        x_enhanced = torch.cat(x_enhanced, dim=1)
        
        # Compute attention with enhanced features
        Q = self.query(x_enhanced)
        K = self.key(x_enhanced)
        V = self.value(x_enhanced)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)
        
        # Output projection and residual connection
        output = self.output_projection(attended)
        return self.layer_norm(x + output)


# =========================================================================
# Advanced Transformer Components for Better Performance
# =========================================================================

class PositionalEncoding3D(nn.Module):
    """3D Positional encoding for spatial grid data"""
    
    def __init__(self, d_model, max_len=64):
        super(PositionalEncoding3D, self).__init__()
        self.d_model = d_model
        
        # Create 3D positional encodings
        pe = torch.zeros(max_len, max_len, max_len, d_model)
        
        # Position indices
        position_d = torch.arange(0, max_len).unsqueeze(1).unsqueeze(2).float()
        position_h = torch.arange(0, max_len).unsqueeze(0).unsqueeze(2).float()
        position_w = torch.arange(0, max_len).unsqueeze(0).unsqueeze(1).float()
        
        # Frequency terms
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Apply sinusoidal encoding to each dimension
        # Distribute encoding evenly across all channels
        channels_per_dim = d_model // 6 if d_model >= 6 else d_model // 3
        
        if d_model >= 6:
            # Use 6-way encoding for better coverage
            for i in range(channels_per_dim):
                if 6*i < d_model:
                    pe[:, :, :, 6*i] = torch.sin(position_d.squeeze() * div_term[i] if i < len(div_term) else 0)
                if 6*i+1 < d_model:
                    pe[:, :, :, 6*i+1] = torch.cos(position_d.squeeze() * div_term[i] if i < len(div_term) else 0)
                if 6*i+2 < d_model:
                    pe[:, :, :, 6*i+2] = torch.sin(position_h.squeeze() * div_term[i] if i < len(div_term) else 0)
                if 6*i+3 < d_model:
                    pe[:, :, :, 6*i+3] = torch.cos(position_h.squeeze() * div_term[i] if i < len(div_term) else 0)
                if 6*i+4 < d_model:
                    pe[:, :, :, 6*i+4] = torch.sin(position_w.squeeze() * div_term[i] if i < len(div_term) else 0)
                if 6*i+5 < d_model:
                    pe[:, :, :, 6*i+5] = torch.cos(position_w.squeeze() * div_term[i] if i < len(div_term) else 0)
        else:
            # Fallback for small d_model
            for i in range(channels_per_dim):
                if 3*i < d_model:
                    pe[:, :, :, 3*i] = torch.sin(position_d.squeeze() * div_term[i] if i < len(div_term) else 0)
                if 3*i+1 < d_model:
                    pe[:, :, :, 3*i+1] = torch.sin(position_h.squeeze() * div_term[i] if i < len(div_term) else 0)
                if 3*i+2 < d_model:
                    pe[:, :, :, 3*i+2] = torch.sin(position_w.squeeze() * div_term[i] if i < len(div_term) else 0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (B, C, D, H, W)
        batch_size, channels, d, h, w = x.size()
        
        # Get positional encoding for current spatial dimensions
        pe = self.pe[:d, :h, :w, :channels].permute(3, 0, 1, 2)  # (C, D, H, W)
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)   # (B, C, D, H, W)
        
        return x + pe


class TransformerBlock3D(nn.Module):
    """3D Transformer block for spatial feature processing"""
    
    def __init__(self, d_model, num_heads=8, dropout=0.1, activation='relu'):
        super(TransformerBlock3D, self).__init__()
        
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        
        # Self-attention with residual connection
        attn_output = self.self_attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class EnhancedFeatureExtractor(nn.Module):
    """Enhanced feature extraction with multiple scales and attention"""
    
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(EnhancedFeatureExtractor, self).__init__()
        
        # Multi-scale convolutions
        self.conv1x1 = nn.Conv3d(in_channels, out_channels//4, 1)
        self.conv3x3 = nn.Conv3d(in_channels, out_channels//4, 3, padding=1)
        self.conv5x5 = nn.Conv3d(in_channels, out_channels//4, 5, padding=2)
        
        # Dilated convolution for larger receptive field
        self.conv_dilated = nn.Conv3d(in_channels, out_channels//4, 3, padding=2, dilation=2)
        
        # Batch normalization and activation
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout)
        
        # Channel attention
        self.channel_attention = ChannelAttention3D(out_channels)
        
    def forward(self, x):
        # Multi-scale feature extraction
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3(x)
        feat3 = self.conv5x5(x)
        feat4 = self.conv_dilated(x)
        
        # Concatenate multi-scale features
        combined = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        
        # Apply normalization and activation
        combined = self.bn(combined)
        combined = self.activation(combined)
        combined = self.dropout(combined)
        
        # Apply channel attention
        combined = self.channel_attention(combined)
        
        return combined


class AdvancedTransformerCNN(nn.Module):
    """
    Advanced CNN-Transformer hybrid model for improved binding affinity prediction.
    Combines multi-scale feature extraction, 3D transformers, and advanced attention.
    """
    
    def __init__(self, num_classes=1, dropout=0.3):
        super(AdvancedTransformerCNN, self).__init__()
        
        # Enhanced feature extractors for each component
        self.protein_extractor = EnhancedFeatureExtractor(19, 128, dropout)
        self.ligand_extractor = EnhancedFeatureExtractor(19, 64, dropout)
        self.pocket_extractor = EnhancedFeatureExtractor(19, 64, dropout)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding3D(256)
        
        # Fusion layer
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)
        )
        
        # Adaptive pooling to reduce spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool3d((8, 8, 8))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock3D(256, num_heads=8, dropout=dropout)
            for _ in range(4)
        ])
        
        # Final processing
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Enhanced classifier with skip connections
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm3d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, protein_grid, ligand_grid, pocket_grid):
        # Extract features from each component
        protein_features = self.protein_extractor(protein_grid)  # (B, 128, D, H, W)
        ligand_features = self.ligand_extractor(ligand_grid)     # (B, 64, D, H, W)
        pocket_features = self.pocket_extractor(pocket_grid)     # (B, 64, D, H, W)
        
        # Combine features
        combined = torch.cat([protein_features, ligand_features, pocket_features], dim=1)  # (B, 256, D, H, W)
        
        # Add positional encoding
        combined = self.pos_encoding(combined)
        
        # Fusion
        fused = self.fusion_conv(combined)  # (B, 256, D, H, W)
        
        # Reduce spatial dimensions
        pooled = self.adaptive_pool(fused)  # (B, 256, 8, 8, 8)
        
        # Reshape for transformer: (B, 512, 256)
        batch_size, channels = pooled.size(0), pooled.size(1)
        transformer_input = pooled.view(batch_size, channels, -1).transpose(1, 2)  # (B, 512, 256)
        
        # Apply transformer layers
        x = transformer_input
        for transformer in self.transformer_layers:
            x = transformer(x)
        
        # Global pooling and classification
        x = x.transpose(1, 2)  # (B, 256, 512)
        x = self.global_pool(x).squeeze(-1)  # (B, 256)
        
        output = self.classifier(x)
        return output


if TORCH_GEOMETRIC_AVAILABLE:
    class HybridCNNGNNTransformer(nn.Module):
        """
        Hybrid model combining CNN, GNN, Transformer, and Quantum architectures
        for enhanced binding affinity prediction accuracy with quantum feature fusion.
        """
        
        def __init__(self, num_classes=1, use_gnn=True, use_quantum=True, dropout=0.3):
            super(HybridCNNGNNTransformer, self).__init__()
            self.use_gnn = use_gnn
            self.use_quantum = use_quantum and QUANTUM_AVAILABLE
            
            # CNN backbone for grid processing
            self.protein_extractor = EnhancedFeatureExtractor(19, 128, dropout)
            self.ligand_extractor = EnhancedFeatureExtractor(19, 64, dropout)
            self.pocket_extractor = EnhancedFeatureExtractor(19, 64, dropout)
            
            # CNN feature processing
            self.cnn_fusion = nn.Sequential(
                nn.Conv3d(256, 256, 3, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(4)
            )
            
            # GNN components
            if use_gnn:
                self.gnn_layers = nn.ModuleList([
                    GATConv(57, 128, heads=4, dropout=dropout),  # 19*3 input features
                    GATConv(128*4, 128, heads=4, dropout=dropout),
                    GATConv(128*4, 64, heads=1, dropout=dropout)
                ])
            
            # Always use CNN-only fusion layer for simplicity and compatibility
            # Even when GNN is available, we'll handle fusion dynamically in forward pass
            self.cnn_feature_size = 256 * 64  # 256 channels * 4^3 spatial
            
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.cnn_feature_size, 512),  # CNN features only
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # Separate GNN fusion layer for when GNN features are available
            if use_gnn:
                self.gnn_fusion_layer = nn.Sequential(
                    nn.Linear(self.cnn_feature_size + 64, 512),  # CNN + GNN features
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            
            # Quantum fusion layers for enhanced representation learning
            if self.use_quantum:
                self.quantum_cnn_fusion = QuantumFeatureFusion(
                    input_size=512, 
                    quantum_layers=8, 
                    encoding='amplitude',
                    ansatz=4, 
                    dropout=dropout
                )
                
                self.quantum_attention_fusion = QuantumAttentionFusion(
                    feature_dim=512,
                    num_heads=8,
                    quantum_layers=6,
                    dropout=dropout
                )
            
            # Final transformer for feature integration
            self.final_transformer = TransformerBlock3D(512, num_heads=8, dropout=dropout)
            
            # Enhanced final classifier
            self.final_classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout/2),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout/2),
                
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout/4),
                
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )
            
            self.apply(self._init_weights)
        
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        def forward(self, protein_grid, ligand_grid, pocket_grid, graph_data=None):
            # CNN feature extraction
            protein_features = self.protein_extractor(protein_grid)
            ligand_features = self.ligand_extractor(ligand_grid)
            pocket_features = self.pocket_extractor(pocket_grid)
            
            # Combine and process CNN features
            combined = torch.cat([protein_features, ligand_features, pocket_features], dim=1)
            cnn_features = self.cnn_fusion(combined)  # (B, 256, 4, 4, 4)
            cnn_output = cnn_features.view(cnn_features.size(0), -1)  # (B, 256*64)
            
            # GNN processing if available and graph_data is provided
            if self.use_gnn and graph_data is not None:
                x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
                
                # Apply GNN layers
                for i, gnn_layer in enumerate(self.gnn_layers):
                    x = gnn_layer(x, edge_index)
                    x = F.relu(x)
                    if i < len(self.gnn_layers) - 1:
                        x = F.dropout(x, training=self.training)
                
                # Global pooling for graph-level representation
                gnn_output = global_add_pool(x, batch)  # (B, 64)
                
                # Fuse CNN and GNN features using dedicated fusion layer
                combined_features = torch.cat([cnn_output, gnn_output], dim=1)
                fused_features = self.gnn_fusion_layer(combined_features)
            else:
                # Use only CNN features
                fused_features = self.fusion_layer(cnn_output)
            
            # Quantum enhancement if available
            if self.use_quantum:
                # Apply quantum feature fusion for enhanced representation
                quantum_enhanced_features = self.quantum_cnn_fusion(fused_features)
                
                # Reshape for quantum attention processing: (B, 1, 512)
                quantum_input = quantum_enhanced_features.unsqueeze(1)
                quantum_attended_features = self.quantum_attention_fusion(quantum_input)
                final_quantum_features = quantum_attended_features.squeeze(1)  # (B, 512)
                
                # Use quantum-enhanced features for transformer input
                transformer_input = final_quantum_features.unsqueeze(1)
            else:
                # Use regular features for transformer input
                transformer_input = fused_features.unsqueeze(1)
            
            # Final transformer processing
            final_features = self.final_transformer(transformer_input).squeeze(1)  # (B, 512)
            
            # Final prediction
            output = self.final_classifier(final_features)
            return output

else:
    class HybridCNNGNNTransformer(nn.Module):
        """
        Fallback implementation of HybridCNNGNNTransformer without torch_geometric.
        Uses CNN-based alternatives to GNN components for structural understanding,
        enhanced with quantum feature fusion capabilities.
        """
        
        def __init__(self, num_classes=1, use_gnn=False, use_quantum=True, dropout=0.3, 
                     grid_size=64, hidden_dim=256):
            super(HybridCNNGNNTransformer, self).__init__()
            self.use_gnn = False  # Force disable GNN when torch_geometric not available
            self.use_quantum = use_quantum and QUANTUM_AVAILABLE
            self.grid_size = grid_size
            self.hidden_dim = hidden_dim
            
            # Enhanced CNN backbone for grid processing
            self.protein_extractor = EnhancedFeatureExtractor(19, 128, dropout)
            self.ligand_extractor = EnhancedFeatureExtractor(19, 64, dropout)
            self.pocket_extractor = EnhancedFeatureExtractor(19, 64, dropout)
            
            # CNN feature processing with multi-scale approach
            self.cnn_fusion = nn.Sequential(
                nn.Conv3d(256, 256, 3, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(),
                nn.Conv3d(256, 256, 3, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(4)
            )
            
            # Graph-inspired CNN layers (simulating GNN behavior)
            # These layers simulate graph convolutions using 3D convolutions
            self.graph_inspired_layers = nn.ModuleList([
                self._make_graph_inspired_block(256, 128, kernel_size=3),
                self._make_graph_inspired_block(128, 128, kernel_size=5),
                self._make_graph_inspired_block(128, 64, kernel_size=7)
            ])
            
            # Structural attention module (replaces graph attention)
            self.structural_attention = nn.ModuleList([
                ChannelAttention3D(256),
                SpatialAttention3D(256)  # Added required in_channels parameter
            ])
            
            # Multi-scale feature pyramid for structural understanding (simplified)
            self.feature_pyramid = nn.ModuleList([
                nn.Sequential(
                    nn.Conv3d(256, 32, 1),  # Reduce channels first
                    nn.AdaptiveAvgPool3d(scale)
                ) for scale in [4, 2, 1]  # Reduced scales
            ])
            
            # Enhanced positional encoding for spatial relationships
            self.pos_encoding_3d = PositionalEncoding3D(256, max_len=grid_size)
            
            # Spatial transformer networks for geometric understanding
            self.spatial_transformers = nn.ModuleList([
                self._make_spatial_transformer(256, grid_size // (2**i))
                for i in range(3)
            ])
            
            # Distance-aware convolutions (simulating graph edges)
            self.distance_conv = nn.Sequential(
                nn.Conv3d(256, 128, 3, padding=1, groups=8),  # Grouped conv for efficiency
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.Conv3d(128, 64, 3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU()
            )
            
            # Calculate fusion layer input size
            # CNN features: 256 * 64 (256 channels * 4^3 spatial)
            # Multi-scale features: 32 * (4^3 + 2^3 + 1^3) = 32 * (64 + 8 + 1) = 32 * 73 = 2,336
            # Distance features: 64 * 64 (64 channels * 4^3 spatial)
            multi_scale_total = 32 * (4**3 + 2**3 + 1**3)  # 32 * 73 = 2,336
            fusion_input_size = 256 * 64 + multi_scale_total + 64 * 64
            
            # Fusion layer for all CNN-based features
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_size, 512),  # Reduced from 1024
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.Linear(512, 256),  # Reduced from 512
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout/2)
            )
            
            # Quantum fusion layers for enhanced representation learning
            if self.use_quantum:
                self.quantum_cnn_fusion = QuantumFeatureFusion(
                    input_size=256, 
                    quantum_layers=10, 
                    encoding='amplitude',
                    ansatz=4, 
                    dropout=dropout
                )
                
                self.quantum_structural_fusion = QuantumFeatureFusion(
                    input_size=256,
                    quantum_layers=8,
                    encoding='amplitude',
                    ansatz=1,  # Use different ansatz for diversity
                    dropout=dropout
                )
                
                self.quantum_attention_fusion = QuantumAttentionFusion(
                    feature_dim=256,
                    num_heads=8,
                    quantum_layers=6,
                    dropout=dropout
                )
            
            # Multi-head transformer for feature integration
            self.feature_transformers = nn.ModuleList([
                TransformerBlock3D(256, num_heads=8, dropout=dropout)  # Updated to match fusion output
                for _ in range(2)  # Reduced number of layers
            ])
            
            # Cross-attention between different feature types
            self.cross_attention = MultiHeadSelfAttention(256, num_heads=8, dropout=dropout)  # Updated size
            
            # Enhanced final classifier with skip connections
            self.final_classifier = nn.Sequential(
                nn.Linear(256, 128),  # Updated input size
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout/2),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout/2),
                
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout/4),
                
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(dropout/4),
                
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, num_classes)
            )
            
            # Residual connection for final features
            self.residual_projection = nn.Linear(fusion_input_size, 256)  # Updated target size
            
            self.apply(self._init_weights)
        
        def _make_graph_inspired_block(self, in_channels, out_channels, kernel_size=3):
            """Create CNN block that mimics graph convolution behavior"""
            padding = kernel_size // 2
            return nn.Sequential(
                # Message passing simulation
                nn.Conv3d(in_channels, out_channels * 2, kernel_size, padding=padding, groups=min(8, in_channels)),
                nn.BatchNorm3d(out_channels * 2),
                nn.ReLU(),
                
                # Aggregation simulation
                nn.Conv3d(out_channels * 2, out_channels, 1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
                
                # Update simulation
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.AdaptiveAvgPool3d(4)
            )
        
        def _make_spatial_transformer(self, channels, spatial_size):
            """Create spatial transformer for geometric understanding"""
            return nn.Sequential(
                nn.AdaptiveAvgPool3d(spatial_size),
                nn.Conv3d(channels, channels // 4, 1),
                nn.ReLU(),
                nn.Conv3d(channels // 4, channels, 1),
                nn.Sigmoid()
            )
        
        def _init_weights(self, module):
            """Initialize model weights"""
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm3d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        def forward(self, protein_grid, ligand_grid, pocket_grid, graph_data=None):
            """
            Forward pass through the hybrid model.
            
            Args:
                protein_grid: Protein voxel grid (B, 19, D, H, W)
                ligand_grid: Ligand voxel grid (B, 19, D, H, W)
                pocket_grid: Pocket voxel grid (B, 19, D, H, W)
                graph_data: Ignored in this implementation (no torch_geometric)
            
            Returns:
                Binding affinity prediction (B, 1)
            """
            batch_size = protein_grid.size(0)
            
            # Step 1: Enhanced CNN feature extraction
            protein_features = self.protein_extractor(protein_grid)  # (B, 128, D, H, W)
            ligand_features = self.ligand_extractor(ligand_grid)     # (B, 64, D, H, W)
            pocket_features = self.pocket_extractor(pocket_grid)     # (B, 64, D, H, W)
            
            # Step 2: Combine and enhance features
            combined = torch.cat([protein_features, ligand_features, pocket_features], dim=1)  # (B, 256, D, H, W)
            
            # Add positional encoding for spatial awareness
            combined = self.pos_encoding_3d(combined)
            
            # Apply structural attention
            for attention_layer in self.structural_attention:
                combined = attention_layer(combined)
            
            # Step 3: CNN fusion with enhanced processing
            cnn_features = self.cnn_fusion(combined)  # (B, 256, 4, 4, 4)
            cnn_output = cnn_features.view(batch_size, -1)  # (B, 256*64)
            
            # Step 4: Graph-inspired processing (replacing GNN)
            graph_inspired_features = combined
            for graph_layer in self.graph_inspired_layers:
                graph_inspired_features = graph_layer(graph_inspired_features)
            
            # Step 5: Multi-scale feature extraction
            multi_scale_features = []
            for pyramid_layer in self.feature_pyramid:
                scale_feat = pyramid_layer(combined)  # (B, 32, scale, scale, scale)
                scale_feat = scale_feat.view(batch_size, -1)  # Flatten
                multi_scale_features.append(scale_feat)
            
            multi_scale_output = torch.cat(multi_scale_features, dim=1)  # (B, 32*73)
            
            # Step 6: Spatial transformation for geometric understanding
            spatial_features = combined
            for i, spatial_transformer in enumerate(self.spatial_transformers):
                # Apply spatial transformation
                spatial_weight = spatial_transformer(spatial_features)
                spatial_features = spatial_features * spatial_weight
                
                # Downsample for next level
                if i < len(self.spatial_transformers) - 1:
                    spatial_features = F.avg_pool3d(spatial_features, 2)
            
            # Step 7: Distance-aware convolutions (simulating graph edges)
            distance_features = self.distance_conv(combined)  # (B, 64, D, H, W)
            distance_features = F.adaptive_avg_pool3d(distance_features, 4)  # (B, 64, 4, 4, 4)
            distance_output = distance_features.view(batch_size, -1)  # (B, 64*64)
            
            # Step 8: Fuse all features
            all_features = torch.cat([cnn_output, multi_scale_output, distance_output], dim=1)
            
            # Process through fusion layer
            fused_features = self.fusion_layer(all_features)  # (B, 256)
            
            # Create residual connection
            residual = self.residual_projection(all_features)  # (B, 256)
            fused_features = fused_features + residual
            
            # Quantum enhancement if available
            if self.use_quantum:
                # Apply multiple quantum processing stages for enhanced representation
                quantum_cnn_features = self.quantum_cnn_fusion(fused_features)
                quantum_structural_features = self.quantum_structural_fusion(quantum_cnn_features)
                
                # Reshape for quantum attention processing: (B, 1, 256)
                quantum_input = quantum_structural_features.unsqueeze(1)
                quantum_attended_features = self.quantum_attention_fusion(quantum_input)
                final_quantum_features = quantum_attended_features.squeeze(1)  # (B, 256)
                
                # Use quantum-enhanced features
                transformer_input = final_quantum_features.unsqueeze(1)
            else:
                # Use regular features
                transformer_input = fused_features.unsqueeze(1)
            
            # Step 9: Transformer processing for feature integration
            # Apply multiple transformer layers
            transformed_features = transformer_input
            for transformer in self.feature_transformers:
                transformed_features = transformer(transformed_features)
            
            # Cross-attention for final feature refinement
            final_features = self.cross_attention(transformed_features).squeeze(1)  # (B, 256)
            
            # Step 10: Final prediction
            output = self.final_classifier(final_features)
            
            return output
        
        def get_feature_importance(self, protein_grid, ligand_grid, pocket_grid):
            """
            Analyze feature importance for interpretability.
            Returns attention weights and feature contributions.
            """
            self.eval()
            with torch.no_grad():
                # Extract intermediate features
                protein_features = self.protein_extractor(protein_grid)
                ligand_features = self.ligand_extractor(ligand_grid)
                pocket_features = self.pocket_extractor(pocket_grid)
                
                # Compute attention weights
                combined = torch.cat([protein_features, ligand_features, pocket_features], dim=1)
                
                importance_scores = {
                    'protein_contribution': torch.mean(protein_features, dim=[2, 3, 4]),
                    'ligand_contribution': torch.mean(ligand_features, dim=[2, 3, 4]),
                    'pocket_contribution': torch.mean(pocket_features, dim=[2, 3, 4]),
                    'spatial_attention': torch.mean(combined, dim=1)
                }
                
                return importance_scores


class ResidualTransformerCNN(nn.Module):
    """
    CNN with residual connections and transformer blocks for improved performance.
    """
    
    def __init__(self, num_classes=1, dropout=0.2):
        super(ResidualTransformerCNN, self).__init__()
        
        # Residual CNN blocks for each component
        self.protein_blocks = nn.ModuleList([
            self._make_residual_block(19, 32),
            self._make_residual_block(32, 64),
            self._make_residual_block(64, 128)
        ])
        
        self.ligand_blocks = nn.ModuleList([
            self._make_residual_block(19, 32),
            self._make_residual_block(32, 64),
            self._make_residual_block(64, 128)
        ])
        
        self.pocket_blocks = nn.ModuleList([
            self._make_residual_block(19, 32),
            self._make_residual_block(32, 64),
            self._make_residual_block(64, 128)
        ])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Transformer for cross-component interaction
        self.cross_transformer = TransformerBlock3D(128, num_heads=8, dropout=dropout)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.MaxPool3d(2)
        )
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm3d, nn.BatchNorm1d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, protein_grid, ligand_grid, pocket_grid):
        # Process each component through residual blocks
        protein_feat = protein_grid
        for block in self.protein_blocks:
            protein_feat = block(protein_feat)
        
        ligand_feat = ligand_grid
        for block in self.ligand_blocks:
            ligand_feat = block(ligand_feat)
        
        pocket_feat = pocket_grid
        for block in self.pocket_blocks:
            pocket_feat = block(pocket_feat)
        
        # Global pooling
        protein_pooled = self.global_pool(protein_feat).flatten(1)  # (B, 128)
        ligand_pooled = self.global_pool(ligand_feat).flatten(1)    # (B, 128)
        pocket_pooled = self.global_pool(pocket_feat).flatten(1)    # (B, 128)
        
        # Stack for transformer
        stacked = torch.stack([protein_pooled, ligand_pooled, pocket_pooled], dim=1)  # (B, 3, 128)
        
        # Apply cross-component transformer
        transformed = self.cross_transformer(stacked)  # (B, 3, 128)
        
        # Flatten and classify
        final_features = transformed.view(transformed.size(0), -1)  # (B, 384)
        output = self.classifier(final_features)
        
        return output


# =========================================================================
# Improved Training Components
# =========================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling imbalanced regression"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        pt = torch.exp(-mse_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * mse_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss function for better training"""
    
    def __init__(self, mse_weight=1.0, mae_weight=0.5, huber_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.huber_weight = huber_weight
        
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        huber_loss = self.huber(pred, target)
        
        combined = (self.mse_weight * mse_loss + 
                   self.mae_weight * mae_loss + 
                   self.huber_weight * huber_loss)
        
        return combined


class WarmupLRScheduler:
    """Learning rate scheduler with warmup"""
    
    def __init__(self, optimizer, warmup_epochs=5, base_lr=1e-4, max_lr=1e-3):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr + (self.max_lr - self.base_lr) * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            lr = self.base_lr + (self.max_lr - self.base_lr) * 0.5 * (
                1 + math.cos(math.pi * (self.current_epoch - self.warmup_epochs) / 50)
            )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


# =========================================================================
# Factory Functions
# =========================================================================

def get_advanced_model(model_name, **kwargs):
    """
    Factory function to create advanced models by name.
    
    Args:
        model_name (str): One of 'advanced_transformer', 'hybrid_cnn_gnn_transformer', 
                         'residual_transformer'
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        torch.nn.Module: The requested model
    """
    models = {
        'advanced_transformer': AdvancedTransformerCNN,
        'hybrid_cnn_gnn_transformer': HybridCNNGNNTransformer,
        'residual_transformer': ResidualTransformerCNN
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)


def advanced_model_summary(model, input_shape=(1, 19, 64, 64, 64), graph_data=None):
    """
    Print a summary of the advanced model including parameter count and memory usage.
    
    Args:
        model (torch.nn.Module): The model to summarize
        input_shape (tuple): Shape of input tensors (batch_size, channels, D, H, W)
        graph_data: Optional graph data for GNN models
    """
    model.eval()
    
    # Create dummy inputs
    device = next(model.parameters()).device
    dummy_protein = torch.randn(*input_shape).to(device)
    dummy_ligand = torch.randn(*input_shape).to(device) 
    dummy_pocket = torch.randn(*input_shape).to(device)
    
    # Forward pass to get output shape
    with torch.no_grad():
        try:
            if isinstance(model, HybridCNNGNNTransformer) and TORCH_GEOMETRIC_AVAILABLE:
                output = model(dummy_protein, dummy_ligand, dummy_pocket, graph_data)
            else:
                output = model(dummy_protein, dummy_ligand, dummy_pocket)
        except Exception as e:
            print(f"Error during forward pass: {e}")
            output = torch.tensor([0.0])  # Fallback
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (rough approximation)
    input_size = np.prod(input_shape) * 4 * 3  # 3 inputs, 4 bytes per float32
    param_size = total_params * 4  # 4 bytes per parameter
    estimated_memory_mb = (input_size + param_size) / (1024 * 1024)
    
    print(f"\n🚀 Advanced Model Summary:")
    print(f"{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {input_shape} (per component)")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Estimated memory usage: {estimated_memory_mb:.1f} MB")
    
    # Model-specific features
    if hasattr(model, 'use_gnn') and model.use_gnn:
        print(f"🧬 Graph Neural Network: Enabled")
    if 'Transformer' in model.__class__.__name__:
        print(f"🤖 Transformer Blocks: Enabled")
    if 'Residual' in model.__class__.__name__:
        print(f"🔗 Residual Connections: Enabled")
    
    print(f"{'='*60}")
