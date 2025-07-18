"""
CNN and Graph Neural Network Model Architectures for Protein-Ligand Binding Affinity Prediction

This module contains CNN architectures and integrated GNN models extracted from step5_basicML.ipynb
and FAST framework, optimized for use with the protein_data_reader.py data loading pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Graph neural network imports
try:
    from torch_geometric.nn import MessagePassing, NNConv, global_add_pool, GCNConv
    from torch_geometric.utils import softmax, add_self_loops, remove_self_loops
    from torch_geometric.utils import to_undirected, is_undirected, contains_self_loops
    from torch_geometric.utils import scatter
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("Warning: torch_geometric not available. Graph neural network models will not work.")
    TORCH_GEOMETRIC_AVAILABLE = False


# =========================================================================
# Self-Attention Components (from Research Paper)
# =========================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism as described in the research paper.
    
    This implements the self-attention architecture shown in Figure 2 of the paper,
    with Query, Key, Value matrices and multi-head attention.
    """
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear transformations for Q, K, V as shown in the paper
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Store residual connection
        residual = x
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention as per Equation (3) in the paper
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.W_o(attention_output)
        
        # Add residual connection
        return output + residual


class SpatialAttention3D(nn.Module):
    """
    3D Spatial attention for processing volumetric protein/ligand/pocket data.
    This helps focus on important spatial regions in the 3D grids.
    """
    
    def __init__(self, in_channels):
        super(SpatialAttention3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv3d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate attention map
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        # Apply attention
        return x * attention


class ChannelAttention3D(nn.Module):
    """
    Channel attention for 3D volumetric data.
    Helps the model focus on important feature channels.
    """
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, d, h, w = x.size()
        
        # Global average pooling and max pooling
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        # Generate channel attention weights
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1, 1)
        
        return x * attention


class DilatedConvBlock3D(nn.Module):
    """
    Dilated convolution block as described in Section 3.4 of the paper.
    Captures multi-scale information without increasing parameters.
    """
    
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4]):
        super(DilatedConvBlock3D, self).__init__()
        
        # Calculate channels per dilated conv, handling remainder
        num_rates = len(dilation_rates)
        base_channels = out_channels // num_rates
        
        # Distribute remainder channels to first few dilated convs
        channels_per_conv = [base_channels + (1 if i < out_channels % num_rates else 0) 
                           for i in range(num_rates)]
        
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, channels_per_conv[i], 
                         kernel_size=3, padding=rate, dilation=rate),
                nn.BatchNorm3d(channels_per_conv[i]),
                nn.ReLU()
            ) for i, rate in enumerate(dilation_rates)
        ])
        
        # 1x1 conv to combine dilated features
        self.combine = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        dilated_outputs = [conv(x) for conv in self.dilated_convs]
        combined = torch.cat(dilated_outputs, dim=1)
        return self.combine(combined)


class AttentionEmbeddingLayer(nn.Module):
    """
    Embedding layer with self-attention as shown in Figure 1 of the paper.
    Processes each component (protein/ligand/pocket) through embedding and attention.
    """
    
    def __init__(self, in_channels, embed_dim, num_attention_heads=8):
        super(AttentionEmbeddingLayer, self).__init__()
        
        # Initial 3D convolution for feature extraction
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim // 2, 3, padding=1),
            nn.BatchNorm3d(embed_dim // 2),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(embed_dim // 2, embed_dim, 3, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(),
        )
        
        # Channel and spatial attention
        self.channel_attention = ChannelAttention3D(embed_dim)
        self.spatial_attention = SpatialAttention3D(embed_dim)
        
        # Adaptive pooling to fixed size for attention
        self.adaptive_pool = nn.AdaptiveAvgPool3d(4)
        
        # Self-attention module
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_attention_heads)
        
    def forward(self, x):
        # 3D convolution feature extraction
        x = self.conv3d(x)
        
        # Apply attention mechanisms
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        
        # Pool to fixed size
        x = self.adaptive_pool(x)  # (batch, embed_dim, 4, 4, 4)
        
        # Reshape for self-attention (batch, sequence_length, embed_dim)
        batch_size, embed_dim, d, h, w = x.size()
        x_flat = x.view(batch_size, embed_dim, -1).transpose(1, 2)  # (batch, 64, embed_dim)
        
        # Apply self-attention
        x_attended = self.self_attention(x_flat)
        
        # Reshape back to 3D
        x_attended = x_attended.transpose(1, 2).view(batch_size, embed_dim, d, h, w)
        
        return x_attended


# =========================================================================
# Enhanced Models with Self-Attention
# =========================================================================

class AttentionEnhancedCNN(nn.Module):
    """
    CNN model enhanced with self-attention mechanisms based on the research paper.
    
    This implements the architecture from Figure 1, with separate embedding layers
    for each component, self-attention, and dilated convolution fusion.
    """
    
    def __init__(self, protein_channels=19, ligand_channels=19, pocket_channels=19, 
                 embed_dim=128, num_attention_heads=8, num_classes=1):
        super(AttentionEnhancedCNN, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Component-specific embedding layers with self-attention
        self.protein_embedding = AttentionEmbeddingLayer(protein_channels, embed_dim, num_attention_heads)
        self.ligand_embedding = AttentionEmbeddingLayer(ligand_channels, embed_dim, num_attention_heads)
        self.pocket_embedding = AttentionEmbeddingLayer(pocket_channels, embed_dim, num_attention_heads)
        
        # Dilated convolution for feature fusion
        self.dilated_fusion = DilatedConvBlock3D(embed_dim * 3, embed_dim * 2, [1, 2, 4])
        
        # Cross-attention between components
        self.cross_attention = MultiHeadSelfAttention(embed_dim * 2, num_attention_heads)
        
        # Final feature processing
        self.final_conv = nn.Sequential(
            nn.Conv3d(embed_dim * 2, embed_dim, 3, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        
        # Classification head matching paper architecture
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 384),  # 384 x 128 as shown in Figure 1
            nn.Dropout(0.3),
            nn.Linear(384, 128),  # 128 x 64
            nn.Dropout(0.2),
            nn.Linear(128, 64),   # 64 x 1
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Proper weight initialization"""
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm3d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, protein_grid, ligand_grid, pocket_grid):
        # Process each component through embedding + self-attention
        protein_features = self.protein_embedding(protein_grid)
        ligand_features = self.ligand_embedding(ligand_grid)
        pocket_features = self.pocket_embedding(pocket_grid)
        
        # Concatenate features for fusion
        combined_features = torch.cat([protein_features, ligand_features, pocket_features], dim=1)
        
        # Apply dilated convolution fusion
        fused_features = self.dilated_fusion(combined_features)
        
        # Reshape for cross-attention
        batch_size, channels, d, h, w = fused_features.size()
        fused_flat = fused_features.view(batch_size, channels, -1).transpose(1, 2)
        
        # Apply cross-attention
        attended_features = self.cross_attention(fused_flat)
        
        # Reshape back and apply final convolution
        attended_features = attended_features.transpose(1, 2).view(batch_size, channels, d, h, w)
        final_features = self.final_conv(attended_features).flatten(1)
        
        # Final prediction
        output = self.classifier(final_features)
        return output


class LightweightAttentionCNN(nn.Module):
    """
    Lightweight version of attention-enhanced CNN for faster training and inference.
    """
    
    def __init__(self, protein_channels=19, ligand_channels=19, pocket_channels=19, 
                 embed_dim=64, num_attention_heads=4, num_classes=1):
        super(LightweightAttentionCNN, self).__init__()
        
        # Smaller embedding layers
        self.protein_embedding = AttentionEmbeddingLayer(protein_channels, embed_dim, num_attention_heads)
        self.ligand_embedding = AttentionEmbeddingLayer(ligand_channels, embed_dim, num_attention_heads)
        self.pocket_embedding = AttentionEmbeddingLayer(pocket_channels, embed_dim, num_attention_heads)
        
        # Simplified fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(embed_dim * 3, embed_dim * 2, 3, padding=1),
            nn.BatchNorm3d(embed_dim * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(2)
        )
        
        # Self-attention on fused features
        self.fusion_attention = MultiHeadSelfAttention(embed_dim * 2, num_attention_heads)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),  # After global average pooling
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm3d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, protein_grid, ligand_grid, pocket_grid):
        # Process components
        protein_features = self.protein_embedding(protein_grid)
        ligand_features = self.ligand_embedding(ligand_grid)
        pocket_features = self.pocket_embedding(pocket_grid)
        
        # Fuse and pool
        combined = torch.cat([protein_features, ligand_features, pocket_features], dim=1)
        fused = self.fusion_conv(combined)
        
        # Apply attention to fused features
        batch_size, channels, d, h, w = fused.size()
        fused_flat = fused.view(batch_size, channels, -1).transpose(1, 2)
        attended = self.fusion_attention(fused_flat)
        
        # Global pooling and classification
        pooled = attended.mean(dim=1)  # Global average pooling over sequence dimension
        output = self.classifier(pooled)
        return output


# =========================================================================
# Graph Neural Network Components (from FAST framework)
# ========================================================================

if TORCH_GEOMETRIC_AVAILABLE:
    
    class GatedGraphConv(MessagePassing):
        """
        Gated Graph Convolution layer adapted from FAST framework.
        """
        
        def __init__(self, out_channels, num_layers, edge_network=None, aggr="add", bias=True):
            super(GatedGraphConv, self).__init__(aggr)
            
            self.out_channels = out_channels
            self.num_layers = num_layers
            self.edge_network = edge_network
            
            self.weight = nn.Parameter(torch.Tensor(num_layers, out_channels, out_channels))
            self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
            self.reset_parameters()
        
        def reset_parameters(self):
            size = self.out_channels
            nn.init.uniform_(self.weight, -1/np.sqrt(size), 1/np.sqrt(size))
            self.rnn.reset_parameters()
        
        def forward(self, x, edge_index, edge_attr=None):
            h = x if x.dim() == 2 else x.unsqueeze(-1)
            assert h.size(1) <= self.out_channels
            
            # Pad with zeros if needed
            if h.size(1) < self.out_channels:
                zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
                h = torch.cat([h, zero], dim=1)
            
            for i in range(self.num_layers):
                m = torch.matmul(h, self.weight[i])
                m = self.propagate(edge_index=edge_index, x=m, aggr="add")
                h = self.rnn(m, h)
            
            return h
        
        def message(self, x_j):
            return x_j
    
    
    class PotentialNetAttention(torch.nn.Module):
        """Attention mechanism for graph neural networks."""
        
        def __init__(self, net_i, net_j):
            super(PotentialNetAttention, self).__init__()
            self.net_i = net_i
            self.net_j = net_j
        
        def forward(self, h_i, h_j):
            return torch.nn.Softmax(dim=1)(
                self.net_i(torch.cat([h_i, h_j], dim=1))
            ) * self.net_j(h_j)
    
    
    class GraphConvolutionalLayer(nn.Module):
        """
        Graph convolutional layer for processing molecular graphs.
        """
        
        def __init__(self, in_features, out_features, edge_dim=1):
            super(GraphConvolutionalLayer, self).__init__()
            
            self.in_features = in_features
            self.out_features = out_features
            
            # Edge network for NNConv
            self.edge_network = nn.Sequential(
                nn.Linear(edge_dim, in_features // 2),
                nn.ReLU(),
                nn.Linear(in_features // 2, in_features * out_features),
                nn.ReLU()
            )
            
            # Graph convolution layer
            self.conv = NNConv(
                in_features, 
                out_features,
                nn=self.edge_network,
                aggr='add'
            )
            
            self.norm = nn.BatchNorm1d(out_features)
            self.activation = nn.ReLU()
        
        def forward(self, x, edge_index, edge_attr, batch=None):
            x = self.conv(x, edge_index, edge_attr)
            x = self.norm(x)
            x = self.activation(x)
            return x
    
    
    class HybridCNNGNN(nn.Module):
        """
        Hybrid model that combines CNN feature extraction with GNN molecular understanding.
        
        This model first uses CNNs to extract spatial features from protein, ligand, and pocket grids,
        then optionally processes molecular graphs with GNNs for enhanced molecular representation.
        """
        
        def __init__(self, protein_channels=19, ligand_channels=19, pocket_channels=19, 
                     node_features=None, edge_features=4, use_gnn=True, num_classes=1):
            super(HybridCNNGNN, self).__init__()
            
            self.use_gnn = use_gnn and TORCH_GEOMETRIC_AVAILABLE
            
            # Calculate automatic node features if not provided
            if node_features is None and self.use_gnn:
                # Node features = concatenated features from all three grids
                node_features = protein_channels + ligand_channels + pocket_channels
            
            self.node_features = node_features
            self.edge_features = edge_features
            
            # CNN Components (similar to StableMultiComponentCNN but smaller)
            self.protein_encoder = nn.Sequential(
                nn.Conv3d(protein_channels, 16, 3, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(16, 32, 3, padding=1),
                nn.BatchNorm3d(32),
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
                nn.AdaptiveAvgPool3d(4)
            )
            
            # Calculate CNN feature dimensions
            cnn_features = 32 * 4 * 4 * 4 * 3  # 3 components
            self.cnn_features = cnn_features
            
            if self.use_gnn and self.node_features is not None:
                # GNN Components
                self.gnn_layers = nn.ModuleList([
                    GraphConvolutionalLayer(self.node_features, 64, self.edge_features),
                    GraphConvolutionalLayer(64, 128, self.edge_features),
                    GraphConvolutionalLayer(128, 64, self.edge_features)
                ])
                
                self.graph_attention = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
                
                # GNN feature dimension
                self.gnn_features = 64
                
                # Create separate fusion layers for CNN-only and CNN+GNN modes
                self.cnn_only_fusion = nn.Sequential(
                    nn.Linear(cnn_features, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU()
                )
                
                self.cnn_gnn_fusion = nn.Sequential(
                    nn.Linear(cnn_features + 64, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU()
                )
            else:
                # CNN-only mode
                self.cnn_only_fusion = nn.Sequential(
                    nn.Linear(cnn_features, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU()
                )
            
            self.classifier = nn.Linear(64, num_classes)
            
            # Initialize weights
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
        
        def forward(self, protein_grid, ligand_grid, pocket_grid, graph_data=None):
            # CNN feature extraction
            protein_features = self.protein_encoder(protein_grid).flatten(1)
            ligand_features = self.ligand_encoder(ligand_grid).flatten(1)
            pocket_features = self.pocket_encoder(pocket_grid).flatten(1)
            
            # Combine CNN features
            cnn_features = torch.cat([protein_features, ligand_features, pocket_features], dim=1)
            
            # GNN processing (if available and graph data provided)
            if self.use_gnn and graph_data is not None and hasattr(self, 'gnn_layers'):
                try:
                    x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr
                    batch = getattr(graph_data, 'batch', None)
                    
                    # Process through GNN layers
                    for gnn_layer in self.gnn_layers:
                        x = gnn_layer(x, edge_index, edge_attr, batch)
                    
                    # Global pooling with attention
                    if batch is not None:
                        # Attention-weighted global pooling
                        attention_weights = self.graph_attention(x)
                        x_weighted = x * attention_weights
                        gnn_features = global_add_pool(x_weighted, batch)
                    else:
                        # Simple global pooling for single graph
                        gnn_features = torch.mean(x, dim=0, keepdim=True)
                    
                    # Use CNN+GNN fusion
                    combined_features = torch.cat([cnn_features, gnn_features], dim=1)
                    fused_features = self.cnn_gnn_fusion(combined_features)
                
                except Exception as e:
                    print(f"Warning: GNN processing failed: {e}, using CNN features only")
                    # Fallback to CNN-only
                    fused_features = self.cnn_only_fusion(cnn_features)
            else:
                # CNN-only mode
                fused_features = self.cnn_only_fusion(cnn_features)
            
            # Final prediction
            output = self.classifier(fused_features)
            
            return output
    
    
    class GraphOnlyModel(nn.Module):
        """
        Pure graph neural network model for molecular property prediction.
        """
        
        def __init__(self, node_features, edge_features=1, hidden_dim=128, num_layers=3, num_classes=1):
            super(GraphOnlyModel, self).__init__()
            
            if not TORCH_GEOMETRIC_AVAILABLE:
                raise ImportError("torch_geometric is required for GraphOnlyModel")
            
            self.num_layers = num_layers
            
            # GNN layers
            self.gnn_layers = nn.ModuleList()
            
            # First layer
            self.gnn_layers.append(
                GraphConvolutionalLayer(node_features, hidden_dim, edge_features)
            )
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.gnn_layers.append(
                    GraphConvolutionalLayer(hidden_dim, hidden_dim, edge_features)
                )
            
            # Last layer
            self.gnn_layers.append(
                GraphConvolutionalLayer(hidden_dim, hidden_dim, edge_features)
            )
            
            # Global attention mechanism
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            
            self.apply(self._init_weights)
        
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        def forward(self, graph_data):
            x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr
            batch = getattr(graph_data, 'batch', None)
            
            # Process through GNN layers
            for gnn_layer in self.gnn_layers:
                x = gnn_layer(x, edge_index, edge_attr, batch)
            
            # Global pooling with attention
            if batch is not None:
                attention_weights = self.attention(x)
                x_weighted = x * attention_weights
                graph_features = global_add_pool(x_weighted, batch)
            else:
                # Simple global pooling for single graph
                graph_features = torch.mean(x, dim=0, keepdim=True)
            
            # Final prediction
            output = self.classifier(graph_features)
            return output

else:
    # Dummy classes when torch_geometric is not available
    class HybridCNNGNN(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("torch_geometric is required for HybridCNNGNN model")
    
    class GraphOnlyModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("torch_geometric is required for GraphOnlyModel")


# =========================================================================
# Original CNN Models (Enhanced with Optional Attention)
# =========================================================================

class StableMultiComponentCNN(nn.Module):
    """
    Stable Multi-Component CNN for protein-ligand binding affinity prediction.
    Enhanced with optional self-attention mechanisms.
    
    This model processes protein, ligand, and pocket grids separately through
    dedicated encoders, then fuses the features for final prediction.
    """
    
    def __init__(self, protein_channels=19, ligand_channels=19, pocket_channels=19, 
                 num_classes=1, use_attention=False):
        super(StableMultiComponentCNN, self).__init__()
        
        self.use_attention = use_attention
        
        # Component-specific encoders with proper initialization
        self.protein_encoder = nn.Sequential(
            nn.Conv3d(protein_channels, 16, 3, padding=1),
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
        
        # Optional attention mechanisms
        if self.use_attention:
            self.protein_attention = ChannelAttention3D(64)
            self.ligand_attention = ChannelAttention3D(64)
            self.pocket_attention = ChannelAttention3D(64)
            
            # Self-attention for fused features
            self.feature_attention = MultiHeadSelfAttention(64 * 4 * 4 * 4, num_heads=8)
        
        # Fusion layer with dropout
        self.fusion = nn.Sequential(
            nn.Linear(64 * 4 * 4 * 4 * 3, 256),
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
        protein_features = self.protein_encoder(protein_grid)
        ligand_features = self.ligand_encoder(ligand_grid)
        pocket_features = self.pocket_encoder(pocket_grid)
        
        # Apply attention if enabled
        if self.use_attention:
            protein_features = self.protein_attention(protein_features)
            ligand_features = self.ligand_attention(ligand_features)
            pocket_features = self.pocket_attention(pocket_features)
        
        # Flatten features
        protein_flat = protein_features.flatten(1)
        ligand_flat = ligand_features.flatten(1)
        pocket_flat = pocket_features.flatten(1)
        
        # Fuse features
        combined = torch.cat([protein_flat, ligand_flat, pocket_flat], dim=1)
        
        # Apply self-attention to combined features if enabled
        if self.use_attention:
            # Reshape for attention (batch, sequence_length=3, feature_dim)
            batch_size = combined.size(0)
            feature_dim = combined.size(1) // 3
            combined_reshaped = combined.view(batch_size, 3, feature_dim)
            attended = self.feature_attention(combined_reshaped)
            combined = attended.view(batch_size, -1)
        
        fused = self.fusion(combined)
        
        # Predict binding affinity
        output = self.classifier(fused)
        return output


class MemoryEfficientCNN(nn.Module):
    """
    Memory-efficient CNN for protein-ligand binding affinity prediction.
    Enhanced with lightweight attention mechanisms.
    
    This model uses smaller feature maps and aggressive pooling to reduce
    memory usage while maintaining predictive performance.
    """
    
    def __init__(self, protein_channels=19, ligand_channels=19, pocket_channels=19, 
                 num_classes=1, use_attention=False):
        super(MemoryEfficientCNN, self).__init__()
        
        self.use_attention = use_attention
        
        # Smaller encoders to reduce memory usage
        self.protein_encoder = nn.Sequential(
            nn.Conv3d(protein_channels, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(4),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(2)
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
        
        # Lightweight attention
        if self.use_attention:
            self.channel_attention = ChannelAttention3D(16, reduction=8)
            
        # Smaller fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(16 * 2 * 2 * 2 * 3, 64),
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
        protein_features = self.protein_encoder(protein_grid)
        ligand_features = self.ligand_encoder(ligand_grid)
        pocket_features = self.pocket_encoder(pocket_grid)
        
        # Apply lightweight attention if enabled
        if self.use_attention:
            protein_features = self.channel_attention(protein_features)
            ligand_features = self.channel_attention(ligand_features)
            pocket_features = self.channel_attention(pocket_features)
        
        # Flatten and fuse features
        protein_flat = protein_features.flatten(1)
        ligand_flat = ligand_features.flatten(1)
        pocket_flat = pocket_features.flatten(1)
        
        combined = torch.cat([protein_flat, ligand_flat, pocket_flat], dim=1)
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


# =========================================================================
# Factory Functions and Utilities
# =========================================================================

def get_model(model_name, **kwargs):
    """
    Factory function to create models by name.
    
    Args:
        model_name (str): One of 'stable', 'memory_efficient', 'simple', 'hybrid_cnn_gnn', 
                         'graph_only', 'attention_enhanced', 'lightweight_attention'
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        torch.nn.Module: The requested model
    """
    models = {
        'stable': StableMultiComponentCNN,
        'memory_efficient': MemoryEfficientCNN,
        'simple': SimpleBindingCNN,
        'hybrid_cnn_gnn': HybridCNNGNN,
        'graph_only': GraphOnlyModel,
        'attention_enhanced': AttentionEnhancedCNN,
        'lightweight_attention': LightweightAttentionCNN
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)


def model_summary(model, input_shape=(1, 19, 64, 64, 64), graph_data=None):
    """
    Print a summary of the model including parameter count and memory usage.
    
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
            if isinstance(model, (HybridCNNGNN, GraphOnlyModel)) and TORCH_GEOMETRIC_AVAILABLE:
                if isinstance(model, GraphOnlyModel):
                    if graph_data is None:
                        print("Warning: GraphOnlyModel requires graph_data for summary")
                        return
                    output = model(graph_data)
                else:
                    # HybridCNNGNN can work with or without graph data
                    output = model(dummy_protein, dummy_ligand, dummy_pocket, graph_data)
            else:
                # Standard CNN models (including new attention models)
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
    
    print(f"\nModel Summary:")
    print(f"{'='*50}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {input_shape} (per component)")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Estimated memory usage: {estimated_memory_mb:.1f} MB")
    if hasattr(model, 'use_gnn') and model.use_gnn:
        print(f"Graph Neural Network: Enabled")
    if hasattr(model, 'use_attention') and model.use_attention:
        print(f"Self-Attention: Enabled")
    print(f"{'='*50}")
