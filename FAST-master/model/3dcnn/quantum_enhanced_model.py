################################################################################
# Copyright 2019-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Quantum-Enhanced 3D CNN for Protein-Ligand Binding Affinity Prediction
# Integrates quantum circuits into the CNN architecture for improved performance
################################################################################

import os
import sys
import math
import numbers
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import PennyLane for quantum capabilities
try:
    import pennylane as qml
    QUANTUM_AVAILABLE = True
    print("PennyLane detected - Quantum layers will be enabled")
except ImportError:
    QUANTUM_AVAILABLE = False
    print("PennyLane not available - Using classical layers only")
    print("To enable quantum features, install with: pip install pennylane")


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix) :]
        state_dict[newkey] = state_dict.pop(key)

    try:
        metadata = state_dict._metadata
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)


class QuantumLayer(nn.Module):
    """
    Quantum layer implementation using PennyLane
    Can be used as a drop-in replacement for classical layers
    """
    def __init__(self, input_dim, output_dim=None, n_qubits=4, n_layers=2, use_cuda=True):
        super(QuantumLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.n_qubits = min(n_qubits, 8)  # Limit to 8 qubits for practical reasons
        self.n_layers = n_layers
        self.use_cuda = use_cuda
        
        if not QUANTUM_AVAILABLE:
            # Fallback to classical layer if quantum not available
            self.classical_layer = nn.Linear(input_dim, self.output_dim)
            self.is_quantum = False
            return
        
        self.is_quantum = True
        
        # Create quantum device with better device selection and error handling
        device_name = "default.qubit"  # Start with most compatible device
        diff_method = "parameter-shift"  # Most compatible differentiation method
        
        if use_cuda and torch.cuda.is_available():
            try:
                # Try GPU-accelerated lightning first
                self.dev = qml.device("lightning.gpu", wires=self.n_qubits)
                diff_method = "adjoint"  # Lightning devices support adjoint method
                device_name = "lightning.gpu"
            except:
                try:
                    # Fallback to CPU lightning
                    self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
                    diff_method = "adjoint"  # Lightning devices support adjoint method
                    device_name = "lightning.qubit"
                except:
                    # Final fallback to default.qubit
                    self.dev = qml.device("default.qubit", wires=self.n_qubits)
                    diff_method = "parameter-shift"
                    device_name = "default.qubit"
        else:
            try:
                # Try lightning.qubit on CPU
                self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
                diff_method = "adjoint"
                device_name = "lightning.qubit"
            except:
                # Fallback to default.qubit
                self.dev = qml.device("default.qubit", wires=self.n_qubits)
                diff_method = "parameter-shift"
                device_name = "default.qubit"
        
        if QUANTUM_AVAILABLE:
            print(f"Using quantum device: {device_name} with diff_method: {diff_method}")

        # Define quantum circuit with compatible differentiation method
        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def quantum_circuit(inputs, weights):
            # Data encoding - angle embedding
            qml.AngleEmbedding(inputs[:self.n_qubits], wires=range(self.n_qubits), rotation='Y')
            
            # Variational ansatz with entangling layers
            for layer in range(self.n_layers):
                # Rotation gates
                for qubit in range(self.n_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)
                
                # Entangling gates
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                # Ring connectivity
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # Measurements
            return [qml.expval(qml.PauliZ(wire)) for wire in range(self.n_qubits)]
        
        self.quantum_circuit = quantum_circuit
        
        # Weight shape: (n_layers, n_qubits, 3)
        self.weights = nn.Parameter(torch.randn(self.n_layers, self.n_qubits, 3, dtype=torch.float32) * 0.1)
        
        # Classical post-processing layer
        self.post_process = nn.Linear(self.n_qubits, self.output_dim)
        
    def forward(self, x):
        if not self.is_quantum:
            return self.classical_layer(x)
        
        batch_size = x.shape[0]
        quantum_outputs = []
        
        for i in range(batch_size):
            # Pad or truncate input to match n_qubits
            if x.shape[1] >= self.n_qubits:
                quantum_input = x[i, :self.n_qubits].float()
            else:
                quantum_input = torch.cat([
                    x[i], 
                    torch.zeros(self.n_qubits - x.shape[1], device=x.device, dtype=x.dtype)
                ]).float()
            
            # Run quantum circuit
            quantum_output = self.quantum_circuit(quantum_input, self.weights)
            quantum_outputs.append(torch.stack(quantum_output).float())
        
        quantum_batch = torch.stack(quantum_outputs).float()
        
        # Post-process through classical layer
        return self.post_process(quantum_batch)


class QuantumAttentionLayer(nn.Module):
    """
    Quantum-enhanced attention mechanism
    """
    def __init__(self, feature_dim, n_qubits=4, use_cuda=True):
        super(QuantumAttentionLayer, self).__init__()
        self.feature_dim = feature_dim
        self.n_qubits = min(n_qubits, 6)  # Smaller for attention
        self.use_cuda = use_cuda
        
        if not QUANTUM_AVAILABLE:
            # Classical attention fallback
            self.attention = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)
            self.is_quantum = False
            return
        
        self.is_quantum = True
        
        # Quantum attention weights computation
        self.quantum_attention = QuantumLayer(
            input_dim=feature_dim, 
            output_dim=feature_dim, 
            n_qubits=self.n_qubits,
            n_layers=1,
            use_cuda=use_cuda
        )
        
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        if not self.is_quantum:
            attn_output, _ = self.attention(x, x, x)
            return attn_output
        
        # Apply quantum transformation for attention weights
        batch_size, seq_len, feature_dim = x.shape
        
        # Reshape for quantum processing
        x_flat = x.view(-1, feature_dim)
        quantum_weights = self.quantum_attention(x_flat)
        quantum_weights = quantum_weights.view(batch_size, seq_len, feature_dim)
        
        # Apply attention weights
        attention_weights = torch.softmax(quantum_weights, dim=1)
        attended_output = x * attention_weights
        
        return self.norm(attended_output)


class QuantumEnhanced3DCNN(nn.Module):
    """
    Quantum-Enhanced 3D CNN Model for protein-ligand binding affinity prediction
    """
    
    def __init__(self, feat_dim=19, output_dim=1, num_filters=[64,128,256], 
                 use_cuda=True, verbose=0, quantum_features=True, quantum_attention=True):
        super(QuantumEnhanced3DCNN, self).__init__()
        
        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.num_filters = num_filters
        self.use_cuda = use_cuda
        self.verbose = verbose
        self.quantum_features = quantum_features and QUANTUM_AVAILABLE
        self.quantum_attention = quantum_attention and QUANTUM_AVAILABLE
        
        # Print configuration
        if self.verbose:
            print(f"Quantum Features: {'Enabled' if self.quantum_features else 'Disabled'}")
            print(f"Quantum Attention: {'Enabled' if self.quantum_attention else 'Disabled'}")
        
        # Traditional CNN layers (same as original)
        self.conv_block1 = self.__conv_layer_set__(self.feat_dim, self.num_filters[0], 7, 2, 3)
        self.res_block1 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[0], 7, 1, 3)
        self.res_block2 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[0], 7, 1, 3)

        self.conv_block2 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[1], 7, 3, 3)
        self.max_pool2 = nn.MaxPool3d(2)

        self.conv_block3 = self.__conv_layer_set__(self.num_filters[1], self.num_filters[2], 5, 2, 2)
        self.max_pool3 = nn.MaxPool3d(2)

        # Quantum-enhanced feature extraction
        fc_len = 16  # Increased from 10 to accommodate quantum features
        
        if self.quantum_features:
            # Quantum layer for feature enhancement
            self.quantum_feature_layer = QuantumLayer(
                input_dim=2048, 
                output_dim=32,  # Quantum-enhanced features
                n_qubits=6,
                n_layers=3,
                use_cuda=use_cuda
            )
            
            # Attention mechanism with quantum enhancement
            if self.quantum_attention:
                self.quantum_attention_layer = QuantumAttentionLayer(
                    feature_dim=32,
                    n_qubits=4,
                    use_cuda=use_cuda
                )
            
            # Combine classical and quantum features
            self.fc1 = nn.Linear(2048 + 32, fc_len)  # Classical + quantum features
        else:
            self.fc1 = nn.Linear(2048, fc_len)
        
        torch.nn.init.normal_(self.fc1.weight, 0, 1)
        self.fc1_bn = nn.BatchNorm1d(num_features=fc_len, affine=True, momentum=0.1).train()
        
        # Quantum-enhanced final prediction layer
        if self.quantum_features:
            self.quantum_prediction = QuantumLayer(
                input_dim=fc_len,
                output_dim=4,  # Intermediate quantum representation
                n_qubits=4,
                n_layers=2,
                use_cuda=use_cuda
            )
            self.fc2 = nn.Linear(4, 1)  # Final prediction
        else:
            self.fc2 = nn.Linear(fc_len, 1)
        
        torch.nn.init.normal_(self.fc2.weight, 0, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def __conv_layer_set__(self, in_c, out_c, k_size, stride, padding):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=k_size, stride=stride, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_c))
        return conv_layer

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        # Classical CNN forward pass (unchanged)
        conv1_h = self.conv_block1(x)
        if self.verbose != 0:
            print(f"conv1_h shape: {conv1_h.shape}")

        conv1_res1_h = self.res_block1(conv1_h)
        if self.verbose != 0:
            print(f"conv1_res1_h shape: {conv1_res1_h.shape}")

        conv1_res1_h2 = conv1_res1_h + conv1_h
        if self.verbose != 0:
            print(f"conv1_res1_h2 shape: {conv1_res1_h2.shape}")

        conv1_res2_h = self.res_block2(conv1_res1_h2)
        if self.verbose != 0:
            print(f"conv1_res2_h shape: {conv1_res2_h.shape}")

        conv1_res2_h2 = conv1_res2_h + conv1_h
        if self.verbose != 0:
            print(f"conv1_res2_h2 shape: {conv1_res2_h2.shape}")

        conv2_h = self.conv_block2(conv1_res2_h2)
        if self.verbose != 0:
            print(f"conv2_h shape: {conv2_h.shape}")

        pool2_h = self.max_pool2(conv2_h)
        if self.verbose != 0:
            print(f"pool2_h shape: {pool2_h.shape}")

        conv3_h = self.conv_block3(pool2_h)
        if self.verbose != 0:
            print(f"conv3_h shape: {conv3_h.shape}")

        pool3_h = conv3_h
        
        flatten_h = pool3_h.view(pool3_h.size(0), -1)
        if self.verbose != 0:
            print(f"flatten_h shape: {flatten_h.shape}")

        # Quantum-enhanced feature processing
        if self.quantum_features:
            # Extract quantum features
            quantum_features = self.quantum_feature_layer(flatten_h)
            if self.verbose != 0:
                print(f"quantum_features shape: {quantum_features.shape}")
            
            # Apply quantum attention if enabled
            if self.quantum_attention:
                # Reshape for attention (add sequence dimension)
                quantum_features_seq = quantum_features.unsqueeze(1)  # [batch, 1, features]
                quantum_features_attended = self.quantum_attention_layer(quantum_features_seq)
                quantum_features = quantum_features_attended.squeeze(1)  # [batch, features]
                if self.verbose != 0:
                    print(f"quantum_features_attended shape: {quantum_features.shape}")
            
            # Combine classical and quantum features
            combined_features = torch.cat([flatten_h, quantum_features], dim=1)
            fc1_z = self.fc1(combined_features)
        else:
            fc1_z = self.fc1(flatten_h)
        
        fc1_y = self.relu(fc1_z)
        fc1_h = self.fc1_bn(fc1_y) if fc1_y.shape[0] > 1 else fc1_y
        if self.verbose != 0:
            print(f"fc1_h shape: {fc1_h.shape}")

        # Quantum-enhanced final prediction
        if self.quantum_features:
            quantum_pred_features = self.quantum_prediction(fc1_h)
            if self.verbose != 0:
                print(f"quantum_pred_features shape: {quantum_pred_features.shape}")
            fc2_z = self.fc2(quantum_pred_features)
        else:
            fc2_z = self.fc2(fc1_h)
        
        if self.verbose != 0:
            print(f"fc2_z shape: {fc2_z.shape}")

        return fc2_z, fc1_z


# Alias for backward compatibility
Model_3DCNN_Quantum = QuantumEnhanced3DCNN


def create_quantum_model(use_quantum=True, **kwargs):
    """
    Factory function to create either quantum-enhanced or classical model
    """
    if use_quantum and QUANTUM_AVAILABLE:
        return QuantumEnhanced3DCNN(quantum_features=True, quantum_attention=True, **kwargs)
    else:
        if use_quantum and not QUANTUM_AVAILABLE:
            print("Warning: Quantum features requested but PennyLane not available. Using classical model.")
        return QuantumEnhanced3DCNN(quantum_features=False, quantum_attention=False, **kwargs)


if __name__ == "__main__":
    # Test the model
    print("Testing Quantum-Enhanced 3D CNN Model...")
    
    # Test with dummy data
    batch_size = 2
    channels = 19
    depth = height = width = 48
    
    x = torch.randn(batch_size, channels, depth, height, width)
    
    # Test quantum model
    model_quantum = create_quantum_model(use_quantum=True, verbose=1)
    print(f"\nModel created with quantum features: {model_quantum.quantum_features}")
    
    with torch.no_grad():
        output, features = model_quantum(x)
        print(f"Output shape: {output.shape}")
        print(f"Features shape: {features.shape}")
    
    print("Model test completed successfully!")
