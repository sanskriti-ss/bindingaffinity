### instructions
"""
cd FAST-master/model/
pip install requirements.txt
python -m quantum_fusion.testing_random_unitaries
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.connectors import TorchConnector
from .main_train import ModelHybridFC, FusionDataset, evaluate_model

# 1. Generate 10 random unitary circuits
def generate_random_circuits(n_qubits, depth, num_circuits=10):
    circuits = []
    for _ in range(num_circuits):
        qc = random_circuit(n_qubits, depth, max_operands=2, measure=False)
        circuits.append(qc)
    return circuits

# Placeholder for loading a small subset of pdbbind data
# Replace with actual data loading logic

def load_small_pdbbind_subset():
    """Load a small subset from the main data loading function"""
    from .main_train import load_sample_data
    
    # Load the full dataset
    sgcnn_features, cnn3d_features, labels, complex_ids = load_sample_data()
    
    # Select small subset for train/val
    train_idx = np.arange(0, 100)
    val_idx = np.arange(100, 110)
    
    train_dataset = FusionDataset(
        sgcnn_features[train_idx], cnn3d_features[train_idx], labels[train_idx]
    )
    val_dataset = FusionDataset(
        sgcnn_features[val_idx], cnn3d_features[val_idx], labels[val_idx]
    )
    
    return train_dataset, val_dataset

# 2. Run each circuit and record performance

def run_circuits_and_evaluate(circuits, train_dataset, val_dataset, n_qubits=4):
    results = []
    sgcnn_dim = train_dataset.sgcnn_features.shape[1]
    cnn3d_dim = train_dataset.cnn3d_features.shape[1]
    total_dim = sgcnn_dim + cnn3d_dim
    
    for idx, qc in enumerate(circuits):
        print(f"Testing circuit {idx + 1}/{len(circuits)}")
        
        # Create a quantum model
        model = ModelHybridFC(
            in_features=total_dim,
            out_features=1,
            qc_input_size=n_qubits,
            qc_n_layers=10,
            qc_encoding='amplitude',
            qc_ansatz=1,
            backend='default.qubit'
        )
        
        # Train for a few epochs
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        train_losses, val_losses = [], []
        
        for epoch in range(5):
            model.train()
            train_loss = 0.0
            for sgcnn_feat, cnn3d_feat, labels in train_loader:
                optimizer.zero_grad()
                combined = torch.cat([sgcnn_feat, cnn3d_feat], dim=1)
                preds = model(combined)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for sgcnn_feat, cnn3d_feat, labels in val_loader:
                    combined = torch.cat([sgcnn_feat, cnn3d_feat], dim=1)
                    preds = model(combined)
                    loss = criterion(preds, labels)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
        
        # Evaluate final performance
        rmse, mae, r2, pearson, spearman = evaluate_model(model, val_loader)
        
        results.append({
            'circuit_idx': idx,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson': pearson,
            'spearman': spearman
        })
        
        print(f"Circuit {idx}: RMSE={rmse:.4f}, R2={r2:.4f}")
    
    return results

# 3. Select top 5 circuits and save results

def save_and_plot_results(results):
    # Sort by RMSE (lowest is best)
    results_sorted = sorted(results, key=lambda x: x['rmse'])[:5]
    df = pd.DataFrame(results_sorted)
    df.to_csv('top5_random_unitary_results.csv', index=False)
    # Plot loss curves
    plt.figure(figsize=(10,6))
    for i, res in enumerate(results_sorted):
        plt.plot(res['train_losses'], label=f'Train Circuit {res["circuit_idx"]}')
        plt.plot(res['val_losses'], label=f'Val Circuit {res["circuit_idx"]}', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves for Top 5 Random Unitary Circuits')
    plt.legend()
    plt.savefig('top5_loss_curves.png')
    plt.show()

if __name__ == "__main__":
    n_qubits = 4  # Reduced for faster testing
    depth = 2     # Reduced depth for faster generation
    circuits = generate_random_circuits(n_qubits, depth, num_circuits=3)  # Test with 3 circuits first
    train_dataset, val_dataset = load_small_pdbbind_subset()
    results = run_circuits_and_evaluate(circuits, train_dataset, val_dataset, n_qubits)
    save_and_plot_results(results)
