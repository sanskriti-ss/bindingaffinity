### instructions
"""
cd FAST-master/model/
pip install requirements.txt
python -m quantum_fusion.testing_random_unitaries

OR

cd FAST-master/model/quantum_fusion/
python testing_random_unitaries.py
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.connectors import TorchConnector
from tqdm import tqdm
import sys
import os

# Handle both relative and direct imports
try:
    from .main_train import ModelHybridFC, FusionDataset, evaluate_model
except ImportError:
    # If running as script directly from quantum_fusion directory
    from main_train import ModelHybridFC, FusionDataset, evaluate_model

# 1. Generate 10 random unitary circuits
def generate_random_circuits(n_qubits, depth, num_circuits=4):
    circuits = []
    for _ in range(num_circuits):
        qc = random_circuit(n_qubits, depth, max_operands=2, measure=False)
        circuits.append(qc)
    return circuits

def load_preprocessed_data():
    """Load preprocessed molecular data from step4 outputs"""
    import json
    import os
    
    # Get the working directory (should be FAST-master/model)
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(current_dir, '..', '..', 'model_ready_data')
    
    print(f"Loading preprocessed data from: {data_dir}")
    
    # Load ligand grids and metadata
    ligand_npz_path = os.path.join(data_dir, 'ligand_data', 'ligand_data.npz')
    ligand_npz = np.load(ligand_npz_path)
    ligand_grids = ligand_npz['arr_0']
    
    with open(os.path.join(data_dir, 'ligand_data', 'ligand_metadata.json'), 'r') as f:
        ligand_metadata = json.load(f)
    
    # Load pocket grids and metadata
    pocket_npz_path = os.path.join(data_dir, 'pocket_data', 'pocket_grids.npz')
    pocket_npz = np.load(pocket_npz_path)
    pocket_grids = pocket_npz['arr_0']
    
    with open(os.path.join(data_dir, 'pocket_data', 'pocket_metadata.json'), 'r') as f:
        pocket_metadata = json.load(f)
    
    # Flatten grids for input to the model
    # ligand_grids shape: (num_samples, height, width, depth, channels)
    # pocket_grids shape: (num_samples, height, width, depth, channels)
    num_samples = min(len(ligand_grids), len(pocket_grids))
    
    ligand_features = ligand_grids[:num_samples].reshape(num_samples, -1)
    pocket_features = pocket_grids[:num_samples].reshape(num_samples, -1)
    
    # Get binding affinity labels from metadata
    # Extract dG values (binding affinity) from ligand metadata
    labels = np.array([ligand_metadata[str(i)].get('dG', 0.0) for i in range(num_samples)], dtype=np.float32)
    labels = labels.reshape(-1, 1)  # Reshape for regression
    
    print(f"Loaded {num_samples} samples")
    print(f"Ligand features shape: {ligand_features.shape}")
    print(f"Pocket features shape: {pocket_features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Split into train/val
    train_size = int(0.9 * num_samples)
    train_idx = np.arange(0, train_size)
    val_idx = np.arange(train_size, num_samples)
    
    train_dataset = FusionDataset(
        torch.tensor(pocket_features[train_idx], dtype=torch.float32),
        torch.tensor(ligand_features[train_idx], dtype=torch.float32),
        torch.tensor(labels[train_idx], dtype=torch.float32)
    )
    val_dataset = FusionDataset(
        torch.tensor(pocket_features[val_idx], dtype=torch.float32),
        torch.tensor(ligand_features[val_idx], dtype=torch.float32),
        torch.tensor(labels[val_idx], dtype=torch.float32)
    )
    
    return train_dataset, val_dataset

# 2. Run each circuit and record performance

def run_circuits_and_evaluate(circuits, train_dataset, val_dataset, n_qubits=4):
    results = []
    sgcnn_dim = train_dataset.sgcnn_features.shape[1]
    cnn3d_dim = train_dataset.cnn3d_features.shape[1]
    total_dim = sgcnn_dim + cnn3d_dim
    
    # Progress bar for circuits
    circuit_progress = tqdm(enumerate(circuits), total=len(circuits), desc="Testing Unitaries", position=0)
    
    for idx, qc in circuit_progress:
        circuit_progress.set_description(f"Testing Unitary {idx + 1}/{len(circuits)}")
        
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
        
        # Progress bar for epochs
        epoch_progress = tqdm(range(5), desc=f"  Training Unitary {idx + 1}", position=1, leave=False)
        
        for epoch in epoch_progress:
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
            
            epoch_progress.set_postfix({'train_loss': f'{train_loss:.4f}', 'val_loss': f'{val_loss:.4f}'})
        
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
        
        circuit_progress.set_postfix({'RMSE': f'{rmse:.4f}', 'R2': f'{r2:.4f}'})
    
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
    circuits = generate_random_circuits(n_qubits, depth, num_circuits=10)
    train_dataset, val_dataset = load_preprocessed_data()
    results = run_circuits_and_evaluate(circuits, train_dataset, val_dataset, n_qubits)
    save_and_plot_results(results)
