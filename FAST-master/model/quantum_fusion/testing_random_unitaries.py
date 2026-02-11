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
from datetime import datetime

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
    
    # Try multiple possible paths for the data directory
    # The script is at: bindingaffinity/FAST-master/model/quantum_fusion/testing_random_unitaries.py
    # We need to reach: bindingaffinity/model_ready_data
    script_dir = os.path.dirname(os.path.abspath(__file__))  # quantum_fusion dir
    
    possible_paths = [
        # Relative to current working directory (when running from bindingaffinity root)
        'model_ready_data',
        # Relative to script location (go up 4 levels: quantum_fusion -> model -> FAST-master -> bindingaffinity)
        os.path.join(script_dir, '..', '..', '..', 'model_ready_data'),
        # Also try from FAST-master root
        os.path.join(script_dir, '..', '..', '..', '..', 'model_ready_data'),
    ]
    
    data_dir = None
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            data_dir = abs_path
            break
    
    if data_dir is None:
        raise FileNotFoundError(
            f"Could not find model_ready_data directory. Tried:\n"
            + "\n".join([f"  - {os.path.abspath(p)}" for p in possible_paths])
        )
    
    print(f"Loading preprocessed data from: {data_dir}")
    
    # Load ligand grids and metadata
    ligand_npz_path = os.path.join(data_dir, 'ligand_grids.npz')
    ligand_npz = np.load(ligand_npz_path)
    ligand_grids = ligand_npz['arr_0']
    
    with open(os.path.join(data_dir, 'ligand_metadata.json'), 'r') as f:
        ligand_metadata = json.load(f)
    
    # Load pocket grids and metadata
    pocket_npz_path = os.path.join(data_dir, 'pocket_grids.npz')
    pocket_npz = np.load(pocket_npz_path)
    pocket_grids = pocket_npz['arr_0']
    
    with open(os.path.join(data_dir, 'pocket_metadata.json'), 'r') as f:
        pocket_metadata = json.load(f)
    
    # Flatten grids for input to the model
    # ligand_grids shape: (num_samples, height, width, depth, channels)
    # pocket_grids shape: (num_samples, height, width, depth, channels)
    num_samples = min(len(ligand_grids), len(pocket_grids))
    
    ligand_features = ligand_grids[:num_samples].reshape(num_samples, -1)
    pocket_features = pocket_grids[:num_samples].reshape(num_samples, -1)
    
    # Get binding affinity labels from CSV
    # Load the binding affinity data from CSV
    import pandas as pd
    csv_path = os.path.join(os.path.dirname(data_dir), 'pdbbind_with_dG.csv')
    if not os.path.exists(csv_path):
        csv_path = 'pdbbind_with_dG.csv'  # Try current directory
    
    df_affinity = pd.read_csv(csv_path)
    
    # Create a mapping from PDB ID to binding affinity
    pdb_to_affinity = {}
    for _, row in df_affinity.iterrows():
        pdb_id = row['protein']
        dg = row['ΔG_kcal_per_mol']
        pdb_to_affinity[pdb_id] = dg
    
    # Extract labels using ligand_id from metadata
    labels = []
    for i in range(num_samples):
        ligand_id = ligand_metadata[i].get('ligand_id', None)
        if ligand_id in pdb_to_affinity:
            labels.append(pdb_to_affinity[ligand_id])
        else:
            labels.append(np.nan)  # Use NaN for missing values
    
    labels = np.array(labels, dtype=np.float32)
    
    # Remove samples with NaN labels
    valid_idx = ~np.isnan(labels)
    ligand_features = ligand_features[valid_idx]
    pocket_features = pocket_features[valid_idx]
    labels = labels[valid_idx]
    num_samples = len(labels)
    # Keep labels as 1D - FusionDataset will unsqueeze(1) to make it 2D
    
    print(f"Loaded {num_samples} samples")
    print(f"Ligand features shape: {ligand_features.shape}")
    print(f"Pocket features shape: {pocket_features.shape}")
    print(f"Labels shape (before unsqueeze): {labels.shape}")
    
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
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = f'plots_{timestamp}'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by RMSE (lowest is best)
    results_sorted = sorted(results, key=lambda x: x['rmse'])[:5]
    
    # Create dataframe with timestamp
    df = pd.DataFrame(results_sorted)
    csv_filename = os.path.join(output_dir, 'top5_random_unitary_results.csv')
    
    # Add metadata row with timestamp
    print(f"\n{'='*60}")
    print(f"Unitary Testing Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}/")
    print(f"Results saved to: {csv_filename}")
    
    df.to_csv(csv_filename, index=False)
    
    # Plot loss curves with timestamp
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for i, res in enumerate(results_sorted):
        ax1.plot(res['train_losses'], label=f'Train Circuit {res["circuit_idx"]}', linewidth=2)
        ax1.plot(res['val_losses'], label=f'Val Circuit {res["circuit_idx"]}', linestyle='--', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Loss Curves for Top 5 Random Unitary Circuits\nRun: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    loss_filename = os.path.join(output_dir, 'top5_loss_curves.png')
    fig1.savefig(loss_filename, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to: {loss_filename}")
    
    # Plot R² scores with timestamp
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    circuit_indices = [res['circuit_idx'] for res in results_sorted]
    r2_scores = [res['r2'] for res in results_sorted]
    rmse_scores = [res['rmse'] for res in results_sorted]
    
    bars = ax2.bar(range(len(circuit_indices)), r2_scores, color='steelblue', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Top Unitary Rank', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title(f'R² Scores for Top 5 Random Unitary Circuits\nRun: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=13)
    ax2.set_xticks(range(len(circuit_indices)))
    ax2.set_xticklabels([f'Circuit {idx}' for idx in circuit_indices])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, r2, rmse) in enumerate(zip(bars, r2_scores, rmse_scores)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'R²={r2:.3f}\nRMSE={rmse:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    r2_filename = os.path.join(output_dir, 'top5_r2_scores.png')
    fig2.savefig(r2_filename, dpi=300, bbox_inches='tight')
    print(f"R² plot saved to: {r2_filename}")
    
    # Print summary table
    print(f"\nTop 5 Unitaries Summary:")
    print(df[['circuit_idx', 'rmse', 'mae', 'r2', 'pearson', 'spearman']].to_string(index=False))
    print(f"{'='*60}\n")
    
    plt.show()

if __name__ == "__main__":
    n_qubits = 4  # Reduced for faster testing
    depth = 2     # Reduced depth for faster generation
    circuits = generate_random_circuits(n_qubits, depth, num_circuits=10)
    train_dataset, val_dataset = load_preprocessed_data()
    results = run_circuits_and_evaluate(circuits, train_dataset, val_dataset, n_qubits)
    save_and_plot_results(results)
