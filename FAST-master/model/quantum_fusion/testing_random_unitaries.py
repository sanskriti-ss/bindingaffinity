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
    from .main_train import (ModelHybridFC, FusionDataset, evaluate_model,
                              ModelHybridFC_Reservoir, ModelHybridFC_VQC)
except ImportError:
    # If running as script directly from quantum_fusion directory
    from main_train import (ModelHybridFC, FusionDataset, evaluate_model,
                            ModelHybridFC_Reservoir, ModelHybridFC_VQC)

# 1. Generate random circuits from G3 gate family {CNOT, H, T}
def generate_g3_random_circuits(n_qubits, depth, num_circuits=10):
    """
    Generate random unitary circuits sampled from the G3 gate family.
    G3 gates: {CNOT, H, T} with uniform random selection (1/3 each in expectation).
    
    Follows the methodology from Domingo et al. (2022): "Optimal quantum reservoir 
    computing for the NISQ era" - circuits are constructed by adding random quantum 
    gates from the G3 family with uniform probability distribution.
    
    Args:
        n_qubits: Number of qubits
        depth: Circuit depth (number of layers)
        num_circuits: Number of circuits to generate
    
    Returns:
        List of QuantumCircuit objects
    """
    from qiskit.circuit import QuantumCircuit, ParameterVector
    import random
    
    circuits = []
    
    for _ in range(num_circuits):
        qc = QuantumCircuit(n_qubits)
        
        # Add random gates: each gate (H, T, CNOT) chosen uniformly with 1/3 probability each
        for layer in range(depth):
            # Generate random gates for this layer
            # Total gates per layer: aim for ~n_qubits gates (mix of single and two-qubit)
            for _ in range(n_qubits):
                # Choose gate type uniformly from {H, T, CNOT}
                gate_type = random.choices(['h', 't', 'cnot'], weights=[1, 1, 1], k=1)[0]
                
                if gate_type == 'h':
                    # Apply H to random qubit
                    qubit = random.randint(0, n_qubits - 1)
                    qc.h(qubit)
                    
                elif gate_type == 't':
                    # Apply T to random qubit
                    qubit = random.randint(0, n_qubits - 1)
                    qc.t(qubit)
                    
                else:  # cnot
                    # Apply CNOT between two random qubits (control != target)
                    control = random.randint(0, n_qubits - 1)
                    target = random.randint(0, n_qubits - 1)
                    # Ensure control and target are different
                    while target == control:
                        target = random.randint(0, n_qubits - 1)
                    qc.cx(control, target)
        
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

def run_circuits_and_evaluate(circuits, train_dataset, val_dataset, n_qubits=4, 
                               model_type='reservoir', num_epochs=5):
    """
    Test circuits with specified model type.
    
    Args:
        circuits: List of Qiskit QuantumCircuit objects (G3 family)
        train_dataset: Training FusionDataset
        val_dataset: Validation FusionDataset
        n_qubits: Number of qubits
        model_type: 'reservoir' (fixed circuit) or 'vqc' (trainable parameters)
        num_epochs: Number of training epochs
    
    Returns:
        List of result dictionaries
    """
    results = []
    sgcnn_dim = train_dataset.sgcnn_features.shape[1]
    cnn3d_dim = train_dataset.cnn3d_features.shape[1]
    total_dim = sgcnn_dim + cnn3d_dim
    
    model_desc = "Reservoir (fixed)" if model_type == 'reservoir' else "VQC (trainable)"
    
    # Progress bar for circuits
    circuit_progress = tqdm(enumerate(circuits), total=len(circuits), 
                           desc=f"Testing {model_desc}", position=0)
    
    for idx, qc in circuit_progress:
        circuit_progress.set_description(f"Testing Unitary {idx + 1}/{len(circuits)} [{model_desc}]")
        
        # Create model based on type
        if model_type == 'reservoir':
            # Option 1: Quantum Reservoir (fixed circuit - Domingo et al.)
            model = ModelHybridFC_Reservoir(
                in_features=total_dim,
                out_features=1,
                qiskit_circuit=qc,
                n_qubits=n_qubits,
                backend='default.qubit'
            )
        elif model_type == 'vqc':
            # Option 2: Variational Quantum Circuit (trainable params)
            model = ModelHybridFC_VQC(
                in_features=total_dim,
                out_features=1,
                qiskit_circuit=qc,
                n_qubits=n_qubits,
                backend='default.qubit'
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'reservoir' or 'vqc'")
        
        # Train for specified epochs
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        train_losses, val_losses = [], []
        
        # Progress bar for epochs
        epoch_progress = tqdm(range(num_epochs), desc=f"  Training Unitary {idx + 1}", position=1, leave=False)
        
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
            'model_type': model_type,
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

def save_and_plot_results(results_reservoir, results_vqc, n_qubits, depth):
    """
    Save and plot comparison results for both model types.
    
    Args:
        results_reservoir: Results from Quantum Reservoir model (fixed circuits)
        results_vqc: Results from VQC model (trainable parameters)
        n_qubits: Number of qubits used
        depth: Circuit depth used
    """
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = f'plots_{timestamp}'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Quantum Circuit Testing Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"Configuration: {n_qubits} qubits, depth {depth}, G3 gate family {{CNOT, H, T}}")
    print(f"Output directory: {output_dir}/")
    
    # Sort by RMSE and get top 5 for each
    reservoir_sorted = sorted(results_reservoir, key=lambda x: x['rmse'])[:5]
    vqc_sorted = sorted(results_vqc, key=lambda x: x['rmse'])[:5]
    
    # Save combined CSV
    df_reservoir = pd.DataFrame(reservoir_sorted)
    df_reservoir['model_type'] = 'reservoir'
    df_vqc = pd.DataFrame(vqc_sorted)
    df_vqc['model_type'] = 'vqc'
    df_combined = pd.concat([df_reservoir, df_vqc], ignore_index=True)
    
    csv_filename = os.path.join(output_dir, 'comparison_results.csv')
    df_combined.to_csv(csv_filename, index=False)
    print(f"Results saved to: {csv_filename}")
    
    # ============ Plot 1: Loss curves comparison ============
    fig1, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Reservoir loss curves
    for i, res in enumerate(reservoir_sorted):
        axes[0].plot(res['train_losses'], label=f'Train C{res["circuit_idx"]}', linewidth=2)
        axes[0].plot(res['val_losses'], label=f'Val C{res["circuit_idx"]}', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Quantum Reservoir (Fixed Circuit)\nDomingo et al. approach', fontsize=12)
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # VQC loss curves
    for i, res in enumerate(vqc_sorted):
        axes[1].plot(res['train_losses'], label=f'Train C{res["circuit_idx"]}', linewidth=2)
        axes[1].plot(res['val_losses'], label=f'Val C{res["circuit_idx"]}', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Variational Quantum Circuit (Trainable)\nParameterized rotations', fontsize=12)
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    fig1.suptitle(f'Loss Curves: Top 5 Circuits per Approach\n{timestamp}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    loss_filename = os.path.join(output_dir, 'loss_curves_comparison.png')
    fig1.savefig(loss_filename, dpi=300, bbox_inches='tight')
    print(f"Loss comparison saved to: {loss_filename}")
    
    # ============ Plot 2: R² comparison bar chart ============
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    x = np.arange(5)
    width = 0.35
    
    r2_reservoir = [res['r2'] for res in reservoir_sorted]
    r2_vqc = [res['r2'] for res in vqc_sorted]
    
    bars1 = ax2.bar(x - width/2, r2_reservoir, width, label='Reservoir (Fixed)', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, r2_vqc, width, label='VQC (Trainable)', color='coral', alpha=0.8)
    
    ax2.set_xlabel('Top Circuit Rank', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title(f'R² Score Comparison: Quantum Reservoir vs VQC\n{timestamp}', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'#{i+1}' for i in range(5)])
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    r2_filename = os.path.join(output_dir, 'r2_comparison.png')
    fig2.savefig(r2_filename, dpi=300, bbox_inches='tight')
    print(f"R² comparison saved to: {r2_filename}")
    
    # ============ Plot 3: RMSE comparison ============
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    rmse_reservoir = [res['rmse'] for res in reservoir_sorted]
    rmse_vqc = [res['rmse'] for res in vqc_sorted]
    
    bars1 = ax3.bar(x - width/2, rmse_reservoir, width, label='Reservoir (Fixed)', color='steelblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2, rmse_vqc, width, label='VQC (Trainable)', color='coral', alpha=0.8)
    
    ax3.set_xlabel('Top Circuit Rank', fontsize=12)
    ax3.set_ylabel('RMSE', fontsize=12)
    ax3.set_title(f'RMSE Comparison: Quantum Reservoir vs VQC\n{timestamp}', fontsize=13)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'#{i+1}' for i in range(5)])
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    rmse_filename = os.path.join(output_dir, 'rmse_comparison.png')
    fig3.savefig(rmse_filename, dpi=300, bbox_inches='tight')
    print(f"RMSE comparison saved to: {rmse_filename}")
    
    # Print summary tables
    print(f"\n{'='*70}")
    print("QUANTUM RESERVOIR (Fixed Circuit - Domingo et al.)")
    print("="*70)
    print(df_reservoir[['circuit_idx', 'rmse', 'mae', 'r2', 'pearson', 'spearman']].to_string(index=False))
    
    print(f"\n{'='*70}")
    print("VARIATIONAL QUANTUM CIRCUIT (Trainable Parameters)")
    print("="*70)
    print(df_vqc[['circuit_idx', 'rmse', 'mae', 'r2', 'pearson', 'spearman']].to_string(index=False))
    
    # Summary comparison
    best_reservoir = reservoir_sorted[0]
    best_vqc = vqc_sorted[0]
    
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"Best Reservoir (Circuit {best_reservoir['circuit_idx']}): RMSE={best_reservoir['rmse']:.4f}, R²={best_reservoir['r2']:.4f}")
    print(f"Best VQC (Circuit {best_vqc['circuit_idx']}): RMSE={best_vqc['rmse']:.4f}, R²={best_vqc['r2']:.4f}")
    
    if best_reservoir['rmse'] < best_vqc['rmse']:
        print(f"\n→ Reservoir approach achieved lower RMSE by {best_vqc['rmse'] - best_reservoir['rmse']:.4f}")
    else:
        print(f"\n→ VQC approach achieved lower RMSE by {best_reservoir['rmse'] - best_vqc['rmse']:.4f}")
    
    print(f"{'='*70}\n")
    
    plt.show()
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test G3 quantum circuits with both model approaches')
    parser.add_argument('--n_qubits', type=int, default=4, help='Number of qubits')
    parser.add_argument('--depth', type=int, default=3, help='Circuit depth (layers)')
    parser.add_argument('--num_circuits', type=int, default=10, help='Number of circuits to test')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs per circuit')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("G3 RANDOM QUANTUM CIRCUIT TESTING")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  - Qubits: {args.n_qubits}")
    print(f"  - Circuit depth: {args.depth}")
    print(f"  - Circuits to test: {args.num_circuits}")
    print(f"  - Epochs per circuit: {args.epochs}")
    print(f"  - Gate family: G3 = {{CNOT, H, T}} with uniform 1/3 probability")
    print(f"{'='*70}\n")
    
    # Generate G3 circuits
    print("Generating G3 random circuits...")
    circuits = generate_g3_random_circuits(args.n_qubits, args.depth, num_circuits=args.num_circuits)
    print(f"Generated {len(circuits)} circuits\n")
    
    # Load data
    print("Loading preprocessed data...")
    train_dataset, val_dataset = load_preprocessed_data()
    
    # Test with Quantum Reservoir (Option 1 - matches Domingo et al.)
    print(f"\n{'='*70}")
    print("OPTION 1: QUANTUM RESERVOIR (Fixed Circuit)")
    print("Following Domingo et al. 'Optimal quantum reservoir computing'")
    print(f"{'='*70}")
    results_reservoir = run_circuits_and_evaluate(
        circuits, train_dataset, val_dataset, 
        n_qubits=args.n_qubits, 
        model_type='reservoir',
        num_epochs=args.epochs
    )
    
    # Test with VQC (Option 2 - trainable parameters)
    print(f"\n{'='*70}")
    print("OPTION 2: VARIATIONAL QUANTUM CIRCUIT (Trainable)")
    print("G3 structure with parameterized rotations")
    print(f"{'='*70}")
    results_vqc = run_circuits_and_evaluate(
        circuits, train_dataset, val_dataset,
        n_qubits=args.n_qubits,
        model_type='vqc',
        num_epochs=args.epochs
    )
    
    # Save and compare results
    output_dir = save_and_plot_results(results_reservoir, results_vqc, args.n_qubits, args.depth)
    print(f"All outputs saved to: {output_dir}/")
