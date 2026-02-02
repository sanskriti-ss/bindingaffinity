import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from quantum_fusion.main_train import QuantumFusionModel, FusionDataset, evaluate_model

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
    # Load features and labels from sample_data/core_test.hdf and pdbbind_2016_train_val_test.csv
    import h5py
    # Paths to sample data
    hdf_path = '../../sample_data/core_test.hdf'
    csv_path = '../../sample_data/pdbbind_2016_train_val_test.csv'
    # Load features from HDF5
    with h5py.File(hdf_path, 'r') as hdf:
        sgcnn_features = np.array(hdf['sgcnn_features'])
        cnn3d_features = np.array(hdf['cnn3d_features'])
        complex_ids = np.array(hdf['complex_ids']).astype(str)
    # Load labels from CSV
    df = pd.read_csv(csv_path)
    id_to_label = dict(zip(df['complex_id'].astype(str), df['label']))
    # Filter to only those with labels
    valid_idx = [i for i, cid in enumerate(complex_ids) if cid in id_to_label]
    sgcnn_features = sgcnn_features[valid_idx]
    cnn3d_features = cnn3d_features[valid_idx]
    complex_ids = complex_ids[valid_idx]
    labels = np.array([id_to_label[cid] for cid in complex_ids])
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

def run_circuits_and_evaluate(circuits, train_dataset, val_dataset, n_qubits):
    results = []
    for idx, qc in enumerate(circuits):
        # Create a QuantumFusionModel with the random circuit
        model = QuantumFusionModel(sgcnn_dim=20, cnn3d_dim=20, n_qubits=n_qubits)
        model.qc = qc  # Replace circuit
        # You may need to update qnn and quantum_layer as well
        model.qnn = model._create_qnn()
        model.quantum_layer = TorchConnector(model.qnn)
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
                preds = model(sgcnn_feat, cnn3d_feat)
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
                    preds = model(sgcnn_feat, cnn3d_feat)
                    loss = criterion(preds, labels)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
        # Evaluate final performance
        rmse, mae, r2, pearson, spearman, labels, preds = evaluate_model(model, val_loader, 'cpu')
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
    n_qubits = 20
    depth = 4
    circuits = generate_random_circuits(n_qubits, depth)
    train_dataset, val_dataset = load_small_pdbbind_subset()
    results = run_circuits_and_evaluate(circuits, train_dataset, val_dataset, n_qubits)
    save_and_plot_results(results)
