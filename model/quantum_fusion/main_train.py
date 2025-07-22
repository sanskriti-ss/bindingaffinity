import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import math

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import Estimator


class FusionDataset(Dataset):
    """Dataset for fusion model training"""
    
    def __init__(self, sgcnn_features, cnn3d_features, labels):
        self.sgcnn_features = torch.FloatTensor(sgcnn_features)
        self.cnn3d_features = torch.FloatTensor(cnn3d_features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sgcnn_features[idx], self.cnn3d_features[idx], self.labels[idx]


class QuantumFusionModel(nn.Module):
    """Quantum fusion model using Qiskit"""
    
    def __init__(self, sgcnn_dim, cnn3d_dim, n_qubits=20):
        super(QuantumFusionModel, self).__init__()
        
        self.n_qubits = n_qubits
        
        # Classical preprocessing layers to reduce dimensions to quantum circuit input
        self.sgcnn_prep = nn.Linear(sgcnn_dim, n_qubits)
        self.cnn3d_prep = nn.Linear(cnn3d_dim, n_qubits)
        
        # Create quantum circuit
        self.qc = self._create_quantum_circuit()
        
        # Create quantum neural network
        self.qnn = self._create_qnn()
        
        # Create TorchConnector
        self.quantum_layer = TorchConnector(self.qnn)
        
        # Classical post-processing
        self.classical_post = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
    def _create_quantum_circuit(self):
        """Create the quantum circuit with parameterized gates"""
        qc = QuantumCircuit(self.n_qubits)
        
        # ParameterVectors for the U-gates (3 parameters per qubit)
        thetas = ParameterVector('θ', self.n_qubits)
        phis = ParameterVector('ϕ', self.n_qubits)
        lambdas = ParameterVector('λ', self.n_qubits)
        
        # Store parameter vectors for later use
        self.thetas = thetas
        self.phis = phis
        self.lambdas = lambdas
        
        # 1) Apply U(θ,ϕ,λ) on each qubit
        for i in range(self.n_qubits):
            qc.u(thetas[i], phis[i], lambdas[i], i)
        
        # 2) Entangle in a ring: CNOT from i → (i+1)%n
        for i in range(self.n_qubits):
            qc.cx(i, (i + 1) % self.n_qubits)
        
        # 3) Second layer of parameterized gates
        thetas2 = ParameterVector('θ2', self.n_qubits)
        phis2 = ParameterVector('ϕ2', self.n_qubits)
        lambdas2 = ParameterVector('λ2', self.n_qubits)
        
        self.thetas2 = thetas2
        self.phis2 = phis2
        self.lambdas2 = lambdas2
        
        for i in range(self.n_qubits):
            qc.u(thetas2[i], phis2[i], lambdas2[i], i)
        
        # 4) Another entangling layer
        for i in range(self.n_qubits):
            qc.cx(i, (i + 1) % self.n_qubits)
        
        return qc
    
    def _create_qnn(self):
        """Create quantum neural network with observable"""
        # Define observable (expectation value of Z on first qubit)
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1.0)])
        
        # Create EstimatorQNN
        estimator = Estimator()
        qnn = EstimatorQNN(
            circuit=self.qc,
            observables=observable,
            input_params=list(self.thetas) + list(self.phis) + list(self.lambdas) +
                        list(self.thetas2) + list(self.phis2) + list(self.lambdas2),
            weight_params=[],  # No trainable weights in circuit, all handled by input
            estimator=estimator
        )
        
        return qnn
    
    def forward(self, sgcnn_feat, cnn3d_feat):
        # Classical preprocessing
        sgcnn_processed = torch.tanh(self.sgcnn_prep(sgcnn_feat))  # Normalize to [-1, 1]
        cnn3d_processed = torch.tanh(self.cnn3d_prep(cnn3d_feat))  # Normalize to [-1, 1]
        
        # Combine features for quantum circuit input
        # Use SGCNN for theta, 3DCNN for phi, and their combination for lambda
        quantum_input = torch.cat([
            sgcnn_processed * np.pi,  # thetas
            cnn3d_processed * np.pi,  # phis
            (sgcnn_processed + cnn3d_processed) * np.pi / 2,  # lambdas
            sgcnn_processed * cnn3d_processed * np.pi,  # thetas2
            (sgcnn_processed - cnn3d_processed) * np.pi / 2,  # phis2
            torch.sin(sgcnn_processed + cnn3d_processed) * np.pi  # lambdas2
        ], dim=1)
        
        # Quantum processing
        quantum_output = self.quantum_layer(quantum_input)
        
        # Classical post-processing
        output = self.classical_post(quantum_output)
        
        return output


class HybridQuantumFusionModel(nn.Module):
    """Hybrid classical-quantum fusion model"""
    
    def __init__(self, sgcnn_dim, cnn3d_dim, n_qubits=10):
        super(HybridQuantumFusionModel, self).__init__()
        
        # Classical branch
        self.classical_sgcnn = nn.Linear(sgcnn_dim, 16)
        self.classical_3dcnn = nn.Linear(cnn3d_dim, 16)
        self.classical_fusion = nn.Linear(32, 8)
        
        # Quantum branch
        self.quantum_branch = QuantumFusionModel(sgcnn_dim, cnn3d_dim, n_qubits)
        
        # Final combination
        self.final_layer = nn.Linear(9, 1)  # 8 from classical + 1 from quantum
        
    def forward(self, sgcnn_feat, cnn3d_feat):
        # Classical processing
        classical_sgcnn = torch.relu(self.classical_sgcnn(sgcnn_feat))
        classical_3dcnn = torch.relu(self.classical_3dcnn(cnn3d_feat))
        classical_concat = torch.cat([classical_sgcnn, classical_3dcnn], dim=1)
        classical_output = torch.relu(self.classical_fusion(classical_concat))
        
        # Quantum processing
        quantum_output = self.quantum_branch(sgcnn_feat, cnn3d_feat)
        
        # Combine classical and quantum
        combined = torch.cat([classical_output, quantum_output], dim=1)
        final_output = self.final_layer(combined)
        
        return final_output


def align_features(sgcnn_feat_path, sgcnn_ids_path, cnn3d_feat_path, cnn3d_ids_path):
    """Align SGCNN and 3DCNN features using complex IDs"""
    
    # Load features and IDs
    sgcnn_features = np.load(sgcnn_feat_path)
    sgcnn_ids = np.load(sgcnn_ids_path)
    cnn3d_features = np.load(cnn3d_feat_path)
    cnn3d_ids = np.load(cnn3d_ids_path)
    
    print(f"SGCNN: {len(sgcnn_features)} samples, {sgcnn_features.shape[1]} features")
    print(f"3DCNN: {len(cnn3d_features)} samples, {cnn3d_features.shape[1]} features")
    
    # Create mapping from IDs to indices
    sgcnn_id_to_idx = {str(id_val): idx for idx, id_val in enumerate(sgcnn_ids)}
    cnn3d_id_to_idx = {str(id_val): idx for idx, id_val in enumerate(cnn3d_ids)}
    
    # Find common IDs
    common_ids = set(sgcnn_id_to_idx.keys()) & set(cnn3d_id_to_idx.keys())
    print(f"Common complexes: {len(common_ids)}")
    
    if len(common_ids) == 0:
        raise ValueError("No common complex IDs found between SGCNN and 3DCNN datasets!")
    
    # Align features
    sgcnn_aligned = []
    cnn3d_aligned = []
    aligned_ids = []
    
    for common_id in common_ids:
        sgcnn_idx = sgcnn_id_to_idx[common_id]
        cnn3d_idx = cnn3d_id_to_idx[common_id]
        sgcnn_aligned.append(sgcnn_features[sgcnn_idx])
        cnn3d_aligned.append(cnn3d_features[cnn3d_idx])
        aligned_ids.append(common_id)
    
    return np.array(sgcnn_aligned), np.array(cnn3d_aligned), aligned_ids


def load_labels_from_csv(csv_path, complex_ids):
    """Load labels from CSV file matching the complex IDs"""
    df = pd.read_csv(csv_path)
    
    # Create mapping from complex_id to label
    id_to_label = {}
    for _, row in df.iterrows():
        complex_id = str(row['complex_id'])
        label = float(row['label'])
        id_to_label[complex_id] = label
    
    # Get labels for our aligned complex IDs
    labels = []
    for complex_id in complex_ids:
        if complex_id in id_to_label:
            labels.append(id_to_label[complex_id])
        else:
            raise ValueError(f"Complex ID {complex_id} not found in CSV file")
    
    return np.array(labels)


def evaluate_model(model, dataloader, device):
    """Evaluate model performance"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sgcnn_feat, cnn3d_feat, labels in dataloader:
            sgcnn_feat = sgcnn_feat.to(device)
            cnn3d_feat = cnn3d_feat.to(device)
            labels = labels.to(device)
            
            preds = model(sgcnn_feat, cnn3d_feat)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    rmse = math.sqrt(mean_squared_error(all_labels, all_preds))
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    pearson, _ = pearsonr(all_labels, all_preds)
    spearman, _ = spearmanr(all_labels, all_preds)
    
    return rmse, mae, r2, pearson, spearman, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgcnn-results-dir", default="/home/karen/Projects/FAST/results/sgcnn", help="SGCNN results directory")
    parser.add_argument("--cnn3d-results-dir", default="/home/karen/Projects/FAST/models", help="3DCNN results directory")
    parser.add_argument("--output-dir", default="/home/karen/Projects/FAST/results/quantum_fusion", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (smaller for quantum)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--n-qubits", type=int, default=10, help="Number of qubits")
    parser.add_argument("--model-type", choices=["quantum", "hybrid"], default="hybrid", help="Model type")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # File paths
    datasets = ['train', 'val', 'test']
    sgcnn_prefix = "refined"
    cnn3d_prefix = "3dcnn_dropout_02_long_22f_best_val_refined_3d"
    
    # Load and align all datasets
    aligned_data = {}
    
    for dataset in datasets:
        print(f"\nLoading {dataset} dataset...")
        
        sgcnn_feat_path = os.path.join(args.sgcnn_results_dir, f"{sgcnn_prefix}_{dataset}_feat.npy")
        sgcnn_ids_path = os.path.join(args.sgcnn_results_dir, f"{sgcnn_prefix}_{dataset}_complex_ids.npy")
        cnn3d_feat_path = os.path.join(args.cnn3d_results_dir, f"{cnn3d_prefix}_{dataset}_feat.npy")
        cnn3d_ids_path = os.path.join(args.cnn3d_results_dir, f"{cnn3d_prefix}_{dataset}_complex_ids.npy")
        sgcnn_csv_path = os.path.join(args.sgcnn_results_dir, f"{sgcnn_prefix}_{dataset}_pred.csv")
        
        # Align features
        sgcnn_feat, cnn3d_feat, common_ids = align_features(
            sgcnn_feat_path, sgcnn_ids_path, cnn3d_feat_path, cnn3d_ids_path
        )
        
        # Load labels
        labels = load_labels_from_csv(sgcnn_csv_path, common_ids)
        
        aligned_data[dataset] = {
            'sgcnn_feat': sgcnn_feat,
            'cnn3d_feat': cnn3d_feat,
            'labels': labels,
            'ids': common_ids
        }
        
        print(f"{dataset}: {len(labels)} aligned samples")
    
    # Create datasets and dataloaders
    train_dataset = FusionDataset(
        aligned_data['train']['sgcnn_feat'],
        aligned_data['train']['cnn3d_feat'],
        aligned_data['train']['labels']
    )
    
    val_dataset = FusionDataset(
        aligned_data['val']['sgcnn_feat'],
        aligned_data['val']['cnn3d_feat'],
        aligned_data['val']['labels']
    )
    
    test_dataset = FusionDataset(
        aligned_data['test']['sgcnn_feat'],
        aligned_data['test']['cnn3d_feat'],
        aligned_data['test']['labels']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    sgcnn_dim = aligned_data['train']['sgcnn_feat'].shape[1]
    cnn3d_dim = aligned_data['train']['cnn3d_feat'].shape[1]
    
    if args.model_type == "quantum":
        model = QuantumFusionModel(sgcnn_dim, cnn3d_dim, args.n_qubits)
    else:  # hybrid
        model = HybridQuantumFusionModel(sgcnn_dim, cnn3d_dim, args.n_qubits)
    
    model.to(device)
    
    print(f"\nModel architecture: {args.model_type}")
    print(f"SGCNN features: {sgcnn_dim}")
    print(f"3DCNN features: {cnn3d_dim}")
    print(f"Qubits: {args.n_qubits}")
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience = 25
    patience_counter = 0
    
    print(f"\nStarting training for up to {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for sgcnn_feat, cnn3d_feat, labels in train_loader:
            sgcnn_feat = sgcnn_feat.to(device)
            cnn3d_feat = cnn3d_feat.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            preds = model(sgcnn_feat, cnn3d_feat)
            loss = criterion(preds, labels)
            loss.backward()
            
            # Gradient clipping for quantum models
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sgcnn_feat, cnn3d_feat, labels in val_loader:
                sgcnn_feat = sgcnn_feat.to(device)
                cnn3d_feat = cnn3d_feat.to(device)
                labels = labels.to(device)
                
                preds = model(sgcnn_feat, cnn3d_feat)
                loss = criterion(preds, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Step scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, os.path.join(args.output_dir, 'best_quantum_fusion_model.pth'))
            print(f"*** Best model saved with validation loss: {val_loss:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_quantum_fusion_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on all datasets
    print(f"\nFinal evaluation:")
    
    for dataset, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        rmse, mae, r2, pearson, spearman, labels, preds = evaluate_model(model, loader, device)
        print(f"{dataset.capitalize()}: RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}, Pearson: {pearson:.3f}, Spearman: {spearman:.3f}")
        
        # Save predictions
        pred_df = pd.DataFrame({
            'cid': range(len(labels)),
            'complex_id': aligned_data[dataset]['ids'],
            'label': labels,
            'pred': preds
        })
        pred_df.to_csv(os.path.join(args.output_dir, f'quantum_fusion_{dataset}_pred.csv'), index=False)
    
    print(f"\nQuantum training completed! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()