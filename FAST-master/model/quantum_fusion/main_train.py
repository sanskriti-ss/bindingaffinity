# -*- coding: utf-8 -*-
"""quantum_fusion main training module

Refactored to use local sample data instead of Colab paths.
"""

from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import math
import h5py
import pennylane as qml
from ingenii_quantum.hybrid_networks.layers import QuantumFCLayer

# Import local sample data paths
try:
    from ..vars import SAMPLE_HDF_PATH, SAMPLE_CSV_PATH
except ImportError:
    # Fallback paths if vars.py not found
    SAMPLE_HDF_PATH = "../sample_data/core_test.hdf"
    SAMPLE_CSV_PATH = "../sample_data/pdbbind_2016_train_val_test.csv"

# Set output directory
output_dir = "../../results"
os.makedirs(output_dir, exist_ok=True)

# ---------------- Data Alignment Functions ----------------
class FusionDataset(Dataset):
    def __init__(self, sgcnn_features, cnn3d_features, labels):
        self.sgcnn_features = torch.FloatTensor(sgcnn_features)
        self.cnn3d_features = torch.FloatTensor(cnn3d_features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.sgcnn_features[idx], self.cnn3d_features[idx], self.labels[idx]


# ---------------- Data Loading Functions ----------------
def load_sample_data():
    """Load data from local sample files - simplified for testing"""
    print(f"Loading data from {SAMPLE_CSV_PATH} for labels")
    
    # Load labels from CSV first to get complex IDs
    df = pd.read_csv(SAMPLE_CSV_PATH)
    print(f"CSV columns: {df.columns.tolist()}")
    print(f"CSV shape: {df.shape}")
    
    # Get complex IDs and labels
    if 'pdbid' in df.columns and '-logKd/Ki' in df.columns:
        complex_ids = df['pdbid'].astype(str).values
        labels = pd.to_numeric(df['-logKd/Ki'], errors='coerce').values
        # Remove entries with NaN labels
        valid_mask = ~np.isnan(labels)
        complex_ids = complex_ids[valid_mask]
        labels = labels[valid_mask]
    elif len(df.columns) >= 2:
        complex_ids = df.iloc[:, 0].astype(str).values
        labels = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        # Remove entries with NaN labels
        valid_mask = ~np.isnan(labels)
        complex_ids = complex_ids[valid_mask]
        labels = labels[valid_mask]
    else:
        raise ValueError("Cannot find complex IDs and labels in CSV")
    
    # For now, create dummy features with consistent size for testing
    n_complexes = min(len(complex_ids), 200)  # Limit to 200 for speed
    feature_dim = 128  # Fixed feature dimension
    
    print(f"Creating dummy features for {n_complexes} complexes with {feature_dim} features each")
    
    ### PLEASE UPDATE SO Not using DUMMY FEATURES
    # Generate dummy features (in practice, these would come from molecular preprocessing)
    np.random.seed(42)  # For reproducible results
    sgcnn_features = np.random.randn(n_complexes, feature_dim).astype(np.float32)
    cnn3d_features = np.random.randn(n_complexes, feature_dim).astype(np.float32)
    
    # Take first n_complexes
    complex_ids = complex_ids[:n_complexes]
    labels = labels[:n_complexes]
    
    print(f"Final dataset: {len(labels)} samples with {feature_dim} features each")
    print(f"Label range: {labels.min():.2f} to {labels.max():.2f}")
    
    return sgcnn_features, cnn3d_features, labels, complex_ids


def evaluate_model(model, loader):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for sg, c3, y in loader:
            combined = torch.cat([sg, c3], dim=1)
            out = model(combined)
            preds.extend(out.cpu().numpy().flatten())
            labs.extend(y.cpu().numpy().flatten())
    preds, labs = np.array(preds), np.array(labs)
    return (math.sqrt(mean_squared_error(labs, preds)),
            mean_absolute_error(labs, preds),
            r2_score(labs, preds),
            pearsonr(labs, preds)[0],
            spearmanr(labs, preds)[0])

# ---------------- Model Definition ----------------
import math
import torch
import torch.nn as nn
from ingenii_quantum.hybrid_networks.layers import QuantumFCLayer

class ModelHybridFC(nn.Module):
    def __init__(self,
                 in_features:   int,
                 out_features:  int,
                 qc_input_size: int = 4,         # number of qubits
                 qc_n_layers:   int = 10,        # depth L = 10
                 qc_encoding:   str = 'amplitude',
                 qc_ansatz:     int = 1,
                 backend:       str = 'default.qubit'
                ):
        super().__init__()
        # 1) Classical compressor → qc_input_size dims
        self.fc1 = nn.Linear(in_features, 2 * qc_input_size)
        self.fc2 = nn.Linear(2 * qc_input_size, qc_input_size)

        # 2) Quantum layer: measures Z on each of qc_input_size wires
        qnn = QuantumFCLayer(
            input_size=qc_input_size,
            n_layers=qc_n_layers,
            encoding=qc_encoding,
            ansatz=qc_ansatz,
            observables=["ZI", "IZ"],      # one Z per qubit by default
            backend=backend
        )
        self.quantum_layer = qnn.create_layer(type_layer='torch')

        # 3) Final head: we’ll sum the per‑qubit outputs to shape [batch,1]
        self.fc_out = nn.Linear(1, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # → Classical preprocessing
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x)) * math.pi

        # → Quantum measurements: out shape [batch, qc_input_size]
        q_out = self.quantum_layer(x)

        # → Sum over qubits → [batch,1]
        q_sum = q_out.sum(dim=1, keepdim=True)

        # → Final regression/classification head
        return self.fc_out(q_sum)


# ============================================================================
# Option 1: Quantum Reservoir Model (Fixed Circuit - matches Domingo et al.)
# ============================================================================
class ModelHybridFC_Reservoir(nn.Module):
    """
    Quantum Reservoir Computing model following Domingo et al. (2022).
    
    The G3 circuit (CNOT, H, T) is FIXED and non-trainable.
    Only the classical layers are trained.
    The quantum circuit acts as a feature extractor/reservoir.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 qiskit_circuit,  # Qiskit QuantumCircuit from G3 family
                 n_qubits: int = 4,
                 backend: str = 'default.qubit'
                ):
        super().__init__()
        self.n_qubits = n_qubits
        
        # 1) Classical compressor → n_qubits dims for encoding
        self.fc1 = nn.Linear(in_features, 2 * n_qubits)
        self.fc2 = nn.Linear(2 * n_qubits, n_qubits)
        
        # 2) Convert Qiskit circuit to PennyLane and create fixed quantum layer
        self.dev = qml.device(backend, wires=n_qubits)
        self.qiskit_circuit = qiskit_circuit
        
        # Create the quantum node (fixed, no trainable params)
        @qml.qnode(self.dev, interface='torch')
        def quantum_reservoir(inputs):
            # Encode classical data via angle encoding
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Apply the fixed G3 circuit
            # Convert Qiskit gates to PennyLane
            for instruction in qiskit_circuit.data:
                gate = instruction.operation
                qubits = [q._index for q in instruction.qubits]
                
                if gate.name == 'h':
                    qml.Hadamard(wires=qubits[0])
                elif gate.name == 't':
                    qml.T(wires=qubits[0])
                elif gate.name == 'cx':
                    qml.CNOT(wires=qubits)
            
            # Measure all qubits in Z basis
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.quantum_reservoir = quantum_reservoir
        
        # 3) Classical head for final prediction
        self.fc_out = nn.Linear(n_qubits, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Classical preprocessing
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x)) * math.pi  # Scale to [-π, π]
        
        # Apply quantum reservoir to each sample (fixed circuit, no gradients through it)
        q_outputs = []
        for i in range(batch_size):
            q_out = self.quantum_reservoir(x[i])
            q_outputs.append(torch.stack(q_out))
        
        q_out = torch.stack(q_outputs)  # [batch, n_qubits]
        
        # Final regression head
        return self.fc_out(q_out)


# ============================================================================
# Option 2: Variational Quantum Circuit Model (Trainable Parameters)
# ============================================================================
class ModelHybridFC_VQC(nn.Module):
    """
    Variational Quantum Circuit model with trainable parameters.
    
    Uses G3 circuit STRUCTURE but replaces fixed gates with trainable rotations:
    - H gate positions → RY(θ) rotations (trainable)
    - T gate positions → RZ(φ) rotations (trainable)
    - CNOT positions → CNOT gates (fixed, but preceded by trainable rotations)
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 qiskit_circuit,  # Qiskit QuantumCircuit to extract structure from
                 n_qubits: int = 4,
                 backend: str = 'default.qubit'
                ):
        super().__init__()
        self.n_qubits = n_qubits
        
        # 1) Classical compressor
        self.fc1 = nn.Linear(in_features, 2 * n_qubits)
        self.fc2 = nn.Linear(2 * n_qubits, n_qubits)
        
        # 2) Extract circuit structure and count trainable parameters
        self.gate_structure = []
        n_params = 0
        for instruction in qiskit_circuit.data:
            gate = instruction.operation
            qubits = [q._index for q in instruction.qubits]
            
            if gate.name == 'h':
                self.gate_structure.append(('ry', qubits[0], n_params))
                n_params += 1
            elif gate.name == 't':
                self.gate_structure.append(('rz', qubits[0], n_params))
                n_params += 1
            elif gate.name == 'cx':
                self.gate_structure.append(('cnot', qubits, None))
        
        # 3) Trainable quantum parameters
        self.quantum_params = nn.Parameter(torch.randn(n_params) * 0.1)
        
        # 4) Create quantum device and circuit
        self.dev = qml.device(backend, wires=n_qubits)
        
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def variational_circuit(inputs, params, gate_structure, n_qubits):
            # Encode classical data
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Apply variational gates following G3 structure
            for gate_type, qubit_info, param_idx in gate_structure:
                if gate_type == 'ry':
                    qml.RY(params[param_idx], wires=qubit_info)
                elif gate_type == 'rz':
                    qml.RZ(params[param_idx], wires=qubit_info)
                elif gate_type == 'cnot':
                    qml.CNOT(wires=qubit_info)
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.variational_circuit = variational_circuit
        
        # 5) Classical head
        self.fc_out = nn.Linear(n_qubits, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Classical preprocessing
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x)) * math.pi
        
        # Apply variational quantum circuit
        q_outputs = []
        for i in range(batch_size):
            q_out = self.variational_circuit(
                x[i], 
                self.quantum_params, 
                self.gate_structure, 
                self.n_qubits
            )
            q_outputs.append(torch.stack(q_out))
        
        q_out = torch.stack(q_outputs)
        
        return self.fc_out(q_out)


# ------------ Hyperparameters ------------------------

epochs = 100
batch_size = 32
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- Load & Prepare Data ----------------
# Load sample data
sgcnn_features, cnn3d_features, labels, complex_ids = load_sample_data()

# Split into train/val/test sets
n_samples = len(labels)
n_train = min(100, int(0.7 * n_samples))
n_val = min(20, int(0.15 * n_samples))
n_test = min(20, n_samples - n_train - n_val)

print(f"Splitting data: {n_train} train, {n_val} val, {n_test} test")

# Create indices for splits
train_idx = np.arange(0, n_train)
val_idx = np.arange(n_train, n_train + n_val)
test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)

# Create datasets dictionary
datasets = {
    'train': {
        'sg': sgcnn_features[train_idx],
        'c3': cnn3d_features[train_idx],
        'y': labels[train_idx],
        'ids': complex_ids[train_idx]
    },
    'val': {
        'sg': sgcnn_features[val_idx],
        'c3': cnn3d_features[val_idx],
        'y': labels[val_idx],
        'ids': complex_ids[val_idx]
    },
    'test': {
        'sg': sgcnn_features[test_idx],
        'c3': cnn3d_features[test_idx],
        'y': labels[test_idx],
        'ids': complex_ids[test_idx]
    }
}

for split in ['train', 'val', 'test']:
    print(f"Loaded {split}: {len(datasets[split]['y'])} samples")

# DataLoaders
loaders = {}
dims = datasets['train']['sg'].shape[1] + datasets['train']['c3'].shape[1]
for split in ['train','val','test']:
    ds = FusionDataset(
        datasets[split]['sg'], datasets[split]['c3'],
        datasets[split]['y']
    )
    shuffle = (split=='train')
    loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# ---------------- Instantiate & Train ----------------
# Number of qubits = ceil(log2(16)) = 4
qc_input_size = 4
# Use Circuit 1
qc_ansatz     = 1
# Use amplitude encoding
qc_encoding   = 'amplitude'
# Depth L = 10
qc_n_layers   = 10

model = ModelHybridFC(
    in_features=dims,
    out_features=1,
    qc_input_size=qc_input_size,
    qc_n_layers=qc_n_layers,
    qc_encoding=qc_encoding,
    qc_ansatz=qc_ansatz,
    backend='default.qubit',             # or another PennyLane device
)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
criterion = nn.MSELoss()

# Training
best_val = float('inf')
for epoch in range(1, epochs+1):
    model.train()
    train_loss = 0.0
    for sg, c3, y in tqdm(loaders['train']):
        x = torch.cat([sg, c3], dim=1).to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(loaders['train'])

    rmse, mae, r2, pearson, spearman = evaluate_model(model, loaders['val'])
    scheduler.step(rmse)

    print(f"Epoch {epoch}: Train RMSE={math.sqrt(train_loss):.4f}, Val RMSE={rmse:.4f}")
    if rmse < best_val:
        best_val = rmse
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

# Final evaluation on test set
if os.path.exists(os.path.join(output_dir, 'best_model.pth')):
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    print("Test metrics:", evaluate_model(model, loaders['test']))
else:
    print("No saved model found, skipping test evaluation")
    
print("Training completed!")