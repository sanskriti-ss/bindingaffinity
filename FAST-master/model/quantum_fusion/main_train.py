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
from datetime import datetime
from ingenii_quantum.hybrid_networks.layers import QuantumFCLayer

# Import local sample data paths
try:
    from ..vars import SAMPLE_HDF_PATH, SAMPLE_CSV_PATH
except ImportError:
    # Fallback paths if vars.py not found
    SAMPLE_HDF_PATH = "../sample_data/core_test.hdf"
    # Fallback: use refined-set INDEX as label source
    _rset = os.path.abspath(os.path.join(os.path.dirname(__file__),
                 '..', '..', '..', 'data', 'refined-set'))
    SAMPLE_CSV_PATH = os.path.join(_rset, 'index', 'INDEX_refined_data.2020')

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
def load_sample_data(max_samples: int = 2000):
    """
    Load real molecular features from the PDBbind refined-set.

    sgcnn_features : (N, 25)   pocket AA composition + physicochemical
    cnn3d_features : (N,  64)  ECFP4+descriptor ligand fingerprint (PCA-reduced)
    labels         : (N,)      -logKd/Ki binding affinity
    complex_ids    : (N,)      PDB IDs
    """
    import os
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    RDLogger.DisableLog('rdApp.*')

    # ---- locate refined-set -----------------------------------------------
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    candidates   = [
        os.path.join(script_dir, '..', '..', '..', 'data', 'refined-set'),
        r'C:\bindingaffinity\data\refined-set',
    ]
    refined_root = next((os.path.abspath(p) for p in candidates
                         if os.path.isdir(os.path.abspath(p))), None)
    if refined_root is None:
        raise FileNotFoundError("Cannot find data/refined-set")
    print(f"Refined-set: {refined_root}")

    # ---- labels -----------------------------------------------------------
    index_path   = os.path.join(refined_root, 'index', 'INDEX_refined_data.2020')
    pdb_to_label = {}
    with open(index_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    pdb_to_label[parts[0].lower()] = float(parts[3])
                except ValueError:
                    pass
    print(f"Labels: {len(pdb_to_label)} entries")

    # ---- AA lookup tables -------------------------------------------------
    AA_IDX = {aa: i for i, aa in enumerate(
        ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
         'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'])}
    HYDRO  = {'ALA':1.8,'ARG':-4.5,'ASN':-3.5,'ASP':-3.5,'CYS':2.5,
               'GLN':-3.5,'GLU':-3.5,'GLY':-0.4,'HIS':-3.2,'ILE':4.5,
               'LEU':3.8,'LYS':-3.9,'MET':1.9,'PHE':2.8,'PRO':-1.6,
               'SER':-0.8,'THR':-0.7,'TRP':-0.9,'TYR':-1.3,'VAL':4.2}
    CHARGE = {'ARG':1,'LYS':1,'ASP':-1,'GLU':-1}
    POLAR  = {'SER','THR','ASN','GLN','TYR','HIS'}
    AROM   = {'PHE','TYR','TRP','HIS'}

    DESC_NAMES = ['MolWt','MolLogP','TPSA','NumHDonors','NumHAcceptors',
                  'NumRotatableBonds','RingCount','FractionCSP3',
                  'HeavyAtomCount','NumAromaticRings']

    def _pocket(pdb_path):
        residues, seen = [], set()
        try:
            with open(pdb_path) as f:
                for line in f:
                    if not (line.startswith('ATOM') or line.startswith('HETATM')):
                        continue
                    rn  = line[17:20].strip()
                    key = (line[21], line[22:26].strip(), rn)
                    if key not in seen and rn in AA_IDX:
                        seen.add(key); residues.append(rn)
        except Exception:
            return None
        if not residues:
            return None
        n    = len(residues)
        comp = np.zeros(20, dtype=np.float32)
        for r in residues:
            comp[AA_IDX[r]] += 1
        comp /= n
        phys = np.array([
            sum(HYDRO.get(r, 0)  for r in residues) / n,
            sum(CHARGE.get(r, 0) for r in residues) / n,
            sum(1 for r in residues if r in POLAR) / n,
            sum(1 for r in residues if r in AROM)  / n,
            float(np.log1p(n)) / 5.0,
        ], dtype=np.float32)
        return np.concatenate([comp, phys])   # (25,)

    def _ligand(sdf_path):
        try:
            mol = next((m for m in
                        Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=True)
                        if m is not None), None)
            if mol is None:
                return None
        except Exception:
            return None
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024),
                      dtype=np.float32)
        descs = []
        for name in DESC_NAMES:
            try:
                v = getattr(Descriptors, name)(mol)
                descs.append(0.0 if (v is None or not np.isfinite(v)) else float(v))
            except Exception:
                descs.append(0.0)
        return np.concatenate([fp, np.array(descs, dtype=np.float32)])  # (1034,)

    # ---- featurize --------------------------------------------------------
    poc_list, lig_list, lab_list, ids_list = [], [], [], []
    for pdb_id in sorted(pdb_to_label):
        folder = os.path.join(refined_root, pdb_id)
        sdf    = os.path.join(folder, f'{pdb_id}_ligand.sdf')
        pdb    = os.path.join(folder, f'{pdb_id}_pocket.pdb')
        if not (os.path.isfile(sdf) and os.path.isfile(pdb)):
            continue
        poc = _pocket(pdb)
        lig = _ligand(sdf)
        if poc is None or lig is None:
            continue
        poc_list.append(poc); lig_list.append(lig)
        lab_list.append(pdb_to_label[pdb_id]); ids_list.append(pdb_id)
        if len(lab_list) >= max_samples:
            break

    print(f"Featurized {len(lab_list)} complexes")
    poc_arr = np.array(poc_list, dtype=np.float32)   # (N, 25)
    lig_arr = np.array(lig_list, dtype=np.float32)   # (N, 1034)
    lab_arr = np.array(lab_list, dtype=np.float32)
    ids_arr = np.array(ids_list)

    # ---- PCA: compress ligand 1034 -> 64 dims ----------------------------
    # Fit scaler and PCA on the approximate train portion only (first 80%)
    # to avoid leaking val-set statistics into the feature transform.
    n_pca      = min(64, len(lab_arr) - 1)
    n_tr_approx = max(1, int(0.80 * len(lab_arr)))
    scaler_l = StandardScaler().fit(lig_arr[:n_tr_approx])
    pca_l    = PCA(n_components=n_pca, random_state=42).fit(
                   scaler_l.transform(lig_arr[:n_tr_approx]))
    lig_pca  = pca_l.transform(scaler_l.transform(lig_arr)).astype(np.float32)

    # StandardScale pocket too (already low-D, skip PCA)
    scaler_p   = StandardScaler().fit(poc_arr[:n_tr_approx])
    poc_scaled = scaler_p.transform(poc_arr).astype(np.float32)

    print(f"Feature dims — pocket: {poc_scaled.shape[1]}  ligand(PCA): {lig_pca.shape[1]}")
    print(f"Label range: {lab_arr.min():.2f} – {lab_arr.max():.2f}  "
          f"mean={lab_arr.mean():.2f}")
    return poc_scaled, lig_pca, lab_arr, ids_arr


def load_with_model_features(
    max_samples: int = 2000,
    dcnn_npz: str = None,   # path to refined_3dcnn_features.npz  (N, 10)
    sgcnn_npz: str = None,  # path to refined_sgcnn_features.npz  (N, 54)
):
    """
    Extends load_sample_data() with optional precomputed 3D-model features.

    Returns the same 4-tuple as load_sample_data(), but with all features
    packed into the first array and an empty (N, 0) placeholder for the
    second so the rest of the pipeline (FusionDataset, dims, model) is
    unchanged.

    Feature layout:
        pocket RDKit (25-dim)
      + ligand PCA  (64-dim)
      [+ 3DCNN fc1  (10-dim)  — when dcnn_npz is provided and exists]
      [+ SGCNN hid  (54-dim)  — when sgcnn_npz is provided and exists]
    """
    from sklearn.preprocessing import StandardScaler
    poc_scaled, lig_pca, lab_arr, ids_arr = load_sample_data(max_samples)
    rdkit_feats = np.hstack([poc_scaled, lig_pca]).astype(np.float32)  # (N, 89)

    parts = [rdkit_feats]

    if dcnn_npz and os.path.exists(dcnn_npz):
        npz = np.load(dcnn_npz, allow_pickle=False)
        dim = 10
        buf = np.zeros((len(ids_arr), dim), dtype=np.float32)
        hits = 0
        for i, pid in enumerate(ids_arr):
            if pid in npz:
                buf[i] = npz[pid]
                hits += 1
        # Fit scaler on first 80% (approx train portion) to avoid val leakage
        n_tr = max(1, int(0.80 * len(ids_arr)))
        buf = StandardScaler().fit(buf[:n_tr]).transform(buf).astype(np.float32)
        parts.append(buf)
        print(f"[load_with_model_features] 3DCNN features ({dim}-dim) "
              f"loaded for {hits}/{len(ids_arr)} complexes")
    else:
        if dcnn_npz:
            print(f"[load_with_model_features] 3DCNN NPZ not found: {dcnn_npz}")

    if sgcnn_npz and os.path.exists(sgcnn_npz):
        npz2 = np.load(sgcnn_npz, allow_pickle=False)
        # detect dim from first available entry (cov+noncov+pool+fc dims vary with arch)
        _sample2 = next(iter(npz2.values())) if len(npz2) > 0 else np.zeros(52)
        dim2 = int(_sample2.shape[0])
        buf2 = np.zeros((len(ids_arr), dim2), dtype=np.float32)
        hits2 = 0
        for i, pid in enumerate(ids_arr):
            if pid in npz2:
                buf2[i] = npz2[pid]
                hits2 += 1
        # Fit scaler on first 80% (approx train portion) to avoid val leakage
        n_tr2 = max(1, int(0.80 * len(ids_arr)))
        buf2 = StandardScaler().fit(buf2[:n_tr2]).transform(buf2).astype(np.float32)
        parts.append(buf2)
        print(f"[load_with_model_features] SGCNN features ({dim2}-dim) "
              f"loaded for {hits2}/{len(ids_arr)} complexes")
    else:
        if sgcnn_npz:
            print(f"[load_with_model_features] SGCNN NPZ not found: {sgcnn_npz}")

    combined = np.hstack(parts).astype(np.float32)
    empty    = np.zeros((combined.shape[0], 0), dtype=np.float32)
    print(f"[load_with_model_features] Final feature dim: {combined.shape[1]}")
    return combined, empty, lab_arr, ids_arr


def evaluate_model(model, loader):
    model.eval()
    dev   = next(model.parameters()).device
    preds, labs = [], []
    with torch.no_grad():
        for sg, c3, y in loader:
            combined = torch.cat([sg, c3], dim=1).to(dev)
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

    - Measures X, Y, Z on every qubit --> 3*n_qubits features (vs just Z before)
    - Skip connection: pre-quantum encoding concatenated with quantum output so
      gradients flow even when the reservoir adds no information for a particular
      circuit.
    - MLP head with BatchNorm + Dropout replaces a single Linear layer
    - Less aggressive compression: in_features -> 4*n_qubits -> n_qubits

    The G3 circuit (CNOT, H, T) is fixed and non-trainable.
    Only the classical layers are trained.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 qiskit_circuit,  # Qiskit QuantumCircuit from G3 family
                 n_qubits: int = 4,
                 backend: str = 'lightning.qubit'
                ):
        super().__init__()
        self.n_qubits = n_qubits

        # 1) Classical compressor: less aggressive squeeze (4x to 1x instead of 2x to 1x)
        self.fc1 = nn.Linear(in_features, 4 * n_qubits)
        self.bn1 = nn.BatchNorm1d(4 * n_qubits)
        self.fc2 = nn.Linear(4 * n_qubits, n_qubits)

        # 2) Fixed quantum reservoir (no trainable params)
        self.dev = qml.device(backend, wires=n_qubits)
        self.qiskit_circuit = qiskit_circuit

        @qml.qnode(self.dev, interface='torch')
        def quantum_reservoir(inputs):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            for instruction in qiskit_circuit.data:
                gate   = instruction.operation
                qubits = [qiskit_circuit.find_bit(q).index for q in instruction.qubits]
                if gate.name == 'h':
                    qml.Hadamard(wires=qubits[0])
                elif gate.name == 't':
                    qml.T(wires=qubits[0])
                elif gate.name == 'cx':
                    qml.CNOT(wires=qubits)
            # X, Y, Z measurements --> 3*n_qubits features instead of just Z
            return (
                [qml.expval(qml.PauliX(i)) for i in range(n_qubits)] +
                [qml.expval(qml.PauliY(i)) for i in range(n_qubits)] +
                [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            )

        self.quantum_reservoir = quantum_reservoir

        # 3) MLP head: (3*n_qubits quantum features + n_qubits skip) --> out_features
        q_features   = 3 * n_qubits
        combined_dim = q_features + n_qubits   # quantum + skip connection
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Classical preprocessing
        x     = torch.relu(self.bn1(self.fc1(x)))
        x_enc = torch.tanh(self.fc2(x)) * math.pi   # [batch, n_qubits]

        # Quantum reservoir (fixed, no grads through it)
        q_outputs = []
        for i in range(batch_size):
            q_out = self.quantum_reservoir(x_enc[i])
            q_outputs.append(torch.stack(q_out))
        q_out = torch.stack(q_outputs).float()   # [batch, 3*n_qubits]

        # Skip connection: append pre-quantum encoding
        combined = torch.cat([q_out, x_enc], dim=1)  # [batch, 3*n+n = 4*n]

        return self.head(combined)


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
            qubits = [qiskit_circuit.find_bit(q).index for q in instruction.qubits]
            
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
                x[i].double(),  # PennyLane expects double
                self.quantum_params.double(), 
                self.gate_structure, 
                self.n_qubits
            )
            q_outputs.append(torch.stack(q_out))
        
        q_out = torch.stack(q_outputs).float()
        
        return self.fc_out(q_out)


if __name__ == "__main__":
    # ---- Previous ModelHybridFC results without the 3dcnn and sgcnn (for reference) ----
    # 100 samples:  R²=0.36  Pearson r=0.71  (tiny test set, overfit)
    # 2000 samples: R²=0.10  Pearson r=0.38  (architecture bottleneck)

    # Import circuit utilities from sibling module
    try:
        from testing_random_unitaries import (
            generate_g3_random_circuits,
            preselect_circuits_by_expressibility,
        )
    except ImportError:
        from quantum_fusion.testing_random_unitaries import (
            generate_g3_random_circuits,
            preselect_circuits_by_expressibility,
        )

    # ---- Hyperparameters ------------------------------------------------
    N_QUBITS   = 6
    DEPTH      = 10
    epochs     = 50 # 200 is way more than needed for 100 samples, but with 2000+ samples and a deeper MLP head it helps to train longer and use a scheduler to reduce LR on plateaus
    batch_size = 64
    lr         = 3e-4   # lower LR suits the deeper MLP head
    device     = torch.device('cpu')   # PennyLane simulators are CPU-only

    # ---- Pick best G3 circuit by RFD expressibility ---------------------
    print("Generating & ranking G3 circuits by RFD expressibility...")
    circuits = generate_g3_random_circuits(N_QUBITS, DEPTH, num_circuits=20)
    indexed  = preselect_circuits_by_expressibility(circuits, N_QUBITS, top_k=1)
    _, best_circuit = indexed[0]
    print(f"Selected circuit #{indexed[0][0]}")

    # ---- Load real molecular features from PDBbind refined-set ----------
    # Auto-detect precomputed 3DCNN / SGCNN NPZ files sitting next to this script.
    _qf_dir    = os.path.dirname(os.path.abspath(__file__))
    _dcnn_npz  = os.path.join(_qf_dir, 'refined_3dcnn_features.npz')
    _sgcnn_npz = os.path.join(_qf_dir, 'refined_sgcnn_features.npz')

    _use_model_feats = os.path.exists(_dcnn_npz) or os.path.exists(_sgcnn_npz)
    if _use_model_feats:
        print("Pre-computed model NPZ file(s) detected — using load_with_model_features()")
        sgcnn_features, cnn3d_features, labels, complex_ids = load_with_model_features(
            max_samples=6000,
            dcnn_npz=_dcnn_npz  if os.path.exists(_dcnn_npz)  else None,
            sgcnn_npz=_sgcnn_npz if os.path.exists(_sgcnn_npz) else None,
        )
    else:
        print("No pre-computed NPZ files found — using RDKit-only features (load_sample_data)")
        sgcnn_features, cnn3d_features, labels, complex_ids = load_sample_data(max_samples=2000)

    n_samples = len(labels)
    n_train   = int(0.70 * n_samples)
    n_val     = int(0.15 * n_samples)
    n_test    = n_samples - n_train - n_val
    print(f"Split: {n_train} train / {n_val} val / {n_test} test")

    train_idx = np.arange(0, n_train)
    val_idx   = np.arange(n_train, n_train + n_val)
    test_idx  = np.arange(n_train + n_val, n_train + n_val + n_test)

    datasets = {
        'train': {'sg': sgcnn_features[train_idx], 'c3': cnn3d_features[train_idx],
                  'y': labels[train_idx],           'ids': complex_ids[train_idx]},
        'val':   {'sg': sgcnn_features[val_idx],   'c3': cnn3d_features[val_idx],
                  'y': labels[val_idx],             'ids': complex_ids[val_idx]},
        'test':  {'sg': sgcnn_features[test_idx],  'c3': cnn3d_features[test_idx],
                  'y': labels[test_idx],            'ids': complex_ids[test_idx]},
    }

    # ---- Normalise labels to zero-mean / unit-std -----------------------
    label_mean = float(datasets['train']['y'].mean())
    label_std  = float(datasets['train']['y'].std()) + 1e-8
    print(f"Label stats — mean={label_mean:.3f}  std={label_std:.3f}")
    for split in datasets:
        datasets[split]['y'] = (datasets[split]['y'] - label_mean) / label_std

    # ---- DataLoaders ----------------------------------------------------
    loaders = {}
    dims    = datasets['train']['sg'].shape[1] + datasets['train']['c3'].shape[1]
    for split in ['train', 'val', 'test']:
        ds = FusionDataset(datasets[split]['sg'], datasets[split]['c3'],
                           datasets[split]['y'])
        loaders[split] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=(split == 'train'))
    print(f"Input dim: {dims}  (pocket {datasets['train']['sg'].shape[1]} "
          f"+ ligand {datasets['train']['c3'].shape[1]})")

    # ---- Model: Reservoir with X/Y/Z observables + skip + MLP head -----
    model = ModelHybridFC_Reservoir(
        in_features=dims,
        out_features=1,
        qiskit_circuit=best_circuit,
        n_qubits=N_QUBITS,
        backend='lightning.qubit',
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()

    # ---- Training loop --------------------------------------------------
    best_val   = float('inf')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              f'reservoir_run_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for sg, c3, y in tqdm(loaders['train'], leave=False):
            x = torch.cat([sg, c3], dim=1).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(loaders['train'])

        rmse, mae, r2, pearson, spearman = evaluate_model(model, loaders['val'])
        scheduler.step(rmse)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: Train RMSE={math.sqrt(train_loss):.4f}  "
                  f"Val RMSE={rmse:.4f}  R²={r2:.4f}  "
                  f"Pearson={pearson:.4f}  lr={current_lr:.2e}")
        if rmse < best_val:
            best_val = rmse
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

    # ---- Final test evaluation ------------------------------------------
    ckpt = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        rmse, mae, r2, pearson, spearman = evaluate_model(model, loaders['test'])
        print(f"\n=== Test Results (ModelHybridFC_Reservoir, {N_QUBITS} qubits) ===")
        print(f"  RMSE (normalised) : {rmse:.4f}   ({rmse * label_std:.4f} pKi units)")
        print(f"  MAE  (normalised) : {mae:.4f}   ({mae  * label_std:.4f} pKi units)")
        print(f"  R²                : {r2:.4f}")
        print(f"  Pearson r         : {pearson:.4f}")
        print(f"  Spearman ρ        : {spearman:.4f}")
        print(f"  Saved to          : {output_dir}")
    else:
        print("No saved model found.")

    print("\nTraining completed!")


    # WITHOUT 3dCNN AND SGCNN FEATURES (for reference):
    # epochs = 100
    # batch_size = 32
    # lr = 0.001
    # # PennyLane quantum simulators (lightning.qubit, default.qubit) run on
    # # CPU only — CUDA causes a device mismatch in the hybrid forward pass.
    # device = torch.device('cpu')

    # # ---------------- Load & Prepare Data ----------------
    # # Load sample data
    print("\nTraining completed!")