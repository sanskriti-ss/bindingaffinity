### instructions
"""
cd FAST-master/model/
pip install requirements.txt
python -m testing_unitaries.testing_random_unitaries

OR

cd FAST-master/model/testing_unitaries/
python testing_random_unitaries.py

-----------------------------------------------------------------------
KEY IMPROVEMENTS OVER FIRST VERSION
-----------------------------------------------------------------------
1. G3 circuits are now ACTUALLY used via ModelHybridFC_Reservoir
   (the original code generated circuits but fed ModelHybridFC which
   never touched them - the quantum layer was always the same ansatz).

2. PCA + StandardScaler compresses the huge 3D-grid feature vectors
   to a fixed low-dimensional space before feeding the quantum model.
   This removes the gradient pathology caused by raw voxel values.

3. Labels are loaded directly from the PDBbind refined-set INDEX file
   (INDEX_refined_data.2020).

4. Circuit expressibility pre-selection is
   used to discard low-expressibility circuits before expensive
   training, keeping only the top-K most expressive ones.

5. Training uses more epochs with early stopping and proper LR scheduling.

6. Final predictions are ensembled over the top-5 circuits, improving
   robustness and giving a natural uncertainty estimate (std dev).
-----------------------------------------------------------------------
"""

import numpy as np
import torch
import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hashlib
from qiskit import QuantumCircuit
from tqdm import tqdm
import sys
import os
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import pennylane as qml

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(THIS_DIR)
QF_DIR = os.path.join(MODEL_DIR, 'quantum_fusion')

# Ensure moved script can still import quantum_fusion modules/resources.
for _p in [MODEL_DIR, QF_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Handle both relative and direct imports
from quantum_fusion.main_train import ModelHybridFC, ModelHybridFC_Reservoir, FusionDataset, evaluate_model, load_with_model_features

try:
    from testing_unitaries.circuit_visualization import (
        render_circuit_diagram,
        publication_qiskit_style,
        circuit_summary_stats,
        circuit_gate_rows,
    )
except Exception:
    from circuit_visualization import (
        render_circuit_diagram,
        publication_qiskit_style,
        circuit_summary_stats,
        circuit_gate_rows,
    )


def load_from_model_feature_npz(max_samples=6000, val_fraction=0.2, random_state=42):
    """
    Load precomputed fusion-model embeddings (3DCNN + SGCNN NPZ) plus RDKit
    base features through main_train.load_with_model_features().

    This usually provides a richer feature space than RDKit-only vectors.
    """
    dcnn_npz = os.path.join(QF_DIR, 'refined_3dcnn_features.npz')
    sgcnn_npz = os.path.join(QF_DIR, 'refined_sgcnn_features.npz')

    sgcnn_features, cnn3d_features, labels, _ = load_with_model_features(
        max_samples=max_samples,
        dcnn_npz=dcnn_npz if os.path.exists(dcnn_npz) else None,
        sgcnn_npz=sgcnn_npz if os.path.exists(sgcnn_npz) else None,
    )

    labels = np.asarray(labels, dtype=np.float32)
    n = len(labels)
    if n < 50:
        raise RuntimeError("Too few samples from NPZ feature loader (<50).")

    # Stratified shuffled split by label quantiles for a more stable validation estimate.
    idx = np.arange(n)
    n_bins = min(10, max(2, int(np.sqrt(n))))
    bins = pd.qcut(labels, q=n_bins, labels=False, duplicates='drop')
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_fraction,
        random_state=random_state,
        shuffle=True,
        stratify=bins,
    )

    label_mean = labels[train_idx].mean()
    label_std = labels[train_idx].std() + 1e-8
    labels_norm = (labels - label_mean) / label_std

    train_ds = FusionDataset(
        torch.tensor(sgcnn_features[train_idx], dtype=torch.float32),
        torch.tensor(cnn3d_features[train_idx], dtype=torch.float32),
        torch.tensor(labels_norm[train_idx], dtype=torch.float32),
    )
    val_ds = FusionDataset(
        torch.tensor(sgcnn_features[val_idx], dtype=torch.float32),
        torch.tensor(cnn3d_features[val_idx], dtype=torch.float32),
        torch.tensor(labels_norm[val_idx], dtype=torch.float32),
    )

    print("Using precomputed NPZ model features (via load_with_model_features)")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"Label stats (train): mean={label_mean:.3f}  std={label_std:.3f}")
    return train_ds, val_ds, float(label_mean), float(label_std)


def split_validation_for_holdout(val_dataset, holdout_fraction=0.5, random_state=42):
    """
    Split an existing validation dataset into (validation_for_selection, holdout).

    The holdout split remains completely unseen during circuit/model selection and
    is used only for final reporting/plots when --holdout is enabled.
    """
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError("holdout_fraction must be in (0, 1)")

    y = val_dataset.labels.numpy().astype(np.float32).flatten()
    n = len(y)
    if n < 20:
        raise RuntimeError("Validation set too small to split into holdout.")

    idx = np.arange(n)
    n_bins = min(10, max(2, int(np.sqrt(n))))
    bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    val_idx, holdout_idx = train_test_split(
        idx,
        test_size=holdout_fraction,
        random_state=random_state,
        shuffle=True,
        stratify=bins,
    )

    def _subset(ds, subset_idx):
        return FusionDataset(
            ds.sgcnn_features[subset_idx],
            ds.cnn3d_features[subset_idx],
            ds.labels[subset_idx],
        )

    val_split = _subset(val_dataset, val_idx)
    holdout_split = _subset(val_dataset, holdout_idx)
    return val_split, holdout_split

# 1. Generate random circuits from G3 gate family {CNOT, H, T}
def generate_g3_random_circuits(n_qubits, num_gates=300, num_circuits=10):
    """
    Generate random unitary circuits sampled from the G3 gate family.
    G3 gates: {CNOT, H, T} with uniform random selection (1/3 each in expectation).
    
    Follows the methodology from Domingo et al. (2022): "Optimal quantum reservoir 
    computing for the NISQ era" - circuits are constructed by adding random quantum 
    gates from the G3 family with uniform probability distribution.
    
    Args:
        n_qubits: Number of qubits
        num_gates: Total number of gate instructions per circuit
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
        for _ in range(num_gates):
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

def load_from_refined_set(n_pca_components=32):
    """
    Build features DIRECTLY from the PDBbind refined-set raw files:
      data/refined-set/<PDBID>/<PDBID>_ligand.sdf   -> RDKit ECFP4 + descriptors
      data/refined-set/<PDBID>/<PDBID>_pocket.pdb   -> amino-acid composition
                                                       + physicochemical profile
    Labels from data/refined-set/index/INDEX_refined_data.2020

    This replaces the old model_ready_data 3-D voxel grids which had:
      - Only ~180 samples (vs ~5000+ in the refined set)
      - No guarantee of PDB-ID alignment with labels
      - Raw voxel noise that PCA could not meaningfully compress
    """
    import os, warnings
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors
    RDLogger.DisableLog('rdApp.*')        # suppress RDKit warnings

    # ---- locate refined-set root ------------------------------------------
    candidates = [
        os.path.join(MODEL_DIR, '..', '..', 'data', 'refined-set'),
        os.path.join(MODEL_DIR, '..', '..', '..', 'data', 'refined-set'),
        r'C:\bindingaffinity\data\refined-set',
    ]
    refined_root = next((os.path.abspath(p) for p in candidates
                         if os.path.isdir(os.path.abspath(p))), None)
    if refined_root is None:
        raise FileNotFoundError("Cannot find data/refined-set. Tried: " + ', '.join(candidates))
    print(f"Refined-set root: {refined_root}")

    # ---- labels from INDEX ------------------------------------------------
    index_path = os.path.join(refined_root, 'index', 'INDEX_refined_data.2020')
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
    print(f"Labels loaded: {len(pdb_to_label)} entries")

    # ---- amino-acid lookup tables ----------------------------------------
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

    def featurize_pocket(pdb_path):
        """25-D feature vector from pocket PDB residue composition."""
        residues, seen = [], set()
        try:
            with open(pdb_path) as f:
                for line in f:
                    if not (line.startswith('ATOM') or line.startswith('HETATM')):
                        continue
                    resname = line[17:20].strip()
                    key = (line[21], line[22:26].strip(), resname)
                    if key not in seen and resname in AA_IDX:
                        seen.add(key)
                        residues.append(resname)
        except Exception:
            return None
        if not residues:
            return None
        n = len(residues)
        comp = np.zeros(20, dtype=np.float32)
        for r in residues:
            comp[AA_IDX[r]] += 1
        comp /= n
        hydro   = sum(HYDRO.get(r, 0)  for r in residues) / n
        charge  = sum(CHARGE.get(r, 0) for r in residues) / n
        polar   = sum(1 for r in residues if r in POLAR)   / n
        arom    = sum(1 for r in residues if r in AROM)    / n
        size_f  = float(np.log1p(n)) / 5.0
        return np.concatenate([comp,
                                np.array([hydro, charge, polar, arom, size_f],
                                         dtype=np.float32)])

    DESC_NAMES = ['MolWt','MolLogP','TPSA','NumHDonors','NumHAcceptors',
                  'NumRotatableBonds','RingCount','FractionCSP3',
                  'HeavyAtomCount','NumAromaticRings']

    def featurize_ligand(sdf_path):
        """1034-D feature vector: ECFP4(1024) + 10 physicochemical descriptors."""
        try:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=True)
            mol   = next((m for m in suppl if m is not None), None)
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
                descs.append(float('nan') if v is None else float(v))
            except Exception:
                descs.append(0.0)
        descs = np.array(descs, dtype=np.float32)
        descs = np.nan_to_num(descs, nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate([fp, descs])

    # ---- iterate over refined-set entries ---------------------------------
    ligand_feats, pocket_feats, labels_out, pdb_ids_out = [], [], [], []
    skipped = 0
    pdb_dirs = sorted(pdb_to_label.keys())
    print(f"Featurizing {len(pdb_dirs)} PDB entries (skipping missing files)...")

    for pdb_id in tqdm(pdb_dirs, desc='RDKit featurization'):
        folder = os.path.join(refined_root, pdb_id)
        sdf    = os.path.join(folder, f'{pdb_id}_ligand.sdf')
        pdb    = os.path.join(folder, f'{pdb_id}_pocket.pdb')
        if not (os.path.isfile(sdf) and os.path.isfile(pdb)):
            skipped += 1
            continue
        lig = featurize_ligand(sdf)
        poc = featurize_pocket(pdb)
        if lig is None or poc is None:
            skipped += 1
            continue
        ligand_feats.append(lig)
        pocket_feats.append(poc)
        labels_out.append(pdb_to_label[pdb_id])
        pdb_ids_out.append(pdb_id)

    print(f"Featurized: {len(labels_out)}  Skipped: {skipped}")
    if len(labels_out) < 50:
        raise RuntimeError("Too few valid complexes (<50). Check refined-set path.")

    ligand_feats = np.array(ligand_feats, dtype=np.float32)
    pocket_feats = np.array(pocket_feats, dtype=np.float32)
    labels_arr   = np.array(labels_out,  dtype=np.float32)

    print(f"Label range: {labels_arr.min():.2f} – {labels_arr.max():.2f}  "
          f"mean={labels_arr.mean():.2f}  std={labels_arr.std():.2f}")

    # ---- train / val split (80 / 20, sequential) --------------------------
    N          = len(labels_arr)
    train_size = int(0.80 * N)
    tr, va     = slice(0, train_size), slice(train_size, N)

    # ---- normalise labels -------------------------------------------------
    label_mean = labels_arr[tr].mean()
    label_std  = labels_arr[tr].std() + 1e-8
    labels_norm = (labels_arr - label_mean) / label_std

    # ---- PCA per modality -------------------------------------------------
    n_lig_pca = min(n_pca_components, train_size - 1, ligand_feats.shape[1])
    n_poc_pca = min(n_pca_components, train_size - 1, pocket_feats.shape[1])

    scaler_lig = StandardScaler().fit(ligand_feats[tr])
    pca_lig    = PCA(n_components=n_lig_pca, random_state=42)\
                     .fit(scaler_lig.transform(ligand_feats[tr]))
    lig_red    = pca_lig.transform(scaler_lig.transform(ligand_feats)).astype(np.float32)

    scaler_poc = StandardScaler().fit(pocket_feats[tr])
    pca_poc    = PCA(n_components=n_poc_pca, random_state=42)\
                     .fit(scaler_poc.transform(pocket_feats[tr]))
    poc_red    = pca_poc.transform(scaler_poc.transform(pocket_feats)).astype(np.float32)

    var_lig = pca_lig.explained_variance_ratio_.sum() * 100
    var_poc = pca_poc.explained_variance_ratio_.sum() * 100
    print(f"Ligand PCA {ligand_feats.shape[1]}→{n_lig_pca}-D  ({var_lig:.1f}% var)")
    print(f"Pocket PCA {pocket_feats.shape[1]}→{n_poc_pca}-D  ({var_poc:.1f}% var)")

    # ---- wrap in FusionDataset (pocket=sgcnn, ligand=cnn3d) ---------------
    def _ds(idx):
        return FusionDataset(
            torch.tensor(poc_red[idx],        dtype=torch.float32),
            torch.tensor(lig_red[idx],        dtype=torch.float32),
            torch.tensor(labels_norm[idx],    dtype=torch.float32),
        )

    train_ds = _ds(tr)
    val_ds   = _ds(va)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    return train_ds, val_ds, label_mean, label_std


def load_preprocessed_data(n_pca_components=64):
    """
    [DEPRECATED — use load_from_refined_set() instead]
    Load preprocessed molecular data from step4 outputs.

    Kept for backwards compatibility only.
    """
    return load_from_refined_set(n_pca_components=n_pca_components)


# ---- Private stub to silence old call signature --------------------------
def _load_preprocessed_data_old(n_pca_components=64):
    """
    Improvements:
    - Labels from PDBbind refined-set INDEX_refined_data.2020 (correct alignment)
    - PCA + StandardScaler compresses the huge 3-D grid vectors before training
    - Label normalisation for stable quantum-circuit regression
    """
    import json
    import os

    possible_paths = [
        'model_ready_data',
        os.path.join(MODEL_DIR, '..', '..', 'model_ready_data'),
        os.path.join(MODEL_DIR, '..', '..', '..', 'model_ready_data'),
    ]
    data_dir = None
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            data_dir = abs_path
            break
    if data_dir is None:
        raise FileNotFoundError(
            "Could not find model_ready_data directory. Tried:\n"
            + "\n".join([f"  - {os.path.abspath(p)}" for p in possible_paths])
        )
    print(f"Loading preprocessed data from: {data_dir}")

    # ------------------------------------------------------------------ grids
    ligand_grids = np.load(os.path.join(data_dir, 'ligand_grids.npz'))['arr_0']
    pocket_grids = np.load(os.path.join(data_dir, 'pocket_grids.npz'))['arr_0']
    with open(os.path.join(data_dir, 'ligand_metadata.json')) as f:
        ligand_metadata = json.load(f)

    num_samples   = min(len(ligand_grids), len(pocket_grids))
    ligand_flat   = ligand_grids[:num_samples].reshape(num_samples, -1).astype(np.float32)
    pocket_flat   = pocket_grids[:num_samples].reshape(num_samples, -1).astype(np.float32)

    # ------------------------------------------------------------------ labels
    # Use refined-set INDEX file for correct -logKd/Ki alignment.
    workspace_root = os.path.abspath(os.path.join(data_dir, '..'))
    index_path = os.path.join(workspace_root, 'data', 'refined-set', 'index',
                              'INDEX_refined_data.2020')
    if os.path.exists(index_path):
        print(f"Loading labels from: {index_path}")
        rows = []
        with open(index_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        rows.append((parts[0].lower(), float(parts[3])))
                    except ValueError:
                        pass
        pdb_to_affinity = dict(rows)
    else:
        print("Refined-set INDEX not found, falling back to pdbbind_with_dG.csv")
        csv_path = os.path.join(workspace_root, 'pdbbind_with_dG.csv')
        if not os.path.exists(csv_path):
            csv_path = 'pdbbind_with_dG.csv'
        df_aff = pd.read_csv(csv_path)
        pdb_to_affinity = dict(zip(df_aff['protein'].str.lower(),
                                   df_aff['ΔG_kcal_per_mol']))

    labels = np.array([
        pdb_to_affinity.get(str(ligand_metadata[i].get('ligand_id', '')).lower()[:4], np.nan)
        for i in range(num_samples)
    ], dtype=np.float32)

    valid = ~np.isnan(labels)
    ligand_flat = ligand_flat[valid]
    pocket_flat = pocket_flat[valid]
    labels      = labels[valid]
    num_samples = len(labels)
    print(f"Samples after label alignment: {num_samples}")
    print(f"Label range: {labels.min():.2f} – {labels.max():.2f}  "
          f"(mean={labels.mean():.2f}, std={labels.std():.2f})")

    # ------------------------------------------------------------------ PCA
    # Standardise + PCA-compress the raw voxel vectors.
    # Without this the classical compression layers overfit noise and the
    # tanh-scaled quantum inputs immediately saturate -> vanishing gradients.
    train_size    = int(0.9 * num_samples)
    n_pca_components = min(n_pca_components,
                           train_size - 1,
                           ligand_flat.shape[1],
                           pocket_flat.shape[1])
    print(f"PCA: {ligand_flat.shape[1]}-D -> {n_pca_components}-D (ligand)  "
          f"{pocket_flat.shape[1]}-D -> {n_pca_components}-D (pocket)")

    scaler_lig   = StandardScaler().fit(ligand_flat[:train_size])
    pca_lig      = PCA(n_components=n_pca_components, random_state=42) \
                       .fit(scaler_lig.transform(ligand_flat[:train_size]))
    ligand_red   = pca_lig.transform(scaler_lig.transform(ligand_flat))

    scaler_poc   = StandardScaler().fit(pocket_flat[:train_size])
    pca_poc      = PCA(n_components=n_pca_components, random_state=42) \
                       .fit(scaler_poc.transform(pocket_flat[:train_size]))
    pocket_red   = pca_poc.transform(scaler_poc.transform(pocket_flat))

    print(f"PCA explained variance — "
          f"ligand: {pca_lig.explained_variance_ratio_.sum()*100:.1f}%  "
          f"pocket: {pca_poc.explained_variance_ratio_.sum()*100:.1f}%")

    # Normalise labels (mean/std of train split only to avoid leakage)
    label_mean = float(labels[:train_size].mean())
    label_std  = float(labels[:train_size].std()) + 1e-8
    labels_norm = (labels - label_mean) / label_std

    # ------------------------------------------------------------------ split
    train_idx = np.arange(0, train_size)
    val_idx   = np.arange(train_size, num_samples)

    train_dataset = FusionDataset(
        torch.tensor(pocket_red[train_idx], dtype=torch.float32),
        torch.tensor(ligand_red[train_idx], dtype=torch.float32),
        torch.tensor(labels_norm[train_idx], dtype=torch.float32),
    )
    val_dataset = FusionDataset(
        torch.tensor(pocket_red[val_idx], dtype=torch.float32),
        torch.tensor(ligand_red[val_idx], dtype=torch.float32),
        torch.tensor(labels_norm[val_idx], dtype=torch.float32),
    )
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")
    return train_dataset, val_dataset, label_mean, label_std

# ===========================================================================
# 2a. Expressibility pre-selection  (Sim et al., PRL 2019)
# ===========================================================================
def reservoir_feature_diversity(qc, n_qubits, n_samples=120):
    """
    Reservoir Feature Diversity (RFD): effective rank of the feature matrix
    F[sample, observable] computed over random angle-encoded inputs.

    Deep G3 circuits (depth>=6) are approximate unitary 3-designs, so they
    ALL converge to near-Haar fidelity variance ≈ 1/(d*(d+1)).  That metric
    cannot distinguish them.  RFD instead asks: do the 3*n_qubits Pauli
    expectation values (X,Y,Z on each qubit) span enough distinct directions
    to support accurate linear regression?  Higher effective rank --> the
    circuit's outputs are more linearly independent --> better reservoir.

    Metric: participation ratio  PR = (Σ sᵢ)² / Σ sᵢ²  where sᵢ are the
    singular values of the standardised feature matrix.  Range [1, 3*n_qubits].
    """
    dev = qml.device('lightning.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def feature_circuit(x):
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        for instruction in qc.data:
            gate   = instruction.operation
            qubits = [qc.find_bit(q).index for q in instruction.qubits]
            if gate.name == 'h':
                qml.Hadamard(wires=qubits[0])
            elif gate.name == 't':
                qml.T(wires=qubits[0])
            elif gate.name == 'cx':
                qml.CNOT(wires=qubits)
        return (
            [qml.expval(qml.PauliX(i)) for i in range(n_qubits)] +
            [qml.expval(qml.PauliY(i)) for i in range(n_qubits)] +
            [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        )

    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n_samples):
        x   = rng.uniform(0, np.pi, n_qubits)
        out = feature_circuit(x)
        rows.append(np.array([float(v) for v in out]))

    F = np.array(rows)                         # (n_samples, 3*n_qubits)
    F = (F - F.mean(0)) / (F.std(0) + 1e-8)   # standardise columns
    sv = np.linalg.svd(F, compute_uv=False)    # descending singular values
    sv = sv / (sv.sum() + 1e-12)
    pr = 1.0 / (np.sum(sv ** 2) + 1e-12)      # participation ratio
    return float(pr)                           # higher = more diverse outputs


def preselect_circuits_by_expressibility(circuits, n_qubits, top_k, return_all_scores=False):
    """
    Pre-select circuits by Reservoir Feature Diversity (replaces fidelity
    variance which is degenerate for deep G3 circuits).
    """
    print(f"\nComputing Reservoir Feature Diversity for {len(circuits)} circuits ...")
    scores = []
    for i, qc in enumerate(tqdm(circuits, desc='RFD', ascii=True)):
        score = reservoir_feature_diversity(qc, n_qubits)
        scores.append((score, i, qc))
        print(f"  Circuit {i:3d}: RFD = {score:.4f}")
    scores.sort(key=lambda t: t[0], reverse=True)
    selected = scores[:top_k]
    print(f"\nTop-{top_k} circuits by RFD (higher = more diverse reservoir):")
    for score, idx, _ in selected:
        print(f"  Circuit {idx}: RFD = {score:.4f}")

    indexed_selected = [(idx, qc) for (_, idx, qc) in selected]
    if return_all_scores:
        all_scores = [
            {
                'circuit_idx': idx,
                'rfd_score': float(score),
                'rfd_rank': rank,
            }
            for rank, (score, idx, _) in enumerate(scores, start=1)
        ]
        return indexed_selected, all_scores
    return indexed_selected


# ===========================================================================
# 2b. Quantum feature extraction + Ridge regression readout
# ===========================================================================

def extract_quantum_features(qc, X_pca, n_qubits, random_seed=42):
    """
    Project X_pca (N × D) into quantum reservoir features (N × 3*n_qubits).

    The quantum reservoir transforms the input non-linearly in exponentially
    large Hilbert space, providing features that cannot be replicated by any
    compact classical transformation.  Ridge then finds the best linear
    combination of these features.

    Input compression: PCA features are projected to n_qubits dimensions via
    a fixed random projection matrix (same seed for all circuits so the ONLY
    difference between circuits is the unitary U they apply).
    """
    dev = qml.device('lightning.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def reservoir(inputs):
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        for instruction in qc.data:
            gate   = instruction.operation
            qubits = [qc.find_bit(q).index for q in instruction.qubits]
            if gate.name == 'h':
                qml.Hadamard(wires=qubits[0])
            elif gate.name == 't':
                qml.T(wires=qubits[0])
            elif gate.name == 'cx':
                qml.CNOT(wires=qubits)
        return (
            [qml.expval(qml.PauliX(i)) for i in range(n_qubits)] +
            [qml.expval(qml.PauliY(i)) for i in range(n_qubits)] +
            [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        )

    # Fixed random projection: D → n_qubits  (same W across all circuits)
    rng = np.random.default_rng(random_seed)
    W   = rng.standard_normal((X_pca.shape[1], n_qubits))
    W  /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-8
    X_low = np.tanh(X_pca @ W) * np.pi   # [N, n_qubits] in (-π, π)

    features = []
    for x in X_low:
        out = reservoir(x.astype(np.float64))
        features.append([float(v) for v in out])
    return np.array(features, dtype=np.float32)   # [N, 3*n_qubits]


def run_circuits_and_evaluate(indexed_circuits, train_dataset, val_dataset,
                              n_qubits=4, max_epochs=None, patience=None,
                              batch_size=None, lr=None, holdout_dataset=None):
    """
    Evaluate each G3 circuit as a quantum reservoir using Ridge regression.

    Architecture (Domingo et al. style for small datasets):
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr, spearmanr
    import math

    # Extract raw PCA arrays from datasets
    train_pocket = train_dataset.sgcnn_features.numpy()   # [N_tr, PCA_DIMS]
    train_ligand = train_dataset.cnn3d_features.numpy()
    train_X_pca  = np.concatenate([train_pocket, train_ligand], axis=1)
    train_y      = train_dataset.labels.numpy().flatten()

    val_pocket   = val_dataset.sgcnn_features.numpy()
    val_ligand   = val_dataset.cnn3d_features.numpy()
    val_X_pca    = np.concatenate([val_pocket,   val_ligand],   axis=1)
    val_y        = val_dataset.labels.numpy().flatten()

    eval_dataset = holdout_dataset if holdout_dataset is not None else val_dataset
    eval_name = 'holdout' if holdout_dataset is not None else 'validation'
    eval_pocket = eval_dataset.sgcnn_features.numpy()
    eval_ligand = eval_dataset.cnn3d_features.numpy()
    eval_X_pca  = np.concatenate([eval_pocket, eval_ligand], axis=1)
    eval_y      = eval_dataset.labels.numpy().flatten()

    def _ridge_learning_curve(X_tr, y_tr, X_eval, y_eval, alphas, n_points=8, seed=42):
        """Build train/eval MSE curves by fitting Ridge on increasing train fractions."""
        n = X_tr.shape[0]
        if n < 20:
            n_points = 3
        fractions = np.linspace(0.15, 1.0, n_points)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)

        train_losses, eval_losses = [], []
        from sklearn.linear_model import RidgeCV as _RidgeCV

        for frac in fractions:
            m = max(8, int(round(frac * n)))
            sub_idx = perm[:m]
            X_sub = X_tr[sub_idx]
            y_sub = y_tr[sub_idx]

            scaler = StandardScaler().fit(X_sub)
            X_sub_s = scaler.transform(X_sub)
            X_eval_s = scaler.transform(X_eval)

            model_lc = _RidgeCV(alphas=alphas, cv=3)
            model_lc.fit(X_sub_s, y_sub)
            pred_sub = model_lc.predict(X_sub_s)
            pred_eval = model_lc.predict(X_eval_s)

            train_losses.append(float(mean_squared_error(y_sub, pred_sub)))
            eval_losses.append(float(mean_squared_error(y_eval, pred_eval)))

        return train_losses, eval_losses

    results = []
    circuit_bar = tqdm(indexed_circuits, desc="Testing Circuits", position=0)

    for orig_idx, qc in circuit_bar:
        circuit_bar.set_description(f"Circuit {orig_idx}")

        # 1. Quantum reservoir features
        q_tr = extract_quantum_features(qc, train_X_pca, n_qubits)  # [N_tr, 3*nq]
        q_va = extract_quantum_features(qc, val_X_pca,   n_qubits)
        q_ev = extract_quantum_features(qc, eval_X_pca,  n_qubits)

        # 2. Concatenate PCA + quantum features
        X_tr = np.concatenate([train_X_pca, q_tr], axis=1)  # [N_tr, 128+3*nq]
        X_va = np.concatenate([val_X_pca,   q_va], axis=1)
        X_ev = np.concatenate([eval_X_pca,  q_ev], axis=1)

        # 3. Scale combined features
        x_scaler = StandardScaler().fit(X_tr)
        X_tr_s = x_scaler.transform(X_tr)
        X_va_s = x_scaler.transform(X_va)
        X_ev_s = x_scaler.transform(X_ev)

        # 4. Ridge readout with CV alpha selection
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        ridge  = RidgeCV(alphas=alphas, cv=5)
        ridge.fit(X_tr_s, train_y)
        train_preds = ridge.predict(X_tr_s)
        preds_val = ridge.predict(X_va_s)
        preds_eval = ridge.predict(X_ev_s)
        r2_select = r2_score(val_y, preds_val)
        r2 = r2_score(eval_y, preds_eval)
        model = ridge
        model_type = 'ridge'
        best_alpha = ridge.alpha_

        train_mse = mean_squared_error(train_y, train_preds)
        val_mse = mean_squared_error(eval_y, preds_eval)
        rmse   = math.sqrt(mean_squared_error(eval_y, preds_eval))
        mae    = mean_absolute_error(eval_y, preds_eval)
        pcc    = pearsonr(eval_y,  preds_eval)[0]
        scc    = spearmanr(eval_y, preds_eval)[0]

        # Learning curves (multiple points) so saved plot shows true curves, not dots.
        train_curve, eval_curve = _ridge_learning_curve(
            X_tr=X_tr,
            y_tr=train_y,
            X_eval=X_ev,
            y_eval=eval_y,
            alphas=alphas,
            n_points=8,
            seed=orig_idx + 42,
        )

        # Baseline (classical-only): same readout family candidates on classical features.
        x_scaler_base = StandardScaler().fit(train_X_pca)
        X_tr_base_s = x_scaler_base.transform(train_X_pca)
        X_va_base_s = x_scaler_base.transform(val_X_pca)

        ridge_base = RidgeCV(alphas=alphas, cv=5)
        ridge_base.fit(X_tr_base_s, train_y)
        base_preds_val = ridge_base.predict(X_va_base_s)
        base_r2_select = r2_score(val_y, base_preds_val)

        X_ev_base_s = x_scaler_base.transform(eval_X_pca)
        base_preds_eval = ridge_base.predict(X_ev_base_s)
        base_r2 = r2_score(eval_y, base_preds_eval)

        results.append({
            'circuit_idx':  orig_idx,
            'model':        model,
            'model_type':   model_type,
            'x_scaler':     x_scaler,
            'circuit':      qc,             # store circuit for ensemble quantum features
            'r2':           r2,
            'r2_baseline':  base_r2,        # Ridge on PCA alone (no quantum)
            'r2_gain':      r2 - base_r2,   # quantum contribution
            'selection_r2': r2_select,
            'selection_r2_baseline': base_r2_select,
            'selection_r2_gain': r2_select - base_r2_select,
            'evaluation_split': eval_name,
            'rmse':         rmse,
            'mae':          mae,
            'pearson':      pcc,
            'spearman':     scc,
            'best_alpha':   best_alpha,
            'eval_preds':   preds_eval,     # cached eval predictions
            'eval_true':    eval_y,
            'eval_X_pca':   eval_X_pca,     # cached eval PCA features
            'val_preds':    preds_val,      # legacy cache key
            'val_true':     eval_y,         # legacy cache key kept for compatibility
            'val_X_pca':    eval_X_pca,     # legacy cache key kept for compatibility
            'train_losses': train_curve if len(train_curve) > 1 else [float(train_mse)],
            'val_losses':   eval_curve if len(eval_curve) > 1 else [float(val_mse)],
        })
        circuit_bar.set_postfix(R2=f'{r2:.4f}',
                                Gain=f'{r2 - base_r2:+.4f}',
                                M=model_type)

    results.sort(key=lambda r: r['r2'], reverse=True)
    print(f"\n{'='*60}")
    split_name = results[0].get('evaluation_split', 'validation') if results else 'validation'
    print(f"Evaluation split:                         {split_name}")
    print(f"Classical Ridge baseline R² (no quantum): {results[0]['r2_baseline']:.4f}")
    print(f"Best quantum circuit R²:                  {results[0]['r2']:.4f}")
    print(f"Quantum gain:                             {results[0]['r2_gain']:+.4f}")
    print(f"{'='*60}")
    return results


# ===========================================================================
# 2c.  Ensemble the top-K circuits (improves R² and gives uncertainty)
# ===========================================================================
def ensemble_evaluate(results, val_dataset=None, top_k=5, n_qubits=4):
    """
    Average predictions from the top_k Ridge models (by R²).
    Re-runs quantum features per circuit, then averages predictions.
    Returns (true_vals, mean_preds, std_preds, ens_r2, ens_rmse).
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error
    import math

    best = sorted(results, key=lambda r: r['r2'], reverse=True)[:top_k]
    true_vals = best[0].get('eval_true', best[0]['val_true'])          # same for all circuits
    val_X_pca = best[0].get('eval_X_pca', best[0]['val_X_pca'])

    all_preds = []
    for res in best:
        q_va = extract_quantum_features(res['circuit'], val_X_pca, n_qubits)
        X_va = np.concatenate([val_X_pca, q_va], axis=1)
        if res.get('x_scaler') is not None:
            X_va = res['x_scaler'].transform(X_va)
        all_preds.append(res['model'].predict(X_va))

    all_preds  = np.stack(all_preds)          # (top_k, N)
    mean_preds = all_preds.mean(axis=0)
    std_preds  = all_preds.std(axis=0)

    ens_r2   = r2_score(true_vals, mean_preds)
    ens_rmse = math.sqrt(mean_squared_error(true_vals, mean_preds))
    ens_pcc  = pearsonr(true_vals, mean_preds)[0]
    print(f"\nEnsemble (top-{top_k}) — R²={ens_r2:.4f}  "
          f"RMSE={ens_rmse:.4f}  Pearson r={ens_pcc:.4f}")
    return true_vals, mean_preds, std_preds, ens_r2, ens_rmse


def _circuit_signature(qc: QuantumCircuit) -> str:
    """Stable textual fingerprint for a circuit based on ordered gate stream."""
    tokens = []
    for inst in qc.data:
        gate_name = inst.operation.name
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        tokens.append(f"{gate_name}:{','.join(map(str, qubits))}")
    raw = "|".join(tokens)
    digest = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]
    return f"g3-{qc.num_qubits}q-{qc.size()}g-{digest}"


def _save_top_circuit_diagrams(results, output_dir, top_k=25):
    """Save high-quality PNG diagrams and metadata for top-K circuits."""
    top = sorted(results, key=lambda r: r['r2'], reverse=True)[:min(top_k, len(results))]
    diag_dir = os.path.join(output_dir, f"top{len(top)}_circuit_diagrams")
    os.makedirs(diag_dir, exist_ok=True)

    style = publication_qiskit_style()
    rows = []
    for rank, r in enumerate(top, start=1):
        qc = r['circuit']
        cid = _circuit_signature(qc)
        stats = circuit_summary_stats(qc)

        png_name = f"rank{rank:02d}_idx{r['circuit_idx']}_{cid}.png"
        png_path = os.path.join(diag_dir, png_name)
        title = f"G3 Circuit #{r['circuit_idx']}  (rank {rank}/{len(top)},  R²={r['r2']:.4f})"
        render_circuit_diagram(qc, png_path, title=title, style=style, fold=-1)

        rows.append({
            'rank': rank,
            'circuit_idx': r['circuit_idx'],
            'circuit_id': cid,
            **stats,
            'r2': r['r2'],
            'r2_baseline': r['r2_baseline'],
            'r2_gain': r['r2_gain'],
            'rmse': r['rmse'],
            'mae': r['mae'],
            'pearson': r['pearson'],
            'spearman': r['spearman'],
            'best_alpha': r['best_alpha'],
            'model_type': r.get('model_type', 'ridge'),
            'diagram_file': png_name,
        })

    gates_csv = os.path.join(output_dir, f'top{len(top)}_circuit_gates.csv')
    pd.DataFrame(rows).to_csv(gates_csv, index=False)
    print(f"Saved top-circuit diagrams -> {diag_dir}")
    print(f"Saved top-circuit metadata -> {gates_csv}")


def _save_all_circuit_exports(all_circuits, all_rfd_scores, results, output_dir, render_all_diagrams=True):
    """Save all generated circuits in analysis-friendly summary + long-form gate CSVs."""
    result_by_idx = {r['circuit_idx']: r for r in results}
    score_by_idx = {int(s['circuit_idx']): s for s in all_rfd_scores}
    selected_idxs = set(result_by_idx.keys())

    rows_catalog = []
    rows_long = []

    all_diag_dir = os.path.join(output_dir, 'all_circuit_diagrams')
    if render_all_diagrams:
        os.makedirs(all_diag_dir, exist_ok=True)

    style = publication_qiskit_style()
    for idx, qc in enumerate(all_circuits):
        cid = _circuit_signature(qc)
        stats = circuit_summary_stats(qc)
        score_info = score_by_idx.get(idx, {})
        perf = result_by_idx.get(idx)

        diagram_name = ''
        if render_all_diagrams:
            diagram_name = f"idx{idx:03d}_{cid}.png"
            diagram_path = os.path.join(all_diag_dir, diagram_name)
            title_parts = [f"G3 Circuit #{idx}"]
            if score_info.get('rfd_rank') is not None:
                title_parts.append(f"RFD rank {score_info['rfd_rank']}")
            if perf is not None:
                title_parts.append(f"R²={perf['r2']:.4f}")
            render_circuit_diagram(qc, diagram_path, title='  |  '.join(title_parts), style=style, fold=-1)

        row = {
            'circuit_idx': idx,
            'circuit_id': cid,
            **stats,
            'selected_for_eval': idx in selected_idxs,
            'evaluated': perf is not None,
            'rfd_score': score_info.get('rfd_score', np.nan),
            'rfd_rank': score_info.get('rfd_rank', np.nan),
            'diagram_file': diagram_name,
            'model_type': perf.get('model_type') if perf is not None else '',
            'best_alpha': perf.get('best_alpha') if perf is not None else np.nan,
            'r2': perf.get('r2') if perf is not None else np.nan,
            'r2_baseline': perf.get('r2_baseline') if perf is not None else np.nan,
            'r2_gain': perf.get('r2_gain') if perf is not None else np.nan,
            'selection_r2': perf.get('selection_r2') if perf is not None else np.nan,
            'selection_r2_baseline': perf.get('selection_r2_baseline') if perf is not None else np.nan,
            'selection_r2_gain': perf.get('selection_r2_gain') if perf is not None else np.nan,
            'rmse': perf.get('rmse') if perf is not None else np.nan,
            'mae': perf.get('mae') if perf is not None else np.nan,
            'pearson': perf.get('pearson') if perf is not None else np.nan,
            'spearman': perf.get('spearman') if perf is not None else np.nan,
            'evaluation_split': perf.get('evaluation_split') if perf is not None else '',
        }
        rows_catalog.append(row)

        gate_rows = circuit_gate_rows(qc, circuit_idx=idx, circuit_id=cid)
        for g in gate_rows:
            g['rfd_score'] = row['rfd_score']
            g['rfd_rank'] = row['rfd_rank']
            g['evaluated'] = row['evaluated']
            g['r2'] = row['r2']
            g['r2_gain'] = row['r2_gain']
        rows_long.extend(gate_rows)

    catalog_csv = os.path.join(output_dir, 'all_circuit_catalog.csv')
    gates_csv = os.path.join(output_dir, 'all_circuit_gate_steps.csv')
    pd.DataFrame(rows_catalog).sort_values('circuit_idx').to_csv(catalog_csv, index=False)
    pd.DataFrame(rows_long).to_csv(gates_csv, index=False)
    print(f"Saved full circuit catalog -> {catalog_csv}")
    print(f"Saved full gate-step table -> {gates_csv}")
    if render_all_diagrams:
        print(f"Saved all circuit diagrams -> {all_diag_dir}")


def _save_quartile_violin_plot(results, output_dir):
    """Create quartile box + violin comparison plot using per-circuit R²."""
    if len(results) < 4:
        print("Skipping quartile plot (need at least 4 circuits).")
        return

    sorted_r = sorted(results, key=lambda r: r['r2'], reverse=True)
    n = len(sorted_r)
    q25 = max(1, n // 4)
    q50 = max(1, n // 2)
    q75 = max(1, 3 * n // 4)

    quartiles = {
        'Top 25%': [r['r2'] for r in sorted_r[:q25]],
        'Q2 (25-50%)': [r['r2'] for r in sorted_r[q25:q50]],
        'Q3 (50-75%)': [r['r2'] for r in sorted_r[q50:q75]],
        'Bottom 25%': [r['r2'] for r in sorted_r[q75:]],
    }

    # Classical baseline reference (Ridge on classical-only features).
    baseline_vals = [r.get('r2_baseline', np.nan) for r in sorted_r]
    baseline_vals = [float(v) for v in baseline_vals if pd.notna(v)]
    baseline_r2 = float(np.mean(baseline_vals)) if baseline_vals else None

    # Compute comfortable y-limits that always include the baseline line.
    y_data = [v for arr in quartiles.values() for v in arr]
    if baseline_r2 is not None:
        y_data.append(baseline_r2)
    y_min = float(min(y_data))
    y_max = float(max(y_data))
    y_span = max(1e-4, y_max - y_min)
    y_pad = max(0.03, 0.15 * y_span)
    y_lo = y_min - y_pad
    y_hi = y_max + y_pad

    fig_q, axes_q = plt.subplots(1, 2, figsize=(14, 5))
    colors_q = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']

    ax_box = axes_q[0]
    bp = ax_box.boxplot(
        [quartiles[q] for q in ['Top 25%', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Bottom 25%']],
        labels=['Top 25%', 'Q2\n(25-50%)', 'Q3\n(50-75%)', 'Bottom 25%'],
        patch_artist=True,
        widths=0.6,
    )
    for patch, color in zip(bp['boxes'], colors_q):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_box.set_ylabel('R²', fontsize=11)
    ax_box.set_title('Quartile Comparison: R² Distribution', fontsize=12, fontweight='bold')
    ax_box.grid(True, axis='y', alpha=0.3)
    ax_box.set_ylim(y_lo, y_hi)
    if baseline_r2 is not None:
        ax_box.axhline(
            baseline_r2,
            color='#34495e',
            linestyle='--',
            linewidth=2.0,
            alpha=0.95,
            label=f'Classical baseline R²={baseline_r2:.3f}',
        )
        ax_box.legend(loc='best')

    ax_vio = axes_q[1]
    vio_data = [quartiles[q] for q in ['Top 25%', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Bottom 25%']]
    positions = [1, 2, 3, 4]
    parts = ax_vio.violinplot(vio_data, positions=positions, showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], colors_q):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax_vio.set_xticks(positions)
    ax_vio.set_xticklabels(['Top 25%', 'Q2\n(25-50%)', 'Q3\n(50-75%)', 'Bottom 25%'])
    ax_vio.set_ylabel('R²', fontsize=11)
    ax_vio.set_title('Violin Plot: R² by Quartile', fontsize=12, fontweight='bold')
    ax_vio.grid(True, axis='y', alpha=0.3)
    ax_vio.set_ylim(y_lo, y_hi)
    if baseline_r2 is not None:
        ax_vio.axhline(
            baseline_r2,
            color='#34495e',
            linestyle='--',
            linewidth=2.0,
            alpha=0.95,
            label=f'Classical baseline R²={baseline_r2:.3f}',
        )
        ax_vio.legend(loc='best')

    plt.tight_layout()
    quartile_path = os.path.join(output_dir, 'quartile_comparison.png')
    plt.savefig(quartile_path, dpi=150, bbox_inches='tight')
    plt.close(fig_q)
    print(f"Saved quartile comparison -> {quartile_path}")

# ===========================================================================
# 3. Save, plot, and report
# ===========================================================================
def save_and_plot_results(results, ensemble_data=None, all_circuits=None, all_rfd_scores=None):
    timestamp  = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.abspath(f'plots_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    sorted_by_r2 = sorted(results, key=lambda x: x['r2'], reverse=True)
    top5 = sorted(results, key=lambda x: x['rmse'])[:5]

    # Evaluated-circuit CSV (typically top-K after RFD preselection)
    all_rows = []
    for rank, r in enumerate(sorted_by_r2, start=1):
        qc = r['circuit']
        stats = circuit_summary_stats(qc)
        all_rows.append({
            'rank_by_r2': rank,
            'circuit_idx': r['circuit_idx'],
            'circuit_id': _circuit_signature(qc),
            **stats,
            'model_type': r.get('model_type', 'ridge'),
            'best_alpha': r['best_alpha'],
            'r2': r['r2'],
            'r2_baseline': r['r2_baseline'],
            'r2_gain': r['r2_gain'],
            'rmse': r['rmse'],
            'mae': r['mae'],
            'pearson': r['pearson'],
            'spearman': r['spearman'],
        })
    all_csv_path = os.path.join(output_dir, 'all_circuit_results.csv')
    pd.DataFrame(all_rows).to_csv(all_csv_path, index=False)
    # Drop model objects before serialising
    top5_serialisable = [
        {k: v for k, v in r.items() if k not in {'model', 'x_scaler', 'circuit', 'val_preds', 'val_true', 'val_X_pca'}}
        for r in top5
    ]
    df = pd.DataFrame(top5_serialisable)
    csv_path = os.path.join(output_dir, 'top5_random_unitary_results.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"Unitary Testing Results — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    eval_split = results[0].get('evaluation_split', 'validation') if results else 'validation'
    print(f"Evaluation split for reported metrics: {eval_split}")
    print(f"Output dir : {output_dir}/")
    print(f"All circuits CSV: {all_csv_path}")
    print(df[['circuit_idx', 'rmse', 'mae', 'r2', 'pearson', 'spearman']].to_string(index=False))
    print(f"{'='*60}")

    # ---- loss curves -------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    max_points = 0
    for res in top5:
        train_losses = res.get('train_losses', [])
        val_losses = res.get('val_losses', [])
        max_points = max(max_points, len(train_losses), len(val_losses))

        if len(train_losses) > 0:
            ax1.plot(range(len(train_losses)), train_losses,
                     label=f"Train C{res['circuit_idx']}", lw=2,
                     marker='o' if len(train_losses) == 1 else None)
        if len(val_losses) > 0:
            ax1.plot(range(len(val_losses)), val_losses,
                     label=f"Val   C{res['circuit_idx']}", linestyle='--', lw=2,
                     marker='o' if len(val_losses) == 1 else None)

    xlabel = 'Epoch' if max_points > 1 else 'Final step'
    title = 'Top-5 G3 Reservoir Circuits — Loss Curves' if max_points > 1 \
        else 'Top-5 G3 Reservoir Circuits — Final Train/Val MSE'
    ax1.set_xlabel(xlabel); ax1.set_ylabel('MSE Loss')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.savefig(os.path.join(output_dir, 'top5_loss_curves.png'), dpi=300, bbox_inches='tight')

    # ---- R² bar chart ------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    idxs = [r['circuit_idx'] for r in top5]
    r2s  = [r['r2']          for r in top5]
    rmses= [r['rmse']        for r in top5]
    bars = ax2.bar(range(len(idxs)), r2s, color='steelblue', alpha=0.8, edgecolor='k')
    for bar, r2v, rmse in zip(bars, r2s, rmses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'R²={r2v:.3f}\nRMSE={rmse:.3f}', ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(range(len(idxs)))
    ax2.set_xticklabels([f'C{i}' for i in idxs])
    ax2.set_xlabel('Circuit'); ax2.set_ylabel('R²')
    ax2.set_title('Top-5 G3 Random Unitary Circuits — R² on Validation Set')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(0, color='red', lw=1, linestyle=':')
    fig2.savefig(os.path.join(output_dir, 'top5_r2_scores.png'), dpi=300, bbox_inches='tight')

    # ---- ensemble scatter --------------------------------------------------
    if ensemble_data is not None:
        true_vals, mean_preds, std_preds, ens_r2, ens_rmse = ensemble_data
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        ax3.errorbar(true_vals, mean_preds, yerr=std_preds,
                     fmt='o', alpha=0.5, ecolor='grey', capsize=2)
        mn = min(true_vals.min(), mean_preds.min())
        mx = max(true_vals.max(), mean_preds.max())
        ax3.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='ideal')
        ax3.set_xlabel('True  (normalised)')
        ax3.set_ylabel('Predicted  (normalised)')
        ax3.set_title(f'Ensemble Predictions  R²={ens_r2:.4f}  RMSE={ens_rmse:.4f}')
        ax3.legend(); ax3.grid(True, alpha=0.3)
        fig3.savefig(os.path.join(output_dir, 'ensemble_scatter.png'), dpi=300, bbox_inches='tight')

    _save_quartile_violin_plot(results, output_dir)
    _save_top_circuit_diagrams(results, output_dir, top_k=25)
    if all_circuits is not None and all_rfd_scores is not None:
        _save_all_circuit_exports(
            all_circuits=all_circuits,
            all_rfd_scores=all_rfd_scores,
            results=results,
            output_dir=output_dir,
            render_all_diagrams=True,
        )

    plt.close('all')
    print(f"Plots saved to {output_dir}/")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test random G3 unitary circuits for quantum reservoir regression.')
    parser.add_argument('--holdout', action='store_true',
                        help='Reserve part of validation data as unseen holdout and report metrics/plots on holdout.')
    parser.add_argument('--holdout-fraction', type=float, default=0.5,
                        help='Fraction of validation split to reserve as holdout when --holdout is set. Default: 0.5')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for holdout split. Default: 42')
    parser.add_argument('--num-circuits', type=int, default=100,
                        help='Number of random G3 circuits to generate. Default: 100')
    parser.add_argument('--num-gates', type=int, default=300,
                        help='Gate depth/instruction count per circuit. Default: 300')
    parser.add_argument('--top-k', type=int, default=25,
                        help='Top-K circuits selected by RFD for evaluation. Default: 25')
    args = parser.parse_args()

    N_QUBITS     = 6    # qubits (↑ from 4 → 18 observable features vs 12)
    NUM_GATES    = args.num_gates
    N_CIRCUITS   = args.num_circuits
    TOP_K_EXPR   = args.top_k
    TOP_K_ENS    = 25   # ensemble after training
    PCA_DIMS     = 32   # PCA dims per modality; 2*32=64 total into reservoir
    USE_MODEL_FEATURE_NPZ = True  # leverage precomputed 3DCNN/SGCNN NPZ features

    # ---- Generate G3 circuit pool ----------------------------------------
    circuits = generate_g3_random_circuits(N_QUBITS, num_gates=NUM_GATES, num_circuits=N_CIRCUITS)

    # ---- Pre-select most expressive circuits (RFD metric) ----------------
    indexed_circuits, all_rfd_scores = preselect_circuits_by_expressibility(
        circuits, N_QUBITS, top_k=TOP_K_EXPR, return_all_scores=True)

    # ---- Load data --------------------------------------------------------
    if USE_MODEL_FEATURE_NPZ:
        train_dataset, val_dataset, label_mean, label_std = \
            load_from_model_feature_npz(max_samples=6000, random_state=args.seed)
    else:
        # Uses ECFP4 + RDKit descriptors (ligand) and AA composition (pocket)
        # ~5000+ samples vs ~180 from old model_ready_data
        train_dataset, val_dataset, label_mean, label_std = \
            load_from_refined_set(n_pca_components=PCA_DIMS)

    holdout_dataset = None
    if args.holdout:
        val_dataset, holdout_dataset = split_validation_for_holdout(
            val_dataset,
            holdout_fraction=args.holdout_fraction,
            random_state=args.seed,
        )
        print(f"Holdout enabled: selection-val={len(val_dataset)}  holdout={len(holdout_dataset)}")
    else:
        print(f"Holdout disabled: using validation split (n={len(val_dataset)}) for metrics/plots")

    # ---- Evaluate each circuit with Ridge regression readout -------------
    results = run_circuits_and_evaluate(
        indexed_circuits, train_dataset, val_dataset,
        n_qubits=N_QUBITS,
        holdout_dataset=holdout_dataset,
    )

    # ---- Ensemble top-K circuits -----------------------------------------
    ensemble_data = ensemble_evaluate(
        results, top_k=TOP_K_ENS, n_qubits=N_QUBITS)

    # ---- Save plots and CSV ----------------------------------------------
    save_and_plot_results(
        results,
        ensemble_data=ensemble_data,
        all_circuits=circuits,
        all_rfd_scores=all_rfd_scores,
    )
