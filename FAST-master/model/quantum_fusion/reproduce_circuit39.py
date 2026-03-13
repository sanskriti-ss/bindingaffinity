#!/usr/bin/env python
"""
reproduce_circuit39.py

Deterministically reproduces Circuit #39: the best G3 random unitary
circuit (R²=0.8798, Adj-R²=0.8459, Pearson r=0.9382) from the gate
sequence stored in top25_circuit_gates.csv.

Outputs (written next to this script):
  circuit39_diagram.png   — Qiskit matplotlib circuit diagram
  circuit39_scatter.png   — Predicted vs actual pKi scatter plot

Run from the quantum_fusion/ directory:
    python reproduce_circuit39.py
"""

import os, sys, math
os.environ.setdefault('PYTHONUTF8', '1')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, TGate, CXGate

import h5py
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_train import (
    FusionDataset,
    ModelHybridFC_Reservoir,
    evaluate_model,
)


def load_from_npz_hdf(hdf_path: str, dcnn_npz: str, sgcnn_npz: str):
    """
    Load features and labels directly from refined_all.hdf + NPZ files,
    bypassing load_sample_data() (which requires physical SDF/PDB files).

    Returns:
        features : (N, 64)  float32  — 3DCNN(10) + SGCNN(54), StandardScaled
        labels   : (N,)     float32  — binding affinity (-logK)
        ids      : (N,)     str      — PDB IDs
    """
    # ── labels from HDF ──────────────────────────────────────────────────────
    with h5py.File(hdf_path, 'r') as hf:
        hdf_ids    = list(hf.keys())
        hdf_labels = {pid: float(hf[pid].attrs['affinity']) for pid in hdf_ids}

    # ── load NPZ files ────────────────────────────────────────────────────────
    npz_3d  = np.load(dcnn_npz,  allow_pickle=False)
    npz_sg  = np.load(sgcnn_npz, allow_pickle=False)

    # ── intersect IDs present in all three sources ────────────────────────────
    valid = sorted(set(hdf_labels) & set(npz_3d.files) & set(npz_sg.files))
    print(f"  HDF: {len(hdf_ids)}   3DCNN NPZ: {len(npz_3d.files)}   "
          f"SGCNN NPZ: {len(npz_sg.files)}   Intersection: {len(valid)}")

    feat_3d  = np.stack([npz_3d[pid] for pid in valid],  axis=0).astype(np.float32)
    feat_sg  = np.stack([npz_sg[pid] for pid in valid],  axis=0).astype(np.float32)
    labels   = np.array([hdf_labels[pid] for pid in valid], dtype=np.float32)
    ids      = np.array(valid)

    # ── scale each block independently ───────────────────────────────────────
    feat_3d = StandardScaler().fit_transform(feat_3d).astype(np.float32)
    feat_sg = StandardScaler().fit_transform(feat_sg).astype(np.float32)

    features = np.hstack([feat_3d, feat_sg])          # (N, 64)
    return features, labels, ids

# ── Config (must match evaluate_top25.py) ────────────────────────────────────
N_QUBITS       = 6
EPOCHS         = 50
BATCH_SIZE     = 64
LR             = 3e-4
EARLY_STOP_PAT = 15
DEVICE         = torch.device('cpu')
CIRCUIT_INDEX  = 39

_qf_dir    = os.path.dirname(os.path.abspath(__file__))
_dcnn_npz  = os.path.join(_qf_dir, 'refined_3dcnn_features.npz')
_sgcnn_npz = os.path.join(_qf_dir, 'refined_sgcnn_features.npz')
_hdf_path  = os.path.join(_qf_dir, 'refined_all.hdf')
_gates_csv = os.path.join(_qf_dir, 'top25_circuit_gates.csv')


# ── 1. Rebuild circuit from stored gate sequence ─────────────────────────────
def build_circuit_from_csv(gates_csv: str, circuit_index: int) -> QuantumCircuit:
    """Reconstruct a Qiskit QuantumCircuit from the gate-sequence CSV."""
    df = pd.read_csv(gates_csv)
    rows = df[df['circuit_index'] == circuit_index].sort_values('step')

    if rows.empty:
        raise ValueError(f"Circuit #{circuit_index} not found in {gates_csv}")

    qc = QuantumCircuit(N_QUBITS)
    for _, row in rows.iterrows():
        gate   = row['gate']
        qubits = [int(q) for q in str(row['qubits']).split(',')]
        if gate == 'h':
            qc.h(qubits[0])
        elif gate == 't':
            qc.t(qubits[0])
        elif gate == 'cx':
            qc.cx(qubits[0], qubits[1])
        else:
            raise ValueError(f"Unknown gate '{gate}' at step {row['step']}")

    return qc


# ── 2. Training helpers ──────────────────────────────────────────────────────
def _train_one(model, loaders):
    optimizer  = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-6)
    criterion  = nn.MSELoss()
    best_val   = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for sg, c3, y in loaders['train']:
            x = torch.cat([sg, c3], dim=1).to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        rmse, *_ = evaluate_model(model, loaders['val'])
        scheduler.step(rmse)
        if rmse < best_val:
            best_val   = rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PAT:
                print(f"    Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


def _get_predictions(model, loader):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for sg, c3, y in loader:
            x   = torch.cat([sg, c3], dim=1).to(DEVICE)
            out = model(x)
            preds.extend(out.cpu().numpy().flatten())
            labs.extend(y.cpu().numpy().flatten())
    return np.array(preds), np.array(labs)


# ── 3. Main ──────────────────────────────────────────────────────────────────
def main():
    # 3a. Rebuild & diagram the circuit
    print(f"Rebuilding Circuit #{CIRCUIT_INDEX} from {_gates_csv} ...")
    circuit = build_circuit_from_csv(_gates_csv, CIRCUIT_INDEX)
    print(f"  {circuit.num_qubits}-qubit circuit, {len(circuit.data)} gates")

    print("Drawing circuit diagram ...")
    fig_c = circuit.draw('mpl', fold=-1, style={'backgroundcolor': '#FFFFFF'})
    fig_c.suptitle(
        f"G3 Circuit #{CIRCUIT_INDEX}  —  Best Quantum Reservoir Circuit\n"
        f"R²=0.8798   Adj-R²=0.8459   Pearson r=0.9382   RMSE=0.6159 pKi",
        fontsize=11, fontweight='bold', y=1.02,
    )
    diag_path = os.path.join(_qf_dir, f'circuit{CIRCUIT_INDEX}_diagram.png')
    fig_c.savefig(diag_path, dpi=200, bbox_inches='tight')
    plt.close(fig_c)
    print(f"  Saved -> {diag_path}")

    # 3b. Load data
    print("\nLoading features ...")
    features, labels, _ = load_from_npz_hdf(_hdf_path, _dcnn_npz, _sgcnn_npz)

    n_samples = len(labels)
    n_train   = int(0.70 * n_samples)
    n_val     = int(0.15 * n_samples)
    train_idx = np.arange(0, n_train)
    val_idx   = np.arange(n_train, n_train + n_val)
    test_idx  = np.arange(n_train + n_val, n_samples)

    label_mean = float(labels[train_idx].mean())
    label_std  = float(labels[train_idx].std()) + 1e-8
    print(f"  Label stats: mean={label_mean:.3f}  std={label_std:.3f}")
    print(f"  Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    def _norm(y): return (y - label_mean) / label_std

    empty = np.zeros((n_samples, 0), dtype=np.float32)

    loaders = {}
    for split, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        ds = FusionDataset(
            features[idx], empty[idx], _norm(labels[idx])
        )
        loaders[split] = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=(split == 'train')
        )

    dims = features.shape[1]
    print(f"  Input dim: {dims}")

    # 3c. Build & train the model
    print(f"\nTraining ModelHybridFC_Reservoir with Circuit #{CIRCUIT_INDEX} ...")
    model = ModelHybridFC_Reservoir(
        in_features=dims,
        out_features=1,
        qiskit_circuit=circuit,
        n_qubits=N_QUBITS,
        backend='lightning.qubit',
    ).to(DEVICE)

    model = _train_one(model, loaders)

    # 3d. Evaluate on test set
    preds_norm, labs_norm = _get_predictions(model, loaders['test'])
    preds_pki = preds_norm * label_std + label_mean
    labs_pki  = labs_norm  * label_std + label_mean

    r2       = r2_score(labs_norm, preds_norm)
    adj_r2   = 1 - (1 - r2) * (len(labs_norm) - 1) / (len(labs_norm) - dims - 1)
    pear     = pearsonr(labs_norm, preds_norm)[0]
    spear    = spearmanr(labs_norm, preds_norm)[0]
    rmse_pki = math.sqrt(mean_squared_error(labs_pki, preds_pki))
    mae_pki  = mean_absolute_error(labs_pki, preds_pki)

    print(f"\n  Circuit #{CIRCUIT_INDEX} reproduced results:")
    print(f"    R²         = {r2:.4f}")
    print(f"    Adj-R²     = {adj_r2:.4f}")
    print(f"    Pearson r  = {pear:.4f}")
    print(f"    Spearman r = {spear:.4f}")
    print(f"    RMSE       = {rmse_pki:.4f} pKi")
    print(f"    MAE        = {mae_pki:.4f} pKi")

    # 3e. Scatter plot
    print("\nGenerating scatter plot ...")
    vmin = min(labs_pki.min(), preds_pki.min()) - 0.3
    vmax = max(labs_pki.max(), preds_pki.max()) + 0.3

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(labs_pki, preds_pki, alpha=0.5, s=20,
                    c=np.abs(preds_pki - labs_pki),
                    cmap='plasma', edgecolors='none')
    plt.colorbar(sc, ax=ax, label='|Error| (pKi)')
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1.8, label='y = x  (ideal)')
    ax.set_xlabel('Experimental  -log K  (pKi)', fontsize=12)
    ax.set_ylabel('Predicted  -log K  (pKi)',    fontsize=12)
    ax.set_title(
        f'Circuit #{CIRCUIT_INDEX} — Binding Affinity Prediction  (Test set, n={len(labs_pki)})\n'
        f'R²={r2:.4f}   Adj-R²={adj_r2:.4f}   Pearson r={pear:.4f}   RMSE={rmse_pki:.4f} pKi',
        fontsize=10,
    )
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    scatter_path = os.path.join(_qf_dir, f'circuit{CIRCUIT_INDEX}_scatter.png')
    fig.savefig(scatter_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {scatter_path}")

    print(f"\nDone. Outputs written to:\n  {diag_path}\n  {scatter_path}")


if __name__ == '__main__':
    main()
