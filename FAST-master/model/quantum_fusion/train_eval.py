"""Training / evaluation helpers for param-vs-fixed quantum fusion studies."""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

_QF_DIR = os.path.dirname(os.path.abspath(__file__))
if _QF_DIR not in sys.path:
    sys.path.insert(0, _QF_DIR)

from main_train import (  # noqa: E402
    FusionDataset,
    ModelHybridFC_Reservoir,
    ModelHybridFC_VQC,
    evaluate_model,
)

_DATA_SEED = 42


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    holdout_loader: DataLoader
    in_features: int
    label_mean: float
    label_std: float
    n_train: int
    n_val: int
    n_holdout: int
    data_source: str


def _load_index_labels(refined_root: str) -> Dict[str, float]:
    index_path = os.path.join(refined_root, "index", "INDEX_refined_data.2020")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"INDEX not found: {index_path}")
    out: Dict[str, float] = {}
    with open(index_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    out[parts[0].lower()] = float(parts[3])
                except ValueError:
                    pass
    return out


def load_data_npz_index(
    dcnn_npz: Optional[str] = None,
    sgcnn_npz: Optional[str] = None,
    holdout_fraction: float = 0.20,
    val_fraction_of_pool: float = 0.20,
    data_seed: int = _DATA_SEED,
) -> Tuple[FusionDataset, FusionDataset, FusionDataset, float, float]:
    """
    Load [3DCNN|SGCNN] features from NPZ + labels from PDBbind INDEX.

    Split (when HDF embed split files are absent):
      80% train pool -> 80/20 quantum-train / selection-val
      20% holdout (approximation of gradient-clean val)
    """
    dcnn_npz = dcnn_npz or os.path.join(_QF_DIR, "refined_3dcnn_features.npz")
    sgcnn_npz = sgcnn_npz or os.path.join(_QF_DIR, "refined_sgcnn_features.npz")

    candidates = [
        os.path.join(_QF_DIR, "..", "..", "..", "data", "refined-set"),
        os.path.join(_QF_DIR, "..", "..", "..", "..", "data", "refined-set"),
    ]
    refined_root = next(
        (os.path.abspath(p) for p in candidates if os.path.isdir(os.path.abspath(p))),
        None,
    )
    if refined_root is None:
        raise FileNotFoundError("Cannot locate data/refined-set for INDEX labels")

    pdb_to_label = _load_index_labels(refined_root)
    npz_3d = np.load(dcnn_npz, allow_pickle=False)
    npz_sg = np.load(sgcnn_npz, allow_pickle=False)
    valid = sorted(set(npz_3d.files) & set(npz_sg.files) & set(pdb_to_label))
    if len(valid) < 100:
        raise RuntimeError(f"Too few labelled NPZ complexes: {len(valid)}")

    feat_3d = np.stack([npz_3d[pid] for pid in valid], axis=0).astype(np.float32)
    feat_sg = np.stack([npz_sg[pid] for pid in valid], axis=0).astype(np.float32)
    labels = np.array([pdb_to_label[pid] for pid in valid], dtype=np.float32)

    n_bins = min(10, max(2, int(np.sqrt(len(valid)))))
    bins = pd.qcut(labels, q=n_bins, labels=False, duplicates="drop")

    idx = np.arange(len(valid))
    pool_idx, ho_idx = train_test_split(
        idx,
        test_size=holdout_fraction,
        random_state=data_seed,
        shuffle=True,
        stratify=bins,
    )
    pool_bins = bins[pool_idx]
    tr_local, sel_local = train_test_split(
        pool_idx,
        test_size=val_fraction_of_pool,
        random_state=data_seed,
        shuffle=True,
        stratify=pool_bins,
    )

    sc_3d = StandardScaler().fit(feat_3d[tr_local])
    sc_sg = StandardScaler().fit(feat_sg[tr_local])
    feat_3d = sc_3d.transform(feat_3d).astype(np.float32)
    feat_sg = sc_sg.transform(feat_sg).astype(np.float32)
    features = np.hstack([feat_3d, feat_sg])

    label_mean = float(labels[tr_local].mean())
    label_std = float(labels[tr_local].std()) + 1e-8
    labels_norm = (labels - label_mean) / label_std

    n = len(valid)
    empty = np.zeros((n, 0), dtype=np.float32)

    def _ds(idxs):
        return FusionDataset(
            torch.tensor(features[idxs], dtype=torch.float32),
            torch.tensor(empty[idxs], dtype=torch.float32),
            torch.tensor(labels_norm[idxs], dtype=torch.float32),
        )

    return (
        _ds(tr_local),
        _ds(sel_local),
        _ds(ho_idx),
        label_mean,
        label_std,
    )


def load_dataloaders(
    batch_size: int = 64,
    data_seed: int = _DATA_SEED,
) -> DataBundle:
    """Prefer HDF embed split; fall back to NPZ+INDEX."""
    try:
        from gate_variance_study import load_data as _load_hdf

        train_ds, val_ds, holdout_ds = _load_hdf()
        source = "hdf_embed_split"
    except Exception as exc:
        print(f"[train_eval] HDF load failed ({exc}); using NPZ+INDEX fallback.")
        train_ds, val_ds, holdout_ds, _, _ = load_data_npz_index(data_seed=data_seed)
        source = "npz_index_split"

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "holdout": DataLoader(holdout_ds, batch_size=batch_size, shuffle=False),
    }
    sample = train_ds[0]
    in_features = int(sample[0].numel() + sample[1].numel())

    return DataBundle(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        holdout_loader=loaders["holdout"],
        in_features=in_features,
        label_mean=0.0,
        label_std=1.0,
        n_train=len(train_ds),
        n_val=len(val_ds),
        n_holdout=len(holdout_ds),
        data_source=source,
    )


def build_model(
    mode: str,
    in_features: int,
    qc,
    n_qubits: int,
    backend: str = "lightning.qubit",
) -> nn.Module:
    if mode == "fixed_e2e":
        return ModelHybridFC_Reservoir(
            in_features=in_features,
            out_features=1,
            qiskit_circuit=qc,
            n_qubits=n_qubits,
            backend=backend,
        )
    if mode == "param_vqc":
        # VQC trains ``quantum_params`` via backprop; lightning.qubit often rejects
        # backprop on mixed gate sets — default.qubit is reliable here.
        return ModelHybridFC_VQC(
            in_features=in_features,
            out_features=1,
            qiskit_circuit=qc,
            n_qubits=n_qubits,
            backend="default.qubit",
        )
    raise ValueError(f"Unknown mode {mode!r}; use fixed_e2e or param_vqc")


def train_model(
    model: nn.Module,
    loaders: DataBundle,
    *,
    epochs: int,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 8,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Train with early stopping on val RMSE; return best val + holdout metrics."""
    device = device or torch.device("cpu")
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=max(3, patience // 2), factor=0.5, min_lr=1e-6,
    )
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    t0 = time.time()

    epoch_iter = range(1, epochs + 1)
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc="epochs", leave=False)

    for epoch in epoch_iter:
        model.train()
        train_loss = 0.0
        for sg, c3, y in loaders.train_loader:
            x = torch.cat([sg, c3], dim=1).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(loaders.train_loader))

        val_rmse, val_mae, val_r2, val_pearson, _ = evaluate_model(
            model, loaders.val_loader,
        )
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    ho_rmse, ho_mae, ho_r2, ho_pearson, ho_spearman = evaluate_model(
        model, loaders.holdout_loader,
    )
    _, _, val_r2, _, _ = evaluate_model(model, loaders.val_loader)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_quantum = 0
    if hasattr(model, "quantum_params"):
        n_quantum = int(model.quantum_params.numel())

    return {
        "r2_holdout": float(ho_r2),
        "rmse_holdout": float(ho_rmse),
        "mae_holdout": float(ho_mae),
        "pearson_holdout": float(ho_pearson),
        "spearman_holdout": float(ho_spearman),
        "r2_val": float(val_r2),
        "best_val_rmse": float(best_val_rmse),
        "train_time_s": time.time() - t0,
        "epochs_run": float(epochs),
        "n_trainable_total": float(n_trainable),
        "n_quantum_params": float(n_quantum),
    }
