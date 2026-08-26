#!/usr/bin/env python
"""
quantum_intervention.py

Causal intervention experiments on the quantum layer of the hybrid
quantum-classical binding affinity model.

Phase A — Train & Save Checkpoints
Phase B — Intervention Evaluation (normal, shuffle_samples, mean, zero, matched_noise)
Phase C — Linear Probe (Ridge regression on pre/post-quantum representations)
Phase D — Statistical Analysis & Plots

Usage:
    # Full pipeline (train + intervene + analyze)
    python quantum_intervention.py --seeds 5 --shuffle-repeats 10 --noise-repeats 10

    # Intervention only (if checkpoints exist)
    python quantum_intervention.py --skip-training --checkpoint-dir results/quantum_intervention/checkpoints

    # Specific interventions
    python quantum_intervention.py --interventions normal shuffle_samples mean
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────────
_QF_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.dirname(_QF_DIR)
_TU_DIR = os.path.join(_MODEL_DIR, "testing_unitaries")
for _p in [_QF_DIR, _MODEL_DIR, _TU_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from main_train import (
    FusionDataset,
    ModelHybridFC_Reservoir,
    evaluate_model,
)
from train_eval import load_data_npz_index, DataBundle, train_model
from best_circuit_study import get_best_circuit
from testing_random_unitaries import generate_g3_random_circuits

# ── constants ─────────────────────────────────────────────────────────────────
BEST_GATE_COUNT = 100
BEST_N_QUBITS = 6
BEST_CIRCUIT_IDX = 27
DEFAULT_EPOCHS = 50
DEFAULT_LR = 3e-4
DEFAULT_BATCH_SIZE = 64
ALL_INTERVENTIONS = ["normal", "shuffle_samples", "mean", "zero", "matched_noise"]


# =============================================================================
# IntervenedModel — wraps ModelHybridFC_Reservoir with intervention hook
# =============================================================================
class IntervenedModel(nn.Module):
    """
    Wraps ``ModelHybridFC_Reservoir`` to intercept quantum output and apply
    causal interventions between the quantum reservoir and the skip connection.

    Intervention modes:
      - normal:          no change (baseline)
      - zero:            replace q_out with zeros
      - mean:            replace q_out with mean activation (from train set)
      - shuffle_samples: permute q_out across samples within the batch
      - matched_noise:   replace q_out with Gaussian noise matching train stats
    """

    def __init__(self, base_model: ModelHybridFC_Reservoir):
        super().__init__()
        self.base = base_model
        self.intervention_mode = "normal"
        self.ref_mean: Optional[torch.Tensor] = None
        self.ref_std: Optional[torch.Tensor] = None
        self.permutation_seed: Optional[int] = None
        self.noise_seed: Optional[int] = None

    def set_intervention(
        self,
        mode: str,
        ref_mean: Optional[torch.Tensor] = None,
        ref_std: Optional[torch.Tensor] = None,
        permutation_seed: Optional[int] = None,
        noise_seed: Optional[int] = None,
    ):
        self.intervention_mode = mode
        self.ref_mean = ref_mean
        self.ref_std = ref_std
        self.permutation_seed = permutation_seed
        self.noise_seed = noise_seed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        base = self.base

        # ── Classical preprocessing (same as ModelHybridFC_Reservoir) ──
        h = torch.relu(base.bn1(base.fc1(x)))
        x_enc = torch.tanh(base.fc2(h)) * math.pi  # [batch, n_qubits]

        # ── Quantum reservoir ──
        q_outputs = []
        for i in range(batch_size):
            q_out = base.quantum_reservoir(x_enc[i])
            q_outputs.append(torch.stack(q_out))
        q_out = torch.stack(q_outputs).float()  # [batch, 3*n_qubits]

        # ── Apply intervention ──
        q_out = self._apply_intervention(q_out, batch_size)

        # ── Skip connection + MLP head ──
        combined = torch.cat([q_out, x_enc], dim=1)
        return base.head(combined)

    def forward_with_cache(self, x: torch.Tensor):
        """Forward pass that also returns (x_enc, q_out) for analysis."""
        batch_size = x.shape[0]
        base = self.base

        h = torch.relu(base.bn1(base.fc1(x)))
        x_enc = torch.tanh(base.fc2(h)) * math.pi

        q_outputs = []
        for i in range(batch_size):
            q_out = base.quantum_reservoir(x_enc[i])
            q_outputs.append(torch.stack(q_out))
        q_out = torch.stack(q_outputs).float()

        combined = torch.cat([q_out, x_enc], dim=1)
        pred = base.head(combined)
        return pred, x_enc.detach(), q_out.detach()

    def _apply_intervention(self, q_out: torch.Tensor, batch_size: int) -> torch.Tensor:
        mode = self.intervention_mode

        if mode == "normal":
            return q_out

        if mode == "zero":
            return torch.zeros_like(q_out)

        if mode == "mean":
            if self.ref_mean is None:
                raise ValueError("ref_mean required for 'mean' intervention")
            return self.ref_mean.unsqueeze(0).expand(batch_size, -1).to(q_out.device)

        if mode == "shuffle_samples":
            rng = torch.Generator()
            if self.permutation_seed is not None:
                rng.manual_seed(self.permutation_seed)
            perm = torch.randperm(batch_size, generator=rng)
            return q_out[perm]

        if mode == "matched_noise":
            if self.ref_mean is None or self.ref_std is None:
                raise ValueError("ref_mean and ref_std required for 'matched_noise'")
            gen = torch.Generator()
            if self.noise_seed is not None:
                gen.manual_seed(self.noise_seed)
            noise = torch.randn(q_out.shape, generator=gen, device=q_out.device)
            return self.ref_mean.to(q_out.device) + noise * self.ref_std.to(q_out.device)

        raise ValueError(f"Unknown intervention mode: {mode}")


# =============================================================================
# Data loading helpers
# =============================================================================
def load_data(batch_size: int = DEFAULT_BATCH_SIZE, data_seed: int = 42) -> DataBundle:
    """Load data using the NPZ+INDEX pipeline from train_eval.py."""
    train_ds, val_ds, holdout_ds, label_mean, label_std = load_data_npz_index(
        data_seed=data_seed
    )

    sample = train_ds[0]
    in_features = int(sample[0].numel() + sample[1].numel())

    return DataBundle(
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        holdout_loader=DataLoader(holdout_ds, batch_size=batch_size, shuffle=False),
        in_features=in_features,
        label_mean=label_mean,
        label_std=label_std,
        n_train=len(train_ds),
        n_val=len(val_ds),
        n_holdout=len(holdout_ds),
        data_source="npz_index_split",
    )


def collect_all_tensors(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect all (X, y) from a DataLoader into single tensors."""
    xs, ys = [], []
    for sg, c3, y in loader:
        xs.append(torch.cat([sg, c3], dim=1))
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0).squeeze()


# =============================================================================
# Phase A — Train & Save Checkpoints
# =============================================================================
def phase_a_train(
    data: DataBundle,
    circuit,
    output_dir: str,
    n_seeds: int = 5,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
) -> List[str]:
    """Train models from different seeds and save checkpoints."""
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_paths = []
    for seed_idx in range(n_seeds):
        print(f"\n{'='*60}")
        print(f"Phase A — Training seed {seed_idx}/{n_seeds-1}")
        print(f"{'='*60}")

        torch.manual_seed(seed_idx)
        np.random.seed(seed_idx)

        model = ModelHybridFC_Reservoir(
            in_features=data.in_features,
            out_features=1,
            qiskit_circuit=circuit,
            n_qubits=BEST_N_QUBITS,
            backend="lightning.qubit",
        )
        device = torch.device("cpu")
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=8, factor=0.5, min_lr=1e-6
        )
        criterion = nn.MSELoss()

        best_val_rmse = float("inf")
        best_state = None

        for epoch in tqdm(range(1, epochs + 1), desc=f"Seed {seed_idx}", leave=False):
            model.train()
            train_loss = 0.0
            for sg, c3, y in data.train_loader:
                x = torch.cat([sg, c3], dim=1).to(device)
                y = y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(1, len(data.train_loader))

            val_rmse, _, val_r2, _, _ = evaluate_model(model, data.val_loader)
            scheduler.step(val_rmse)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 10 == 0 or epoch == 1:
                tqdm.write(
                    f"  Epoch {epoch:3d}: train_rmse={math.sqrt(train_loss):.4f}  "
                    f"val_rmse={val_rmse:.4f}  val_r2={val_r2:.4f}"
                )

        if best_state is not None:
            model.load_state_dict(best_state)

        ckpt_path = os.path.join(ckpt_dir, f"seed_{seed_idx}.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "seed": seed_idx,
                "best_val_rmse": best_val_rmse,
                "epochs": epochs,
                "in_features": data.in_features,
                "n_qubits": BEST_N_QUBITS,
            },
            ckpt_path,
        )
        checkpoint_paths.append(ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}  (val_rmse={best_val_rmse:.4f})")

    return checkpoint_paths


# =============================================================================
# Phase B — Intervention Evaluation
# =============================================================================
def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    return {
        "r2": float(r2_score(labels, preds)),
        "rmse": float(math.sqrt(mean_squared_error(labels, preds))),
        "mae": float(mean_absolute_error(labels, preds)),
        "pearson": float(pearsonr(labels, preds)[0]),
        "spearman": float(spearmanr(labels, preds)[0]),
        "n_samples": int(len(labels)),
    }


def collect_quantum_stats(
    model: IntervenedModel, loader: DataLoader
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect mean and std of quantum output over a dataset (for train set)."""
    model.eval()
    model.set_intervention("normal")
    all_q = []
    with torch.no_grad():
        for sg, c3, y in loader:
            x = torch.cat([sg, c3], dim=1)
            _, _, q_out = model.forward_with_cache(x)
            all_q.append(q_out)
    all_q = torch.cat(all_q, dim=0)
    return all_q.mean(dim=0), all_q.std(dim=0)


def run_intervention_eval(
    model: IntervenedModel,
    loader: DataLoader,
    mode: str,
    ref_mean: Optional[torch.Tensor] = None,
    ref_std: Optional[torch.Tensor] = None,
    repeat_seed: Optional[int] = None,
) -> Dict[str, float]:
    """Run one intervention mode and return metrics."""
    perm_seed = repeat_seed if mode == "shuffle_samples" else None
    noise_seed = repeat_seed if mode == "matched_noise" else None

    model.set_intervention(
        mode,
        ref_mean=ref_mean,
        ref_std=ref_std,
        permutation_seed=perm_seed,
        noise_seed=noise_seed,
    )
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for sg, c3, y in loader:
            x = torch.cat([sg, c3], dim=1)
            out = model(x)
            preds.extend(out.cpu().numpy().flatten())
            labels.extend(y.cpu().numpy().flatten())

    preds = np.array(preds)
    labels = np.array(labels)
    metrics = compute_metrics(preds, labels)
    metrics["intervention"] = mode
    metrics["repeat_seed"] = repeat_seed
    return metrics


def phase_b_interventions(
    data: DataBundle,
    circuit,
    checkpoint_paths: List[str],
    interventions: List[str],
    shuffle_repeats: int = 10,
    noise_repeats: int = 10,
) -> pd.DataFrame:
    """Run all interventions across all checkpoints."""
    rows = []

    for ckpt_idx, ckpt_path in enumerate(checkpoint_paths):
        print(f"\n{'='*60}")
        print(f"Phase B — Checkpoint {ckpt_idx}: {os.path.basename(ckpt_path)}")
        print(f"{'='*60}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        base_model = ModelHybridFC_Reservoir(
            in_features=ckpt["in_features"],
            out_features=1,
            qiskit_circuit=circuit,
            n_qubits=ckpt["n_qubits"],
            backend="lightning.qubit",
        )
        base_model.load_state_dict(ckpt["model_state_dict"])
        model = IntervenedModel(base_model)

        # Collect reference stats from TRAIN set only (no data leakage)
        print("  Computing train-set quantum activation stats...")
        ref_mean, ref_std = collect_quantum_stats(model, data.train_loader)

        for mode in interventions:
            if mode in ("shuffle_samples", "matched_noise"):
                n_repeats = shuffle_repeats if mode == "shuffle_samples" else noise_repeats
                for rep_seed in tqdm(
                    range(n_repeats), desc=f"  {mode}", leave=False
                ):
                    m = run_intervention_eval(
                        model, data.holdout_loader, mode,
                        ref_mean=ref_mean, ref_std=ref_std,
                        repeat_seed=rep_seed,
                    )
                    m["checkpoint"] = ckpt_idx
                    m["checkpoint_path"] = os.path.basename(ckpt_path)
                    rows.append(m)
            else:
                m = run_intervention_eval(
                    model, data.holdout_loader, mode,
                    ref_mean=ref_mean, ref_std=ref_std,
                )
                m["checkpoint"] = ckpt_idx
                m["checkpoint_path"] = os.path.basename(ckpt_path)
                rows.append(m)
                print(f"  {mode:20s}  R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}")

    return pd.DataFrame(rows)


# =============================================================================
# Phase C — Linear Probe
# =============================================================================
def phase_c_linear_probe(
    data: DataBundle,
    circuit,
    checkpoint_paths: List[str],
) -> pd.DataFrame:
    """Fit Ridge regression from pre-quantum and post-quantum representations."""
    rows = []

    for ckpt_idx, ckpt_path in enumerate(checkpoint_paths):
        print(f"\n{'='*60}")
        print(f"Phase C — Linear Probe: checkpoint {ckpt_idx}")
        print(f"{'='*60}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        base_model = ModelHybridFC_Reservoir(
            in_features=ckpt["in_features"],
            out_features=1,
            qiskit_circuit=circuit,
            n_qubits=ckpt["n_qubits"],
            backend="lightning.qubit",
        )
        base_model.load_state_dict(ckpt["model_state_dict"])
        model = IntervenedModel(base_model)
        model.eval()

        # Extract representations from train and holdout
        for split_name, loader in [
            ("train", data.train_loader),
            ("holdout", data.holdout_loader),
        ]:
            x_encs, q_outs, labels = [], [], []
            with torch.no_grad():
                for sg, c3, y in loader:
                    x = torch.cat([sg, c3], dim=1)
                    _, x_enc, q_out = model.forward_with_cache(x)
                    x_encs.append(x_enc.numpy())
                    q_outs.append(q_out.numpy())
                    labels.append(y.numpy().flatten())

            if split_name == "train":
                X_enc_tr = np.concatenate(x_encs)
                Q_out_tr = np.concatenate(q_outs)
                y_tr = np.concatenate(labels)
            else:
                X_enc_ho = np.concatenate(x_encs)
                Q_out_ho = np.concatenate(q_outs)
                y_ho = np.concatenate(labels)

        # Ridge: pre-quantum (x_enc) → affinity
        sc_enc = StandardScaler().fit(X_enc_tr)
        ridge_enc = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000], cv=5)
        ridge_enc.fit(sc_enc.transform(X_enc_tr), y_tr)
        preds_enc = ridge_enc.predict(sc_enc.transform(X_enc_ho))
        r2_enc = r2_score(y_ho, preds_enc)

        # Ridge: quantum output (q_out) → affinity
        sc_q = StandardScaler().fit(Q_out_tr)
        ridge_q = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000], cv=5)
        ridge_q.fit(sc_q.transform(Q_out_tr), y_tr)
        preds_q = ridge_q.predict(sc_q.transform(Q_out_ho))
        r2_q = r2_score(y_ho, preds_q)

        # Ridge: combined (x_enc + q_out) → affinity
        combined_tr = np.concatenate([X_enc_tr, Q_out_tr], axis=1)
        combined_ho = np.concatenate([X_enc_ho, Q_out_ho], axis=1)
        sc_comb = StandardScaler().fit(combined_tr)
        ridge_comb = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000], cv=5)
        ridge_comb.fit(sc_comb.transform(combined_tr), y_tr)
        preds_comb = ridge_comb.predict(sc_comb.transform(combined_ho))
        r2_comb = r2_score(y_ho, preds_comb)

        print(f"  Pre-quantum (x_enc) R²:  {r2_enc:.4f}  (alpha={ridge_enc.alpha_})")
        print(f"  Quantum output (q_out) R²: {r2_q:.4f}  (alpha={ridge_q.alpha_})")
        print(f"  Combined R²:             {r2_comb:.4f}  (alpha={ridge_comb.alpha_})")

        rows.append({
            "checkpoint": ckpt_idx,
            "representation": "pre_quantum_x_enc",
            "r2": r2_enc,
            "dim": X_enc_tr.shape[1],
            "best_alpha": float(ridge_enc.alpha_),
        })
        rows.append({
            "checkpoint": ckpt_idx,
            "representation": "quantum_q_out",
            "r2": r2_q,
            "dim": Q_out_tr.shape[1],
            "best_alpha": float(ridge_q.alpha_),
        })
        rows.append({
            "checkpoint": ckpt_idx,
            "representation": "combined",
            "r2": r2_comb,
            "dim": combined_tr.shape[1],
            "best_alpha": float(ridge_comb.alpha_),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Phase D — Statistical Analysis & Plots
# =============================================================================
def phase_d_analysis(
    intervention_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    output_dir: str,
):
    """Compute summary statistics, bootstrap CIs, and generate plots."""

    # ── Aggregate intervention results ──
    summary_rows = []
    for mode in intervention_df["intervention"].unique():
        sub = intervention_df[intervention_df["intervention"] == mode]
        summary_rows.append({
            "intervention": mode,
            "r2_mean": sub["r2"].mean(),
            "r2_std": sub["r2"].std(),
            "r2_median": sub["r2"].median(),
            "rmse_mean": sub["rmse"].mean(),
            "rmse_std": sub["rmse"].std(),
            "mae_mean": sub["mae"].mean(),
            "pearson_mean": sub["pearson"].mean(),
            "spearman_mean": sub["spearman"].mean(),
            "n_evaluations": len(sub),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "intervention_summary.csv"), index=False)

    # ── Paired delta R² (per checkpoint: intervention - normal) ──
    normal_r2 = {}
    for _, row in intervention_df[intervention_df["intervention"] == "normal"].iterrows():
        normal_r2[row["checkpoint"]] = row["r2"]

    delta_rows = []
    for _, row in intervention_df[intervention_df["intervention"] != "normal"].iterrows():
        ckpt = row["checkpoint"]
        if ckpt in normal_r2:
            delta_rows.append({
                "checkpoint": ckpt,
                "intervention": row["intervention"],
                "repeat_seed": row["repeat_seed"],
                "r2_intervention": row["r2"],
                "r2_normal": normal_r2[ckpt],
                "delta_r2": row["r2"] - normal_r2[ckpt],
            })
    delta_df = pd.DataFrame(delta_rows) if delta_rows else pd.DataFrame()

    # ── Bootstrap 95% CI for ΔR² ──
    bootstrap_rows = []
    if len(delta_df) > 0:
        for mode in delta_df["intervention"].unique():
            deltas = delta_df[delta_df["intervention"] == mode]["delta_r2"].values
            if len(deltas) < 2:
                bootstrap_rows.append({
                    "intervention": mode,
                    "delta_r2_mean": float(deltas.mean()) if len(deltas) else 0.0,
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                })
                continue
            rng = np.random.RandomState(42)
            boot_means = []
            for _ in range(10000):
                sample = rng.choice(deltas, size=len(deltas), replace=True)
                boot_means.append(sample.mean())
            boot_means = np.array(boot_means)
            bootstrap_rows.append({
                "intervention": mode,
                "delta_r2_mean": float(deltas.mean()),
                "ci_lower": float(np.percentile(boot_means, 2.5)),
                "ci_upper": float(np.percentile(boot_means, 97.5)),
            })

    # ── Generate plots ──
    _plot_intervention_r2(summary_df, output_dir)
    if len(delta_df) > 0:
        _plot_delta_r2(delta_df, bootstrap_rows, output_dir)
    _plot_paired_r2(intervention_df, output_dir)
    _plot_linear_probe(probe_df, output_dir)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("Phase D — Summary")
    print(f"{'='*60}")
    print("\nIntervention R² (mean ± std):")
    for _, row in summary_df.iterrows():
        print(f"  {row['intervention']:20s}  R² = {row['r2_mean']:.4f} ± {row['r2_std']:.4f}")
    if bootstrap_rows:
        print("\nΔR² (intervention - normal), bootstrap 95% CI:")
        for b in bootstrap_rows:
            print(
                f"  {b['intervention']:20s}  ΔR² = {b['delta_r2_mean']:+.4f}  "
                f"[{b['ci_lower']:+.4f}, {b['ci_upper']:+.4f}]"
            )
    print("\nLinear Probe R² (mean across checkpoints):")
    for rep in probe_df["representation"].unique():
        sub = probe_df[probe_df["representation"] == rep]
        print(f"  {rep:25s}  R² = {sub['r2'].mean():.4f} ± {sub['r2'].std():.4f}")


def _plot_intervention_r2(summary_df: pd.DataFrame, output_dir: str):
    """Bar plot of R² per intervention mode."""
    fig, ax = plt.subplots(figsize=(8, 5))
    modes = summary_df["intervention"].values
    means = summary_df["r2_mean"].values
    stds = summary_df["r2_std"].values

    colors = {
        "normal": "#4C72B0",
        "shuffle_samples": "#DD8452",
        "mean": "#55A868",
        "zero": "#C44E52",
        "matched_noise": "#8172B3",
    }
    bar_colors = [colors.get(m, "#888888") for m in modes]

    x = np.arange(len(modes))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors,
                  alpha=0.85, edgecolor="k", linewidth=0.6,
                  error_kw={"elinewidth": 1.5})

    for bar, val, err in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + err + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=10)
    ax.set_ylabel("Holdout R²", fontsize=12)
    ax.set_title("Quantum Layer Intervention: Holdout R²", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"intervention_r2.{fmt}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved intervention_r2.png/.pdf")


def _plot_delta_r2(delta_df: pd.DataFrame, bootstrap_rows: list, output_dir: str):
    """Box plot of ΔR² per intervention mode with bootstrap CI markers."""
    modes = sorted(delta_df["intervention"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))

    data_to_plot = [delta_df[delta_df["intervention"] == m]["delta_r2"].values for m in modes]
    bp = ax.boxplot(data_to_plot, tick_labels=[m.replace("_", "\n") for m in modes],
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=6))

    colors = ["#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for patch, c in zip(bp["boxes"], colors[:len(modes)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    ax.axhline(0, color="black", lw=1.0, linestyle="--")
    ax.set_ylabel("ΔR² (intervention − normal)", fontsize=12)
    ax.set_title("Quantum Intervention Effect: ΔR²", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"delta_r2.{fmt}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved delta_r2.png/.pdf")


def _plot_paired_r2(intervention_df: pd.DataFrame, output_dir: str):
    """Paired dot-line plot: normal vs each intervention per checkpoint."""
    modes = [m for m in intervention_df["intervention"].unique() if m != "normal"]
    if not modes:
        return

    normal_by_ckpt = {}
    for _, row in intervention_df[intervention_df["intervention"] == "normal"].iterrows():
        normal_by_ckpt[row["checkpoint"]] = row["r2"]

    fig, ax = plt.subplots(figsize=(8, 5))
    color_map = {
        "shuffle_samples": "#DD8452",
        "mean": "#55A868",
        "zero": "#C44E52",
        "matched_noise": "#8172B3",
    }

    for mode in modes:
        sub = intervention_df[intervention_df["intervention"] == mode]
        # average over repeat seeds per checkpoint
        mode_by_ckpt = sub.groupby("checkpoint")["r2"].mean()
        for ckpt in mode_by_ckpt.index:
            if ckpt in normal_by_ckpt:
                ax.plot(
                    [normal_by_ckpt[ckpt], mode_by_ckpt[ckpt]],
                    [0, 1],
                    "o-",
                    color=color_map.get(mode, "#888"),
                    alpha=0.5,
                    markersize=6,
                )

    # Use scatter for legend
    for mode in modes:
        sub = intervention_df[intervention_df["intervention"] == mode]
        mean_r2 = sub["r2"].mean()
        ax.scatter([], [], color=color_map.get(mode, "#888"),
                   label=f"{mode} (mean={mean_r2:.3f})", s=50)

    normal_vals = list(normal_by_ckpt.values())
    if normal_vals:
        ax.axvline(np.mean(normal_vals), color="#4C72B0", linestyle="--",
                   label=f"normal (mean={np.mean(normal_vals):.3f})")

    ax.set_xlabel("R²", fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Intervened"], fontsize=11)
    ax.set_title("Paired R²: Normal vs Interventions", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"paired_r2.{fmt}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved paired_r2.png/.pdf")


def _plot_linear_probe(probe_df: pd.DataFrame, output_dir: str):
    """Bar plot of linear probe R² for each representation type."""
    fig, ax = plt.subplots(figsize=(7, 5))
    reps = probe_df["representation"].unique()

    r2_means = [probe_df[probe_df["representation"] == r]["r2"].mean() for r in reps]
    r2_stds = [probe_df[probe_df["representation"] == r]["r2"].std() for r in reps]

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    x = np.arange(len(reps))
    bars = ax.bar(x, r2_means, yerr=r2_stds, capsize=6, color=colors[:len(reps)],
                  alpha=0.85, edgecolor="k", linewidth=0.6,
                  error_kw={"elinewidth": 1.5})

    for bar, val, err in zip(bars, r2_means, r2_stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + err + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    labels = [r.replace("_", "\n") for r in reps]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Ridge R² on Holdout", fontsize=12)
    ax.set_title("Linear Probe: Representation Quality", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"linear_probe_r2.{fmt}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved linear_probe_r2.png/.pdf")


# =============================================================================
# Architecture Audit
# =============================================================================
def write_architecture_audit(output_dir: str):
    """Generate architecture_audit.md answering the 8 key questions."""
    audit = """\
# Architecture Audit: ModelHybridFC_Reservoir

## 1. What is the input to the quantum layer?

`x_enc` — the output of `tanh(fc2(relu(bn1(fc1(x))))) * π`, shape `[batch, n_qubits]`.
This is the classical encoding compressed from the full input features down to `n_qubits`
dimensions, scaled to `[-π, π]` via `tanh * π`.

## 2. What is the output from the quantum layer?

`q_out` — expectation values of Pauli X, Y, Z on each qubit, shape `[batch, 3*n_qubits]`.
The quantum reservoir is a fixed (non-trainable) G3 circuit (CNOT, H, T gates).

## 3. What are the dimensions?

For the 6-qubit model:
- Input to quantum: `[batch, 6]` (x_enc)
- Output from quantum: `[batch, 18]` (3 × 6 = 18 expectation values)
- Combined after skip: `[batch, 24]` (18 quantum + 6 skip)

## 4. Is the quantum output concatenated with classical features?

**YES.** `x_enc` (the pre-quantum encoding) is concatenated with `q_out` via skip connection:
`combined = torch.cat([q_out, x_enc], dim=1)  # [batch, 4*n_qubits]`

## 5. Are there skip connections?

**YES.** The skip connection passes `x_enc` directly to the MLP head, bypassing the quantum
layer. The MLP head receives `[q_out, x_enc]`, meaning it sees both quantum and classical
representations. This allows the network to potentially route around the quantum layer.

## 6. What normalization is applied?

- **Before quantum:** BatchNorm1d on `fc1` output (4*n_qubits dims), then tanh scaling
- **After quantum:** None — raw expectation values in `[-1, 1]` go directly to concat
- **In MLP head:** BatchNorm1d(64) after first linear layer

## 7. How expressive is the downstream network?

The MLP head is:
```
Linear(24 → 64) → BatchNorm1d(64) → ReLU → Dropout(0.2)
→ Linear(64 → 32) → ReLU → Linear(32 → 1)
```
With 24 input features (18 quantum + 6 skip), the head has 64×24 + 64 + 64×32 + 32 +
32×1 + 1 = 3,777 parameters. This is expressive enough to selectively ignore quantum
features and rely entirely on the 6-dimensional skip connection.

## 8. Multi-layer bypasses?

In `ModelMultiLayerG3Hybrid`, the initial encoding `x_enc_0` is concatenated with ALL
layer outputs: `combined = cat([q_out_1, q_out_2, ..., q_out_L, x_enc_0])`.
This provides a full classical bypass — the MLP head can ignore every quantum layer
and still access the original classical encoding.
"""
    path = os.path.join(output_dir, "architecture_audit.md")
    with open(path, "w") as f:
        f.write(audit)
    print(f"Saved: {path}")


# =============================================================================
# Main entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Quantum layer causal intervention experiments."
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of training seeds for Phase A (default: 5)",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Training epochs per seed (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--shuffle-repeats", type=int, default=10,
        help="Permutation seeds for shuffle_samples intervention (default: 10)",
    )
    parser.add_argument(
        "--noise-repeats", type=int, default=10,
        help="Noise seeds for matched_noise intervention (default: 10)",
    )
    parser.add_argument(
        "--interventions", nargs="+", default=ALL_INTERVENTIONS,
        choices=ALL_INTERVENTIONS, metavar="MODE",
        help=f"Intervention modes (default: {' '.join(ALL_INTERVENTIONS)})",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip Phase A; load existing checkpoints",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Directory with existing checkpoints (requires --skip-training)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/quantum_intervention)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    args = parser.parse_args()

    # ── Output directory ──
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(_QF_DIR, "..", "..", "results", "quantum_intervention")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ── Save run config ──
    run_config = {
        "timestamp": datetime.now().isoformat(),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "shuffle_repeats": args.shuffle_repeats,
        "noise_repeats": args.noise_repeats,
        "interventions": args.interventions,
        "skip_training": args.skip_training,
        "batch_size": args.batch_size,
        "best_gate_count": BEST_GATE_COUNT,
        "best_n_qubits": BEST_N_QUBITS,
        "best_circuit_idx": BEST_CIRCUIT_IDX,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # ── Load data ──
    print("\nLoading data...")
    data = load_data(batch_size=args.batch_size)
    print(f"  Train: {data.n_train}  Val: {data.n_val}  Holdout: {data.n_holdout}")
    print(f"  Features: {data.in_features}  Source: {data.data_source}")

    # ── Reproduce best circuit ──
    print("\nReproducing best G3 circuit...")
    circuit = get_best_circuit(n_qubits=BEST_N_QUBITS)
    print(f"  Gate count={BEST_GATE_COUNT}  Qubits={BEST_N_QUBITS}  "
          f"Circuit idx={BEST_CIRCUIT_IDX}")

    # ── Architecture audit ──
    write_architecture_audit(output_dir)

    # ── Phase A: Train & Save ──
    if args.skip_training:
        ckpt_dir = args.checkpoint_dir or os.path.join(output_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        checkpoint_paths = sorted(
            [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
        )
        if not checkpoint_paths:
            raise FileNotFoundError(f"No .pth files found in {ckpt_dir}")
        print(f"\nSkipping training. Found {len(checkpoint_paths)} checkpoints in {ckpt_dir}")
    else:
        checkpoint_paths = phase_a_train(
            data, circuit, output_dir,
            n_seeds=args.seeds, epochs=args.epochs,
        )

    # ── Phase B: Interventions ──
    print("\n" + "=" * 70)
    print("Phase B — Intervention Evaluation")
    print("=" * 70)
    intervention_df = phase_b_interventions(
        data, circuit, checkpoint_paths,
        interventions=args.interventions,
        shuffle_repeats=args.shuffle_repeats,
        noise_repeats=args.noise_repeats,
    )
    intervention_df.to_csv(
        os.path.join(output_dir, "intervention_results.csv"), index=False
    )
    print(f"\nSaved: intervention_results.csv ({len(intervention_df)} rows)")

    # ── Phase C: Linear Probe ──
    print("\n" + "=" * 70)
    print("Phase C — Linear Probe")
    print("=" * 70)
    probe_df = phase_c_linear_probe(data, circuit, checkpoint_paths)
    probe_df.to_csv(os.path.join(output_dir, "linear_probe_results.csv"), index=False)
    print(f"\nSaved: linear_probe_results.csv ({len(probe_df)} rows)")

    # ── Phase D: Analysis & Plots ──
    print("\n" + "=" * 70)
    print("Phase D — Statistical Analysis & Plots")
    print("=" * 70)
    phase_d_analysis(intervention_df, probe_df, output_dir)

    print(f"\nAll outputs saved to: {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
