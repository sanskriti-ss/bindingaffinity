#!/usr/bin/env python
"""
quantum_kernel_study.py — Quantum Kernel Experiments
=====================================================

Implements quantum reservoir computing as a proper kernel method
following Domingo et al. (2022) and Schuld & Killoran (2022):

    K_Q(i,j) = φ(xᵢ)ᵀ φ(xⱼ)

where φ(x) ∈ ℝ^{3·n_qubits} is the vector of Pauli X/Y/Z expectation
values of the quantum circuit after encoding x.  This is a *linear*
kernel in quantum feature space — KernelRidge with this K is exactly
equivalent to Ridge regression on the quantum features, but the kernel
view makes the geometry explicit.

Experiments
-----------
1. Sweep N circuits, score each by:
     - RFD (expressibility)
     - Centered Kernel Alignment (CKA) with the ideal label kernel
     - KernelRidge test R²
   to scatter plots to see which metric predicts R²

2. Compare quantum kernel against classical baselines:
     - Linear kernel on raw features
     - RBF kernel (σ auto-tuned by median heuristic)
     - Polynomial kernel (degree 2, 3)
  to bar chart

3. Best circuit deep-dive:
     - Kernel matrix heatmap (samples sorted by affinity)
     - KPCA (first 2 PCs of K) coloured by pKi
     - Prediction scatter (predicted vs experimental pKi)
     - CKA across the circuit sweep

Run from quantum_fusion/ directory:
    python quantum_kernel_study.py [--n-circuits N] [--n-qubits Q]

Outputs (all written to quantum_kernel_study_output/):
    kernel_matrix_heatmap.png
    kpca_affinity.png
    circuit_sweep_scatter.png   (R² vs RFD, R² vs CKA)
    classical_vs_quantum_bar.png
    kernel_study_results.csv
"""

import os, sys, math, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import KernelPCA
from scipy.stats import pearsonr, spearmanr
import pennylane as qml

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_train import load_with_model_features
from testing_random_unitaries import (
    generate_g3_random_circuits,
    preselect_circuits_by_expressibility,
    reservoir_feature_diversity,
)

_qf_dir    = os.path.dirname(os.path.abspath(__file__))
_dcnn_npz  = os.path.join(_qf_dir, 'refined_3dcnn_features.npz')
_sgcnn_npz = os.path.join(_qf_dir, 'refined_sgcnn_features.npz')


# ── Kernel utilities ──────────────────────────────────────────────────────────

def quantum_feature_map(qc, X_encoded: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Map data through the quantum reservoir.

    X_encoded: (N, n_qubits) array already compressed + tanh-scaled to (-π, π).
    Returns φ(X): (N, 3*n_qubits) Pauli X/Y/Z expectation values.
    """
    dev = qml.device('lightning.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs):
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        for inst in qc.data:
            gate   = inst.operation
            qubits = [qc.find_bit(q).index for q in inst.qubits]
            if gate.name == 'h':   qml.Hadamard(wires=qubits[0])
            elif gate.name == 't': qml.T(wires=qubits[0])
            elif gate.name == 'cx':qml.CNOT(wires=qubits)
        return (
            [qml.expval(qml.PauliX(i)) for i in range(n_qubits)] +
            [qml.expval(qml.PauliY(i)) for i in range(n_qubits)] +
            [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        )

    out = []
    for x in X_encoded:
        row = circuit(x.astype(np.float64))
        out.append([float(v) for v in row])
    return np.array(out, dtype=np.float32)


def encode(X_raw: np.ndarray, n_qubits: int, random_seed: int = 42) -> np.ndarray:
    """
    Compress D-dim features to n_qubits angles via a fixed random projection.
    Same projection is used for all circuits so circuit differences are isolated.
    Returns: (N, n_qubits) in (-π, π).
    """
    rng = np.random.default_rng(random_seed)
    W   = rng.standard_normal((X_raw.shape[1], n_qubits))
    W  /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-8
    return np.tanh(X_raw @ W) * math.pi


def linear_quantum_kernel(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    """K(i,j) = φ(xᵢ)ᵀ φ(xⱼ)  — linear kernel in quantum feature space."""
    return phi_a @ phi_b.T


def centered_kernel_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Centered Kernel Alignment (Cortes et al. 2012) between kernel K and the
    ideal kernel K_y = y·yᵀ.  Range [0, 1]; higher = K is better aligned
    with the regression target.
    """
    n = len(y)
    # Centre K
    H   = np.eye(n) - np.ones((n, n)) / n
    Kc  = H @ K @ H
    Ky  = np.outer(y, y)
    Kyc = H @ Ky @ H
    num  = np.trace(Kc @ Kyc)
    denom= np.sqrt(np.trace(Kc @ Kc) * np.trace(Kyc @ Kyc)) + 1e-12
    return float(num / denom)


def rbf_kernel(X_a: np.ndarray, X_b: np.ndarray, sigma: float = None) -> np.ndarray:
    """RBF kernel with median-heuristic σ when not specified."""
    if sigma is None:
        dists = np.sum((X_a[:, None, :] - X_a[None, :, :]) ** 2, axis=-1)
        sigma = np.sqrt(np.median(dists[dists > 0]) / 2)
    sq = np.sum((X_a[:, None, :] - X_b[None, :, :]) ** 2, axis=-1)
    return np.exp(-sq / (2 * sigma ** 2))


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_kernel_ridge(K_tr, K_te, y_tr, y_te, alphas=(0.001, 0.01, 0.1, 1, 10, 100)):
    """Cross-validate α on training kernel, then score on test kernel."""
    best_r2, best_alpha = -np.inf, alphas[0]
    for a in alphas:
        kr = KernelRidge(alpha=a, kernel='precomputed')
        scores = cross_val_score(kr, K_tr, y_tr, cv=5, scoring='r2')
        if scores.mean() > best_r2:
            best_r2, best_alpha = scores.mean(), a

    kr = KernelRidge(alpha=best_alpha, kernel='precomputed')
    kr.fit(K_tr, y_tr)
    preds   = kr.predict(K_te)
    r2      = r2_score(y_te, preds)
    rmse    = math.sqrt(mean_squared_error(y_te, preds))
    mae     = mean_absolute_error(y_te, preds)
    pearson = pearsonr(y_te, preds)[0]
    spearman= spearmanr(y_te, preds)[0]
    return {'r2': r2, 'rmse': rmse, 'mae': mae,
            'pearson': pearson, 'spearman': spearman,
            'best_alpha': best_alpha, 'preds': preds, 'true': y_te}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_kernel_heatmap(K: np.ndarray, y_sorted_idx: np.ndarray,
                        title: str, out_path: str):
    """Kernel matrix heatmap with rows/cols sorted by affinity."""
    K_sorted = K[np.ix_(y_sorted_idx, y_sorted_idx)]
    fig, ax  = plt.subplots(figsize=(7, 6))
    im = ax.imshow(K_sorted, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Kernel value')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Sample (sorted by pKi)')
    ax.set_ylabel('Sample (sorted by pKi)')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_kpca(phi: np.ndarray, y: np.ndarray, title: str, out_path: str):
    """KPCA (first 2 PCs of linear quantum kernel) coloured by pKi."""
    K   = linear_quantum_kernel(phi, phi)
    kpca = KernelPCA(n_components=2, kernel='precomputed', eigen_solver='dense')
    emb  = kpca.fit_transform(K)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap='RdYlBu_r',
                    s=18, alpha=0.75, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='pKi (−log Kd)')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('KPCA 1')
    ax.set_ylabel('KPCA 2')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_sweep_scatter(df: pd.DataFrame, out_path: str):
    """2-panel: R² vs RFD, R² vs CKA."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Quantum Circuit Sweep — What Predicts Test R²?',
                 fontsize=13, fontweight='bold')

    for ax, xcol, xlabel in [
        (axes[0], 'rfd',  'RFD (Reservoir Feature Diversity)'),
        (axes[1], 'cka',  'CKA (Kernel Alignment with Labels)'),
    ]:
        ax.scatter(df[xcol], df['r2'], s=55, alpha=0.8,
                   c=df['r2'], cmap='viridis', edgecolors='k', linewidths=0.4)
        # Pearson r annotation
        r, p = pearsonr(df[xcol], df['r2'])
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Test R²', fontsize=11)
        ax.set_title(f'Pearson r = {r:.3f}  (p={p:.3f})', fontsize=10)
        # Trend line
        z = np.polyfit(df[xcol], df['r2'], 1)
        xr = np.linspace(df[xcol].min(), df[xcol].max(), 100)
        ax.plot(xr, np.polyval(z, xr), 'r--', lw=1.3, alpha=0.7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_classical_vs_quantum(classical_results: dict, q_results: dict,
                               out_path: str):
    """Horizontal bar chart: classical baselines vs quantum kernel variants."""
    labels, r2s, colors = [], [], []

    # Classical baselines
    palette = {
        'Linear (classical)': '#4878d0',
        'RBF (σ=median)':      '#6acc65',
        'Polynomial deg-2':    '#d65f5f',
        'Polynomial deg-3':    '#b47cc7',
    }
    for name, color in palette.items():
        if name in classical_results:
            labels.append(name); r2s.append(classical_results[name]); colors.append(color)

    # Quantum kernel results
    for name, val in q_results.items():
        labels.append(name); r2s.append(val); colors.append('#ee8866')

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.55)))
    bars = ax.barh(labels[::-1], r2s[::-1], color=colors[::-1], height=0.55)
    ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=9)
    ax.set_xlabel('Test R²', fontsize=12)
    ax.set_title('Quantum Kernel vs Classical Baselines  (KernelRidge readout)',
                 fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', lw=0.8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_prediction_scatter(res_dict: dict, label_std: float,
                             label_mean: float, title: str, out_path: str):
    """Predicted vs experimental pKi scatter for the best quantum circuit."""
    true_pki  = res_dict['true']  * label_std + label_mean
    pred_pki  = res_dict['preds'] * label_std + label_mean
    r2, pearson = res_dict['r2'], res_dict['pearson']
    rmse_pki  = res_dict['rmse']  * label_std

    vmin = min(true_pki.min(), pred_pki.min()) - 0.3
    vmax = max(true_pki.max(), pred_pki.max()) + 0.3

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_pki, pred_pki, alpha=0.55, s=22,
               color='steelblue', edgecolors='none')
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1.5)
    ax.set_xlabel('Experimental pKi', fontsize=12)
    ax.set_ylabel('Predicted pKi',    fontsize=12)
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.text(0.97, 0.05, f'R² = {r2:.4f}\nPearson r = {pearson:.4f}'
                         f'\nRMSE = {rmse_pki:.3f} pKi',
            transform=ax.transAxes, fontsize=10,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_cka_bar(df: pd.DataFrame, out_path: str):
    """CKA bar chart across all circuits, sorted by R²."""
    df_s = df.sort_values('r2', ascending=True)
    colors = plt.cm.RdYlGn(  # type: ignore[attr-defined]
        (df_s['r2'] - df_s['r2'].min()) /
        (df_s['r2'].max() - df_s['r2'].min() + 1e-8))

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(df_s) * 0.35)))
    y_labels = [f"#{int(r['circ_idx'])}" for _, r in df_s.iterrows()]
    for ax, col, xlabel in [
        (axes[0], 'cka', 'CKA (Kernel Alignment)'),
        (axes[1], 'r2',  'Test R²'),
    ]:
        bars = ax.barh(y_labels, df_s[col].values, color=colors, height=0.6)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(f'Per-circuit {xlabel}', fontsize=11)

    fig.suptitle('Quantum Circuits — CKA Alignment & Test R²  (sorted by R²)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_circuits: int = 30, n_qubits: int = 6):
    out_dir = os.path.join(_qf_dir, 'quantum_kernel_study_output')
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading features …")
    sgcnn_f, cnn3d_f, labels_raw, ids = load_with_model_features(
        max_samples=6000,
        dcnn_npz=_dcnn_npz  if os.path.exists(_dcnn_npz)  else None,
        sgcnn_npz=_sgcnn_npz if os.path.exists(_sgcnn_npz) else None,
    )
    X_all = np.hstack([sgcnn_f, cnn3d_f]).astype(np.float32)

    n = len(labels_raw)
    all_idx = np.arange(n)
    n_bins  = min(10, max(2, int(np.sqrt(n))))
    bins    = pd.qcut(labels_raw, q=n_bins, labels=False, duplicates='drop')
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.20, random_state=42, shuffle=True, stratify=bins)

    # Scale on train only
    scaler    = StandardScaler().fit(X_all[train_idx])
    X_all_sc  = scaler.transform(X_all).astype(np.float32)

    label_mean = float(labels_raw[train_idx].mean())
    label_std  = float(labels_raw[train_idx].std()) + 1e-8
    y_all      = (labels_raw - label_mean) / label_std
    y_tr, y_te = y_all[train_idx], y_all[test_idx]

    print(f"Samples: {n}  Train: {len(train_idx)}  Test: {len(test_idx)}")
    print(f"Feature dim: {X_all_sc.shape[1]}  |  n_qubits: {n_qubits}\n")

    # ── 2. Encode to quantum angles (fixed projection for all circuits) ────────
    X_enc_all = encode(X_all_sc, n_qubits, random_seed=42)
    X_enc_tr  = X_enc_all[train_idx]
    X_enc_te  = X_enc_all[test_idx]

    # ── 3. Classical baseline kernels ─────────────────────────────────────────
    print("Classical kernel baselines …")
    classical_results = {}

    # Linear
    K_lin_tr = X_enc_tr @ X_enc_tr.T
    K_lin_te = X_enc_te @ X_enc_tr.T
    classical_results['Linear (classical)'] = eval_kernel_ridge(
        K_lin_tr, K_lin_te, y_tr, y_te)['r2']

    # RBF
    K_rbf_tr = rbf_kernel(X_enc_tr, X_enc_tr)
    K_rbf_te = rbf_kernel(X_enc_te, X_enc_tr)
    classical_results['RBF (σ=median)'] = eval_kernel_ridge(
        K_rbf_tr, K_rbf_te, y_tr, y_te)['r2']

    # Polynomial deg-2, 3
    for deg in [2, 3]:
        K_p_tr = (X_enc_tr @ X_enc_tr.T + 1) ** deg
        K_p_te = (X_enc_te @ X_enc_tr.T + 1) ** deg
        classical_results[f'Polynomial deg-{deg}'] = eval_kernel_ridge(
            K_p_tr, K_p_te, y_tr, y_te)['r2']

    for name, r2 in classical_results.items():
        print(f"  {name:25s}  R² = {r2:.4f}")

    # Also run on full 153-dim (not projected) for honest comparison
    X_full_tr = X_all_sc[train_idx]
    X_full_te = X_all_sc[test_idx]
    K_full_rbf_tr = rbf_kernel(X_full_tr, X_full_tr)
    K_full_rbf_te = rbf_kernel(X_full_te, X_full_tr)
    r2_full_rbf = eval_kernel_ridge(K_full_rbf_tr, K_full_rbf_te, y_tr, y_te)['r2']
    classical_results['RBF on full 153-dim'] = r2_full_rbf
    print(f"  {'RBF on full 153-dim':25s}  R² = {r2_full_rbf:.4f}")

    # ── 4. Generate & score circuits ──────────────────────────────────────────
    print(f"\nGenerating {n_circuits} G3 circuits …")
    circuits = generate_g3_random_circuits(n_qubits, num_gates=300,
                                            num_circuits=n_circuits)

    print("\nEvaluating all circuits (quantum features + KernelRidge) …")
    sweep_rows = []
    best_r2, best_phi_tr, best_phi_te, best_res, best_circ_idx = -np.inf, None, None, None, -1

    for circ_idx, qc in enumerate(tqdm(circuits, desc='Circuit sweep', ascii=True)):
        # RFD score
        rfd = reservoir_feature_diversity(qc, n_qubits, real_inputs=X_enc_tr)

        # Quantum feature map
        phi_tr = quantum_feature_map(qc, X_enc_tr, n_qubits)
        phi_te = quantum_feature_map(qc, X_enc_te, n_qubits)

        # Quantum kernel (linear in feature space)
        K_q_tr = linear_quantum_kernel(phi_tr, phi_tr)
        K_q_te = linear_quantum_kernel(phi_te, phi_tr)

        # CKA with label kernel on train set
        cka = centered_kernel_alignment(K_q_tr, y_tr)

        # KernelRidge evaluation
        res = eval_kernel_ridge(K_q_tr, K_q_te, y_tr, y_te)

        sweep_rows.append({
            'circ_idx': circ_idx,
            'rfd':      round(rfd,        4),
            'cka':      round(cka,        4),
            'r2':       round(res['r2'],  4),
            'rmse':     round(res['rmse'] * label_std, 4),
            'mae':      round(res['mae']  * label_std, 4),
            'pearson':  round(res['pearson'],  4),
            'spearman': round(res['spearman'], 4),
            'alpha':    res['best_alpha'],
        })

        if res['r2'] > best_r2:
            best_r2 = res['r2']
            best_phi_tr   = phi_tr
            best_phi_te   = phi_te
            best_res      = res
            best_circ_idx = circ_idx

        print(f"  Circuit {circ_idx:3d}: RFD={rfd:.3f}  CKA={cka:.3f}  R²={res['r2']:.4f}")

    df_sweep = pd.DataFrame(sweep_rows)
    csv_path = os.path.join(out_dir, 'kernel_study_results.csv')
    df_sweep.to_csv(csv_path, index=False)
    print(f"\nSaved sweep CSV -> {csv_path}")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    best_row = df_sweep.loc[df_sweep['r2'].idxmax()]
    print(f"\n{'='*60}")
    print(f"Best quantum kernel circuit: #{int(best_row['circ_idx'])}")
    print(f"  RFD       = {best_row['rfd']:.4f}")
    print(f"  CKA       = {best_row['cka']:.4f}")
    print(f"  R²        = {best_row['r2']:.4f}")
    print(f"  RMSE      = {best_row['rmse']:.4f} pKi")
    print(f"  Pearson r = {best_row['pearson']:.4f}")
    print(f"  Spearman ρ= {best_row['spearman']:.4f}")
    print(f"\nClassical RBF (full 153-dim)  R² = {r2_full_rbf:.4f}")
    print(f"Classical RBF (6-dim encoded) R² = {classical_results['RBF (σ=median)']:.4f}")
    print(f"Quantum kernel (best circuit) R² = {best_row['r2']:.4f}")
    gap = best_row['r2'] - classical_results['RBF (σ=median)']
    print(f"Quantum gain over same-dim RBF   = {gap:+.4f}")
    print(f"{'='*60}")

    # Quantum kernel variants for bar chart
    q_bar = {
        f"Quantum linear (best #{int(best_row['circ_idx'])})": best_row['r2'],
        f"Quantum linear (median circuit)":                     float(df_sweep['r2'].median()),
        f"Quantum linear (worst circuit)":                      float(df_sweep['r2'].min()),
    }

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots …")

    # 6a. Kernel matrix heatmap (best circuit, train set only for visibility)
    K_best_tr = linear_quantum_kernel(best_phi_tr, best_phi_tr)
    sort_idx  = np.argsort(y_tr)
    plot_kernel_heatmap(
        K_best_tr, sort_idx,
        title=f'Quantum Kernel Matrix — Circuit #{best_circ_idx}\n'
              f'(train set, sorted by pKi)',
        out_path=os.path.join(out_dir, 'kernel_matrix_heatmap.png'))

    # 6b. KPCA coloured by affinity (best circuit, test set)
    plot_kpca(
        best_phi_te, y_te * label_std + label_mean,
        title=f'Quantum KPCA — Circuit #{best_circ_idx}  (test set)',
        out_path=os.path.join(out_dir, 'kpca_affinity.png'))

    # 6c. Circuit sweep scatter: R² vs RFD + R² vs CKA
    if len(df_sweep) >= 4:
        plot_sweep_scatter(df_sweep, os.path.join(out_dir, 'circuit_sweep_scatter.png'))

    # 6d. Classical vs quantum comparison bar
    plot_classical_vs_quantum(
        classical_results, q_bar,
        os.path.join(out_dir, 'classical_vs_quantum_bar.png'))

    # 6e. Best circuit prediction scatter
    plot_prediction_scatter(
        best_res, label_std, label_mean,
        title=f'Quantum Kernel Ridge — Circuit #{best_circ_idx}\n'
              f'R²={best_res["r2"]:.4f}  Pearson r={best_res["pearson"]:.4f}',
        out_path=os.path.join(out_dir, 'best_circuit_scatter.png'))

    # 6f. CKA + R² bars across all circuits
    plot_cka_bar(df_sweep, os.path.join(out_dir, 'cka_r2_bars.png'))

    # ── 7. Print RFD–R² and CKA–R² correlations ──────────────────────────────
    r_rfd_r2, p_rfd  = pearsonr(df_sweep['rfd'], df_sweep['r2'])
    r_cka_r2, p_cka  = pearsonr(df_sweep['cka'], df_sweep['r2'])
    print(f"\nRFD  ↔ R² Pearson r = {r_rfd_r2:+.3f}  (p={p_rfd:.3f})  "
          + ("← RFD predicts performance" if abs(r_rfd_r2) > 0.4 else
             "← RFD does NOT reliably predict R²"))
    print(f"CKA  ↔ R² Pearson r = {r_cka_r2:+.3f}  (p={p_cka:.3f})  "
          + ("← CKA is a good selection metric" if abs(r_cka_r2) > 0.4 else
             "← CKA does NOT reliably predict R²"))

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-circuits', type=int, default=30,
                        help='Number of G3 circuits to sweep (default: 30)')
    parser.add_argument('--n-qubits',   type=int, default=6,
                        help='Number of qubits (default: 6)')
    args = parser.parse_args()
    main(n_circuits=args.n_circuits, n_qubits=args.n_qubits)
