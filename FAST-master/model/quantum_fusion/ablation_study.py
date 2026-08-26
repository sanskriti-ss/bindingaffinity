#!/usr/bin/env python
"""
ablation_study.py — Quantum Fusion Ablation Experiments
========================================================

Answers the core question: is R²=0.86 from the quantum reservoir,
or from the rich classical features feeding into a classical MLP head?

Runs four conditions on the SAME train/val/test split:

  A) Classical MLP only      — no quantum at all (ClassicalMLPBaseline, use_skip=False)
  B) Quantum + skip (original) — ModelHybridFC_Reservoir, use_skip=True
  C) Quantum, no skip        — ModelHybridFC_Reservoir, use_skip=False
                               Head forced to read ONLY quantum features.
  D) Random circuit, no skip — same as C but with a randomly-picked (non-RFD)
                               circuit to check whether RFD selection matters.

Expected outcome if quantum is bypassed by skip:
  R²(A) ≈ R²(B) >> R²(C)

Expected outcome if quantum actually contributes:
  R²(C) significantly above R²(A)

Run from quantum_fusion/ directory:
    python ablation_study.py [--n-circuits N] [--epochs E]
"""

import os, sys, math, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_train import (
    load_with_model_features,
    FusionDataset,
    ModelHybridFC_Reservoir,
    ClassicalMLPBaseline,
    evaluate_model,
)
from testing_random_unitaries import (
    generate_g3_random_circuits,
    preselect_circuits_by_expressibility,
)

# ── Config ────────────────────────────────────────────────────────────────────
N_QUBITS   = 6
NUM_GATES  = 100   # enough to get diversity without being slow
EPOCHS     = 50
BATCH_SIZE = 64
LR         = 3e-4
EARLY_STOP = 15
DEVICE     = torch.device('cpu')

_qf_dir    = os.path.dirname(os.path.abspath(__file__))
_dcnn_npz  = os.path.join(_qf_dir, 'refined_3dcnn_features.npz')
_sgcnn_npz = os.path.join(_qf_dir, 'refined_sgcnn_features.npz')


def train_and_eval(model, loaders, epochs=EPOCHS, lr=LR):
    """Train model, return best-val-checkpoint test metrics dict."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()
    best_val, best_state, no_imp = float('inf'), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for sg, c3, y in loaders['train']:
            x = torch.cat([sg, c3], dim=1).to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        rmse_v, *_ = evaluate_model(model, loaders['val'])
        scheduler.step(rmse_v)
        if rmse_v < best_val:
            best_val  = rmse_v
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= EARLY_STOP:
                break

    model.load_state_dict(best_state)
    rmse, mae, r2, pear, spear = evaluate_model(model, loaders['test'])
    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'pearson': pear, 'spearman': spear}


def build_loaders(combined, empty, labels, train_idx, val_idx, test_idx,
                  label_mean, label_std):
    def _norm(y): return (y - label_mean) / label_std
    loaders = {}
    for split, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        ds = FusionDataset(combined[idx], empty[idx], _norm(labels[idx]))
        loaders[split] = DataLoader(ds, batch_size=BATCH_SIZE,
                                    shuffle=(split == 'train'))
    return loaders


def main(n_circuits: int = 10, epochs: int = EPOCHS):
    # ── 1. Load & split data ─────────────────────────────────────────────────
    print("Loading features …")
    sgcnn_f, cnn3d_f, labels, ids = load_with_model_features(
        max_samples=6000,
        dcnn_npz=_dcnn_npz  if os.path.exists(_dcnn_npz)  else None,
        sgcnn_npz=_sgcnn_npz if os.path.exists(_sgcnn_npz) else None,
    )
    n = len(labels)
    all_idx = np.arange(n)
    n_bins  = min(10, max(2, int(np.sqrt(n))))
    bins    = pd.qcut(labels, q=n_bins, labels=False, duplicates='drop')

    train_idx, tmp = train_test_split(all_idx, test_size=0.30,
                                      random_state=42, shuffle=True, stratify=bins)
    val_idx, test_idx = train_test_split(tmp, test_size=0.50,
                                         random_state=42, shuffle=True,
                                         stratify=bins[tmp])

    # Scale features on train only
    combined = np.hstack([sgcnn_f, cnn3d_f]).astype(np.float32)
    scaler   = StandardScaler().fit(combined[train_idx])
    combined = scaler.transform(combined).astype(np.float32)
    empty    = np.zeros((n, 0), dtype=np.float32)

    label_mean = float(labels[train_idx].mean())
    label_std  = float(labels[train_idx].std()) + 1e-8
    dims = combined.shape[1]
    print(f"Feature dim: {dims}  Train/Val/Test: "
          f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"Label stats (train): mean={label_mean:.3f}  std={label_std:.3f}\n")

    loaders = build_loaders(combined, empty, labels,
                            train_idx, val_idx, test_idx, label_mean, label_std)

    # ── 2. Pick best RFD circuit + one random circuit ────────────────────────
    # Compute real encoder inputs from train data so RFD uses the correct
    # input distribution (tanh-compressed, not uniform).
    print(f"Generating {n_circuits} G3 circuits …")
    circuits = generate_g3_random_circuits(N_QUBITS, num_gates=NUM_GATES,
                                           num_circuits=n_circuits)
    import torch.nn as _nn
    _rfd_sample = combined[train_idx[:min(300, len(train_idx))]]
    _rfd_t      = torch.tensor(_rfd_sample, dtype=torch.float32)
    with torch.no_grad():
        _fc1 = _nn.Linear(dims, 4 * N_QUBITS)
        _bn1 = _nn.BatchNorm1d(4 * N_QUBITS); _bn1.eval()
        _fc2 = _nn.Linear(4 * N_QUBITS, N_QUBITS)
        real_rfd_inputs = (torch.tanh(_fc2(torch.relu(
            _bn1(_fc1(_rfd_t))))) * math.pi).numpy()
    del _fc1, _bn1, _fc2, _rfd_t

    indexed  = preselect_circuits_by_expressibility(
        circuits, N_QUBITS, top_k=1, real_inputs=real_rfd_inputs)
    _, best_circuit = indexed[0]
    import random
    random_circuit = random.choice(circuits)
    print(f"Best circuit by RFD: #{indexed[0][0]}")

    # ── 3. Run ablation conditions ────────────────────────────────────────────
    results = {}

    # ── A: Classical MLP (no quantum) ────────────────────────────────────────
    print("\n[A] Classical MLP baseline (no quantum) …")
    model_A = ClassicalMLPBaseline(
        in_features=dims, out_features=1, n_qubits=N_QUBITS, use_skip=False).to(DEVICE)
    results['A_classical'] = train_and_eval(model_A, loaders, epochs=epochs)
    print(f"    R²={results['A_classical']['r2']:.4f}  "
          f"RMSE={results['A_classical']['rmse']*label_std:.4f} pKi")

    # ── B: Quantum reservoir + skip (original) ────────────────────────────────
    print("\n[B] Quantum reservoir + skip connection (original) …")
    model_B = ModelHybridFC_Reservoir(
        in_features=dims, out_features=1,
        qiskit_circuit=best_circuit, n_qubits=N_QUBITS,
        backend='lightning.qubit', use_skip=True).to(DEVICE)
    results['B_quantum_skip'] = train_and_eval(model_B, loaders, epochs=epochs)
    print(f"    R²={results['B_quantum_skip']['r2']:.4f}  "
          f"RMSE={results['B_quantum_skip']['rmse']*label_std:.4f} pKi")

    # ── C: Quantum reservoir, NO skip (honest quantum eval) ──────────────────
    print("\n[C] Quantum reservoir, no skip (head reads only quantum features) …")
    model_C = ModelHybridFC_Reservoir(
        in_features=dims, out_features=1,
        qiskit_circuit=best_circuit, n_qubits=N_QUBITS,
        backend='lightning.qubit', use_skip=False).to(DEVICE)
    results['C_quantum_noskip'] = train_and_eval(model_C, loaders, epochs=epochs)
    print(f"    R²={results['C_quantum_noskip']['r2']:.4f}  "
          f"RMSE={results['C_quantum_noskip']['rmse']*label_std:.4f} pKi")

    # ── D: Random circuit, NO skip ────────────────────────────────────────────
    print("\n[D] Random G3 circuit (non-RFD), no skip …")
    model_D = ModelHybridFC_Reservoir(
        in_features=dims, out_features=1,
        qiskit_circuit=random_circuit, n_qubits=N_QUBITS,
        backend='lightning.qubit', use_skip=False).to(DEVICE)
    results['D_random_noskip'] = train_and_eval(model_D, loaders, epochs=epochs)
    print(f"    R²={results['D_random_noskip']['r2']:.4f}  "
          f"RMSE={results['D_random_noskip']['rmse']*label_std:.4f} pKi")

    # ── 4. Summary table ──────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("ABLATION RESULTS (test set, normalised-label RMSE × std → pKi RMSE)")
    print("="*72)
    rows = []
    labels_map = {
        'A_classical':     'A  Classical MLP (no quantum)',
        'B_quantum_skip':  'B  Quantum + skip  (original)',
        'C_quantum_noskip':'C  Quantum, no skip  [honest]',
        'D_random_noskip': 'D  Random circuit, no skip   ',
    }
    for key, label in labels_map.items():
        m = results[key]
        rmse_pki = m['rmse'] * label_std
        print(f"  {label}  R²={m['r2']:.4f}  Pearson={m['pearson']:.4f}"
              f"  RMSE={rmse_pki:.4f} pKi")
        rows.append({'condition': label.strip(), **{k: round(v, 4) for k, v in m.items()},
                     'rmse_pki': round(rmse_pki, 4)})
    print("="*72)

    gap_AB = results['B_quantum_skip']['r2'] - results['A_classical']['r2']
    gap_CA = results['C_quantum_noskip']['r2'] - results['A_classical']['r2']
    # D - C: positive means random circuit outperforms RFD-selected (bad for RFD)
    gap_DC = results['D_random_noskip']['r2'] - results['C_quantum_noskip']['r2']

    print(f"\nDiagnosis:")
    print(f"  B - A (quantum gain with skip)        = {gap_AB:+.4f}  "
          + ("← skip dominates: quantum mostly ignored" if abs(gap_AB) < 0.03 else "← meaningful quantum gain with skip"))
    print(f"  C - A (quantum gain, no skip)          = {gap_CA:+.4f}  "
          + ("← quantum features alone add real signal" if gap_CA > 0.02
             else "← quantum bottleneck: 18 measurements can't represent 153-dim features"))
    print(f"  D - C (random vs RFD, no skip)         = {gap_DC:+.4f}  "
          + ("← RFD selection helps quantum-only performance" if gap_DC < -0.01
             else ("← random circuit beats RFD-selected: RFD optimises diversity, not predictive power"
                   if gap_DC > 0.01 else "← RFD selection has negligible effect")))

    # ── 5. Save results ───────────────────────────────────────────────────────
    out_csv = os.path.join(_qf_dir, 'ablation_results.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    conds = [r['condition'] for r in rows]
    r2s   = [r['r2'] for r in rows]
    colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
    bars = ax.barh(conds, r2s, color=colors, height=0.5)
    ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=10)
    ax.set_xlabel('Test R²', fontsize=12)
    ax.set_title('Ablation Study — Quantum Reservoir vs Classical Baseline', fontsize=13)
    ax.set_xlim(0, max(r2s) * 1.15)
    ax.axvline(results['A_classical']['r2'], color='black', linestyle='--',
               linewidth=1.2, label='Classical baseline')
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(_qf_dir, 'ablation_r2_bar.png'), dpi=150)
    plt.close(fig)
    print(f"Saved -> {os.path.join(_qf_dir, 'ablation_r2_bar.png')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-circuits', type=int, default=10,
                        help='circuits to generate for RFD pre-selection (default: 10)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='training epochs per condition (default: 50)')
    args = parser.parse_args()
    main(n_circuits=args.n_circuits, epochs=args.epochs)
