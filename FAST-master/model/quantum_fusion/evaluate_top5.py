#!/usr/bin/env python
"""
evaluate_top25.py

1. Loads pre-computed features (same dataset as main_train.py)
2. Generates 100 G3 circuits, selects top-25 by RFD expressibility
3. Trains ModelHybridFC_Reservoir for each of the 25 circuits (50 epochs)
4. Saves top5_unitary_results.csv  (circuit index, RFD, RMSE, MAE, R², Adj R², Pearson r, Spearman ρ)
5. Performs quartile analysis: compares top-25% circuits vs other quartiles
6. Plots scatter (predicted vs actual) for best/worst and existing best_model.pth

Run from quantum_fusion/ directory:
    python evaluate_top5.py
"""

import os, sys, math, glob
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
from datetime import datetime
from tqdm import tqdm

# ── imports from main_train ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_train import (
    load_with_model_features,
    FusionDataset,
    ModelHybridFC_Reservoir,
    evaluate_model,
)
from testing_random_unitaries import (
    generate_g3_random_circuits,
    preselect_circuits_by_expressibility,
)

# ── Config ───────────────────────────────────────────────────────────────────
N_QUBITS   = 6
DEPTH      = 10
EPOCHS     = 50
BATCH_SIZE = 64
LR         = 3e-4
TOP_K      = 25           # Train on top-25 by RFD
NUM_CIRCS  = 100          # Generate 100 circuits for quartile analysis
DEVICE     = torch.device('cpu')   # PennyLane simulators are CPU-only

_qf_dir    = os.path.dirname(os.path.abspath(__file__))
_dcnn_npz  = os.path.join(_qf_dir, 'refined_3dcnn_features.npz')
_sgcnn_npz = os.path.join(_qf_dir, 'refined_sgcnn_features.npz')
OUT_DIR    = _qf_dir   # write outputs next to this script


def _train_one(model, loaders, epochs=EPOCHS, lr=LR, device=DEVICE):
    """Train a single reservoir model, return best val RMSE and its test metrics."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()

    best_val  = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for sg, c3, y in loaders['train']:
            x = torch.cat([sg, c3], dim=1).to(device)
            y = y.to(device)
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

    model.load_state_dict(best_state)
    return best_val, best_state


def _get_predictions(model, loader, device=DEVICE):
    """Return (preds, labels) numpy arrays (de-normalised not needed here)."""
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for sg, c3, y in loader:
            x = torch.cat([sg, c3], dim=1).to(device)
            out = model(x)
            preds.extend(out.cpu().numpy().flatten())
            labs.extend(y.cpu().numpy().flatten())
    return np.array(preds), np.array(labs)


def main():
    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("Loading features …")
    sgcnn_features, cnn3d_features, labels, complex_ids = load_with_model_features(
        max_samples=6000,
        dcnn_npz=_dcnn_npz  if os.path.exists(_dcnn_npz)  else None,
        sgcnn_npz=_sgcnn_npz if os.path.exists(_sgcnn_npz) else None,
    )

    n_samples = len(labels)
    n_train   = int(0.70 * n_samples)
    n_val     = int(0.15 * n_samples)

    train_idx = np.arange(0, n_train)
    val_idx   = np.arange(n_train, n_train + n_val)
    test_idx  = np.arange(n_train + n_val, n_samples)

    label_mean = float(labels[train_idx].mean())
    label_std  = float(labels[train_idx].std()) + 1e-8
    print(f"Label stats — mean={label_mean:.3f}  std={label_std:.3f}")

    def _norm(y): return (y - label_mean) / label_std

    datasets = {}
    for split, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        datasets[split] = {
            'sg': sgcnn_features[idx],
            'c3': cnn3d_features[idx],
            'y':  _norm(labels[idx]),
        }

    loaders = {}
    for split in ['train', 'val', 'test']:
        ds = FusionDataset(datasets[split]['sg'], datasets[split]['c3'],
                           datasets[split]['y'])
        loaders[split] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split == 'train'))

    dims = datasets['train']['sg'].shape[1] + datasets['train']['c3'].shape[1]
    print(f"Input dim: {dims}   Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    # ── 2. Generate & rank circuits ──────────────────────────────────────────
    print(f"\nGenerating {NUM_CIRCS} G3 circuits and selecting top {TOP_K} by RFD …")
    circuits = generate_g3_random_circuits(N_QUBITS, DEPTH, num_circuits=NUM_CIRCS)
    indexed  = preselect_circuits_by_expressibility(circuits, N_QUBITS, top_k=TOP_K)
    print(f"Top-{TOP_K} circuit indices (by RFD): {[i for i,_ in indexed]}")

    # ── 3. Train each circuit ────────────────────────────────────────────────
    results = []
    best_r2       = -np.inf
    best_preds    = None
    best_labs     = None
    best_circ_idx = None

    for rank, (circ_idx, circuit) in enumerate(indexed, 1):
        print(f"\n{'='*60}")
        print(f"Circuit {circ_idx}  (rank {rank}/{TOP_K})")
        print(f"{'='*60}")

        model = ModelHybridFC_Reservoir(
            in_features=dims,
            out_features=1,
            qiskit_circuit=circuit,
            n_qubits=N_QUBITS,
            backend='lightning.qubit',
        ).to(DEVICE)

        _, best_state = _train_one(model, loaders)
        model.load_state_dict(best_state)

        # Evaluate on test set
        preds, labs = _get_predictions(model, loaders['test'])
        rmse  = math.sqrt(mean_squared_error(labs, preds))
        mae   = mean_absolute_error(labs, preds)
        r2    = r2_score(labs, preds)
        pear  = pearsonr(labs, preds)[0]
        spear = spearmanr(labs, preds)[0]

        # Adjusted R²: 1 - (1-R²)*(n-1)/(n-p-1)
        n_test = len(labs)
        adj_r2 = 1 - (1 - r2) * (n_test - 1) / (n_test - dims - 1)

        # Convert RMSE/MAE back to pKi units
        rmse_pki = rmse * label_std
        mae_pki  = mae  * label_std

        print(f"  Test  R²={r2:.4f}  Adj-R²={adj_r2:.4f}  Pearson={pear:.4f}  "
              f"Spearman={spear:.4f}  RMSE={rmse_pki:.4f} pKi  MAE={mae_pki:.4f} pKi")

        results.append({
            'rank':          rank,
            'circuit_index': circ_idx,
            'r2':            round(r2,     4),
            'adj_r2':        round(adj_r2, 4),
            'pearson_r':     round(pear,   4),
            'spearman_rho':  round(spear,  4),
            'rmse_pki':      round(rmse_pki, 4),
            'mae_pki':       round(mae_pki,  4),
            'rmse_norm':     round(rmse, 4),
            'mae_norm':      round(mae,  4),
        })

        if r2 > best_r2:
            best_r2       = r2
            best_preds    = preds
            best_labs     = labs
            best_circ_idx = circ_idx
            best_state_save = best_state

    # ── 4. Save results CSV ──────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, 'top5_unitary_results.csv')
    df = pd.DataFrame(results).sort_values('r2', ascending=False).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV → {csv_path}")
    print(df.to_string(index=False))

    # ── 4b. Circuit diagram PNGs + gate-sequence CSV ──────────────────────────
    circs_dir = os.path.join(OUT_DIR, 'top5_circuit_diagrams')
    os.makedirs(circs_dir, exist_ok=True)

    gate_rows = []
    for rank, (circ_idx, circuit) in enumerate(indexed, 1):
        r2_val = next((r['r2'] for r in results if r['circuit_index'] == circ_idx), None)

        # ── PNG diagram ────────────────────────────────────────────────────
        try:
            fig_c = circuit.draw('mpl', fold=-1, style={'backgroundcolor': '#FFFFFF'})
            fig_c.suptitle(
                f"G3 Circuit #{circ_idx}  (rank {rank}/{TOP_K},  R²={r2_val})",
                fontsize=11, fontweight='bold', y=1.01,
            )
            diag_path = os.path.join(circs_dir, f'circuit_{circ_idx}_rank{rank}.png')
            fig_c.savefig(diag_path, dpi=150, bbox_inches='tight')
            plt.close(fig_c)
            print(f"  Saved circuit diagram → {diag_path}")
        except Exception as e:
            print(f"  Could not draw circuit #{circ_idx}: {e}")

        # ── Gate-sequence rows ─────────────────────────────────────────────
        for step, instruction in enumerate(circuit.data):
            gate   = instruction.operation
            qubits = [circuit.find_bit(q).index for q in instruction.qubits]
            gate_rows.append({
                'circuit_index': circ_idx,
                'rank':          rank,
                'r2':            r2_val,
                'step':          step,
                'gate':          gate.name,
                'qubits':        ','.join(str(q) for q in qubits),
                'n_qubits_involved': len(qubits),
            })

    gates_csv = os.path.join(OUT_DIR, 'top5_circuit_gates.csv')
    pd.DataFrame(gate_rows).to_csv(gates_csv, index=False)
    print(f"Saved gate-sequence CSV → {gates_csv}  ({len(gate_rows)} gate entries)")

    # ── 5. Scatter plot (best circuit + optional existing best_model) ─────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Quantum Fusion — Binding Affinity Predictions', fontsize=14, fontweight='bold')

    # ── Left panel: best circuit from this run ────────────────────────────
    ax = axes[0]
    labs_pki  = best_labs  * label_std + label_mean
    preds_pki = best_preds * label_std + label_mean
    vmin = min(labs_pki.min(), preds_pki.min()) - 0.3
    vmax = max(labs_pki.max(), preds_pki.max()) + 0.3

    ax.scatter(labs_pki, preds_pki, alpha=0.4, s=18, color='steelblue', edgecolors='none')
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1.5, label='y = x')
    ax.set_xlabel('Experimental  −log K  (pKi)', fontsize=11)
    ax.set_ylabel('Predicted  −log K  (pKi)', fontsize=11)
    ax.set_title(f'Best circuit (#{best_circ_idx})  R²={best_r2:.4f}  Pearson r={pearsonr(labs_pki, preds_pki)[0]:.4f}',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    # ── Right panel: existing best_model.pth if present ───────────────────
    ax2 = axes[1]
    existing_ckpt = None
    runs = sorted(glob.glob(os.path.join(_qf_dir, 'reservoir_run_*')))
    if runs:
        candidate = os.path.join(runs[-1], 'best_model.pth')
        if os.path.exists(candidate):
            existing_ckpt = candidate

    if existing_ckpt:
        print(f"\nLoading existing best_model from: {existing_ckpt}")
        # We need to rebuild the model with a circuit — reuse the best circuit from this run
        model_prev = ModelHybridFC_Reservoir(
            in_features=dims,
            out_features=1,
            qiskit_circuit=indexed[0][1],   # use top-1 circuit for architecture
            n_qubits=N_QUBITS,
            backend='lightning.qubit',
        ).to(DEVICE)
        state = torch.load(existing_ckpt, map_location=DEVICE, weights_only=False)
        # state may be full state_dict or wrapped
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        try:
            model_prev.load_state_dict(state, strict=True)
            prev_preds, prev_labs = _get_predictions(model_prev, loaders['test'])
            prev_labs_pki  = prev_labs  * label_std + label_mean
            prev_preds_pki = prev_preds * label_std + label_mean
            prev_r2   = r2_score(prev_labs, prev_preds)
            prev_pear = pearsonr(prev_labs, prev_preds)[0]
            vmin2 = min(prev_labs_pki.min(), prev_preds_pki.min()) - 0.3
            vmax2 = max(prev_labs_pki.max(), prev_preds_pki.max()) + 0.3
            ax2.scatter(prev_labs_pki, prev_preds_pki, alpha=0.4, s=18,
                        color='darkorange', edgecolors='none')
            ax2.plot([vmin2, vmax2], [vmin2, vmax2], 'r--', lw=1.5, label='y = x')
            ax2.set_title(f'Existing best_model.pth  R²={prev_r2:.4f}  Pearson r={prev_pear:.4f}',
                          fontsize=10)
            ax2.set_xlim(vmin2, vmax2)
            ax2.set_ylim(vmin2, vmax2)
            ax2.set_aspect('equal', adjustable='box')
            ax2.grid(True, alpha=0.3)
            print(f"  Existing model → Test R²={prev_r2:.4f}  Pearson={prev_pear:.4f}")
        except Exception as e:
            ax2.text(0.5, 0.5, f'Could not load\nbest_model.pth\n{e}',
                     ha='center', va='center', transform=ax2.transAxes, fontsize=9)
            print(f"  Could not load existing model: {e}")
    else:
        ax2.text(0.5, 0.5, 'No existing\nbest_model.pth found',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=11)

    ax2.set_xlabel('Experimental  −log K  (pKi)', fontsize=11)
    ax2.set_ylabel('Predicted  −log K  (pKi)', fontsize=11)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    scatter_path = os.path.join(OUT_DIR, 'scatter_best.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot → {scatter_path}")

    # ── 6. Bar chart of R² across top-5 ──────────────────────────────────────
    fig2, ax3 = plt.subplots(figsize=(8, 4))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    x_pos   = np.arange(len(results))
    width   = 0.35
    bars_r2 = ax3.bar(x_pos - width/2, [r['r2']     for r in results],
                      width, color=colors[:len(results)], edgecolor='white', label='R²')
    bars_ar = ax3.bar(x_pos + width/2, [r['adj_r2'] for r in results],
                      width, color=colors[:len(results)], edgecolor='white',
                      alpha=0.55, hatch='//', label='Adj R²')
    for bar, r in zip(bars_r2, results):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{r['r2']:.4f}", ha='center', va='bottom', fontsize=8)
    for bar, r in zip(bars_ar, results):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{r['adj_r2']:.4f}", ha='center', va='bottom', fontsize=8)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"Circuit #{r['circuit_index']}" for r in results])
    ax3.set_ylabel('Test R² / Adj R²', fontsize=11)
    ax3.set_title(f'Top-{TOP_K} G3 Circuits — R² and Adjusted R² Comparison', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, min(1.0, max(r['r2'] for r in results) + 0.10))
    ax3.axhline(y=max(r['r2'] for r in results), color='red', linestyle='--',
                alpha=0.5, linewidth=1, label=f"Best R²={max(r['r2'] for r in results):.4f}")
    ax3.legend(fontsize=9)
    ax3.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    bar_path = os.path.join(OUT_DIR, 'top5_r2_bar.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved bar chart → {bar_path}")

    # ── 7. Quartile Analysis: Performance by Top-25% vs Other Quartiles ──────
    # Sort results by R² to establish quartiles
    results_sorted = sorted(results, key=lambda x: x['r2'], reverse=True)
    n_results = len(results_sorted)
    q25_idx = max(1, n_results // 4)  # Top 25% (Q1)
    q50_idx = max(1, n_results // 2)  # Top 50% (Q2)
    q75_idx = max(1, 3 * n_results // 4)  # Top 75% (Q3)

    top_25pct = results_sorted[:q25_idx]
    q2_25to50 = results_sorted[q25_idx:q50_idx]
    q3_50to75 = results_sorted[q50_idx:q75_idx]
    bottom_25  = results_sorted[q75_idx:]

    quartiles = {
        'Top 25%':     [r['adj_r2'] for r in top_25pct],
        'Q2 (25-50%)': [r['adj_r2'] for r in q2_25to50],
        'Q3 (50-75%)': [r['adj_r2'] for r in q3_50to75],
        'Bottom 25%':  [r['adj_r2'] for r in bottom_25],
    }

    # Build circuit lookup by index
    circuit_by_idx = {circ_idx: circuit for circ_idx, circuit in indexed}

    # Print quartile statistics
    print(f"\n{'='*70}")
    print(f"QUARTILE ANALYSIS (based on {n_results} trained circuits)")
    print(f"{'='*70}")
    for q_name, vals in quartiles.items():
        if vals:
            print(f"\n{q_name:15} ({len(vals):2d} circuits):")
            print(f"  Adj-R² range:   {min(vals):.4f} – {max(vals):.4f}")
            print(f"  Mean Adj-R²:    {np.mean(vals):.4f}")
            print(f"  Std  Adj-R²:    {np.std(vals):.4f}")

    # Best vs Worst comparison
    best_circuit = results_sorted[0]
    worst_circuit = results_sorted[-1]
    print(f"\n{'='*70}")
    print(f"BEST VS WORST CIRCUIT")
    print(f"{'='*70}")
    print(f"BEST:  Circuit #{best_circuit['circuit_index']}  (rank {best_circuit['rank']})")
    print(f"       R²={best_circuit['r2']:.4f}  Adj-R²={best_circuit['adj_r2']:.4f}")
    print(f"       Pearson={best_circuit['pearson_r']:.4f}  Spearman={best_circuit['spearman_rho']:.4f}")
    print(f"\nWORST: Circuit #{worst_circuit['circuit_index']}  (rank {worst_circuit['rank']})")
    print(f"       R²={worst_circuit['r2']:.4f}  Adj-R²={worst_circuit['adj_r2']:.4f}")
    print(f"       Pearson={worst_circuit['pearson_r']:.4f}  Spearman={worst_circuit['spearman_rho']:.4f}")
    delta_adj_r2 = best_circuit['adj_r2'] - worst_circuit['adj_r2']
    print(f"\nDelta Adj-R²:  {delta_adj_r2:+.4f}  ({100*delta_adj_r2/abs(worst_circuit['adj_r2']):+.1f}% relative gain)")
    print(f"{'='*70}\n")

    # ── 7a. Box plot + Violin plot comparison ────────────────────────────────
    fig_q, axes_q = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    ax_box = axes_q[0]
    bp = ax_box.boxplot(
        [quartiles[q] for q in ['Top 25%', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Bottom 25%']],
        labels=['Top 25%', 'Q2\n(25-50%)', 'Q3\n(50-75%)', 'Bottom 25%'],
        patch_artist=True,
        widths=0.6,
    )
    colors_q = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
    for patch, color in zip(bp['boxes'], colors_q):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_box.set_ylabel('Adjusted R²', fontsize=11)
    ax_box.set_title('Quartile Comparison: Adjusted R² Distribution', fontsize=12, fontweight='bold')
    ax_box.grid(True, axis='y', alpha=0.3)
    ax_box.set_ylim(min(min(v) for v in quartiles.values()) - 0.05,
                    max(max(v) for v in quartiles.values()) + 0.05)

    # Violin plot
    ax_vio = axes_q[1]
    vio_data = [quartiles[q] for q in ['Top 25%', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Bottom 25%']]
    positions = [1, 2, 3, 4]
    parts = ax_vio.violinplot(vio_data, positions=positions, showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], colors_q):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax_vio.set_xticks(positions)
    ax_vio.set_xticklabels(['Top 25%', 'Q2\n(25-50%)', 'Q3\n(50-75%)', 'Bottom 25%'])
    ax_vio.set_ylabel('Adjusted R²', fontsize=11)
    ax_vio.set_title('Violin Plot: Adjusted R² by Quartile', fontsize=12, fontweight='bold')
    ax_vio.grid(True, axis='y', alpha=0.3)
    ax_vio.set_ylim(min(min(v) for v in vio_data) - 0.05,
                    max(max(v) for v in vio_data) + 0.05)

    plt.tight_layout()
    quartile_path = os.path.join(OUT_DIR, 'quartile_comparison.png')
    plt.savefig(quartile_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved quartile comparison → {quartile_path}")

    # ── 7b. Best vs Worst scatter (side-by-side) ────────────────────────────
    # Rebuild models for best and worst circuits to get their predictions
    fig_bw, axes_bw = plt.subplots(1, 2, figsize=(14, 6))
    fig_bw.suptitle('Best Circuit vs Worst Circuit — Predictions', fontsize=14, fontweight='bold')

    for ax_idx, circuit_info_dict in enumerate([best_circuit, worst_circuit]):
        circ_idx = circuit_info_dict['circuit_index']
        circuit = circuit_by_idx[circ_idx]
        title_prefix = 'BEST' if ax_idx == 0 else 'WORST'

        model_q = ModelHybridFC_Reservoir(
            in_features=dims,
            out_features=1,
            qiskit_circuit=circuit,
            n_qubits=N_QUBITS,
            backend='lightning.qubit',
        ).to(DEVICE)

        # Quick retrain to get final predictions
        _, best_state_q = _train_one(model_q, loaders)
        model_q.load_state_dict(best_state_q)
        preds_q, labs_q = _get_predictions(model_q, loaders['test'])
        labs_pki_q = labs_q * label_std + label_mean
        preds_pki_q = preds_q * label_std + label_mean

        ax = axes_bw[ax_idx]
        vmin = min(labs_pki_q.min(), preds_pki_q.min()) - 0.3
        vmax = max(labs_pki_q.max(), preds_pki_q.max()) + 0.3

        ax.scatter(labs_pki_q, preds_pki_q, alpha=0.4, s=18, edgecolors='none',
                   color='steelblue' if title_prefix == 'BEST' else 'coral')
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1.5, label='y = x')
        ax.set_xlabel('Experimental  −log K  (pKi)', fontsize=11)
        ax.set_ylabel('Predicted  −log K  (pKi)', fontsize=11)
        ax.set_title(
            f"{title_prefix}: Circuit #{circuit_info_dict['circuit_index']}  "
            f"R²={circuit_info_dict['r2']:.4f}  Adj-R²={circuit_info_dict['adj_r2']:.4f}",
            fontsize=11
        )
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    best_worst_path = os.path.join(OUT_DIR, 'best_vs_worst_scatter.png')
    plt.savefig(best_worst_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved best vs worst scatter → {best_worst_path}")

    print("\nDone!")


if __name__ == '__main__':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    main()
