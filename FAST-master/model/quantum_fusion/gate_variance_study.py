#!/usr/bin/env python
"""
gate_variance_study.py

Studies how R² performance and its variance scale with the number of G3
gates per circuit.

Methodology
-----------
For each gate count in GATE_COUNTS = [30, 100, 300, 1000]:
  - Run N_SHOTS = 3 independent trials with different random seeds
  - Each trial:
      1. Generate N_CIRCUITS=100 G3 random circuits with that gate count
      2. RFD-preselect top TOP_K=25 by Reservoir Feature Diversity
      3. Evaluate each selected circuit with Ridge regression readout
  - Record per-circuit R² for all TOP_K circuits × N_SHOTS seeds
    → TOP_K * N_SHOTS = 75 R² values per gate count

Outputs (saved to gate_variance_study_<timestamp>/)
-----------------------------------------------------
  gate_variance_results.csv   — per-circuit R² for every run
  variance_violin_box.png     — violin + box plots, R² by gate count
  variance_mean_std.png       — mean ± std R² vs gate count
  best_per_seed.png           — best R² per seed, grouped by gate count

Runtime estimate (6-qubit lightning.qubit, N_CIRCUITS=100)
-----------------------------------------------------------
  30   gates → ~2 min   per shot
  100  gates → ~5 min   per shot
  300  gates → ~12 min  per shot
  1000 gates → ~35 min  per shot
  Total (4 gate counts × 3 shots) ≈ 3-4 hours

  Use --n-circuits 30 --top-k 10 for a quick smoke-test (<15 min).

Run from the quantum_fusion/ directory:
    python gate_variance_study.py
    python gate_variance_study.py --gate-counts 30 100 300 --shots 2
    python gate_variance_study.py --n-circuits 30 --top-k 10   # fast test
"""

import os, sys, random, argparse
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── path setup ───────────────────────────────────────────────────────────────
_qf_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _qf_dir)

from main_train import FusionDataset
from testing_random_unitaries import (
    generate_g3_random_circuits,
    reservoir_feature_diversity,
    run_circuits_and_evaluate,
)

# ── Default config (all overridable via CLI) ─────────────────────────────────
_GATE_COUNTS = [30, 100, 300, 1000]
_SHOT_SEEDS  = [0, 1, 2]       # 3 independent trials per gate count
_N_CIRCUITS  = 100              # random circuit pool per trial
_TOP_K       = 25               # kept after RFD filter
_N_QUBITS    = 6
_DATA_SEED   = 42               # fixed data-split seed (same for all runs)

_dcnn_npz       = os.path.join(_qf_dir, 'refined_3dcnn_features.npz')
_sgcnn_npz      = os.path.join(_qf_dir, 'refined_sgcnn_features.npz')
_hdf_path       = os.path.join(_qf_dir, 'refined_all.hdf')
_train_hdf_path = os.path.join(_qf_dir, 'sgcnn_train.hdf')  # 4519 IDs  gradient updates applied
_val_hdf_path   = os.path.join(_qf_dir, 'sgcnn_val.hdf')    # 797  IDs  forward pass only (gradient-clean)


# ── Data loading (HDF + NPZ, bypasses RDKit/SDF requirement) ─────────────────
def load_data():
    """
    Load features and labels from refined_all.hdf + NPZ embeddings.
    Returns train / selection-val / holdout FusionDatasets.

    Upstream embedding leakage is eliminated by honouring the same
    train/val split used when training the 3DCNN and SGCNN embedding
    models:

      sgcnn_train.hdf (4519 IDs) — gradient updates were applied to
        these complexes by both upstream models.  They form the quantum
        training pool and are sub-split 80/20 into quantum-train and
        selection-val.

      sgcnn_val.hdf (797 IDs) — both upstream models ran ONLY forward
        passes on these complexes (early-stopping validation).  No weight
        updates were ever applied.  Their embeddings are produced by
        models that never learned from them --> rigorous holdout.

    Feature layout: [3DCNN(10) | SGCNN(54)] = 64-dim per complex.
    """
    print("Loading data from HDF + NPZ ...")
    with h5py.File(_hdf_path, 'r') as hf:
        hdf_ids    = list(hf.keys())
        hdf_labels = {pid: float(hf[pid].attrs['affinity']) for pid in hdf_ids}

    npz_3d = np.load(_dcnn_npz, allow_pickle=False)
    npz_sg = np.load(_sgcnn_npz, allow_pickle=False)

    valid = sorted(set(hdf_labels) & set(npz_3d.files) & set(npz_sg.files))
    print(f"  HDF: {len(hdf_ids)}  3DCNN NPZ: {len(npz_3d.files)}  "
          f"SGCNN NPZ: {len(npz_sg.files)}  Intersection: {len(valid)}")

    feat_3d  = np.stack([npz_3d[pid] for pid in valid], axis=0).astype(np.float32)
    feat_sg  = np.stack([npz_sg[pid] for pid in valid], axis=0).astype(np.float32)
    labels   = np.array([hdf_labels[pid] for pid in valid], dtype=np.float32)
    valid_idx = {pid: i for i, pid in enumerate(valid)}

    # ── Use the HDF split from upstream model training → no embedding leakage ──
    with h5py.File(_train_hdf_path, 'r') as hf:
        embed_train_ids = set(hf.keys())   # 4519  gradient updates applied
    with h5py.File(_val_hdf_path, 'r') as hf:
        embed_val_ids   = set(hf.keys())   # 797  forward pass only (gradient-clean)

    train_pool_pids = [pid for pid in valid if pid in embed_train_ids]
    holdout_pids    = [pid for pid in valid if pid in embed_val_ids]
    print(f"  Embedding train pool: {len(train_pool_pids)}  "
          f"Gradient-clean holdout: {len(holdout_pids)}")

    # ── Sub-split train pool 80/20 into quantum-train / selection-val ──────
    tp_arr    = np.array(train_pool_pids)
    tp_idx    = np.arange(len(tp_arr))
    tp_labels = np.array([hdf_labels[pid] for pid in train_pool_pids], dtype=np.float32)
    n_bins_tp = min(10, max(2, int(np.sqrt(len(tp_arr)))))
    bins_tp   = pd.qcut(tp_labels, q=n_bins_tp, labels=False, duplicates='drop')
    tr_local, sel_local = train_test_split(
        tp_idx, test_size=0.20, random_state=_DATA_SEED,
        shuffle=True, stratify=bins_tp,
    )
    tr_idx  = np.array([valid_idx[pid] for pid in tp_arr[tr_local]])
    sel_idx = np.array([valid_idx[pid] for pid in tp_arr[sel_local]])
    ho_idx  = np.array([valid_idx[pid] for pid in holdout_pids])

    # ── Fit scalers on TRAIN ONLY, then transform all (no val/holdout stats) ──
    sc_3d   = StandardScaler().fit(feat_3d[tr_idx])
    sc_sg   = StandardScaler().fit(feat_sg[tr_idx])
    feat_3d = sc_3d.transform(feat_3d).astype(np.float32)
    feat_sg = sc_sg.transform(feat_sg).astype(np.float32)
    features = np.hstack([feat_3d, feat_sg])  # (N, 64)

    label_mean = float(labels[tr_idx].mean())
    label_std  = float(labels[tr_idx].std()) + 1e-8
    labels_norm = (labels - label_mean) / label_std

    # Empty cnn3d slot — run_circuits_and_evaluate concatenates both slots;
    # torch.cat([feats, empty], dim=1) is a no-op that preserves shape.
    n     = len(valid)
    empty = np.zeros((n, 0), dtype=np.float32)

    def _ds(idx):
        return FusionDataset(
            torch.tensor(features[idx], dtype=torch.float32),
            torch.tensor(empty[idx],    dtype=torch.float32),
            torch.tensor(labels_norm[idx], dtype=torch.float32),
        )

    train_ds   = _ds(tr_idx)
    val_ds     = _ds(sel_idx)     # selection-val: used to rank circuits / tune alpha
    holdout_ds = _ds(ho_idx)     # true holdout: 797 gradient-clean IDs
    print(f"  Train: {len(train_ds)}  Val(selection): {len(val_ds)}  "
          f"Holdout: {len(holdout_ds)}")
    print(f"  Label stats (train): mean={label_mean:.3f}  std={label_std:.3f}")
    return train_ds, val_ds, holdout_ds


# ── RFD pre-selection (captures scores for CSV) ──────────────────────────────
def _preselect_with_scores(circuits, n_qubits, top_k):
    """
    Like preselect_circuits_by_expressibility() but also returns a dict
    mapping circuit_index -> RFD score for downstream recording.
    """
    print(f"  Computing RFD for {len(circuits)} circuits ...")
    scored = []
    for i, qc in enumerate(tqdm(circuits, desc='RFD', ascii=True)):
        score = reservoir_feature_diversity(qc, n_qubits)
        scored.append((score, i, qc))

    scored.sort(key=lambda t: t[0], reverse=True)
    selected  = scored[:top_k]
    rfd_map   = {idx: sc for (sc, idx, _qc) in scored}
    indexed   = [(idx, qc) for (_, idx, qc) in selected]
    return indexed, rfd_map


# ── single (gate_count, seed) trial ─────────────────────────────────────────
def run_trial(gate_count, seed, train_ds, val_ds, holdout_ds, n_circuits, top_k, n_qubits):
    """
    One full trial: generate circuits → RFD filter → Ridge evaluate.

    Circuit selection uses val_ds (Ridge trained on train, alpha picked by
    val R²).  Reported metrics are computed on holdout_ds which is NEVER
    used for selection — eliminating selection bias in the reported numbers.

    Returns a list of row dicts ready for DataFrame construction.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    circuits = generate_g3_random_circuits(
        n_qubits, num_gates=gate_count, num_circuits=n_circuits
    )

    indexed, rfd_map = _preselect_with_scores(circuits, n_qubits, top_k)

    results = run_circuits_and_evaluate(
        indexed, train_ds, val_ds, n_qubits=n_qubits,
        holdout_dataset=holdout_ds,
    )

    # results is already sorted by R² (descending)
    rows = []
    for rank, res in enumerate(results, start=1):
        rows.append({
            'gate_count':   gate_count,
            'seed':         seed,
            'circuit_rank': rank,          # 1 = best R² within this trial
            'circuit_idx':  res['circuit_idx'],
            'rfd_score':    rfd_map.get(res['circuit_idx'], float('nan')),
            'r2':           res['r2'],
            'rmse':         res['rmse'],
            'mae':          res['mae'],
            'pearson':      res['pearson'],
            'spearman':     res['spearman'],
            'r2_baseline':  res['r2_baseline'],
            'r2_gain':      res['r2_gain'],
        })
    return rows


# ── plotting ─────────────────────────────────────────────────────────────────
def plot_results(df, output_dir, gate_counts, shot_seeds, n_circuits, top_k):
    x_pos   = np.arange(len(gate_counts))
    x_labs  = [str(g) for g in gate_counts]
    colors  = plt.cm.viridis(np.linspace(0.15, 0.85, len(gate_counts)))

    # ── 1. Violin + box ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    data_by_gate = [df[df['gate_count'] == g]['r2'].values for g in gate_counts]

    ax = axes[0]
    vp = ax.violinplot(data_by_gate, positions=x_pos, showmeans=True, showmedians=True)
    for pc, c in zip(vp['bodies'], colors):
        pc.set_facecolor(c); pc.set_alpha(0.72)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{g}\ngates' for g in gate_counts], fontsize=11)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('R² Distribution per Gate Count\n(violin)', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(0, color='red', lw=1, linestyle=':')

    ax = axes[1]
    bp = ax.boxplot(data_by_gate, positions=x_pos, patch_artist=True, widths=0.55)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.72)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{g}\ngates' for g in gate_counts], fontsize=11)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('R² Distribution per Gate Count\n(box)', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(0, color='red', lw=1, linestyle=':')

    n_pts = top_k * len(shot_seeds)
    fig.suptitle(
        f'G3 Quantum Reservoir — R² Variance vs Gate Count\n'
        f'(pool={n_circuits} circuits/seed, top-{top_k} by RFD, '
        f'{len(shot_seeds)} seeds, {n_pts} total pts per gate count)',
        y=1.02, fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    out = os.path.join(output_dir, 'variance_violin_box.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {out}")

    # ── 2. Mean ± std line plot ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    means, stds, bests = [], [], []
    for g in gate_counts:
        sub = df[df['gate_count'] == g]['r2']
        means.append(sub.mean()); stds.append(sub.std()); bests.append(sub.max())
    means, stds = np.array(means), np.array(stds)

    ax.plot(x_pos, means, 'o-', color='steelblue', lw=2.2, ms=9, label='Mean R²')
    ax.fill_between(x_pos, means - stds, means + stds,
                    alpha=0.22, color='steelblue', label='±1 std dev')
    ax.plot(x_pos, bests, 's--', color='darkorange', lw=1.8, ms=8, label='Best R²')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{g}\ngates' for g in gate_counts], fontsize=11)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Mean ± Std R² vs Gate Count\n(all circuits × all seeds)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='red', lw=1, linestyle=':')
    plt.tight_layout()
    out = os.path.join(output_dir, 'variance_mean_std.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {out}")

    # ── 3. Best R² per seed (reproducibility across seeds) ────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = ['o', 's', '^', 'D', 'v', 'P']
    for j, seed in enumerate(sorted(shot_seeds)):
        sub = df[df['seed'] == seed]
        best_per_gate = [sub[sub['gate_count'] == g]['r2'].max()
                         for g in gate_counts]
        ax.plot(x_pos, best_per_gate,
                marker=markers[j % len(markers)], lw=1.8, ms=9,
                label=f'Seed {seed}')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{g}\ngates' for g in gate_counts], fontsize=11)
    ax.set_ylabel('Best R² (per seed)', fontsize=12)
    ax.set_title('Best Circuit R² per Seed vs Gate Count\n(statistical consistency)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, 'best_per_seed.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {out}")

    # ── 4. Per-seed mean R² with error bars ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    width  = 0.25
    seed_list = sorted(shot_seeds)
    for j, seed in enumerate(seed_list):
        sub  = df[df['seed'] == seed]
        vals = [sub[sub['gate_count'] == g]['r2'].values for g in gate_counts]
        m    = [v.mean() for v in vals]
        e    = [v.std()  for v in vals]
        pos  = x_pos + (j - (len(seed_list) - 1) / 2) * width
        ax.bar(pos, m, width=width * 0.9, yerr=e, capsize=4,
               alpha=0.75, label=f'Seed {seed}', error_kw={'elinewidth': 1.5})

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{g}\ngates' for g in gate_counts], fontsize=11)
    ax.set_ylabel('Mean R² (± std within seed)', fontsize=12)
    ax.set_title('R² per Gate Count, Grouped by Seed\n(bars = mean of top-K circuits)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(0, color='red', lw=1, linestyle=':')
    plt.tight_layout()
    out = os.path.join(output_dir, 'r2_by_seed_grouped.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {out}")


# ── entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='G3 quantum reservoir R² variance study across gate counts.'
    )
    parser.add_argument(
        '--gate-counts', nargs='+', type=int, default=_GATE_COUNTS,
        metavar='N',
        help=f'Gate counts to compare (default: {_GATE_COUNTS})',
    )
    parser.add_argument(
        '--shots', type=int, default=len(_SHOT_SEEDS),
        help=f'Number of independent seeds per gate count (default: {len(_SHOT_SEEDS)})',
    )
    parser.add_argument(
        '--n-circuits', type=int, default=_N_CIRCUITS,
        help=f'Random circuit pool size per trial (default: {_N_CIRCUITS})',
    )
    parser.add_argument(
        '--top-k', type=int, default=_TOP_K,
        help=f'Circuits to keep after RFD pre-selection (default: {_TOP_K})',
    )
    parser.add_argument(
        '--n-qubits', type=int, default=_N_QUBITS,
        help=f'Number of qubits (default: {_N_QUBITS})',
    )
    args = parser.parse_args()

    gate_counts = args.gate_counts
    shot_seeds  = list(range(args.shots))
    n_circuits  = args.n_circuits
    top_k       = args.top_k
    n_qubits    = args.n_qubits

    timestamp  = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(_qf_dir, f'gate_variance_study_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Gate counts:   {gate_counts}")
    print(f"Seeds (shots): {shot_seeds}")
    print(f"Circuits/shot: {n_circuits}  top-K: {top_k}  qubits: {n_qubits}")

    # ── load data ONCE (shared across all runs) ───────────────────────────
    print()
    train_ds, val_ds, holdout_ds = load_data()

    # ── main loop ─────────────────────────────────────────────────────────
    all_rows = []
    total    = len(gate_counts) * len(shot_seeds)
    run_no   = 0

    for gate_count in gate_counts:
        for seed in shot_seeds:
            run_no += 1
            print(f"\n{'='*65}")
            print(f"Run {run_no}/{total}  |  gate_count={gate_count}  seed={seed}")
            print(f"{'='*65}")

            rows = run_trial(
                gate_count, seed, train_ds, val_ds, holdout_ds,
                n_circuits=n_circuits, top_k=top_k, n_qubits=n_qubits,
            )
            all_rows.extend(rows)

            best  = max(r['r2'] for r in rows)
            mean  = np.mean([r['r2'] for r in rows])
            print(f"  >> Best R²={best:.4f}  Mean R²={mean:.4f}  "
                  f"(over top-{len(rows)} circuits)")

            # Save incrementally in case of crash
            pd.DataFrame(all_rows).to_csv(
                os.path.join(output_dir, 'gate_variance_results.csv'), index=False
            )

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(output_dir, 'gate_variance_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults table saved -> {csv_path}")

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("Summary — Mean R² ± std  (all circuits × all seeds per gate count)")
    print("="*65)
    summary = (
        df.groupby('gate_count')['r2']
          .agg(mean='mean', std='std', best='max', worst='min', n='count')
          .round(4)
    )
    print(summary.to_string())

    # ── plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_results(df, output_dir, gate_counts, shot_seeds,
                 n_circuits=n_circuits, top_k=top_k)

    print(f"\nAll outputs in: {output_dir}/")


if __name__ == '__main__':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    os.environ.setdefault('PYTHONUTF8', '1')
    main()
