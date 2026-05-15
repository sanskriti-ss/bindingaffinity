#!/usr/bin/env python
"""
best_circuit_study.py

Three-part follow-up analysis of the best quantum reservoir circuit found in
the gate variance study (gate=100, seed=2, circuit_idx=27, R²=0.4763).

Part A — Stability  (~5 min):
    Evaluate the fixed best circuit over 15 different train/val sub-splits.
    The holdout (797 gradient-clean IDs) is always the same; only the 80/20
    sub-split within the 4519-ID train pool varies.  Tests whether the
    +0.0011 quantum gain is robust to the particular sub-split chosen.

Part B — Qubits  (~8 min):
    Compare n_qubits in {6, 8, 10}.  For each, generate 20 fresh G3-100
    circuits, RFD-select the top-3, evaluate with Ridge on holdout.
    Tests whether additional qubits unlock more signal.
uh
Part C — Readout  (~4 min):
    Fix the 6-qubit best circuit; compare:
      - Ridge (current approach, 5 alpha-CV splits)
      - MLP  (Linear 64→32→1, GELU, Dropout 0.2, Adam, early-stop)
            run 5 independent torch seeds for variance estimate.
    Tests whether the Ridge linear constraint is a bottleneck.

Outputs saved to:  best_circuit_study_<timestamp>/
    stability_results.csv  /  stability.png
    qubit_results.csv      /  qubits.png
    readout_results.csv    /  readout.png
    summary.txt

Run from the quantum_fusion/ directory:
    python best_circuit_study.py
    python best_circuit_study.py --parts A B        # run only A and B
    python best_circuit_study.py --parts C          # run readout only
"""

import os, sys, random, argparse, math, warnings
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)

# ── path setup ────────────────────────────────────────────────────────────────
_qf_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _qf_dir)

from main_train import FusionDataset
from testing_random_unitaries import (
    generate_g3_random_circuits,
    reservoir_feature_diversity,
    extract_quantum_features,
)

# ── known-best circuit identity (from gate_variance_study intermediate run) ──
BEST_GATE_COUNT   = 100
BEST_SEED         = 2
BEST_CIRCUIT_IDX  = 27
BEST_POOL_SIZE    = 50    # pool used when circuit was discovered
BEST_N_QUBITS     = 6

# ── data paths ────────────────────────────────────────────────────────────────
_dcnn_npz       = os.path.join(_qf_dir, 'refined_3dcnn_features.npz')
_sgcnn_npz      = os.path.join(_qf_dir, 'refined_sgcnn_features.npz')
_hdf_path       = os.path.join(_qf_dir, 'refined_all.hdf')
_train_hdf_path = os.path.join(_qf_dir, 'sgcnn_train.hdf')
_val_hdf_path   = os.path.join(_qf_dir, 'sgcnn_val.hdf')


# ── data loading ──────────────────────────────────────────────────────────────
def _load_raw_arrays():
    """Load features/labels/ID-sets from disk once; return raw arrays."""
    with h5py.File(_hdf_path, 'r') as hf:
        hdf_ids    = list(hf.keys())
        hdf_labels = {pid: float(hf[pid].attrs['affinity']) for pid in hdf_ids}

    npz_3d = np.load(_dcnn_npz, allow_pickle=False)
    npz_sg = np.load(_sgcnn_npz, allow_pickle=False)

    valid     = sorted(set(hdf_labels) & set(npz_3d.files) & set(npz_sg.files))
    feat_3d   = np.stack([npz_3d[pid] for pid in valid], axis=0).astype(np.float32)
    feat_sg   = np.stack([npz_sg[pid] for pid in valid], axis=0).astype(np.float32)
    labels    = np.array([hdf_labels[pid] for pid in valid], dtype=np.float32)
    valid_idx = {pid: i for i, pid in enumerate(valid)}

    with h5py.File(_train_hdf_path, 'r') as hf:
        embed_train_ids = set(hf.keys())
    with h5py.File(_val_hdf_path, 'r') as hf:
        embed_val_ids   = set(hf.keys())

    train_pool_pids = [pid for pid in valid if pid in embed_train_ids]
    holdout_pids    = [pid for pid in valid if pid in embed_val_ids]

    return feat_3d, feat_sg, labels, valid_idx, train_pool_pids, holdout_pids, hdf_labels


def load_data_seeded(raw_arrays, data_seed=42):
    """
    Build train / val / holdout FusionDatasets from the raw arrays using a
    given split seed.  The holdout (797 gradient-clean IDs) is always fixed;
    only the 80/20 sub-split of the 4519-ID train pool varies.
    """
    feat_3d, feat_sg, labels, valid_idx, train_pool_pids, holdout_pids, hdf_labels = raw_arrays

    tp_arr    = np.array(train_pool_pids)
    tp_idx    = np.arange(len(tp_arr))
    tp_labels = np.array([hdf_labels[pid] for pid in train_pool_pids], dtype=np.float32)
    n_bins    = min(10, max(2, int(np.sqrt(len(tp_arr)))))
    bins      = pd.qcut(tp_labels, q=n_bins, labels=False, duplicates='drop')
    tr_local, sel_local = train_test_split(
        tp_idx, test_size=0.20, random_state=data_seed,
        shuffle=True, stratify=bins,
    )
    tr_idx  = np.array([valid_idx[pid] for pid in tp_arr[tr_local]])
    sel_idx = np.array([valid_idx[pid] for pid in tp_arr[sel_local]])
    ho_idx  = np.array([valid_idx[pid] for pid in holdout_pids])

    sc_3d = StandardScaler().fit(feat_3d[tr_idx])
    sc_sg = StandardScaler().fit(feat_sg[tr_idx])
    f3 = sc_3d.transform(feat_3d).astype(np.float32)
    fs = sc_sg.transform(feat_sg).astype(np.float32)
    features = np.hstack([f3, fs])  # (N, 64)

    label_mean = float(labels[tr_idx].mean())
    label_std  = float(labels[tr_idx].std()) + 1e-8
    labels_n   = (labels - label_mean) / label_std

    n     = len(features)
    empty = np.zeros((n, 0), dtype=np.float32)

    def _ds(idx):
        return FusionDataset(
            torch.tensor(features[idx], dtype=torch.float32),
            torch.tensor(empty[idx],    dtype=torch.float32),
            torch.tensor(labels_n[idx], dtype=torch.float32),
        )

    return _ds(tr_idx), _ds(sel_idx), _ds(ho_idx)


# ── circuit reproduction ──────────────────────────────────────────────────────
def get_best_circuit(n_qubits=BEST_N_QUBITS):
    """Reproduce circuit_idx=27 from (gate=100, seed=2, pool=50)."""
    random.seed(BEST_SEED)
    np.random.seed(BEST_SEED)
    torch.manual_seed(BEST_SEED)
    circuits = generate_g3_random_circuits(
        n_qubits, num_gates=BEST_GATE_COUNT, num_circuits=BEST_POOL_SIZE
    )
    return circuits[BEST_CIRCUIT_IDX]


# ── feature helpers ───────────────────────────────────────────────────────────
def _get_pca_arrays(ds):
    X = np.concatenate([ds.sgcnn_features.numpy(), ds.cnn3d_features.numpy()], axis=1)
    y = ds.labels.numpy().flatten()
    return X, y


def _ridge_eval(X_tr, y_tr, X_ho, y_ho):
    """Fit Ridge on X_tr/y_tr and return R² on X_ho/y_ho.  Assumes scaled input."""
    ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000], cv=5)
    ridge.fit(X_tr, y_tr)
    preds = ridge.predict(X_ho)
    r2    = r2_score(y_ho, preds)
    return r2, ridge.alpha_, preds


# ── full single-circuit evaluation ───────────────────────────────────────────
def evaluate_circuit(qc, train_ds, holdout_ds, n_qubits):
    """
    Quantum+classical Ridge evaluation of one circuit.
    Returns a metrics dict with r2, r2_baseline, r2_gain, rmse, mae, pearson,
    spearman, best_alpha.
    """
    X_tr, y_tr = _get_pca_arrays(train_ds)
    X_ho, y_ho = _get_pca_arrays(holdout_ds)

    # ── quantum features ──
    q_tr = extract_quantum_features(qc, X_tr, n_qubits)
    q_ho = extract_quantum_features(qc, X_ho, n_qubits)

    Xq_tr = np.concatenate([X_tr, q_tr], axis=1)
    Xq_ho = np.concatenate([X_ho, q_ho], axis=1)

    sc = StandardScaler().fit(Xq_tr)
    r2, alpha, preds = _ridge_eval(sc.transform(Xq_tr), y_tr,
                                   sc.transform(Xq_ho), y_ho)

    # ── classical baseline ──
    sc_b = StandardScaler().fit(X_tr)
    r2_b, _, base_preds = _ridge_eval(sc_b.transform(X_tr), y_tr,
                                      sc_b.transform(X_ho), y_ho)

    return {
        'r2':          r2,
        'r2_baseline': r2_b,
        'r2_gain':     r2 - r2_b,
        'rmse':        math.sqrt(mean_squared_error(y_ho, preds)),
        'mae':         mean_absolute_error(y_ho, preds),
        'pearson':     float(pearsonr(y_ho, preds)[0]),
        'spearman':    float(spearmanr(y_ho, preds)[0]),
        'best_alpha':  float(alpha),
    }


# ── MLP readout ───────────────────────────────────────────────────────────────
class _MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_mlp(X_tr, y_tr, X_va, y_va, in_dim, hidden=64, dropout=0.2,
               lr=5e-4, wd=1e-4, max_epochs=400, patience=30, seed=0):
    torch.manual_seed(seed)
    model  = _MLP(in_dim, hidden=hidden, dropout=dropout)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    loss_fn = nn.MSELoss()

    t_X = torch.tensor(X_tr, dtype=torch.float32)
    t_y = torch.tensor(y_tr, dtype=torch.float32)
    v_X = torch.tensor(X_va, dtype=torch.float32)
    v_y = torch.tensor(y_va, dtype=torch.float32)

    best_loss, best_state, stale = float('inf'), None, 0
    for ep in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss_fn(model(t_X), t_y).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(v_X), v_y).item()
        sched.step(vl)

        if vl < best_loss - 1e-6:
            best_loss  = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_circuit_mlp(qc, train_ds, val_ds, holdout_ds, n_qubits,
                          hidden=64, dropout=0.2, n_seeds=5):
    """
    MLP readout on the same quantum+classical features.
    Runs n_seeds torch seeds and returns mean/std R² on holdout.
    Also returns classical MLP baseline.
    """
    X_tr, y_tr = _get_pca_arrays(train_ds)
    X_va, y_va = _get_pca_arrays(val_ds)
    X_ho, y_ho = _get_pca_arrays(holdout_ds)

    q_tr = extract_quantum_features(qc, X_tr, n_qubits)
    q_va = extract_quantum_features(qc, X_va, n_qubits)
    q_ho = extract_quantum_features(qc, X_ho, n_qubits)

    Xq_tr = np.concatenate([X_tr, q_tr], axis=1)
    Xq_va = np.concatenate([X_va, q_va], axis=1)
    Xq_ho = np.concatenate([X_ho, q_ho], axis=1)

    sc = StandardScaler().fit(Xq_tr)
    Xq_tr_s = sc.transform(Xq_tr).astype(np.float32)
    Xq_va_s = sc.transform(Xq_va).astype(np.float32)
    Xq_ho_s = sc.transform(Xq_ho).astype(np.float32)

    in_dim = Xq_tr_s.shape[1]
    r2_vals = []
    for seed in range(n_seeds):
        mlp   = _train_mlp(Xq_tr_s, y_tr.astype(np.float32),
                            Xq_va_s, y_va.astype(np.float32),
                            in_dim, hidden=hidden, dropout=dropout, seed=seed)
        mlp.eval()
        with torch.no_grad():
            preds = mlp(torch.tensor(Xq_ho_s, dtype=torch.float32)).numpy()
        r2_vals.append(r2_score(y_ho, preds))

    # Classical MLP baseline (no quantum)
    sc_b   = StandardScaler().fit(X_tr)
    Xb_tr  = sc_b.transform(X_tr).astype(np.float32)
    Xb_va  = sc_b.transform(X_va).astype(np.float32)
    Xb_ho  = sc_b.transform(X_ho).astype(np.float32)

    in_dim_b = Xb_tr.shape[1]
    r2_base_vals = []
    for seed in range(n_seeds):
        mlp_b = _train_mlp(Xb_tr, y_tr.astype(np.float32),
                            Xb_va, y_va.astype(np.float32),
                            in_dim_b, hidden=hidden, dropout=dropout, seed=seed)
        mlp_b.eval()
        with torch.no_grad():
            preds_b = mlp_b(torch.tensor(Xb_ho, dtype=torch.float32)).numpy()
        r2_base_vals.append(r2_score(y_ho, preds_b))

    return {
        'r2_mean':      float(np.mean(r2_vals)),
        'r2_std':       float(np.std(r2_vals)),
        'r2_vals':      r2_vals,
        'r2_base_mean': float(np.mean(r2_base_vals)),
        'r2_base_std':  float(np.std(r2_base_vals)),
        'r2_base_vals': r2_base_vals,
        'r2_gain_mean': float(np.mean(r2_vals) - np.mean(r2_base_vals)),
    }


# ── Part A: Stability across split seeds ─────────────────────────────────────
def run_part_a(raw_arrays, output_dir, n_splits=15):
    print("\n" + "=" * 70)
    print("Part A — Stability: fixed best circuit × 15 split seeds")
    print("=" * 70)
    qc = get_best_circuit(n_qubits=BEST_N_QUBITS)
    print(f"Reproduced circuit: gate={BEST_GATE_COUNT}  seed={BEST_SEED}  "
          f"idx={BEST_CIRCUIT_IDX}  qubits={BEST_N_QUBITS}")

    rows = []
    for split_seed in tqdm(range(n_splits), desc='Split seeds', ascii=True):
        train_ds, val_ds, holdout_ds = load_data_seeded(raw_arrays, data_seed=split_seed)
        m = evaluate_circuit(qc, train_ds, holdout_ds, BEST_N_QUBITS)
        m['split_seed'] = split_seed
        rows.append(m)
        tqdm.write(f"  seed={split_seed:2d}  "
                   f"R²={m['r2']:.6f}  gain={m['r2_gain']:+.6f}  "
                   f"base={m['r2_baseline']:.6f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'stability_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    print("\n--- Stability summary ---")
    print(f"  R² (quantum+classical): mean={df['r2'].mean():.6f}  "
          f"std={df['r2'].std():.6f}  "
          f"min={df['r2'].min():.6f}  max={df['r2'].max():.6f}")
    print(f"  R² (classical only):    mean={df['r2_baseline'].mean():.6f}  "
          f"std={df['r2_baseline'].std():.6f}")
    print(f"  Gain (quantum delta):   mean={df['r2_gain'].mean():+.6f}  "
          f"std={df['r2_gain'].std():.6f}  "
          f"always_positive={int((df['r2_gain'] > 0).sum())}/{len(df)}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    seeds = df['split_seed'].values

    ax = axes[0]
    ax.plot(seeds, df['r2'].values,          'o-', color='steelblue', lw=1.8, ms=7, label='Quantum+Classical')
    ax.plot(seeds, df['r2_baseline'].values, 's--', color='darkorange', lw=1.5, ms=7, label='Classical only')
    ax.fill_between(seeds,
                    df['r2'].mean() - df['r2'].std(),
                    df['r2'].mean() + df['r2'].std(),
                    alpha=0.15, color='steelblue')
    ax.set_xlabel('Data split seed', fontsize=11)
    ax.set_ylabel('R² on holdout (797 IDs)', fontsize=11)
    ax.set_title('Stability: R² vs split seed\n(fixed best circuit, gate=100, nq=6)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(seeds, df['r2_gain'].values, color=['seagreen' if g > 0 else 'tomato'
                                                for g in df['r2_gain'].values],
           alpha=0.8, edgecolor='k', linewidth=0.5)
    ax.axhline(0, color='black', lw=1.0)
    ax.axhline(df['r2_gain'].mean(), color='purple', lw=1.5, linestyle='--',
               label=f"Mean gain = {df['r2_gain'].mean():+.4f}")
    ax.set_xlabel('Data split seed', fontsize=11)
    ax.set_ylabel('R² gain (quantum - classical)', fontsize=11)
    ax.set_title('Quantum gain per split seed', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'stability.png')
    plt.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")
    return df


# ── Part B: Qubit count scaling ────────────────────────────────────────────────
def run_part_b(raw_arrays, output_dir,
               qubit_counts=(6, 8, 10), n_circuits_pool=20, top_k=3):
    print("\n" + "=" * 70)
    print("Part B — Qubits: n_qubits in {" + ", ".join(map(str, qubit_counts)) + "}")
    print(f"         pool={n_circuits_pool} circuits/qubit count, top-{top_k} by RFD")
    print("=" * 70)

    train_ds, val_ds, holdout_ds = load_data_seeded(raw_arrays, data_seed=42)
    X_tr, _ = _get_pca_arrays(train_ds)

    rows = []
    for nq in qubit_counts:
        print(f"\n--- n_qubits={nq} ---")
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        circuits = generate_g3_random_circuits(nq, num_gates=BEST_GATE_COUNT,
                                               num_circuits=n_circuits_pool)

        # RFD pre-select
        scored = []
        for i, qc in enumerate(tqdm(circuits, desc=f'RFD nq={nq}', ascii=True)):
            score = reservoir_feature_diversity(qc, nq)
            scored.append((score, i, qc))
        scored.sort(key=lambda t: t[0], reverse=True)
        selected = [(idx, qc) for (_, idx, qc) in scored[:top_k]]
        rfd_map  = {idx: sc for (sc, idx, _) in scored}

        for rank, (idx, qc) in enumerate(
                tqdm(selected, desc=f'Eval nq={nq}', ascii=True), start=1):
            m = evaluate_circuit(qc, train_ds, holdout_ds, nq)
            m['n_qubits']     = nq
            m['circuit_idx']  = idx
            m['circuit_rank'] = rank
            m['rfd_score']    = rfd_map[idx]
            rows.append(m)
            tqdm.write(f"  nq={nq}  rank={rank}  idx={idx}  "
                       f"R²={m['r2']:.6f}  gain={m['r2_gain']:+.6f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'qubit_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    print("\n--- Qubit summary (best circuit per qubit count) ---")
    for nq in qubit_counts:
        sub  = df[df['n_qubits'] == nq]
        best = sub.loc[sub['r2'].idxmax()]
        print(f"  nq={nq:2d}  best R²={best['r2']:.6f}  "
              f"gain={best['r2_gain']:+.6f}  "
              f"base={best['r2_baseline']:.6f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    best_r2  = [df[df['n_qubits'] == nq]['r2'].max()          for nq in qubit_counts]
    mean_r2  = [df[df['n_qubits'] == nq]['r2'].mean()         for nq in qubit_counts]
    base_r2  = [df[df['n_qubits'] == nq]['r2_baseline'].mean() for nq in qubit_counts]
    x = np.arange(len(qubit_counts))
    ax.plot(x, best_r2,  'o-', color='steelblue',  ms=10, lw=2,   label='Best R²')
    ax.plot(x, mean_r2,  's--', color='dodgerblue', ms=8,  lw=1.5, label='Mean R²')
    ax.plot(x, base_r2,  'D:',  color='darkorange',  ms=8,  lw=1.5, label='Classical baseline')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{nq}\nqubits' for nq in qubit_counts], fontsize=11)
    ax.set_ylabel('R² on holdout', fontsize=11)
    ax.set_title('Qubit Scaling: R² vs n_qubits\n(G3-100 circuits, top-3 by RFD)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    best_gain = [df[df['n_qubits'] == nq]['r2_gain'].max()  for nq in qubit_counts]
    mean_gain = [df[df['n_qubits'] == nq]['r2_gain'].mean() for nq in qubit_counts]
    ax.bar(x - 0.18, best_gain, width=0.35, label='Best gain',
           color='steelblue', alpha=0.8, edgecolor='k', linewidth=0.5)
    ax.bar(x + 0.18, mean_gain, width=0.35, label='Mean gain',
           color='dodgerblue', alpha=0.6, edgecolor='k', linewidth=0.5)
    ax.axhline(0, color='black', lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{nq}\nqubits' for nq in qubit_counts], fontsize=11)
    ax.set_ylabel('R² gain (quantum - classical)', fontsize=11)
    ax.set_title('Quantum gain per qubit count', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'qubits.png')
    plt.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")
    return df


# ── Part C: MLP vs Ridge readout ──────────────────────────────────────────────
def run_part_c(raw_arrays, output_dir, n_mlp_seeds=5):
    print("\n" + "=" * 70)
    print("Part C — Readout: Ridge vs MLP on fixed best circuit")
    print(f"         MLP: Linear 64→32→1, GELU, Dropout 0.2, {n_mlp_seeds} seeds")
    print("=" * 70)

    qc = get_best_circuit(n_qubits=BEST_N_QUBITS)
    train_ds, val_ds, holdout_ds = load_data_seeded(raw_arrays, data_seed=42)

    # Ridge (single deterministic evaluation)
    print("\n  Evaluating Ridge readout ...")
    ridge_m = evaluate_circuit(qc, train_ds, holdout_ds, BEST_N_QUBITS)
    print(f"  Ridge:  R²={ridge_m['r2']:.6f}  "
          f"gain={ridge_m['r2_gain']:+.6f}  "
          f"base={ridge_m['r2_baseline']:.6f}")

    # MLP (n_seeds runs for variance)
    print(f"\n  Evaluating MLP readout ({n_mlp_seeds} seeds) ...")
    mlp_m = evaluate_circuit_mlp(qc, train_ds, val_ds, holdout_ds, BEST_N_QUBITS,
                                  n_seeds=n_mlp_seeds)
    print(f"  MLP:   R²={mlp_m['r2_mean']:.6f} ± {mlp_m['r2_std']:.6f}  "
          f"gain={mlp_m['r2_gain_mean']:+.6f}  "
          f"base={mlp_m['r2_base_mean']:.6f} ± {mlp_m['r2_base_std']:.6f}")

    rows = [
        {'readout': 'Ridge', 'type': 'quantum+classical',
         'r2_mean': ridge_m['r2'],  'r2_std': 0.0,
         'r2_baseline': ridge_m['r2_baseline'], 'r2_gain': ridge_m['r2_gain']},
        {'readout': 'Ridge', 'type': 'classical only',
         'r2_mean': ridge_m['r2_baseline'], 'r2_std': 0.0,
         'r2_baseline': ridge_m['r2_baseline'], 'r2_gain': 0.0},
        {'readout': 'MLP',   'type': 'quantum+classical',
         'r2_mean': mlp_m['r2_mean'], 'r2_std': mlp_m['r2_std'],
         'r2_baseline': mlp_m['r2_base_mean'], 'r2_gain': mlp_m['r2_gain_mean']},
        {'readout': 'MLP',   'type': 'classical only',
         'r2_mean': mlp_m['r2_base_mean'], 'r2_std': mlp_m['r2_base_std'],
         'r2_baseline': mlp_m['r2_base_mean'], 'r2_gain': 0.0},
    ]
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'readout_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels    = ['Ridge\nClassical', 'Ridge\nQuantum+Class.', 'MLP\nClassical', 'MLP\nQuantum+Class.']
    r2_means  = [ridge_m['r2_baseline'], ridge_m['r2'],
                 mlp_m['r2_base_mean'],  mlp_m['r2_mean']]
    r2_errs   = [0.0, 0.0, mlp_m['r2_base_std'], mlp_m['r2_std']]
    colors    = ['darkorange', 'steelblue', 'darkorange', 'steelblue']
    hatches   = ['', '', '///', '///']
    x = np.arange(4)

    ax = axes[0]
    bars = ax.bar(x, r2_means, yerr=r2_errs, capsize=6,
                  color=colors, alpha=0.80, edgecolor='k', linewidth=0.7,
                  error_kw={'elinewidth': 2})
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('R² on holdout', fontsize=11)
    ax.set_title('Readout Comparison: R²\n(fixed best circuit, gate=100, nq=6)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, r2_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax = axes[1]
    gain_labels = ['Ridge', 'MLP']
    gains_mean  = [ridge_m['r2_gain'],    mlp_m['r2_gain_mean']]
    gains_err   = [0.0,                  mlp_m['r2_std'] + mlp_m['r2_base_std']]
    gains_color = ['steelblue' if g > 0 else 'tomato' for g in gains_mean]
    xg = np.arange(2)
    ax.bar(xg, gains_mean, yerr=gains_err, capsize=8,
           color=gains_color, alpha=0.85, edgecolor='k', linewidth=0.7,
           error_kw={'elinewidth': 2})
    ax.axhline(0, color='black', lw=1.0)
    ax.set_xticks(xg)
    ax.set_xticklabels(gain_labels, fontsize=12)
    ax.set_ylabel('R² gain (quantum - classical)', fontsize=11)
    ax.set_title('Quantum gain: Ridge vs MLP\n(error bar = propagated std)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'readout.png')
    plt.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")
    return df, ridge_m, mlp_m


# ── summary writer ────────────────────────────────────────────────────────────
def write_summary(output_dir, results):
    lines = [
        "=" * 70,
        "BEST CIRCUIT STUDY — SUMMARY",
        "=" * 70,
        f"Best circuit: gate_count={BEST_GATE_COUNT}  seed={BEST_SEED}  "
        f"circuit_idx={BEST_CIRCUIT_IDX}  n_qubits={BEST_N_QUBITS}",
        f"Baseline R² (classical Ridge, split_seed=42): "
        f"{results.get('baseline_r2', 'n/a')}",
        "",
    ]
    if 'a' in results:
        df = results['a']
        lines += [
            "Part A — Stability (15 split seeds, fixed circuit):",
            f"  R² quantum+classical:  {df['r2'].mean():.6f} ± {df['r2'].std():.6f}  "
            f"[{df['r2'].min():.6f}, {df['r2'].max():.6f}]",
            f"  R² classical only:     {df['r2_baseline'].mean():.6f} ± {df['r2_baseline'].std():.6f}",
            f"  Gain:                  {df['r2_gain'].mean():+.6f} ± {df['r2_gain'].std():.6f}",
            f"  Splits w/ positive gain: {int((df['r2_gain'] > 0).sum())}/15",
            "",
        ]
    if 'b' in results:
        df = results['b']
        lines += ["Part B — Qubit Scaling:"]
        for nq in sorted(df['n_qubits'].unique()):
            sub  = df[df['n_qubits'] == nq]
            best = sub.loc[sub['r2'].idxmax()]
            lines.append(f"  nq={nq:2d}  best R²={best['r2']:.6f}  "
                         f"mean R²={sub['r2'].mean():.6f}  "
                         f"gain={best['r2_gain']:+.6f}")
        lines.append("")
    if 'c' in results:
        r_df, ridge_m, mlp_m = results['c']
        lines += [
            "Part C — Readout Comparison (fixed best circuit, split_seed=42):",
            f"  Ridge  quantum+classical: R²={ridge_m['r2']:.6f}  "
            f"gain={ridge_m['r2_gain']:+.6f}",
            f"  Ridge  classical only:   R²={ridge_m['r2_baseline']:.6f}",
            f"  MLP    quantum+classical: R²={mlp_m['r2_mean']:.6f} ± {mlp_m['r2_std']:.6f}  "
            f"gain={mlp_m['r2_gain_mean']:+.6f}",
            f"  MLP    classical only:   R²={mlp_m['r2_base_mean']:.6f} ± {mlp_m['r2_base_std']:.6f}",
            "",
        ]
    lines += ["=" * 70]
    txt = "\n".join(lines)
    print("\n" + txt)
    path = os.path.join(output_dir, 'summary.txt')
    with open(path, 'w') as f:
        f.write(txt + "\n")
    print(f"\nSaved: {path}")


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Best circuit follow-up study: stability, qubits, readout.'
    )
    parser.add_argument(
        '--parts', nargs='+', default=['A', 'B', 'C'],
        choices=['A', 'B', 'C'], metavar='PART',
        help='Parts to run (default: A B C)',
    )
    parser.add_argument(
        '--n-splits', type=int, default=15,
        help='Number of data sub-split seeds for Part A (default: 15)',
    )
    parser.add_argument(
        '--qubit-counts', nargs='+', type=int, default=[6, 8, 10],
        metavar='N', help='Qubit counts for Part B (default: 6 8 10)',
    )
    parser.add_argument(
        '--qubit-pool', type=int, default=20,
        help='Circuit pool size per qubit count in Part B (default: 20)',
    )
    parser.add_argument(
        '--qubit-topk', type=int, default=3,
        help='Circuits to keep after RFD in Part B (default: 3)',
    )
    parser.add_argument(
        '--mlp-seeds', type=int, default=5,
        help='MLP init seeds for Part C variance estimate (default: 5)',
    )
    args = parser.parse_args()

    parts = [p.upper() for p in args.parts]

    ts  = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out = os.path.join(_qf_dir, f'best_circuit_study_{ts}')
    os.makedirs(out, exist_ok=True)
    print(f"Output directory: {out}")
    print(f"Parts to run:     {parts}")
    print(f"Best circuit:     gate={BEST_GATE_COUNT}  seed={BEST_SEED}  "
          f"idx={BEST_CIRCUIT_IDX}  nq={BEST_N_QUBITS}")

    print("\nLoading raw feature arrays (once) ...")
    raw_arrays = _load_raw_arrays()
    print(f"  Train pool: {len(raw_arrays[4])}  Holdout: {len(raw_arrays[5])}")

    results = {}

    if 'A' in parts:
        results['a'] = run_part_a(raw_arrays, out, n_splits=args.n_splits)

    if 'B' in parts:
        results['b'] = run_part_b(
            raw_arrays, out,
            qubit_counts=tuple(args.qubit_counts),
            n_circuits_pool=args.qubit_pool,
            top_k=args.qubit_topk,
        )

    if 'C' in parts:
        results['c'] = run_part_c(raw_arrays, out, n_mlp_seeds=args.mlp_seeds)

    write_summary(out, results)
    print(f"\nAll outputs in: {out}/")


if __name__ == '__main__':
    main()
