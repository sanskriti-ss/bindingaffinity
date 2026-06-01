#!/usr/bin/env python
"""
param_vs_fixed_study.py
=======================

Compare end-to-end hybrid models on the same random G3 circuit draw:

  * ``fixed_e2e``  — :class:`ModelHybridFC_Reservoir` (fixed H/T/CNOT, train classical head)
  * ``param_vqc``  — :class:`ModelHybridFC_VQC` (trainable RY/RZ + classical head)

Future: ``fixed_ridge`` via ``--modes fixed_ridge`` (Ridge readout, not in smoke default).

Smoke test (user spec):
  2 replicas × gate counts [3, 10, 30, 100, 300] × modes [fixed_e2e, param_vqc]
  → CSV + violin plots of holdout R².

Run from ``quantum_fusion/``::

    python param_vs_fixed_study.py
    python param_vs_fixed_study.py --epochs 12 --gate-counts 3 10 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_QF_DIR = Path(__file__).resolve().parent
if str(_QF_DIR) not in sys.path:
    sys.path.insert(0, str(_QF_DIR))

from circuit_common import (  # noqa: E402
    circuit_fingerprint,
    count_vqc_params,
    gate_structure_summary,
    sample_g3_circuit,
)
from train_eval import build_model, load_dataloaders, train_model  # noqa: E402

DEFAULT_GATE_COUNTS = [3, 10, 30, 100, 300]
DEFAULT_MODES = ("fixed_e2e", "param_vqc")
REPLICA_SEEDS = (0, 1)


def plot_violins(df: pd.DataFrame, out_dir: Path, gate_counts: Sequence[int]) -> None:
    """Violin (+box) of holdout R² by gate count, coloured by mode."""
    modes = sorted(df["mode"].unique())
    colors = {"fixed_e2e": "#4C72B0", "param_vqc": "#DD8452", "fixed_ridge": "#55A868"}
    x_pos = np.arange(len(gate_counts))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, plot_kind in zip(axes, ("violin", "box")):
        for j, mode in enumerate(modes):
            offsets = x_pos + (j - (len(modes) - 1) / 2) * 0.28
            data = [
                df[(df["gate_count"] == g) & (df["mode"] == mode)]["r2_holdout"].values
                for g in gate_counts
            ]
            c = colors.get(mode, None)
            if plot_kind == "violin":
                parts = ax.violinplot(
                    data, positions=offsets, showmeans=True, showmedians=True,
                    widths=0.22,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(c)
                    body.set_alpha(0.72)
            else:
                bp = ax.boxplot(
                    data, positions=offsets, widths=0.20, patch_artist=True,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(c)
                    patch.set_alpha(0.72)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(g) for g in gate_counts])
        ax.set_xlabel("G3 gate count")
        ax.set_ylabel("Holdout R²")
        ax.set_title(f"R² by gate count ({plot_kind})")
        ax.axhline(0, color="red", ls=":", lw=1)
        ax.grid(True, axis="y", alpha=0.3)

    handles = [
        plt.Line2D([0], [0], color=colors.get(m, "gray"), lw=6, label=m)
        for m in modes
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(modes), bbox_to_anchor=(0.5, 1.02))
    n_rep = df.groupby(["gate_count", "mode"]).size().max()
    fig.suptitle(
        f"Parameterized vs fixed G3 reservoir (e2e)\n"
        f"up to {int(n_rep)} replicas per cell",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "violins_and_boxes_r2_by_gate_and_mode.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Single combined violin: x = gate_count, hue = mode (seaborn if available)
    try:
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(11, 5.5))
        plot_df = df.copy()
        plot_df["gate_count"] = plot_df["gate_count"].astype(str)
        sns.violinplot(
            data=plot_df,
            x="gate_count",
            y="r2_holdout",
            hue="mode",
            order=[str(g) for g in gate_counts],
            hue_order=list(modes),
            ax=ax,
            cut=0,
            inner="box",
        )
        ax.set_xlabel("G3 gate count")
        ax.set_ylabel("Holdout R²")
        ax.set_title("Holdout R²: fixed reservoir (e2e) vs variational G3")
        ax.axhline(0, color="red", ls=":", lw=1)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "seaborn_violin_r2.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass


def run_study(
    gate_counts: Sequence[int],
    modes: Sequence[str],
    replicas: int,
    *,
    n_qubits: int = 6,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 3e-4,
    output_dir: Path | None = None,
    base_seed: int = 1000,
) -> pd.DataFrame:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_dir or (_QF_DIR / "results" / f"param_vs_fixed_{ts}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataloaders(batch_size=batch_size)
    print(
        f"Data: {data.data_source}  train={data.n_train}  val={data.n_val}  "
        f"holdout={data.n_holdout}  in_features={data.in_features}"
    )

    rows = []
    for gate_count in gate_counts:
        for replica in range(replicas):
            circuit_seed = base_seed + replica + gate_count * 17
            qc = sample_g3_circuit(n_qubits, gate_count, seed=circuit_seed)
            fp = circuit_fingerprint(qc)
            nh, nt, nc = gate_structure_summary(qc)
            n_qparams = count_vqc_params(qc)

            for mode in modes:
                print(
                    f"\n=== gate={gate_count} replica={replica} mode={mode} "
                    f"(H={nh} T={nt} CNOT={nc}) ==="
                )
                model = build_model(
                    mode, data.in_features, qc, n_qubits,
                    backend="lightning.qubit",
                )
                metrics = train_model(
                    model, data,
                    epochs=epochs, lr=lr, verbose=True,
                )
                row = {
                    "gate_count": gate_count,
                    "replica": replica,
                    "circuit_seed": circuit_seed,
                    "mode": mode,
                    "circuit_fingerprint": fp,
                    "n_h_slots": nh,
                    "n_t_slots": nt,
                    "n_cnot": nc,
                    "n_gates_total": gate_count,
                    "n_quantum_params_if_vqc": n_qparams,
                    **metrics,
                }
                rows.append(row)
                print(
                    f"  holdout R²={metrics['r2_holdout']:.4f}  "
                    f"val R²={metrics['r2_val']:.4f}  "
                    f"time={metrics['train_time_s']:.1f}s"
                )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "param_vs_fixed_results.csv"
    df.to_csv(csv_path, index=False)

    cfg = {
        "timestamp": ts,
        "gate_counts": list(gate_counts),
        "modes": list(modes),
        "replicas": replicas,
        "n_qubits": n_qubits,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "data_source": data.data_source,
        "n_train": data.n_train,
        "n_val": data.n_val,
        "n_holdout": data.n_holdout,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    plot_violins(df, out_dir, gate_counts)
    print(f"\nSaved CSV -> {csv_path}")
    print(f"Saved plots -> {out_dir}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description="Parameterized vs fixed G3 hybrid models (e2e smoke/full).",
    )
    p.add_argument("--gate-counts", nargs="+", type=int, default=DEFAULT_GATE_COUNTS)
    p.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    p.add_argument("--replicas", type=int, default=2)
    p.add_argument("--n-qubits", type=int, default=6)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--base-seed", type=int, default=1000)
    args = p.parse_args()

    run_study(
        args.gate_counts,
        args.modes,
        args.replicas,
        n_qubits=args.n_qubits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
