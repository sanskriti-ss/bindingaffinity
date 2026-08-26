#!/usr/bin/env python
"""
replacing_multiple_layers_study.py
==================================

Stack multiple G3 unitary layers (fixed reservoir or param VQC) in a hybrid model.

Run from ``quantum_fusion/``::

    python replacing_multiple_layers_study.py --smoke --mode fixed --circuit-sharing both
    python replacing_multiple_layers_study.py --mode fixed --layer-sweep --gates-per-layer 3 \\
        --circuit-sharing independent --shots 3 --epochs 15
"""

from __future__ import annotations

import argparse
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
    sample_g3_circuits,
)
from experiment_io import (  # noqa: E402
    format_run_info,
    make_run_dir,
    setup_run_logging,
    write_run_config,
    write_run_info,
)


def plot_layer_comparison(df: pd.DataFrame, plots_dir: Path) -> None:
    """Primary layer-depth comparison: holdout R² vs n_quantum_layers."""
    if "n_quantum_layers" not in df.columns or df["n_quantum_layers"].nunique() < 1:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    layer_vals = sorted(df["n_quantum_layers"].unique().astype(int))
    x_pos = np.arange(len(layer_vals))

    mode = df["mode"].iloc[0] if "mode" in df.columns else ""
    sharing = (
        df["circuit_sharing"].iloc[0]
        if "circuit_sharing" in df.columns and df["circuit_sharing"].nunique() == 1
        else "mixed"
    )
    gates = int(df["gates_per_layer"].iloc[0]) if "gates_per_layer" in df.columns else ""
    n_shots = int(df.groupby("n_quantum_layers").size().max())

    fig, ax = plt.subplots(figsize=(11, 6))
    data_by_layer = [
        df[df["n_quantum_layers"] == g]["r2_holdout"].values for g in layer_vals
    ]

    parts = ax.violinplot(
        data_by_layer, positions=x_pos, showmeans=True, showmedians=True, widths=0.65,
    )
    for body in parts["bodies"]:
        body.set_facecolor("#4C72B0")
        body.set_alpha(0.55)

    bp = ax.boxplot(
        data_by_layer, positions=x_pos, widths=0.22, patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#DD8452")
        patch.set_alpha(0.45)

    means = [np.mean(d) for d in data_by_layer]
    sems = [np.std(d, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0 for d in data_by_layer]
    ax.errorbar(
        x_pos, means, yerr=sems, fmt="o-", color="#2171B5", lw=2, ms=8,
        capsize=4, label="Mean ± SEM", zorder=5,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(g) for g in layer_vals])
    ax.set_xlabel("Number of quantum layers replaced", fontsize=12)
    ax.set_ylabel("Holdout R²", fontsize=12)
    ax.set_title(
        f"Holdout R² vs layer depth ({mode}, {gates} gates/layer, {sharing})\n"
        f"up to {n_shots} shot(s) per layer count",
        fontsize=12,
    )
    ax.axhline(0, color="red", ls=":", lw=1)
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "layer_comparison_holdout_r2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_results(df: pd.DataFrame, plots_dir: Path) -> None:
    """Generate comparison plots for multi-layer study results."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    sharing_vals = sorted(df["circuit_sharing"].unique())

    plot_layer_comparison(df, plots_dir)

    if len(sharing_vals) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [
            df[df["circuit_sharing"] == s]["r2_holdout"].values
            for s in sharing_vals
        ]
        parts = ax.violinplot(data, positions=range(len(sharing_vals)), showmeans=True)
        for body in parts["bodies"]:
            body.set_alpha(0.7)
        ax.boxplot(data, positions=range(len(sharing_vals)), widths=0.15)
        ax.set_xticks(range(len(sharing_vals)))
        ax.set_xticklabels(sharing_vals)
        ax.set_ylabel("Holdout R²")
        ax.set_title("Holdout R²: independent vs shared G3 circuits")
        ax.axhline(0, color="red", ls=":", lw=1)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "r2_independent_vs_shared.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if "n_quantum_layers" in df.columns and df["n_quantum_layers"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped = df.groupby("n_quantum_layers")["r2_holdout"].agg(["mean", "std"])
        x = grouped.index.astype(int).values
        ax.errorbar(x, grouped["mean"], yerr=grouped["std"], fmt="o-", capsize=4)
        ax.set_xlabel("Number of quantum layers")
        ax.set_ylabel("Mean holdout R²")
        ax.set_title("Holdout R² vs stack depth (mean ± std)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "r2_by_layer_count.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    if "circuit_sharing" in df.columns and len(sharing_vals) > 1:
        for sharing in sharing_vals:
            sub = df[df["circuit_sharing"] == sharing]
            ax.scatter(sub["shot"], sub["r2_holdout"], label=sharing, alpha=0.8, s=60)
        ax.legend()
    elif "n_quantum_layers" in df.columns:
        for n_layers in sorted(df["n_quantum_layers"].unique()):
            sub = df[df["n_quantum_layers"] == n_layers]
            ax.scatter(
                sub["shot"], sub["r2_holdout"], label=f"{int(n_layers)} layers",
                alpha=0.8, s=60,
            )
        ax.legend(fontsize=8, ncol=2)
    else:
        ax.scatter(df["shot"], df["r2_holdout"], alpha=0.8)
    ax.set_xlabel("Shot index")
    ax.set_ylabel("Holdout R²")
    ax.set_title("Holdout R² by shot")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "violins_by_sharing.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_from_csv(csv_path: Path, out_dir: Path | None = None) -> Path:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    plots_dir = Path(out_dir or csv_path.parent) / "plots"
    plot_results(df, plots_dir)
    print(f"Saved plots -> {plots_dir}")
    return plots_dir


def _layer_metadata(circuits, base_seed: int, sharing: str) -> List[dict]:
    entries = []
    for layer_idx, qc in enumerate(circuits):
        nh, nt, nc = gate_structure_summary(qc)
        seed = base_seed if sharing == "shared" else base_seed + layer_idx * 17
        entries.append({
            "layer_idx": layer_idx,
            "seed": seed,
            "n_h": nh,
            "n_t": nt,
            "n_cnot": nc,
            "fingerprint": circuit_fingerprint(qc),
            "n_quantum_params": count_vqc_params(qc),
        })
    return entries


def _append_csv_row(csv_path: Path, row: dict, *, write_header: bool) -> None:
    pd.DataFrame([row]).to_csv(
        csv_path, mode="w" if write_header else "a", header=write_header, index=False,
    )


def run_study(
    mode: str,
    *,
    n_quantum_layers: int = 2,
    layer_counts: Sequence[int] | None = None,
    gates_per_layer: int = 3,
    circuit_sharing: str = "both",
    shots: int = 1,
    n_qubits: int = 6,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 3e-4,
    base_seed: int = 2000,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    from train_eval import build_multi_layer_model, load_dataloaders, train_model  # noqa: E402

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_dir or make_run_dir(_QF_DIR / "results", mode)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    depths = list(layer_counts) if layer_counts is not None else [n_quantum_layers]
    sharing_modes = (
        ["independent", "shared"]
        if circuit_sharing == "both"
        else [circuit_sharing]
    )

    data = load_dataloaders(batch_size=batch_size)
    print(
        f"Data: {data.data_source}  train={data.n_train}  val={data.n_val}  "
        f"holdout={data.n_holdout}  in_features={data.in_features}"
    )
    print(f"Layer depths to run: {depths}")

    csv_path = out_dir / "multi_layer_results.csv"
    if csv_path.exists():
        csv_path.unlink()

    rows: List[dict] = []
    last_layer_meta: List[dict] = []
    csv_initialized = False

    for depth in depths:
        for sharing in sharing_modes:
            for shot in range(shots):
                circuit_seed = base_seed + shot * 101 + depth * 1009
                circuits = sample_g3_circuits(
                    n_qubits,
                    gates_per_layer,
                    depth,
                    circuit_seed,
                    sharing=sharing,  # type: ignore[arg-type]
                )
                layer_meta = _layer_metadata(circuits, circuit_seed, sharing)
                last_layer_meta = layer_meta
                fps = [m["fingerprint"] for m in layer_meta]

                print(
                    f"\n=== mode={mode} sharing={sharing} shot={shot} "
                    f"layers={depth} gates/layer={gates_per_layer} ==="
                )
                for m in layer_meta:
                    print(
                        f"  layer {m['layer_idx']}: H={m['n_h']} T={m['n_t']} "
                        f"CNOT={m['n_cnot']}"
                    )

                model = build_multi_layer_model(
                    mode, data.in_features, circuits, n_qubits,
                )
                metrics = train_model(
                    model, data, epochs=epochs, lr=lr, verbose=True,
                )
                row = {
                    "mode": mode,
                    "circuit_sharing": sharing,
                    "shot": shot,
                    "circuit_seed": circuit_seed,
                    "n_quantum_layers": depth,
                    "gates_per_layer": gates_per_layer,
                    "layer_fingerprints": "||".join(fps),
                    "n_quantum_params_total": sum(m["n_quantum_params"] for m in layer_meta),
                    **metrics,
                }
                rows.append(row)
                _append_csv_row(csv_path, row, write_header=not csv_initialized)
                csv_initialized = True
                print(
                    f"  holdout R²={metrics['r2_holdout']:.4f}  "
                    f"val R²={metrics['r2_val']:.4f}  "
                    f"time={metrics['train_time_s']:.1f}s"
                )

    df = pd.DataFrame(rows)

    config = {
        "timestamp": ts,
        "circuit_type": mode,
        "n_quantum_layers": depths[-1] if len(depths) == 1 else None,
        "layer_counts": depths,
        "gates_per_layer": gates_per_layer,
        "circuit_sharing": circuit_sharing,
        "shots": shots,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "n_qubits": n_qubits,
        "base_seed": base_seed,
        "data_source": data.data_source,
        "n_train": data.n_train,
        "n_val": data.n_val,
        "n_holdout": data.n_holdout,
        "layer_circuits": last_layer_meta,
    }
    write_run_config(out_dir, config)

    summary = [
        f"Layer depths:    {depths}",
        f"Best holdout R²: {df['r2_holdout'].max():.4f}",
        f"Mean holdout R²: {df['r2_holdout'].mean():.4f}",
        f"Best val R²:     {df['r2_val'].max():.4f}",
    ]
    if len(depths) > 1:
        by_layer = df.groupby("n_quantum_layers")["r2_holdout"].mean()
        summary.append("Mean holdout R² by layer count:")
        for n_layers, mean_r2 in by_layer.items():
            summary.append(f"  {int(n_layers)} layers: {mean_r2:.4f}")
    write_run_info(out_dir, format_run_info(config, summary_lines=summary))

    plot_results(df, out_dir / "plots")
    print(f"\nSaved CSV -> {csv_path}")
    print(f"Saved plots -> {out_dir / 'plots'}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description="Multi-layer G3 replacement study (fixed or param).",
    )
    p.add_argument("--mode", required=False, choices=["fixed", "param"])
    p.add_argument("--n-quantum-layers", type=int, default=2)
    p.add_argument(
        "--layer-counts", nargs="+", type=int, default=None,
        metavar="N",
        help="Sweep over multiple layer depths (e.g. 1 2 3 4 5 6 7 8 9 10).",
    )
    p.add_argument(
        "--layer-sweep",
        action="store_true",
        help="Preset: sweep layer depths 1 through 10.",
    )
    p.add_argument("--gates-per-layer", type=int, default=3)
    p.add_argument(
        "--circuit-sharing",
        choices=["independent", "shared", "both"],
        default="both",
    )
    p.add_argument("--shots", type=int, default=1)
    p.add_argument("--n-qubits", type=int, default=6)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--base-seed", type=int, default=2000)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick run: 2 layers, 3 gates/layer, 1 shot, 3 epochs.",
    )
    p.add_argument(
        "--plot-only",
        type=Path,
        metavar="CSV",
        default=None,
        help="Regenerate plots from an existing multi_layer_results.csv.",
    )
    args = p.parse_args()

    if args.plot_only is not None:
        plot_from_csv(args.plot_only, out_dir=args.output_dir)
        return

    if args.smoke:
        args.n_quantum_layers = 2
        args.gates_per_layer = 3
        args.shots = 1
        args.epochs = 3
        args.layer_counts = None
        args.layer_sweep = False

    if args.layer_sweep and args.layer_counts is not None:
        p.error("Use only one of --layer-sweep or --layer-counts")

    layer_counts = None
    if args.layer_sweep:
        layer_counts = list(range(1, 11))
    elif args.layer_counts is not None:
        layer_counts = args.layer_counts

    if args.mode is None:
        p.error("--mode is required unless using --plot-only")

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = make_run_dir(_QF_DIR / "results", args.mode)
    setup_run_logging(Path(out_dir))

    run_study(
        args.mode,
        n_quantum_layers=args.n_quantum_layers,
        layer_counts=layer_counts,
        gates_per_layer=args.gates_per_layer,
        circuit_sharing=args.circuit_sharing,
        shots=args.shots,
        n_qubits=args.n_qubits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_seed=args.base_seed,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
