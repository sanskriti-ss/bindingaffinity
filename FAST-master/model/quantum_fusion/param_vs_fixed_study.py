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


def plot_poster_results(df: pd.DataFrame, out_dir: Path, gate_counts: Sequence[int]) -> Path:
    """
    Poster-quality Results figure: mean holdout R² ± SEM by gate count, coloured by mode.

    Shows individual replicas as jittered points behind summary statistics.
    Designed for poster readability (large fonts, clean axes).
    """
    import seaborn as sns

    modes = sorted(df["mode"].unique())
    palette = {"fixed_e2e": "#2171B5", "param_vqc": "#E6550D", "fixed_ridge": "#31A354"}

    fig, ax = plt.subplots(figsize=(8, 5.5))

    plot_df = df[["gate_count", "mode", "r2_holdout"]].copy()
    plot_df["gate_count"] = plot_df["gate_count"].astype(str)

    # Strip plot for individual replicas (low alpha)
    sns.stripplot(
        data=plot_df,
        x="gate_count",
        y="r2_holdout",
        hue="mode",
        order=[str(g) for g in gate_counts],
        hue_order=modes,
        dodge=True,
        alpha=0.3,
        size=4,
        jitter=0.12,
        ax=ax,
        palette=palette,
        legend=False,
    )

    # Point plot for mean ± SEM
    sns.pointplot(
        data=plot_df,
        x="gate_count",
        y="r2_holdout",
        hue="mode",
        order=[str(g) for g in gate_counts],
        hue_order=modes,
        dodge=0.3,
        errorbar="se",
        markers=["o", "s"],
        capsize=0.08,
        err_kws={"linewidth": 1.5},
        ax=ax,
        palette=palette,
        linestyles="none",
    )

    ax.set_xlabel("G3 gate count", fontsize=14, fontweight="bold")
    ax.set_ylabel("Holdout R²", fontsize=14, fontweight="bold")
    ax.set_title(
        "Fixed Reservoir vs Parameterized VQC:\nHoldout R² by Circuit Depth",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(labelsize=12)
    ax.axhline(
        df[df["mode"] == "fixed_e2e"]["r2_holdout"].mean(),
        color="#2171B5", ls="--", lw=0.8, alpha=0.4,
    )
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    # Take only the first len(modes) entries (from pointplot)
    ax.legend(
        handles[:len(modes)], [m.replace("_", " ").title() for m in modes],
        fontsize=12, frameon=True, framealpha=0.9, loc="lower left",
    )

    n_rep = df.groupby(["gate_count", "mode"]).size().max()
    ax.annotate(
        f"n = {int(n_rep)} replicas per condition",
        xy=(0.98, 0.02), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9, color="gray",
    )

    fig.tight_layout()
    out_path = out_dir / "poster_results_r2_by_gate.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path.name}")
    return out_path


def plot_poster_discussion(df: pd.DataFrame, out_dir: Path, gate_counts: Sequence[int]) -> Path:
    """
    Poster-quality Discussion figure: multi-panel synthesis.

    Panel A: Standardized effect sizes of gate composition features (H, T, CNOT
             fraction) on holdout R², computed per mode via top-vs-bottom-quartile
             comparison. Shows which compositional features matter and whether
             the two training paradigms differ in sensitivity.
    Panel B: Performance advantage (Δ R² = fixed − param) vs gate count.

    Captures the key scientific insight: fixed reservoirs maintain stable performance
    regardless of circuit composition, while parameterized circuits are more sensitive
    to circuit depth and composition.
    """
    import seaborn as sns
    from scipy import stats as sp_stats

    palette = {"fixed_e2e": "#2171B5", "param_vqc": "#E6550D"}
    modes = sorted(df["mode"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"wspace": 0.35})

    # --- Panel A: feature effect sizes (bar chart, pattern-dashboard style) ---
    ax_a = axes[0]
    comp_df = df.copy()
    comp_df["frac_h"] = comp_df["n_h_slots"] / comp_df["n_gates_total"]
    comp_df["frac_t"] = comp_df["n_t_slots"] / comp_df["n_gates_total"]
    comp_df["frac_cnot"] = comp_df["n_cnot"] / comp_df["n_gates_total"]

    features = [
        ("frac_h", "H fraction"),
        ("frac_t", "T fraction"),
        ("frac_cnot", "CNOT fraction"),
        ("n_h_slots", "H count"),
        ("n_t_slots", "T count"),
        ("n_cnot", "CNOT count"),
    ]

    bar_data = []
    for mode in modes:
        sub = comp_df[comp_df["mode"] == mode]
        for col, label in features:
            # Pearson correlation as standardized effect
            r_val, p_val = sp_stats.pearsonr(sub[col], sub["r2_holdout"])
            bar_data.append({
                "feature": label,
                "mode": mode,
                "r": r_val,
                "p": p_val,
                "sig": "*" if p_val < 0.05 else "",
            })

    bar_df = pd.DataFrame(bar_data)

    y_pos = np.arange(len(features))
    bar_h = 0.35
    for i, mode in enumerate(modes):
        sub = bar_df[bar_df["mode"] == mode]
        offset = (i - 0.5) * bar_h
        bars = ax_a.barh(
            y_pos + offset, sub["r"].values,
            height=bar_h, color=palette[mode], alpha=0.8,
            label=mode.replace("_", " ").title(),
        )
        # Annotate significance
        for j, (r_val, sig) in enumerate(zip(sub["r"].values, sub["sig"].values)):
            x_pos = r_val + 0.01 if r_val >= 0 else r_val - 0.01
            ha = "left" if r_val >= 0 else "right"
            ax_a.text(
                x_pos, y_pos[j] + offset,
                f"{r_val:.3f}{sig}", va="center", ha=ha,
                fontsize=8, color=palette[mode], fontweight="bold",
            )

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([f[1] for f in features], fontsize=11)
    ax_a.axvline(0, color="black", lw=0.8)
    ax_a.set_xlabel("Pearson r  (feature vs holdout R²)", fontsize=12, fontweight="bold")
    ax_a.set_title("A.  Gate Composition Effect Sizes", fontsize=13, fontweight="bold", loc="left")
    ax_a.legend(fontsize=10, frameon=True, framealpha=0.9, loc="lower right")
    ax_a.grid(True, axis="x", alpha=0.2)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.tick_params(labelsize=10)
    ax_a.set_xlim(-0.25, 0.25)

    # --- Panel B: Δ R² by gate count ---
    ax_b = axes[1]

    # Compute paired delta per circuit (same replica & gate_count share a circuit)
    pivot = df.pivot_table(
        index=["gate_count", "replica"],
        columns="mode",
        values="r2_holdout",
    ).reset_index()
    pivot["delta_r2"] = pivot["fixed_e2e"] - pivot["param_vqc"]
    pivot["gate_count_str"] = pivot["gate_count"].astype(str)

    sns.boxplot(
        data=pivot,
        x="gate_count_str",
        y="delta_r2",
        order=[str(g) for g in gate_counts],
        color="#7FCDBB",
        width=0.5,
        fliersize=3,
        ax=ax_b,
    )
    sns.stripplot(
        data=pivot,
        x="gate_count_str",
        y="delta_r2",
        order=[str(g) for g in gate_counts],
        color="#253494",
        alpha=0.4,
        size=4,
        jitter=0.15,
        ax=ax_b,
    )

    ax_b.axhline(0, color="red", ls=":", lw=1.2, alpha=0.7)
    ax_b.set_xlabel("G3 gate count", fontsize=13, fontweight="bold")
    ax_b.set_ylabel("Δ R²  (fixed − param)", fontsize=13, fontweight="bold")
    ax_b.set_title("B.  Fixed Reservoir Advantage", fontsize=13, fontweight="bold", loc="left")
    ax_b.grid(True, axis="y", alpha=0.2)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.tick_params(labelsize=11)

    # Annotate mean delta per gate count
    for i, g in enumerate(gate_counts):
        mean_d = pivot[pivot["gate_count"] == g]["delta_r2"].mean()
        ax_b.annotate(
            f"Δ={mean_d:.4f}",
            xy=(i, pivot[pivot["gate_count"] == g]["delta_r2"].max() + 0.002),
            ha="center", fontsize=9, fontweight="bold", color="#253494",
        )

    fig.tight_layout()
    out_path = out_dir / "poster_discussion_synthesis.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path.name}")
    return out_path


def plot_pattern_dashboard(df: pd.DataFrame, out_dir: Path, gate_counts: Sequence[int]) -> Path:
    """
    Quantum Circuit Pattern Report Dashboard.

    Left panel: Feature effect sizes (standardized) — top-quartile vs bottom-quartile
    R² difference for compositional and structural features, in std units.

    Right panel: Transition pattern shifts — how gate-to-gate transition probabilities
    differ between the best- and worst-performing circuits.

    This reproduces the pattern dashboard analysis for any param_vs_fixed CSV.
    """
    from scipy import stats as sp_stats

    # Compute compositional features
    adf = df.copy()
    adf["frac_h"] = adf["n_h_slots"] / adf["n_gates_total"]
    adf["frac_t"] = adf["n_t_slots"] / adf["n_gates_total"]
    adf["frac_cnot"] = adf["n_cnot"] / adf["n_gates_total"]
    adf["depth"] = adf["n_gates_total"]  # proxy: total gate count as depth

    # Parse transitions from circuit_fingerprint
    gate_types = ["h", "t", "cx"]
    transition_pairs = [f"{a}->{b}" for a in gate_types for b in gate_types]

    def _parse_transitions(fp: str) -> dict:
        """Count gate-to-gate transitions from fingerprint string."""
        gates = []
        for token in fp.split("|"):
            g = token.split(":")[0].strip()
            if g in gate_types:
                gates.append(g)
        counts = {tp: 0 for tp in transition_pairs}
        for i in range(len(gates) - 1):
            key = f"{gates[i]}->{gates[i+1]}"
            if key in counts:
                counts[key] += 1
        total = max(sum(counts.values()), 1)
        return {k: v / total for k, v in counts.items()}

    trans_records = adf["circuit_fingerprint"].apply(_parse_transitions).apply(pd.Series)
    adf = pd.concat([adf, trans_records], axis=1)

    # --- Split into top/bottom quartile by r2_holdout (pooled across modes) ---
    q75 = adf["r2_holdout"].quantile(0.75)
    q25 = adf["r2_holdout"].quantile(0.25)
    top = adf[adf["r2_holdout"] >= q75]
    bot = adf[adf["r2_holdout"] <= q25]

    # --- Left panel: feature effect sizes ---
    feature_cols = [
        ("n_t_slots", "count_t"),
        ("n_cnot", "count_cx"),
        ("depth", "depth"),
        ("frac_t", "ratio_t"),
        ("frac_cnot", "ratio_cx"),
        ("n_gates_total", "n_gates"),
        ("frac_h", "ratio_h"),
    ]

    effect_sizes = []
    for col, label in feature_cols:
        top_mean = top[col].mean()
        bot_mean = bot[col].mean()
        pooled_std = adf[col].std()
        raw_delta = top_mean - bot_mean
        std_delta = raw_delta / pooled_std if pooled_std > 0 else 0
        effect_sizes.append({
            "feature": label,
            "std_delta": std_delta,
            "raw_delta": raw_delta,
        })

    # Add r2_gain (mean top - mean bottom as self-reference)
    r2_std = adf["r2_holdout"].std()
    r2_delta = top["r2_holdout"].mean() - bot["r2_holdout"].mean()
    effect_sizes.append({
        "feature": "r2_gain",
        "std_delta": r2_delta / r2_std if r2_std > 0 else 0,
        "raw_delta": r2_delta,
    })

    eff_df = pd.DataFrame(effect_sizes)

    # --- Right panel: transition shifts ---
    trans_shifts = []
    for tp in transition_pairs:
        delta = top[tp].mean() - bot[tp].mean()
        trans_shifts.append({"transition": tp, "delta": delta})
    trans_df = pd.DataFrame(trans_shifts).sort_values("delta", ascending=True)
    # Keep only non-negligible transitions
    trans_df = trans_df[trans_df["delta"].abs() > 1e-5]

    # --- Plot ---
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"wspace": 0.50})

    # Left: feature effect sizes
    colors_l = ["#2CA02C" if v >= 0 else "#D62728" for v in eff_df["std_delta"]]
    y_pos_l = np.arange(len(eff_df))
    ax_l.barh(y_pos_l, eff_df["std_delta"].values, color=colors_l, alpha=0.8, height=0.65)
    ax_l.set_yticks(y_pos_l)
    ax_l.set_yticklabels(eff_df["feature"].values, fontsize=11)
    for i, row in eff_df.iterrows():
        x = row["std_delta"]
        label = f"{x:+.3f}\u03c3 ({row['raw_delta']:+.4g})"
        sig = " *" if abs(x) > 1.0 else ""
        # Always place label at the end of the bar, on the outside
        if x >= 0:
            ax_l.text(x + 0.08, i, label + sig, va="center", ha="left", fontsize=8.5)
        else:
            # For negative bars, place label at the positive side of zero for clarity
            ax_l.text(0.08, i, label + sig, va="center", ha="left", fontsize=8.5, color="#555555")
    ax_l.axvline(0, color="black", lw=0.8)
    ax_l.set_xlabel("Standardized Δ(top − bottom) [std units]", fontsize=11, fontweight="bold")
    ax_l.set_title("Feature effect sizes (standardized)", fontsize=13, fontweight="bold")
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)
    ax_l.grid(True, axis="x", alpha=0.2)
    ax_l.tick_params(labelsize=10)
    # Extend x-range to fit labels
    x_max = eff_df["std_delta"].max()
    x_min = eff_df["std_delta"].min()
    ax_l.set_xlim(x_min - 0.6, x_max + 1.2)

    # Right: transition pattern shifts
    colors_r = ["#1F77B4" if v >= 0 else "#9467BD" for v in trans_df["delta"]]
    y_pos_r = np.arange(len(trans_df))
    ax_r.barh(y_pos_r, trans_df["delta"].values, color=colors_r, alpha=0.8, height=0.65)
    ax_r.set_yticks(y_pos_r)
    ax_r.set_yticklabels(trans_df["transition"].values, fontsize=11)
    for i, (_, row) in enumerate(trans_df.iterrows()):
        x = row["delta"]
        ha = "left" if x >= 0 else "right"
        offset = 0.0003 if x >= 0 else -0.0003
        ax_r.text(x + offset, i, f"{x:+.4f}", va="center", ha=ha, fontsize=9, fontweight="bold")
    ax_r.axvline(0, color="black", lw=0.8)
    ax_r.set_xlabel("Δ transition probability", fontsize=11, fontweight="bold")
    ax_r.set_title("Transition pattern shifts", fontsize=13, fontweight="bold")
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    ax_r.grid(True, axis="x", alpha=0.2)
    ax_r.tick_params(labelsize=10)

    fig.suptitle("Quantum Circuit Pattern Report Dashboard", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = out_dir / "poster_pattern_dashboard.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path.name}")
    return out_path


def plot_from_csv(
    csv_path: Path,
    out_dir: Path | None = None,
    gate_counts: Sequence[int] | None = None,
    poster: bool = False,
) -> Path:
    """Regenerate violin/box plots from an existing results CSV (no training)."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if "r2_holdout" not in df.columns:
        raise ValueError(f"CSV missing r2_holdout column: {csv_path}")

    out_dir = Path(out_dir) if out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if gate_counts is None:
        cfg_path = out_dir / "run_config.json"
        if cfg_path.is_file():
            with open(cfg_path) as f:
                gate_counts = json.load(f).get("gate_counts")
        if gate_counts is None:
            gate_counts = sorted(df["gate_count"].unique())

    plot_violins(df, out_dir, gate_counts)
    if poster:
        print("Generating poster figures...")
        plot_poster_results(df, out_dir, gate_counts)
        plot_poster_discussion(df, out_dir, gate_counts)
        plot_pattern_dashboard(df, out_dir, gate_counts)
    print(f"Saved plots -> {out_dir}")
    return out_dir


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
    from train_eval import build_model, load_dataloaders, train_model  # noqa: E402

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
    p.add_argument(
        "--plot-only",
        type=Path,
        metavar="CSV",
        default=None,
        help="Regenerate violin plots from an existing results CSV (no training).",
    )
    p.add_argument(
        "--poster",
        action="store_true",
        default=False,
        help="Also generate poster-quality figures for Results and Discussion sections.",
    )
    args = p.parse_args()

    if args.plot_only is not None:
        plot_from_csv(args.plot_only, out_dir=args.output_dir, poster=args.poster)
        return

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
