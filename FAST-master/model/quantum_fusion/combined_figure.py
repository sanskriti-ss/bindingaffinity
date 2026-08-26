#!/usr/bin/env python
"""
combined_figure.py — Publication-quality combined figure from three studies.

Generates a 2×2 panel figure:
  (a) Holdout R² vs layer depth        [multi_layer_results.csv]
  (b) Holdout R² vs gate count          [param_vs_fixed_results.csv]
  (c) Feature effect sizes              [param_vs_fixed_results.csv]
  (d) Transition pattern shifts         [param_vs_fixed_results.csv]

Data is loaded from CSVs extracted from git stash@{0} (parameterized branch).

Usage:
    python combined_figure.py
    python combined_figure.py --param-csv path/to/param.csv --layer-csv path/to/layer.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colour palette
C_FIXED = "#4878CF"
C_PARAM = "#E8915A"
C_POS   = "#6CA86C"
C_NEG   = "#CD5C5C"
C_BLUE  = "#6C9BD2"
C_PURPLE = "#B07AB0"


def _try_extract_from_stash(stash_path: str, fallback: str | None) -> str | None:
    """Try to extract a file from git stash; return path or None."""
    if fallback and os.path.isfile(fallback):
        return fallback
    repo_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__),
    ).stdout.strip()
    if not repo_root:
        return None
    try:
        content = subprocess.run(
            ["git", "show", f"stash@{{0}}:{stash_path}"],
            capture_output=True, text=True, cwd=repo_root, check=True,
        ).stdout
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        tmp.write(content)
        tmp.close()
        return tmp.name
    except subprocess.CalledProcessError:
        return None


def compute_effects_and_transitions(df: pd.DataFrame):
    """Compute feature effect sizes and transition shifts from param_vs_fixed CSV."""
    adf = df.copy()
    adf["frac_h"] = adf["n_h_slots"] / adf["n_gates_total"]
    adf["frac_t"] = adf["n_t_slots"] / adf["n_gates_total"]
    adf["frac_cnot"] = adf["n_cnot"] / adf["n_gates_total"]
    adf["depth"] = adf["n_gates_total"]

    # Parse transitions from circuit_fingerprint
    gate_types = ["h", "t", "cx"]
    transition_pairs = [f"{a}->{b}" for a in gate_types for b in gate_types]

    def _parse_transitions(fp: str) -> dict:
        gates = []
        for token in str(fp).split("|"):
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

    # Top/bottom quartile by r2_holdout
    q75 = adf["r2_holdout"].quantile(0.75)
    q25 = adf["r2_holdout"].quantile(0.25)
    top = adf[adf["r2_holdout"] >= q75]
    bot = adf[adf["r2_holdout"] <= q25]

    # Feature effect sizes
    feature_cols = [
        ("n_t_slots", "count_t"),
        ("n_cnot", "count_cx"),
        ("depth", "depth"),
        ("frac_t", "ratio_t"),
        ("frac_cnot", "ratio_cx"),
        ("n_gates_total", "n_gates"),
        ("frac_h", "ratio_h"),
    ]

    effects = []
    for col, label in feature_cols:
        top_mean = top[col].mean()
        bot_mean = bot[col].mean()
        pooled_std = adf[col].std()
        raw_delta = top_mean - bot_mean
        std_delta = raw_delta / pooled_std if pooled_std > 0 else 0
        effects.append({"feature": label, "std_delta": std_delta, "raw_delta": raw_delta})

    r2_std = adf["r2_holdout"].std()
    r2_delta = top["r2_holdout"].mean() - bot["r2_holdout"].mean()
    effects.append({
        "feature": "r2_gain",
        "std_delta": r2_delta / r2_std if r2_std > 0 else 0,
        "raw_delta": r2_delta,
    })

    eff_df = pd.DataFrame(effects)

    # Transition shifts
    trans_shifts = []
    for tp in transition_pairs:
        delta = top[tp].mean() - bot[tp].mean()
        trans_shifts.append({"transition": tp, "delta": delta})
    trans_df = pd.DataFrame(trans_shifts).sort_values("delta", ascending=True)
    trans_df = trans_df[trans_df["delta"].abs() > 1e-5]

    return eff_df, trans_df


def main():
    parser = argparse.ArgumentParser(description="Combined publication figure")
    parser.add_argument("--param-csv", default=None, help="param_vs_fixed_results.csv path")
    parser.add_argument("--layer-csv", default=None, help="multi_layer_results.csv path")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    param_csv = _try_extract_from_stash(
        "FAST-master/model/quantum_fusion/results/param_vs_fixed_20260531_182941/param_vs_fixed_results.csv",
        args.param_csv,
    )
    layer_csv = _try_extract_from_stash(
        "FAST-master/model/quantum_fusion/results/replacing_multiple_layers_fixed_20260617_145059/multi_layer_results.csv",
        args.layer_csv,
    )

    if param_csv is None:
        sys.exit("Cannot find param_vs_fixed_results.csv (provide --param-csv or ensure stash@{0})")
    if layer_csv is None:
        sys.exit("Cannot find multi_layer_results.csv (provide --layer-csv or ensure stash@{0})")

    pvf = pd.read_csv(param_csv)
    ml = pd.read_csv(layer_csv)
    print(f"Loaded param_vs_fixed: {len(pvf)} rows, gate_counts={sorted(pvf['gate_count'].unique())}")
    print(f"Loaded multi_layer:    {len(ml)} rows, layers={sorted(ml['n_quantum_layers'].unique())}")

    eff_df, trans_df = compute_effects_and_transitions(pvf)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.2, 5.8))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.40,
                           left=0.09, right=0.97, top=0.94, bottom=0.08)

    # ── Panel (a): Layer depth ────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    layer_vals = sorted(ml["n_quantum_layers"].unique().astype(int))
    positions_a = np.arange(len(layer_vals))
    all_data_a = [ml[ml["n_quantum_layers"] == d]["r2_holdout"].values for d in layer_vals]
    means_a = [np.mean(v) for v in all_data_a]
    sems_a = [np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for v in all_data_a]
    n_shots = max(len(v) for v in all_data_a)

    vp = ax_a.violinplot(all_data_a, positions=positions_a, showmeans=False,
                          showextrema=False, widths=0.6)
    for body in vp["bodies"]:
        body.set_facecolor(C_FIXED)
        body.set_alpha(0.20)
        body.set_edgecolor(C_FIXED)
        body.set_linewidth(0.5)

    bp = ax_a.boxplot(all_data_a, positions=positions_a, widths=0.25,
                       patch_artist=True, showfliers=True,
                       flierprops=dict(marker="o", markersize=2, alpha=0.5, color=C_PARAM),
                       medianprops=dict(color="black", linewidth=0.8),
                       whiskerprops=dict(linewidth=0.6),
                       capprops=dict(linewidth=0.6))
    for patch in bp["boxes"]:
        patch.set_facecolor(C_PARAM)
        patch.set_alpha(0.40)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)

    ax_a.errorbar(positions_a, means_a, yerr=sems_a, fmt="o-", color=C_FIXED,
                   markersize=5, linewidth=1.2, capsize=3, zorder=5,
                   label="Mean $\\pm$ SEM")

    ax_a.set_xticks(positions_a)
    ax_a.set_xticklabels([str(d) for d in layer_vals])
    ax_a.set_xlabel("Number of quantum layers replaced")
    ax_a.set_ylabel("Holdout $R^2$")
    ax_a.set_ylim(0.40, 0.50)
    ax_a.set_title(f"(a)  Layer depth (fixed, 3 gates/layer, {n_shots} shots)",
                   fontweight="bold", loc="left")
    ax_a.legend(loc="lower left", framealpha=0.8, edgecolor="none")
    ax_a.grid(axis="y", alpha=0.2, linewidth=0.4)

    # ── Panel (b): Fixed vs param ────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    gate_counts = sorted(pvf["gate_count"].unique())
    x_pos = np.arange(len(gate_counts))
    w = 0.32
    off = 0.17

    all_fixed = [pvf[(pvf["gate_count"] == g) & (pvf["mode"] == "fixed_e2e")]["r2_holdout"].values
                 for g in gate_counts]
    all_param = [pvf[(pvf["gate_count"] == g) & (pvf["mode"] == "param_vqc")]["r2_holdout"].values
                 for g in gate_counts]

    n_reps = max(max(len(v) for v in all_fixed), max(len(v) for v in all_param))

    # Violins
    for data_list, offset_sign, colour in [(all_fixed, -1, C_FIXED), (all_param, 1, C_PARAM)]:
        pos = x_pos + offset_sign * off
        vp_i = ax_b.violinplot(data_list, positions=pos, showmeans=False,
                                showextrema=False, widths=w)
        for body in vp_i["bodies"]:
            body.set_facecolor(colour)
            body.set_alpha(0.20)
            body.set_edgecolor(colour)
            body.set_linewidth(0.5)

        bp_i = ax_b.boxplot(data_list, positions=pos, widths=w * 0.55,
                             patch_artist=True, showfliers=False,
                             medianprops=dict(color="black", linewidth=0.8),
                             whiskerprops=dict(linewidth=0.6),
                             capprops=dict(linewidth=0.6))
        for patch in bp_i["boxes"]:
            patch.set_facecolor(colour)
            patch.set_alpha(0.40)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.5)

    # Mean ± SEM
    for data_list, offset_sign, colour, label in [
        (all_fixed, -1, C_FIXED, "Fixed mean $\\pm$ SEM"),
        (all_param, 1, C_PARAM, "Param mean $\\pm$ SEM"),
    ]:
        means = [np.mean(v) for v in data_list]
        sems = [np.std(v, ddof=1)/np.sqrt(len(v)) if len(v) > 1 else 0 for v in data_list]
        pos = x_pos + offset_sign * off
        ax_b.errorbar(pos, means, yerr=sems, fmt="o", color=colour,
                       markersize=4, capsize=3, linewidth=1.0, label=label, zorder=5)
        ax_b.plot(pos, means, "-", color=colour, linewidth=0.8, alpha=0.7)

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels([str(g) for g in gate_counts])
    ax_b.set_xlabel("G3 gate count")
    ax_b.set_ylabel("Holdout $R^2$")
    ax_b.set_ylim(0.40, 0.50)
    ax_b.set_title(f"(b)  Fixed vs parameterised ({n_reps} replicas/cell)",
                   fontweight="bold", loc="left")
    ax_b.legend(loc="upper right", framealpha=0.8, edgecolor="none")
    ax_b.grid(axis="y", alpha=0.2, linewidth=0.4)

    # ── Panel (c): Feature effect sizes ──────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])

    eff_sorted = eff_df.sort_values("std_delta", key=abs).reset_index(drop=True)
    y_pos_c = np.arange(len(eff_sorted))
    bar_colors_c = [C_POS if v >= 0 else C_NEG for v in eff_sorted["std_delta"]]

    ax_c.barh(y_pos_c, eff_sorted["std_delta"].values, color=bar_colors_c,
              alpha=0.75, edgecolor="black", linewidth=0.3, height=0.6)

    for i, row in eff_sorted.iterrows():
        v = row["std_delta"]
        raw = row["raw_delta"]
        label = f"{v:+.3f}$\\sigma$ ({raw:+.4g})"
        sig = " *" if abs(v) > 1.0 else ""
        if v >= 0:
            ax_c.text(v + 0.08, i, label + sig, va="center", ha="left",
                      fontsize=5.5, color="#333333")
        else:
            ax_c.text(0.08, i, label + sig, va="center", ha="left",
                      fontsize=5.5, color="#555555")

    ax_c.set_yticks(y_pos_c)
    ax_c.set_yticklabels(eff_sorted["feature"].values, fontsize=7)
    ax_c.set_xlabel("Standardised $\\Delta$ (top $-$ bottom) [std units]")
    ax_c.set_title("(c)  Feature effect sizes", fontweight="bold", loc="left")
    ax_c.axvline(0, color="black", linewidth=0.5)
    ax_c.grid(axis="x", alpha=0.2, linewidth=0.4)

    # ── Panel (d): Transition pattern shifts ─────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])

    y_pos_d = np.arange(len(trans_df))
    bar_colors_d = [C_BLUE if v >= 0 else C_PURPLE for v in trans_df["delta"]]

    ax_d.barh(y_pos_d, trans_df["delta"].values, color=bar_colors_d,
              alpha=0.75, edgecolor="black", linewidth=0.3, height=0.6)

    for i, (_, row) in enumerate(trans_df.iterrows()):
        v = row["delta"]
        pad = 0.0005 if v >= 0 else -0.0005
        ha = "left" if v >= 0 else "right"
        ax_d.text(v + pad, i, f"{v:+.4f}", va="center", ha=ha,
                  fontsize=6, color="#333333")

    ax_d.set_yticks(y_pos_d)
    ax_d.set_yticklabels(trans_df["transition"].values, fontsize=7)
    ax_d.set_xlabel("$\\Delta$ transition probability")
    ax_d.set_title("(d)  Transition pattern shifts", fontweight="bold", loc="left")
    ax_d.axvline(0, color="black", linewidth=0.5)
    ax_d.grid(axis="x", alpha=0.2, linewidth=0.4)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"
    )
    os.makedirs(out_dir, exist_ok=True)

    for fmt in ("png", "pdf"):
        path = os.path.join(out_dir, f"quantum_fusion_combined.{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
