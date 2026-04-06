#!/usr/bin/env python
"""
Analyze pattern differences between better and worse circuits from testing_random_unitaries outputs.

Usage:
  python -m testing_unitaries.analyze_circuit_patterns --input-dir testing_unitaries/plots_YYYY-mm-dd_HH-MM-SS
  python testing_unitaries/analyze_circuit_patterns.py --input-dir testing_unitaries/plots_YYYY-mm-dd_HH-MM-SS
"""

import argparse
import json
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _find_latest_plots_dir(base_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(base_dir, 'plots_*')))
    if not candidates:
        raise FileNotFoundError(f"No plots_* directories found under {base_dir}")
    return candidates[-1]


def _safe_mean(x):
    return float(np.nanmean(x)) if len(x) else np.nan


def _safe_std(x):
    return float(np.nanstd(x)) if len(x) else np.nan


def _compute_transition_table(df_steps: pd.DataFrame, circuit_ids: set) -> pd.DataFrame:
    rows = []
    local = df_steps[df_steps['circuit_idx'].isin(circuit_ids)].copy()
    for circuit_idx, g in local.groupby('circuit_idx'):
        g = g.sort_values('step_idx')
        gates = g['gate_name'].astype(str).tolist()
        if len(gates) < 2:
            continue
        for a, b in zip(gates[:-1], gates[1:]):
            rows.append({'circuit_idx': circuit_idx, 'transition': f'{a}->{b}'})

    if not rows:
        return pd.DataFrame(columns=['transition', 'count', 'prob'])

    trans = pd.DataFrame(rows)
    agg = trans.groupby('transition').size().rename('count').reset_index()
    total = agg['count'].sum()
    agg['prob'] = agg['count'] / max(1, total)
    return agg.sort_values('prob', ascending=False).reset_index(drop=True)


def analyze(input_dir: str, top_fraction: float = 0.25):
    catalog_path = os.path.join(input_dir, 'all_circuit_catalog.csv')
    steps_path = os.path.join(input_dir, 'all_circuit_gate_steps.csv')

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Missing required file: {catalog_path}")
    if not os.path.exists(steps_path):
        raise FileNotFoundError(f"Missing required file: {steps_path}")

    df = pd.read_csv(catalog_path)
    steps = pd.read_csv(steps_path)

    evaluated = df[df['evaluated'] == True].copy()
    evaluated = evaluated.dropna(subset=['r2'])
    if len(evaluated) < 8:
        raise RuntimeError(f"Need at least 8 evaluated circuits for robust analysis; found {len(evaluated)}")

    q_hi = evaluated['r2'].quantile(1.0 - top_fraction)
    q_lo = evaluated['r2'].quantile(top_fraction)

    top = evaluated[evaluated['r2'] >= q_hi].copy()
    bottom = evaluated[evaluated['r2'] <= q_lo].copy()

    feature_cols = [
        'rfd_score', 'depth', 'n_gates',
        'count_h', 'count_t', 'count_cx',
        'ratio_h', 'ratio_t', 'ratio_cx',
        'r2_gain',
    ]
    feature_cols = [c for c in feature_cols if c in evaluated.columns]

    rows = []
    for c in feature_cols:
        top_vals = top[c].dropna().to_numpy()
        bot_vals = bottom[c].dropna().to_numpy()
        rows.append({
            'feature': c,
            'top_mean': _safe_mean(top_vals),
            'bottom_mean': _safe_mean(bot_vals),
            'delta_top_minus_bottom': _safe_mean(top_vals) - _safe_mean(bot_vals),
            'top_std': _safe_std(top_vals),
            'bottom_std': _safe_std(bot_vals),
            'n_top': len(top_vals),
            'n_bottom': len(bot_vals),
        })

    comp = pd.DataFrame(rows).sort_values('delta_top_minus_bottom', ascending=False)
    comp_path = os.path.join(input_dir, 'pattern_comparison_top_vs_bottom.csv')
    comp.to_csv(comp_path, index=False)

    top_trans = _compute_transition_table(steps, set(top['circuit_idx'].tolist()))
    bot_trans = _compute_transition_table(steps, set(bottom['circuit_idx'].tolist()))
    trans = top_trans.merge(bot_trans, on='transition', how='outer', suffixes=('_top', '_bottom')).fillna(0.0)
    trans['delta_prob_top_minus_bottom'] = trans['prob_top'] - trans['prob_bottom']
    trans = trans.sort_values('delta_prob_top_minus_bottom', ascending=False)
    trans_path = os.path.join(input_dir, 'pattern_transition_differences.csv')
    trans.to_csv(trans_path, index=False)

    # Plot 1: gate-ratio comparison
    gate_ratio_cols = [c for c in ['ratio_h', 'ratio_t', 'ratio_cx'] if c in evaluated.columns]
    if gate_ratio_cols:
        x = np.arange(len(gate_ratio_cols))
        w = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - w / 2, [top[c].mean() for c in gate_ratio_cols], w, label='Top bucket', color='#2ecc71')
        ax.bar(x + w / 2, [bottom[c].mean() for c in gate_ratio_cols], w, label='Bottom bucket', color='#e74c3c')
        ax.set_xticks(x)
        ax.set_xticklabels(gate_ratio_cols)
        ax.set_ylabel('Mean ratio')
        ax.set_title('Gate composition by performance bucket')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(input_dir, 'pattern_gate_ratio_comparison.png'), dpi=180)
        plt.close(fig)

    # Plot 2: RFD vs R²
    if 'rfd_score' in evaluated.columns:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(evaluated['rfd_score'], evaluated['r2'], alpha=0.7, color='#3498db', edgecolor='k', linewidth=0.4)
        ax2.set_xlabel('RFD score')
        ax2.set_ylabel('R²')
        ax2.set_title('RFD vs performance (evaluated circuits)')
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(input_dir, 'pattern_rfd_vs_r2.png'), dpi=180)
        plt.close(fig2)

    report = {
        'input_dir': input_dir,
        'n_total_circuits': int(len(df)),
        'n_evaluated': int(len(evaluated)),
        'n_top_bucket': int(len(top)),
        'n_bottom_bucket': int(len(bottom)),
        'r2_top_threshold': float(q_hi),
        'r2_bottom_threshold': float(q_lo),
        'top_feature_gains': comp.head(8).to_dict(orient='records'),
        'top_transition_gains': trans.head(10).to_dict(orient='records'),
    }
    report_path = os.path.join(input_dir, 'pattern_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"Input: {input_dir}")
    print(f"Saved: {comp_path}")
    print(f"Saved: {trans_path}")
    print(f"Saved: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze performance patterns across generated quantum circuits.')
    parser.add_argument('--input-dir', type=str, default='', help='Path to plots_* output directory')
    parser.add_argument('--top-fraction', type=float, default=0.25, help='Fraction for top/bottom buckets (default: 0.25)')
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir.strip() if args.input_dir else _find_latest_plots_dir(this_dir)
    analyze(input_dir=input_dir, top_fraction=args.top_fraction)
