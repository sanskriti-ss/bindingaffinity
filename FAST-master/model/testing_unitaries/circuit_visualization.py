import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def publication_qiskit_style(background_color: str = "#f3f3f3") -> Dict:
    """Return a clean, high-contrast Qiskit mpl style similar to publication screenshots."""
    return {
        "name": "iqp",
        "backgroundcolor": background_color,
        "linecolor": "#202020",
        "fontsize": 15,
        "subfontsize": 11,
        "displaycolor": {
            "h": "#ff5b66",
            "t": "#41a8f0",
            "x": "#0b3aa7",
            "cx": "#0b3aa7",
        },
    }


def render_circuit_diagram(
    qc,
    output_path: str,
    title: Optional[str] = None,
    fold: int = -1,
    style: Optional[Dict] = None,
    dpi: int = 220,
) -> str:
    """Render a Qiskit circuit to PNG with a consistent style."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    style = style or publication_qiskit_style()

    try:
        fig = qc.draw(output="mpl", fold=fold, style=style)
    except Exception:
        # Graceful fallback to default style if local qiskit style options differ.
        fig = qc.draw(output="mpl", fold=fold)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def circuit_summary_stats(qc) -> Dict:
    """Return compact per-circuit summary features."""
    counts = qc.count_ops()
    n_gates = int(qc.size())
    count_h = int(counts.get("h", 0))
    count_t = int(counts.get("t", 0))
    count_cx = int(counts.get("cx", 0))
    depth = int(qc.depth()) if qc.depth() is not None else -1

    denom = max(1, n_gates)
    return {
        "n_qubits": int(qc.num_qubits),
        "n_gates": n_gates,
        "depth": depth,
        "count_h": count_h,
        "count_t": count_t,
        "count_cx": count_cx,
        "ratio_h": count_h / denom,
        "ratio_t": count_t / denom,
        "ratio_cx": count_cx / denom,
    }


def circuit_gate_rows(qc, circuit_idx: int, circuit_id: str) -> List[Dict]:
    """Return long-form gate rows for downstream pattern analysis."""
    rows: List[Dict] = []
    for step_idx, inst in enumerate(qc.data):
        gate_name = inst.operation.name
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        control = qubits[0] if gate_name == "cx" and len(qubits) > 0 else None
        target = qubits[1] if gate_name == "cx" and len(qubits) > 1 else (qubits[0] if qubits else None)

        rows.append(
            {
                "circuit_idx": circuit_idx,
                "circuit_id": circuit_id,
                "step_idx": step_idx,
                "gate_name": gate_name,
                "num_qubits": len(qubits),
                "control_qubit": control,
                "target_qubit": target,
                "qubits": "|".join(str(q) for q in qubits),
            }
        )
    return rows
