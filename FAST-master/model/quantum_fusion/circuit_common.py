"""Shared G3 circuit sampling helpers for param-vs-fixed studies."""

from __future__ import annotations

import random
from typing import List, Literal, Sequence, Tuple, Union

from qiskit.circuit import QuantumCircuit

GateStructureEntry = Tuple[str, Union[int, List[int]], Union[int, None]]
CircuitSharing = Literal["independent", "shared"]


def sample_g3_circuit(
    n_qubits: int,
    num_gates: int,
    seed: int,
) -> QuantumCircuit:
    """Return one random G3 circuit (H, T, CNOT) with ``num_gates`` instructions."""
    from testing_random_unitaries import generate_g3_random_circuits

    random.seed(seed)
    circuits = generate_g3_random_circuits(
        n_qubits, num_gates=num_gates, num_circuits=1,
    )
    return circuits[0]


def circuit_fingerprint(qc: QuantumCircuit) -> str:
    """Stable string id from ordered (gate, qubits) stream."""
    tokens = []
    for inst in qc.data:
        gate = inst.operation.name
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        tokens.append(f"{gate}:{','.join(map(str, qubits))}")
    return "|".join(tokens)


def count_vqc_params(qc: QuantumCircuit) -> int:
    """Trainable angles in ``ModelHybridFC_VQC`` (one per H/T slot)."""
    n = 0
    for inst in qc.data:
        name = inst.operation.name
        if name in ("h", "t"):
            n += 1
    return n


def gate_structure_summary(qc: QuantumCircuit) -> Tuple[int, int, int]:
    """Return (n_h_slots, n_t_slots, n_cnot) for logging."""
    nh = nt = nc = 0
    for inst in qc.data:
        name = inst.operation.name
        if name == "h":
            nh += 1
        elif name == "t":
            nt += 1
        elif name == "cx":
            nc += 1
    return nh, nt, nc


def extract_gate_structure(qc: QuantumCircuit, *, parametric: bool) -> List[GateStructureEntry]:
    """
    Parse a G3 Qiskit circuit into PennyLane gate descriptors.

    When ``parametric`` is True, H/T slots become trainable RY/RZ with param indices.
    """
    structure: List[GateStructureEntry] = []
    n_params = 0
    for inst in qc.data:
        gate = inst.operation
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        if gate.name == "h":
            if parametric:
                structure.append(("ry", qubits[0], n_params))
                n_params += 1
            else:
                structure.append(("h", qubits[0], None))
        elif gate.name == "t":
            if parametric:
                structure.append(("rz", qubits[0], n_params))
                n_params += 1
            else:
                structure.append(("t", qubits[0], None))
        elif gate.name == "cx":
            structure.append(("cnot", qubits, None))
    return structure


def count_parametric_g3_params(structure: Sequence[GateStructureEntry]) -> int:
    """Number of trainable angles implied by a parametric gate structure."""
    return sum(1 for g, _, idx in structure if g in ("ry", "rz") and idx is not None)


def sample_g3_circuits(
    n_qubits: int,
    gates_per_layer: int,
    n_layers: int,
    seed: int,
    sharing: CircuitSharing = "independent",
) -> List[QuantumCircuit]:
    """
    Sample one G3 circuit per quantum layer.

    ``independent``: distinct random circuit per layer (seed + layer_idx * 17).
    ``shared``: one draw reused at every layer.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if sharing == "shared":
        qc = sample_g3_circuit(n_qubits, gates_per_layer, seed)
        return [qc.copy() for _ in range(n_layers)]
    return [
        sample_g3_circuit(n_qubits, gates_per_layer, seed + layer_idx * 17)
        for layer_idx in range(n_layers)
    ]
