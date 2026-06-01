"""Shared G3 circuit sampling helpers for param-vs-fixed studies."""

from __future__ import annotations

import random
from typing import List, Tuple

from qiskit.circuit import QuantumCircuit


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
