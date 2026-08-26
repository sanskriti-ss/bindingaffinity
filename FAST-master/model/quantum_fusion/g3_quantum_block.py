"""Reusable fixed and parametric G3 PennyLane blocks."""

from __future__ import annotations

from typing import List, Sequence, Union

import pennylane as qml
import torch
import torch.nn as nn
from qiskit.circuit import QuantumCircuit

from circuit_common import GateStructureEntry, extract_gate_structure


def apply_g3_structure(
    structure: Sequence[GateStructureEntry],
    params: torch.Tensor | None,
) -> None:
    """Apply parsed G3 structure inside an active QNode."""
    for gate_type, qubit_info, param_idx in structure:
        if gate_type == "h":
            qml.Hadamard(wires=qubit_info)
        elif gate_type == "t":
            qml.T(wires=qubit_info)
        elif gate_type == "ry":
            assert params is not None and param_idx is not None
            qml.RY(params[param_idx], wires=qubit_info)
        elif gate_type == "rz":
            assert params is not None and param_idx is not None
            qml.RZ(params[param_idx], wires=qubit_info)
        elif gate_type == "cnot":
            qml.CNOT(wires=qubit_info)


def _encode_ry(inputs: torch.Tensor, n_qubits: int) -> None:
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)


class G3FixedBlock(nn.Module):
    """Fixed G3 reservoir block with X/Y/Z measurements."""

    feature_dim: int

    def __init__(
        self,
        qc: QuantumCircuit,
        n_qubits: int,
        backend: str = "lightning.qubit",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.feature_dim = 3 * n_qubits
        self.gate_structure = extract_gate_structure(qc, parametric=False)
        self.dev = qml.device(backend, wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor) -> List:
            _encode_ry(inputs, n_qubits)
            apply_g3_structure(self.gate_structure, None)
            return (
                [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
                + [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
                + [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            )

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs: [n_qubits] -> [3 * n_qubits]"""
        return torch.stack(self.circuit(inputs)).float()


class G3ParamBlock(nn.Module):
    """Parametric G3 block (H->RY, T->RZ) with Z measurements."""

    feature_dim: int

    def __init__(
        self,
        qc: QuantumCircuit,
        n_qubits: int,
        backend: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.feature_dim = n_qubits
        self.gate_structure = extract_gate_structure(qc, parametric=True)
        n_params = max(
            (idx for _, _, idx in self.gate_structure if idx is not None),
            default=-1,
        ) + 1
        self.quantum_params = nn.Parameter(torch.randn(max(n_params, 0)) * 0.1)
        self.dev = qml.device(backend, wires=n_qubits)

        structure = self.gate_structure

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, params: torch.Tensor) -> List:
            _encode_ry(inputs, n_qubits)
            apply_g3_structure(structure, params)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs: [n_qubits] -> [n_qubits]"""
        return torch.stack(self.circuit(inputs, self.quantum_params)).float()
