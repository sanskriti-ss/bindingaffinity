"""Stacked G3 hybrid models with configurable layer count."""

from __future__ import annotations

import math
from typing import List, Literal, Sequence

import torch
import torch.nn as nn
from qiskit.circuit import QuantumCircuit

from g3_quantum_block import G3FixedBlock, G3ParamBlock

Mode = Literal["fixed", "param"]


class ModelMultiLayerG3Hybrid(nn.Module):
    """
    Classical encoder -> N G3 quantum blocks -> MLP head (+ initial-encoding skip).

    Between layers, quantum measurements are remapped to ``n_qubits`` angles via
  trainable Linear layers and ``tanh * pi`` for the next RY encoding.
    """

    def __init__(
        self,
        in_features: int,
        circuits: Sequence[QuantumCircuit],
        *,
        mode: Mode,
        n_qubits: int = 6,
        out_features: int = 1,
        fixed_backend: str = "lightning.qubit",
        param_backend: str = "default.qubit",
    ) -> None:
        super().__init__()
        if len(circuits) < 1:
            raise ValueError("At least one G3 circuit is required")

        self.mode = mode
        self.n_qubits = n_qubits
        self.n_quantum_layers = len(circuits)

        self.fc1 = nn.Linear(in_features, 4 * n_qubits)
        self.bn1 = nn.BatchNorm1d(4 * n_qubits)
        self.fc2 = nn.Linear(4 * n_qubits, n_qubits)

        blocks: List[nn.Module] = []
        for qc in circuits:
            if mode == "fixed":
                blocks.append(G3FixedBlock(qc, n_qubits, backend=fixed_backend))
            else:
                blocks.append(G3ParamBlock(qc, n_qubits, backend=param_backend))
        self.blocks = nn.ModuleList(blocks)

        self.inter_layer = nn.ModuleList()
        for block in blocks[:-1]:
            self.inter_layer.append(nn.Linear(block.feature_dim, n_qubits))

        total_q_features = sum(b.feature_dim for b in blocks)
        combined_dim = total_q_features + n_qubits
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_features),
        )

    def _run_block_batch(self, block: nn.Module, angles: torch.Tensor) -> torch.Tensor:
        batch_size = angles.shape[0]
        outputs = []
        for i in range(batch_size):
            outputs.append(block(angles[i]))
        return torch.stack(outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.fc1(x)))
        angles = torch.tanh(self.fc2(x)) * math.pi
        initial_encoding = angles

        layer_features: List[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            q_feat = self._run_block_batch(block, angles)
            layer_features.append(q_feat)
            if layer_idx < len(self.inter_layer):
                angles = torch.tanh(self.inter_layer[layer_idx](q_feat)) * math.pi

        combined = torch.cat(layer_features + [initial_encoding], dim=1)
        return self.head(combined)
