"""Standardized experiment directories, metadata, and log teeing."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TextIO


class TeeStream:
    """Write stdout/stderr to console and a log file."""

    def __init__(self, stream: TextIO, log_path: Path) -> None:
        self._stream = stream
        self._log = open(log_path, "a", encoding="utf-8")

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def close(self) -> None:
        self._log.close()

    def isatty(self) -> bool:
        return self._stream.isatty()


def mode_folder_tag(mode: str) -> str:
    """Map mode to folder suffix (fixed / para)."""
    if mode in ("fixed", "fixed_e2e"):
        return "fixed"
    if mode in ("param", "param_vqc"):
        return "para"
    raise ValueError(f"Unknown mode {mode!r}; use fixed or param")


def make_run_dir(
    base_dir: Path,
    mode: str,
    *,
    prefix: str = "replacing_multiple_layers",
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = mode_folder_tag(mode)
    out_dir = base_dir / f"{prefix}_{tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    return out_dir


def setup_run_logging(out_dir: Path) -> TeeStream:
    log_path = out_dir / "run.log"
    log_path.write_text("", encoding="utf-8")
    tee = TeeStream(sys.stdout, log_path)
    sys.stdout = tee
    return tee


def write_run_config(out_dir: Path, config: Dict[str, Any]) -> Path:
    path = out_dir / "run_config.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    return path


def write_run_info(out_dir: Path, lines: list[str]) -> Path:
    path = out_dir / "run_info.txt"
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    print(f"\nSaved run info -> {path}")
    return path


def format_run_info(
    config: Dict[str, Any],
    *,
    summary_lines: list[str] | None = None,
) -> list[str]:
    lines = [
        "=" * 70,
        "Multi-layer G3 replacement study",
        "=" * 70,
        f"Timestamp:        {config.get('timestamp', '')}",
        f"Circuit type:     {config.get('circuit_type', '')}",
        f"Quantum layers:   {config.get('n_quantum_layers', '')}",
        f"Layer sweep:      {config.get('layer_counts', '')}",
        f"Gates per layer:  {config.get('gates_per_layer', '')}",
        f"Circuit sharing:  {config.get('circuit_sharing', '')}",
        f"Shots:            {config.get('shots', '')}",
        f"Epochs:           {config.get('epochs', '')}",
        f"Batch size:       {config.get('batch_size', '')}",
        f"Learning rate:    {config.get('lr', '')}",
        f"N qubits:         {config.get('n_qubits', '')}",
        f"Data source:      {config.get('data_source', '')}",
        f"Train / val / holdout: {config.get('n_train', '')} / "
        f"{config.get('n_val', '')} / {config.get('n_holdout', '')}",
        "",
        "Per-layer circuits:",
    ]
    for entry in config.get("layer_circuits", []):
        lines.append(
            f"  layer {entry.get('layer_idx')}: seed={entry.get('seed')} "
            f"H={entry.get('n_h')} T={entry.get('n_t')} CNOT={entry.get('n_cnot')} "
            f"fp={entry.get('fingerprint', '')[:60]}..."
        )
    if summary_lines:
        lines.extend(["", "Run summary:"] + summary_lines)
    lines.append("=" * 70)
    return lines
