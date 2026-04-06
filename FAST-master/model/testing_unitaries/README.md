# Testing Unitaries

This folder contains the random-unitary reservoir testing workflows moved out of `quantum_fusion/`.

## Scripts

- `testing_random_unitaries.py`
  - Generates random G3 circuits (`H`, `T`, `CNOT`)
  - Ranks circuits by Reservoir Feature Diversity (RFD)
  - Evaluates top circuits with ridge readout
  - Supports unseen holdout reporting via `--holdout`
  - Saves plots/CSVs to a timestamped `plots_YYYY-mm-dd_HH-MM-SS/` folder
  - Exports all generated circuits in analysis-ready files:
    - `all_circuit_catalog.csv` (one row per circuit)
    - `all_circuit_gate_steps.csv` (one row per gate instruction)
    - `all_circuit_diagrams/` (PNG diagrams using modular Qiskit style)

- `evaluate_top25.py`
  - Generates 100 circuits and selects top 25 by RFD
  - Trains reservoir models and compares circuit performance quartiles
  - Saves comparison plots and CSV summaries

- `analyze_circuit_patterns.py`
  - Reads `all_circuit_catalog.csv` + `all_circuit_gate_steps.csv`
  - Compares top vs bottom performance buckets
  - Produces pattern CSV/JSON outputs and summary plots

## Run commands

From `FAST-master/model/`:

```bash
python -m testing_unitaries.testing_random_unitaries
python -m testing_unitaries.testing_random_unitaries --holdout
python -m testing_unitaries.evaluate_top25
python -m testing_unitaries.analyze_circuit_patterns --input-dir testing_unitaries/plots_YYYY-mm-dd_HH-MM-SS
```

From `FAST-master/model/testing_unitaries/`:

```bash
python testing_random_unitaries.py
python testing_random_unitaries.py --holdout
python evaluate_top25.py
python analyze_circuit_patterns.py --input-dir plots_YYYY-mm-dd_HH-MM-SS
```

## Holdout mode

Use `--holdout` to report final metrics and plots on unseen data:

```bash
python -m testing_unitaries.testing_random_unitaries --holdout --holdout-fraction 0.5 --seed 42
```

- Selection still happens on the selection-validation split.
- Reported `R²`, `RMSE`, and plots are computed on holdout.

## Notes

- The scripts load feature NPZ files from `model/quantum_fusion/`.
- `evaluate_top25.py` also imports `quantum_fusion.main_train`.
- Output location for `evaluate_top25.py` is this folder (`testing_unitaries/`).
