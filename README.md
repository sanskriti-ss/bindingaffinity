# Protein-Ligand Binding Affinity Prediction
## Quantum Reservoir Computing as a Nonlinear Feature Map — A Proof of Concept

A research framework for protein-ligand binding affinity prediction (PDBbind refined set, 836 complexes) that explores **quantum reservoir computing as a fixed nonlinear feature map** fused with classical deep learning. Built as an honest proof-of-concept and invitation for future quantum ML research in drug discovery.

> **Honest framing:** This work does **not** claim quantum advantage over classical methods.
> The classical pipeline (RBF kernel on 153-dim features) achieves R² = 0.85 on the test set.
> The quantum reservoir kernel achieves R² ≈ 0.20. The gap is explained by a fundamental
> **information bottleneck**: 153 classical features must be compressed to 6 qubit angles before
> the quantum circuit sees them, discarding ~96% of the input information.
> We document this bottleneck carefully so future researchers know exactly what needs to change.

---

## What This Project Does

**Pipeline:**
1. Load 836 protein-ligand complexes from PDBbind 2020 refined set
2. Extract 3DCNN (fc1, 10-dim) and SGCNN (hidden, 54-dim) deep feature embeddings
3. Fuse with RDKit pocket (25-dim) + ligand PCA (64-dim) features → **153-dim** combined vector
4. Compress 153→6 dimensions via a fixed random projection (tanh-scaled to [−π, π])
5. Pass through a fixed **G3 quantum reservoir** (H, T, CNOT gates; 6 qubits)
6. Read out 18 Pauli X/Y/Z expectation values as quantum features
7. Fit a ridge regression or MLP head on quantum features

**Circuit selection:** G3 circuits are pre-screened by Reservoir Feature Diversity (RFD), a spectral expressibility score based on the participation ratio of SVD of the quantum feature matrix.

---

## Key Results

### Ablation Study (MLP readout, 50 epochs, stratified 70/15/15 split)

## The ablation study uses a stratified 70/15/15 training/validation/test split with random seed 42. The quantum-kernel study uses a stratified 80/20 training/test split with random seed 42. Experiment-specific preprocessing and model-selection procedures are implemented in the corresponding scripts.


| Condition | R² | Pearson r | RMSE (pKi) | Notes |
|-----------|------|-----------|------------|-------|
| **A — Classical MLP (no quantum)** | 0.7775 | 0.8893 | 0.881 | Baseline |
| **B — Quantum reservoir + skip** | 0.8041 | 0.9011 | 0.827 | Skip gives MLP a classical shortcut |
| **C — Quantum only, no skip** | 0.4518 | 0.6803 | 1.383 | Quantum features alone |
| **D — Random circuit, no skip** | 0.5237 | 0.7379 | 1.289 | Unselected circuit |

**Diagnosis:**
- `B − A` = +0.027 (apparent quantum gain with skip connection)
- `C − A` = **−0.326** ← quantum features alone are far below classical
- `C − D` = −0.072 ← RFD-selected circuit is not more informative than random

The skip connection in condition B lets the MLP route around the quantum layer entirely.
Condition C is the honest measure: the quantum reservoir, used alone, performs well below
the classical MLP — confirming the model was not using quantum features when skip was enabled.

### Quantum Kernel Study (KernelRidge readout, 20-circuit sweep)

| Method | R² | Notes |
|--------|------|-------|
| **Classical RBF — full 153-dim** | **0.8532** | Classical ceiling |
| Classical linear — 6-dim encoded | 0.4536 | Same compressed input as quantum |
| Classical RBF — 6-dim encoded | 0.3855 | Same compressed input as quantum |
| **Quantum kernel (best circuit #9)** | **0.1967** | Worse than classical on same dims |
| Quantum kernel (median circuit) | ~0.05 | Most circuits near zero |

**Centered Kernel Alignment (CKA) — quantum kernel vs label kernel:**
All 20 circuits produced CKA ∈ [0.008, 0.036] — near zero. The quantum feature space
has almost no geometric alignment with the binding affinity regression target.

**Circuit selection predictors:**
- RFD ↔ R²: Pearson r = +0.23, p = 0.34 (not significant)
- CKA ↔ R²: Pearson r = +0.38, p = 0.10 (not significant)

Neither RFD nor CKA reliably predicts which circuit will perform best on the task.

### Generated Plots

All plots live in `FAST-master/model/quantum_fusion/`:

| File | Content |
|------|---------|
| `ablation_r2_bar.png` | Bar chart: A/B/C/D ablation conditions |
| `quantum_kernel_study_output/kernel_matrix_heatmap.png` | Quantum K(i,j) matrix sorted by pKi |
| `quantum_kernel_study_output/kpca_affinity.png` | KPCA embedding of quantum features, coloured by pKi |
| `quantum_kernel_study_output/circuit_sweep_scatter.png` | R² vs RFD and R² vs CKA per circuit |
| `quantum_kernel_study_output/classical_vs_quantum_bar.png` | Classical kernels vs quantum kernel R² |
| `quantum_kernel_study_output/best_circuit_scatter.png` | Predicted vs experimental pKi, best quantum circuit |
| `quantum_kernel_study_output/cka_r2_bars.png` | Per-circuit CKA and R² sorted by performance |
| `quantum_kernel_study_output/kernel_study_results.csv` | Full 20-circuit sweep data |
| `ablation_results.csv` | Full ablation table |

---

## Why Quantum Doesn't Win Here (and What Would Help)

### The Information Bottleneck

The fundamental limitation is dimensional compression before the quantum layer:

```
153-dim fused features
        ↓   fixed random projection + tanh
    6-dim angles   ← only 6 numbers reach the quantum circuit
        ↓   G3 circuit (fixed, non-trainable)
   18 Pauli measurements
        ↓   KernelRidge / MLP
      pKi prediction
```

With 153 input features compressed to 6 angles, **~96% of the input information is discarded
before the quantum circuit ever sees it**. A classical RBF kernel on the same 6-dim space
achieves R² = 0.39 — already above the quantum kernel (0.20). The quantum transformation
adds noise rather than signal in this low-dimensional regime.

The full 153-dim classical RBF achieves R² = 0.85 without any compression, setting the
ceiling that any quantum method must surpass.

### What Future Researchers Should Try

To achieve genuine quantum advantage on this task:

| Limitation | Fix |
|------------|-----|
| 6-qubit information bottleneck | **More qubits** — 20-50 qubits could encode much more of the 153-dim space |
| Fixed non-trainable encoding | **Data re-uploading** (Pérez-Salinas et al. 2020) — trainable angle encoding layers interleaved with the circuit |
| Linear quantum kernel only | **Fidelity kernel** K(i,j) = &#124;⟨0&#124;U†(xᵢ)U(xⱼ)&#124;0⟩&#124;² — the true inner product in Hilbert space |
| G3 random circuits | **Variational Quantum Circuit (VQC)** — trainable parameterised gates optimised end-to-end |
| 836-sample dataset | **Larger dataset** — quantum kernels are O(N²) in circuit evaluations; need hardware acceleration |
| Skip connection bypass | Remove skip; force quantum features to carry all information |

### What This Codebase Does Establish

1. A clean **quantum reservoir computing pipeline** for molecular property prediction
2. **Honest ablation methodology**: stratified splits, train-only scaling, skip-removed evaluation
3. **RFD circuit selection** as a dataset-aware pre-screening tool (though its predictive power for R² is weak at 6 qubits)
4. **CKA as a diagnostic**: measuring quantum kernel alignment with the label kernel before training
5. A **reproducible benchmark** on PDBbind for future quantum ML methods to compare against

---

## Project Structure

```
FAST-master/model/
├── 3dcnn/              # 3D CNN (PyTorch) voxel grid → 10-dim fc1 embedding
├── sgcnn/              # Spatial Graph CNN (PotentialNet) atom graph → 54-dim embedding
├── quantum_fusion/     # Main research code
│   ├── main_train.py                  # Model definitions, data loading, ClassicalMLPBaseline
│   ├── testing_random_unitaries.py    # G3 circuit generation, RFD, extract_quantum_features
│   ├── ablation_study.py              # Runs A/B/C/D ablation — honest quantum contribution
│   ├── quantum_kernel_study.py        # Quantum kernel sweep, CKA, KPCA, all kernel plots
│   ├── evaluate_top25.py              # Historical: 25-circuit MLP sweep with skip connection
│   ├── extract_3dcnn_features.py      # Extracts 3DCNN fc1 embeddings → refined_3dcnn_features.npz
│   ├── extract_sgcnn_features.py      # Extracts SGCNN hidden embeddings → refined_sgcnn_features.npz
│   ├── refined_3dcnn_features.npz     # Precomputed 3DCNN embeddings (836 complexes, 10-dim)
│   ├── refined_sgcnn_features.npz     # Precomputed SGCNN embeddings (836 complexes, 54-dim)
│   ├── ablation_results.csv           # A/B/C/D ablation table
│   ├── ablation_r2_bar.png            # Ablation bar chart
│   └── quantum_kernel_study_output/   # All kernel study plots and CSV
└── fusion_tf/          # TensorFlow fusion baseline
```

---

## Reproducing Results

```bash
cd FAST-master/model/quantum_fusion

# Ablation study (A/B/C/D conditions, ~5 min)
python ablation_study.py --n-circuits 20 --epochs 50

# Quantum kernel sweep (CKA, KPCA, all kernel plots, ~5 min)
python quantum_kernel_study.py --n-circuits 20 --n-qubits 6
```

Both scripts use the same stratified random split (seed=42, 80/20 train/test) and
train-only StandardScaler to prevent data leakage.

---

## Getting Started

### Prerequisites

- Python 3.8+
- ChimeraX (for molecular preprocessing, optional)
- CUDA GPU recommended for 3DCNN/SGCNN embedding extraction

### Installation

**Option A — conda (recommended, matches the environment used to produce the paper's results):**

```bash
git clone https://github.com/sanskriti-ss/bindingaffinity.git
cd bindingaffinity
conda env create -f environment.yml
conda activate acnn_env
```

**Option B — pip only:**

```bash
git clone https://github.com/sanskriti-ss/bindingaffinity.git
cd bindingaffinity
pip install -r requirements.txt
```

`requirements.txt` is fully pinned to the versions used to generate the results in this
repository (PyTorch 2.10, PennyLane 0.44, Qiskit 2.3, Ingenii Quantum 0.1.1, etc.).
`torch-geometric` wheels are platform/CUDA-specific — if the plain `pip install` fails for
your platform, install PyTorch first, then follow the
[PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
for a matching `torch-geometric` build before re-running `pip install -r requirements.txt`.

### Dataset

Uses PDBbind 2020 refined set (5316 complexes; 836 have complete featurization):

1. Go to https://www.pdbbind-plus.org.cn/ → Download → Log in
2. Download "Protein-ligand complexes: The refined set" (~658 MB)
3. Place at `data/refined-set/`

---

## References

- Domingo et al. (2022). *Taking advantage of noise in quantum reservoir computing.* Scientific Reports.
- Schuld & Killoran (2022). *Is quantum advantage the right goal for quantum machine learning?* PRX Quantum.
- Pérez-Salinas et al. (2020). *Data re-uploading for a universal quantum classifier.* Quantum.
- Cortes et al. (2012). *Algorithms for learning kernels based on centered alignment.* JMLR.

---

## Acknowledgments

- Fetch.AI × BruinAI × QCSA Team
- RDKit, PennyLane, PyTorch, Qiskit communities
- PDBbind team for structural biology data
- ChimeraX development team

## Contributing & Contact

Contributions welcome — especially implementations of data re-uploading, fidelity kernels, or VQC variants. Open an issue or pull request on GitHub.

