# Protein-Ligand Binding Affinity Prediction

A binding affinity prediction project exploring the frontier of quantum-classical hybrid machine learning for molecular property prediction. This project implements and compares 3D CNN, Spatial Graph CNN (SGCNN/PotentialNet), and a novel **Quantum Fusion** architecture: a quantum reservoir computing model that fuses deep learning embeddings with fixed G3 random circuits. Built as a proof-of-concept and research framework for drug discovery applications.


## Project Overview

This project implements and compares multiple approaches for protein-ligand binding affinity prediction:

**Pipeline:**
- Processing protein and ligand structures from PDB files
- Converting molecular data to MOL2 format with charges and hydrogens
- Creating 3D voxel representations of protein-ligand binding sites
- Extracting 3DCNN (fc1, 10-dim) and SGCNN (hidden, 54-dim) deep feature embeddings
- Fusing embeddings with RDKit pocket/ligand features into a 153-dim vector
- Training a quantum reservoir computing model using G3 random circuits (H, T, CNOT)
- Circuit selection via Reservoir Feature Diversity (RFD) expressibility scoring
- Analysis and performance benchmarking across all model families

**Research Insights:**
- Quantum reservoir computing (fixed G3 circuits + classical MLP head) achieves **R² ≈ 0.86** on the full PDBbind refined set (4641 complexes)
- Fusing 3DCNN and SGCNN embeddings with RDKit features into a 153-dim input is critical — RDKit-only features give R² ≈ 0.10
- Circuit expressibility pre-selection (RFD) consistently identifies circuits that generalise better
- Demonstrates a viable NISQ-era approach: the quantum reservoir is fixed and non-trainable; only classical layers are optimised

## Project Structure

```
FAST-master/model/
├── 3dcnn/              # 3D CNN (PyTorch) voxel grid 10-dim fc1 embedding
├── sgcnn/              # Spatial Graph CNN (PotentialNet) atom graph 54-dim embedding
├── quantum_fusion/     # Quantum Fusion: fused features --> G3 reservoir --> MLP head
│   ├── main_train.py            # Model definitions & data loading
│   ├── testing_random_unitaries.py  # G3 circuit generation & RFD selection
│   ├── evaluate_top5.py         # 100-circuit sweep, top-25 by RFD, top-5 results
│   ├── extract_3dcnn_features.py
│   ├── extract_sgcnn_features.py
│   ├── top5_unitary_results.csv # latest benchmark results
│   ├── scatter_best.png         # predicted vs actual for best circuit
│   └── top5_r2_bar.png          # R² bar chart for top-5 circuits
└── fusion_tf/          # TF fusion baseline
```

## Recent Developments

### Quantum Fusion Architecture

The quantum fusion model (`ModelHybridFC_Reservoir`) follows the Quantum Reservoir Computing paradigm of Domingo et al. (2022):

1. **Feature fusion** — pocket AA-composition + physicochemical (25-dim), ligand ECFP4+descriptor PCA (64-dim), 3DCNN fc1 embedding (10-dim), SGCNN hidden embedding (54-dim) → **153-dim** total input
2. **Classical compressor** — FC(153→24) with BatchNorm, FC(24 --> 6), tanh·π encoding
3. **Fixed quantum reservoir** — 6-qubit G3 circuit (H, T, CNOT gates); X/Y/Z Pauli measurements yield **18 quantum features**; skip-connect appends the 6-dim encoding → **24-dim combined**
4. **MLP regression head** — Linear(24→64) → BN → ReLU → Dropout(0.2) → Linear(64→32) → ReLU → Linear(32→1)

**Circuit selection:** 100 G3 circuits are generated, scored by Reservoir Feature Diversity (RFD) expressibility, and the top-25 are fully trained (50 epochs, Adam, LR=3×10⁻⁴). The best 5 are reported below.

### Performance Results — Top-5 Quantum Reservoir Circuits (Note that whenever you run main_train in quantum fusion, you get different circuits everytime hence random unitary circuits!)

Evaluated on 697-sample held-out test set (PDBbind 2020 refined set, 4641 complexes total, 3248/696/697 train/val/test split, label mean=6.42, std=1.976 pKi):

| Rank | Circuit | Test R² | Adj R² | Pearson r | Spearman ρ | RMSE (pKi) | MAE (pKi) |
|------|---------|---------|--------|-----------|------------|-----------|----------|
| 1 | #12 | **0.8603** | 0.821 | **0.9281** | **0.920** | **0.6638** | 0.4948 |
| 2 | #15 | 0.8548 | 0.814 | 0.9246 | 0.916 | 0.6768 | 0.4886 |
| 3 | #10 | 0.8558 | 0.815 | 0.9268 | 0.920 | 0.6744 | 0.4908 |
| 4 | #2  | 0.8480 | 0.805 | 0.9213 | 0.914 | 0.6925 | 0.4838 |
| 5 | #8  | 0.8586 | 0.819 | 0.9272 | 0.918 | 0.6679 | **0.4686** |

Scatter plot (predicted vs actual, best circuit): `FAST-master/model/quantum_fusion/scatter_best.png`
R² comparison bar chart: `FAST-master/model/quantum_fusion/top5_r2_bar.png`

**Key Findings:**
- All top-5 circuits achieve R² > 0.848 and Pearson r > 0.92, indicating strong predictive correlation
- Top circuit (#12) achieves RMSE ≈ 0.66 pKi — competitive with classical deep learning baselines on PDBbind
- Variance across top-5 is small (ΔR² < 0.013), suggesting stable learning regardless of circuit topology
- The quantum reservoir adds complementary non-linear projections that consistently benefit the MLP head when the input feature space is rich (153-dim fused)


## Getting Started

### Prerequisites

- Python 3.8+
- ChimeraX (for molecular preprocessing)
- CUDA-capable GPU (recommended for deep learning)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/sanskriti-ss/bindingaffinity.git
cd bindingaffinity
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt (and other requirements.txt if needed)
```

3. (Optional for exploratory steps) Install ChimeraX (for preprocessing):
   - Download from [ChimeraX website](https://www.cgl.ucsf.edu/chimerax/)
   - Update the path in `batch_process_chimerax.py`

```bash
# Molecular preprocessing with ChimeraX
python batch_process_chimerax.py
```

(Optional, NOT the code, just for demos)

```bash
jupyter notebook step4_spatial_representation_3d.ipynb
jupyter notebook step5_basicML.ipynb
```

4. Downlaod the datasets (instructions below in ## Datasets)

5. Run the files in FAST-master
Instructions in the repo!
e.g. To train or test fusion model, run `model/fusion/main_fusion_pdbbind.py`
but really read the repo :)


## Dataset

The project uses 5316 protein-ligand complexes with binding affinity data (ΔG) processed into multiple formats:

**CSV Data (pdbbind_with_dG.csv):**
- Binding constants (Ki, Kd) with automatic unit conversion to nM
- Experimental binding free energies (ΔG) in kcal/mol
- Protein resolution and experimental conditions
- Quality-controlled and outlier-filtered dataset

**3D Molecular Grids:**
- Ligand grids: (229, 19, 64, 64, 64) - Ligand spatial representations
- Pocket grids: (210, 19, 64, 64, 64) - Binding site representations  
- Protein grids: (188, 19, 64, 64, 64) - Full protein context
- 19-channel feature encoding including atoms, bonds, charges, properties

**Processed Structures:**
- 228 protein-ligand complexes from PDB
- Complete hydrogen atoms and partial charges
- MOL2 format with binding pocket definitions
- Quality-controlled molecular structures

### To download the data:
1) Go to https://www.pdbbind-plus.org.cn/
2) Click 'Download' in the upper tabs. Log in! Make an account if you haven't already.
3) Download "Protein-ligand complexes: The refined set" with 5316 protein-ligand complexes. It is 658MB so it might take a bit.
4) Make sure you 

## Technical Details

### Voxelization Parameters

- **Grid Size**: 48x48x48 voxels 
- **Voxel Size**: 1.0 Å
- **Gaussian Radius**: 2 voxels
- **Gaussian Sigma**: 1.0 voxel
- **Binding Site Cutoff**: 6.0 Å

### Feature Engineering

The system generates 19-channel feature tensors capturing:

- Atomic composition and properties
- Structural characteristics
- Electrostatic properties
- Molecular context (protein vs. ligand)


## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest enhancements.

## License

This project is open source. Please see the LICENSE file for details.

## Acknowledgments

- ChimeraX development team for molecular visualization and processing tools
- RDKit community for cheminformatics libraries
- PDB for providing structural biology data
- Fetch.AI X BruinAI X QCSA Team

## Contact

For questions or collaborations, please open an issue on GitHub.
