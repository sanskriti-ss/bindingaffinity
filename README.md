# Protein-Ligand Binding Affinity Prediction

A comprehensive quantum computational chemistry project for predicting protein-ligand binding affinity using 3D spatial representations and deep learning techniques. This project replicates and extends hybrid methodologies from a recent a research paper [binding affinity](https://www.nature.com/articles/s41598-023-45269-y) in molecular modeling and drug discovery.

## 🧬 Project Overview

This project implements a complete pipeline for:

- Processing protein and ligand structures from PDB files
- Converting molecular data to MOL2 format with charges and hydrogens
- Creating 3D voxel representations of protein-ligand binding sites
- Building spatial feature tensors for machine learning models

## Project Structure

```text
bindingaffinity/
├── batch_process_chimerax.py          # ChimeraX batch processing script
├── step4_spatial_representation_3d.ipynb  # Main analysis notebook
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
└── demo_charges/                      # Processed molecular data
    ├── 1a30/                         # PDB ID: 1a30
    │   ├── 1a30_protein.mol2         # Protein structure with charges
    │   ├── 1a30_ligand.mol2          # Ligand structure with charges
    │   └── 1a30_pocket.mol2          # Binding pocket region
    ├── 1bcu/                         # Additional protein complexes
    └── ... (228 total protein complexes)
```

## Key Features

### 1. Molecular Preprocessing

- **ChimeraX Integration**: Automated batch processing of PDB files
- **Hydrogen Addition**: Adds missing hydrogens using appropriate protonation states
- **Charge Assignment**: Computes partial charges for all atoms
- **MOL2 Conversion**: Converts structures to MOL2 format for downstream analysis

### 2. 3D Spatial Representation

- **Voxelization**: Converts 3D molecular structures to regular grids
- **Multi-channel Features**: 19-channel feature representation including:
  - Element one-hot encoding (B, C, N, O, P, S, Se, halogens, metals)
  - Hybridization states
  - Bond connectivity information
  - Structural properties (hydrophobic, aromatic, H-bond donor/acceptor)
  - Partial charges (Gasteiger method)
  - Molecule type classification

### 3. Binding Site Analysis

- **Pocket Detection**: Identifies binding sites within protein structures
- **Cutoff-based Selection**: Focuses on protein atoms within specified distance of ligands
- **Gaussian Smoothing**: Applies 3D Gaussian kernels for feature distribution

### 4. Data Processing Pipeline

- **Batch Processing**: Handles 228 protein complexes efficiently
- **Error Handling**: Robust processing with comprehensive logging
- **Quality Control**: Validates molecular structures and filters problematic cases

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
pip install -r requirements.txt
```

3. Install ChimeraX (for preprocessing):
   - Download from [ChimeraX website](https://www.cgl.ucsf.edu/chimerax/)
   - Update the path in `batch_process_chimerax.py`

### Usage

1. **Molecular Preprocessing**:

```bash
python batch_process_chimerax.py
```

2. **3D Spatial Analysis**:
   Open and run the Jupyter notebook:

```bash
jupyter notebook "Copy_of_[clyde_explore]_step4_spatial_representation_3d.ipynb"
```

## Dataset
(FOR NOW)
The project includes 228 protein-ligand complexes from the Protein Data Bank (PDB), processed into standardized MOL2 format with:

- Complete hydrogen atoms
- Partial charge assignments
- Binding pocket definitions
- Quality-controlled structures

## Technical Details

### Voxelization Parameters

- **Grid Size**: 48×48×48 voxels
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

## Applications

This pipeline enables:

- **Binding Affinity Prediction**: Train CNNs on 3D molecular representations
- **Virtual Screening**: Evaluate potential drug compounds
- **Structure-Activity Relationships**: Analyze molecular binding patterns
- **Drug Discovery**: Support lead optimization and design

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest enhancements.

##  License

This project is open source. Please see the LICENSE file for details.

## Acknowledgments

- ChimeraX development team for molecular visualization and processing tools
- RDKit community for cheminformatics libraries
- PDB for providing structural biology data
- Fetch.AI X BruinAI X QCSA Team

## Contact

For questions or collaborations, please open an issue on GitHub.
