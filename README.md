# Protein-Ligand Binding Affinity Prediction

A binding affinity prediction project demonstrating that traditional machine learning with proper feature engineering dramatically outperforms deep learning approaches on molecular datasets. This project provides a complete comparison framework across hybrid CNNs SGCNNs, QMLs, and advanced deep learning techniques, with practical insights for molecular property prediction in drug discovery, in order to speet up and provide a proof-of-concept to other similar research papers


## Project Overview

This project implements and compares multiple approaches for protein-ligand binding affinity prediction:


**Pipeline:**
- Processing protein and ligand structures from PDB files
- Converting molecular data to MOL2 format with charges and hydrogens  
- Creating 3D voxel representations of protein-ligand binding sites
- Extracting statistical features from molecular grids
- Training and comparing traditional ML, hybrid, and deep learning models
- Analysis and performance benchmarking

**Research Insights:**
- Demonstrates that traditional ML dramatically outperforms deep learning on small molecular datasets
- Shows the critical importance of feature engineering over model complexity
- Provides guidance on when to use different approaches based on dataset size
- Offers a complete framework for molecular property prediction research NPY Files

## Project Structure


## Recent Developments

### Comprehensive Model Comparison

We've implemented and evaluated multiple approaches for binding affinity prediction:

**Key Insights:**
- Small datasets strongly favor traditional ML over deep learning
- Feature engineering with molecular descriptors is crucial for performance
- 3D CNN approaches require significantly more data to be effective
- Hybrid approaches don't improve over pure traditional ML but provide research value

### Performance Results

| Model | Test R² | Test MAE | RMSE | Status |
|-------|---------|----------|------|--------|


**Key Findings:**
- Traditional ML dramatically outperforms deep learning approaches on this dataset
- Feature engineering with statistical descriptors achieves near-perfect performance
- Hybrid approaches provide good results but don't improve over pure traditional ML
- Small dataset size favors traditional ML over deep learning


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

3. Install ChimeraX (for preprocessing):
   - Download from [ChimeraX website](https://www.cgl.ucsf.edu/chimerax/)
   - Update the path in `batch_process_chimerax.py`


#### Data Processing

```bash
# Molecular preprocessing with ChimeraX
python batch_process_chimerax.py
```

#### Analysis Notebooks

```bash
jupyter notebook step4_spatial_representation_3d.ipynb
jupyter notebook step5_basicML.ipynb
```

## Dataset

The project uses 279 protein-ligand complexes with binding affinity data (ΔG) processed into multiple formats:

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
