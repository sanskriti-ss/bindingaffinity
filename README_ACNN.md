# Atomic Convolutions for Protein-Ligand Interactions

This repository contains Python scripts converted from the Jupyter notebook "14_Modeling_Protein_Ligand_Interactions_With_Atomic_Convolutions.ipynb".

## Overview

The scripts implement Atomic Convolutional Neural Networks (ACNNs) for predicting protein-ligand binding affinities using the PDBbind dataset. The implementation includes both classical deep learning and optional quantum neural network extensions.

## Files

- `atomic_convolutions_simple.py` - Streamlined version focusing only on classical ACNN
- `atomic_convolutions_protein_ligand.py` - Complete version with quantum neural network extensions
- `requirements_acnn.txt` - Python dependencies (Python 3.9-3.12)
- `README_ACNN.md` - This documentation

## Quick Start


#### Option A: Use pyenv (recommended)
```powershell
# Install pyenv for Windows: https://github.com/pyenv-win/pyenv-win
pyenv install 3.11.9
pyenv local 3.11.9
```

#### Option B: Download Python 3.11 from python.org
Download from https://www.python.org/downloads/ and install Python 3.11.x

### 2. Install Dependencies

⚠️ **Windows Users**: If you encounter compilation errors, use one of these methods:

#### Method A: Automated Windows Installation (Recommended)
```powershell
# Run the automated installation script
install_windows.bat
```

#### Method B: Use Conda (Most Reliable) (clyde: I used this to install the packages from deepchem tutorial)
```powershell
# Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
conda env create -f environment.yml
conda activate acnn_env
```

#### Method C: Pip with Pre-compiled Wheels Only
```powershell
# Create virtual environment
python -m venv acnn_env
acnn_env\Scripts\activate

# Install using only pre-compiled wheels (no compilation)
pip install --only-binary=all -r requirements_minimal.txt

# Then install chemistry packages with conda
conda install -c conda-forge rdkit deepchem mdtraj pdbfixer openmm
```

#### Method D: Standard Installation (May Require Build Tools)
```powershell
# Create a virtual environment (recommended)
python -m venv acnn_env
acnn_env\Scripts\activate

# Install requirements
pip install -r requirements_acnn.txt
```

### 2. Run the Simple Version

```powershell
python atomic_convolutions_simple.py
```

This will:
- Load the PDBbind core dataset (~200 complexes)
- Train an ACNN model for 50 epochs
- Display training progress and final evaluation metrics
- Save a training curve plot as `acnn_training_curves.png`

### 3. Run the Complete Version (with Quantum Extensions)

```powershell
# First install quantum dependencies =
pip install qiskit qiskit-machine-learning

# Run the full script
python atomic_convolutions_protein_ligand.py
```

## Key Features

### Atomic Convolutional Neural Network (ACNN)
- **Distance Matrix**: Constructs spatial relationships from 3D coordinates
- **Atom Type Convolution**: Exploits local chemical environments
- **Radial Pooling**: Dimensionality reduction to prevent overfitting
- **Fully Connected Layers**: Final prediction network

### Model Architecture
- Input: Protein-ligand complex with up to 1100 atoms (100 ligand + 1000 protein)
- Feature extraction: AtomicConvFeaturizer with 4Å neighbor cutoff
- Network: [32, 32, 16] hidden layers
- Output: Binding affinity prediction (pKd/pKi values)

### Dataset
- **PDBbind Core**: ~200 high-quality protein-ligand complexes
- **PDBbind Refined**: ~5000 complexes (change `set_name='refined'` in the code)
- Target: Binding affinity (Kd/Ki values)

## Expected Results

Based on the original paper (Gomes et al., 2017):
- Training R² ≈ 0.91
- Test R² ≈ 0.45

## Our optimized Results [WIP]
 - 

## Customization

### Hyperparameters

Modify these parameters in the scripts:

```python
# Featurizer parameters
f1_num_atoms = 100      # ligand atoms
f2_num_atoms = 1000     # protein atoms
max_num_neighbors = 12  # spatial neighbors
neighbor_cutoff = 4     # Angstrom cutoff

# Model parameters
batch_size = 12
layer_sizes = [32, 32, 16]
learning_rate = 0.003
max_epochs = 50
```

### Dataset Selection

```python
# Use core dataset (fast, ~200 complexes)
set_name = 'core'

# Use refined dataset (slower, ~5000 complexes)
set_name = 'refined'
```

### Using Full Protein vs Binding Pocket

```python
# Use only binding pocket (faster, less memory)
pocket = True

# Use entire protein (slower, more memory)
pocket = False
```

## Troubleshooting

### Windows Build Tools Issues

**Error**: `ERROR: Unknown compiler(s): [['icl'], ['cl'], ['cc'], ['gcc'], ['clang']...]`

**Cause**: Missing C++ compiler needed to build packages from source.

**Solutions**:

1. **Use pre-compiled wheels** (easiest):
```powershell
pip install --only-binary=all -r requirements_minimal.txt
```

2. **Install Visual Studio Build Tools**:
   - Download "Build Tools for Visual Studio" from Microsoft
   - Install with C++ build tools
   - Restart PowerShell and try again

3. **Use conda instead** (recommended):
```powershell
conda env create -f environment.yml
conda activate acnn_env
```


1. **Use conda with Python 3.11** (recommended):
```powershell
# Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
conda create -n acnn_env python=3.11
conda activate acnn_env
pip install -r requirements_acnn.txt
```

2. **Use pyenv-win**:
```powershell
# Install from https://github.com/pyenv-win/pyenv-win
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv acnn_env
acnn_env\Scripts\activate
pip install -r requirements_acnn.txt
```

3. **Manual Python 3.11 installation**:
   - Download Python 3.11.x from python.org
   - Install alongside your current Python
   - Use `py -3.11` instead of `python` in commands

### Common Issues

1. **Memory Errors**: Reduce `f2_num_atoms` or use `pocket=True`
2. **Installation Issues**: Try conda instead of pip for scientific packages
3. **GPU Issues**: The model will automatically use CPU if GPU is unavailable

### DeepChem Installation

If you encounter issues with DeepChem:

```powershell
# Alternative installation methods
conda install -c conda-forge deepchem
# or
pip install --pre deepchem
```

### Quantum Dependencies 

```powershell
pip install qiskit qiskit-machine-learning
```

## Performance Notes

- **Core dataset**: ~1-2 minutes training time
- **Refined dataset**: ~10-30 minutes training time
- **Memory usage**: 2-8 GB RAM depending on configuration
- **GPU**: Not required but will accelerate training

## References

1. Gomes, J., et al. "Atomic convolutional networks for predicting protein-ligand binding affinity." arXiv:1703.10603 (2017).
2. Original DeepChem tutorial by Nathan C. Frey and Bharath Ramsundar
3. PDBbind database: http://www.pdbbind.org.cn/

## License

This code is derived from DeepChem tutorials and follows the same open-source principles.
