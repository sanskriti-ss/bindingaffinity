# Protein-Ligand Binding Affinity Prediction

A comprehensive quantum computational chemistry project for predicting protein-ligand binding affinity using 3D spatial representations and advanced deep learning techniques. This project replicates and extends hybrid methodologies from a recent research paper [binding affinity](https://www.nature.com/articles/s41598-023-45269-y) in molecular modeling and drug discovery, featuring quantum-enhanced neural architectures and transformer-based models.

## 🧬 Project Overview

This project implements a complete pipeline for:

- Processing protein and ligand structures from PDB files
- Converting molecular data to MOL2 format with charges and hydrogens
- Creating 3D voxel representations of protein-ligand binding sites
- Building spatial feature tensors for machine learning models
- Training advanced CNN-Transformer  and quantum-enhanced models (Work in progress)
- Comprehensive model comparison and evaluation  (Work in progress)

## Project Structure

```text
bindingaffinity/
├── advanced_training.py               # Advanced CNN-Transformer training pipeline
├── integrated_training_clean.py       # Memory-optimized training system
├── models.py                          # Classical CNN architectures
├── models_2.py                        # Advanced Transformer-CNN models
├── quantum_models.py                  # Quantum-enhanced neural networks
├── step4_spatial_representation_3d.ipynb  # Main analysis notebook
├── step5_basicML.ipynb               # Machine learning experiments
├── protein_data_reader.py            # Dataset loading and processing
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
├── best_*.pth                        # Trained model checkpoints
├── *_predictions.png                 # Model evaluation plots
├── *_training_history.png           # Training progress visualizations
├── model_comparison_results.csv      # Comprehensive model performance
└── demo_charges/                     # Processed molecular data
    ├── 1a30/                        # PDB ID: 1a30
    │   ├── 1a30_protein.mol2        # Protein structure with charges
    │   ├── 1a30_ligand.mol2         # Ligand structure with charges
    │   └── 1a30_pocket.mol2         # Binding pocket region
    ├── 1bcu/                        # Additional protein complexes
    └── ... (228 total protein complexes)
```

## Recent Developments

### Advanced Model Architectures

We've implemented cutting-edge neural architectures for binding affinity prediction:

**Classical Models:**
- **SimpleBidirectionalCNN**: Basic CNN baseline
- **StableMultiComponentCNN**: Memory-optimized multi-scale CNN
- **MemoryEfficientCNN**: Lightweight CNN with attention mechanisms
- **AttentionEnhancedCNN**: Self-attention integrated CNN
- **LightweightAttentionCNN**: Efficient attention-based architecture

**Advanced Models:**
- **HybridCNNGNN**: Graph Neural Network + CNN fusion **(Current Best: R² = 0.13)**
- **AdvancedTransformerCNN**: Transformer encoder with CNN backbone
- **ResidualTransformerCNN**: ResNet-style Transformer architecture
- **HybridCNNGNNTransformer**: Complete CNN-GNN-Transformer fusion

**Quantum-Enhanced Models:**
- **QuantumEnhancedBindingAffinityPredictor**: Quantum circuit integration
- **QuantumHybridEnsemble**: Quantum ensemble methods
- **QuantumResilientCNN**: Noise-resistant quantum CNN

### Performance Results

| Model | Test R² | Test MAE | Status |
|-------|---------|----------|--------|
| HybridCNNGNN | 0.13 | 1.24 |  Best |
| StableMultiComponentCNN | 0.09 | 1.31 | Meh |
| MemoryEfficientCNN | 0.07 | 1.35 | bad |
| SimpleBidirectionalCNN | 0.04 | 1.42 | very bad |

*Note: Training is ongoing for advanced models (AdvancedTransformerCNN, ResidualTransformerCNN, QuantumEnhanced models)*

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

### 3. Advanced Training Features

- **Memory Optimization**: Gradient accumulation, mixed precision training, aggressive memory cleanup
- **Advanced Optimizers**: AdamW with different learning rates for CNN, Transformer, and classifier components
- **Sophisticated Scheduling**: Warmup + Cosine Annealing with Warm Restarts
- **Multiple Loss Functions**: Combined Loss, Focal Loss, Quantum-Aware Loss
- **Early Stopping**: R²-based with patience for optimal convergence
- **Model Checkpointing**: Automatic saving of best performing models

### 4. Quantum Computing Integration

- **Quantum Circuits**: Native quantum computing integration for molecular property prediction
- **Error Correction**: Quantum error mitigation and noise resilience
- **Hybrid Classical-Quantum**: Seamless integration with classical CNN architectures
- **Quantum Measurements**: Configurable shot counts and circuit depths

### 5. Binding Site Analysis

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

2. **Advanced Model Training**:

```bash
# Run the complete advanced training pipeline
python advanced_training.py

# For memory-optimized classical models
python integrated_training_clean.py
```

3. **Individual Model Testing**:

```bash
# Quick start example
python quick_start_example.py
```

4. **Jupyter Analysis**:
   Open and run the analysis notebooks:

```bash
jupyter notebook step4_spatial_representation_3d.ipynb
jupyter notebook step5_basicML.ipynb
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

- **Binding Affinity Prediction**: Train advanced CNNs, Transformers, and Quantum models on 3D molecular representations
- **Virtual Screening**: Evaluate potential drug compounds using state-of-the-art neural architectures
- **Structure-Activity Relationships**: Analyze molecular binding patterns with attention mechanisms
- **Drug Discovery**: Support lead optimization using quantum-enhanced computational methods
- **Model Comparison**: Comprehensive benchmarking across classical and quantum approaches
- **Research**: Cutting-edge experiments in quantum machine learning for molecular modeling

## Technical Implementation

### Training Configuration

The advanced training pipeline (`advanced_training.py`) includes:

```python
config = {
    'batch_size': 4,                    # Memory-optimized batch size
    'num_epochs': 100,                  # Extended training epochs
    'learning_rate': 1e-4,              # Adaptive learning rate
    'gradient_accumulation_steps': 4,   # Simulate larger batches
    'mixed_precision': True,            # Memory optimization
    'loss_type': 'combined',            # Advanced loss functions
    'quantum_enabled': True,            # Quantum model support
    'early_stopping_patience': 30,     # Intelligent stopping
}
```

### Memory Optimizations

- **Gradient Accumulation**: Simulates larger batch sizes
- **Mixed Precision**: FP16 training for GPU memory efficiency  
- **Aggressive Cleanup**: Automatic memory management
- **Pin Memory**: Optimized data transfer
- **Non-blocking**: Asynchronous GPU operations

### Model Architectures

**CNN Backbone**: Multi-scale feature extraction from 3D voxelized molecular data  
**Transformer Layers**: Self-attention for long-range molecular interactions  
**Graph Neural Networks**: Explicit molecular graph representation  
**Quantum Circuits**: Quantum feature encoding and processing  
**Fusion Mechanisms**: Advanced feature combination strategies

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
