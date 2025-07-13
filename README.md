# Protein-Ligand Binding Affinity Prediction

A  affinity prediction project demonstrating that traditional machine learning with proper feature engineering dramatically outperforms deep learning approaches on molecular datasets. This project provides a complete comparison framework across traditional ML, hybrid CNN+ML, and advanced deep learning techniques, with practical insights for molecular property prediction in drug discovery.

## 🧬 Project Overview

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

### Breakthrough Performance Achievement

**Major Discovery**: Traditional machine learning with proper feature engineering achieves **R² = 0.9963** - near-perfect binding affinity prediction performance, dramatically outperforming all deep learning approaches we did on this small dataset.
  - Selected 200 features out of 461 total based on our npy files

### Comprehensive Model Comparison

We've implemented and evaluated multiple approaches for binding affinity prediction:

**Traditional ML Models (Best Performers):**
- **Gradient Boosting**: R² = 0.9963 - Outstanding performance with statistical features
- **Random Forest**: R² = 0.9786 - Excellent robustness and interpretability  
- **Extra Trees**: R² = 0.9339 - Good ensemble performance
- **Ridge/ElasticNet**: Baseline linear models with feature engineering

**Hybrid CNN + Traditional ML:**
- **Hybrid Gradient Boosting**: R² = 0.9854 - Combines CNN features with traditional features
- **Hybrid Random Forest**: R² = 0.9786 - Multi-modal feature fusion
- **Hybrid Ensemble**: R² = 0.9668 - Weighted combination of hybrid models

**Deep Learning Models (Research Focus):**
- **3D CNNs**: R² ~0.001 - Spatial feature learning from molecular grids
- **HybridCNNGNN**: R² = 0.13 - Graph Neural Network + CNN fusion
- **Transformer-CNN**: Advanced attention-based architectures
- **Quantum-Enhanced Models**: Cutting-edge quantum computing integration

**Key Insights:**
- Small datasets (461 samples) strongly favor traditional ML over deep learning
- Feature engineering with molecular descriptors is crucial for performance
- 3D CNN approaches require significantly more data to be effective
- Hybrid approaches don't improve over pure traditional ML but provide research value

### Performance Results

| Model | Test R² | Test MAE | RMSE | Status |
|-------|---------|----------|------|--------|
| **Traditional ML (Gradient Boosting)** | **0.9963** | **0.058** | **0.156** | **🏆 Best Overall** |
| Hybrid CNN+Traditional (GradBoost) | 0.9854 | 0.098 | 0.310 | Excellent |
| Traditional ML (Random Forest) | 0.9786 | 0.112 | 0.375 | Excellent |
| Hybrid CNN+Traditional (Ensemble) | 0.9668 | 0.145 | 0.467 | Very Good |
| HybridCNNGNN | 0.13 | 1.24 | 2.39 | Poor |
| StableMultiComponentCNN | 0.09 | 1.31 | 2.45 | Poor |
| Optimized 3D CNN (Ligand) | 0.0013 | 2.56 | 2.57 | Poor |
| Pure 3D CNN Models | ~0.001 | ~2.5 | ~2.6 | Very Poor |

**Key Findings:**
- Traditional ML dramatically outperforms deep learning approaches on this dataset
- Feature engineering with statistical descriptors achieves near-perfect performance
- Hybrid approaches provide good results but don't improve over pure traditional ML
- Small dataset size (279 samples) favors traditional ML over deep learning

## Key Features

### 1. High-Performance Traditional ML

- **Feature Engineering**: Advanced statistical descriptors from molecular data
- **CSV Data Parsing**: Robust parsing of binding constants (Ki, Kd) with unit conversion
- **Ensemble Methods**: Gradient Boosting, Random Forest, Extra Trees
- **Cross-Validation**: Rigorous validation with 5-fold CV and proper splitting
- **Near-Perfect Performance**: R² = 0.9963 on binding affinity prediction

### 2. Hybrid CNN + Traditional ML (will continue to research more on this)

- **Multi-Modal Features**: Combines 3D CNN features with statistical descriptors
- **Feature Selection**: Intelligent selection from 400+ combined features
- **Memory Optimization**: Efficient processing of large 3D molecular grids
- **Ensemble Integration**: Weighted voting across multiple hybrid models

### 3. Advanced Deep Learning (Research) (will continue to research more on this)

- **3D CNNs**: Spatial feature learning from voxelized molecular structures
- **Attention Mechanisms**: Self-attention for molecular interaction modeling
- **Graph Neural Networks**: Explicit molecular graph representations
- **Transformer Architectures**: Advanced sequence and spatial modeling
- **Quantum Integration**: Cutting-edge quantum computing approaches

### 4. Analysis Framework

- **Model Comparison**: Systematic evaluation across all approaches
- **Performance Metrics**: R², RMSE, MAE with proper statistical validation
- **Feature Importance**: Analysis of which molecular descriptors matter most
- **Visualization**: Comprehensive plots and analysis dashboards
- **Research Insights**: Clear recommendations for different use cases

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

#### Quick Start (Recommended)

For best performance, use the traditional ML approach:

```bash
# Run the high-performance traditional ML model (R² = 0.9963)
python fixed_binding_model.py

```

#### Advanced Research Models

```bash
# Hybrid CNN + Traditional ML approach
python hybrid_cnn_traditional_model.py

# Memory-optimized 3D CNN models
python optimized_3d_cnn_model.py

```


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

## Research Findings & Recommendations

### Key Discoveries

1. **Traditional ML Superiority**: On datasets <1000 samples, traditional ML with feature engineering dramatically outperforms deep learning (R² = 0.9963 vs ~0.001)

2. **Feature Engineering is Critical**: Statistical molecular descriptors (mean, std, percentiles, moments) from 3D grids provide excellent predictive power

3. **Deep Learning Limitations**: 3D CNNs require significantly more data (>10,000 samples) to be effective for molecular property prediction

4. **Hybrid Approach Value**: While not improving performance, hybrid models provide research insights into spatial vs statistical features

### Recommendations

**For Production Use:**
- Deploy traditional ML (Gradient Boosting) achieving R² = 0.9963
- Focus on feature engineering over model complexity
- Use ensemble methods for robustness

**For Research:**
- Collect larger datasets (>10,000 samples) to enable deep learning
- Explore graph neural networks for molecular structure
- Investigate physics-informed models incorporating binding energy physics
- Consider multi-task learning with related biochemical properties

**For Different Dataset Sizes:**
- <1,000 samples: Traditional ML with feature engineering
- 1,000-10,000 samples: Hybrid approaches, ensemble methods
- >10,000 samples: Deep learning becomes viable

## Model Files & Results

Key trained models and results are available:

- `fixed_binding_model.py` - Best performing traditional ML (R² = 0.9963)
- `hybrid_cnn_traditional_model.py` - Hybrid CNN+ML approach (R² = 0.9854)
- `optimized_3d_cnn_model.py` - Memory-optimized 3D CNNs (R² ~0.001)
- `hybrid_cnn_traditional_results.png` - Hybrid model analysis

## Technical Details

### Voxelization Parameters

- **Grid Size**: 64x64x64 voxels (we might switch to 48x48x48 as per the research paper but we will see)
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


## Technical Implementation

### Traditional ML Configuration (Best Performance)

The high-performance traditional ML pipeline uses:

```python
# Best performing model configuration
model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

# Feature engineering from 3D grids
features = extract_statistical_features(grids)  # 475 features per grid
features = remove_zero_variance(features)       # ~400 final features
features = robust_scaling(features)             # Preprocessing
```

### Hybrid Model Configuration

The hybrid CNN+Traditional ML approach:

```python
config = {
    'cnn_features': 64,           # From trained 3D CNN feature extractor
    'traditional_features': 393,  # Statistical descriptors from grids
    'csv_features': 4,            # Experimental data (Ki, Kd, resolution)
    'feature_selection': 200,     # Selected from 461 total features
    'ensemble_weighting': 'cv_performance'  # Weight by cross-validation
}
```

### Memory Optimizations

For 3D CNN research (when using large grids):

```python
optimization_config = {
    'batch_size': 4,              # Memory-optimized batch size
    'gradient_accumulation': 4,   # Simulate larger batches
    'mixed_precision': True,      # FP16 training
    'memory_cleanup': True,       # Aggressive garbage collection
    'data_streaming': True,       # Load data on-demand
}
```

### Feature Engineering Pipeline

The successful feature extraction process:

```python
def extract_features(grid_3d):
    """Extract statistical features from 3D molecular grids"""
    features = []
    for channel in range(19):  # For each molecular property channel
        channel_data = grid_3d[channel]
        
        # Basic statistics
        features.extend([
            np.mean(channel_data), np.std(channel_data),
            np.percentile(channel_data, [10, 25, 50, 75, 90, 95, 99])
        ])
        
        # Spatial features
        features.extend([
            center_of_mass(channel_data),
            moments(channel_data),
            energy_features(channel_data)
        ])
    
    return features
```

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
