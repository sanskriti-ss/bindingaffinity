# Quantum-Enhanced 3D CNN for Protein-Ligand Binding Affinity

This directory contains a quantum-enhanced version of the FAST 3D CNN model for predicting protein-ligand binding affinity. The quantum enhancement aims to improve model performance through quantum computing techniques.

## Key Features

### Quantum Enhancements
- **Quantum Feature Layers**: Variational quantum circuits for feature transformation
- **Quantum Attention Mechanism**: Quantum-enhanced attention for improved feature selection
- **Hybrid Architecture**: Seamless integration of classical CNN with quantum layers
- **Fallback Support**: Automatically falls back to classical layers if quantum libraries unavailable

### Model Architecture
```
Input (19 x 48 x 48 x 48) 
    ↓
Classical 3D CNN Layers
    ↓
Quantum Feature Layer (6 qubits, 3 layers)
    ↓
Quantum Attention Layer (4 qubits)
    ↓
Quantum Prediction Layer (4 qubits, 2 layers)
    ↓
Output (Binding Affinity)
```

## Files

- `quantum_enhanced_model.py` - Quantum-enhanced 3D CNN model implementation
- `main_train_quantum.py` - Training script with quantum support
- `compare_models.py` - Utility to compare classical vs quantum performance
- `requirements_quantum.txt` - Dependencies including PennyLane

## Installation

### Install Quantum Dependencies
```bash
pip install -r requirements_quantum.txt
```

### For GPU-accelerated quantum computing (optional): You gotta follow pennylane ai instructions on how to install their lightning GPU
```bash
pip install pennylane-lightning[gpu]
```

## Usage

### Quick Start - Train Quantum Model
```bash
python main_train_quantum.py --checkpoint-dir quantum-train --checkpoint-iter 1 --epoch-count 50 --batch-size 64 --learning-rate 1e-3 --decay-rate 0.97 --use-quantum --quantum-features --quantum-attention --quantum-qubits 6 --quantum-layers 3  
```

### Compare Classical vs Quantum Models
```bash
python compare_models.py --mode both --epochs 5
```

### Train Classical Model Only
```bash
python main_train_quantum.py --epochs 10 --batch-size 8
```

### Advanced Quantum Configuration
```bash
python main_train_quantum.py \
    --use-quantum \
    --quantum-features \
    --quantum-attention \
    --quantum-qubits 6 \
    --quantum-layers 3 \
    --epochs 20 \
    --batch-size 4 \
    --learning-rate 2e-3
```

## ⚙️ Quantum Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--use-quantum` | Enable quantum enhancement | False |
| `--quantum-features` | Use quantum feature layers | True |
| `--quantum-attention` | Use quantum attention | True |
| `--quantum-qubits` | Number of qubits | 6 |
| `--quantum-layers` | Quantum circuit depth | 3 |

## 🧮 Quantum Circuit Details

### Feature Layer Circuit
- **Encoding**: Angle embedding with Y rotations
- **Ansatz**: Parameterized rotation gates (RX, RY, RZ) with CNOT entanglement
- **Connectivity**: Ring topology for enhanced entanglement
- **Measurements**: Pauli-Z expectation values

### Attention Layer Circuit
- **Purpose**: Quantum-enhanced attention weight computation
- **Qubits**: 4 qubits (smaller for attention efficiency)
- **Output**: Softmax-normalized attention weights

### Prediction Layer Circuit
- **Function**: Final binding affinity prediction
- **Architecture**: 2-layer variational circuit
- **Output**: 4-dimensional quantum features → classical linear layer


## 🔧 Troubleshooting

### PennyLane Not Available
The model automatically falls back to classical layers if PennyLane is not installed:
```
PennyLane not available - Using classical layers only
To enable quantum features, install with: pip install pennylane
```

### GPU Memory Issues
If you encounter GPU memory issues with quantum circuits:
- Reduce batch size: `--batch-size 2`
- Reduce number of qubits: `--quantum-qubits 4`
- Use CPU backend by setting device to "cpu"

### Slow Training
Quantum circuits add computational overhead:
- Start with fewer epochs for testing: `--epochs 5`
- Use smaller models for initial experiments
- Consider using `lightning.gpu` backend for acceleration

## References

1. Quantum Machine Learning for Molecular Property Prediction
2. Variational Quantum Algorithms for Classification
3. Quantum Neural Networks and Deep Learning
4. PennyLane Documentation: https://pennylane.ai/

## Contributing

To extend the quantum functionality:
1. Add new quantum layers in `quantum_enhanced_model.py`
2. Experiment with different quantum ansätze
3. Try alternative encoding strategies
4. Test on different molecular datasets

## License

Same as the original FAST project (MIT License)
