#!/usr/bin/env python3
"""
Modeling Protein-Ligand Interactions with Atomic Convolutions and Quantum Neural Networks

This script is converted from the Jupyter notebook:
14_Modeling_Protein_Ligand_Interactions_With_Atomic_Convolutions.ipynb

Based on the DeepChem tutorial by Nathan C. Frey and Bharath Ramsundar
Original paper: "Atomic convolutional networks for predicting protein-ligand binding affinity"
arXiv:1703.10603 (2017)

ACNN Architecture Overview:
- Distance Matrix: Constructed from Cartesian atomic coordinates
- Atom type convolution: Uses neighbor-listed distance matrix
- Radial Pooling layer: Dimensionality reduction to prevent overfitting
- Atomistic fully connected network: Final prediction layers

This script includes both classical ACNN training and quantum neural network extensions.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from rdkit import Chem

# DeepChem imports
import deepchem as dc
from deepchem.molnet import load_pdbbind
from deepchem.models import AtomicConvModel
from deepchem.feat import AtomicConvFeaturizer

# Quantum ML imports 
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    from qiskit.circuit.library import ZFeatureMap
    from qiskit_machine_learning.algorithms import VQR
    from qiskit.primitives import Estimator
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit_algorithms.optimizers import L_BFGS_B
    QUANTUM_AVAILABLE = True
    print("Quantum ML libraries available. Quantum extensions will be included.")
except ImportError as e:
    print(f"Quantum ML libraries not available. Classical ACNN only. Error: {e}")
    QUANTUM_AVAILABLE = False


def setup_environment():
    """
    Setup the environment for running the script.
    Note: This replaces the Colab-specific setup from the notebook.
    """
    print("Setting up environment...")
    print(f"DeepChem version: {dc.__version__}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"NumPy version: {np.__version__}")
    

def configure_atomic_conv_featurizer():
    """
    Configure the AtomicConvFeaturizer with appropriate parameters.
    
    Returns:
        AtomicConvFeaturizer: Configured featurizer
    """
    f1_num_atoms = 100  # maximum number of atoms to consider in the ligand
    f2_num_atoms = 1000  # maximum number of atoms to consider in the protein
    max_num_neighbors = 12  # maximum number of spatial neighbors for an atom

    acf = AtomicConvFeaturizer(
        frag1_num_atoms=f1_num_atoms,
        frag2_num_atoms=f2_num_atoms,
        complex_num_atoms=f1_num_atoms + f2_num_atoms,
        max_num_neighbors=max_num_neighbors,
        neighbor_cutoff=4
    )
    
    print(f"Configured AtomicConvFeaturizer:")
    print(f"  - Ligand atoms: {f1_num_atoms}")
    print(f"  - Protein atoms: {f2_num_atoms}")
    print(f"  - Max neighbors: {max_num_neighbors}")
    
    return acf


def load_pdbbind_data(featurizer, dataset_size='core'):
    """
    Load and preprocess PDBbind dataset.
    
    Args:
        featurizer: AtomicConvFeaturizer instance
        dataset_size: 'core' or 'refined'
    
    Returns:
        tuple: (tasks, datasets, transformers)
    """
    print(f"\nLoading PDBbind dataset ({dataset_size})...")
    start_time = time.time()
    
    # Load PDBbind dataset
    # pocket=True uses only binding pocket instead of entire protein (saves memory)
    tasks, datasets, transformers = load_pdbbind(
        featurizer=featurizer,
        save_dir='.',
        data_dir='.',
        pocket=True,
        reload=False,
        set_name=dataset_size
    )
    
    load_time = time.time() - start_time
    print(f"Dataset loaded in {load_time:.2f} seconds")
    
    return tasks, datasets, transformers


class DataTransformer(dc.trans.Transformer):
    """Remove samples where features are None"""
    
    def __init__(self):
        super().__init__(transform_X=True, transform_y=True, transform_w=True, transform_ids=True)
    
    def transform_array(self, x, y, w, ids):
        # Check if any samples have None features
        if x is None:
            return x, y, w, ids
        
        # For arrays, check for None values element-wise
        kept_rows = np.array([sample is not None for sample in x])
        
        if not np.any(kept_rows):
            # If no valid samples, return empty arrays
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        return x[kept_rows], y[kept_rows], w[kept_rows], ids[kept_rows]


def preprocess_datasets(datasets):
    """
    Apply data preprocessing to remove invalid samples.
    
    Args:
        datasets: List of DeepChem datasets
    
    Returns:
        tuple: (train, val, test) datasets
    """
    print("\nPreprocessing datasets...")
    
    # Apply transformer to remove None values
    transformer = DataTransformer()
    datasets = [d.transform(transformer) for d in datasets]
    
    train, val, test = datasets
    
    print(f"Dataset sizes after preprocessing:")
    print(f"  - Training: {len(train)}")
    print(f"  - Validation: {len(val)}")
    print(f"  - Test: {len(test)}")
    
    return train, val, test


def train_atomic_conv_model(train_dataset, val_dataset, 
                           f1_num_atoms=100, f2_num_atoms=1000, 
                           max_num_neighbors=12, max_epochs=50):
    """
    Train the Atomic Convolutional Neural Network.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        f1_num_atoms: Number of ligand atoms
        f2_num_atoms: Number of protein atoms
        max_num_neighbors: Maximum neighbors per atom
        max_epochs: Maximum training epochs
    
    Returns:
        tuple: (model, losses, val_losses)
    """
    print("\nInitializing AtomicConvModel...")
    
    # Create the model with hyperparameters from the original paper
    acm = AtomicConvModel(
        n_tasks=1,
        frag1_num_atoms=f1_num_atoms,
        frag2_num_atoms=f2_num_atoms,
        complex_num_atoms=f1_num_atoms + f2_num_atoms,
        max_num_neighbors=max_num_neighbors,
        batch_size=12,
        layer_sizes=[32, 32, 16],
        learning_rate=0.003,
    )
    
    # Initialize loss tracking
    losses, val_losses = [], []
    
    # Define validation callback
    metric = dc.metrics.Metric(dc.metrics.score_function.rms_score)
    step_cutoff = len(train_dataset) // 12
    
    def val_cb(model, step):
        """Validation callback to track training progress"""
        if step % step_cutoff != 0:
            return
        val_loss = model.evaluate(val_dataset, metrics=[metric])['rms_score']**2
        train_loss = model.evaluate(train_dataset, metrics=[metric])['rms_score']**2
        val_losses.append(val_loss)
        losses.append(train_loss)
        print(f"Step {step}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Train the model
    print(f"Training model for {max_epochs} epochs...")
    start_time = time.time()
    
    acm.fit(
        train_dataset, 
        nb_epoch=max_epochs, 
        max_checkpoints_to_keep=1,
        callbacks=[val_cb]
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return acm, losses, val_losses


def plot_training_curves(losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        losses: Training losses
        val_losses: Validation losses
        save_path: Optional path to save the plot
    """
    print("\nPlotting training curves...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(range(len(losses)), losses, label='Train Loss', alpha=0.7)
    ax.scatter(range(len(val_losses)), val_losses, label='Validation Loss', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (RMS²)')
    ax.set_title('ACNN Training Progress')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def evaluate_model(model, datasets, dataset_names=['train', 'val', 'test']):
    """
    Evaluate the trained model on all datasets.
    
    Args:
        model: Trained AtomicConvModel
        datasets: List of datasets
        dataset_names: Names for the datasets
    
    Returns:
        dict: Evaluation results
    """
    print("\nEvaluating model performance...")
    
    score_metric = dc.metrics.Metric(dc.metrics.score_function.pearson_r2_score)
    results = {}
    
    for name, dataset in zip(dataset_names, datasets):
        score = model.evaluate(dataset, metrics=[score_metric])
        results[name] = score['pearson_r2_score']
        print(f"{name.capitalize()} R² score: {score['pearson_r2_score']:.4f}")
    
    return results


# Quantum Neural Network Extension
def flatten_dc_features(X_dc):
    """
    Flatten DeepChem features into classical matrix format.
    
    Args:
        X_dc: DeepChem feature arrays
    
    Returns:
        np.ndarray: Flattened feature matrix
    """
    flat = []
    for sample in X_dc:
        # Each sample is a tuple/list of 9 arrays (one per feature type)
        arrays = [arr.flatten() for arr in sample]
        flat.append(np.concatenate(arrays))
    return np.vstack(flat)


def conv_instruction(n, prefix):
    """
    Create quantum convolutional instruction.
    
    Args:
        n: Number of qubits
        prefix: Parameter prefix
    
    Returns:
        Instruction: Quantum circuit instruction
    """
    params = ParameterVector(prefix, length=3*n)
    qc = QuantumCircuit(n, name="Conv")
    idx = 0
    
    for q1, q2 in zip(range(0, n, 2), range(1, n, 2)):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi/2, 1)
        sub.cx(1, 0)
        sub.rz(params[idx], 0)
        sub.ry(params[idx+1], 1)
        sub.cx(0, 1)
        sub.ry(params[idx+2], 1)
        qc.compose(sub, [q1, q2], inplace=True)
        qc.barrier()
        idx += 3
    
    return qc.to_instruction()


def pool_instruction(srcs, sinks, prefix):
    """
    Create quantum pooling instruction.
    
    Args:
        srcs: Source qubits
        sinks: Sink qubits
        prefix: Parameter prefix
    
    Returns:
        Instruction: Quantum circuit instruction
    """
    params = ParameterVector(prefix, length=3*len(srcs))
    n = len(srcs) + len(sinks)
    qc = QuantumCircuit(n, name="Pool")
    idx = 0
    
    for s, t in zip(srcs, sinks):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi/2, 1)
        sub.cx(1, 0)
        sub.rz(params[idx], 0)
        sub.ry(params[idx+1], 1)
        sub.cx(0, 1)
        sub.ry(params[idx+2], 1)
        qc.compose(sub, [s, t], inplace=True)
        qc.barrier()
        idx += 3
    
    return qc.to_instruction()


def train_quantum_model(datasets, n_qubits=6):
    """
    Train quantum neural network on the datasets.
    
    Args:
        datasets: Tuple of (train, val, test) datasets
        n_qubits: Number of qubits to use
    
    Returns:
        tuple: (vqr_model, predictions, scalers)
    """
    if not QUANTUM_AVAILABLE:
        print("Quantum ML libraries not available. Skipping quantum training.")
        return None, None, None
    
    print(f"\nTraining Quantum Neural Network with {n_qubits} qubits...")
    
    train, val, test = datasets
    
    # Extract features and targets
    X_train_dc, y_train = train.X, train.y
    X_val_dc, y_val = val.X, val.y
    X_test_dc, y_test = test.X, test.y
    
    # Flatten DeepChem features
    X_train_flat = flatten_dc_features(X_train_dc)
    X_val_flat = flatten_dc_features(X_val_dc)
    X_test_flat = flatten_dc_features(X_test_dc)
    
    # Scale and apply PCA
    fm_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_all = fm_scaler.fit_transform(
        np.vstack([X_train_flat, X_val_flat, X_test_flat])
    )
    
    pca = PCA(n_components=n_qubits)
    X_all_pca = pca.fit_transform(X_all)
    
    # Split back
    N_train = X_train_flat.shape[0]
    N_val = X_val_flat.shape[0]
    X_train_pca = X_all_pca[:N_train]
    X_val_pca = X_all_pca[N_train:N_train+N_val]
    X_test_pca = X_all_pca[N_train+N_val:]
    
    # Scale targets
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_all = y_scaler.fit_transform(
        np.vstack([y_train.reshape(-1,1), y_val.reshape(-1,1), y_test.reshape(-1,1)])
    )
    y_train_scaled = y_all[:N_train].ravel()
    y_val_scaled = y_all[N_train:N_train+N_val].ravel()
    y_test_scaled = y_all[N_train+N_val:].ravel()
    
    # Build quantum circuit
    feature_map = ZFeatureMap(num_qubits=n_qubits, reps=1)
    
    qc_cnn = QuantumCircuit(n_qubits)
    qc_cnn.compose(feature_map, range(n_qubits), inplace=True)
    qc_cnn.append(conv_instruction(n_qubits, "c1"), range(n_qubits))
    
    # Pooling
    src = list(range(0, n_qubits, 2))
    snk = list(range(1, n_qubits, 2))
    qc_cnn.append(pool_instruction(src, snk, "p1"), range(n_qubits))
    
    # Train VQR
    vqr = VQR(
        feature_map=feature_map,
        ansatz=qc_cnn,
        optimizer=L_BFGS_B(maxiter=150),
        estimator=Estimator(),
    )
    
    print("Fitting quantum model...")
    vqr.fit(X_train_pca, y_train_scaled)
    
    # Make predictions
    y_val_pred_scaled = vqr.predict(X_val_pca)
    y_test_pred_scaled = vqr.predict(X_test_pca)
    
    y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled.reshape(-1,1)).ravel()
    y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1,1)).ravel()
    
    print("Quantum model training completed!")
    
    return vqr, (y_val_pred, y_test_pred), (fm_scaler, pca, y_scaler)


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("Modeling Protein-Ligand Interactions with Atomic Convolutions")
    print("=" * 80)
    
    # Setup
    setup_environment()
    
    # Configure featurizer
    acf = configure_atomic_conv_featurizer()
    
    # Load data
    tasks, datasets, transformers = load_pdbbind_data(acf, dataset_size='core')
    
    # Preprocess data
    train, val, test = preprocess_datasets(datasets)
    
    # Train classical ACNN
    model, losses, val_losses = train_atomic_conv_model(
        train, val, 
        f1_num_atoms=100, 
        f2_num_atoms=1000, 
        max_epochs=50
    )
    
    # Plot training curves
    plot_training_curves(losses, val_losses, save_path='training_curves.png')
    
    # Evaluate classical model
    results = evaluate_model(model, [train, val, test])
    
    # Train quantum model (if available)
    if QUANTUM_AVAILABLE:
        print("\n" + "="*60)
        print("Quantum Neural Network Extension")
        print("="*60)
        
        qnn_model, qnn_predictions, qnn_scalers = train_quantum_model(
            (train, val, test), n_qubits=6
        )
        
        if qnn_model is not None:
            y_val_pred, y_test_pred = qnn_predictions
            print(f"Quantum model validation predictions shape: {y_val_pred.shape}")
            print(f"Quantum model test predictions shape: {y_test_pred.shape}")
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
    
    # Summary
    print(f"\nFinal Results:")
    print(f"Classical ACNN Performance:")
    for dataset, score in results.items():
        print(f"  {dataset.capitalize()} R²: {score:.4f}")
    
    print(f"\nThe model has been trained on {len(train)} training samples")
    print(f"and validated on {len(val)} validation samples.")
    print(f"Test set contains {len(test)} samples.")
    
    return model, (train, val, test), results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run the main analysis
    try:
        model, datasets, results = main()
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
