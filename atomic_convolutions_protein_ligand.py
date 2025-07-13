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
# set environment variable to disable oneDNN optimizations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for reproducibility
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
    from sklearn.metrics import r2_score
    from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
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
    for i, sample in enumerate(X_dc):
        try:
            # Handle different feature structures
            if isinstance(sample, dict):
                # If sample is a dictionary, extract numeric arrays
                arrays = []
                for key, value in sample.items():
                    if isinstance(value, np.ndarray):
                        arrays.append(value.flatten())
                    elif isinstance(value, (list, tuple)):
                        arrays.append(np.array(value, dtype=np.float32).flatten())
                if arrays:
                    flat.append(np.concatenate(arrays))
                else:
                    # If no arrays found, create a small dummy feature
                    flat.append(np.array([0.0], dtype=np.float32))
            elif isinstance(sample, (list, tuple)):
                # Each sample is a tuple/list of arrays (one per feature type)
                arrays = []
                for arr in sample:
                    if isinstance(arr, np.ndarray):
                        # Convert to float and flatten
                        arrays.append(arr.astype(np.float32).flatten())
                    elif isinstance(arr, dict):
                        # Handle nested dictionaries
                        for v in arr.values():
                            if isinstance(v, np.ndarray):
                                arrays.append(v.astype(np.float32).flatten())
                            elif isinstance(v, (int, float)):
                                arrays.append(np.array([float(v)], dtype=np.float32))
                    elif isinstance(arr, (list, tuple)):
                        arrays.append(np.array(arr, dtype=np.float32).flatten())
                    elif isinstance(arr, (int, float)):
                        arrays.append(np.array([float(arr)], dtype=np.float32))
                if arrays:
                    flat.append(np.concatenate(arrays))
                else:
                    flat.append(np.array([0.0], dtype=np.float32))
            elif isinstance(sample, np.ndarray):
                # Direct numpy array
                if sample.ndim == 0:  # scalar
                    flat.append(np.array([float(sample.item())], dtype=np.float32))
                else:
                    # Ensure it's numeric and flatten
                    try:
                        flat.append(sample.astype(np.float32).flatten())
                    except (ValueError, TypeError):
                        # If conversion fails, try to extract numeric values
                        numeric_vals = []
                        flat_sample = sample.flatten()
                        for val in flat_sample:
                            if isinstance(val, (int, float, np.number)):
                                numeric_vals.append(float(val))
                            elif hasattr(val, '__len__') and len(val) > 0:
                                # Try to extract first numeric value
                                try:
                                    numeric_vals.append(float(val[0]))
                                except (IndexError, TypeError, ValueError):
                                    numeric_vals.append(0.0)
                            else:
                                numeric_vals.append(0.0)
                        flat.append(np.array(numeric_vals, dtype=np.float32))
            else:
                # Unknown structure, create dummy feature
                print(f"Warning: Unknown feature structure type {type(sample)} at index {i}, using dummy feature")
                flat.append(np.array([0.0], dtype=np.float32))
        except Exception as e:
            print(f"Warning: Error processing sample {i}: {e}, using dummy feature")
            flat.append(np.array([0.0], dtype=np.float32))
    
    if not flat:
        raise ValueError("No valid features found in the dataset")
    
    # Ensure all feature vectors have the same length
    max_len = max(len(f) for f in flat)
    normalized_flat = []
    for f in flat:
        if len(f) < max_len:
            # Pad with zeros if necessary
            padded = np.zeros(max_len, dtype=np.float32)
            padded[:len(f)] = f
            normalized_flat.append(padded)
        else:
            normalized_flat.append(f)
    
    result = np.vstack(normalized_flat).astype(np.float32)
    print(f"  Flattened features: {len(flat)} samples -> {result.shape} (dtype: {result.dtype})")
    return result


def conv_instruction(n, prefix):
    """
    Create an enhanced quantum convolutional instruction with more expressiveness.
    
    Args:
        n: Number of qubits
        prefix: Parameter prefix
    
    Returns:
        Instruction: Quantum circuit instruction
    """
    params = ParameterVector(prefix, length=6*n)  # More parameters for richer circuit
    qc = QuantumCircuit(n, name="Conv")
    idx = 0

    # Layer 1: Parameterized single-qubit rotations on all qubits
    for q in range(n):
        qc.rx(params[idx], q)
        qc.ry(params[idx+1], q)
        idx += 2

    qc.barrier()

    # Layer 2: Entangle pairs with parameterized two-qubit gates
    for q1, q2 in zip(range(0, n - 1, 2), range(1, n, 2)):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi/2, 1)
        sub.cx(1, 0)
        sub.rz(params[idx], 0)
        sub.ry(params[idx+1], 1)
        sub.cx(0, 1)
        sub.ry(params[idx+2], 1)
        sub.rx(params[idx+3], 0)
        sub.rz(params[idx+4], 1)
        qc.compose(sub, [q1, q2], inplace=True)
        idx += 5

    qc.barrier()

    # Layer 3: Additional entanglement between neighboring pairs (overlapping pairs)
    for q1, q2 in zip(range(1, n - 2, 2), range(2, n - 1, 2)):
        sub = QuantumCircuit(2)
        sub.rx(params[idx], 0)
        sub.ry(params[idx+1], 1)
        sub.cx(0, 1)
        sub.rz(params[idx+2], 1)
        sub.cx(1, 0)
        sub.rz(params[idx+3], 0)
        qc.compose(sub, [q1, q2], inplace=True)
        idx += 4

    qc.barrier()

    # Layer 4: Parameterized single-qubit rotations on all qubits again
    for q in range(n):
        qc.ry(params[idx], q)
        qc.rz(params[idx+1], q)
        idx += 2

    return qc.to_instruction()


def pool_instruction(srcs, sinks, prefix):
    """
    Create an enhanced quantum pooling instruction with more expressiveness.
    
    Args:
        srcs: Source qubits
        sinks: Sink qubits
        prefix: Parameter prefix
    
    Returns:
        Instruction: Quantum circuit instruction
    """
    n = len(srcs) + len(sinks)
    params = ParameterVector(prefix, length=6*len(srcs))  # More parameters for richer circuit
    qc = QuantumCircuit(n, name="Pool")
    idx = 0

    # Layer 1: Parameterized single-qubit rotations on source and sink qubits
    for s, t in zip(srcs, sinks):
        qc.rx(params[idx], s)
        qc.ry(params[idx+1], s)
        qc.rz(params[idx+2], t)
        qc.ry(params[idx+3], t)
        idx += 4

    qc.barrier()

    # Layer 2: Entangling gates with parameters
    for s, t in zip(srcs, sinks):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi/2, 1)
        sub.cx(1, 0)
        sub.rz(params[idx], 0)
        sub.ry(params[idx+1], 1)
        sub.cx(0, 1)
        sub.ry(params[idx+2], 1)
        qc.compose(sub, [s, t], inplace=True)
        idx += 3

    qc.barrier()

    # Layer 3: Parameterized single-qubit rotations again
    for s, t in zip(srcs, sinks):
        qc.rx(params[idx], s)
        qc.rz(params[idx+1], t)
        idx += 2

    return qc.to_instruction()


def train_quantum_model(datasets, n_qubits=12, feature_map_reps=2, ansatz_reps=2, optimizer_maxiter=300): # Original, 6 qubits, feature_map_reps=1, ansatz_reps=1, optimizer_maxiter=100
    """
    Train quantum neural network on the datasets with improved hyperparameters.
    
    Args:
        datasets: Tuple of (train, val, test) datasets
        n_qubits: Number of qubits to use
        feature_map_reps: Number of repetitions in feature map
        ansatz_reps: Number of repetitions in ansatz circuit
        optimizer_maxiter: Maximum iterations for optimizer
    
    Returns:
        tuple: (vqr_model, predictions, scalers)
    """
    if not QUANTUM_AVAILABLE:
        print("Quantum ML libraries not available. Skipping quantum training.")
        return None, None, None
    
    print(f"\nTraining Quantum Neural Network with {n_qubits} qubits, feature_map_reps={feature_map_reps}, ansatz_reps={ansatz_reps}, optimizer_maxiter={optimizer_maxiter}...")
    
    train, val, test = datasets
    
    # Extract features and targets
    X_train_dc, y_train = train.X, train.y
    X_val_dc, y_val = val.X, val.y
    X_test_dc, y_test = test.X, test.y
    
    # Debug: Print feature structure
    print(f"Feature structure debugging:")
    print(f"  Train features type: {type(X_train_dc)}")
    if len(X_train_dc) > 0:
        print(f"  First sample type: {type(X_train_dc[0])}")
        if hasattr(X_train_dc[0], '__len__') and len(X_train_dc[0]) > 0:
            print(f"  First element of first sample type: {type(X_train_dc[0][0]) if isinstance(X_train_dc[0], (list, tuple)) else 'N/A'}")
    
    # Flatten DeepChem features
    try:
        X_train_flat = flatten_dc_features(X_train_dc)
        X_val_flat = flatten_dc_features(X_val_dc)
        X_test_flat = flatten_dc_features(X_test_dc)
        print(f"  Flattened feature shapes: Train={X_train_flat.shape}, Val={X_val_flat.shape}, Test={X_test_flat.shape}")
    except Exception as e:
        print(f"Error flattening features: {e}")
        return None, None, None
    
    # Scale and apply PCA
    fm_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    
    # Ensure all arrays are properly 2D and consistent
    print(f"  Pre-scaling shapes: Train={X_train_flat.shape}, Val={X_val_flat.shape}, Test={X_test_flat.shape}")
    print(f"  Pre-scaling dtypes: Train={X_train_flat.dtype}, Val={X_val_flat.dtype}, Test={X_test_flat.dtype}")
    
    # Check for any problematic values
    def check_array_validity(arr, name):
        # Ensure array is numeric
        if arr.dtype == object:
            print(f"Warning: {name} has object dtype, attempting to convert to float32")
            try:
                arr = arr.astype(np.float32)
            except (ValueError, TypeError) as e:
                print(f"Error converting {name} to numeric: {e}")
                # Create a fallback array
                arr = np.zeros(arr.shape, dtype=np.float32)
        
        if not np.all(np.isfinite(arr)):
            print(f"Warning: {name} contains non-finite values")
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        return arr.astype(np.float32)
    
    X_train_flat = check_array_validity(X_train_flat, "X_train_flat")
    X_val_flat = check_array_validity(X_val_flat, "X_val_flat")
    X_test_flat = check_array_validity(X_test_flat, "X_test_flat")
    
    try:
        X_all = np.vstack([X_train_flat, X_val_flat, X_test_flat])
        X_all = fm_scaler.fit_transform(X_all)
    except Exception as e:
        print(f"Error during stacking/scaling: {e}")
        # Alternative: scale each dataset separately then combine
        X_train_scaled = fm_scaler.fit_transform(X_train_flat)
        X_val_scaled = fm_scaler.transform(X_val_flat)
        X_test_scaled = fm_scaler.transform(X_test_flat)
        X_all = np.vstack([X_train_scaled, X_val_scaled, X_test_scaled])
    
    # Ensure we don't request more PCA components than features
    n_features = X_all.shape[1]
    n_components = min(n_qubits, n_features)
    if n_components != n_qubits:
        print(f"Warning: Reducing PCA components from {n_qubits} to {n_components} (limited by feature count)")
    
    pca = PCA(n_components=n_components)
    X_all_pca = pca.fit_transform(X_all)
    print(f"  PCA reduced features from {n_features} to {n_components} dimensions")
    
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
    
    # Build quantum circuit with increased reps
    feature_map = ZFeatureMap(feature_dimension=n_components, reps=feature_map_reps)
    ansatz = RealAmplitudes(num_qubits=n_components, reps=ansatz_reps)
    
    
    # Train VQR
    vqr = VQR(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=L_BFGS_B(maxiter=optimizer_maxiter),
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


def evaluate_quantum_predictions(y_true_val, y_pred_val, y_true_test, y_pred_test):
    """
    Calculate R² scores for quantum model predictions.
    
    Args:
        y_true_val: True validation labels
        y_pred_val: Predicted validation values
        y_true_test: True test labels  
        y_pred_test: Predicted test values
    
    Returns:
        dict: Dictionary with R² scores
    """
    val_r2 = r2_score(y_true_val, y_pred_val)
    test_r2 = r2_score(y_true_test, y_pred_test)
    
    return {
        'val': val_r2,
        'test': test_r2
    }


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
    tasks, datasets, transformers = load_pdbbind_data(acf, dataset_size='refined')
    
    # Preprocess data
    train, val, test = preprocess_datasets(datasets)
    
    # # Train classical ACNN
    # model, losses, val_losses = train_atomic_conv_model(
    #     train, val, 
    #     f1_num_atoms=100, 
    #     f2_num_atoms=1000, 
    #     max_epochs=50
    # )
    
    # # Plot training curves
    # plot_training_curves(losses, val_losses, save_path='training_curves.png')
    
    # # Evaluate classical model
    # results = evaluate_model(model, [train, val, test])
    
    # Train quantum model (if available)
    quantum_results = {}
    if QUANTUM_AVAILABLE:
        print("\n" + "="*60)
        print("Quantum Neural Network Extension")
        print("="*60)
        
        qnn_model, qnn_predictions, qnn_scalers = train_quantum_model(
            (train, val, test), n_qubits=12, feature_map_reps=2, ansatz_reps=2, optimizer_maxiter=300
        )
        
        if qnn_model is not None:
            y_val_pred, y_test_pred = qnn_predictions
            print(f"Quantum model validation predictions shape: {y_val_pred.shape}")
            print(f"Quantum model test predictions shape: {y_test_pred.shape}")
            
            # Calculate quantum model R² scores
            quantum_results = evaluate_quantum_predictions(
                val.y.ravel(), y_val_pred,
                test.y.ravel(), y_test_pred
            )
            
            print(f"\nQuantum Model Performance:")
            print(f"  Validation R²: {quantum_results['val']:.4f}")
            print(f"  Test R²: {quantum_results['test']:.4f}")
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
    
    # Summary
    print(f"\nFinal Results:")
    # print(f"Classical ACNN Performance:")
    # for dataset, score in results.items():
    #     print(f"  {dataset.capitalize()} R²: {score:.4f}")
        
    if QUANTUM_AVAILABLE and quantum_results:
        print(f"\nQuantum ACNN Performance:")
        print(f"  Validation R²: {quantum_results['val']:.4f}")
        print(f"  Test R²: {quantum_results['test']:.4f}")
    
    print(f"\nThe model has been trained on {len(train)} training samples")
    print(f"and validated on {len(val)} validation samples.")
    print(f"Test set contains {len(test)} samples.")
    
    # Return results including quantum performance
    results = {'quantum': quantum_results}  # Placeholder for quantum results
    results['quantum'] = quantum_results
    return qnn_model, (train, val, test), results


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
