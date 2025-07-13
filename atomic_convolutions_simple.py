#!/usr/bin/env python3
"""
Simplified Atomic Convolutions for Protein-Ligand Binding Prediction
This script trains a simplified Atomic Conv model on the PDBbind dataset.
Usage:
    python atomic_convolutions_simple.py

Requirements:
    - deepchem
    - tensorflow
    - numpy
    - matplotlib
    - rdkit
"""

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Configure TensorFlow to suppress warnings and use legacy behavior if needed
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set shorter data directory to avoid Windows path length issues
data_dir = 'C:\\bindingaffinity\\dc'
os.makedirs(data_dir, exist_ok=True)
os.environ['DEEPCHEM_DATA_DIR'] = data_dir

# DeepChem imports
import deepchem as dc
from deepchem.molnet import load_pdbbind
from deepchem.models import AtomicConvModel
from deepchem.feat import AtomicConvFeaturizer


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


def main():
    print("Atomic Convolutions for Protein-Ligand Interactions")
    print("=" * 55)
    
    # 1. Setup featurizer
    print("Setting up featurizer...")
    f1_num_atoms = 50    # reduced ligand atoms 
    f2_num_atoms = 500   # reduced protein atoms 
    max_num_neighbors = 4  # further reduced
    
    acf = AtomicConvFeaturizer(
        frag1_num_atoms=f1_num_atoms,
        frag2_num_atoms=f2_num_atoms,
        complex_num_atoms=f1_num_atoms + f2_num_atoms,
        max_num_neighbors=max_num_neighbors,
        neighbor_cutoff=4.0  # Ensure float type
    )
    
    # 2. Load PDBbind dataset
    print("Loading PDBbind dataset...")
    start_time = time.time()
    
    # Use a shorter data directory to avoid Windows path length limits
    data_dir = 'C:\\bindingaffinity\\data'
    os.makedirs(data_dir, exist_ok=True)
    
    tasks, datasets, transformers = load_pdbbind(
        featurizer=acf,
        save_dir=data_dir,
        data_dir=data_dir,
        pocket=True,
        splitter='random',
        reload=False,  # Changed to False to use cached data if available
        subset='core'  # Use 'refined' for larger dataset
    )
    
    print(f"Data loaded in {time.time() - start_time:.1f}s")
    
    # 3. Clean data
    print("Preprocessing datasets...")
    transformer = DataTransformer()
    datasets = [d.transform(transformer) for d in datasets]
    train, val, test = datasets
    
    print(f"Dataset sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Check for valid data and print feature information
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError("One or more datasets are empty after preprocessing!")
    
    # Print sample feature information for debugging
    print(f"Sample features shape: {train.X[0] if len(train.X) > 0 else 'No features'}")
    if hasattr(train.X[0], 'shape'):
        print(f"Feature shape: {train.X[0].shape}")
    
    # 4. Create and train model
    print("Creating AtomicConvModel...")
    try:
        # Try with the original parameters first
        model = AtomicConvModel(
            n_tasks=1,
            frag1_num_atoms=f1_num_atoms,
            frag2_num_atoms=f2_num_atoms,
            complex_num_atoms=f1_num_atoms + f2_num_atoms,
            max_num_neighbors=max_num_neighbors,
            batch_size=12,  
            layer_sizes=[32, 32, 16],
            learning_rate=0.001,
        )
        print("Model created successfully!")
    except Exception as e:
        print(f"Error creating model with original parameters: {e}")
        print("Trying with simplified parameters...")
        # Try with minimal parameters and different architecture
        try:
            model = AtomicConvModel(
                n_tasks=1,
                frag1_num_atoms=f1_num_atoms,
                frag2_num_atoms=f2_num_atoms,
                complex_num_atoms=f1_num_atoms + f2_num_atoms,
                max_num_neighbors=max_num_neighbors,
                batch_size=8,
                layer_sizes=[16, 8],
                learning_rate=0.001,
            )
            print("Model created with simplified parameters!")
        except Exception as e2:
            print(f"Error with simplified parameters: {e2}")
            # Try with even more basic settings
            model = AtomicConvModel(
                n_tasks=1,
                frag1_num_atoms=f1_num_atoms,
                frag2_num_atoms=f2_num_atoms,
                complex_num_atoms=f1_num_atoms + f2_num_atoms,
                max_num_neighbors=max_num_neighbors,
                batch_size=4,
                layer_sizes=[8],
                learning_rate=0.001,
            )
            print("Model created with minimal parameters!")
    
    # Training with progress tracking
    losses, val_losses = [], []
    metric = dc.metrics.Metric(dc.metrics.score_function.rms_score)
    step_cutoff = max(1, len(train) // 12)  # Ensure at least 1
    
    def validation_callback(model, step):
        if step % step_cutoff != 0:
            return
        try:
            val_loss = model.evaluate(val, metrics=[metric])['rms_score']**2
            train_loss = model.evaluate(train, metrics=[metric])['rms_score']**2
            val_losses.append(val_loss)
            losses.append(train_loss)
            print(f"  Epoch {len(losses)}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        except Exception as e:
            print(f"Error in validation callback: {e}")
    
    print("Training model...")
    start_time = time.time()
    model.fit(
        train,
        nb_epoch=50,
        max_checkpoints_to_keep=1,
        callbacks=[validation_callback]
    )
    print(f"Training completed in {time.time() - start_time:.1f}s")
    
    # 5. Plot training curves
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'o-', label='Training Loss', alpha=0.7)
    plt.plot(val_losses, 's-', label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (RMS²)')
    plt.title('ACNN Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('acnn_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
     # Additional plot for actual vs predicted binding energies
    y_pred, y_test = model.predict(test)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Binding Energy')
    plt.ylabel('Predicted Binding Energy')
    plt.title('Actual vs Predicted Binding Energies')
    plt.grid(True, alpha=0.3)
    plt.savefig('acnn_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Final evaluation
    print("\nFinal Evaluation:")
    r2_metric = dc.metrics.Metric(dc.metrics.score_function.pearson_r2_score)
    
    for name, dataset in [('Train', train), ('Validation', val), ('Test', test)]:
        score = model.evaluate(dataset, metrics=[r2_metric])['pearson_r2_score']
        print(f"  {name} R² Score: {score:.4f}")
    
    print("\nTraining completed successfully!")
    print("Model saved automatically by DeepChem.")
    
    return model, (train, val, test)


if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    try:
        model, datasets = main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
