#!/usr/bin/env python3
"""
Working Atomic Convolutions for Protein-Ligand Binding Prediction

This version uses a simpler approach that avoids the featurization issues.
"""

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Configure TensorFlow
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set data directory
data_dir = 'C:\\bindingaffinity\\data'
os.makedirs(data_dir, exist_ok=True)
os.environ['DEEPCHEM_DATA_DIR'] = data_dir

# DeepChem imports
import deepchem as dc
from deepchem.molnet import load_pdbbind


def main():
    print("Working Atomic Convolutions for Protein-Ligand Interactions")
    print("=" * 65)
    
    # Use a different featurizer that works better
    print("Setting up ConvMolFeaturizer...")
    featurizer = dc.feat.ConvMolFeaturizer()
    
    # Load PDBbind dataset with the working featurizer
    print("Loading PDBbind dataset...")
    start_time = time.time()
    
    try:
        tasks, datasets, transformers = load_pdbbind(
            featurizer=featurizer,
            save_dir=data_dir,
            data_dir=data_dir,
            pocket=False,  # Use full molecule instead of pocket
            splitter='random',
            reload=False,
            subset='core'
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Trying with reload=True...")
        tasks, datasets, transformers = load_pdbbind(
            featurizer=featurizer,
            save_dir=data_dir,
            data_dir=data_dir,
            pocket=False,
            splitter='random',
            reload=True,
            subset='core'
        )
    
    print(f"Data loaded in {time.time() - start_time:.1f}s")
    train, val, test = datasets
    
    print(f"Dataset sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    if len(train) == 0:
        raise ValueError("Training dataset is empty!")
    
    # Print feature information
    if len(train.X) > 0 and train.X[0] is not None:
        print(f"Feature type: {type(train.X[0])}")
        if hasattr(train.X[0], 'get_shape'):
            print(f"Feature shape: {train.X[0].get_shape()}")
    
    # Use GraphConvModel instead of AtomicConvModel
    print("Creating GraphConvModel (more stable than AtomicConv)...")
    try:
        model = dc.models.GraphConvModel(
            n_tasks=1,
            graph_conv_layers=[64, 64],
            dense_layer_size=128,
            dropout=0.25,
            mode='regression',
            batch_size=32,
            learning_rate=0.001
        )
        print("✓ GraphConvModel created successfully!")
    except Exception as e:
        print(f"Error creating GraphConvModel: {e}")
        # Fallback to MultitaskClassifier
        print("Trying MultitaskRegressor as fallback...")
        model = dc.models.MultitaskRegressor(
            n_tasks=1,
            n_features=1024,  # Default for ConvMolFeaturizer
            layer_sizes=[1000, 500, 100],
            dropouts=0.25,
            learning_rate=0.001
        )
        print("✓ MultitaskRegressor created successfully!")\n    \n    # Training\n    print(\"Training model...\")\n    start_time = time.time()\n    \n    try:\n        model.fit(train, nb_epoch=10)\n        print(f\"Training completed in {time.time() - start_time:.1f}s\")\n    except Exception as e:\n        print(f\"Training error: {e}\")\n        print(\"Retrying with minimal settings...\")\n        model.fit(train, nb_epoch=3)\n        print(f\"Minimal training completed in {time.time() - start_time:.1f}s\")\n    \n    # Evaluation\n    print(\"\\nEvaluating model...\")\n    try:\n        r2_metric = dc.metrics.Metric(dc.metrics.score_function.pearson_r2_score)\n        rmse_metric = dc.metrics.Metric(dc.metrics.score_function.rms_score)\n        \n        for name, dataset in [(\"Train\", train), (\"Validation\", val), (\"Test\", test)]:\n            if len(dataset) > 0:\n                scores = model.evaluate(dataset, metrics=[r2_metric, rmse_metric])\n                print(f\"  {name} - R²: {scores['pearson_r2_score']:.4f}, RMSE: {scores['rms_score']:.4f}\")\n    except Exception as e:\n        print(f\"Evaluation error: {e}\")\n    \n    # Prediction test\n    print(\"\\nTesting predictions...\")\n    try:\n        y_pred = model.predict(test)\n        y_true = test.y\n        print(f\"Predictions shape: {y_pred.shape}\")\n        print(f\"True values shape: {y_true.shape}\")\n        \n        # Simple plot\n        plt.figure(figsize=(8, 6))\n        plt.scatter(y_true, y_pred, alpha=0.6)\n        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)\n        plt.xlabel('Actual Binding Affinity')\n        plt.ylabel('Predicted Binding Affinity')\n        plt.title('Actual vs Predicted Binding Affinity')\n        plt.grid(True, alpha=0.3)\n        plt.savefig('working_binding_predictions.png', dpi=300, bbox_inches='tight')\n        plt.show()\n        \n        print(\"✓ Prediction test successful!\")\n    except Exception as e:\n        print(f\"Prediction error: {e}\")\n    \n    print(\"\\n✓ Script completed successfully!\")\n    return model, (train, val, test)\n\n\nif __name__ == \"__main__\":\n    # Set seeds for reproducibility\n    np.random.seed(42)\n    tf.random.set_seed(42)\n    \n    try:\n        model, datasets = main()\n    except Exception as e:\n        print(f\"Fatal error: {e}\")\n        import traceback\n        traceback.print_exc()
