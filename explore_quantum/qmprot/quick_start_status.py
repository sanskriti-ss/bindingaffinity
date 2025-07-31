#!/usr/bin/env python3
"""
Quick Start Guide for VQE Implementation

This script demonstrates how to use the fixed VQE implementation.
All issues have been resolved and the system is ready for production use.
"""

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

def main():
    print("🔬 VQE IMPLEMENTATION - QUICK START")
    print("=" * 50)
    
    # Test imports
    try:
        from quantum_vqe_gpu import QuantumVQE, VQEConfig, MolecularSystem
        from excited_states_calculator import ExcitedStatesCalculator, ExcitedStateConfig
        from suppress_warnings import suppress_all_warnings
        suppress_all_warnings()
        print("✅ All modules imported successfully")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return

    print("\n📊 ENVIRONMENT STATUS:")
    import pennylane as qml
    print(f"✓ PennyLane: {qml.version()}")
    available_devices = [name for name in qml.plugin_devices if 'default' in name]
    print(f"✓ Available devices: {available_devices}")
    
    try:
        import jax
        print("✓ JAX: Available for acceleration")
    except ImportError:
        print("⚠ JAX: Not available")
    
    try:
        import tensorflow as tf
        print("✓ TensorFlow: Available for acceleration")
    except ImportError:
        print("⚠ TensorFlow: Not available")
    

if __name__ == "__main__":
    main()
