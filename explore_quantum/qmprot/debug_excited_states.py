#!/usr/bin/env python3
"""
Debug script for excited states calculation
"""

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import numpy as np
import pennylane as qml
from excited_states_calculator import ExcitedStatesCalculator, ExcitedStateConfig

def debug_excited_states():
    print("🔍 Debugging Excited States Calculator")
    print("=" * 50)
    
    # Create a simple test Hamiltonian
    n_qubits = 4
    coefficients = [1.0, -0.5, 0.3]
    operators = [qml.PauliZ(0), qml.PauliX(1), qml.PauliZ(0) @ qml.PauliZ(1)]
    hamiltonian = qml.Hamiltonian(coefficients, operators)
    
    print(f"Test system: {n_qubits} qubits")
    print(f"Hamiltonian terms: {len(coefficients)}")
    
    # Create excited states calculator
    config = ExcitedStateConfig(
        method="deflation",
        n_excited_states=1,
        max_iterations=5
    )
    
    calculator = ExcitedStatesCalculator(config)
    calculator.setup_device(n_qubits)
    
    print(f"Device created: {calculator.device}")
    print(f"Device qubits: {calculator.n_qubits}")
    
    # Calculate expected number of parameters
    # For hardware-efficient ansatz: (2 * n_qubits * n_layers) + n_qubits
    n_layers = 3
    expected_params = (2 * n_qubits * n_layers) + n_qubits
    print(f"Expected parameters: {expected_params}")
    
    # Create mock ground state parameters
    ground_params = np.random.uniform(0, 2*np.pi, expected_params)
    ground_energy = -1.0
    
    print(f"Ground state params shape: {ground_params.shape}")
    print(f"Ground state energy: {ground_energy}")
    
    # Test the ansatz directly
    device = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(device)
    def test_ansatz(params):
        calculator.hardware_efficient_ansatz(params, n_layers=3)
        return qml.state()
    
    print("\nTesting ansatz...")
    try:
        state = test_ansatz(ground_params)
        print(f"✓ Ansatz works! State shape: {state.shape}")
    except Exception as e:
        print(f"❌ Ansatz failed: {e}")
        return
    
    # Test deflation method
    print("\nTesting deflation method...")
    try:
        results = calculator.calculate_excited_states(
            hamiltonian, ground_params, ground_energy
        )
        print(f"✓ Deflation method works!")
        print(f"Results: {results}")
    except Exception as e:
        print(f"❌ Deflation method failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_excited_states()
