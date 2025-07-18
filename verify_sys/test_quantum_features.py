#!/usr/bin/env python3
"""
Test script for quantum feature extraction debugging
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit import transpile

print("Testing quantum feature extraction fix...")

# Test quantum circuit execution
n_qubits = 4
test_features = np.asarray([1.5, 2.1, 0.8, 3.0])  # Test feature values

try:
    # Create quantum circuit
    qc_sample = QuantumCircuit(n_qubits)
    
    # Encode features as rotation angles
    for j, feature_val in enumerate(test_features):
        # Ensure feature value is within valid range [0, 2π]
        normalized_val = np.clip(feature_val, 0, 2*np.pi)
        qc_sample.ry(normalized_val, j)
    
    # Add some entanglement
    for j in range(n_qubits - 1):
        qc_sample.cx(j, j + 1)
    
    # Execute circuit
    backend = Aer.get_backend('statevector_simulator')
    transpiled_qc = transpile(qc_sample, backend)
    job = backend.run(transpiled_qc)
    result = job.result()
    statevector = result.get_statevector()
    
    print(f"✅ Quantum circuit executed successfully")
    print(f"   Statevector type: {type(statevector)}")
    print(f"   Statevector shape: {len(statevector)}")
    
    # Test feature extraction
    statevector_array = np.asarray(statevector.data)
    print(f"   Statevector array shape: {statevector_array.shape}")
    
    # Extract features
    max_features = min(8, len(statevector_array))
    amplitudes = np.abs(statevector_array[:max_features])
    phases = np.angle(statevector_array[:max_features])
    
    # Pad if needed
    if max_features < 8:
        amplitudes = np.pad(amplitudes, (0, 8 - max_features), 'constant')
        phases = np.pad(phases, (0, 8 - max_features), 'constant')
    
    quantum_features = np.concatenate([amplitudes, phases])
    
    print(f"✅ Quantum features extracted successfully")
    print(f"   Amplitudes shape: {amplitudes.shape}")
    print(f"   Phases shape: {phases.shape}")
    print(f"   Combined features shape: {quantum_features.shape}")
    print(f"   Sample values: {quantum_features[:4]}")
    
    print("\n🌌 Quantum feature extraction is working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
