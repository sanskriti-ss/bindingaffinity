#!/usr/bin/env python3
"""
Simple test script to verify Qiskit installation and functionality
"""

print("Testing Qiskit installation...")

try:
    # Test basic Qiskit imports (v1.0+ compatible)
    from qiskit import QuantumCircuit
    from qiskit_aer import Aer  # Changed in v1.0+
    from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
    from qiskit.primitives import Sampler, Estimator
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    import qiskit
    
    print(f"✅ Qiskit version: {qiskit.__version__}")
    print("✅ All quantum computing libraries loaded successfully")
    
    # Test basic quantum circuit
    print("\nTesting basic quantum circuit...")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    print(f"✅ Created quantum circuit with {qc.num_qubits} qubits")
    
    # Test backend
    print("\nTesting quantum backend...")
    backend = Aer.get_backend('aer_simulator')
    print(f"✅ Backend loaded: {backend.name}")
    
    # Test simple execution
    from qiskit import transpile
    transpiled_qc = transpile(qc, backend)
    job = backend.run(transpiled_qc, shots=1000)
    result = job.result()
    counts = result.get_counts()
    print(f"✅ Quantum circuit executed successfully")
    print(f"   Results: {counts}")
    
    # Test feature map
    print("\nTesting quantum feature map...")
    feature_map = ZZFeatureMap(feature_dimension=4, reps=1)
    print(f"✅ Feature map created with {feature_map.num_qubits} qubits")
    
    # Test variational circuit
    print("\nTesting variational circuit...")
    ansatz = RealAmplitudes(4, reps=1)
    print(f"✅ Variational circuit created with {ansatz.num_qubits} qubits")
    
    print("\n🌌 QUANTUM COMPUTING FULLY ENABLED! 🌌")
    print("Your quantum hybrid model is ready to run!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
