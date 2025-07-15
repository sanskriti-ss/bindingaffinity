# Import necessary libraries
from qiskit import QuantumCircuit, transpile
from qiskit_ionq import IonQProvider
# Import native IonQ gates correctly
from qiskit_ionq import GPIGate, GPI2Gate, MSGate, ZZGate
import numpy as np

"""
NOTE: AI generated code, need to debug errors

To optimise FRQI circuit using native IonQ gates
"""

def frqi_ionq_native(image, eps=0.02, granularity=np.pi/32, system="aria"):
    """
    Build an approximate FRQI circuit using only native IonQ gates.
    
    Args:
        image: np.ndarray - 1-D or 2-D real array in [0, 1]
        eps: float - angles |θ| < eps are ignored (pruned)
        granularity: float - coarse-grain θ to nearest multiple of this value
        system: str - "aria" or "forte" to select appropriate gates
        
    Returns:
        QuantumCircuit (n_address + 1 qubits)
    """
    # Pre-compute angles
    flat = image.flatten()
    thetas = 2 * np.arcsin(flat)  # exact FRQI angles
    n_addr = int(np.log2(flat.size))
    qc = QuantumCircuit(n_addr + 1, name="FRQI_native")

    # Put address register into equal superposition
    # Use GPI2(0) which is equivalent to H up to global phase
    for i in range(n_addr):
        qc.append(GPI2Gate(0), [i])

    # Map address -> θ and drop tiny rotations
    addr2θ = {idx: θ for idx, θ in enumerate(thetas) if abs(θ) > eps}

    # Coarse-grain (angle binning)
    def bin_angle(angle):
        return round(angle / granularity) * granularity

    bins = {}
    for addr, θ in addr2θ.items():
        bins.setdefault(bin_angle(θ), []).append(addr)

    # One rotation per bin
    colour = n_addr  # index of colour qubit
    
    # Select appropriate entangling gate based on the system
    if system.lower() == "aria":
        # MS gate for Aria systems - parameters are (phase, angle)
        def create_entangle_gate():
            return MSGate(0, np.pi/2)
    else:  # forte
        # ZZ gate for Forte systems - parameter is angle
        def create_entangle_gate():
            return ZZGate(np.pi/2)

    for θ, addresses in bins.items():
        for addr in addresses:
            controls = [i for i in range(n_addr) if (addr >> i) & 1]
            
            if not controls:
                # If no controls, just apply the rotation directly
                qc.append(GPIGate(θ-np.pi/2), [colour])
                continue

            # For each control, we need to create the equivalent of controlled rotation
            # using native gates
            
            # First, apply GPI2 to prepare the color qubit
            qc.append(GPI2Gate(0), [colour])
            
            for c in controls:
                # Apply entangling gate between control and color qubits
                qc.append(create_entangle_gate(), [c, colour])
                
                # Apply single-qubit rotations to implement the controlled operation
                qc.append(GPIGate(-np.pi/2), [c])
                qc.append(GPIGate(θ/len(controls)), [colour])
                
                # Apply entangling gate again to disentangle
                qc.append(create_entangle_gate(), [c, colour])
                
                # Reset control qubit phase
                qc.append(GPIGate(np.pi/2), [c])
            
            # Final GPI2 to complete the operation
            qc.append(GPI2Gate(0), [colour])

    qc.barrier()
    return qc

# Example usage:
# Initialize IonQ provider (replace with your actual API key)
ionq_provider = IonQProvider("YOUR_API_KEY_HERE")

# Select backend based on system
system_type = "aria"  # or "forte"
if system_type.lower() == "aria":
    noise_model = "aria-1", 

    basis_gates = ['gpi', 'gpi2', 'ms']
elif system_type.lower() == "forte":
    noise_model = "forte-1"
    basis_gates = ['gpi', 'gpi2', 'zz']


backend =ionq_provider.get_backend("ionq_simulator", simulator_params={
    "noise_model": noise_model, 
    "shots": 1000
})

# Create a test image (8 pixels -> 7 qubits)
test_image = np.random.rand(8)  # Replace with your actual image data

# Build the native FRQI circuit
native_qc = frqi_ionq_native(test_image, eps=0.02, granularity=np.pi/32, system=system_type)

# Transpile for the selected IonQ system
transpiled_qc = transpile(native_qc,
                         backend=backend,
                         basis_gates=basis_gates,
                         optimization_level=3)

# Print depth comparison
print(f"Original FRQI depth: ~650")
print(f"Native FRQI depth: {transpiled_qc.depth()}")