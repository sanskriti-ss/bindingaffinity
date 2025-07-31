#!/usr/bin/env python3
"""
Excited States Viewer - Display VQE excited states results clearly

This script reads the JSON results from VQE calculations and displays
the excited states in a clear, formatted way.
"""

import json
import os
import glob
from pathlib import Path

def display_excited_states(json_file):
    """Display excited states from JSON results file"""
    
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    print("=" * 70)
    print(f"EXCITED STATES ANALYSIS")
    print("=" * 70)
    
    # System information
    metadata = results.get('metadata', {})
    mol_system = metadata.get('molecular_system', {})
    
    print(f"Molecule: {mol_system.get('name', 'Unknown')}")
    print(f"Qubits: {mol_system.get('n_qubits', 'Unknown')}")
    print(f"Electrons: {mol_system.get('n_electrons', 'Unknown')}")
    print(f"Exact Energy: {mol_system.get('exact_energy', 'Unknown'):.6f} Hartree")
    print()
    
    # Ground state
    ground_state = results.get('ground_state', {})
    ground_energy = ground_state.get('energy', 0)
    print(f"Ground State Energy: {ground_energy:.6f} Hartree")
    
    # Accuracy
    accuracy = results.get('accuracy_metrics', {})
    if accuracy:
        print(f"VQE Error: {accuracy.get('absolute_error', 0):.6f} Hartree")
        print(f"Relative Error: {accuracy.get('relative_error', 0):.3f}%")
    print()
    
    # Excited states
    excited_data = results.get('excited_states', {})
    if not excited_data:
        print("❌ No excited states data found")
        return
    
    method = excited_data.get('method', 'Unknown')
    excited_states = excited_data.get('excited_states', [])
    
    print(f"Excited States Method: {method.upper()}")
    print(f"Number of Excited States: {len(excited_states)}")
    print()
    
    if not excited_states:
        print("⚠️  No excited states calculated")
        return
    
    print("ENERGY LEVELS:")
    print("-" * 50)
    print(f"{'State':<12} {'Energy (Ha)':<15} {'Transition (Ha)':<15} {'Transition (eV)':<15}")
    print("-" * 50)
    
    # Ground state
    print(f"{'Ground':<12} {ground_energy:<15.6f} {'0.000000':<15} {'0.000':<15}")
    
    # Excited states
    hartree_to_ev = 27.2114  # Conversion factor
    
    for i, (energy, params) in enumerate(excited_states):
        transition_ha = energy - ground_energy
        transition_ev = transition_ha * hartree_to_ev
        
        state_label = f"Excited {i+1}"
        print(f"{state_label:<12} {energy:<15.6f} {transition_ha:<15.6f} {transition_ev:<15.3f}")
    
    print("-" * 50)
    print()
    
    # Energy gaps
    if len(excited_states) > 0:
        homo_lumo_gap = excited_states[0][0] - ground_energy
        print(f"HOMO-LUMO Gap: {homo_lumo_gap:.6f} Hartree ({homo_lumo_gap * hartree_to_ev:.3f} eV)")
    
    # Additional analysis
    if len(excited_states) > 1:
        print("\nEXCITED STATE SPACING:")
        print("-" * 30)
        for i in range(len(excited_states) - 1):
            gap = excited_states[i+1][0] - excited_states[i][0]
            print(f"S{i+1} → S{i+2}: {gap:.6f} Ha ({gap * hartree_to_ev:.3f} eV)")
    
    print()
    
    # Computational details
    if 'computation_time' in excited_data:
        print(f"Computation Time: {excited_data['computation_time']:.2f} seconds")
    
    if 'converged' in excited_data:
        print(f"Converged: {'✓' if excited_data['converged'] else '✗'}")
    
    if 'error' in excited_data:
        print(f"❌ Error in calculation: {excited_data['error']}")

def find_latest_results():
    """Find the most recent results file"""
    pattern = "comprehensive_results_*.json"
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Sort by modification time, newest first
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = find_latest_results()
        
    if not json_file:
        print("❌ No results file found. Run VQE calculation first.")
        print("Usage: python view_excited_states.py [results_file.json]")
        return
    
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        return
    
    print(f"📄 Reading results from: {json_file}")
    print()
    
    try:
        display_excited_states(json_file)
    except Exception as e:
        print(f"❌ Error reading results: {e}")

if __name__ == "__main__":
    main()
