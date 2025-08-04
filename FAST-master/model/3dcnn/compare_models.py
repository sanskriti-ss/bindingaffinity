#!/usr/bin/env python3
"""
Quantum vs Classical 3D CNN Comparison Script
This script allows easy comparison between classical and quantum-enhanced models
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def run_training(use_quantum=False, **kwargs):
    """Run training with specified parameters"""
    
    # Base arguments
    base_args = [
        sys.executable, "main_train_quantum.py",
        "--data-dir", kwargs.get("data_dir", "data"),
        "--mlhdf-fn", kwargs.get("train_hdf", "pdbbind_demo_2021_with_dG.csv.hdf"),
        "--vmlhdf-fn", kwargs.get("val_hdf", "pdbbind_demo_2021_with_dG.csv.hdf"),
        "--epoch-count", str(kwargs.get("epochs", 10)),
        "--batch-size", str(kwargs.get("batch_size", 8)),
        "--learning-rate", str(kwargs.get("learning_rate", 4e-3)),
        "--checkpoint-iter", "1",
        "--verbose", str(kwargs.get("verbose", 1)),
    ]
    
    # Quantum-specific arguments
    if use_quantum:
        base_args.extend([
            "--use-quantum",
            "--quantum-features",
            "--quantum-attention",
            "--checkpoint-dir", "checkpoint_quantum",
        ])
        model_type = "quantum"
    else:
        base_args.extend([
            "--checkpoint-dir", "checkpoint_classical",
        ])
        model_type = "classical"
    
    print(f"\n{'='*60}")
    print(f"TRAINING {model_type.upper()} MODEL")
    print(f"{'='*60}")
    
    # Log the command
    print("Running command:")
    print(" ".join(base_args))
    print()
    
    start_time = time.time()
    
    try:
        # Run the training
        result = subprocess.run(base_args, check=True, capture_output=False)
        end_time = time.time()
        
        print(f"\n{model_type.upper()} training completed successfully!")
        print(f"Training time: {end_time - start_time:.2f} seconds")
        
        return True, end_time - start_time
        
    except subprocess.CalledProcessError as e:
        print(f"\n{model_type.upper()} training failed with return code {e.returncode}")
        return False, 0
    except Exception as e:
        print(f"\nError running {model_type} training: {e}")
        return False, 0


def compare_checkpoints(classical_dir="checkpoint_classical", quantum_dir="checkpoint_quantum"):
    """Compare the best checkpoints from classical and quantum training"""
    
    classical_checkpoint = Path(classical_dir) / "best_classical_checkpoint.pth"
    quantum_checkpoint = Path(quantum_dir) / "best_quantum_checkpoint.pth"
    
    results = {}
    
    if classical_checkpoint.exists():
        try:
            import torch
            checkpoint = torch.load(classical_checkpoint, map_location="cpu")
            results["classical"] = {
                "r2": checkpoint["validate_dict"]["r2"],
                "loss": checkpoint["validate_dict"]["loss"],
                "train_r2": checkpoint["train_dict"]["r2"],
                "train_loss": checkpoint["train_dict"]["loss"],
            }
            print(f"Classical model - Validation R²: {results['classical']['r2']:.4f}, Loss: {results['classical']['loss']:.4f}")
        except Exception as e:
            print(f"Error loading classical checkpoint: {e}")
    
    if quantum_checkpoint.exists():
        try:
            import torch
            checkpoint = torch.load(quantum_checkpoint, map_location="cpu")
            results["quantum"] = {
                "r2": checkpoint["validate_dict"]["r2"],
                "loss": checkpoint["validate_dict"]["loss"],
                "train_r2": checkpoint["train_dict"]["r2"],
                "train_loss": checkpoint["train_dict"]["loss"],
            }
            print(f"Quantum model - Validation R²: {results['quantum']['r2']:.4f}, Loss: {results['quantum']['loss']:.4f}")
        except Exception as e:
            print(f"Error loading quantum checkpoint: {e}")
    
    # Compare results
    if "classical" in results and "quantum" in results:
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        classical_r2 = results["classical"]["r2"]
        quantum_r2 = results["quantum"]["r2"]
        
        improvement = quantum_r2 - classical_r2
        improvement_pct = (improvement / abs(classical_r2)) * 100 if classical_r2 != 0 else 0
        
        print(f"Classical R²:  {classical_r2:.6f}")
        print(f"Quantum R²:    {quantum_r2:.6f}")
        print(f"Improvement:   {improvement:+.6f} ({improvement_pct:+.2f}%)")
        
        if improvement > 0:
            print("🎉 Quantum model shows improvement!")
        else:
            print("📊 Classical model performed better this time")
        
        # Loss comparison
        classical_loss = results["classical"]["loss"]
        quantum_loss = results["quantum"]["loss"]
        loss_improvement = classical_loss - quantum_loss
        
        print(f"\nClassical Loss: {classical_loss:.6f}")
        print(f"Quantum Loss:   {quantum_loss:.6f}")
        print(f"Loss reduction: {loss_improvement:+.6f}")
        
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Classical vs Quantum 3D CNN models")
    parser.add_argument("--mode", choices=["classical", "quantum", "both", "compare"], 
                       default="both", help="Training mode")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=4e-3, help="Learning rate")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--train-hdf", default="pdbbind_demo_2021_with_dG.csv.hdf", help="Training HDF file")
    parser.add_argument("--val-hdf", default="pdbbind_demo_2021_with_dG.csv.hdf", help="Validation HDF file")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not os.path.exists("main_train_quantum.py"):
        print("Error: main_train_quantum.py not found in current directory")
        print("Please run this script from the 3dcnn model directory")
        return
    
    results = {}
    
    if args.mode in ["classical", "both"]:
        print("Starting classical model training...")
        success, time_taken = run_training(
            use_quantum=False,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            data_dir=args.data_dir,
            train_hdf=args.train_hdf,
            val_hdf=args.val_hdf,
            verbose=args.verbose
        )
        results["classical"] = {"success": success, "time": time_taken}
    
    if args.mode in ["quantum", "both"]:
        print("Starting quantum model training...")
        success, time_taken = run_training(
            use_quantum=True,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            data_dir=args.data_dir,
            train_hdf=args.train_hdf,
            val_hdf=args.val_hdf,
            verbose=args.verbose
        )
        results["quantum"] = {"success": success, "time": time_taken}
    
    # Compare results if both were run or if compare mode
    if args.mode in ["both", "compare"]:
        print("\nComparing model performance...")
        checkpoint_results = compare_checkpoints()
        
        # Print timing comparison if both models were trained
        if "classical" in results and "quantum" in results:
            print(f"\nTraining Time Comparison:")
            print(f"Classical: {results['classical']['time']:.2f} seconds")
            print(f"Quantum:   {results['quantum']['time']:.2f} seconds")
            
            if results['quantum']['time'] > results['classical']['time']:
                overhead = results['quantum']['time'] - results['classical']['time']
                print(f"Quantum overhead: +{overhead:.2f} seconds ({overhead/results['classical']['time']*100:.1f}%)")
            else:
                speedup = results['classical']['time'] - results['quantum']['time']
                print(f"Quantum speedup: -{speedup:.2f} seconds")


if __name__ == "__main__":
    main()
